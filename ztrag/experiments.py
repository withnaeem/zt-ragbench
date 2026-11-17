"""Experiments for ZT-RAGBench. Run: python -m ztrag.experiments --seeds 5"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import (DEPT_WORDS, GENERIC, PROJECTS, DEPTS, Index, add_derived_artifacts, chunk_auth_primary,
                   chunk_auth_truth, doc_auth_mask, generate_corpus, generate_users, ingestion_quarantine,
                   label_content_check, topk, topk_masked)

K = 5
SEMANTIC = set(GENERIC) | set(PROJECTS) | {w for ws in DEPT_WORDS.values() for w in ws}

ARCHS = ["A0 Shared index, no ACL", "A1 Tenant-partitioned index", "A2 Post-retrieval filter",
         "A3 Post-filter + 4x over-fetch", "A4 Pre-filter (ACL snapshot)", "A5 Zero-trust pipeline"]


# ---------------------------------------------------------------------------
def doc_first_chunk(corpus):
    first = {}
    for i, d in enumerate(corpus.chunk_doc):
        if len(corpus.chunk_sources[i]) == 1 and int(d) not in first:
            first[int(d)] = i
    return first


def make_query(rng, corpus, first_chunk, d, n_words, key_prob):
    words = [w for w in corpus.texts[first_chunk[d]].split() if w in SEMANTIC]
    q = list(rng.choice(words, n_words, replace=False))
    if rng.random() < key_prob:
        q.append(corpus.doc_keys[d][rng.integers(2)])
    return " ".join(q)


def all_masks(users, corpus, levels=None):
    lv = corpus.doc_level if levels is None else levels
    return np.stack([doc_auth_mask(u, users, corpus.doc_tenant, corpus.doc_dept, lv, corpus.doc_acl)
                     for u in range(len(users.tenant))])


def batched_scores(index, queries, bs=400):
    for s in range(0, len(queries), bs):
        Q = index.embed(queries[s:s + bs])
        yield s, index.scores(Q)


def ci(vals):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return float(v.mean()), float(t.get(len(v), 1.96) * v.std(ddof=1) / np.sqrt(len(v)))


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
def workloads(rng, corpus, users, doc_masks, first_chunk, n_benign=1500, n_insider=1000, n_cross=500):
    n_users = len(users.tenant)
    W = []
    for _ in range(n_benign):
        u = rng.integers(n_users)
        d = rng.choice(np.where(doc_masks[u])[0])
        W.append(("benign", u, d, make_query(rng, corpus, first_chunk, d, 4, 0.5)))
    for _ in range(n_insider):
        u = rng.integers(n_users)
        pool = np.where((~doc_masks[u]) & (corpus.doc_tenant == users.tenant[u]) & (corpus.doc_level >= 1))[0]
        d = rng.choice(pool)
        W.append(("insider", u, d, make_query(rng, corpus, first_chunk, d, 3, 1.0)))
    for _ in range(n_cross):
        u = rng.integers(n_users)
        pool = np.where(corpus.doc_tenant != users.tenant[u])[0]
        d = rng.choice(pool)
        W.append(("cross", u, d, make_query(rng, corpus, first_chunk, d, 3, 1.0)))
    return W


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
def run_arch(arch, s, u, ctx):
    """Return retrieved chunk ids for architecture `arch` given score vector s and user u."""
    c = ctx
    if arch == 0:
        return topk(s, K)
    if arch == 1:
        return topk_masked(s, c["chunk_tenant"] == c["users"].tenant[u], K)
    if arch == 2:
        top = topk(s, K)
        return top[c["snap_chunk_masks"][u][top]]
    if arch == 3:
        top = topk(s, 4 * K)
        return top[c["snap_chunk_masks"][u][top]][:K]
    if arch == 4:
        return topk_masked(s, c["snap_chunk_masks"][u], K)
    if arch == 5:
        # live policy decision per request (no cached entitlements), all-source label propagation,
        # then label/content consistency check.
        corpus, users = c["corpus"], c["live_users"]
        dm = doc_auth_mask(u, users, corpus.doc_tenant, corpus.doc_dept, c["stored_levels"], corpus.doc_acl)
        cm = chunk_auth_truth(dm, corpus) if c["has_derived"] else dm[corpus.chunk_doc]
        ids = topk_masked(s, cm, K)
        return label_content_check(corpus.texts, ids, c["chunk_stored_level"], c.get("quarantine"))
    if arch == 6:  # A5 with query-time chunk-level check only (ablation)
        corpus, users = c["corpus"], c["live_users"]
        dm = doc_auth_mask(u, users, corpus.doc_tenant, corpus.doc_dept, c["stored_levels"], corpus.doc_acl)
        cm = chunk_auth_truth(dm, corpus) if c["has_derived"] else dm[corpus.chunk_doc]
        return label_content_check(corpus.texts, topk_masked(s, cm, K), c["chunk_stored_level"])
    raise ValueError(arch)


def evaluate(W, results_ids, corpus, truth_chunk_masks, oracle_ids=None):
    out = {}
    for kind in ("benign", "insider", "cross"):
        idx = [i for i, w in enumerate(W) if w[0] == kind]
        if not idx:
            continue
        exp, n_unauth, hit, fill, overlap = [], [], [], [], []
        for i in idx:
            _, u, d, _ = W[i]
            ids = results_ids[i]
            un = (~truth_chunk_masks[u][ids]).sum() if len(ids) else 0
            exp.append(un > 0)
            n_unauth.append(un)
            hit.append(bool(np.any(corpus.chunk_doc[ids] == d)) if len(ids) else False)
            fill.append(len(ids) / K)
            if oracle_ids is not None:
                o = oracle_ids[i]
                overlap.append(len(set(ids.tolist()) & set(o.tolist())) / max(len(o), 1))
        out[kind] = {"exposure": float(np.mean(exp)), "unauth_per_q": float(np.mean(n_unauth)),
                     "target_hit": float(np.mean(hit)), "fill": float(np.mean(fill))}
        if overlap:
            out[kind]["oracle_overlap"] = float(np.mean(overlap))
    return out


def build_ctx(corpus, users, snap_users=None, stored_levels=None, derived_policy="truth"):
    stored_levels = corpus.doc_level if stored_levels is None else stored_levels
    snap_users = users if snap_users is None else snap_users
    has_derived = any(len(s) > 1 for s in corpus.chunk_sources)
    snap_doc = all_masks(snap_users, corpus, stored_levels)
    f = chunk_auth_truth if derived_policy == "truth" else chunk_auth_primary
    snap_chunk = np.stack([f(m, corpus) for m in snap_doc])
    chunk_stored_level = np.array([max(stored_levels[s] for s in srcs) for srcs in corpus.chunk_sources])
    return {"corpus": corpus, "users": users, "live_users": users, "chunk_tenant": corpus.doc_tenant[corpus.chunk_doc],
            "snap_chunk_masks": snap_chunk, "stored_levels": stored_levels, "has_derived": has_derived,
            "chunk_stored_level": chunk_stored_level, "quarantine": ingestion_quarantine(corpus, stored_levels)}


# ---------------------------------------------------------------------------
# E1: architecture comparison (static, correctly labelled corpus)
# ---------------------------------------------------------------------------
def exp_main(seed, kind):
    rng = np.random.default_rng(seed)
    corpus = generate_corpus(rng)
    users = generate_users(rng)
    index = Index(corpus.texts, kind=kind, seed=seed)
    live_doc = all_masks(users, corpus)
    truth = np.stack([chunk_auth_truth(m, corpus) for m in live_doc])
    W = workloads(rng, corpus, users, live_doc, doc_first_chunk(corpus))
    ctx = build_ctx(corpus, users)
    ids = {a: [None] * len(W) for a in range(6)}
    oracle = [None] * len(W)
    lat = {a: [] for a in range(6)}
    for s0, S in batched_scores(index, [w[3] for w in W]):
        for j in range(S.shape[0]):
            i = s0 + j
            u = W[i][1]
            oracle[i] = topk_masked(S[j], truth[u], K)
            for a in range(6):
                t0 = time.perf_counter()
                ids[a][i] = run_arch(a, S[j], u, ctx)
                lat[a].append(time.perf_counter() - t0)
    res = {}
    for a in range(6):
        res[ARCHS[a]] = evaluate(W, ids[a], corpus, truth, oracle_ids=oracle)
        res[ARCHS[a]]["latency_ms"] = 1000 * float(np.median(lat[a]))
    res["_meta"] = {"n_chunks": len(corpus.texts), "n_docs": corpus.n_docs, "n_users": len(users.tenant),
                    "n_queries": {k: sum(w[0] == k for w in W) for k in ("benign", "insider", "cross")},
                    "authorised_doc_fraction": float(live_doc.mean() * 3)}
    return res


# ---------------------------------------------------------------------------
# E2: permission revocation lag (dynamic entitlements)
# ---------------------------------------------------------------------------
DELTAS_H = [0, 0.25, 1, 6, 24, 168]


def exp_revocation(seed, revoke_frac=0.2, rate_per_day=6, horizon_h=168):
    rng = np.random.default_rng(seed + 1000)
    corpus = generate_corpus(rng)
    before = generate_users(rng)
    index = Index(corpus.texts, kind="tfidf", seed=seed)
    after = before.copy()
    cand = np.where(before.clearance >= 1)[0]
    revoked = rng.choice(cand, int(revoke_frac * len(cand)), replace=False)
    r_time = {}
    for u in revoked:
        if before.clearance[u] >= 2 and before.groups[u] != 0 and rng.random() < 0.5:
            after.groups[u] = np.uint64(0)          # removed from all project groups
        else:
            after.dept[u] = (before.dept[u] + rng.integers(1, len(DEPTS))) % len(DEPTS)  # transfer
        r_time[int(u)] = rng.uniform(0, horizon_h)
    first = doc_first_chunk(corpus)
    Q = []  # (user, time, query)
    for u in revoked:
        u = int(u)
        mb = doc_auth_mask(u, before, corpus.doc_tenant, corpus.doc_dept, corpus.doc_level, corpus.doc_acl)
        ma = doc_auth_mask(u, after, corpus.doc_tenant, corpus.doc_dept, corpus.doc_level, corpus.doc_acl)
        lost = np.where(mb & ~ma)[0]
        now = np.where(ma)[0]
        remaining = horizon_h - r_time[u]
        n = rng.poisson(rate_per_day * remaining / 24)
        for t in rng.uniform(r_time[u], horizon_h, n):
            d = rng.choice(lost) if (len(lost) and rng.random() < 0.5) else rng.choice(now)
            Q.append((u, t, make_query(rng, corpus, first, d, 4, 0.5)))
    res = {}
    if not Q:
        return res
    live_masks = {u: doc_auth_mask(u, after, corpus.doc_tenant, corpus.doc_dept, corpus.doc_level, corpus.doc_acl)[corpus.chunk_doc]
                  for u in r_time}
    old_masks = {u: doc_auth_mask(u, before, corpus.doc_tenant, corpus.doc_dept, corpus.doc_level, corpus.doc_acl)[corpus.chunk_doc]
                 for u in r_time}
    leaks = {d: [] for d in DELTAS_H}
    leaks["zt"] = []
    for s0, S in batched_scores(index, [q[2] for q in Q]):
        for j in range(S.shape[0]):
            u, t, _ = Q[s0 + j]
            for D in DELTAS_H:
                snap_t = 0.0 if D >= horizon_h else (t if D == 0 else np.floor(t / D) * D)
                m = old_masks[u] if r_time[u] > snap_t else live_masks[u]
                ids = topk_masked(S[j], m, K)
                leaks[D].append(bool((~live_masks[u][ids]).any()))
            ids = topk_masked(S[j], live_masks[u], K)
            leaks["zt"].append(bool((~live_masks[u][ids]).any()))
    for k, v in leaks.items():
        res[str(k)] = float(np.mean(v))
    res["_n_queries"] = len(Q)
    res["_n_revoked"] = len(revoked)
    return res


# ---------------------------------------------------------------------------
# E3: identity-agnostic semantic response cache
# ---------------------------------------------------------------------------
TAUS = [0.80, 0.85, 0.90, 0.95, 0.98]


def exp_cache(seed, n_stream=6000, pool_per_tenant=100):
    rng = np.random.default_rng(seed + 2000)
    corpus = generate_corpus(rng)
    users = generate_users(rng)
    index = Index(corpus.texts, kind="lsa", seed=seed)
    first = doc_first_chunk(corpus)
    live = all_masks(users, corpus)
    cmask = live[:, corpus.chunk_doc]
    n_t = int(corpus.doc_tenant.max()) + 1
    pools = []
    for t in range(n_t):
        docs = rng.choice(np.where(corpus.doc_tenant == t)[0], pool_per_tenant, replace=False)
        pools.append([make_query(rng, corpus, first, int(d), 4, 0.3) for d in docs])
    zipf = 1.0 / np.arange(1, pool_per_tenant + 1) ** 1.1
    zipf /= zipf.sum()
    stream = []
    for _ in range(n_stream):
        u = int(rng.integers(len(users.tenant)))
        words = pools[users.tenant[u]][rng.choice(pool_per_tenant, p=zipf)].split()
        if rng.random() < 0.5:   # paraphrase: substitute a word
            words[rng.integers(len(words))] = rng.choice(GENERIC)
        if rng.random() < 0.3 and len(words) > 3:
            words.pop(rng.integers(len(words)))
        stream.append((u, " ".join(words)))
    E = index.embed([s[1] for s in stream])
    S_all = index.scores(E)
    fresh = [topk_masked(S_all[i], cmask[stream[i][0]], K) for i in range(n_stream)]
    res = {}
    for tau in TAUS:
        out = {}
        for variant in ("naive_global", "per_tenant", "zt_revalidate"):
            keys, vals, owners = [], [], []
            leak, hit = [], []
            for i, (u, _) in enumerate(stream):
                served = None
                if keys:
                    K_ = np.asarray(keys)
                    sims = K_ @ E[i]
                    if variant == "per_tenant":
                        sims = np.where(users.tenant[np.asarray(owners)] == users.tenant[u], sims, -1)
                    j = int(np.argmax(sims))
                    if sims[j] >= tau:
                        cand = vals[j]
                        if variant != "zt_revalidate" or cmask[u][cand].all():
                            served = cand
                if served is None:
                    served = fresh[i]
                    keys.append(E[i]); vals.append(served); owners.append(u)
                    hit.append(False)
                else:
                    hit.append(True)
                leak.append(bool((~cmask[u][served]).any()))
            out[variant] = {"leak_rate": float(np.mean(leak)), "hit_rate": float(np.mean(hit))}
        res[str(tau)] = out
    return res


# ---------------------------------------------------------------------------
# E4: derived artifacts (label propagation)
# ---------------------------------------------------------------------------
def exp_derived(seed):
    rng = np.random.default_rng(seed + 3000)
    base = generate_corpus(rng)
    users = generate_users(rng)
    corpus = add_derived_artifacts(rng, base, frac=0.15)
    index = Index(corpus.texts, kind="tfidf", seed=seed)
    live_doc = all_masks(users, corpus)
    truth = np.stack([chunk_auth_truth(m, corpus) for m in live_doc])
    naive = np.stack([chunk_auth_primary(m, corpus) for m in live_doc])
    W = workloads(rng, corpus, users, live_doc, doc_first_chunk(corpus), n_cross=0)
    derived_idx = np.array([i for i, s in enumerate(corpus.chunk_sources) if len(s) > 1])
    lv = corpus.doc_level
    mixed = np.mean([len({(lv[s], corpus.doc_dept[s] if lv[s] == 1 else -1, int(corpus.doc_acl[s])) for s in corpus.chunk_sources[c]}) > 1
                     for c in derived_idx])
    out = {"naive": [None] * len(W), "zt": [None] * len(W)}
    for s0, S in batched_scores(index, [w[3] for w in W]):
        for j in range(S.shape[0]):
            u = W[s0 + j][1]
            out["naive"][s0 + j] = topk_masked(S[j], naive[u], K)
            out["zt"][s0 + j] = topk_masked(S[j], truth[u], K)
    res = {k: evaluate(W, v, corpus, truth) for k, v in out.items()}
    # share of exposures caused by derived chunks under naive labels
    n_der = sum(int(np.isin(ids[~truth[W[i][1]][ids]], derived_idx).sum()) for i, ids in enumerate(out["naive"]))
    n_all = sum(int((~truth[W[i][1]][ids]).sum()) for i, ids in enumerate(out["naive"]))
    res["_meta"] = {"n_derived": int(len(derived_idx)), "label_mixed_fraction": float(mixed),
                    "unauth_from_derived": n_der, "unauth_total": n_all,
                    "naive_overpermissive_pairs": float((naive[:, derived_idx] & ~truth[:, derived_idx]).mean())}
    return res


# ---------------------------------------------------------------------------
# E5: label noise and the label/content consistency check
# ---------------------------------------------------------------------------
EPS = [0.0, 0.01, 0.02, 0.05, 0.10]


def exp_labelnoise(seed):
    rng = np.random.default_rng(seed + 4000)
    corpus = generate_corpus(rng)
    users = generate_users(rng)
    index = Index(corpus.texts, kind="tfidf", seed=seed)
    live_doc = all_masks(users, corpus)
    truth = np.stack([chunk_auth_truth(m, corpus) for m in live_doc])
    W = workloads(rng, corpus, users, live_doc, doc_first_chunk(corpus), n_cross=0)
    flip_u = rng.random(corpus.n_docs)
    scores = [S for _, S in batched_scores(index, [w[3] for w in W], bs=len(W))][0]
    res = {}
    for eps in EPS:
        stored = corpus.doc_level.copy()
        stored[(corpus.doc_level >= 1) & (flip_u < eps)] = 0
        ctx = build_ctx(corpus, users, stored_levels=stored)
        r_a4 = [run_arch(4, scores[i], W[i][1], ctx) for i in range(len(W))]
        r_a5 = [run_arch(5, scores[i], W[i][1], ctx) for i in range(len(W))]
        r_a5c = [run_arch(6, scores[i], W[i][1], ctx) for i in range(len(W))]
        fb = {}
        for name, rr in (("chunk", r_a5c), ("doc", r_a5)):
            fb[name] = float(np.mean([int(truth[W[i][1]][np.setdiff1d(r_a4[i], rr[i])].sum()) > 0
                                      for i in range(len(W)) if W[i][0] == "benign"]))
        res[str(eps)] = {"A4": evaluate(W, r_a4, corpus, truth), "A5_chunk": evaluate(W, r_a5c, corpus, truth),
                         "A5": evaluate(W, r_a5, corpus, truth),
                         "benign_false_block": fb,
                         "n_mislabelled": int(((corpus.doc_level >= 1) & (flip_u < eps)).sum())}
    return res


# ---------------------------------------------------------------------------
def aggregate(list_of_dicts):
    """Recursively aggregate a list of identically-shaped nested dicts into mean/ci."""
    first = list_of_dicts[0]
    if isinstance(first, dict):
        return {k: aggregate([d[k] for d in list_of_dicts]) for k in first}
    if isinstance(first, (int, float, np.floating, np.integer, bool)):
        m, h = ci([float(x) for x in list_of_dicts])
        return {"mean": m, "ci95": h}
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    seeds = list(range(a.seeds))
    allres = {}
    t0 = time.time()
    for name, fn in [("main_tfidf", lambda s: exp_main(s, "tfidf")), ("main_lsa", lambda s: exp_main(s, "lsa")),
                     ("revocation", exp_revocation), ("cache", exp_cache), ("derived", exp_derived),
                     ("labelnoise", exp_labelnoise)]:
        runs = []
        for s in seeds:
            runs.append(fn(s))
            print(f"[{time.time() - t0:7.1f}s] {name} seed {s} done", flush=True)
        (out / f"{name}_raw.json").write_text(json.dumps(runs, indent=1))
        allres[name] = aggregate(runs)
    (out / "results.json").write_text(json.dumps(allres, indent=1))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
