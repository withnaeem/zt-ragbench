"""In-flight revocation experiments (E6-E9): what happens when a user's rights are revoked while a
RAG request that already retrieved the affected evidence is still being generated.

Stochastic simulation over ZT-RAGBench retrieval. Each "revocation event" is a departing-insider
session: a user issues requests back-to-back (closed loop) for 2 h; at a uniformly random time in
[0.5 h, 1.5 h] the authoritative policy revokes part of their access (department transfer or
project-group removal). We count releases after the revocation whose evidence includes a lost
document. Run: python -m ztrag.inflight --seeds 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import DEPTS, Index, doc_auth_mask, generate_corpus, generate_users, topk_masked
from .experiments import K, aggregate, doc_first_chunk, make_query

SESSION_H = 2.0
WORKLOADS = {
    # generation time lognormal(median, sigma) in seconds; hops = retrieval rounds; think = mean s
    "chat": {"g_med": 4.0, "g_sig": 0.5, "hops": 1, "think": 30.0},
    "agentic": {"g_med": 60.0, "g_sig": 0.7, "hops": 3, "think": 30.0},
}
DEFAULTS = {"ttl": 300.0, "lag_mean": 2.0, "delta_med": 0.3, "delta_sig": 0.8,
            "rtt_replica_med": 0.010, "rtt_auth_med": 0.040, "rtt_sig": 0.5, "p_target": 0.7}
STRATS = ["S0 Retrieval check, cached entitlements (TTL)", "S1 Retrieval check, live",
          "S2 S1 + non-atomic release recheck", "S3 S1 + atomic gate on replica",
          "S4 S1 + atomic gate on authority (fenced)"]


# ---------------------------------------------------------------------------
class World:
    """Corpus, index and departing-insider sessions with retrieval outcomes precomputed for the
    pre- and post-revocation policy states."""

    def __init__(self, seed, n_sessions=200, workload="chat", p_target=0.7):
        rng = np.random.default_rng(seed + 5000 + (0 if workload == "chat" else 77))
        self.rng = rng
        self.corpus = c = generate_corpus(rng)
        users = generate_users(rng)
        self.index = Index(c.texts, kind="tfidf", seed=seed)
        first = doc_first_chunk(c)
        wl = WORKLOADS[workload]
        self.wl = wl
        cand = np.where(users.clearance >= 1)[0]
        self.sessions = []
        queries = []
        for _ in range(n_sessions):
            u = int(rng.choice(cand))
            after = users.copy()
            if users.clearance[u] >= 2 and users.groups[u] != 0 and rng.random() < 0.5:
                after.groups[u] = np.uint64(0)
            else:
                after.dept[u] = (users.dept[u] + rng.integers(1, len(DEPTS))) % len(DEPTS)
            mb = doc_auth_mask(u, users, c.doc_tenant, c.doc_dept, c.doc_level, c.doc_acl)
            ma = doc_auth_mask(u, after, c.doc_tenant, c.doc_dept, c.doc_level, c.doc_acl)
            lost = np.where(mb & ~ma)[0]
            if len(lost) == 0:
                continue
            keep = np.where(ma)[0]
            # enough queries for the session (upper bound on number of requests x hops)
            n_req = int(SESSION_H * 3600 / (wl["think"] + wl["g_med"]) * 2.5) + 5
            q0 = len(queries)
            for _ in range(n_req * wl["hops"]):
                d = rng.choice(lost) if rng.random() < p_target else rng.choice(keep)
                queries.append(make_query(rng, c, first, int(d), 4, 0.6))
            self.sessions.append({"u": u, "mb": mb[c.chunk_doc], "ma": ma[c.chunk_doc], "lost": set(lost.tolist()),
                                  "q0": q0, "nq": len(queries) - q0,
                                  "r": rng.uniform(0.5 * 3600, 1.5 * 3600)})
        # retrieval under both states
        self.dep_before = [None] * len(queries)
        self.dep_after = [None] * len(queries)
        owner = np.zeros(len(queries), dtype=int)
        for si, s in enumerate(self.sessions):
            owner[s["q0"]:s["q0"] + s["nq"]] = si
        for s0 in range(0, len(queries), 500):
            S = self.index.scores(self.index.embed(queries[s0:s0 + 500]))
            for j in range(S.shape[0]):
                s = self.sessions[owner[s0 + j]]
                self.dep_before[s0 + j] = frozenset(c.chunk_doc[topk_masked(S[j], s["mb"], K)].tolist())
                self.dep_after[s0 + j] = frozenset(c.chunk_doc[topk_masked(S[j], s["ma"], K)].tolist())


def lognorm(rng, med, sig, size=None):
    return med * np.exp(sig * rng.standard_normal(size))


# ---------------------------------------------------------------------------
def simulate(world: World, strat: int, rng, ttl=None, lag_mean=None, delta_med=None, provenance_miss=0.0,
             stream_lease=None, n_tokens=300, ttft=0.5):
    """Simulate every session under one strategy. Returns per-session metrics and per-request overhead.

    A read of policy state at wall time t through a replica reflects the authority as of t - lag,
    lag ~ Exp(lag_mean). The authority itself is linearisable (lag 0).
    """
    P = dict(DEFAULTS)
    ttl = P["ttl"] if ttl is None else ttl
    lag_mean = P["lag_mean"] if lag_mean is None else lag_mean
    delta_med = P["delta_med"] if delta_med is None else delta_med
    wl = world.wl
    out_unsafe, out_any, out_window, out_tokens, overhead, n_req_total, n_checks = [], [], [], [], [], 0, 0

    def sees_revocation(t_read, r, authority=False):
        lag = 0.0 if authority else rng.exponential(lag_mean) if lag_mean > 0 else 0.0
        return (t_read - lag) >= r

    for s in world.sessions:
        r = s["r"]
        t = rng.uniform(0, 60)
        qi = s["q0"]
        cache_t = -1e9           # time entitlements were last fetched (S0)
        cache_revoked = False    # did that fetch see the revocation
        unsafe, last_unsafe, leaked_tok = 0, None, 0
        while t < SESSION_H * 3600 and qi + wl["hops"] <= s["q0"] + s["nq"]:
            n_req_total += 1
            G = float(lognorm(rng, wl["g_med"], wl["g_sig"]))
            deps = set()
            for h in range(wl["hops"]):
                th = t + G * h / wl["hops"]
                if strat == 0:
                    if th - cache_t > ttl:
                        cache_t = th
                        cache_revoked = sees_revocation(th, r)
                    revoked_view = cache_revoked
                else:
                    revoked_view = sees_revocation(th, r)
                deps |= world.dep_after[qi] if revoked_view else world.dep_before[qi]
                qi += 1
            recorded = {d for d in deps if rng.random() >= provenance_miss}
            touches_lost = bool(deps & s["lost"])
            rec_touches_lost = bool(recorded & s["lost"])
            t_gen_end = t + G
            delta = float(lognorm(rng, delta_med, P["delta_sig"]))
            rtt_r = float(lognorm(rng, P["rtt_replica_med"], P["rtt_sig"]))
            rtt_a = float(lognorm(rng, P["rtt_auth_med"], P["rtt_sig"]))
            released, t_commit, extra = True, t_gen_end + delta, 0.0

            if stream_lease is not None:
                # streaming: tokens uniform over [t+ttft, t+G]; lease re-checks every stream_lease s
                # (authority reads); stream aborted at first check that sees the revocation.
                t0s, t1s = t + ttft, t_gen_end
                if touches_lost and r < t1s:
                    if np.isinf(stream_lease):
                        stop = t1s
                    else:
                        k = np.ceil(max(r - t0s, 0) / stream_lease)
                        stop = min(t0s + k * stream_lease, t1s)
                    a = max(r, t0s)
                    if stop > a:
                        leaked_tok += int(round(n_tokens * (stop - a) / (t1s - t0s)))
                if not np.isinf(stream_lease):
                    n_checks += int((t1s - t0s) / stream_lease) + 1
                t = t1s + rng.exponential(wl["think"])
                continue

            if strat in (0, 1):
                pass
            elif strat == 2:
                t_check = t_gen_end
                extra = rtt_r
                if rec_touches_lost and sees_revocation(t_check, r):
                    released = False
                t_commit = t_gen_end + rtt_r + delta
            elif strat == 3:
                # atomic: decision and release ordered together, but decision read from a replica
                t_commit = t_gen_end + delta + rtt_r
                extra = rtt_r
                if rec_touches_lost and sees_revocation(t_commit, r):
                    released = False
            elif strat == 4:
                t_commit = t_gen_end + delta + rtt_a
                extra = rtt_a
                if rec_touches_lost and sees_revocation(t_commit, r, authority=True):
                    released = False
            overhead.append(extra)
            if released and touches_lost and t_commit > r:
                unsafe += 1
                last_unsafe = t_commit
            t = (t_commit if released else t_gen_end + extra) + rng.exponential(wl["think"])
        out_unsafe.append(unsafe)
        out_any.append(unsafe > 0)
        out_window.append((last_unsafe - r) if last_unsafe is not None else 0.0)
        out_tokens.append(leaked_tok)
    ov = np.array(overhead) if overhead else np.zeros(1)
    return {"unsafe_per_event": float(np.mean(out_unsafe)), "p_any_unsafe": float(np.mean(out_any)),
            "exposure_window_s": float(np.mean(out_window)), "leaked_tokens_per_event": float(np.mean(out_tokens)),
            "overhead_p50_ms": float(1000 * np.median(ov)), "overhead_p95_ms": float(1000 * np.percentile(ov, 95)),
            "requests_per_event": n_req_total / max(len(world.sessions), 1),
            "checks_per_request": n_checks / max(n_req_total, 1), "n_events": len(world.sessions)}


# ---------------------------------------------------------------------------
TTLS = [0, 60, 300, 900, 3600]
DELTAS = [0.05, 0.3, 1.0, 3.0]
LAGS = [0.0, 0.5, 2.0, 10.0, 30.0]
MISS = [0.0, 0.01, 0.05, 0.10, 0.25]
LEASES = [np.inf, 10.0, 2.0, 0.5]


def run_seed(seed):
    res = {}
    for wl in ("chat", "agentic"):
        w = World(seed, workload=wl)
        R = {}
        R["main"] = {STRATS[s]: simulate(w, s, np.random.default_rng(seed * 10 + s)) for s in range(5)}
        R["ttl"] = {str(x): simulate(w, 0, np.random.default_rng(seed), ttl=x)["unsafe_per_event"] for x in TTLS}
        R["delta"] = {str(x): {"S2": simulate(w, 2, np.random.default_rng(seed), delta_med=x)["unsafe_per_event"],
                               "S4": simulate(w, 4, np.random.default_rng(seed), delta_med=x)["unsafe_per_event"]}
                      for x in DELTAS}
        R["lag"] = {str(x): {"S2": simulate(w, 2, np.random.default_rng(seed), lag_mean=x)["unsafe_per_event"],
                             "S3": simulate(w, 3, np.random.default_rng(seed), lag_mean=x)["unsafe_per_event"],
                             "S4": simulate(w, 4, np.random.default_rng(seed), lag_mean=x)["unsafe_per_event"]}
                    for x in LAGS}
        R["provenance"] = {str(x): simulate(w, 4, np.random.default_rng(seed), provenance_miss=x)["unsafe_per_event"]
                           for x in MISS}
        R["stream"] = {}
        for L in LEASES:
            m = simulate(w, 1, np.random.default_rng(seed), stream_lease=L)
            R["stream"]["none" if np.isinf(L) else str(L)] = {"leaked_tokens_per_event": m["leaked_tokens_per_event"],
                                                               "checks_per_request": m["checks_per_request"]}
        res[wl] = R
    return res


def availability(seed, n=200_000):
    """E9: fail-closed release gate under PDP outages. Outages: alternating renewal process with
    exponential down-times (mean 30 s); a gate that finds the PDP down retries until its budget B."""
    rng = np.random.default_rng(seed + 9000)
    out = {}
    for A in (0.999, 0.995, 0.99):
        down_mean = 30.0
        up_mean = down_mean * A / (1 - A)
        horizon = n * 20.0
        # build outage intervals
        t, iv = 0.0, []
        while t < horizon:
            t += rng.exponential(up_mean)
            d = rng.exponential(down_mean)
            iv.append((t, t + d))
            t += d
        starts = np.array([a for a, _ in iv])
        ends = np.array([b for _, b in iv])
        q = np.sort(rng.uniform(0, horizon, n))
        j = np.searchsorted(starts, q, side="right") - 1
        in_out = (j >= 0) & (q < ends[np.clip(j, 0, None)])
        remaining = np.where(in_out, ends[np.clip(j, 0, None)] - q, 0.0)
        row = {"measured_unavailability": float(in_out.mean())}
        for B in (0.0, 2.0, 10.0):
            blocked = remaining > B
            delayed = in_out & ~blocked
            row[str(B)] = {"false_block_rate": float(blocked.mean()),
                           "delayed_rate": float(delayed.mean()),
                           "mean_delay_when_delayed_s": float(remaining[delayed].mean()) if delayed.any() else 0.0}
        out[str(A)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(s))
        print(f"[{time.time() - t0:6.1f}s] inflight seed {s}", flush=True)
    av = [availability(s) for s in range(a.seeds)]
    (out / "inflight_raw.json").write_text(json.dumps(runs, indent=1))
    (out / "inflight.json").write_text(json.dumps({"sim": aggregate(runs), "availability": aggregate(av)}, indent=1))
    print("wrote", out / "inflight.json")


if __name__ == "__main__":
    main()
