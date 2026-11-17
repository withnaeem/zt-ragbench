"""E10: multithreaded prototype of the release gate.

Real threads, real interleavings. An authority store serialises policy writes and release commits
under one lock and stamps each with a global sequence number, so an offline auditor can decide
exactly whether a release was committed after a revocation of one of its dependencies.

Strategies:
  B0  retrieval-time check only
  B1  non-atomic recheck: check under the lock, release the lock, finish post-processing, commit
  P   fenced commit: same as B1, but the commit re-validates the dependencies inside the same
      critical section that orders revocations (linearisable check-and-commit)

A revoker thread adversarially revokes a dependency of a randomly chosen in-flight request.
Run: python -m ztrag.proto --reps 5
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import numpy as np


class Authority:
    def __init__(self, n_users, n_docs, rng):
        self.lock = threading.Lock()
        self.seq = 0
        self.allowed = [set(rng.sample(range(n_docs), n_docs // 2)) for _ in range(n_users)]
        self.revocations = []   # (seq, user, doc)
        self.releases = []      # (seq, user, deps, strategy)
        self.blocks = 0

    def _next(self):
        self.seq += 1
        return self.seq

    def check(self, u, deps):
        with self.lock:
            self._next()
            return all(d in self.allowed[u] for d in deps)

    def revoke(self, u, d):
        with self.lock:
            if d in self.allowed[u]:
                self.allowed[u].discard(d)
                self.revocations.append((self._next(), u, d))

    def regrant(self, u, d):
        with self.lock:
            self.allowed[u].add(d)
            self._next()

    def commit(self, u, deps, strat):
        with self.lock:
            self.releases.append((self._next(), u, tuple(deps), strat))

    def fenced_commit(self, u, deps, strat):
        with self.lock:
            if not all(d in self.allowed[u] for d in deps):
                self.blocks += 1
                return False
            self.releases.append((self._next(), u, tuple(deps), strat))
            return True


def run(strategy, n_threads=32, duration=10.0, seed=0, gen_ms=(5, 50), post_ms=(1, 10), revoke_every_ms=2.0,
        n_users=200, n_docs=500):
    rng = random.Random(seed)
    A = Authority(n_users, n_docs, rng)
    inflight = {}
    inflight_lock = threading.Lock()
    stop = threading.Event()
    gate_times = []
    gt_lock = threading.Lock()

    def worker(wid):
        r = random.Random(seed * 1000 + wid)
        while not stop.is_set():
            u = r.randrange(n_users)
            with A.lock:
                pool = list(A.allowed[u])
            if len(pool) < 3:
                continue
            deps = r.sample(pool, 3)
            if not A.check(u, deps):          # retrieval-time check
                continue
            key = (wid, time.perf_counter_ns())
            with inflight_lock:
                inflight[key] = (u, deps)
            time.sleep(r.uniform(*gen_ms) / 1000)        # generation
            t0 = time.perf_counter()
            if strategy == "B0":
                time.sleep(r.uniform(*post_ms) / 1000)
                A.commit(u, deps, strategy)
            elif strategy == "B1":
                ok = A.check(u, deps)
                time.sleep(r.uniform(*post_ms) / 1000)    # post-processing between check and commit
                if ok:
                    A.commit(u, deps, strategy)
                else:
                    with A.lock:
                        A.blocks += 1
            elif strategy == "P":
                ok = A.check(u, deps)
                time.sleep(r.uniform(*post_ms) / 1000)
                if ok:
                    A.fenced_commit(u, deps, strategy)
                else:
                    with A.lock:
                        A.blocks += 1
            dt = time.perf_counter() - t0
            with gt_lock:
                gate_times.append(dt)
            with inflight_lock:
                inflight.pop(key, None)

    def revoker():
        r = random.Random(seed + 7)
        while not stop.is_set():
            time.sleep(revoke_every_ms / 1000)
            with inflight_lock:
                if not inflight:
                    continue
                u, deps = r.choice(list(inflight.values()))
            d = r.choice(deps)
            A.revoke(u, d)
            # restore access later so the population does not drain
            threading.Timer(0.2, A.regrant, args=(u, d)).start()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    threads.append(threading.Thread(target=revoker, daemon=True))
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    time.sleep(0.3)

    return A, gate_times


def run_exact(strategy, **kw):
    """Same as run(), but logs re-grants so the auditor replays exact state."""
    orig = Authority.regrant

    def logged_regrant(self, u, d):
        with self.lock:
            self.allowed[u].add(d)
            self.revocations.append((self._next(), u, -d - 1))   # negative marks re-grant

    Authority.regrant = logged_regrant
    try:
        A, gate_times = run(strategy, **kw)
    finally:
        Authority.regrant = orig
    events = [(s, "rev", u, d) for s, u, d in A.revocations] + [(s, "rel", u, deps) for s, u, deps, _ in A.releases]
    events.sort(key=lambda e: e[0])
    revoked = set()
    unsafe = 0
    for s, kind, u, x in events:
        if kind == "rev":
            if x >= 0:
                revoked.add((u, x))
            else:
                revoked.discard((u, -x - 1))
        else:
            if any((u, d) in revoked for d in x):
                unsafe += 1
    n_rel = len(A.releases)
    n_rev = sum(1 for _, _, d in A.revocations if d >= 0)
    gt = np.array(gate_times) * 1000
    return {"releases": n_rel, "revocations": n_rev, "unsafe": unsafe, "blocks": A.blocks,
            "unsafe_per_10k_releases": 1e4 * unsafe / max(n_rel, 1),
            "unsafe_per_100_revocations": 100 * unsafe / max(n_rev, 1),
            "gate_ms_p50": float(np.median(gt)), "gate_ms_p95": float(np.percentile(gt, 95)),
            "throughput_rps": n_rel / kw.get("duration", 10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    res = {}
    for strat in ("B0", "B1", "P"):
        res[strat] = [run_exact(strat, seed=s, duration=a.duration) for s in range(a.reps)]
        print(strat, [r["unsafe"] for r in res[strat]], flush=True)
    scale = {}
    for nt in (8, 32, 128):
        scale[str(nt)] = [run_exact("P", seed=s, duration=a.duration / 2, n_threads=nt) for s in range(3)]
        print("P threads", nt, [round(r["gate_ms_p95"], 2) for r in scale[str(nt)]], flush=True)
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "proto.json").write_text(json.dumps({"strategies": res, "scaling": scale}, indent=1))


if __name__ == "__main__":
    main()
