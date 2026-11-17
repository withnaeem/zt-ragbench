# ZT-RAGBench

Code and data generators for *Authorization Across the Enterprise RAG Request Lifecycle: An Empirical Zero-Trust Evaluation of Retrieval-Time and Release-Time Data Leakage* (Naeem Shahzad).

ZT-RAGBench is a synthetic multi-tenant enterprise corpus with attribute-based access-control (ABAC) policies. It compares six RAG access-control architectures (A0–A5) under five experiments:

| Exp | What it measures |
|-----|------------------|
| E1 `main_tfidf`, `main_lsa` | Exposure, attack success, Hit@5 and policy overhead for A0–A5 on a static corpus |
| E2 `revocation` | Leakage caused by the ACL snapshot sync interval after permissions are revoked |
| E3 `cache` | Cross-user leakage through a semantic response cache |
| E4 `derived` | Leakage from derived artefacts (digests) that inherit only their primary source's label |
| E5 `labelnoise` | Residual risk from mislabelled documents, with and without label/content checks |
| E6–E9 `inflight.py` | In-flight revocation: entitlement TTL, release recheck vs atomic vs fenced gate, replica lag, provenance gaps, streaming leases, PDP outages |
| E10 `proto.py` | Multithreaded prototype of B0 / B1 / fenced commit, with an offline auditor that replays the event log |

## Reproduce

```bash
pip install numpy scikit-learn matplotlib
python -m ztrag.experiments --seeds 5 --out results   # ~6 min on a laptop CPU
python -m ztrag.inflight --seeds 5 --out results      # ~6 min
python -m ztrag.proto --reps 5 --out results          # ~4 min
python -m ztrag.figures                               # writes paper/figs/*.pdf
```

`results/results.json` holds the means and 95% confidence intervals reported in the paper. `results/*_raw.json` holds the per-seed values.

## Layout

- `ztrag/core.py`: corpus and user generation, the ABAC policy, retrievers (TF-IDF, LSA) and the label/content check
- `ztrag/experiments.py`: architectures A0–A5, workloads, experiments E1–E5 and aggregation
- `ztrag/figures.py`: paper figures
- `ztrag/inflight.py`, `ztrag/proto.py`: release-stage experiments
- `paper2/`: LaTeX source of the paper

All data is synthetic. No real personal or company data is used.
