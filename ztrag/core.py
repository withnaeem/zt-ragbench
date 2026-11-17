"""ZT-RAGBench core: synthetic multi-tenant enterprise corpus, ABAC policy,
retrievers and access-control architectures for RAG.

Everything is deterministic given a seed. No external data or models are used.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
DEPTS = ["hr", "finance", "legal", "engineering", "sales", "executive"]
DEPT_WORDS = {
    "hr": "salary compensation appraisal hiring termination benefits leave payroll grievance onboarding "
          "promotion disciplinary recruitment headcount retention attrition bonus pension medical insurance "
          "probation contract employee manager succession training wellbeing overtime allowance relocation "
          "visa sponsorship iqama gosi saudization interview offer resignation investigation complaint".split(),
    "finance": "budget forecast revenue ledger audit invoice capex opex cashflow liquidity treasury receivable "
               "payable accrual depreciation ebitda margin variance reconciliation tax zakat vat procurement "
               "expense reimbursement quarterly annual statement balance profit loss impairment covenant loan "
               "hedging dividend provision writeoff valuation".split(),
    "legal": "contract litigation nda compliance settlement clause indemnity liability arbitration regulator "
             "subpoena counsel privilege dispute breach warranty jurisdiction termination amendment license "
             "intellectual patent trademark sanction antitrust whistleblower investigation consent notice "
             "agreement obligation penalty court filing memorandum".split(),
    "engineering": "deployment kubernetes latency api incident architecture microservice database migration "
                   "pipeline release rollback outage postmortem endpoint gateway cluster scaling vulnerability "
                   "patch credential secret token certificate firewall telemetry logging monitoring backup "
                   "replication schema refactor sprint backlog testing staging production".split(),
    "sales": "pipeline quota client deal discount renewal prospect proposal pricing territory commission "
             "forecast opportunity churn upsell crosssell tender rfp negotiation account partner reseller "
             "margin target lead conversion campaign demo contract signature handover onboarding "
             "satisfaction escalation competitor".split(),
    "executive": "acquisition board strategy merger reorganization valuation investor shareholder roadmap "
                 "restructuring layoff divestment ipo synergy governance succession committee resolution "
                 "confidential partnership expansion vision transformation initiative jointventure "
                 "duediligence term sheet funding capital allocation risk appetite".split(),
}
GENERIC = ("update review meeting plan status report team project timeline action item owner deadline "
           "summary note draft final approved pending discussion decision follow next week month quarter "
           "priority issue risk scope request document process policy").split()
PROJECTS = ["falcon", "orion", "atlas", "nimbus", "sahara", "najm", "qamar", "dune", "oasis", "zephyr",
            "cobalt", "onyx", "aurora", "vertex", "summit", "harbor", "mirage", "sapphire", "titan", "nova"]
LEVELS = {0: "Internal", 1: "Confidential", 2: "Restricted"}

PII_PATTERNS = [
    re.compile(r"\bnid_[12]\d{9}\b"),          # national-ID-like
    re.compile(r"\biban_sa\d{22}\b"),          # IBAN-like
    re.compile(r"\bsar_\d{4,6}\b"),            # salary-like amount
]


@dataclass
class Corpus:
    texts: list            # chunk texts
    chunk_doc: np.ndarray  # chunk -> source doc id (primary); derived chunks have -1
    chunk_sources: list    # chunk -> tuple of source doc ids (len 1 for ordinary chunks)
    doc_tenant: np.ndarray
    doc_dept: np.ndarray
    doc_level: np.ndarray
    doc_acl: np.ndarray    # uint64 bitmask of project groups (restricted docs)
    doc_keys: list         # doc -> list of unique key terms
    doc_pii: np.ndarray    # bool: doc carries PII-like fields
    n_docs: int


@dataclass
class Users:
    tenant: np.ndarray
    dept: np.ndarray
    clearance: np.ndarray  # 0 internal, 1 confidential, 2 restricted
    groups: np.ndarray     # uint64 bitmask

    def copy(self):
        return Users(self.tenant.copy(), self.dept.copy(), self.clearance.copy(), self.groups.copy())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _pii_tokens(rng):
    nid = f"nid_{rng.integers(1, 3)}{rng.integers(0, 10**9):09d}"
    iban = "iban_sa" + "".join(str(d) for d in rng.integers(0, 10, 22))
    sal = f"sar_{rng.integers(4000, 99999)}"
    return [nid, iban, sal]


def generate_corpus(rng, n_tenants=3, docs_per_tenant=2000, chunks_per_doc=3, words_per_chunk=30,
                    n_groups=40, pii_rate_sensitive=0.5, pii_rate_internal=0.02,
                    level_probs=(0.5, 0.3, 0.2)):
    n_docs = n_tenants * docs_per_tenant
    doc_tenant = np.repeat(np.arange(n_tenants), docs_per_tenant)
    doc_dept = rng.integers(0, len(DEPTS), n_docs)
    doc_level = rng.choice(3, n_docs, p=level_probs)
    doc_acl = np.zeros(n_docs, dtype=np.uint64)
    for d in np.where(doc_level == 2)[0]:
        for g in rng.choice(n_groups, rng.integers(1, 3), replace=False):
            doc_acl[d] |= np.uint64(1) << np.uint64(g)
    sens = doc_level >= 1
    doc_pii = np.where(sens, rng.random(n_docs) < pii_rate_sensitive, rng.random(n_docs) < pii_rate_internal)

    texts, chunk_doc, chunk_sources, doc_keys = [], [], [], []
    for d in range(n_docs):
        dept = DEPTS[doc_dept[d]]
        neighbour = DEPTS[(doc_dept[d] + 1) % len(DEPTS)]
        proj = PROJECTS[rng.integers(len(PROJECTS))]
        keys = [f"{proj}_{rng.integers(100, 999)}{chr(97 + rng.integers(26))}", f"t{doc_tenant[d]}client_{rng.integers(0, 400)}"]
        doc_keys.append(keys)
        pii = _pii_tokens(rng) if doc_pii[d] else []
        for c in range(chunks_per_doc):
            n_dept = int(words_per_chunk * 0.6)
            words = list(rng.choice(DEPT_WORDS[dept], n_dept))
            words += list(rng.choice(DEPT_WORDS[neighbour], 3))
            words += list(rng.choice(GENERIC, words_per_chunk - n_dept - 3 - 2))
            words += [proj, keys[c % 2]]
            if c == 0 or (c == 1 and chunks_per_doc > 2):
                words.append(keys[(c + 1) % 2])
            if pii and c == 1:
                words += pii
            rng.shuffle(words)
            texts.append(" ".join(words))
            chunk_doc.append(d)
            chunk_sources.append((d,))
    return Corpus(texts, np.array(chunk_doc), chunk_sources, doc_tenant, doc_dept, doc_level, doc_acl,
                  doc_keys, doc_pii, n_docs)


def add_derived_artifacts(rng, corpus: Corpus, frac=0.15):
    """Append 'digest' chunks that merge content from 2-4 source documents of the same tenant
    (e.g. auto-generated meeting digests or summaries)."""
    n_new = int(frac * corpus.n_docs)
    texts = list(corpus.texts)
    chunk_doc = list(corpus.chunk_doc)
    chunk_sources = list(corpus.chunk_sources)
    doc_chunks = {}
    for i, d in enumerate(corpus.chunk_doc):
        doc_chunks.setdefault(int(d), []).append(i)
    n_tenants = int(corpus.doc_tenant.max()) + 1
    for _ in range(n_new):
        t = rng.integers(n_tenants)
        pool = np.where(corpus.doc_tenant == t)[0]
        srcs = rng.choice(pool, rng.integers(2, 5), replace=False)
        words = []
        for s in srcs:
            src_words = corpus.texts[doc_chunks[int(s)][rng.integers(len(doc_chunks[int(s)]))]].split()
            words += list(rng.choice(src_words, 12, replace=False))
        words += list(rng.choice(GENERIC, 4))
        rng.shuffle(words)
        texts.append(" ".join(words))
        chunk_doc.append(int(srcs[0]))      # primary source (what naive pipelines inherit)
        chunk_sources.append(tuple(int(s) for s in srcs))
    return Corpus(texts, np.array(chunk_doc), chunk_sources, corpus.doc_tenant, corpus.doc_dept,
                  corpus.doc_level, corpus.doc_acl, corpus.doc_keys, corpus.doc_pii, corpus.n_docs)


def generate_users(rng, n_tenants=3, users_per_tenant=100, n_groups=40, clearance_probs=(0.5, 0.35, 0.15)):
    n = n_tenants * users_per_tenant
    tenant = np.repeat(np.arange(n_tenants), users_per_tenant)
    dept = rng.integers(0, len(DEPTS), n)
    clearance = rng.choice(3, n, p=clearance_probs)
    groups = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        for g in rng.choice(n_groups, rng.integers(0, 4), replace=False):
            groups[u] |= np.uint64(1) << np.uint64(g)
    return Users(tenant, dept, clearance, groups)


# ---------------------------------------------------------------------------
# Policy (ABAC). Ground truth = this function evaluated on LIVE attributes and TRUE labels.
# ---------------------------------------------------------------------------
def doc_auth_mask(u, users: Users, doc_tenant, doc_dept, doc_level, doc_acl):
    """Boolean vector over documents: may user u read each document?"""
    same_t = doc_tenant == users.tenant[u]
    internal = doc_level == 0
    conf = (doc_level == 1) & (doc_dept == users.dept[u]) & (users.clearance[u] >= 1)
    restr = (doc_level == 2) & ((doc_acl & users.groups[u]) != 0) & (users.clearance[u] >= 2)
    return same_t & (internal | conf | restr)


def chunk_auth_truth(doc_mask, corpus: Corpus):
    """A chunk is authorised iff the user may read ALL of its source documents."""
    m = doc_mask[corpus.chunk_doc].copy()
    for i, srcs in enumerate(corpus.chunk_sources):
        if len(srcs) > 1:
            m[i] = all(doc_mask[s] for s in srcs)
    return m


def chunk_auth_primary(doc_mask, corpus: Corpus):
    """Naive label inheritance: a derived chunk inherits the primary source's ACL."""
    return doc_mask[corpus.chunk_doc]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class Index:
    def __init__(self, texts, kind="tfidf", dim=256, seed=0):
        self.kind = kind
        self.vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        if kind == "tfidf":
            self.X = normalize(X).tocsr()
            self.svd = None
        else:  # dense proxy: LSA embeddings
            self.svd = TruncatedSVD(dim, random_state=seed)
            self.X = normalize(self.svd.fit_transform(X))

    def embed(self, queries):
        Q = self.vec.transform(queries)
        if self.svd is None:
            return normalize(Q)
        return normalize(self.svd.transform(Q))

    def scores(self, Q):
        S = Q @ self.X.T
        return S.toarray() if hasattr(S, "toarray") else np.asarray(S)


def topk_masked(scores, mask, k):
    s = np.where(mask, scores, -np.inf)
    idx = np.argpartition(-s, k)[:k]
    idx = idx[np.argsort(-s[idx])]
    return idx[np.isfinite(s[idx])]


def topk(scores, k):
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


def label_content_check(texts, chunk_ids, stored_level_of_chunk, quarantined_chunk=None):
    """DLP-style label/content consistency check.

    Chunk-level (query time): block a retrieved chunk labelled Internal that itself contains PII-like
    fields. Document-level (ingestion time): if `quarantined_chunk` is given, also block every chunk of
    a document in which ANY chunk was flagged, so the inconsistency propagates to the whole document.
    """
    keep = []
    for c in chunk_ids:
        if stored_level_of_chunk[c] == 0:
            if quarantined_chunk is not None and quarantined_chunk[c]:
                continue
            if any(p.search(texts[c]) for p in PII_PATTERNS):
                continue
        keep.append(c)
    return np.array(keep, dtype=int)


def ingestion_quarantine(corpus, stored_levels):
    """Flag documents stored as Internal although some chunk carries PII-like fields; return a
    per-chunk boolean (chunk belongs to a flagged document)."""
    flagged = np.zeros(corpus.n_docs, dtype=bool)
    for i, t in enumerate(corpus.texts):
        if len(corpus.chunk_sources[i]) == 1 and any(p.search(t) for p in PII_PATTERNS):
            flagged[corpus.chunk_doc[i]] = True
    flagged &= stored_levels == 0
    return np.array([any(flagged[s] for s in srcs) for srcs in corpus.chunk_sources])
