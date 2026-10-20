#!/usr/bin/env python3
"""Advanced RAG demo: naive dense retrieval vs hybrid (BM25+dense, RRF) + cross-encoder rerank.
Corpus is a fictional Nimbus Cloud knowledge base with deliberate traps.
Run: python3 demo_rag.py
"""
import json, math, numpy as np

CORPUS = [
    ("sla_standard",
     "Nimbus Cloud SLA: Standard plan customers receive a 10% service credit when monthly "
     "downtime exceeds 1 hour. Credits are applied automatically to the next invoice. "
     "The Standard plan SLA covers compute and storage services only."),
    ("sla_enterprise",
     "Nimbus Cloud SLA: Enterprise tier customers receive a 25% service credit when monthly "
     "downtime exceeds 4 hours. The Enterprise SLA covers compute, storage, and networking. "
     "Claims must be filed within 30 days of the incident."),
    ("enterprise_support",
     "Enterprise support includes 24/7 phone support with a 15-minute response time for "
     "critical issues. Enterprise customers get a dedicated account manager and quarterly "
     "business reviews."),
    ("refund_policy",
     "Refund policy: monthly plans can be refunded within 7 days of purchase, annual plans "
     "within 30 days. Enterprise contracts are non-refundable except as stated in the SLA."),
    ("sla_pro",
     "Nimbus Cloud SLA: Pro plan customers receive a 15% service credit when monthly "
     "downtime exceeds 2 hours. Credits are applied automatically to the next invoice."),
    ("downtime_definition",
     "Downtime is measured as minutes when the service returns 5xx errors for over 60 "
     "consecutive seconds, excluding scheduled maintenance windows announced 72 hours in advance."),
    ("status_page",
     "Check status.nimbus.example for real-time availability. Historical uptime for the last "
     "90 days is published there with per-region breakdowns."),
    ("credits_process",
     "Service credits are applied to the next invoice automatically. Credits cannot be "
     "converted to cash refunds under any plan."),
    ("sla_overview",
     "Nimbus Cloud's SLA program reflects our commitment to reliability. Every plan includes "
     "service level objectives for uptime, measured monthly. Enterprise customers enjoy "
     "enhanced SLA terms with higher service credits for extended outages. See the "
     "plan-specific pages for exact percentages."),
    ("outage_postmortem",
     "Postmortem: the March 3 enterprise outage lasted 4.5 hours and SLA credits were issued "
     "per policy. Root cause was a load balancer misconfiguration. No action needed."),
    ("sla_faq",
     "SLA FAQ: what is an SLA refund? Service credits compensate customers for downtime. "
     "How do I claim? File a claim within 30 days. Which plans are covered? All paid plans "
     "include an SLA."),
]

QUERY = "What is the SLA refund percentage for enterprise downtime exceeding 4 hours?"
CORRECT_ID = "sla_enterprise"
CORRECT_ANSWER = "25% service credit"

def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def show(title, order, score_fn):
    print(f"\n=== {title} ===")
    for r, i in enumerate(order):
        cid, _ = CORPUS[i]
        mark = "  <-- CORRECT" if cid == CORRECT_ID else ""
        print(f"  #{r+1} {cid:20s} score={score_fn(i):.4f}{mark}")

def main():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi

    texts = [t for _, t in CORPUS]
    tok = [t.lower().split() for t in texts]
    qtok = QUERY.lower().split()

    print("loading models...")
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    E = emb.encode(texts, normalize_embeddings=True)
    q = emb.encode([QUERY], normalize_embeddings=True)[0]

    # ---- naive: dense cosine top-3 ----
    cos = E @ q
    naive = list(np.argsort(-cos))[:3]
    show("NAIVE RAG (dense cosine, top-3)", naive, lambda i: float(cos[i]))

    # ---- advanced: BM25 + dense -> RRF -> cross-encoder rerank ----
    bm25 = BM25Okapi(tok)
    b = bm25.get_scores(qtok)
    top_bm25 = list(np.argsort(-b))[:8]
    top_dense = list(np.argsort(-cos))[:8]
    fused = rrf([top_bm25, top_dense])[:8]

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(QUERY, texts[i]) for i in fused]
    rs = reranker.predict(pairs)
    reranked = [fused[i] for i in np.argsort(-rs)]
    show("ADVANCED RAG (hybrid RRF -> cross-encoder rerank, top-3)", reranked[:3],
         lambda i: float(rs[list(fused).index(i)]) if i in fused else 0.0)

    print("\n--- what the LLM would see (advanced top-1) ---")
    cid, txt = CORPUS[reranked[0]]
    print(f"[{cid}] {txt}")
    print(f"\nNaive top-1 was [{CORPUS[naive[0]][0]}]: talks about SLA refunds, "
          f"contains no percentage -> the LLM rambles or punts.")
    print(f"Advanced top-1 [{cid}] contains the answer: {CORRECT_ANSWER}.")

if __name__ == "__main__":
    main()
