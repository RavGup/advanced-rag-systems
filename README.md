# Advanced RAG Systems

Hands-on demos from the *Inside the LLM* series — naive retrieval vs the production playbook, with real scores.

## #1 — Hybrid Search + Reranking

Naive vector search vs BM25 (keyword) + dense hybrid search, Reciprocal Rank Fusion, and cross-encoder reranking — on an 11-chunk fictional knowledge base (Nimbus Cloud), with deliberate traps.

**Run it in your browser:** [Open in Google Colab](https://colab.research.google.com/github/RavGup/advanced-rag-systems/blob/main/advanced-rag-1-hybrid-rerank.ipynb)

Or locally:

```
pip install sentence-transformers rank-bm25
python demo_rag.py
```

**The result:** naive search ranks an FAQ with no numbers at #1 (the correct chunk sits at #2). The advanced pipeline puts the chunk containing the answer — 25% service credit — at #1.
