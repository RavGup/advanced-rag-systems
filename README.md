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

## #3 — Fixing Retrieval When the Words Don't Match

A RAG answer is wrong even though the correct document exists. Debug it stage by stage on a 20-document store-policy corpus: vector search, BM25, hybrid search with RRF, synonym expansion, LLM query rewriting, and HyDE — with a scoreboard of gold-document ranks. No API key needed; the LLM steps run on a small local model.

**Run it in your browser:** [Open in Google Colab](https://colab.research.google.com/github/RavGup/advanced-rag-systems/blob/main/advanced-rag-3-debugging-wrong-answers.ipynb)

**The result:** naive vector search ranks the domestic 30-day policy at #1 (the correct 60-day policy sits at #2). Synonym expansion fixes recall, and LLM rewriting puts the right document at #1 — but hybrid RRF and HyDE do not reliably help, and HyDE hallucinates "30 days" into its own hypothetical document.
