# MiniRAG From Scratch (NumPy + Optional OpenAI)

This is a tiny companion project designed to prove fundamentals under abstractions.

- No LangChain
- No vector DB
- No framework orchestration
- Just chunking, TF-IDF retrieval, cosine ranking, and simple answer synthesis

## Why this exists

The main repo shows system breadth. This folder shows I can still build the core RAG loop from first principles in a few hundred lines.

## What it does

1. Splits a small document set into chunks.
2. Builds a TF-IDF matrix with NumPy.
3. Retrieves top-k chunks via cosine similarity.
4. Synthesizes an answer:
   - If `OPENAI_API_KEY` exists, calls OpenAI for a grounded answer.
   - Otherwise returns a deterministic extractive summary from retrieved chunks.

## Run

```bash
python tiny_companion/minirag_numpy_openai/minirag.py --query "How does reranking improve RAG?" --top-k 3
```

Optional model-based synthesis:

```bash
export OPENAI_API_KEY=your_key
python tiny_companion/minirag_numpy_openai/minirag.py --query "How does reranking improve RAG?" --top-k 3 --use-openai
```

## Methodology notes

- Retrieval metric is cosine similarity over TF-IDF vectors.
- Corpus is an embedded sample set (`n=6` short documents) for deterministic demos.
- This is intentionally minimal and educational, not production traffic ready.

