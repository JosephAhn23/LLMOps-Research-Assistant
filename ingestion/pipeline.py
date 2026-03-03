"""
Document ingestion pipeline - chunking, embedding, FAISS indexing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss.index")
META_PATH = os.getenv("META_PATH", "data/metadata.json")


class EmbeddingModel:
    """
    Raw HuggingFace transformer with mean-pool + L2-norm embeddings.

    Mean pooling over all non-padding tokens is the recommended strategy for
    sentence-transformers models (e.g. all-MiniLM-L6-v2) and produces
    embeddings that are consistent with the retrieval query embeddings used
    by RetrieverAgent.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        self._torch = torch
        self._F = F
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask) -> "torch.Tensor":
        import torch

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**encoded)
                embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
                embeddings = self._F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        import numpy as _np
        return _np.vstack(all_embeddings).astype(np.float32)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


class IngestionPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        Path("data").mkdir(exist_ok=True)

    def ingest_documents(self, docs: List[Dict[str, str]]):
        """docs: list of {"id": ..., "text": ..., "source": ...}"""
        all_chunks = []
        all_meta = []

        for doc in docs:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{doc['id']}_{i}".encode()).hexdigest()
                all_chunks.append(chunk)
                all_meta.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc["id"],
                    "source": doc.get("source", ""),
                    "chunk_index": i,
                    "text": chunk,
                })

        logger.info("Embedding %d chunks...", len(all_chunks))
        embeddings = self.embedder.embed(all_chunks)

        dim = embeddings.shape[1]
        import faiss

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            # Append to the existing index so prior documents are preserved.
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, encoding="utf-8") as f:
                existing_meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
            existing_meta = []

        self.index.add(embeddings)
        self.metadata = existing_meta + all_meta

        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        logger.info(
            "Indexed %d new chunks (total: %d) into FAISS.",
            len(all_chunks),
            self.index.ntotal,
        )

    def load(self):
        import faiss

        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)
