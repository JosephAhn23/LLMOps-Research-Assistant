"""
Distributed FAISS with IVF partitioning + multi-shard search.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import faiss
import numpy as np

logger = logging.getLogger(__name__)

NLIST = 100
NPROBE = 10
N_SHARDS = 4


class DistributedFAISSIndex:
    """
    Sharded IVF index simulating distributed search across N_SHARDS nodes.
    Each shard holds a partition of the corpus - results are merged and re-ranked.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.shards: List[faiss.IndexIVFFlat] = []
        self.shard_metadata: List[List[Dict]] = []
        self.lock = threading.Lock()

    def _build_shard(self, embeddings: np.ndarray, metadata: List[Dict]) -> faiss.IndexIVFFlat:
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, NLIST, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = NPROBE
        return index

    def build(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Partition embeddings across shards."""
        n = len(embeddings)
        shard_size = n // N_SHARDS

        self.shards = []
        self.shard_metadata = []

        for i in range(N_SHARDS):
            start = i * shard_size
            end = n if i == N_SHARDS - 1 else (i + 1) * shard_size
            shard_emb = embeddings[start:end]
            shard_meta = metadata[start:end]

            if len(shard_emb) < NLIST:
                quantizer = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIDMap(quantizer)
                ids = np.arange(len(shard_emb)).astype(np.int64)
                index.add_with_ids(shard_emb, ids)
            else:
                index = self._build_shard(shard_emb, shard_meta)

            self.shards.append(index)
            self.shard_metadata.append(shard_meta)

        logger.info("Built %d shards, ~%d vectors each.", N_SHARDS, shard_size)

    def _search_shard(self, shard_idx: int, query: np.ndarray, top_k: int) -> List[Dict]:
        """Search a single shard - runs in thread pool."""
        index = self.shards[shard_idx]
        metadata = self.shard_metadata[shard_idx]

        scores, indices = index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            chunk = metadata[idx].copy()
            chunk["retrieval_score"] = float(score)
            chunk["shard"] = shard_idx
            results.append(chunk)
        return results

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Parallel search across all shards using ThreadPoolExecutor.
        Merges results and returns global top-k.
        """
        all_results = []

        with ThreadPoolExecutor(max_workers=N_SHARDS) as executor:
            futures = {
                executor.submit(self._search_shard, i, query_vec, top_k): i
                for i in range(len(self.shards))
            }
            for future in as_completed(futures):
                shard_results = future.result()
                all_results.extend(shard_results)

        all_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return all_results[:top_k]

    def save(self, path: str = "data/shards"):
        import json
        import os

        os.makedirs(path, exist_ok=True)
        for i, shard in enumerate(self.shards):
            faiss.write_index(shard, f"{path}/shard_{i}.index")
            with open(f"{path}/shard_{i}.meta.json", "w", encoding="utf-8") as f:
                json.dump(self.shard_metadata[i], f)

    def load(self, path: str = "data/shards"):
        import json
        import os

        self.shards = []
        self.shard_metadata = []
        i = 0
        while os.path.exists(f"{path}/shard_{i}.index"):
            self.shards.append(faiss.read_index(f"{path}/shard_{i}.index"))
            meta_path = f"{path}/shard_{i}.meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    self.shard_metadata.append(json.load(f))
            else:
                self.shard_metadata.append([])
            i += 1
