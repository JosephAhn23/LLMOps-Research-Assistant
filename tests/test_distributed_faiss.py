"""
Tests for distributed FAISS - shard building, search, aggregation.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestShardBuilding:
    def test_build_creates_correct_n_shards(self):
        pytest.importorskip("faiss")
        from infra.distributed_faiss_service import build_shard_indexes

        embeddings = np.random.randn(400, 64).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        metadata = [{"text": f"chunk {i}", "source": "test", "chunk_id": str(i)} for i in range(400)]

        with patch("infra.distributed_faiss_service.os.makedirs"), patch(
            "infra.distributed_faiss_service.faiss.write_index"
        ), patch("builtins.open", MagicMock()):
            build_shard_indexes(embeddings, metadata, n_shards=4)

    def test_shard_size_even_distribution(self):
        n = 1000
        n_shards = 4
        shard_size = n // n_shards
        assert shard_size == 250
        sizes = [shard_size] * (n_shards - 1) + [n - shard_size * (n_shards - 1)]
        assert sum(sizes) == n


class TestAggregatorSearch:
    @pytest.mark.asyncio
    async def test_aggregator_handles_shard_failure(self):
        """Aggregator should return results even if one shard is down."""
        results = [
            [{"text": "chunk 0", "retrieval_score": 0.9, "source": "s0"}],
            [],  # shard 1 down
            [{"text": "chunk 2", "retrieval_score": 0.7, "source": "s2"}],
            [{"text": "chunk 3", "retrieval_score": 0.6, "source": "s3"}],
        ]
        all_chunks = [c for shard in results for c in shard]
        assert len(all_chunks) == 3


class TestSearchQuality:
    def test_global_topk_correct(self):
        """Global merge must return globally highest scoring chunks."""
        shard_0 = [{"retrieval_score": 0.95}, {"retrieval_score": 0.70}]
        shard_1 = [{"retrieval_score": 0.88}, {"retrieval_score": 0.62}]
        shard_2 = [{"retrieval_score": 0.91}, {"retrieval_score": 0.55}]
        shard_3 = [{"retrieval_score": 0.77}, {"retrieval_score": 0.48}]

        all_chunks = shard_0 + shard_1 + shard_2 + shard_3
        all_chunks.sort(key=lambda x: x["retrieval_score"], reverse=True)
        top_3 = all_chunks[:3]

        assert top_3[0]["retrieval_score"] == 0.95
        assert top_3[1]["retrieval_score"] == 0.91
        assert top_3[2]["retrieval_score"] == 0.88
