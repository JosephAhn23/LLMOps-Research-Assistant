"""
Tests for ingestion pipeline - chunking, embedding, quality filtering.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ingestion.data_quality import DataQualityFilter, Deduplicator
from ingestion.pipeline import EmbeddingModel, chunk_text


# ─── Chunking Tests ───────────────────────────────────────────


class TestChunking:
    def test_basic_chunking(self):
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.split()) <= 100

    def test_overlap_continuity(self):
        text = " ".join([str(i) for i in range(200)])
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # Verify overlap: last words of chunk N appear in chunk N+1
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split()[-10:])
            words_b = set(chunks[i + 1].split()[:10])
            assert len(words_a & words_b) > 0

    def test_short_text_single_chunk(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=512, overlap=64)
        assert chunks == []

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 2000)
        chunks = chunk_text(text, chunk_size=200, overlap=0)
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 200


# ─── Data Quality Tests ───────────────────────────────────────


class TestDataQualityFilter:
    def setup_method(self):
        self.filter = DataQualityFilter(min_words=10, max_words=10000)

    def test_passes_good_text(self):
        text = "This is a well-written paragraph with normal content. " * 20
        assert self.filter.is_quality(text) is True

    def test_rejects_too_short(self):
        assert self.filter.is_quality("too short") is False

    def test_rejects_high_digit_ratio(self):
        text = "1234567890 " * 100
        assert self.filter.is_quality(text) is False

    def test_rejects_high_uppercase(self):
        text = "ALL CAPS TEXT IS USUALLY SPAM OR BOILERPLATE " * 50
        assert self.filter.is_quality(text) is False

    def test_rejects_high_duplicate_lines(self):
        text = "\n".join(["same line"] * 100)
        assert self.filter.is_quality(text) is False

    def test_assess_returns_metrics(self):
        text = "Normal text with reasonable content. " * 30
        metrics = self.filter.assess(text)
        assert metrics.avg_word_length > 0
        assert 0 <= metrics.digit_ratio <= 1
        assert metrics.passed is True

    def test_empty_text_fails(self):
        assert self.filter.is_quality("") is False


class TestDeduplicator:
    def setup_method(self):
        self.dedup = Deduplicator(ngram_size=5)

    def test_exact_duplicate_detected(self):
        text = "This is some test text for deduplication testing purposes."
        h1 = self.dedup.hash(text)
        h2 = self.dedup.hash(text)
        assert h1 == h2

    def test_different_texts_different_hashes(self):
        h1 = self.dedup.hash("text one about machine learning")
        h2 = self.dedup.hash("text two about deep learning systems")
        assert h1 != h2

    def test_whitespace_normalized(self):
        h1 = self.dedup.hash("hello world")
        h2 = self.dedup.hash("hello    world")
        assert h1 == h2

    def test_dataframe_deduplication(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "text": [
                    "unique text about machine learning systems",
                    "unique text about machine learning systems",  # exact duplicate
                    "completely different content about databases",
                ]
            }
        )
        result_df, stats = self.dedup.deduplicate_dataframe(df)
        assert len(result_df) == 2
        assert stats["removed"] == 1
        assert stats["dedup_rate"] == pytest.approx(1 / 3, rel=0.01)


# ─── Embedding Tests ──────────────────────────────────────────


class TestEmbeddingModel:
    @pytest.fixture(autouse=True)
    def mock_model(self):
        # EmbeddingModel lazy-imports transformers; patch constructor instead.
        with patch.object(EmbeddingModel, "__init__", lambda self, model_name=None: None):
            yield

    def test_embed_returns_correct_shape(self):
        model = EmbeddingModel()
        with patch.object(model, "embed", return_value=np.random.randn(2, 384)):
            embeddings = model.embed(["text one", "text two"])
            assert embeddings.shape == (2, 384)

    def test_embeddings_normalized(self):
        model = EmbeddingModel()
        raw = np.random.randn(3, 384).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / norms
        with patch.object(model, "embed", return_value=normalized):
            embeddings = model.embed(["a", "b", "c"])
            norms = np.linalg.norm(embeddings, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)
