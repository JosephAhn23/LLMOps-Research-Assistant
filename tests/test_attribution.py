from agents.attribution import attribute_answer_to_chunks


def test_attribution_links_sentence_to_chunk():
    chunks = [
        {"text": "The RAG pipeline uses retrieval before generation.", "source": "doc.md"},
        {"text": "Unrelated content about cats.", "source": "other.md"},
    ]
    answer = "We use retrieval before generation in our stack. Cats are nice."
    spans = attribute_answer_to_chunks(answer, chunks, speculative_threshold=0.05)
    assert len(spans) >= 1
    rag_span = next(s for s in spans if "retrieval" in s["sentence"].lower())
    assert rag_span["best_source_index"] == 1
    assert rag_span["speculative"] is False


def test_speculative_when_no_overlap():
    chunks = [{"text": "Alpha beta gamma.", "source": "a"}]
    answer = "The quantum chromodynamics result is exactly eleven point four."
    spans = attribute_answer_to_chunks(answer, chunks, speculative_threshold=0.2)
    assert spans
    assert all(s["speculative"] for s in spans)
