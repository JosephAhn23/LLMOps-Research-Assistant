from ingestion.pipeline import chunk_text


def test_chunk_text_with_overlap() -> None:
    chunks = chunk_text("one two three four five six seven", chunk_size=3, overlap=1)
    assert len(chunks) >= 3
    assert chunks[0] == "one two three"
