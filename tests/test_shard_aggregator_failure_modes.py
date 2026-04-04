import time

from api import shard_aggregator as aggregator


def test_search_one_returns_empty_on_timeout(monkeypatch) -> None:
    def _raise_timeout(*_args, **_kwargs):
        raise aggregator.httpx.ReadTimeout("timeout")

    monkeypatch.setattr(aggregator.httpx, "post", _raise_timeout)

    result = aggregator._search_one("http://shard-a", {"query_vector": [0.1], "top_k": 3})
    assert result == []


def test_aggregator_handles_partial_shard_outage(monkeypatch) -> None:
    monkeypatch.setattr(aggregator, "_shard_urls", lambda: ["a", "b", "c"])

    def _fake_search_one(url: str, _payload: dict) -> list[dict]:
        if url == "b":
            return []
        return [{"source": f"{url}.md", "retrieval_score": 0.7 if url == "a" else 0.9}]

    monkeypatch.setattr(aggregator, "_search_one", _fake_search_one)

    result = aggregator.search(aggregator.SearchRequest(query_vector=[0.2], top_k=2))
    assert [r["source"] for r in result["results"]] == ["c.md", "a.md"]


def test_aggregator_tolerates_latency_spike(monkeypatch) -> None:
    monkeypatch.setattr(aggregator, "_shard_urls", lambda: ["fast", "slow"])

    def _fake_search_one(url: str, _payload: dict) -> list[dict]:
        if url == "slow":
            time.sleep(0.05)
        return [{"source": f"{url}.md", "retrieval_score": 0.8 if url == "slow" else 0.6}]

    monkeypatch.setattr(aggregator, "_search_one", _fake_search_one)

    result = aggregator.search(aggregator.SearchRequest(query_vector=[0.2], top_k=2))
    assert len(result["results"]) == 2
    assert result["results"][0]["source"] == "slow.md"
