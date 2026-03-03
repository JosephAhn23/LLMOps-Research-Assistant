from agents.orchestrator import should_continue


def test_should_continue_routes_on_error() -> None:
    state = {"error": "failed"}
    assert should_continue(state) == "__end__"


def test_should_continue_routes_to_rerank() -> None:
    state = {"error": ""}
    assert should_continue(state) == "rerank"
