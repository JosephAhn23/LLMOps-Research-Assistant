from mlops.evaluation import EvalSample, retrieval_hit_rate


def test_retrieval_hit_rate() -> None:
    samples = [EvalSample(question='Q', answer='A', contexts=['the model uses redis cache'], expected='redis cache')]
    assert retrieval_hit_rate(samples) == 1.0
