from unittest.mock import MagicMock, patch

from governance.constitution import (
    ConstitutionalClassifier,
    ConstitutionalResult,
    _parse_score,
    _parse_violations,
)


def test_parse_score_and_violations():
    raw = "SCORE: 88\nVIOLATIONS: too vague, no caveats\nRATIONALE: weak."
    assert _parse_score(raw) == 88.0
    v = _parse_violations(raw)
    assert "too vague" in v[0] or any("vague" in x for x in v)


def test_classifier_uses_openai_mock():
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="SCORE: 95\nVIOLATIONS: NONE\nRATIONALE: ok."))]

    with patch("openai.OpenAI") as m_client:
        m_client.return_value.chat.completions.create.return_value = mock_resp
        clf = ConstitutionalClassifier(model="gpt-4o-mini", pass_threshold=90.0)
        r = clf.grade("A careful answer with [source_1] citation.")
    assert isinstance(r, ConstitutionalResult)
    assert r.score == 95.0
    assert r.passed is True
