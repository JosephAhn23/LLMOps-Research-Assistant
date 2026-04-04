from unittest.mock import MagicMock

from sandbox.code_sandbox import ExecutionResult
from sandbox.validation_loop import ValidationLoop, default_claim_matcher


def test_default_claim_matcher_numeric():
    assert default_claim_matcher("The result is 3.14", "computed: 3.14\n", "")
    assert not default_claim_matcher("The result is 3.14", "computed: 2.71\n", "")


def test_validation_loop_verifies_after_revision():
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        ExecutionResult(exit_code=1, stdout="", stderr="Error", language="python"),
        ExecutionResult(exit_code=0, stdout="3.14", stderr="", language="python"),
    ]

    def revise(_code: str, _fb: str) -> str:
        return "print(3.14)"

    loop = ValidationLoop(sandbox=sandbox, max_rounds=3)
    r = loop.run("The result is 3.14", "print(0)", revise)
    assert r.claim_verified is True
    assert r.rounds_used == 2


def test_validation_loop_exhausts_rounds():
    sandbox = MagicMock()
    sandbox.run.return_value = ExecutionResult(
        exit_code=0, stdout="wrong", stderr="", language="python"
    )

    def revise(_c: str, _f: str) -> str:
        return "print('wrong')"

    loop = ValidationLoop(sandbox=sandbox, max_rounds=2)
    r = loop.run("expect 99", "print('wrong')", revise)
    assert r.claim_verified is False
    assert r.rounds_used == 2
