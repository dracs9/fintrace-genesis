"""Safe calculator (SPEC §8.3)."""

from __future__ import annotations

import pytest

from fintrace.env.calculator import evaluate
from fintrace.env.core import AuditEnv
from fintrace.env.errors import EnvError

GOOD = [
    ("1+2", "3"),
    ("2*3+4", "10"),
    ("2*(3+4)", "14"),
    ("10-4-3", "3"),
    ("7/2", "3.5"),
    ("1/3", "0.3333333333333333333333333333"),
    ("2/3", "0.6666666666666666666666666667"),
    ("0.1+0.2", "0.3"),
    ("-5", "-5"),
    ("+5", "5"),
    ("--5", "5"),
    ("-(2+3)*2", "-10"),
    ("100", "100"),
    ("1e3", "1000"),
    ("2.50*4", "10"),
    ("512.340000*10000", "5123400"),
    ("1000000*512.34", "512340000"),
    ("5-5", "0"),
    ("-0", "0"),
    ("0.0000001*1", "0.0000001"),
    ("1_000+1", "1001"),
    ("1000000000000000000", "1000000000000000000"),
    ("-1000000000000000000", "-1000000000000000000"),
    ("  3 * 3  ", "9"),
    ("(((((1)))))", "1"),
    ("4/2", "2"),
]

BAD = [
    "2**8",
    "7//2",
    "7%2",
    "__import__('os')",
    "__import__('os').system('ls')",
    "x",
    "x+1",
    "abs(-1)",
    "(1).real",
    "'1'+'2'",
    "b'1'",
    "[1,2]",
    "(1,2)",
    "{1:2}",
    "lambda: 1",
    "True+1",
    "None",
    "1j",
    "1 if 1 else 2",
    "1<2",
    "1 and 2",
    "~1",
    "not 1",
    "1<<2",
    "1&1",
    "...",
    "1/0",
    "0/0",
    "1/(2-2)",
    "1000000000000000001",
    "1000000000*1000000000*10",
    "1e18*10/10",
    "1e999",
    "",
    "   ",
    "1+",
    "(1",
    "1;2",
    "a=1",
    "1" + "+1" * 100,          # 201 chars
    "-" * 21 + "1",             # depth 22
    "(" * 25 + "1" + ")" * 25 + "+" + "1+" * 20 + "1",
]


@pytest.mark.parametrize("expression,expected", GOOD)
def test_good_expressions(expression: str, expected: str) -> None:
    assert evaluate(expression) == expected


@pytest.mark.parametrize("expression", BAD)
def test_bad_expressions(expression: str) -> None:
    with pytest.raises(EnvError) as err:
        evaluate(expression)
    assert err.value.code == "CALC_ERROR"


def test_length_boundary() -> None:
    expr = "1" + "+1" * 99 + "+" + "1"  # 201 chars
    assert len(expr) == 201
    with pytest.raises(EnvError):
        evaluate(expr)
    ok = "1" * 18 + "+" + "0" * 181
    assert len(ok) == 200
    assert evaluate(ok) == "1" * 18


def test_depth_boundary() -> None:
    assert evaluate("-" * 19 + "1") == "-1"     # depth 20
    with pytest.raises(EnvError):
        evaluate("-" * 20 + "1")                # depth 21


def test_non_string_rejected() -> None:
    with pytest.raises(EnvError):
        evaluate(5)  # type: ignore[arg-type]


def test_never_scientific_notation() -> None:
    for expr in ("1e-20*1", "123456789*1000000000", "1/1024"):
        assert "e" not in evaluate(expr).lower()


def test_calculator_through_env(started: AuditEnv) -> None:
    obs, *_ = started.step({"tool": "calculator", "args": {"expression": "50004500-50002400"}})
    assert obs["result"] == {"value": "2100"}
    obs, *_ = started.step({"tool": "calculator", "args": {"expression": "2**64"}})
    assert obs["error"] == {"code": "CALC_ERROR", "message": "invalid expression"}
