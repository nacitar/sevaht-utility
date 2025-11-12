from __future__ import annotations

from dataclasses import InitVar
from typing import Any, Union

import pytest

from sevaht_utility.hinting import (
    get_callable_argument_hints,
    iterate_types,
    verified_cast,
)

# --- iterate_types -----------------------------------------------------------


def test_iterate_types_flattening() -> None:
    # Mixed PEP 604 and typing.Union
    result = list(iterate_types(int | float | Union[str, bytes] | list))
    assert result == [int, float, str, bytes, list]


def test_iterate_types_deduplication() -> None:
    result = list(iterate_types(int | int | (str | str)))
    assert result == [int, str]


def test_iterate_types_nested_unions() -> None:
    nested = int | (float | (str | bytes))
    assert set(iterate_types(nested)) == {int, float, str, bytes}


def test_iterate_types_non_union() -> None:
    result = list(iterate_types(dict))
    assert result == [dict]


# --- verified_cast -----------------------------------------------------------


def test_verified_cast_accepts_valid_type() -> None:
    assert verified_cast(int | str, 5) == 5
    assert verified_cast(int | str, "ok") == "ok"


def test_verified_cast_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        verified_cast(int | str, 1.2)


def test_verified_cast_with_union_and_subtypes() -> None:
    class A:
        pass

    class B(A):
        pass

    assert verified_cast(A | B, A()).__class__ is A
    assert verified_cast(A | B, B()).__class__ is B


# --- get_callable_argument_hints ---------------------------------------------


def _example_func(a: int, b: str, c: InitVar[float]) -> bool:
    return True


def _example_func_no_annotations(x, y):  # type: ignore[no-untyped-def]
    pass


def test_get_callable_argument_hints_extracts_types() -> None:
    hints = get_callable_argument_hints(_example_func)
    assert hints == {"a": int, "b": str, "c": float}


def test_get_callable_argument_hints_ignores_return() -> None:
    assert "return" not in get_callable_argument_hints(_example_func)


def test_get_callable_argument_hints_handles_unannotated() -> None:
    hints = get_callable_argument_hints(_example_func_no_annotations)
    assert hints == {"x": Any, "y": Any}


# --- type-safety smoke check -------------------------------------------------


def test_mypy_type_safety() -> None:
    def typed_func(x: int) -> None:
        assert isinstance(x, int)

    # mypy should see that verified_cast returns int
    val: int = verified_cast(int, 123)
    typed_func(val)
