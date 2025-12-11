from __future__ import annotations

from dataclasses import InitVar  # noqa: TC003
from typing import Any, Union

import pytest

from sevaht_utility.hinting import (
    InvalidTypeError,
    get_callable_argument_hints,
    iterate_types,
    verify_type,
)

# --- iterate_types -----------------------------------------------------------


def test_iterate_types_flattening() -> None:
    # Mixed PEP 604 and typing.Union
    result = list(
        iterate_types(int | float | Union[str, bytes] | list)  # noqa: UP007
    )
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


# --- verify_type -----------------------------------------------------------


def test_verify_type_accepts_valid_type() -> None:
    assert verify_type(int | str, 5) == 5
    assert verify_type(int | str, "ok") == "ok"


def test_verify_type_rejects_invalid_type() -> None:
    with pytest.raises(InvalidTypeError):
        verify_type(int | str, 1.2)


def test_verify_type_with_union_and_subtypes() -> None:
    class A:
        pass

    class B(A):
        pass

    assert verify_type(A | B, A()).__class__ is A
    assert verify_type(A | B, B()).__class__ is B


# --- get_callable_argument_hints ---------------------------------------------


def _example_func(a: int, b: str, c: InitVar[float]) -> bool:  # noqa: ARG001
    return True


def _example_undecorated_func(  # type: ignore[no-untyped-def]  # noqa: ANN202
    x, y  # noqa: ANN001
):
    pass


def test_get_callable_argument_hints_extracts_types() -> None:
    hints = get_callable_argument_hints(_example_func)
    assert hints == {"a": int, "b": str, "c": float}


def test_get_callable_argument_hints_ignores_return() -> None:
    assert "return" not in get_callable_argument_hints(_example_func)


def test_get_callable_argument_hints_handles_unannotated() -> None:
    hints = get_callable_argument_hints(_example_undecorated_func)
    assert hints == {"x": Any, "y": Any}
