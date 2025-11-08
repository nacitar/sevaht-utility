from __future__ import annotations

from collections import deque
from dataclasses import InitVar
from inspect import signature
from types import UnionType
from typing import (
    Any,
    Callable,
    Iterator,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
)

T = TypeVar("T")


def iterate_types(*source_types: type) -> Iterator[type]:
    stack = deque(source_types)
    seen: set[type] = set()
    while stack:
        current = stack.popleft()
        if isinstance(current, UnionType):
            stack.extendleft(reversed(get_args(current)))
        elif current not in seen:
            seen.add(current)
            yield current


def verified_cast(expected_type: type[T], value: Any) -> T:
    if isinstance(value, *iterate_types(expected_type)):
        return cast(T, value)
    raise TypeError(
        f"Expected: {expected_type}, Actual: {type(value)}, Value: {value}"
    )


def get_callable_argument_hints(
    function: Callable[..., Any],
) -> dict[str, type]:
    type_hints = {
        member: (
            member_type
            if not isinstance(member_type, InitVar)
            else member_type.type
        )
        for member, member_type in get_type_hints(function).items()
    }
    return {
        member: type_hints[member]
        for member in signature(function).parameters.keys()
        if member != "return"
    }
