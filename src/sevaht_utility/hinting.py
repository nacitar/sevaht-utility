from __future__ import annotations

from collections import deque
from dataclasses import InitVar
from inspect import signature
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


T = TypeVar("T")


class InvalidTypeError(TypeError):
    """Raised when a value's type does not match the expected type."""

    def __init__(
        self, value: object, *, expected_type: type[T] | UnionType
    ) -> None:
        super().__init__(
            f"Expected: {expected_type}, "
            f"Actual: {type(value)}, "
            f"Value: {value}"
        )
        self.expected_type = expected_type
        self.value = value


def iterate_types(*source_types: type | UnionType) -> Iterator[type]:
    stack = deque(source_types)
    seen: set[type] = set()
    while stack:
        current = stack.popleft()
        if isinstance(current, UnionType) or get_origin(current) is Union:
            stack.extendleft(reversed(get_args(current)))
        elif current not in seen:
            seen.add(current)
            yield current


def verified_cast(expected_type: type[T] | UnionType, value: object) -> T:
    for candidate_type in iterate_types(expected_type):
        if candidate_type in (Any, object) or isinstance(
            value, candidate_type
        ):
            return cast("T", value)
    raise InvalidTypeError(value, expected_type=expected_type)


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
        member: type_hints.get(member, Any)
        for member in signature(function).parameters
        if member != "return"
    }
