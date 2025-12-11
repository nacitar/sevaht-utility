from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto, unique
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class NameStyleConfig:
    separator: str  # "_", "-", or ""
    capitalize_first: bool  # capitalize first word?
    capitalize_rest: bool  # capitalize words after first?


@unique
class NameStyle(Enum):
    _config: NameStyleConfig

    SNAKE_CASE = auto(), NameStyleConfig("_", False, False)  # snake_case
    KEBAB_CASE = auto(), NameStyleConfig("-", False, False)  # kebab-case
    CAMEL_CASE = auto(), NameStyleConfig("", False, True)  # camelCase
    PASCAL_CASE = auto(), NameStyleConfig("", True, True)  # PascalCase

    def __new__(cls, value: int, config: NameStyleConfig) -> NameStyle:
        obj = object.__new__(cls)
        obj._value_ = value
        obj._config = config
        return obj

    @property
    def config(self) -> NameStyleConfig:
        return self._config


def split_into_words(name: str) -> list[str]:
    offset: int | None = None
    last = "X"  # anything that isupper() works
    words: list[str] = []
    for i, current in enumerate(chain(name, "-")):
        is_delimiter = current in ("-", "_") or current.isspace()
        is_lower_to_upper = not last.isupper() and current.isupper()
        if is_delimiter or is_lower_to_upper:
            if offset is not None and offset != i:
                words.append(name[offset:i].lower())
            offset = i if is_lower_to_upper else None
        elif offset is None:
            offset = i
        last = current
    return words


def join_words(words: Sequence[str], style: NameStyle) -> str:
    normalized_words = [word.lower() for word in words if word]
    if not normalized_words:
        return ""
    cfg = style.config
    transformed = [
        (
            w.capitalize()
            if (i == 0 and cfg.capitalize_first)
            or (i > 0 and cfg.capitalize_rest)
            else w
        )
        for i, w in enumerate(normalized_words)
    ]
    return cfg.separator.join(transformed)


def convert_name(name: str, style: NameStyle) -> str:
    """Convert a name in any supported style to the given target style."""
    words = split_into_words(name)
    return join_words(words, style)
