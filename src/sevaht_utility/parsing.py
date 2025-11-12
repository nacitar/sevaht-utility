from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from functools import cache
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TextIO,
    TypeVar,
    cast,
    overload,
)

from .hinting import get_callable_argument_hints, iterate_types, verified_cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
TextProvider = str | Path | TextIO | list[str]
StringConverter = Callable[[str], Any]


def get_text(source: TextProvider) -> str:
    """Return the full text from any supported TextProvider."""
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return os.linesep.join(source)
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    return source.read()  # TextIO


@contextmanager
def open_text(source: TextProvider) -> Iterator[TextIO]:
    """Yield a readable TextIO.  Must always be used as a context manager."""
    if isinstance(source, Path):
        yield source.open(encoding="utf-8")
    elif isinstance(source, (str, list)):
        yield StringIO(get_text(source))
    else:
        yield source  # TextIO; already open, do not close


def parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


@cache
def default_string_converters() -> Mapping[type[Any], StringConverter]:
    return MappingProxyType(
        {Any: str, str: str, int: int, float: float, bool: parse_bool}
    )


@dataclass
class StringParser:
    _CONVERTERS: dict[type[Any], StringConverter] = field(
        init=False, default_factory=lambda: dict(default_string_converters())
    )
    _CONVERTER_LOCK: threading.Lock = field(
        init=False, default_factory=lambda: threading.Lock()
    )

    @staticmethod
    @cache
    def default() -> StringParser:
        return StringParser()

    def converters(
        self, target: type[Any]
    ) -> list[tuple[StringConverter, type[Any]]]:
        converters: list[tuple[StringConverter, type[Any]]] = []
        with self._CONVERTER_LOCK:
            for candidate_type in iterate_types(target):
                if (
                    converter := self._CONVERTERS.get(candidate_type)
                ) is None and callable(
                    method := getattr(candidate_type, "from_string", None)
                ):
                    self._CONVERTERS[candidate_type] = converter = method
                if converter is not None:
                    converters.append((converter, candidate_type))
                else:
                    logger.debug(
                        f"Skipping type without converter: {candidate_type}"
                    )
        return converters

    def parse(self, source: TextProvider, *, target: type[T]) -> T:
        # first_valid_conversion verifies the cast to the specific type
        # whereas T might be a Union here.
        source = get_text(source)
        try:
            return cast(
                T,
                type(self).first_valid_conversion(
                    source, converters=self.converters(target)
                ),
            )
        except TypeError:
            logger.exception(f"Could not parse to {target}: {source}")
            raise

    @staticmethod
    def first_valid_conversion(
        source: TextProvider,
        *,
        converters: Iterable[tuple[StringConverter, type[Any]]],
    ) -> Any:
        source = get_text(source)
        for converter, converter_type in converters:
            try:
                return verified_cast(converter_type, converter(source))
            except Exception:
                logger.exception(
                    f"Failed to convert to {converter_type}: {source}"
                )
        raise TypeError(f"No converters worked for source: {source}")

    def set_converter(
        self, target: type[T], *, converter: StringConverter
    ) -> None:
        with self._CONVERTER_LOCK:
            for candidate_type in iterate_types(target):
                self._CONVERTERS[target] = converter


class ColumnSubsetError(Exception):
    pass


@overload
def csv_load(
    source: TextProvider,
    *,
    delimiter: str = ...,
    column_names: Sequence[str] | None = ...,
    dataclass: None = ...,
    field_metadata_key: str = ...,
    field_to_column_name: dict[str, str] | None = ...,
    init_function: Callable[..., dict[str, str]] | None = ...,
    allow_column_subset: bool = ...,
    string_parser: StringParser | None = ...,
) -> Iterator[dict[str, str]]: ...


@overload
def csv_load(
    source: TextProvider,
    *,
    delimiter: str = ...,
    column_names: Sequence[str] | None = ...,
    dataclass: type[T] = ...,
    field_metadata_key: str = ...,
    field_to_column_name: dict[str, str] | None = ...,
    init_function: Callable[..., T] | None = ...,
    allow_column_subset: bool = ...,
    string_parser: StringParser | None = ...,
) -> Iterator[T]: ...


def csv_load(
    source: TextProvider,
    *,
    delimiter: str = ",",
    column_names: Sequence[str] | None = None,
    dataclass: type[T] | None = None,
    field_metadata_key: str = "csv_key",
    field_to_column_name: dict[str, str] | None = None,
    init_function: Callable[..., T | dict[str, str]] | None = None,
    allow_column_subset: bool = True,
    string_parser: StringParser | None = None,
) -> Iterator[T] | Iterator[dict[str, str]]:
    """Load CSV data into dicts or dataclass instances.

    For custom field types, a classmethod `from_string(cls, s: str)` may be
    implemented to control how an instance is created from a CSV cell string.
    """
    with open_text(source) as source_io:
        reader = csv.reader(source_io, delimiter=delimiter)
        if string_parser is None:
            string_parser = StringParser.default()
        if column_names is None:
            column_names = next(reader)
        if init_function is not None:
            type_hints = get_callable_argument_hints(init_function)
            if field_to_column_name is None:
                field_to_column_name = {key: key for key in type_hints}
        else:
            type_hints = None
        column_indices = {name: i for i, name in enumerate(column_names)}
        if dataclass is not None:
            if not is_dataclass(dataclass):
                raise TypeError(
                    f"dataclass argument isn't a dataclass: {dataclass}"
                )
            if init_function is None:
                init_function = dataclass
            if type_hints is None:
                type_hints = get_callable_argument_hints(dataclass)
            if field_to_column_name is None:
                field_to_column_name = {
                    name: dataclass.__dataclass_fields__[name].metadata.get(
                        field_metadata_key, name
                    )
                    for name in type_hints
                }
        else:
            if init_function is None:
                init_function = dict
            if type_hints is None:
                type_hints = {}
            if field_to_column_name is None:
                field_to_column_name = {
                    column_name: column_name for column_name in column_names
                }

        field_to_index_and_converters = {
            field_name: (
                index,
                string_parser.converters(type_hints.get(field_name, str)),
            )
            for field_name, column_name in field_to_column_name.items()
            if (index := column_indices.get(column_name)) is not None
        }
        if len(field_to_column_name) < len(column_names):
            message = (
                f"Only {len(field_to_index_and_converters)} fields"
                f" read of the {len(column_names)} present in CSV data."
            )
            logger.debug(message)
            if not allow_column_subset:
                raise ColumnSubsetError(message)
        yield from (
            init_function(
                **{
                    name: StringParser.first_valid_conversion(
                        row[index], converters=converters
                    )
                    for name, (
                        index,
                        converters,
                    ) in field_to_index_and_converters.items()
                }
            )
            for row in reader
        )


_JSON5_COMMENT_PATTERN = re.compile(
    r"""
    (                               # 1: double-quoted string
        "(?:\\.|[^"\\])*"
    )
  | (                               # 2: single-quoted string
        '(?:\\.|[^'\\])*'
    )
  | (?:[ \t]*//[^\r\n]*)            # remove spaces + single-line comment
  | (?:[ \t]*/\*.*?\*/)             # remove spaces + block comment (ungreedy)
    """,
    re.VERBOSE | re.DOTALL,
)


def json5_load(source: TextProvider) -> Any:
    def comment_replacer(match: re.Match[str]) -> str:
        return match.group(1) or match.group(2) or ""

    no_comments = _JSON5_COMMENT_PATTERN.sub(
        comment_replacer, get_text(source)
    )
    return json.loads(re.sub(r",(?=\s*[\]}])", "", no_comments))
