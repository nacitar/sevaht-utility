from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from functools import cache
from io import StringIO
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import Any, TextIO, TypeAlias, TypeVar, cast, overload

from .hinting import get_callable_argument_hints, iterate_types, verify_type
from .naming import NameStyle, convert_name

logger = logging.getLogger(__name__)

T = TypeVar("T")
TextProvider: TypeAlias = str | Path | TextIO | list[str]
StringConverter: TypeAlias = Callable[[str], object]
# NOTE: must quote recursive type aliases even with future annotations
JsonValue: TypeAlias = (
    dict[str, "JsonValue"]
    | list["JsonValue"]
    | str
    | int
    | float
    | bool
    | None
)


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


class StringParserError(TypeError):
    def __init__(self, value: object) -> None:
        super().__init__(f"Could not parse string: {value}")
        self.value = value


class StringConverterError(TypeError):
    def __init__(self, value: object) -> None:
        super().__init__(f"Could not parse string: {value}")
        self.value = value


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
        self, target: type[Any] | UnionType
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
        # first_valid_conversion verifies the cast to a specific type
        # whereas T might be a Union here.
        source = get_text(source)
        try:
            return cast(
                "T",
                type(self).first_valid_conversion(
                    source, converters=self.converters(target)
                ),
            )
        except StringParserError:
            logger.exception(f"Could not parse to {target}: {source}")
            raise

    @staticmethod
    def first_valid_conversion(
        source: TextProvider,
        *,
        converters: Iterable[tuple[StringConverter, type[Any]]],
    ) -> object:
        source = get_text(source)
        for converter, converter_type in converters:
            try:
                return verify_type(converter_type, converter(source))
            except Exception:  # noqa: BLE001
                # catching bare exception because user-provided converters may
                # raise anything; this code cannot control that.
                logger.debug(
                    f"Failed to convert to {converter_type}: {source}"
                )
        raise StringParserError(source)

    def set_converter(
        self, target: type[T], *, converter: StringConverter
    ) -> None:
        with self._CONVERTER_LOCK:
            for candidate_type in iterate_types(target):
                self._CONVERTERS[candidate_type] = converter


class UnconsumedColumnsError(Exception):
    def __init__(self, columns: Sequence[str]) -> None:
        super().__init__(
            f"{len(columns)} columns were not consumed: {', '.join(columns)}"
        )


class NotADataclassError(TypeError):
    """Raised when an argument expected to be a dataclass is not one."""

    def __init__(self, obj: object) -> None:
        super().__init__(f"Dataclass argument isn't a dataclass: {obj}")
        self.obj = obj


class MutuallyExclusiveArgumentsError(Exception):
    def __init__(self, arguments: Sequence[str]) -> None:
        super().__init__(
            f"Mutually exclusive arguments provided: {', '.join(arguments)}"
        )


@dataclass
class DataMapping:
    column_names: Sequence[str] | None = None
    field_to_column_name: Mapping[str, str] | None = None
    field_to_column_index: Mapping[str, int] | None = None
    name_style: NameStyle | None = None


class AmbiguousColumnNamesError(ValueError):
    def __init__(
        self, *, canonical_name: str, columns: Sequence[tuple[int, str]]
    ) -> None:
        formatted_columns = ", ".join(
            f"{name!r} at index {index}" for index, name in columns
        )
        super().__init__(
            "Ambiguous column names after normalization "
            f"for key {canonical_name!r}: {formatted_columns}. "
            "Use DataMapping.column_names or DataMapping.field_to_column_index "
            "to disambiguate."
        )
        self.canonical_name = canonical_name
        self.columns = columns


class AmbiguousFieldMappingsError(ValueError):
    def __init__(self, *, canonical_name: str, fields: Sequence[str]) -> None:
        super().__init__(
            "Ambiguous field mappings after normalization "
            f"for key {canonical_name!r}: {', '.join(fields)}. "
            "Use DataMapping.field_to_column_index or distinct names to "
            "disambiguate."
        )
        self.canonical_name = canonical_name
        self.fields = fields


class ColumnIndexOutOfRangeError(ValueError):
    def __init__(
        self, *, field_name: str, column_index: int, column_count: int
    ) -> None:
        super().__init__(
            f"Column index out of range for field {field_name!r}: "
            f"{column_index} (column count: {column_count})"
        )
        self.field_name = field_name
        self.column_index = column_index
        self.column_count = column_count


@dataclass
class CsvLoadOptions:
    delimiter: str = ","
    field_metadata_key: str = "csv_key"
    allow_column_subset: bool = True
    string_parser: StringParser = field(
        default_factory=lambda: StringParser.default()
    )


FieldMapping: TypeAlias = Mapping[str, str]
FieldToColumnIndexMapping: TypeAlias = Mapping[str, int]
FieldConverters: TypeAlias = list[tuple[StringConverter, type[Any]]]
FieldIndexAndConverters: TypeAlias = dict[str, tuple[int, FieldConverters]]
CanonicalColumnIndices: TypeAlias = dict[str, list[int]]
AmbiguousColumns: TypeAlias = dict[str, list[tuple[int, str]]]


@dataclass(frozen=True)
class ColumnResolution:
    resolved_indices: Mapping[str, int]
    ambiguous_columns: AmbiguousColumns
    column_count: int
    mapping: DataMapping


def _convert_name_if_needed(value: str, *, mapping: DataMapping) -> str:
    return (
        convert_name(value, style=mapping.name_style)
        if mapping.name_style
        else value
    )


def _build_canonical_column_indices(
    *, column_names: Sequence[str], mapping: DataMapping
) -> CanonicalColumnIndices:
    canonical_column_indices: CanonicalColumnIndices = {}
    for index, column_name in enumerate(column_names):
        canonical_name = _convert_name_if_needed(column_name, mapping=mapping)
        canonical_column_indices.setdefault(canonical_name, []).append(index)
    return canonical_column_indices


def _split_ambiguous_columns(
    *,
    column_names: Sequence[str],
    canonical_column_indices: CanonicalColumnIndices,
) -> tuple[dict[str, int], AmbiguousColumns]:
    resolved_column_indices = {
        name: indices[0]
        for name, indices in canonical_column_indices.items()
        if len(indices) == 1
    }
    ambiguous_columns = {
        canonical_name: [(index, column_names[index]) for index in indices]
        for canonical_name, indices in canonical_column_indices.items()
        if len(indices) > 1
    }
    return resolved_column_indices, ambiguous_columns


def _resolve_field_to_column_name(
    *,
    type_hints: Mapping[str, type[Any]],
    field_to_column_name: FieldMapping | None,
    mapping: DataMapping,
    options: CsvLoadOptions,
    dataclass_type: type[Any] | None,
) -> FieldMapping:
    if field_to_column_name is not None:
        return field_to_column_name
    if dataclass_type is None:
        return {
            key: _convert_name_if_needed(key, mapping=mapping)
            for key in type_hints
        }
    return {
        name: _convert_name_if_needed(
            dataclass_type.__dataclass_fields__[name].metadata.get(
                options.field_metadata_key, name
            ),
            mapping=mapping,
        )
        for name in type_hints
    }


def _build_field_indices_and_converters(
    *,
    field_to_column_name: Mapping[str, str],
    field_to_column_index: FieldToColumnIndexMapping | None,
    column_resolution: ColumnResolution,
    type_hints: Mapping[str, type[Any]],
    string_parser: StringParser,
) -> FieldIndexAndConverters:
    field_to_canonical_name: dict[str, str] = {}
    field_to_index: dict[str, int] = {}
    for field_name, column_name in field_to_column_name.items():
        if (
            field_to_column_index is not None
            and (field_index := field_to_column_index.get(field_name))
            is not None
        ):
            if (
                field_index < 0
                or field_index >= column_resolution.column_count
            ):
                raise ColumnIndexOutOfRangeError(
                    field_name=field_name,
                    column_index=field_index,
                    column_count=column_resolution.column_count,
                )
            field_to_index[field_name] = field_index
            continue

        canonical_name = _convert_name_if_needed(
            column_name, mapping=column_resolution.mapping
        )
        field_to_canonical_name[field_name] = canonical_name
        if ambiguous_name_columns := column_resolution.ambiguous_columns.get(
            canonical_name
        ):
            raise AmbiguousColumnNamesError(
                canonical_name=canonical_name, columns=ambiguous_name_columns
            )
        if (
            index := column_resolution.resolved_indices.get(canonical_name)
        ) is not None:
            field_to_index[field_name] = index

    canonical_to_fields: dict[str, list[str]] = {}
    for field_name, canonical_name in field_to_canonical_name.items():
        canonical_to_fields.setdefault(canonical_name, []).append(field_name)
    if ambiguous_fields := [
        (canonical_name, fields)
        for canonical_name, fields in canonical_to_fields.items()
        if len(fields) > 1
    ]:
        canonical_name, fields = ambiguous_fields[0]
        raise AmbiguousFieldMappingsError(
            canonical_name=canonical_name, fields=fields
        )

    return {
        field_name: (
            index,
            string_parser.converters(type_hints.get(field_name, str)),
        )
        for field_name, index in field_to_index.items()
    }


def _assert_or_allow_unconsumed_columns(
    *,
    column_names: Sequence[str],
    field_to_index_and_converters: FieldIndexAndConverters,
    allow_column_subset: bool,
) -> None:
    consumed_indices = {i for i, _ in field_to_index_and_converters.values()}
    if len(consumed_indices) >= len(column_names):
        return
    unconsumed_column_names = [
        name
        for i, name in enumerate(column_names)
        if i not in consumed_indices
    ]
    if not unconsumed_column_names:
        return
    error = UnconsumedColumnsError(unconsumed_column_names)
    if not allow_column_subset:
        raise error


@overload  # dict case, no init_function for type hints; dict[str, str]
def csv_load(
    source: TextProvider,
    *,
    dataclass: None = ...,
    init_function: None = None,
    mapping: DataMapping | None = ...,
    options: CsvLoadOptions | None = ...,
) -> Iterator[dict[str, str]]: ...


@overload  # dict case, YES init_function for type hints; dict[str, object]
def csv_load(
    source: TextProvider,
    *,
    dataclass: None = None,
    init_function: Callable[..., dict[str, object]],  # REQUIRED
    mapping: DataMapping | None = ...,
    options: CsvLoadOptions | None = ...,
) -> Iterator[dict[str, object]]: ...


@overload  # dataclass case
def csv_load(
    source: TextProvider,
    *,
    dataclass: type[T],  # REQUIRED
    init_function: Callable[..., T] | None = ...,
    mapping: DataMapping | None = ...,
    options: CsvLoadOptions | None = ...,
) -> Iterator[T]: ...


def csv_load(
    source: TextProvider,
    *,
    dataclass: type[T] | None = None,
    init_function: Callable[..., object] | None = None,
    mapping: DataMapping | None = None,
    options: CsvLoadOptions | None = None,
) -> Iterator[T] | Iterator[dict[str, str]] | Iterator[dict[str, object]]:
    """Load CSV data into dicts or dataclass instances.

    For custom field types, a classmethod `from_string(cls, s: str)` may be
    implemented to control how an instance is created from a CSV cell string.

    Field mapping precedence (highest to lowest):
    1) `mapping.field_to_column_index`
    2) `mapping.field_to_column_name`
    3) `init_function` parameter names (optionally normalized by `name_style`)
    4) Dataclass field metadata key (`options.field_metadata_key`) or field name
       (optionally normalized by `name_style`)
    5) Dict mode defaults to CSV column names (optionally normalized by
       `name_style`)

    When `name_style` is provided, both source column names and target mapping
    names are normalized before matching. Ambiguous normalized matches raise
    `AmbiguousColumnNamesError`.
    """
    mapping = mapping or DataMapping()
    options = options or CsvLoadOptions()
    field_to_column_name = mapping.field_to_column_name
    field_to_column_index = mapping.field_to_column_index

    with open_text(source) as source_io:
        reader = csv.reader(source_io, delimiter=options.delimiter)
        string_parser = options.string_parser or StringParser.default()
        column_names = mapping.column_names
        if column_names is None:
            try:
                column_names = next(reader)
            except StopIteration:
                logger.debug("No column names provided and source is empty.")
                return

        resolved_init_function = init_function
        if init_function is not None:
            type_hints = get_callable_argument_hints(init_function)
        else:
            type_hints = None
        canonical_column_indices = _build_canonical_column_indices(
            column_names=column_names, mapping=mapping
        )
        resolved_column_indices, ambiguous_columns = _split_ambiguous_columns(
            column_names=column_names,
            canonical_column_indices=canonical_column_indices,
        )
        if dataclass is not None:
            if not is_dataclass(dataclass):
                raise NotADataclassError(dataclass)
            resolved_init_function = cast(
                "Callable[..., T]", resolved_init_function or dataclass
            )
            type_hints = type_hints or get_callable_argument_hints(dataclass)
            if field_to_column_name is None and init_function is not None:
                field_to_column_name = {
                    key: _convert_name_if_needed(key, mapping=mapping)
                    for key in type_hints
                }
            else:
                field_to_column_name = _resolve_field_to_column_name(
                    type_hints=type_hints,
                    field_to_column_name=field_to_column_name,
                    mapping=mapping,
                    options=options,
                    dataclass_type=dataclass,
                )
        else:
            if resolved_init_function is None:
                resolved_init_function = cast(
                    "Callable[..., dict[str, str]]", dict
                )
            else:
                resolved_init_function = cast(
                    "Callable[..., dict[str, object]]", resolved_init_function
                )
            type_hints = type_hints or {}
            field_to_column_name = field_to_column_name or {
                column_name: _convert_name_if_needed(
                    column_name, mapping=mapping
                )
                for column_name in column_names
            }

        field_to_index_and_converters = _build_field_indices_and_converters(
            field_to_column_name=field_to_column_name,
            field_to_column_index=field_to_column_index,
            column_resolution=ColumnResolution(
                resolved_indices=resolved_column_indices,
                ambiguous_columns=ambiguous_columns,
                column_count=len(column_names),
                mapping=mapping,
            ),
            type_hints=type_hints,
            string_parser=string_parser,
        )
        _assert_or_allow_unconsumed_columns(
            column_names=column_names,
            field_to_index_and_converters=field_to_index_and_converters,
            allow_column_subset=options.allow_column_subset,
        )
        first_valid_conversion = StringParser.first_valid_conversion
        yield from (
            resolved_init_function(
                **{
                    name: first_valid_conversion(
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


def json5_load(source: TextProvider) -> JsonValue:
    def comment_replacer(match: re.Match[str]) -> str:
        return match.group(1) or match.group(2) or ""

    no_comments = _JSON5_COMMENT_PATTERN.sub(
        comment_replacer, get_text(source)
    )
    return cast(
        "JsonValue", json.loads(re.sub(r",(?=\s*[\]}])", "", no_comments))
    )
