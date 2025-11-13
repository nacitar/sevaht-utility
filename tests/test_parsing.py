from __future__ import annotations

import io
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sevaht_utility.parsing import (
    ColumnSubsetError,
    StringConverter,
    StringParser,
    csv_load,
    get_text,
    json5_load,
    open_text,
    parse_bool,
)


def test_get_text_from_string() -> None:
    assert get_text("abc") == "abc"


def test_get_text_from_list() -> None:
    lines = ["a", "b", "c"]
    result = get_text(lines)
    # should join with os.linesep
    assert "a" in result and "b" in result and "c" in result
    assert result.count("\n") == 2 or result.count("\r\n") == 2


def test_get_text_from_path(tmp_path: Path) -> None:
    p = tmp_path / "sample.txt"
    p.write_text("hello world", encoding="utf-8")
    assert get_text(p) == "hello world"


def test_get_text_from_textio() -> None:
    s = io.StringIO("xyz")
    assert get_text(s) == "xyz"


def test_open_text_with_path(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("content")
    with open_text(p) as f:
        assert f.read() == "content"


def test_open_text_with_string() -> None:
    with open_text("abc") as f:
        assert f.read() == "abc"


def test_open_text_with_list() -> None:
    with open_text(["x", "y"]) as f:
        text = f.read()
    assert "x" in text and "y" in text


def test_open_text_with_textio() -> None:
    s = io.StringIO("something")
    with open_text(s) as f:
        assert f is s  # should yield same object
        assert f.read() == "something"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("no", False),
        ("FALSE", False),
        ("TrUe", True),
    ],
)
def test_parse_bool_various(value: str, expected: bool) -> None:
    assert parse_bool(value) is expected


def test_stringparser_default_converters_include_basic_types() -> None:
    parser = StringParser.default()
    converters = parser.converters(int)
    assert any(conv is int for conv, _ in converters)


def test_stringparser_handles_from_string_classmethod() -> None:
    @dataclass
    class Custom:
        value: int

        @classmethod
        def from_string(cls, value: str) -> Custom:
            return cls(int(value))

    parser = StringParser()
    result = parser.parse("42", target=Custom)
    assert isinstance(result, Custom)
    assert result.value == 42


def test_stringparser_with_custom_registered_converter() -> None:
    class Custom:
        def __init__(self, s: str) -> None:
            self.value = int(s) + 1

    def custom_conv(s: str) -> Custom:
        return Custom(s)

    parser = StringParser()
    parser.set_converter(Custom, converter=custom_conv)
    result = parser.parse("9", target=Custom)
    assert result.value == 10


def test_stringparser_fallback_raises() -> None:
    class Bad:
        pass

    parser = StringParser()
    with pytest.raises(TypeError):
        parser.parse("text", target=Bad)


def test_stringparser_first_valid_conversion_picks_first() -> None:
    converters = [(int, int), (float, float)]
    result = StringParser.first_valid_conversion("7", converters=converters)
    assert result == 7


def test_stringparser_first_valid_conversion_skips_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_conv(_: str) -> int:
        raise ValueError("bad")

    converters: list[tuple[StringConverter, type[Any]]] = [
        (fail_conv, int),
        (float, float),
    ]
    result = StringParser.first_valid_conversion("3.14", converters=converters)
    assert result == 3.14
    assert any(
        "Failed to convert" in line for line in caplog.text.splitlines()
    )


@dataclass
class FromRegistered:
    value: int = field(init=False, default=0)
    str_value: InitVar[str | None] = None

    def __post_init__(self, str_value: str | None) -> None:
        if str_value is not None:
            self.value = int(str_value)


@pytest.fixture
def string_parser_from_registered() -> StringParser:
    parser = StringParser()
    parser.set_converter(
        FromRegistered, converter=FromRegistered  # constructor
    )
    return parser


@dataclass
class FromMethod:
    value: int

    @classmethod
    def from_string(cls, value: str) -> FromMethod:
        return cls(int(value))


@dataclass
class ComplexClass:
    number: int = field(metadata={"csv_key": "custom_number"})
    float_number: float
    from_registered: FromRegistered
    from_method: FromMethod
    string: str
    extra_argument: int = 0


@pytest.fixture
def csv_header() -> str:
    return "custom_number,float_number,from_registered,from_method,string"


@pytest.fixture
def csv_rows() -> list[str]:
    return [
        "1,3.14,2,3,Hello",
        "4,2.71,5,6,Goodbye",
        "9,11.72,311,123,Longer Text",
    ]


@pytest.fixture
def csv_lines(csv_header: str, csv_rows: list[str]) -> list[str]:
    return [csv_header, *csv_rows]


def test_csv_load_into_dataclass(
    csv_lines: list[str],
    csv_rows: list[str],
    string_parser_from_registered: StringParser,
) -> None:
    """Verify normal dataclass loading with automatic converters
    and metadata mapping."""
    instances = list(
        csv_load(
            csv_lines,
            dataclass=ComplexClass,
            string_parser=string_parser_from_registered,
        )
    )
    assert len(instances) == len(csv_rows)

    for instance, row in zip(instances, csv_rows):
        n, f, r, m, s = row.split(",")
        assert instance.number == int(n)
        assert instance.float_number == float(f)
        assert instance.from_registered.value == int(r)
        assert instance.from_method.value == int(m)
        assert instance.string == s


def test_csv_load_raises_if_not_dataclass(csv_lines: list[str]) -> None:
    """Passing a non-dataclass type to `dataclass` should raise TypeError."""

    class NotADataClass:
        pass

    with pytest.raises(TypeError, match="isn't a dataclass"):
        list(csv_load(csv_lines, dataclass=NotADataClass))


def test_csv_load_dataclass_with_custom_init(
    csv_lines: list[str],
    csv_rows: list[str],
    string_parser_from_registered: StringParser,
) -> None:
    """Verify `init_function` and `init_arguments` alter the resulting
    instances as expected."""

    def custom_factory(
        custom_number: int,
        float_number: float,
        from_registered: FromRegistered,
        from_method: FromMethod,
        string: str,
    ) -> ComplexClass:
        return ComplexClass(
            number=custom_number + 10,
            float_number=float_number + 1.0,
            from_registered=from_registered,
            from_method=from_method,
            string=string,
            extra_argument=5,
        )

    instances = list(
        csv_load(
            csv_lines,
            dataclass=ComplexClass,
            init_function=custom_factory,
            string_parser=string_parser_from_registered,
        )
    )
    for instance, row in zip(instances, csv_rows):
        n, f, r, m, s = row.split(",")
        assert instance.number == int(n) + 10
        assert instance.float_number == float(f) + 1
        assert instance.from_registered.value == int(r)
        assert instance.from_method.value == int(m)
        assert instance.string == s
        assert instance.extra_argument == 5


def test_csv_load_dataclass_with_custom_init_and_field_names(
    csv_lines: list[str],
    csv_rows: list[str],
    string_parser_from_registered: StringParser,
) -> None:
    """Verify `init_function` and `init_arguments` alter the resulting
    instances as expected."""

    def custom_factory_with_field_names(
        number: int,
        float_number: float,
        from_registered: FromRegistered,
        from_method: FromMethod,
        string: str,
    ) -> ComplexClass:
        return ComplexClass(
            number=number + 10,
            float_number=float_number + 1.0,
            from_registered=from_registered,
            from_method=from_method,
            string=string,
            extra_argument=5,
        )

    instances = list(
        csv_load(
            csv_lines,
            dataclass=ComplexClass,
            init_function=custom_factory_with_field_names,
            field_to_column_name={
                "number": "custom_number",
                "float_number": "float_number",
                "from_registered": "from_registered",
                "from_method": "from_method",
                "string": "string",
            },
            string_parser=string_parser_from_registered,
        )
    )
    for instance, row in zip(instances, csv_rows):
        n, f, r, m, s = row.split(",")
        assert instance.number == int(n) + 10
        assert instance.float_number == float(f) + 1
        assert instance.from_registered.value == int(r)
        assert instance.from_method.value == int(m)
        assert instance.string == s
        assert instance.extra_argument == 5


def test_csv_load_into_dict(csv_rows: list[str]) -> None:
    """Ensure csv_load produces dicts when no dataclass is provided."""
    header = "number,float_number,from_registered,from_method,string"
    lines = [header, *csv_rows]
    mapping = {"the_float": "float_number", "the_string": "string"}

    with pytest.raises(ColumnSubsetError):
        next(
            csv_load(
                lines, field_to_column_name=mapping, allow_column_subset=False
            )
        )

    results = list(csv_load(lines, field_to_column_name=mapping))

    for row, result in zip(csv_rows, results):
        n, f, r, m, s = row.split(",")
        assert result["the_float"] == f
        assert result["the_string"] == s
        # Only mapped + injected keys should exist
        assert set(result.keys()) == {"the_float", "the_string"}


def test_csv_load_missing_column_name(csv_rows: list[str]) -> None:
    """Unknown column names should simply be skipped (converter not built)."""
    header = "only_this"
    result = list(csv_load([header, *csv_rows]))
    assert all(isinstance(r, dict) for r in result)
    assert all(
        r == {"only_this": v.split(",")[0]} for r, v in zip(result, csv_rows)
    )


def test_csv_load_custom_delimiter() -> None:
    """Verify custom delimiter works as expected."""
    csv_data = ["a|b|c", "1|2|3"]
    results = list(csv_load(csv_data, delimiter="|"))
    assert results == [{"a": "1", "b": "2", "c": "3"}]


@pytest.fixture
def data_scores_header() -> str:
    return "id,name,score_1,score_2,bonus_score"


@pytest.fixture
def data_scores_rows() -> list[str]:
    return ["7,John Doe,97.2,79,50", "abc-123,John Doe,97.2,79,50"]


@pytest.fixture
def data_scores_lines(
    data_scores_header: str, data_scores_rows: list[str]
) -> list[str]:
    return [data_scores_header, *data_scores_rows]


def test_csv_load_dataclass_with_initvar_and_init_false(
    data_scores_lines: list[str], data_scores_rows: list[str]
) -> None:
    @dataclass
    class TestClass:
        id: int | str
        name: str
        scores: list[float] = field(init=False, default_factory=list)
        score_1: InitVar[float]
        score_2: InitVar[float]
        score_3: InitVar[float] = field(metadata={"csv_key": "bonus_score"})

        def __post_init__(
            self, score_1: float, score_2: float, score_3: float
        ) -> None:
            self.scores = [score_1, score_2, score_3]

    instances = list(csv_load(data_scores_lines, dataclass=TestClass))

    def int_or_str(value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value

    for instance, row in zip(instances, data_scores_rows):
        id, name, score_1, score_2, bonus_score = row.split(",")
        assert instance.id == int_or_str(id)
        assert instance.name == name
        assert instance.scores == [
            float(score_1),
            float(score_2),
            float(bonus_score),
        ]


def test_csv_load_with_repeated_headers_is_ok(tmp_path: Path) -> None:
    data = ["a,a,b", "1,2,3"]
    results = list(csv_load(data))
    assert isinstance(results[0], dict)
    assert set(results[0].keys()) == {"a", "b"}


def test_csv_load_with_empty_file_yields_nothing(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert list(csv_load(p)) == []


def test_strip_json5_comments_and_trailing_commas() -> None:
    samples = {
        '{"a": "simple", // comment\n "b": "text",}': {
            "a": "simple",
            "b": "text",
        },
        '{"q": "has quote: \\"inner\\"", "x": 1,}': {
            "q": 'has quote: "inner"',
            "x": 1,
        },
        '{"b": "escaped backslash: \\\\", "y": 2, // comment\n}': {
            "b": "escaped backslash: \\",
            "y": 2,
        },
        """
        {
            "msg": "Line1\\nLine2", // multi-line escape
            "num": 5,
        }
        """: {
            "msg": "Line1\nLine2",
            "num": 5,
        },
    }

    for src, expected_obj in samples.items():
        parsed = json5_load(src)
        assert parsed == expected_obj, (
            f"\nInput:\n{src}"
            f"\nParsed:\n{parsed}"
            f"\nExpected:\n{expected_obj}"
        )


def test_json5_load_with_empty_object() -> None:
    assert json5_load("{}") == {}


def test_json5_load_ignores_comments_inside_strings() -> None:
    text = '{"key": "// not a comment"}'
    result = json5_load(text)
    assert result == {"key": "// not a comment"}


def test_json5_load_trailing_comma_in_array() -> None:
    text = '{"arr": [1,2,3,], "x": 5,}'
    result = json5_load(text)
    assert result == {"arr": [1, 2, 3], "x": 5}
