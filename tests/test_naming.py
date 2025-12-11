from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sevaht_utility.naming import (
    NameStyle,
    convert_name,
    join_words,
    split_into_words,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.mark.parametrize(
    ("name", "expected_words"),
    [
        ("some_sample_name", ["some", "sample", "name"]),
        ("some-sample-name", ["some", "sample", "name"]),
        ("someSampleName", ["some", "sample", "name"]),
        ("SomeSampleName", ["some", "sample", "name"]),
        ("single", ["single"]),
        ("Single", ["single"]),
        ("  some_sample_name  ", ["some", "sample", "name"]),
        ("", []),
        ("___", []),
        ("---", []),
    ],
)
def test_split_into_words(name: str, expected_words: list[str]) -> None:
    words = split_into_words(name)
    assert words == expected_words


@pytest.mark.parametrize(
    ("words", "style", "expected_name"),
    [
        (["some", "sample", "name"], NameStyle.SNAKE_CASE, "some_sample_name"),
        (["some", "sample", "name"], NameStyle.KEBAB_CASE, "some-sample-name"),
        (["some", "sample", "name"], NameStyle.CAMEL_CASE, "someSampleName"),
        (["some", "sample", "name"], NameStyle.PASCAL_CASE, "SomeSampleName"),
        (["single"], NameStyle.SNAKE_CASE, "single"),
        (["single"], NameStyle.CAMEL_CASE, "single"),
        (["single"], NameStyle.PASCAL_CASE, "Single"),
    ],
)
def test_join_words_basic(
    words: Sequence[str], style: NameStyle, expected_name: str
) -> None:
    name = join_words(words, style)
    assert name == expected_name


def test_join_words_ignores_empty_words() -> None:
    words: list[str] = ["", "some", "", "name", ""]
    name_snake = join_words(words, NameStyle.SNAKE_CASE)
    name_pascal = join_words(words, NameStyle.PASCAL_CASE)
    assert name_snake == "some_name"
    assert name_pascal == "SomeName"


@pytest.mark.parametrize(
    ("original_name", "style", "expected_name"),
    [
        ("someSampleName", NameStyle.SNAKE_CASE, "some_sample_name"),
        ("some-sample-name", NameStyle.PASCAL_CASE, "SomeSampleName"),
        ("SomeSampleName", NameStyle.KEBAB_CASE, "some-sample-name"),
        ("some_sample_name", NameStyle.CAMEL_CASE, "someSampleName"),
        ("some_sample_name", NameStyle.SNAKE_CASE, "some_sample_name"),
        ("SomeSampleName", NameStyle.PASCAL_CASE, "SomeSampleName"),
    ],
)
def test_convert_name_examples(
    original_name: str, style: NameStyle, expected_name: str
) -> None:
    converted = convert_name(original_name, style=style)
    assert converted == expected_name


@pytest.mark.parametrize(
    "original_name",
    [
        "some_sample_name",
        "some-sample-name",
        "someSampleName",
        "SomeSampleName",
        "single",
    ],
)
@pytest.mark.parametrize("intermediate_style", list(NameStyle))
@pytest.mark.parametrize("style", list(NameStyle))
def test_convert_name_idempotent_over_words(
    original_name: str, intermediate_style: NameStyle, style: NameStyle
) -> None:
    """Converting via an intermediate style should not change the final result.

    convert_name(original, target) == convert_name(
        convert_name(original, intermediate), target
    )
    """
    direct_conversion = convert_name(original_name, style=style)
    via_intermediate = convert_name(
        convert_name(original_name, style=intermediate_style), style=style
    )
    assert via_intermediate == direct_conversion


def test_convert_name_empty() -> None:
    assert convert_name("", style=NameStyle.SNAKE_CASE) == ""
    assert convert_name("", style=NameStyle.CAMEL_CASE) == ""
    assert convert_name("", style=NameStyle.PASCAL_CASE) == ""
    assert convert_name("", style=NameStyle.KEBAB_CASE) == ""
