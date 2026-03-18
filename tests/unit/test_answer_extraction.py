"""Tests for answer extraction from model outputs."""

import pytest

from scripts.eval.extraction import extract_answer, normalize_answer


class TestExtractAnswer:
    # -- \boxed{} format --
    def test_boxed_integer(self):
        assert extract_answer("The answer is \\boxed{42}") == "42"

    def test_boxed_negative(self):
        assert extract_answer("\\boxed{-7}") == "-7"

    def test_boxed_decimal(self):
        assert extract_answer("\\boxed{3.14}") == "3.14"

    def test_boxed_multiple_takes_last(self):
        assert extract_answer("First \\boxed{5}, then \\boxed{10}") == "10"

    def test_boxed_with_surrounding_text(self):
        text = "After solving step by step, we get \\boxed{123}. Done."
        assert extract_answer(text) == "123"

    # -- #### format (GSM8K) --
    def test_hash_format(self):
        assert extract_answer("#### 42") == "42"

    def test_hash_with_comma(self):
        assert extract_answer("#### 1,234") == "1234"

    # -- "The answer is" format --
    def test_answer_is(self):
        assert extract_answer("The answer is 42.") == "42"

    def test_final_answer_is(self):
        assert extract_answer("The final answer is 42.") == "42"

    # -- Fallback: last number --
    def test_last_number_fallback(self):
        assert extract_answer("So we get 3 + 4 = 7") == "7"

    # -- Edge cases --
    def test_empty_string(self):
        assert extract_answer("") is None

    def test_no_numbers(self):
        assert extract_answer("I don't know the answer.") is None

    def test_only_whitespace(self):
        assert extract_answer("   \n\n  ") is None

    def test_very_long_output(self):
        text = "Step 1: ...\n" * 500 + "The answer is \\boxed{42}"
        assert extract_answer(text) == "42"


class TestNormalizeAnswer:
    def test_strip_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_strip_dollar_signs(self):
        assert normalize_answer("$42$") == "42"

    def test_strip_commas(self):
        assert normalize_answer("1,234") == "1234"

    def test_float_to_int(self):
        assert normalize_answer("42.0") == "42"

    def test_preserve_decimal(self):
        assert normalize_answer("3.14") == "3.14"

    def test_negative(self):
        assert normalize_answer("-7") == "-7"

    def test_equivalent_representations(self):
        representations = ["42", "42.0", "42.00", " 42 ", "$42$"]
        normalized = set(normalize_answer(r) for r in representations)
        assert len(normalized) == 1, f"Not all equivalent: {normalized}"
