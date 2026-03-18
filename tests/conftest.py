"""Root conftest — shared fixtures for all tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_gsm8k(fixtures_dir):
    """10 GSM8K test problems for eval testing."""
    path = fixtures_dir / "sample_gsm8k.jsonl"
    if not path.exists():
        pytest.skip("sample_gsm8k.jsonl fixture not yet created")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def sample_sft_data(fixtures_dir):
    """20 SFT samples for formatting tests."""
    path = fixtures_dir / "sample_sft.jsonl"
    if not path.exists():
        pytest.skip("sample_sft.jsonl fixture not yet created")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
