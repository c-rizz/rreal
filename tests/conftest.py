"""Root pytest configuration for the rreal test suite.

Layout
------
    tests/nets/                 encoder/decoder network unit tests (fast, CPU)
    tests/feature_extractors/   feature-extractor tests (fast, CPU)

Slicing the suite (markers are declared in pyproject.toml)::

    pytest -m "not gpu"                 # fast CPU-only tests
    pytest tests/feature_extractors     # just the feature-extractor tests

Tests that need hardware skip themselves when the requirement is missing, so the suite is
always runnable everywhere.
"""
from __future__ import annotations

import pytest
import torch as th


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Every test starts from the same RNG state, so network inits are reproducible."""
    th.manual_seed(0)


@pytest.fixture(scope="session")
def cpu_device() -> th.device:
    return th.device("cpu")
