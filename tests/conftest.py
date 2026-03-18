"""Root conftest.py — ensures project root is on sys.path for all tests."""

import os
import sys

# Add project root to path so `math_nano` and `scripts` are importable
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
