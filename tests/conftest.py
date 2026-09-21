"""Shared test configuration: make build/ scripts importable as modules."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "build"), str(ROOT / "build" / "installer")):
    if p not in sys.path:
        sys.path.insert(0, p)
