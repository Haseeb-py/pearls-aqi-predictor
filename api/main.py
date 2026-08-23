"""Uvicorn entry point that exposes the application in the src layout."""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pearls_aqi.api.main import app  # noqa: F401
