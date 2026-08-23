"""Streamlit Community Cloud entry point."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
