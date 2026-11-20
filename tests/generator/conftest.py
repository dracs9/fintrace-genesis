"""Make the src-layout package importable while the shared packaging is pending."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
