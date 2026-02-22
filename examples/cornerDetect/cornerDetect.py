from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cornerDetect import run_cli


if __name__ == "__main__":
    run_cli(base_dir=Path(__file__).resolve().parent)
