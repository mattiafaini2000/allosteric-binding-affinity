#!/usr/bin/env python3
"""Run asd expand online activity labels."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from allosteric_affinity.online_activity import main

if __name__ == "__main__":
    raise SystemExit(main())
