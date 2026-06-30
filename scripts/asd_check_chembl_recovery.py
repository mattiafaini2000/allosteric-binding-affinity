#!/usr/bin/env python3
"""Run asd check chembl recovery."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from allosteric_affinity.discovery_chembl import main

if __name__ == "__main__":
    raise SystemExit(main())
