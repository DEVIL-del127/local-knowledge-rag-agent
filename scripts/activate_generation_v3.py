from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import PDFSearchSystem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_id")
    args = parser.parse_args()
    system = PDFSearchSystem(db="main")
    revision = system.activate_generation(args.generation_id)
    print(f"active_generation={args.generation_id} registry_revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
