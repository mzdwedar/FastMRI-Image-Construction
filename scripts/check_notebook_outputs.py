"""Fail if any tracked notebook has committed outputs or execution counts (strip with nbstripout)."""

import json
import subprocess
import sys


def dirty_cells(path: str) -> list[int]:
    with open(path) as f:
        cells = json.load(f)["cells"]
    return [i for i, c in enumerate(cells) if c.get("outputs") or c.get("execution_count")]


def main() -> int:
    files = subprocess.run(["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=True).stdout.split()
    bad = {f: cells for f in files if (cells := dirty_cells(f))}
    for f, cells in bad.items():
        print(f"{f}: outputs or execution counts in cells {cells}; run `uv run nbstripout {f}`", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
