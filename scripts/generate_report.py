"""Generate research_report.md from the saved outputs of a results directory.

Example (from the project root)::

    python scripts/generate_report.py --results-dir results/synthetic_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import var_es  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from var_es.reporting import build_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Markdown research report from saved CSV/JSON results.")
    ap.add_argument("--results-dir", required=True, help="Results directory containing run_metadata.json and CSVs.")
    args = ap.parse_args(argv)
    rd = Path(args.results_dir)
    if not rd.is_absolute() and not rd.exists():
        rd = PROJECT_ROOT / rd
    if not (rd / "run_metadata.json").exists():
        raise SystemExit(f"{rd.name}: run_metadata.json not found; run scripts/run_backtest.py first")
    text = build_report(rd)
    (rd / "research_report.md").write_text(text, encoding="utf-8")
    print(f"Wrote {rd.name}/research_report.md ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
