"""Generate the reproducible synthetic equity-return dataset.

Example (from the project root)::

    python scripts/generate_data.py --config configs/synthetic_baseline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import var_es  # noqa: F401
except ImportError:  # allow running without an editable install
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from var_es.data import load_config  # noqa: E402
from var_es.synthetic import write_synthetic_dataset  # noqa: E402


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else root / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic five-asset equity returns from a JSON config.")
    ap.add_argument("--config", required=True, help="Path to a JSON config with a 'synthetic' section.")
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="Root directory that relative data paths in the config refer to (default: project root).")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    cfg = load_config(resolve(args.config, root))
    if cfg["data"]["source"] != "synthetic" or "synthetic" not in cfg:
        ap.error("config is not a synthetic-data configuration")
    meta = write_synthetic_dataset(cfg, root)
    print(f"Generated {meta['n_obs']} observations x {len(meta['assets'])} assets "
          f"({meta['first_date']} to {meta['last_date']}), seed {meta['seed']}.")
    for k, v in meta["files"].items():
        print(f"  {k}: {v['path']}  sha256={v['sha256'][:16]}...")
    print(f"  metadata: {cfg['data']['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
