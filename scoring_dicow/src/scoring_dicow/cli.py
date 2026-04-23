from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io import write_json
from .metrics import score_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--collar", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    output_root = args.output_root or cfg.output_root
    collar = args.collar if args.collar is not None else cfg.collar
    selected = args.datasets or list(cfg.datasets.keys())

    output_root.mkdir(parents=True, exist_ok=True)
    combined = {}
    for dataset_name in selected:
        dataset_cfg = cfg.datasets[dataset_name]
        combined[dataset_name] = score_dataset(
            cfg.testset_root / dataset_name,
            dataset_cfg,
            output_root / dataset_name,
            collar=collar,
        )

    write_json(output_root / "metric_summary_normalized.json", combined)
    print((output_root / "metric_summary_normalized.json").read_text(encoding="utf-8"))
