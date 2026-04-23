from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    predictions: str
    mapping: str


@dataclass(frozen=True)
class AppConfig:
    testset_root: Path
    output_root: Path
    collar: int
    datasets: dict[str, DatasetConfig]


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: Path) -> AppConfig:
    base_dir = path.resolve().parent
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = {
        name: DatasetConfig(
            name=name,
            predictions=str(_resolve_path(item["predictions"], base_dir)),
            mapping=item["mapping"],
        )
        for name, item in data["datasets"].items()
    }
    return AppConfig(
        testset_root=_resolve_path(data["testset_root"], base_dir),
        output_root=_resolve_path(data.get("output_root", "./output"), base_dir),
        collar=int(data.get("collar", 5)),
        datasets=datasets,
    )
