"""Shared fail-closed guards for frozen RDD-P0 analysis stages."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml


def require_frozen_no_overwrite(config_path: Path, targets: Iterable[Path]) -> None:
    """Refuse a stage unless the frozen config forbids overwrite and all targets are absent."""
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    overwrite = config.get("boundaries", {}).get("overwrite_existing_outputs")
    if overwrite is not False:
        raise RuntimeError(
            "frozen config must set boundaries.overwrite_existing_outputs=false: %s"
            % config_path
        )

    unique_targets = list(dict.fromkeys(Path(path) for path in targets))
    existing = [path for path in unique_targets if path.exists()]
    if existing:
        rendered = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "refusing to overwrite frozen RDD-P0 outputs; use a new analysis ID:\n  "
            + rendered
        )
