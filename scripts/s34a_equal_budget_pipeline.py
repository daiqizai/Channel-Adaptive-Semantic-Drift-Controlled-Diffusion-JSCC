#!/usr/bin/env python3
"""Persistent fail-closed runner for the authorized S34A equal-budget pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/train/EXP-S34A-SWINJSCC-BASE-SA-EQUAL-BUDGET-001"
CM = ROOT / "outputs/train/EXP-S34A-SWINJSCC-CM-SA-EQUAL-BUDGET-001"
EVAL = ROOT / "outputs/external_baselines/ANALYSIS-S34A-SWINJSCC-EQUAL-BUDGET-COMPARISON-001"
STATE = ROOT / "outputs/train/S34A-SWINJSCC-PERSISTENT-PIPELINE-STATE.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_state(status: str, **extra) -> None:
    payload = {
        "status": status,
        "extension_run_allowed": False,
        "official_imagenette_validation_accessed": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def run(command: list[str], stage: str) -> None:
    write_state(stage, command=command)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if result.returncode != 0:
        write_state(f"{stage}_failed", returncode=result.returncode, command=command)
        raise SystemExit(result.returncode)


def require_complete_equal_budget(directory: Path, arm: str) -> None:
    state = read_json(directory / "STATE.json")
    summary = read_json(directory / "summary.json")
    if not state or not summary:
        raise RuntimeError(f"missing final state/summary for {arm}")
    if state.get("status") != "complete_equal_budget_only_waiting_for_user_extension_decision":
        raise RuntimeError(f"unexpected final state for {arm}: {state.get('status')}")
    if summary.get("status") != "complete_equal_budget_only" or summary.get("arm") != arm:
        raise RuntimeError(f"unexpected summary for {arm}")
    if int(summary.get("epochs_completed", -1)) != 12:
        raise RuntimeError(f"{arm} did not complete exactly 12 epochs")
    if summary.get("extension_executed") is not False:
        raise RuntimeError(f"extension detected for {arm}")


def main() -> None:
    train_script = str(ROOT / "scripts/s34a_train_swinjscc_equal_budget.py")
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    base_state = read_json(BASE / "STATE.json")
    if not base_state or base_state.get("status") != "complete_equal_budget_only_waiting_for_user_extension_decision":
        run(
            [sys.executable, train_script, "--arm", "official_base_sa", "--device", "cuda:0", "--resume"],
            "resuming_base_equal_budget",
        )
    require_complete_equal_budget(BASE, "official_base_sa")

    cm_state = read_json(CM / "STATE.json")
    if not cm_state or cm_state.get("status") != "complete_equal_budget_only_waiting_for_user_extension_decision":
        command = [
            sys.executable,
            train_script,
            "--arm",
            "capacity_matched_sa",
            "--device",
            "cuda:0",
        ]
        if CM.exists():
            command.append("--resume")
        run(command, "running_cm_equal_budget")
    require_complete_equal_budget(CM, "capacity_matched_sa")

    evaluation_state = read_json(EVAL / "STATE.json")
    if not evaluation_state or evaluation_state.get("status") != "complete":
        command = [
            sys.executable,
            str(ROOT / "scripts/s34a_evaluate_swinjscc_equal_budget.py"),
            "--config",
            str(ROOT / "configs/s34a_swinjscc_equal_budget_evaluation.yaml"),
            "--device",
            "cuda:0",
        ]
        if EVAL.exists():
            command.append("--resume")
        run(command, "running_equal_budget_policy_dev_evaluation")
    evaluation_state = read_json(EVAL / "STATE.json")
    if not evaluation_state or evaluation_state.get("status") != "complete":
        raise RuntimeError("equal-budget evaluation did not complete")
    write_state("complete_waiting_for_user_extension_decision", evaluation_state=evaluation_state)


if __name__ == "__main__":
    main()
