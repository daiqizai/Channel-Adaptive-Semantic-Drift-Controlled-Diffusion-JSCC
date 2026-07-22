#!/usr/bin/env python3
"""Run the authorized S34A policy-dev evaluation after both 12-epoch arms finish."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CM_STATE = ROOT / "outputs/train/EXP-S34A-SWINJSCC-CM-SA-EQUAL-BUDGET-001/STATE.json"
DUAL_STATE = ROOT / "outputs/train/S34A-SWINJSCC-DUAL-ARM-LAUNCHER-STATE.json"
EVAL_OUTPUT = ROOT / "outputs/external_baselines/ANALYSIS-S34A-SWINJSCC-EQUAL-BUDGET-COMPARISON-001"
STATE = ROOT / "outputs/train/S34A-SWINJSCC-EVALUATION-LAUNCHER-STATE.json"


def write_state(value: dict) -> None:
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    if EVAL_OUTPUT.exists():
        raise FileExistsError(EVAL_OUTPUT)
    write_state(
        {
            "status": "waiting_for_cm_equal_budget",
            "extension_run_allowed": False,
            "official_imagenette_validation_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    while True:
        cm_state = read_json(CM_STATE)
        if cm_state and cm_state.get("status") == "complete_equal_budget_only_waiting_for_user_extension_decision":
            break
        dual_state = read_json(DUAL_STATE)
        if dual_state and dual_state.get("status") in {
            "base_not_complete_cm_not_started",
            "cm_process_finished",
        }:
            if not cm_state or cm_state.get("status") != "complete_equal_budget_only_waiting_for_user_extension_decision":
                write_state(
                    {
                        "status": "training_failed_evaluation_not_started",
                        "dual_state": dual_state,
                        "cm_state": cm_state,
                        "extension_run_allowed": False,
                        "official_imagenette_validation_accessed": False,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                raise RuntimeError("dual-arm training did not complete")
        time.sleep(30)

    if int(cm_state.get("epochs_completed", -1)) != 12 or cm_state.get("extension_executed") is not False:
        raise RuntimeError("CM equal-budget epoch/extension ledger failed")
    command = [
        sys.executable,
        str(ROOT / "scripts/s34a_evaluate_swinjscc_equal_budget.py"),
        "--config",
        str(ROOT / "configs/s34a_swinjscc_equal_budget_evaluation.yaml"),
        "--device",
        "cuda:0",
    ]
    write_state(
        {
            "status": "launching_equal_budget_policy_dev_evaluation",
            "command": command,
            "extension_run_allowed": False,
            "official_imagenette_validation_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    write_state(
        {
            "status": "evaluation_process_finished",
            "returncode": result.returncode,
            "evaluation_state": read_json(EVAL_OUTPUT / "STATE.json"),
            "extension_run_allowed": False,
            "official_imagenette_validation_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
