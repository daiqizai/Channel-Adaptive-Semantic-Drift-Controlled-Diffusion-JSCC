#!/usr/bin/env python3
"""Launch only CM-SA-12ep after the active Base-SA-12ep process completes.

This is a narrow fail-closed queue helper for the user-authorized equal-budget
dual arm.  It never launches an extension and refuses to start CM unless the
Base state says all 12 authorized epochs completed successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/train/EXP-S34A-SWINJSCC-BASE-SA-EQUAL-BUDGET-001"
CM = ROOT / "outputs/train/EXP-S34A-SWINJSCC-CM-SA-EQUAL-BUDGET-001"
STATE = ROOT / "outputs/train/S34A-SWINJSCC-DUAL-ARM-LAUNCHER-STATE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def write_state(value: dict) -> None:
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def process_is_expected_base(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return False
    command = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    return (
        "s34a_train_swinjscc_equal_budget.py" in command
        and "official_base_sa" in command
    )


def main() -> None:
    args = parse_args()
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError("poll interval must be between 5 and 60 seconds")
    if CM.exists():
        raise FileExistsError(CM)
    write_state(
        {
            "status": "waiting_for_base_equal_budget",
            "base_pid": args.base_pid,
            "extension_run_allowed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    while process_is_expected_base(args.base_pid):
        time.sleep(args.poll_seconds)

    base_state_path = BASE / "STATE.json"
    if not base_state_path.is_file():
        raise FileNotFoundError(base_state_path)
    base_state = json.loads(base_state_path.read_text(encoding="utf-8"))
    if base_state.get("status") != "complete_equal_budget_only_waiting_for_user_extension_decision":
        write_state(
            {
                "status": "base_not_complete_cm_not_started",
                "base_state": base_state,
                "extension_run_allowed": False,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        raise RuntimeError("Base-SA did not complete its authorized 12 epochs")
    if int(base_state.get("epochs_completed", -1)) != 12:
        raise RuntimeError("Base-SA epoch ledger is not exactly 12")
    if base_state.get("extension_executed") is not False:
        raise RuntimeError("unexpected extension state")

    command = [
        sys.executable,
        str(ROOT / "scripts/s34a_train_swinjscc_equal_budget.py"),
        "--arm",
        "capacity_matched_sa",
        "--device",
        "cuda:0",
    ]
    write_state(
        {
            "status": "launching_cm_equal_budget",
            "command": command,
            "maximum_cm_epochs": 12,
            "extension_run_allowed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    cm_state_path = CM / "STATE.json"
    cm_state = (
        json.loads(cm_state_path.read_text(encoding="utf-8"))
        if cm_state_path.is_file()
        else None
    )
    write_state(
        {
            "status": "cm_process_finished",
            "returncode": result.returncode,
            "cm_state": cm_state,
            "extension_run_allowed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
