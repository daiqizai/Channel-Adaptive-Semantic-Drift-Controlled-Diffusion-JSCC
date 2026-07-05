from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep receiver-side semantic gate policies for EXP-S4 residual outputs.")
    parser.add_argument("--input-csv", default="outputs/EXP-S4-006/per_sample.csv")
    parser.add_argument("--config", default="outputs/EXP-S4-006/config.yaml")
    parser.add_argument("--output-dir", default="outputs/analysis/exp_s4_006_gate_policy_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return transforms.ToTensor()(load_rgb_pil(path))


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = torch.mean((candidate - reference) ** 2).item()
    if mse <= 0.0:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def safe_name(value: float) -> str:
    return str(value).replace(".", "p")


def read_input_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["snr_db"] = float(str(row["snr_db"]).strip())
        row["residual_gate"] = float(str(row["residual_gate"]).strip())
        for key in [
            "detector_accept_refined",
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "m3_matches_original_top1",
        ]:
            row[key] = parse_bool(row[key])
    return rows


def load_classifier(config: dict[str, Any], device: torch.device):
    weights_file = resolve_project_path(config["classifier"]["weights_file"])
    if not weights_file.is_file() or weights_file.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {weights_file}")
    os.environ.setdefault("TORCH_HOME", str(resolve_project_path(config["classifier"]["cache_dir"])))
    import torchvision.models as models

    weights = getattr(models.AlexNet_Weights, str(config["classifier"]["weights"]))
    model = models.alexnet(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


@torch.no_grad()
def classify_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    topk: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], float]:
    predictions: dict[str, dict[str, Any]] = {}
    elapsed = 0.0
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([preprocess(load_rgb_pil(path)) for path in batch_paths]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=-1)
        values, indices = torch.topk(probabilities, k=topk, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for path, row_values, row_indices in zip(batch_paths, values.cpu(), indices.cpu()):
            predictions[project_relative(path)] = {
                "top_indices": [int(item) for item in row_indices.tolist()],
                "top_probs": [float(item) for item in row_values.tolist()],
            }
    return predictions, elapsed


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def enrich_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    categories: list[str],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    tensor_cache: dict[str, torch.Tensor] = {}

    def tensor(path_text: str) -> torch.Tensor:
        if path_text not in tensor_cache:
            tensor_cache[path_text] = load_rgb_tensor(resolve_project_path(path_text))
        return tensor_cache[path_text]

    for row in rows:
        out = dict(row)
        for prefix, path_key in [
            ("original", "original"),
            ("m0", "m0_reconstruction"),
            ("refined", "refined"),
        ]:
            pred = predictions[str(out[path_key])]
            indices = list(pred["top_indices"])
            probs = list(pred["top_probs"])
            out[f"{prefix}_top_indices"] = indices
            out[f"{prefix}_top_probs"] = probs
            out[f"{prefix}_top_labels"] = [label_for(categories, index) for index in indices]
            out[f"{prefix}_top1_index"] = indices[0]
            out[f"{prefix}_top1_prob"] = probs[0]
            out[f"{prefix}_top1_label"] = label_for(categories, indices[0])

        original_top1 = int(out["original_top1_index"])
        m0_top1 = int(out["m0_top1_index"])
        refined_top1 = int(out["refined_top1_index"])
        out["m0_matches_original_top1"] = m0_top1 == original_top1
        out["refined_matches_original_top1"] = refined_top1 == original_top1
        out["refined_matches_m0_top1"] = refined_top1 == m0_top1
        original = tensor(str(out["original"]))
        m0 = tensor(str(out["m0_reconstruction"]))
        refined = tensor(str(out["refined"]))
        out["m0_psnr_db"] = psnr(original, m0)
        out["refined_psnr_db"] = psnr(original, refined)
        enriched.append(out)
    return enriched


def list_text(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, list):
            output[key] = list_text(value)
        elif isinstance(value, bool):
            output[key] = bool_text(value)
        else:
            output[key] = value
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(serialize_row(row))


GatePolicy = Callable[[dict[str, Any]], bool]


def top1_equal(row: dict[str, Any]) -> bool:
    return int(row["refined_top1_index"]) == int(row["m0_top1_index"])


def refined_top1_in_m0_topk(row: dict[str, Any]) -> bool:
    return int(row["refined_top1_index"]) in set(row["m0_top_indices"])


def m0_top1_in_refined_topk(row: dict[str, Any]) -> bool:
    return int(row["m0_top1_index"]) in set(row["refined_top_indices"])


def mutual_top1_in_topk(row: dict[str, Any]) -> bool:
    return refined_top1_in_m0_topk(row) and m0_top1_in_refined_topk(row)


def any_topk_overlap(row: dict[str, Any]) -> bool:
    return bool(set(row["m0_top_indices"]) & set(row["refined_top_indices"]))


def build_policies() -> list[tuple[str, GatePolicy]]:
    policies: list[tuple[str, GatePolicy]] = [
        ("top1_equal", top1_equal),
        ("refined_top1_in_m0_top5", refined_top1_in_m0_topk),
        ("m0_top1_in_refined_top5", m0_top1_in_refined_topk),
        ("mutual_top1_in_top5", mutual_top1_in_topk),
        ("any_top5_overlap", any_topk_overlap),
    ]

    for margin in [0.05, 0.10, 0.15, 0.20, 0.30]:
        policies.append(
            (
                f"top1_equal_or_refined_conf_gain_ge_{safe_name(margin)}",
                lambda row, margin=margin: top1_equal(row)
                or float(row["refined_top1_prob"]) >= float(row["m0_top1_prob"]) + margin,
            )
        )

    for refined_min in [0.30, 0.40, 0.50]:
        policies.append(
            (
                f"top1_equal_or_refined_conf_ge_{safe_name(refined_min)}_and_m0_top1_in_refined_top5",
                lambda row, refined_min=refined_min: top1_equal(row)
                or (
                    m0_top1_in_refined_topk(row)
                    and float(row["refined_top1_prob"]) >= refined_min
                ),
            )
        )

    for m0_max in [0.20, 0.30, 0.40]:
        for refined_min in [0.30, 0.40, 0.50]:
            policies.append(
                (
                    f"top1_equal_or_m0conf_le_{safe_name(m0_max)}_refconf_ge_{safe_name(refined_min)}",
                    lambda row, m0_max=m0_max, refined_min=refined_min: top1_equal(row)
                    or (
                        float(row["m0_top1_prob"]) <= m0_max
                        and float(row["refined_top1_prob"]) >= refined_min
                    ),
                )
            )
    return policies


def evaluate_policy(
    rows: list[dict[str, Any]],
    policy_name: str,
    accept_fn: GatePolicy,
    snr: float | None,
) -> dict[str, Any]:
    subset = rows if snr is None else [row for row in rows if float(row["snr_db"]) == snr]
    total = len(subset)
    accepted = 0
    m0_correct = 0
    refined_correct = 0
    final_correct = 0
    false_accept = 0
    false_reject = 0
    accepted_repair = 0
    missed_repair = 0
    protective_reject = 0
    accepted_new_error = 0
    rejected_both_wrong = 0
    final_psnrs: list[float] = []
    m0_psnrs: list[float] = []
    refined_psnrs: list[float] = []

    for row in subset:
        accept = bool(accept_fn(row))
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        final_ok = refined_ok if accept else m0_ok
        accepted += int(accept)
        m0_correct += int(m0_ok)
        refined_correct += int(refined_ok)
        final_correct += int(final_ok)
        false_accept += int(accept and not refined_ok)
        false_reject += int((not accept) and refined_ok)
        accepted_repair += int(accept and (not m0_ok) and refined_ok)
        missed_repair += int((not accept) and (not m0_ok) and refined_ok)
        protective_reject += int((not accept) and m0_ok and not refined_ok)
        accepted_new_error += int(accept and m0_ok and not refined_ok)
        rejected_both_wrong += int((not accept) and (not m0_ok) and (not refined_ok))
        m0_psnr = float(row["m0_psnr_db"])
        refined_psnr = float(row["refined_psnr_db"])
        final_psnrs.append(refined_psnr if accept else m0_psnr)
        m0_psnrs.append(m0_psnr)
        refined_psnrs.append(refined_psnr)

    final_failure = 1.0 - rate(final_correct, total)
    m0_failure = 1.0 - rate(m0_correct, total)
    refined_failure = 1.0 - rate(refined_correct, total)
    return {
        "policy": policy_name,
        "snr_db": "all" if snr is None else float(snr),
        "num_images": total,
        "accept_count": accepted,
        "accept_rate": rate(accepted, total),
        "reject_count": total - accepted,
        "reject_rate": rate(total - accepted, total),
        "m0_failure_rate": m0_failure,
        "refined_failure_rate": refined_failure,
        "final_failure_rate": final_failure,
        "final_correct_rate": rate(final_correct, total),
        "false_accept_count": false_accept,
        "false_accept_rate": rate(false_accept, total),
        "false_reject_count": false_reject,
        "false_reject_rate": rate(false_reject, total),
        "accepted_repair_count": accepted_repair,
        "accepted_repair_rate": rate(accepted_repair, total),
        "missed_repair_count": missed_repair,
        "missed_repair_rate": rate(missed_repair, total),
        "protective_reject_count": protective_reject,
        "protective_reject_rate": rate(protective_reject, total),
        "accepted_new_error_count": accepted_new_error,
        "accepted_new_error_rate": rate(accepted_new_error, total),
        "rejected_both_wrong_count": rejected_both_wrong,
        "rejected_both_wrong_rate": rate(rejected_both_wrong, total),
        "m0_psnr_db": mean(m0_psnrs),
        "refined_psnr_db": mean(refined_psnrs),
        "final_psnr_db": mean(final_psnrs),
        "final_delta_psnr_vs_m0_db": mean(final_psnrs) - mean(m0_psnrs),
        "final_delta_psnr_vs_refined_db": mean(final_psnrs) - mean(refined_psnrs),
    }


def add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["policy"] == "top1_equal":
            baselines[str(row["snr_db"])] = row
    output = []
    for row in rows:
        baseline = baselines[str(row["snr_db"])]
        enriched = dict(row)
        enriched["delta_final_failure_vs_top1_equal"] = (
            float(row["final_failure_rate"]) - float(baseline["final_failure_rate"])
        )
        enriched["delta_final_psnr_vs_top1_equal_db"] = (
            float(row["final_psnr_db"]) - float(baseline["final_psnr_db"])
        )
        enriched["delta_missed_repair_vs_top1_equal"] = (
            int(row["missed_repair_count"]) - int(baseline["missed_repair_count"])
        )
        enriched["delta_accepted_new_error_vs_top1_equal"] = (
            int(row["accepted_new_error_count"]) - int(baseline["accepted_new_error_count"])
        )
        output.append(enriched)
    return output


def make_report(
    global_rows: list[dict[str, Any]],
    by_snr_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    baseline = next(row for row in global_rows if row["policy"] == "top1_equal")
    nonworse = [
        row
        for row in global_rows
        if float(row["delta_final_failure_vs_top1_equal"]) <= 0.0
    ]
    nonworse_sorted = sorted(
        nonworse,
        key=lambda row: (
            float(row["delta_final_psnr_vs_top1_equal_db"]),
            -float(row["final_failure_rate"]),
            -float(row["missed_repair_count"]),
        ),
        reverse=True,
    )
    repair_sorted = sorted(
        global_rows,
        key=lambda row: (
            int(row["missed_repair_count"]),
            float(row["final_failure_rate"]),
            -float(row["final_psnr_db"]),
        ),
    )
    lines = [
        "# EXP-S4-006 Gate Policy Sweep",
        "",
        "This derived analysis recomputes AlexNet top-5 predictions for original/M0/refined images and evaluates receiver-side gate policies offline.",
        "",
        "The policies do not use original labels at decision time. Original pseudo top-1 is used only for offline evaluation.",
        "",
        "## Baseline",
        "",
        "| Policy | Final Failure | Final PSNR | Accept | Missed Repair | Accepted New Error |",
        "|---|---:|---:|---:|---:|---:|",
        "| {policy} | {fail:.4f} | {psnr:.4f} | {accept:.4f} | {missed} | {new_error} |".format(
            policy=baseline["policy"],
            fail=float(baseline["final_failure_rate"]),
            psnr=float(baseline["final_psnr_db"]),
            accept=float(baseline["accept_rate"]),
            missed=int(baseline["missed_repair_count"]),
            new_error=int(baseline["accepted_new_error_count"]),
        ),
        "",
        "## Best Policies With No Global Final-Failure Increase",
        "",
        "| Policy | Final Failure | Delta Failure | Final PSNR | Delta PSNR | Missed Repair | Accepted New Error | Accept |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in nonworse_sorted[:12]:
        lines.append(
            "| {policy} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {missed} | {new_error} | {accept:.4f} |".format(
                policy=row["policy"],
                fail=float(row["final_failure_rate"]),
                dfail=float(row["delta_final_failure_vs_top1_equal"]),
                psnr=float(row["final_psnr_db"]),
                dpsnr=float(row["delta_final_psnr_vs_top1_equal_db"]),
                missed=int(row["missed_repair_count"]),
                new_error=int(row["accepted_new_error_count"]),
                accept=float(row["accept_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Lowest Missed-Repair Policies",
            "",
            "| Policy | Missed Repair | Final Failure | Delta Failure | Final PSNR | Accepted Repair | Accepted New Error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repair_sorted[:12]:
        lines.append(
            "| {policy} | {missed} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {accepted_repair} | {new_error} |".format(
                policy=row["policy"],
                missed=int(row["missed_repair_count"]),
                fail=float(row["final_failure_rate"]),
                dfail=float(row["delta_final_failure_vs_top1_equal"]),
                psnr=float(row["final_psnr_db"]),
                accepted_repair=int(row["accepted_repair_count"]),
                new_error=int(row["accepted_new_error_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `top1_equal` remains the safest baseline under the same AlexNet pseudo-label metric.",
            "- A useful next gate should reduce `missed_repair_count` without increasing `accepted_new_error_count` or final failure.",
            "- Any selected policy is validation-tuned and must be checked on a held-out split before being called final M3/Ours.",
            "",
            "## Output Files",
            "",
            f"- `topk_predictions.csv`: `{metadata['topk_predictions_csv']}`",
            f"- `policy_summary.csv`: `{metadata['policy_summary_csv']}`",
            f"- `policy_by_snr.csv`: `{metadata['policy_by_snr_csv']}`",
            f"- `metadata.json`: `{metadata['metadata_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_csv = resolve_project_path(args.input_csv)
    config_path = resolve_project_path(args.config)
    output_dir = resolve_project_path(args.output_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = resolve_device(args.device)
    rows = read_input_rows(input_csv)
    unique_paths = sorted(
        {
            project_relative(resolve_project_path(row[key]))
            for row in rows
            for key in ["original", "m0_reconstruction", "refined"]
        }
    )
    path_objects = [resolve_project_path(path) for path in unique_paths]
    batch_size = int(args.batch_size or config["classifier"]["batch_size"])
    model, preprocess, categories = load_classifier(config, device)
    predictions, classify_seconds = classify_paths(
        model=model,
        preprocess=preprocess,
        paths=path_objects,
        batch_size=batch_size,
        topk=int(args.topk),
        device=device,
    )
    enriched_rows = enrich_rows(rows, predictions, categories)

    policies = build_policies()
    snrs = sorted({float(row["snr_db"]) for row in enriched_rows})
    global_rows = [evaluate_policy(enriched_rows, name, fn, snr=None) for name, fn in policies]
    by_snr_rows = [
        evaluate_policy(enriched_rows, name, fn, snr=snr)
        for name, fn in policies
        for snr in snrs
    ]
    global_rows = add_baseline_deltas(global_rows)
    by_snr_rows = add_baseline_deltas(by_snr_rows)

    topk_csv = output_dir / "topk_predictions.csv"
    summary_csv = output_dir / "policy_summary.csv"
    by_snr_csv = output_dir / "policy_by_snr.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(topk_csv, enriched_rows)
    write_csv(summary_csv, global_rows)
    write_csv(by_snr_csv, by_snr_rows)

    metadata = {
        "input_csv": project_relative(input_csv),
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "topk_predictions_csv": project_relative(topk_csv),
        "policy_summary_csv": project_relative(summary_csv),
        "policy_by_snr_csv": project_relative(by_snr_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "topk": int(args.topk),
        "num_unique_images_classified": len(unique_paths),
        "classification_seconds": classify_seconds,
        "classifier": config["classifier"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; AlexNet weights are loaded from the local project cache.",
    }
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    report_md.write_text(make_report(global_rows, by_snr_rows, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_policies": len(policies),
                "num_rows": len(enriched_rows),
                "num_unique_images_classified": len(unique_paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
