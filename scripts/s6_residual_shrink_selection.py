from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate residual-strength shrink candidates over existing residual-refiner outputs."
    )
    parser.add_argument("--config", default="configs/s6_residual_shrink_selection_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def alpha_name(alpha: float) -> str:
    text = f"{alpha:.4f}".rstrip("0").rstrip(".")
    return "alpha_" + text.replace(".", "p")


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):+.{digits}f}"


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


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
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "N/A"


def git_dirty_state() -> str:
    try:
        output = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return "dirty" if output else "clean"


def proxy_environment_present() -> list[str]:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"]
    return [key for key in keys if os.environ.get(key)]


def validate_inputs(config: dict[str, Any], snrs: list[float]) -> dict[str, Any]:
    paths = {
        "source_config": resolve_project_path(config["inputs"]["source_config"]),
        "source_summary_csv": resolve_project_path(config["inputs"]["source_summary_csv"]),
        "source_per_sample_csv": resolve_project_path(config["inputs"]["source_per_sample_csv"]),
        "original_dir": resolve_project_path(config["inputs"]["original_dir"]),
        "m0_export_dir": resolve_project_path(config["inputs"]["m0_export_dir"]),
        "refined_export_dir": resolve_project_path(config["inputs"]["refined_export_dir"]),
        "jscc_checkpoint": resolve_project_path(config["inputs"]["jscc_checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
        "classifier_weights": resolve_project_path(config["classifier"]["weights_file"]),
    }
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "forbidden_checkpoint" and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if paths["jscc_checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if not paths["classifier_weights"].is_file() or paths["classifier_weights"].stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {paths['classifier_weights']}")
    for snr in snrs:
        refined_dir = paths["refined_export_dir"] / snr_name(snr) / "refined"
        if not refined_dir.exists():
            raise FileNotFoundError(f"Refined directory missing: {refined_dir}")
    return {key: project_relative(path) for key, path in paths.items()}


def load_classifier(config: dict[str, Any], device: torch.device):
    os.environ["TORCH_HOME"] = str(resolve_project_path(config["classifier"]["cache_dir"]))
    import torchvision.models as models

    weights = getattr(models.AlexNet_Weights, str(config["classifier"]["weights"]))
    model = models.alexnet(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def try_load_lpips(device: torch.device, cache_root: Path):
    try:
        os.environ["TORCH_HOME"] = str(cache_root)
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def classify_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    topk: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    outputs: list[dict[str, Any]] = []
    elapsed = 0.0
    with torch.no_grad():
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
            for row_values, row_indices in zip(values.cpu(), indices.cpu()):
                outputs.append(
                    {
                        "top_indices": [int(item) for item in row_indices.tolist()],
                        "top_probs": [float(item) for item in row_values.tolist()],
                    }
                )
    return outputs, elapsed


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | None]:
    metrics: dict[str, list[float]] = {"mse": [], "psnr_db": [], "ssim": [], "ms_ssim": []}
    lpips_values: list[float] = []
    with torch.no_grad():
        for start in range(0, reference.shape[0], batch_size):
            ref = reference[start : start + batch_size].to(device)
            cand = candidate[start : start + batch_size].to(device)
            mse_values = F.mse_loss(cand, ref, reduction="none").flatten(start_dim=1).mean(dim=1)
            metrics["mse"].extend(mse_values.detach().cpu().tolist())
            metrics["psnr_db"].extend(psnr_per_sample(cand, ref).detach().cpu().tolist())
            metrics["ssim"].extend(ssim_per_sample(cand, ref).detach().cpu().tolist())
            metrics["ms_ssim"].extend(ms_ssim_per_sample(cand, ref).detach().cpu().tolist())
            if lpips_model is not None:
                values = lpips_model(cand * 2.0 - 1.0, ref * 2.0 - 1.0)
                lpips_values.extend(values.flatten().detach().cpu().tolist())
    return {
        "mse": mean(metrics["mse"]),
        "psnr_db": mean(metrics["psnr_db"]),
        "ssim": mean(metrics["ssim"]),
        "ms_ssim": mean(metrics["ms_ssim"]),
        "lpips": mean(lpips_values) if lpips_model is not None else None,
    }


def path_from_row(row: dict[str, str], key: str) -> Path:
    return resolve_project_path(row[key])


def bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def build_candidate_images(
    rows: list[dict[str, str]],
    alpha: float,
    candidate_dir: Path,
) -> dict[tuple[float, str], Path]:
    candidate_paths: dict[tuple[float, str], Path] = {}
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["snr_db"]), []).append(row)
    for snr, snr_rows in grouped.items():
        out_dir = candidate_dir / alpha_name(alpha) / snr_name(snr)
        out_dir.mkdir(parents=True, exist_ok=False)
        for row in snr_rows:
            name = row["sample"]
            m0 = load_rgb_tensor(path_from_row(row, "m0_reconstruction"))
            refined = load_rgb_tensor(path_from_row(row, "refined"))
            candidate = (m0 + float(alpha) * (refined - m0)).clamp(0.0, 1.0)
            out_path = out_dir / name
            save_image(candidate, out_path)
            candidate_paths[(snr, name)] = out_path
    return candidate_paths


def summarize_policy(rows: list[dict[str, Any]], policy: str, alpha: float | None, snr: float | str) -> dict[str, Any]:
    if not rows:
        return {}
    m0_failure = 1.0 - rate([bool(row["m0_matches_original_top1"]) for row in rows])
    final_failure = 1.0 - rate([bool(row["final_matches_original_top1"]) for row in rows])
    candidate_failure = 1.0 - rate([bool(row["candidate_matches_original_top1"]) for row in rows])
    accept_rate = rate([bool(row["accept_candidate"]) for row in rows])
    repair_count = sum(
        1
        for row in rows
        if bool(row["accept_candidate"])
        and not bool(row["m0_matches_original_top1"])
        and bool(row["candidate_matches_original_top1"])
    )
    accepted_new_error_count = sum(
        1
        for row in rows
        if bool(row["accept_candidate"])
        and bool(row["m0_matches_original_top1"])
        and not bool(row["candidate_matches_original_top1"])
    )
    rejected_good_count = sum(
        1 for row in rows if not bool(row["accept_candidate"]) and bool(row["candidate_matches_original_top1"])
    )
    return {
        "policy": policy,
        "alpha": "" if alpha is None else float(alpha),
        "snr_db": snr,
        "num_images": len(rows),
        "accept_rate": accept_rate,
        "m0_failure_rate": m0_failure,
        "candidate_failure_rate": candidate_failure,
        "final_failure_rate": final_failure,
        "delta_final_failure_vs_m0": final_failure - m0_failure,
        "candidate_refinement_drift_rate": 1.0
        - rate([bool(row["candidate_matches_m0_top1"]) for row in rows]),
        "final_refinement_drift_rate": 1.0 - rate([bool(row["final_matches_m0_top1"]) for row in rows]),
        "repair_count": repair_count,
        "accepted_new_error_count": accepted_new_error_count,
        "rejected_good_count": rejected_good_count,
    }


def attach_metrics(summary: dict[str, Any], metrics: dict[str, float | None], m0_metrics: dict[str, float | None]) -> None:
    for key, value in metrics.items():
        summary[f"final_{key}"] = value
    summary["delta_psnr_vs_m0_db"] = (
        None if metrics["psnr_db"] is None or m0_metrics["psnr_db"] is None else metrics["psnr_db"] - m0_metrics["psnr_db"]
    )
    summary["delta_lpips_vs_m0"] = (
        None if metrics["lpips"] is None or m0_metrics["lpips"] is None else metrics["lpips"] - m0_metrics["lpips"]
    )
    summary["delta_ms_ssim_vs_m0"] = (
        None
        if metrics["ms_ssim"] is None or m0_metrics["ms_ssim"] is None
        else metrics["ms_ssim"] - m0_metrics["ms_ssim"]
    )


def make_policy_rows_for_snr(
    snr_rows: list[dict[str, str]],
    alpha: float | None,
    candidate_preds: list[dict[str, Any]] | None,
    candidate_paths: list[Path] | None,
    policy: str,
    categories: list[str],
) -> list[dict[str, Any]]:
    policy_rows: list[dict[str, Any]] = []
    for idx, source in enumerate(snr_rows):
        original_top1 = int(source["original_top1_index"])
        m0_top1 = int(source["m0_top1_index"])
        m0_top1_prob = float(source["m0_top1_prob"])
        if policy == "m0":
            candidate_top1 = m0_top1
            candidate_prob = m0_top1_prob
            candidate_path = path_from_row(source, "m0_reconstruction")
            accept = False
        else:
            assert candidate_preds is not None
            assert candidate_paths is not None
            candidate_top1 = int(candidate_preds[idx]["top_indices"][0])
            candidate_prob = float(candidate_preds[idx]["top_probs"][0])
            candidate_path = candidate_paths[idx]
            accept = policy == "always_alpha" or candidate_top1 == m0_top1
        final_top1 = candidate_top1 if accept else m0_top1
        final_prob = candidate_prob if accept else m0_top1_prob
        candidate_matches_origin = candidate_top1 == original_top1
        candidate_matches_m0 = candidate_top1 == m0_top1
        m0_matches_origin = bool_from_csv(source["m0_matches_original_top1"])
        row = {
            "policy": policy,
            "alpha": "" if alpha is None else float(alpha),
            "snr_db": float(source["snr_db"]),
            "sample": source["sample"],
            "original": source["original"],
            "m0_reconstruction": source["m0_reconstruction"],
            "source_refined": source["refined"],
            "candidate": project_relative(candidate_path),
            "accept_candidate": accept,
            "final_source": project_relative(candidate_path if accept else path_from_row(source, "m0_reconstruction")),
            "original_top1_index": original_top1,
            "original_top1_label": source["original_top1_label"],
            "original_top1_prob": float(source["original_top1_prob"]),
            "m0_top1_index": m0_top1,
            "m0_top1_label": source["m0_top1_label"],
            "m0_top1_prob": m0_top1_prob,
            "candidate_top1_index": candidate_top1,
            "candidate_top1_label": label_for(categories, candidate_top1),
            "candidate_top1_prob": candidate_prob,
            "final_top1_index": final_top1,
            "final_top1_label": label_for(categories, final_top1),
            "final_top1_prob": final_prob,
            "m0_matches_original_top1": m0_matches_origin,
            "candidate_matches_original_top1": candidate_matches_origin,
            "candidate_matches_m0_top1": candidate_matches_m0,
            "final_matches_original_top1": final_top1 == original_top1,
            "final_matches_m0_top1": final_top1 == m0_top1,
            "accepted_repair": accept and (not m0_matches_origin) and candidate_matches_origin,
            "accepted_new_error": accept and m0_matches_origin and (not candidate_matches_origin),
            "rejected_good": (not accept) and candidate_matches_origin,
        }
        policy_rows.append(row)
    return policy_rows


def stack_paths(paths: list[Path]) -> torch.Tensor:
    return torch.stack([load_rgb_tensor(path) for path in paths])


def tensors_for_policy(snr_rows: list[dict[str, str]], policy_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    references = stack_paths([path_from_row(row, "original") for row in snr_rows])
    m0 = stack_paths([path_from_row(row, "m0_reconstruction") for row in snr_rows])
    final = stack_paths([resolve_project_path(row["final_source"]) for row in policy_rows])
    return references, m0, final


def choose_schedule(
    summary_rows: list[dict[str, Any]],
    policy: str,
    snrs: list[float],
    residual_gates: dict[float, float] | None = None,
    enforce_effective_strength_nonincreasing: bool = False,
) -> tuple[dict[float, float | None], list[dict[str, Any]]]:
    candidates_by_snr: dict[float, list[dict[str, Any]]] = {}
    m0_by_snr: dict[float, dict[str, Any]] = {}
    for snr in snrs:
        m0_row = next(row for row in summary_rows if row["policy"] == "m0" and float(row["snr_db"]) == float(snr))
        m0_by_snr[float(snr)] = m0_row
        candidates_by_snr[float(snr)] = [
            row
            for row in summary_rows
            if row["policy"] == policy
            and row["snr_db"] != "all"
            and float(row["snr_db"]) == float(snr)
            and float(row["final_failure_rate"]) <= float(m0_row["final_failure_rate"]) + 1e-12
        ]

    if enforce_effective_strength_nonincreasing:
        if residual_gates is None:
            raise ValueError("residual_gates are required for monotonic effective-strength selection")
        option_lists: list[list[dict[str, Any]]] = []
        for snr in snrs:
            options = [*candidates_by_snr[float(snr)], {**m0_by_snr[float(snr)], "alpha": None}]
            option_lists.append(options)

        valid_schedules: list[tuple[tuple[dict[str, Any], ...], list[float]]] = []
        for schedule in itertools.product(*option_lists):
            strengths = [
                float(residual_gates[float(snr)]) * (0.0 if row.get("alpha") in (None, "") else float(row["alpha"]))
                for snr, row in zip(snrs, schedule)
            ]
            if all(left + 1e-12 >= right for left, right in zip(strengths, strengths[1:])):
                valid_schedules.append((schedule, strengths))
        if not valid_schedules:
            raise RuntimeError("No schedule satisfies the effective-strength monotonicity constraint")

        best_schedule, best_strengths = max(
            valid_schedules,
            key=lambda item: (
                sum(float(row["final_psnr_db"]) for row in item[0]),
                -sum(float(row["final_failure_rate"]) for row in item[0]),
            ),
        )
        choices: dict[float, float | None] = {}
        choice_rows: list[dict[str, Any]] = []
        for snr, row, strength in zip(snrs, best_schedule, best_strengths):
            alpha = None if row.get("alpha") in (None, "") else float(row["alpha"])
            choices[float(snr)] = alpha
            choice_rows.append(
                {
                    **row,
                    "selected_for_schedule": policy,
                    "residual_gate": float(residual_gates[float(snr)]),
                    "effective_strength": strength,
                    "selection_reason": "max_mean_psnr_under_m0_failure_and_monotonic_effective_strength",
                }
            )
        return choices, choice_rows

    choices: dict[float, float | None] = {}
    choice_rows: list[dict[str, Any]] = []
    for snr in snrs:
        m0_row = m0_by_snr[float(snr)]
        candidates = candidates_by_snr[float(snr)]
        if not candidates:
            choices[float(snr)] = None
            choice_rows.append({**m0_row, "selected_for_schedule": policy, "selection_reason": "no_safe_alpha_fallback_to_m0"})
            continue
        best = max(candidates, key=lambda row: (float(row["final_psnr_db"]), -float(row["final_failure_rate"])))
        choices[float(snr)] = float(best["alpha"])
        choice_rows.append({**best, "selected_for_schedule": policy, "selection_reason": "max_psnr_under_m0_failure"})
    return choices, choice_rows


def build_scheduled_policy_rows(
    all_policy_rows: list[dict[str, Any]],
    choices: dict[float, float | None],
    source_policy: str,
    schedule_policy: str,
) -> list[dict[str, Any]]:
    scheduled: list[dict[str, Any]] = []
    for snr, alpha in choices.items():
        if alpha is None:
            rows = [
                row
                for row in all_policy_rows
                if row["policy"] == "m0" and abs(float(row["snr_db"]) - float(snr)) < 1e-9
            ]
        else:
            rows = [
                row
                for row in all_policy_rows
                if row["policy"] == source_policy
                and abs(float(row["snr_db"]) - float(snr)) < 1e-9
                and abs(float(row["alpha"]) - float(alpha)) < 1e-9
            ]
        for row in rows:
            scheduled.append({**row, "policy": schedule_policy, "selected_alpha": "" if alpha is None else float(alpha)})
    return scheduled


def aggregate_all_summary(
    policy_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    lpips_model,
    device: torch.device,
    batch_size: int,
    alpha: float | None = None,
) -> dict[str, Any]:
    grouped_by_snr: dict[float, list[dict[str, Any]]] = {}
    for row in policy_rows:
        grouped_by_snr.setdefault(float(row["snr_db"]), []).append(row)
    all_references: list[torch.Tensor] = []
    all_m0: list[torch.Tensor] = []
    all_final: list[torch.Tensor] = []
    for rows in grouped_by_snr.values():
        all_references.extend([load_rgb_tensor(resolve_project_path(row["original"])) for row in rows])
        all_m0.extend([load_rgb_tensor(resolve_project_path(row["m0_reconstruction"])) for row in rows])
        all_final.extend([load_rgb_tensor(resolve_project_path(row["final_source"])) for row in rows])
    summary = summarize_policy(policy_rows, policy_rows[0]["policy"], alpha, "all")
    metrics = compute_pair_metrics(torch.stack(all_references), torch.stack(all_final), lpips_model, device, batch_size)
    m0_metrics = compute_pair_metrics(torch.stack(all_references), torch.stack(all_m0), lpips_model, device, batch_size)
    attach_metrics(summary, metrics, m0_metrics)
    summary_rows.append(summary)
    return summary


def style_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_tradeoff(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    rows = [row for row in summary_rows if row["snr_db"] != "all" and row["policy"] in {"always_alpha", "top1_fallback_alpha"}]
    snrs = sorted({float(row["snr_db"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"always_alpha": "#6b8f3a", "top1_fallback_alpha": "#b57b2a"}
    labels = {"always_alpha": "always alpha", "top1_fallback_alpha": "top-1 fallback alpha"}
    for policy in ["always_alpha", "top1_fallback_alpha"]:
        for snr in snrs:
            snr_rows = sorted(
                [row for row in rows if row["policy"] == policy and float(row["snr_db"]) == snr],
                key=lambda row: float(row["alpha"]),
            )
            line_label = labels[policy] if snr == snrs[0] else None
            axes[0].plot(
                [float(row["alpha"]) for row in snr_rows],
                [float(row["delta_psnr_vs_m0_db"]) for row in snr_rows],
                marker="o",
                color=colors[policy],
                alpha=0.35 + 0.1 * snrs.index(snr),
                label=line_label,
            )
            axes[1].plot(
                [float(row["alpha"]) for row in snr_rows],
                [float(row["final_failure_rate"]) for row in snr_rows],
                marker="o",
                color=colors[policy],
                alpha=0.35 + 0.1 * snrs.index(snr),
            )
            axes[2].plot(
                [float(row["alpha"]) for row in snr_rows],
                [float(row["accept_rate"]) for row in snr_rows],
                marker="o",
                color=colors[policy],
                alpha=0.35 + 0.1 * snrs.index(snr),
            )
    style_axes(axes[0], "Quality vs Residual Strength", "alpha", "PSNR delta vs M0 (dB)")
    style_axes(axes[1], "Pseudo Failure vs Residual Strength", "alpha", "failure rate")
    style_axes(axes[2], "Top-1 Gate Acceptance", "alpha", "accept rate")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_sample_grids(
    rows: list[dict[str, Any]],
    output_dir: Path,
    sample_grid_count: int,
) -> list[str]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["snr_db"]), []).append(row)
    for snr, snr_rows in sorted(grouped.items()):
        selected = snr_rows[:sample_grid_count]
        tensors: list[torch.Tensor] = []
        tensors.extend([load_rgb_tensor(resolve_project_path(row["original"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["m0_reconstruction"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["candidate"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["final_source"])) for row in selected])
        path = sample_dir / f"{snr_name(snr)}_original_m0_candidate_final.png"
        save_image(torch.stack(tensors), path, nrow=len(selected))
        paths.append(project_relative(path))
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    headers = [label for _key, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        values = []
        for key, _label in columns:
            values.append(fmt(row.get(key, "")))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(
    summary_rows: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    all_rows = {row["policy"]: row for row in summary_rows if row["snr_db"] == "all"}
    selected = all_rows["selected_top1_fallback_shrink_schedule"]
    always_constrained = all_rows.get("selected_always_m0_failure_constrained_schedule")
    alpha_one_top1 = next(
        row for row in summary_rows if row["snr_db"] == "all" and row["policy"] == "top1_fallback_alpha" and float(row["alpha"]) == 1.0
    )
    alpha_one_always = next(
        row for row in summary_rows if row["snr_db"] == "all" and row["policy"] == "always_alpha" and float(row["alpha"]) == 1.0
    )

    source_experiment = str(metadata.get("source_experiment") or "source experiment")
    split_note = str(metadata.get("selection_split") or f"{source_experiment} validation split")

    lines = [
        f"# {source_experiment} Residual Shrink Selection",
        "",
        "This derived validation-only analysis evaluates whether shrinking the existing residual refinement can improve the quality/semantic tradeoff.",
        "It reads existing M0 and refined PNG files, forms `x_alpha = clamp(m0 + alpha * (refined - m0), 0, 1)`, then evaluates frozen AlexNet pseudo-label consistency.",
        "",
        "## Bottom Line",
        "",
        f"- Existing full-strength top-1 fallback (`alpha=1.0`) gives mean PSNR delta `{signed(alpha_one_top1['delta_psnr_vs_m0_db'])}` dB vs M0 with final failure delta `{signed(alpha_one_top1['delta_final_failure_vs_m0'])}`.",
        f"- The validation-selected per-SNR top-1 fallback shrink schedule gives mean PSNR delta `{signed(selected['delta_psnr_vs_m0_db'])}` dB vs M0 with final failure delta `{signed(selected['delta_final_failure_vs_m0'])}`.",
        f"- Always accepting full-strength residual (`alpha=1.0`) gives mean PSNR delta `{signed(alpha_one_always['delta_psnr_vs_m0_db'])}` dB but final failure delta `{signed(alpha_one_always['delta_final_failure_vs_m0'])}`.",
    ]
    if always_constrained is not None:
        lines.append(
            f"- The best always-accept schedule under the coarse M0 failure-rate constraint gives mean PSNR delta `{signed(always_constrained['delta_psnr_vs_m0_db'])}` dB, but it still has `{always_constrained['accepted_new_error_count']}` accepted new errors, so it is not a safe M3."
        )
    lines.extend(
        [
            "- Interpretation: if lower alpha improves acceptance or semantic risk, the next trainable M3 should control residual strength directly instead of only adding receiver-side vetoes.",
            "",
            "## Selected Schedule",
            "",
        ]
    )
    lines += markdown_table(
        schedule_rows,
        [
            ("selected_for_schedule", "Schedule"),
            ("snr_db", "SNR"),
            ("alpha", "Alpha"),
            ("effective_strength", "Effective Strength"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("final_failure_rate", "Failure"),
            ("accept_rate", "Accept"),
            ("repair_count", "Repair"),
            ("accepted_new_error_count", "New Error"),
            ("selection_reason", "Reason"),
        ],
    )
    lines.extend(["", "## All-Policy Summary", ""])
    compact = [
        row
        for row in summary_rows
        if row["snr_db"] == "all"
        or row["policy"]
        in {"m0", "selected_top1_fallback_shrink_schedule", "selected_always_m0_failure_constrained_schedule"}
    ]
    lines += markdown_table(
        compact,
        [
            ("policy", "Policy"),
            ("alpha", "Alpha"),
            ("snr_db", "SNR"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_lpips_vs_m0", "Delta LPIPS"),
            ("final_failure_rate", "Failure"),
            ("delta_final_failure_vs_m0", "Delta Failure"),
            ("accept_rate", "Accept"),
            ("repair_count", "Repair"),
            ("accepted_new_error_count", "New Error"),
        ],
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Schedule JSON: `{metadata['schedule_json']}`",
            f"- Tradeoff figure: `{metadata['tradeoff_figure']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
            "## Caveats",
            "",
            f"- This uses the same {split_note}, so selected alphas are not test-safe.",
            "- The semantic metric is still frozen AlexNet pseudo-label consistency on COCO images.",
            "- Lower alpha is a proxy for residual-strength control, not a trained semantic-risk-aware refiner yet.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    snrs = [float(item) for item in config["snrs"]]
    alphas = [float(item) for item in config["shrink"]["alphas"]]
    input_manifest = validate_inputs(config, snrs)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "snrs": snrs,
        "alphas": alphas,
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    rows = read_csv(resolve_project_path(config["inputs"]["source_per_sample_csv"]))
    rows = sorted(rows, key=lambda row: (float(row["snr_db"]), row["sample"]))
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["snr_db"]), []).append(row)

    classifier_model, classifier_preprocess, categories = load_classifier(config, device)
    lpips_model = None
    lpips_error = None
    if not args.skip_lpips:
        lpips_model, lpips_error = try_load_lpips(
            device,
            resolve_project_path(config["classifier"]["cache_dir"]),
        )

    candidate_root = output_dir / "candidates"
    candidate_path_map: dict[float, dict[tuple[float, str], Path]] = {}
    candidate_pred_map: dict[float, dict[float, list[dict[str, Any]]]] = {}
    classification_times: dict[str, float] = {}
    for alpha in alphas:
        candidate_path_map[alpha] = build_candidate_images(rows, alpha, candidate_root)
        candidate_pred_map[alpha] = {}
        for snr in snrs:
            snr_rows = grouped[snr]
            paths = [candidate_path_map[alpha][(snr, row["sample"])] for row in snr_rows]
            preds, elapsed = classify_paths(
                classifier_model,
                classifier_preprocess,
                paths,
                int(config["classifier"]["batch_size"]),
                int(config["classifier"]["topk"]),
                device,
            )
            candidate_pred_map[alpha][snr] = preds
            classification_times[f"{alpha_name(alpha)}_{snr_name(snr)}"] = elapsed

    policy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    image_batch_size = int(config["evaluation"]["image_batch_size"])

    for snr in snrs:
        snr_rows = grouped[snr]
        m0_policy_rows = make_policy_rows_for_snr(snr_rows, None, None, None, "m0", categories)
        references, m0_tensor, final_tensor = tensors_for_policy(snr_rows, m0_policy_rows)
        m0_metrics = compute_pair_metrics(references, m0_tensor, lpips_model, device, image_batch_size)
        m0_summary = summarize_policy(m0_policy_rows, "m0", None, snr)
        attach_metrics(m0_summary, m0_metrics, m0_metrics)
        policy_rows.extend(m0_policy_rows)
        summary_rows.append(m0_summary)

        for alpha in alphas:
            candidate_paths = [candidate_path_map[alpha][(snr, row["sample"])] for row in snr_rows]
            candidate_preds = candidate_pred_map[alpha][snr]
            for policy in ["always_alpha", "top1_fallback_alpha"]:
                rows_for_policy = make_policy_rows_for_snr(
                    snr_rows,
                    alpha,
                    candidate_preds,
                    candidate_paths,
                    policy,
                    categories,
                )
                references, _m0_tensor, final_tensor = tensors_for_policy(snr_rows, rows_for_policy)
                metrics = compute_pair_metrics(references, final_tensor, lpips_model, device, image_batch_size)
                summary = summarize_policy(rows_for_policy, policy, alpha, snr)
                attach_metrics(summary, metrics, m0_metrics)
                policy_rows.extend(rows_for_policy)
                summary_rows.append(summary)

    for policy in ["m0", "always_alpha", "top1_fallback_alpha"]:
        if policy == "m0":
            rows_for_policy = [row for row in policy_rows if row["policy"] == "m0"]
            aggregate_all_summary(rows_for_policy, summary_rows, lpips_model, device, image_batch_size)
        else:
            for alpha in alphas:
                rows_for_policy = [
                    row for row in policy_rows if row["policy"] == policy and abs(float(row["alpha"]) - alpha) < 1e-9
                ]
                aggregate_all_summary(rows_for_policy, summary_rows, lpips_model, device, image_batch_size, alpha=alpha)

    selection_cfg = config.get("shrink", {}).get("selection", {})
    enforce_monotonic = bool(selection_cfg.get("enforce_effective_strength_nonincreasing", False))
    source_config_path = resolve_project_path(config["inputs"]["source_config"])
    with source_config_path.open("r", encoding="utf-8") as handle:
        source_config = yaml.safe_load(handle)
    residual_gates = {
        float(key): float(value) for key, value in source_config["model"]["residual_gates"].items()
    }
    top1_choices, top1_choice_rows = choose_schedule(
        summary_rows,
        "top1_fallback_alpha",
        snrs,
        residual_gates=residual_gates,
        enforce_effective_strength_nonincreasing=enforce_monotonic,
    )
    always_choices, always_choice_rows = choose_schedule(
        summary_rows,
        "always_alpha",
        snrs,
        residual_gates=residual_gates,
        enforce_effective_strength_nonincreasing=enforce_monotonic,
    )
    scheduled_top1_rows = build_scheduled_policy_rows(
        policy_rows,
        top1_choices,
        "top1_fallback_alpha",
        "selected_top1_fallback_shrink_schedule",
    )
    scheduled_always_rows = build_scheduled_policy_rows(
        policy_rows,
        always_choices,
        "always_alpha",
        "selected_always_m0_failure_constrained_schedule",
    )
    policy_rows.extend(scheduled_top1_rows)
    policy_rows.extend(scheduled_always_rows)
    aggregate_all_summary(scheduled_top1_rows, summary_rows, lpips_model, device, image_batch_size)
    aggregate_all_summary(scheduled_always_rows, summary_rows, lpips_model, device, image_batch_size)

    schedule_rows = top1_choice_rows + always_choice_rows
    tradeoff_figure = output_dir / "alpha_tradeoff.png"
    plot_tradeoff(summary_rows, tradeoff_figure)
    sample_grids = make_sample_grids(
        scheduled_top1_rows,
        output_dir,
        int(config["evaluation"]["sample_grid_count"]),
    )

    summary_csv = output_dir / "summary.csv"
    per_sample_csv = output_dir / "per_sample.csv"
    schedule_json = output_dir / "selected_schedule.json"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, policy_rows)
    schedule_payload = {
        "top1_fallback_alpha": {str(key): value for key, value in top1_choices.items()},
        "always_alpha": {str(key): value for key, value in always_choices.items()},
        "effective_strength_nonincreasing": enforce_monotonic,
        "residual_gates": {str(key): value for key, value in residual_gates.items()},
        "selection_rows": schedule_rows,
    }
    save_json(schedule_json, schedule_payload)
    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "source_experiment": config["inputs"].get("source_experiment", ""),
        "selection_split": config.get("shrink", {}).get("selection", {}).get("split", ""),
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "config": project_relative(config_path),
        "input_manifest": input_manifest,
        "summary_csv": project_relative(summary_csv),
        "per_sample_csv": project_relative(per_sample_csv),
        "schedule_json": project_relative(schedule_json),
        "tradeoff_figure": project_relative(tradeoff_figure),
        "sample_grids": sample_grids,
        "metadata_json": project_relative(metadata_json),
        "classification_times_sec": classification_times,
        "lpips_error": lpips_error,
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report = make_report(summary_rows, schedule_rows, metadata)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
