from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit continuous-alpha tail refiner with LPIPS and classifier ensemble."
    )
    parser.add_argument("--config", default="configs/s6_continuous_alpha_tail_refiner_audit_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


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


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
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


def classifier_status(config: dict[str, Any]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for model_cfg in config["classifiers"]["models"]:
        weights_file = resolve_project_path(model_cfg["weights_file"])
        statuses.append(
            {
                "key": model_cfg["key"],
                "model_name": model_cfg["model_name"],
                "weights": model_cfg["weights"],
                "weights_file": project_relative(weights_file),
                "available": weights_file.is_file() and weights_file.stat().st_size > 1024 * 1024,
                "bytes": weights_file.stat().st_size if weights_file.exists() else 0,
            }
        )
    return statuses


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "source_per_sample_csv": resolve_project_path(config["inputs"]["source_per_sample_csv"]),
        "source_summary_csv": resolve_project_path(config["inputs"]["source_summary_csv"]),
        "source_config": resolve_project_path(config["inputs"]["source_config"]),
        "checkpoint": resolve_project_path(config["inputs"]["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    missing_weights = [
        status["weights_file"]
        for status in classifier_status(config)
        if not bool(status["available"])
    ]
    if missing_weights and not bool(config.get("allow_missing_classifier_weights", False)):
        raise FileNotFoundError("Missing local classifier weights:\n" + "\n".join(missing_weights))
    return {key: project_relative(path) for key, path in paths.items()}


def normalize_policy_rows(source_rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    policies = config["policies"]
    for policy_cfg in policies:
        source_policy = str(policy_cfg["source_policy"])
        for source in source_rows:
            if str(source.get("policy", "")) != source_policy:
                continue
            candidate_key = str(policy_cfg["candidate_path_field"])
            final_key = str(policy_cfg["final_path_field"])
            row = {
                "policy": str(policy_cfg["key"]),
                "source_policy": source_policy,
                "split": source["split"],
                "snr_db": float(source["snr_db"]),
                "sample": source["sample"],
                "target_mode": source.get("target_mode", ""),
                "target_alpha": source.get("target_alpha", ""),
                "utility_target_alpha": source.get("utility_target_alpha", ""),
                "predicted_alpha": source.get("predicted_alpha", ""),
                "original": source["original"],
                "m0_reconstruction": source["m0_reconstruction"],
                "candidate": source[candidate_key],
                "final": source[final_key],
                "candidate_path_field": candidate_key,
                "final_path_field": final_key,
                "accepted": parse_bool(source["accepted"]),
                "source_m0_matches_original_top1": parse_bool(source["m0_matches_original_top1"]),
                "source_candidate_matches_original_top1": parse_bool(source["candidate_matches_original_top1"]),
                "source_candidate_matches_m0_top1": parse_bool(source["candidate_matches_m0_top1"]),
                "source_final_matches_original_top1": parse_bool(source["final_matches_original_top1"]),
                "source_accepted_repair": parse_bool(source["accepted_repair"]),
                "source_accepted_new_error": parse_bool(source["accepted_new_error"]),
                "source_missed_repair": parse_bool(source["missed_repair"]),
                "source_original_top1_index": source.get("original_top1_index", ""),
                "source_m0_top1_index": source.get("m0_top1_index", ""),
                "source_candidate_top1_index": source.get("alpha_top1_index", source.get("full_top1_index", "")),
            }
            output.append(row)
    if not output:
        raise RuntimeError("No policy rows matched the configured source policies.")
    return sorted(output, key=lambda row: (str(row["policy"]), str(row["split"]), float(row["snr_db"]), str(row["sample"])))


def validate_policy_images(rows: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "candidate", "final"]:
            path = resolve_project_path(row[key])
            if not path.exists():
                missing.append(project_relative(path))
    return sorted(set(missing))


def stack_paths(paths: list[str]) -> torch.Tensor:
    return torch.stack([load_rgb_tensor(resolve_project_path(path)) for path in paths])


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | None]:
    metric_values: dict[str, list[float]] = {"mse": [], "psnr_db": [], "ssim": [], "ms_ssim": []}
    lpips_values: list[float] = []
    with torch.no_grad():
        for start in range(0, reference.shape[0], batch_size):
            ref = reference[start : start + batch_size].to(device)
            cand = candidate[start : start + batch_size].to(device)
            mse_values = F.mse_loss(cand, ref, reduction="none").flatten(start_dim=1).mean(dim=1)
            metric_values["mse"].extend(mse_values.detach().cpu().tolist())
            metric_values["psnr_db"].extend(psnr_per_sample(cand, ref).detach().cpu().tolist())
            metric_values["ssim"].extend(ssim_per_sample(cand, ref).detach().cpu().tolist())
            metric_values["ms_ssim"].extend(ms_ssim_per_sample(cand, ref).detach().cpu().tolist())
            if lpips_model is not None:
                values = lpips_model(cand * 2.0 - 1.0, ref * 2.0 - 1.0)
                lpips_values.extend(values.flatten().detach().cpu().tolist())
    return {
        "mse": mean(metric_values["mse"]),
        "psnr_db": mean(metric_values["psnr_db"]),
        "ssim": mean(metric_values["ssim"]),
        "ms_ssim": mean(metric_values["ms_ssim"]),
        "lpips": mean(lpips_values) if lpips_model is not None else None,
    }


def attach_delta(summary: dict[str, Any], prefix: str, metrics: dict[str, float | None], m0: dict[str, float | None]) -> None:
    for key, value in metrics.items():
        summary[f"{prefix}_{key}"] = value
    summary[f"delta_{prefix}_psnr_vs_m0_db"] = (
        None if metrics["psnr_db"] is None or m0["psnr_db"] is None else metrics["psnr_db"] - m0["psnr_db"]
    )
    summary[f"delta_{prefix}_lpips_vs_m0"] = (
        None if metrics["lpips"] is None or m0["lpips"] is None else metrics["lpips"] - m0["lpips"]
    )
    summary[f"delta_{prefix}_ms_ssim_vs_m0"] = (
        None if metrics["ms_ssim"] is None or m0["ms_ssim"] is None else metrics["ms_ssim"] - m0["ms_ssim"]
    )


def summarize_source_semantics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_alexnet_m0_failure_rate": 1.0 - rate([bool(row["source_m0_matches_original_top1"]) for row in rows]),
        "source_alexnet_candidate_failure_rate": 1.0
        - rate([bool(row["source_candidate_matches_original_top1"]) for row in rows]),
        "source_alexnet_final_failure_rate": 1.0
        - rate([bool(row["source_final_matches_original_top1"]) for row in rows]),
        "source_alexnet_delta_final_failure_vs_m0": (
            1.0 - rate([bool(row["source_final_matches_original_top1"]) for row in rows])
        )
        - (1.0 - rate([bool(row["source_m0_matches_original_top1"]) for row in rows])),
        "accept_rate": rate([bool(row["accepted"]) for row in rows]),
        "source_alexnet_repair_count": sum(bool(row["source_accepted_repair"]) for row in rows),
        "source_alexnet_accepted_new_error_count": sum(bool(row["source_accepted_new_error"]) for row in rows),
        "source_alexnet_missed_repair_count": sum(bool(row["source_missed_repair"]) for row in rows),
    }


def make_quality_summary(
    rows: list[dict[str, Any]],
    lpips_model,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    group_keys: list[tuple[str, str, str]] = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if str(row["policy"]) == policy]
        group_keys.append((policy, "all", "all"))
        for split in sorted({str(row["split"]) for row in policy_rows}):
            split_rows = [row for row in policy_rows if str(row["split"]) == split]
            group_keys.append((policy, split, "all"))
            for snr in sorted({float(row["snr_db"]) for row in split_rows}):
                group_keys.append((policy, split, snr_name(snr)))

    for policy, split, snr in group_keys:
        subset = [row for row in rows if str(row["policy"]) == policy]
        if split != "all":
            subset = [row for row in subset if str(row["split"]) == split]
        if snr != "all":
            subset = [row for row in subset if snr_name(float(row["snr_db"])) == snr]
        references = stack_paths([str(row["original"]) for row in subset])
        m0_tensor = stack_paths([str(row["m0_reconstruction"]) for row in subset])
        candidate_tensor = stack_paths([str(row["candidate"]) for row in subset])
        final_tensor = stack_paths([str(row["final"]) for row in subset])
        m0_metrics = compute_pair_metrics(references, m0_tensor, lpips_model, device, batch_size)
        candidate_metrics = compute_pair_metrics(references, candidate_tensor, lpips_model, device, batch_size)
        final_metrics = compute_pair_metrics(references, final_tensor, lpips_model, device, batch_size)
        summary: dict[str, Any] = {
            "level": "policy_split_snr" if snr != "all" else ("policy_split" if split != "all" else "policy"),
            "policy": policy,
            "split": split,
            "snr_db": snr,
            "num_images": len(subset),
        }
        attach_delta(summary, "m0", m0_metrics, m0_metrics)
        attach_delta(summary, "candidate", candidate_metrics, m0_metrics)
        attach_delta(summary, "final", final_metrics, m0_metrics)
        summary.update(summarize_source_semantics(subset))
        summary_rows.append(summary)
    return summary_rows


def try_load_lpips(device: torch.device, cache_dir: Path):
    try:
        os.environ.setdefault("TORCH_HOME", str(cache_dir))
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def load_classifier(model_cfg: dict[str, Any], config: dict[str, Any], device: torch.device, allow_download: bool):
    weights_file = resolve_project_path(model_cfg["weights_file"])
    if (not weights_file.exists() or weights_file.stat().st_size < 1024 * 1024) and not allow_download:
        raise RuntimeError(f"Classifier weights missing from local cache: {weights_file}")
    cache_dir = resolve_project_path(config["classifiers"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))

    import torchvision.models as models

    weights_enum = getattr(models, str(model_cfg["weights_enum"]))
    weights = getattr(weights_enum, str(model_cfg["weights"]))
    builder = getattr(models, str(model_cfg["model_name"]))
    model = builder(weights=weights).to(device)
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
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = torch.topk(probs, k=topk, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for path, row_values, row_indices in zip(batch_paths, values.cpu(), indices.cpu()):
            predictions[project_relative(path)] = {
                "top_indices": [int(item) for item in row_indices.tolist()],
                "top_probs": [float(item) for item in row_values.tolist()],
            }
    return predictions, elapsed


def unique_classifier_paths(rows: list[dict[str, Any]]) -> list[Path]:
    seen: set[str] = set()
    output: list[Path] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "candidate"]:
            rel = str(row[key])
            if rel not in seen:
                seen.add(rel)
                output.append(resolve_project_path(rel))
    return output


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def eval_model_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    categories: list[str],
    model_key: str,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        original_pred = predictions[project_relative(resolve_project_path(row["original"]))]
        m0_pred = predictions[project_relative(resolve_project_path(row["m0_reconstruction"]))]
        candidate_pred = predictions[project_relative(resolve_project_path(row["candidate"]))]
        original_top1 = int(original_pred["top_indices"][0])
        m0_top1 = int(m0_pred["top_indices"][0])
        candidate_top1 = int(candidate_pred["top_indices"][0])
        accepted = bool(row["accepted"])
        final_top1 = candidate_top1 if accepted else m0_top1
        m0_ok = m0_top1 == original_top1
        candidate_ok = candidate_top1 == original_top1
        final_ok = final_top1 == original_top1
        out_rows.append(
            {
                "classifier": model_key,
                "policy": row["policy"],
                "split": row["split"],
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "accepted": accepted,
                "predicted_alpha": row.get("predicted_alpha", ""),
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "candidate": row["candidate"],
                "final": row["final"],
                "original_top1_index": original_top1,
                "original_top1_label": label_for(categories, original_top1),
                "original_top1_prob": float(original_pred["top_probs"][0]),
                "m0_top1_index": m0_top1,
                "m0_top1_label": label_for(categories, m0_top1),
                "m0_top1_prob": float(m0_pred["top_probs"][0]),
                "candidate_top1_index": candidate_top1,
                "candidate_top1_label": label_for(categories, candidate_top1),
                "candidate_top1_prob": float(candidate_pred["top_probs"][0]),
                "final_top1_index": final_top1,
                "final_top1_label": label_for(categories, final_top1),
                "m0_matches_original_top1": m0_ok,
                "candidate_matches_original_top1": candidate_ok,
                "candidate_matches_m0_top1": candidate_top1 == m0_top1,
                "final_matches_original_top1": final_ok,
                "accepted_repair": accepted and (not m0_ok) and candidate_ok,
                "accepted_new_error": accepted and m0_ok and (not candidate_ok),
                "missed_repair": (not accepted) and (not m0_ok) and candidate_ok,
                "protective_reject": (not accepted) and m0_ok and (not candidate_ok),
            }
        )
    return out_rows


def summarize_model_rows(rows: list[dict[str, Any]], level: str, classifier: str, policy: str, split: str, snr: str) -> dict[str, Any]:
    m0_failure = 1.0 - rate([bool(row["m0_matches_original_top1"]) for row in rows])
    candidate_failure = 1.0 - rate([bool(row["candidate_matches_original_top1"]) for row in rows])
    final_failure = 1.0 - rate([bool(row["final_matches_original_top1"]) for row in rows])
    return {
        "level": level,
        "classifier": classifier,
        "policy": policy,
        "split": split,
        "snr_db": snr,
        "num_images": len(rows),
        "accept_rate": rate([bool(row["accepted"]) for row in rows]),
        "m0_failure_rate": m0_failure,
        "candidate_failure_rate": candidate_failure,
        "final_failure_rate": final_failure,
        "delta_final_failure_vs_m0": final_failure - m0_failure,
        "accepted_repair_count": sum(bool(row["accepted_repair"]) for row in rows),
        "accepted_new_error_count": sum(bool(row["accepted_new_error"]) for row in rows),
        "missed_repair_count": sum(bool(row["missed_repair"]) for row in rows),
        "protective_reject_count": sum(bool(row["protective_reject"]) for row in rows),
    }


def make_model_summary(per_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for classifier in sorted({str(row["classifier"]) for row in per_model_rows}):
        cls_rows = [row for row in per_model_rows if str(row["classifier"]) == classifier]
        for policy in sorted({str(row["policy"]) for row in cls_rows}):
            policy_rows = [row for row in cls_rows if str(row["policy"]) == policy]
            output.append(summarize_model_rows(policy_rows, "classifier_policy", classifier, policy, "all", "all"))
            for split in sorted({str(row["split"]) for row in policy_rows}):
                split_rows = [row for row in policy_rows if str(row["split"]) == split]
                output.append(summarize_model_rows(split_rows, "classifier_policy_split", classifier, policy, split, "all"))
                for snr in sorted({float(row["snr_db"]) for row in split_rows}):
                    subset = [row for row in split_rows if float(row["snr_db"]) == snr]
                    output.append(
                        summarize_model_rows(
                            subset,
                            "classifier_policy_split_snr",
                            classifier,
                            policy,
                            split,
                            snr_name(snr),
                        )
                    )
    return output


def make_vote_rows(source_rows: list[dict[str, Any]], per_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_model_rows:
        by_key[(str(row["policy"]), str(row["split"]), float(row["snr_db"]), str(row["sample"]))].append(row)
    output: list[dict[str, Any]] = []
    for row in source_rows:
        model_rows = sorted(
            by_key[(str(row["policy"]), str(row["split"]), float(row["snr_db"]), str(row["sample"]))],
            key=lambda item: str(item["classifier"]),
        )
        classifiers = [str(item["classifier"]) for item in model_rows]
        new_error_models = [str(item["classifier"]) for item in model_rows if bool(item["accepted_new_error"])]
        repair_models = [str(item["classifier"]) for item in model_rows if bool(item["accepted_repair"])]
        missed_repair_models = [str(item["classifier"]) for item in model_rows if bool(item["missed_repair"])]
        protective_models = [str(item["classifier"]) for item in model_rows if bool(item["protective_reject"])]
        output.append(
            {
                "policy": row["policy"],
                "split": row["split"],
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "accepted": bool(row["accepted"]),
                "predicted_alpha": row.get("predicted_alpha", ""),
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "candidate": row["candidate"],
                "final": row["final"],
                "classifier_count": len(model_rows),
                "classifiers": "|".join(classifiers),
                "accepted_new_error_vote_count": len(new_error_models),
                "accepted_new_error_models": "|".join(new_error_models),
                "accepted_repair_vote_count": len(repair_models),
                "accepted_repair_models": "|".join(repair_models),
                "missed_repair_vote_count": len(missed_repair_models),
                "missed_repair_models": "|".join(missed_repair_models),
                "protective_reject_vote_count": len(protective_models),
                "protective_reject_models": "|".join(protective_models),
            }
        )
    return output


def summarize_vote_rows(rows: list[dict[str, Any]], level: str, policy: str, split: str, snr: str) -> dict[str, Any]:
    classifier_count = int(rows[0]["classifier_count"]) if rows else 0
    return {
        "level": level,
        "policy": policy,
        "split": split,
        "snr_db": snr,
        "num_images": len(rows),
        "classifier_count": classifier_count,
        "accept_count": sum(bool(row["accepted"]) for row in rows),
        "any_classifier_new_error_count": sum(int(row["accepted_new_error_vote_count"]) >= 1 for row in rows),
        "majority_classifier_new_error_count": sum(
            int(row["accepted_new_error_vote_count"]) > classifier_count / 2 for row in rows
        ),
        "all_classifier_new_error_count": sum(int(row["accepted_new_error_vote_count"]) == classifier_count for row in rows),
        "any_classifier_repair_count": sum(int(row["accepted_repair_vote_count"]) >= 1 for row in rows),
        "majority_classifier_repair_count": sum(
            int(row["accepted_repair_vote_count"]) > classifier_count / 2 for row in rows
        ),
        "all_classifier_repair_count": sum(int(row["accepted_repair_vote_count"]) == classifier_count for row in rows),
        "any_classifier_missed_repair_count": sum(int(row["missed_repair_vote_count"]) >= 1 for row in rows),
        "any_classifier_protective_reject_count": sum(int(row["protective_reject_vote_count"]) >= 1 for row in rows),
    }


def make_vote_summary(vote_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in sorted({str(row["policy"]) for row in vote_rows}):
        policy_rows = [row for row in vote_rows if str(row["policy"]) == policy]
        output.append(summarize_vote_rows(policy_rows, "policy", policy, "all", "all"))
        for split in sorted({str(row["split"]) for row in policy_rows}):
            split_rows = [row for row in policy_rows if str(row["split"]) == split]
            output.append(summarize_vote_rows(split_rows, "policy_split", policy, split, "all"))
            for snr in sorted({float(row["snr_db"]) for row in split_rows}):
                subset = [row for row in split_rows if float(row["snr_db"]) == snr]
                output.append(summarize_vote_rows(subset, "policy_split_snr", policy, split, snr_name(snr)))
    return output


def make_grid(rows: list[dict[str, Any]], output_path: Path, count: int) -> None:
    if not rows:
        return
    rows = rows[:count]
    tile = 160
    label_height = 54
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (tile + label_height)
        images = [
            ("original", resolve_project_path(row["original"])),
            ("m0", resolve_project_path(row["m0_reconstruction"])),
            ("candidate", resolve_project_path(row["candidate"])),
            ("final", resolve_project_path(row["final"])),
        ]
        for col, (label, path) in enumerate(images):
            x = col * tile
            canvas.paste(load_rgb_pil(path).resize((tile, tile), Image.Resampling.BICUBIC), (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        detail = (
            f"{row['policy']} {row['split']} {row['sample']} {snr_name(float(row['snr_db']))} "
            f"accept={bool_text(bool(row['accepted']))}"
        )
        draw.text((4, y + 18), detail[:118], fill=(0, 0, 0), font=font)
        votes = (
            f"new={row['accepted_new_error_vote_count']}:{row['accepted_new_error_models']} "
            f"repair={row['accepted_repair_vote_count']}:{row['accepted_repair_models']}"
        )
        draw.text((4, y + 34), votes[:118], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_galleries(vote_rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    gallery_dir = output_dir / "galleries"
    count = int(config["evaluation"]["gallery_rows"])
    manifest: dict[str, str] = {}
    groups = {
        "continuous_any_classifier_new_errors": [
            row
            for row in vote_rows
            if row["policy"] == "continuous_alpha_top1_fallback" and int(row["accepted_new_error_vote_count"]) >= 1
        ],
        "continuous_majority_classifier_new_errors": [
            row
            for row in vote_rows
            if row["policy"] == "continuous_alpha_top1_fallback"
            and int(row["accepted_new_error_vote_count"]) > int(row["classifier_count"]) / 2
        ],
        "continuous_any_classifier_repairs": [
            row
            for row in vote_rows
            if row["policy"] == "continuous_alpha_top1_fallback" and int(row["accepted_repair_vote_count"]) >= 1
        ],
        "full_strength_any_classifier_new_errors": [
            row
            for row in vote_rows
            if row["policy"] == "full_strength_top1_fallback" and int(row["accepted_new_error_vote_count"]) >= 1
        ],
    }
    for name, rows in groups.items():
        rows = sorted(
            rows,
            key=lambda row: (
                -int(row["accepted_new_error_vote_count"]),
                -int(row["accepted_repair_vote_count"]),
                str(row["split"]),
                float(row["snr_db"]),
                str(row["sample"]),
            ),
        )
        if not rows:
            continue
        path = gallery_dir / f"{name}.png"
        make_grid(rows, path, count)
        manifest[name] = project_relative(path)
    return manifest


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    labels = [label for _key, label in columns]
    lines.append("| " + " | ".join(labels) + " |")
    lines.append("|" + "|".join(["---" for _ in labels]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key, _label in columns) + " |")
    return lines


def make_report(
    quality_summary: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    vote_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    quality_split = [row for row in quality_summary if row["level"] == "policy_split"]
    vote_split = [row for row in vote_summary if row["level"] == "policy_split"]
    model_split = [row for row in model_summary if row["level"] == "classifier_policy_split"]
    continuous_all = next(
        row for row in quality_summary if row["level"] == "policy" and row["policy"] == "continuous_alpha_top1_fallback"
    )
    full_all = next(row for row in quality_summary if row["level"] == "policy" and row["policy"] == "full_strength_top1_fallback")
    lines = [
        "# Continuous-Alpha Tail Refiner Audit",
        "",
        "This derived audit reads the existing continuous-alpha tail-only residual refiner outputs. It does not train a model, run diffusion, or tune a policy.",
        "",
        "## Bottom Line",
        "",
        f"- Continuous-alpha top-1 fallback: PSNR delta `{signed(continuous_all['delta_final_psnr_vs_m0_db'])}` dB, LPIPS delta `{signed(continuous_all['delta_final_lpips_vs_m0'])}`, source AlexNet new error `{continuous_all['source_alexnet_accepted_new_error_count']}`.",
        f"- Full-strength top-1 fallback from the same checkpoint: PSNR delta `{signed(full_all['delta_final_psnr_vs_m0_db'])}` dB, LPIPS delta `{signed(full_all['delta_final_lpips_vs_m0'])}`, source AlexNet new error `{full_all['source_alexnet_accepted_new_error_count']}`.",
        "- The classifier ensemble is an offline robustness audit only; COCO pseudo labels remain auxiliary, not final clean-correct supervision.",
        "",
        "## Quality And Source-AlexNet Summary",
        "",
    ]
    lines += markdown_table(
        quality_split,
        [
            ("policy", "Policy"),
            ("split", "Split"),
            ("delta_final_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_final_lpips_vs_m0", "Delta LPIPS"),
            ("source_alexnet_final_failure_rate", "AlexNet Failure"),
            ("source_alexnet_accepted_new_error_count", "AlexNet New Error"),
            ("source_alexnet_missed_repair_count", "Missed Repair"),
            ("accept_rate", "Accept"),
        ],
    )
    lines.extend(["", "## Ensemble Vote Summary", ""])
    lines += markdown_table(
        vote_split,
        [
            ("policy", "Policy"),
            ("split", "Split"),
            ("num_images", "Images"),
            ("any_classifier_new_error_count", "Any New Error"),
            ("majority_classifier_new_error_count", "Majority New Error"),
            ("any_classifier_repair_count", "Any Repair"),
            ("majority_classifier_repair_count", "Majority Repair"),
            ("any_classifier_missed_repair_count", "Any Missed Repair"),
        ],
    )
    lines.extend(["", "## Per-Classifier Split Summary", ""])
    lines += markdown_table(
        model_split,
        [
            ("classifier", "Classifier"),
            ("policy", "Policy"),
            ("split", "Split"),
            ("final_failure_rate", "Final Failure"),
            ("delta_final_failure_vs_m0", "Delta Failure"),
            ("accepted_new_error_count", "New Error"),
            ("accepted_repair_count", "Repair"),
            ("missed_repair_count", "Missed Repair"),
        ],
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-sample policy CSV: `{metadata['per_sample_csv']}`",
            f"- Quality summary CSV: `{metadata['quality_summary_csv']}`",
            f"- Per-model CSV: `{metadata['per_model_csv']}`",
            f"- Model summary CSV: `{metadata['model_summary_csv']}`",
            f"- Vote CSV: `{metadata['vote_csv']}`",
            f"- Vote summary CSV: `{metadata['vote_summary_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Galleries: `{metadata['gallery_dir']}`",
            "",
            "## Caveats",
            "",
            "- The receiver-side decision is still the original AlexNet top-1 fallback saved by the source run.",
            "- ResNet18 and MobileNetV3-Small are only used after the fact to probe cross-model semantic risk.",
            "- This audit does not replace a future labeled clean-correct evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = validate_inputs(config)
    source_rows = read_csv(resolve_project_path(config["inputs"]["source_per_sample_csv"]))
    policy_rows = normalize_policy_rows(source_rows, config)
    missing_images = validate_policy_images(policy_rows)
    if missing_images:
        raise FileNotFoundError("Missing policy images:\n" + "\n".join(missing_images[:30]))

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_run_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "num_policy_rows": len(policy_rows),
        "rows_by_policy": {
            policy: sum(1 for row in policy_rows if row["policy"] == policy)
            for policy in sorted({str(row["policy"]) for row in policy_rows})
        },
        "rows_by_split": {
            split: sum(1 for row in policy_rows if row["split"] == split)
            for split in sorted({str(row["split"]) for row in policy_rows})
        },
        "unique_images_to_classify": len(unique_classifier_paths(policy_rows)),
        "classifier_status": classifier_status(config),
        "device": str(resolve_device(args.device)),
        "lpips_requested": not args.skip_lpips,
        "allow_download": bool(args.allow_download),
        "proxy_environment_present": proxy_environment_present(),
        "manifest": manifest,
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    lpips_model = None
    lpips_error = "Skipped by --skip-lpips" if args.skip_lpips else None
    if not args.skip_lpips:
        lpips_model, lpips_error = try_load_lpips(device, resolve_project_path(config["classifiers"]["cache_dir"]))
    quality_summary = make_quality_summary(
        policy_rows,
        lpips_model,
        device,
        int(config["evaluation"]["image_batch_size"]),
    )

    all_paths = unique_classifier_paths(policy_rows)
    per_model_rows: list[dict[str, Any]] = []
    classifier_runtime: dict[str, Any] = {}
    loaded_classifiers: list[str] = []
    for model_cfg in config["classifiers"]["models"]:
        key = str(model_cfg["key"])
        model, preprocess, categories = load_classifier(model_cfg, config, device, allow_download=args.allow_download)
        predictions, elapsed = classify_paths(
            model,
            preprocess,
            all_paths,
            int(config["classifiers"]["batch_size"]),
            int(config["classifiers"]["topk"]),
            device,
        )
        per_model_rows.extend(eval_model_rows(policy_rows, predictions, categories, key))
        classifier_runtime[key] = {
            "num_images": len(all_paths),
            "inference_seconds": elapsed,
            "device": str(device),
            "weights_file": project_relative(resolve_project_path(model_cfg["weights_file"])),
        }
        loaded_classifiers.append(key)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_summary = make_model_summary(per_model_rows)
    vote_rows = make_vote_rows(policy_rows, per_model_rows)
    vote_summary = make_vote_summary(vote_rows)
    galleries = write_galleries(vote_rows, config, output_dir)

    per_sample_csv = output_dir / "per_sample.csv"
    quality_summary_csv = output_dir / "quality_summary.csv"
    per_model_csv = output_dir / "per_model_per_sample.csv"
    model_summary_csv = output_dir / "model_summary.csv"
    vote_csv = output_dir / "per_sample_votes.csv"
    vote_summary_csv = output_dir / "vote_summary.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(per_sample_csv, policy_rows)
    write_csv(quality_summary_csv, quality_summary)
    write_csv(per_model_csv, per_model_rows)
    write_csv(model_summary_csv, model_summary)
    write_csv(vote_csv, vote_rows)
    write_csv(vote_summary_csv, vote_summary)

    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "source_inputs": manifest,
        "per_sample_csv": project_relative(per_sample_csv),
        "quality_summary_csv": project_relative(quality_summary_csv),
        "per_model_csv": project_relative(per_model_csv),
        "model_summary_csv": project_relative(model_summary_csv),
        "vote_csv": project_relative(vote_csv),
        "vote_summary_csv": project_relative(vote_summary_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "gallery_dir": project_relative(output_dir / "galleries"),
        "galleries": galleries,
        "loaded_classifiers": loaded_classifiers,
        "classifier_runtime": classifier_runtime,
        "lpips_error": lpips_error,
        "dry_run_payload": dry_run_payload,
        "run_command": " ".join(sys.argv),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": (
            "No download is required when local classifier and LPIPS/AlexNet weights are present under the project cache. "
            "Use --allow-download only deliberately with cleared proxy variables."
        ),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(quality_summary, model_summary, vote_summary, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "report_md": project_relative(report_md),
                "policy_rows": len(policy_rows),
                "per_model_rows": len(per_model_rows),
                "vote_rows": len(vote_rows),
                "lpips_error": lpips_error,
                "loaded_classifiers": loaded_classifiers,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
