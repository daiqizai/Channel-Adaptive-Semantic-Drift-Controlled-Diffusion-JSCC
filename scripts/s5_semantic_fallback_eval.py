from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

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
        description="Evaluate a receiver-side semantic fallback rule over existing M0/M1 outputs."
    )
    parser.add_argument("--config", default="configs/s5_semantic_fallback_m1_exp_s2_002.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--skip-lpips", action="store_true")
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


def parse_snrs(value: str | None, config: dict[str, Any]) -> list[float]:
    if value is None:
        return [float(item) for item in config["snrs"]]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_classifier_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (snr_name(float(row["snr_db"])), str(row["sample"]))
            if key in rows:
                raise RuntimeError(f"Duplicate classifier row for {key}")
            rows[key] = row
    return rows


def list_sample_names(config: dict[str, Any], snr: float, num_samples: int) -> list[str]:
    manifest_path = resolve_project_path(config["inputs"]["source_manifest"])
    manifest = load_json(manifest_path)
    names = manifest.get(snr_name(snr))
    if names is None:
        raise KeyError(f"{snr_name(snr)} not found in {manifest_path}")
    if len(names) < num_samples:
        raise RuntimeError(f"Need {num_samples} samples for {snr_name(snr)}, found {len(names)}")
    return list(names[:num_samples])


def validate_inputs(
    config: dict[str, Any],
    snrs: list[float],
    num_samples: int,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], dict[str, Any]]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    m1_output_dir = resolve_project_path(config["inputs"]["m1_output_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    classifier_per_sample = resolve_project_path(config["inputs"]["classifier_per_sample"])
    required = [
        original_dir,
        m0_export_dir,
        m1_output_dir,
        resolve_project_path(config["inputs"]["m1_metrics"]),
        classifier_per_sample,
        resolve_project_path(config["inputs"]["classifier_metrics"]),
        resolve_project_path(config["inputs"]["source_manifest"]),
        checkpoint,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    classifier_rows = load_classifier_rows(classifier_per_sample)
    names_by_snr: dict[str, list[str]] = {}
    for snr in snrs:
        snr_key = snr_name(snr)
        names = list_sample_names(config, snr, num_samples)
        m0_dir = m0_export_dir / "exports" / snr_key / "reconstruction"
        m1_dir = m1_output_dir / "exports" / snr_key / "refined"
        for name in names:
            for path in [original_dir / name, m0_dir / name, m1_dir / name]:
                if not path.exists():
                    raise FileNotFoundError(f"Matched sample not found: {path}")
            if (snr_key, name) not in classifier_rows:
                raise KeyError(f"Classifier row missing for {snr_key}/{name}")
        names_by_snr[snr_key] = names
    return names_by_snr, classifier_rows


def detector_accepts(row: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    detector = str(config["control"]["detector"])
    if detector == "top1_agreement_with_m0":
        return bool_value(row["m1_matches_m0_top1"]), "m1_top1_equals_m0_top1"
    raise ValueError(f"Unsupported detector: {detector}")


def try_load_lpips(device: torch.device):
    cache_root = resolve_project_path("outputs/cache")
    weights_file = cache_root / "torch" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    if not weights_file.is_file():
        return None, f"Skipped because local AlexNet weights are missing: {project_relative(weights_file)}"
    try:
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001 - optional metric should not abort this evaluation.
        return None, f"{type(exc).__name__}: {exc}"


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, float | None]:
    reference = reference.to(device)
    candidate = candidate.to(device)
    metrics: dict[str, float | None] = {}
    with torch.no_grad():
        mse_values = F.mse_loss(candidate, reference, reduction="none").flatten(start_dim=1).mean(dim=1)
        metrics["mse"] = mean(mse_values.detach().cpu().tolist())
        metrics["psnr_db"] = mean(psnr_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ssim"] = mean(ssim_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ms_ssim"] = mean(ms_ssim_per_sample(candidate, reference).detach().cpu().tolist())
        if lpips_model is not None:
            lpips_values = lpips_model(candidate * 2.0 - 1.0, reference * 2.0 - 1.0)
            metrics["lpips"] = mean(lpips_values.flatten().detach().cpu().tolist())
        else:
            metrics["lpips"] = None
    return metrics


def semantic_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_images": 0,
            "accept_rate": 0.0,
            "reject_rate": 0.0,
            "m0_final_failure": 0.0,
            "m1_final_failure": 0.0,
            "m3_final_failure": 0.0,
            "m3_prediction_consistency": 0.0,
            "m1_refinement_drift": 0.0,
            "m3_refinement_drift": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "fallback_helped_rate": 0.0,
            "fallback_hurt_rate": 0.0,
        }
    accepted = [bool(row["detector_accept_refined"]) for row in rows]
    m0_match_origin = [bool(row["m0_matches_original_top1"]) for row in rows]
    m1_match_origin = [bool(row["m1_matches_original_top1"]) for row in rows]
    m1_match_m0 = [bool(row["m1_matches_m0_top1"]) for row in rows]
    m3_match_origin = [bool(row["m3_matches_original_top1"]) for row in rows]
    m3_match_m0 = [bool(row["m3_matches_m0_top1"]) for row in rows]
    false_accept = [bool(row["detector_accept_refined"]) and not bool(row["m1_matches_original_top1"]) for row in rows]
    false_reject = [not bool(row["detector_accept_refined"]) and bool(row["m1_matches_original_top1"]) for row in rows]
    fallback_helped = [
        not bool(row["detector_accept_refined"])
        and bool(row["m0_matches_original_top1"])
        and not bool(row["m1_matches_original_top1"])
        for row in rows
    ]
    fallback_hurt = [
        not bool(row["detector_accept_refined"])
        and not bool(row["m0_matches_original_top1"])
        and bool(row["m1_matches_original_top1"])
        for row in rows
    ]
    m0_failure = 1.0 - rate(m0_match_origin)
    m1_failure = 1.0 - rate(m1_match_origin)
    m3_failure = 1.0 - rate(m3_match_origin)
    return {
        "num_images": len(rows),
        "accept_rate": rate(accepted),
        "reject_rate": 1.0 - rate(accepted),
        "m0_final_failure": m0_failure,
        "m1_final_failure": m1_failure,
        "m3_final_failure": m3_failure,
        "m3_prediction_consistency": 1.0 - m3_failure,
        "m1_refinement_drift": 1.0 - rate(m1_match_m0),
        "m3_refinement_drift": 1.0 - rate(m3_match_m0),
        "m3_minus_m1_final_failure": m3_failure - m1_failure,
        "m3_minus_m0_final_failure": m3_failure - m0_failure,
        "false_accept_rate": rate(false_accept),
        "false_reject_rate": rate(false_reject),
        "fallback_helped_rate": rate(fallback_helped),
        "fallback_hurt_rate": rate(fallback_hurt),
        "false_accept_count": int(sum(false_accept)),
        "false_reject_count": int(sum(false_reject)),
        "fallback_helped_count": int(sum(fallback_helped)),
        "fallback_hurt_count": int(sum(fallback_hurt)),
    }


def subset_summaries(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, dict[str, float | int]]:
    summaries = {"all": semantic_summary(rows)}
    for threshold in thresholds:
        key = f"original_conf_ge_{str(threshold).replace('.', 'p')}"
        subset = [row for row in rows if float(row["original_top1_prob"]) >= threshold]
        summaries[key] = semantic_summary(subset)
    return summaries


def copy_final_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def evaluate_snr(
    snr: float,
    names: list[str],
    classifier_rows: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    snr_key = snr_name(snr)
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_key / "reconstruction"
    m1_dir = resolve_project_path(config["inputs"]["m1_output_dir"]) / "exports" / snr_key / "refined"
    final_dir = output_dir / "exports" / snr_key / "final"
    final_dir.mkdir(parents=True, exist_ok=False)

    references: list[torch.Tensor] = []
    m0_tensors: list[torch.Tensor] = []
    m1_tensors: list[torch.Tensor] = []
    m3_tensors: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []

    for name in names:
        row = classifier_rows[(snr_key, name)]
        accept, reason = detector_accepts(row, config)
        source_path = m1_dir / name if accept else m0_dir / name
        final_path = final_dir / name
        copy_final_image(source_path, final_path)

        m3_prefix = "m1" if accept else "m0"
        m3_matches_original = bool_value(row[f"{m3_prefix}_matches_original_top1"])
        m3_matches_m0 = bool_value(row["m1_matches_m0_top1"]) if accept else True
        output_kind = "accepted_refined" if accept else "fallback_m0"

        sample_row = {
            "snr_db": float(snr),
            "sample": name,
            "detector": str(config["control"]["detector"]),
            "detector_reason": reason,
            "detector_accept_refined": accept,
            "m3_output_kind": output_kind,
            "original": project_relative(original_dir / name),
            "m0_reconstruction": project_relative(m0_dir / name),
            "m1_refined": project_relative(m1_dir / name),
            "m3_final": project_relative(final_path),
            "original_top1_index": int(row["original_top1_index"]),
            "original_top1_label": row["original_top1_label"],
            "original_top1_prob": float(row["original_top1_prob"]),
            "m0_top1_index": int(row["m0_top1_index"]),
            "m0_top1_label": row["m0_top1_label"],
            "m0_top1_prob": float(row["m0_top1_prob"]),
            "m1_top1_index": int(row["m1_top1_index"]),
            "m1_top1_label": row["m1_top1_label"],
            "m1_top1_prob": float(row["m1_top1_prob"]),
            "m3_top1_index": int(row[f"{m3_prefix}_top1_index"]),
            "m3_top1_label": row[f"{m3_prefix}_top1_label"],
            "m3_top1_prob": float(row[f"{m3_prefix}_top1_prob"]),
            "m0_matches_original_top1": bool_value(row["m0_matches_original_top1"]),
            "m1_matches_original_top1": bool_value(row["m1_matches_original_top1"]),
            "m1_matches_m0_top1": bool_value(row["m1_matches_m0_top1"]),
            "m3_matches_original_top1": m3_matches_original,
            "m3_matches_m0_top1": m3_matches_m0,
            "false_accept": accept and not bool_value(row["m1_matches_original_top1"]),
            "false_reject": (not accept) and bool_value(row["m1_matches_original_top1"]),
        }
        per_sample.append(sample_row)

        references.append(load_rgb_tensor(original_dir / name))
        m0_tensors.append(load_rgb_tensor(m0_dir / name))
        m1_tensors.append(load_rgb_tensor(m1_dir / name))
        m3_tensors.append(load_rgb_tensor(final_path))

    reference = torch.stack(references)
    m0 = torch.stack(m0_tensors)
    m1 = torch.stack(m1_tensors)
    m3 = torch.stack(m3_tensors)

    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_grid = sample_dir / f"{snr_key}_original_m0_m1_m3final.png"
    save_image(
        torch.cat([reference[:sample_count], m0[:sample_count], m1[:sample_count], m3[:sample_count]], dim=0),
        sample_grid,
        nrow=sample_count,
    )

    thresholds = [float(item) for item in config["evaluation"].get("pseudo_clean_conf_thresholds", [])]
    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "sample_names": names,
        "final_dir": project_relative(final_dir),
        "sample_grid": project_relative(sample_grid),
        "image_quality": {
            "m0_reconstruction_vs_original": compute_pair_metrics(reference, m0, lpips_model, device),
            "m1_refined_vs_original": compute_pair_metrics(reference, m1, lpips_model, device),
            "m3_final_vs_original": compute_pair_metrics(reference, m3, lpips_model, device),
        },
        "semantic_reliability": subset_summaries(per_sample, thresholds),
        "per_sample": per_sample,
    }


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def make_report(results: list[dict[str, Any]], sources: dict[str, str], config: dict[str, Any]) -> str:
    lines = [
        "# Semantic Fallback Pilot",
        "",
        "This is a derived S5 pilot over existing M0/M1 images and frozen-classifier diagnostics. It does not run diffusion again.",
        "",
        "## Sources",
        "",
        f"- M1 image outputs: `{sources['m1_output_dir']}`",
        f"- Classifier per-sample CSV: `{sources['classifier_per_sample']}`",
        f"- M0 export: `{sources['m0_export_dir']}`",
        "",
        "## Control Rule",
        "",
        f"- Detector: `{config['control']['detector']}`",
        "- Accept M1 only when the frozen classifier top-1 label of M1 equals the top-1 label of M0.",
        "- Otherwise, fallback to M0 reconstruction.",
        "- The detector does not use the original image; original pseudo-labels are used only for offline evaluation.",
        "",
        "## Main Table",
        "",
        "| SNR(dB) | Accept | M0 PSNR | M1 PSNR | M3 PSNR | M0 failure | M1 failure | M3 failure | False accept | False reject |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        quality = result["image_quality"]
        semantic = result["semantic_reliability"]["all"]
        lines.append(
            "| {snr:g} | {accept} | {m0_psnr} | {m1_psnr} | {m3_psnr} | {m0_fail} | {m1_fail} | {m3_fail} | {fa} | {fr} |".format(
                snr=result["snr_db"],
                accept=fmt(float(semantic["accept_rate"])),
                m0_psnr=fmt(float(quality["m0_reconstruction_vs_original"]["psnr_db"])),
                m1_psnr=fmt(float(quality["m1_refined_vs_original"]["psnr_db"])),
                m3_psnr=fmt(float(quality["m3_final_vs_original"]["psnr_db"])),
                m0_fail=fmt(float(semantic["m0_final_failure"])),
                m1_fail=fmt(float(semantic["m1_final_failure"])),
                m3_fail=fmt(float(semantic["m3_final_failure"])),
                fa=fmt(float(semantic["false_accept_rate"])),
                fr=fmt(float(semantic["false_reject_rate"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The fallback clamps semantic failure close to M0 because most M1 outputs change the frozen classifier top-1 label.",
            "- This is not the final M3 result: it uses the already-negative fixed-strength M1 outputs and a pseudo-label classifier diagnostic.",
            "- The next S5 step should combine this failure handling with a conservative SNR-adaptive diffusion-strength grid.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(args.snrs, config)
    num_samples = int(args.num_samples or config["evaluation"]["num_samples"])
    config["evaluation"]["num_samples"] = num_samples
    names_by_snr, classifier_rows = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(json.dumps({"status": "ok", "snrs": snrs, "sample_names_by_snr": names_by_snr}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    lpips_model, lpips_error = (None, "Skipped by --skip-lpips") if args.skip_lpips else try_load_lpips(device)

    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", names_by_snr)

    results = []
    csv_rows: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            snr=snr,
            names=names_by_snr[snr_name(snr)],
            classifier_rows=classifier_rows,
            config=config,
            output_dir=output_dir,
            lpips_model=lpips_model,
            device=device,
        )
        results.append(result)
        csv_rows.extend(result["per_sample"])
        printable = {key: value for key, value in result.items() if key != "per_sample"}
        print(json.dumps(printable, indent=2))

    import importlib.metadata as md
    import torchvision

    sources = {
        "m0_export_dir": config["inputs"]["m0_export_dir"],
        "m1_output_dir": config["inputs"]["m1_output_dir"],
        "classifier_per_sample": config["inputs"]["classifier_per_sample"],
    }
    metadata = {
        "project_version": "N/A (not a project git repo)",
        "config": project_relative(config_path),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dataset": config["dataset"],
        "image_size": int(config["image_size"]),
        "channel": str(config["channel"]),
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": int(config["seed"]),
        "num_samples": num_samples,
        "inputs": config["inputs"],
        "control": config["control"],
        "lpips_error": lpips_error,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "note": (
            "This is a derived semantic fallback pilot. It uses original pseudo-labels only for evaluation, "
            "not for the detector decision. COCO GT classification labels are not used."
        ),
        "key_sources": [
            "scripts/s5_semantic_fallback_eval.py",
            "src/cadsd_jscc/metrics.py",
            "scripts/s4_classifier_consistency_eval.py",
        ],
    }
    payload = {"metadata": metadata, "results": results}
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    (output_dir / "REPORT.md").write_text(make_report(results, sources, config), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
