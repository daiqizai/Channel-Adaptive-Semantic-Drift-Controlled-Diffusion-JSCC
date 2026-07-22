from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import yaml
from PIL import Image, __version__ as PIL_VERSION
from torch.utils.data import DataLoader, Dataset
from torchvision import __version__ as TORCHVISION_VERSION
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/s6_imagenette_supervised_clean_eval.yaml"
SPLIT_NAMES = ("cls_train", "cls_cal", "policy_dev")
MANIFEST_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train randomly initialized Imagenette semantic classifiers using only the "
            "deterministic cls_train split; cls_cal is reserved for model selection and calibration."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=None,
        help="Classifier roles to train (space- or comma-separated), for example: G_gate T_cls.",
    )
    parser.add_argument(
        "--finalize-completed",
        nargs="+",
        default=None,
        metavar="ROLE",
        help=(
            "Finalize an interrupted role whose complete epoch history and best-in-progress "
            "checkpoint already exist. This recovery mode only loads cls_cal; it never resumes training."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Config must contain a YAML mapping: {path}")
    return payload


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key, "")) for key in fieldnames})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required only to match the publisher's integrity checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_sha256(config: dict[str, Any]) -> str:
    """Hash the preregistered method while allowing only final-lock metadata to be populated later."""
    protocol_config = {key: value for key, value in config.items() if key != "final_lock"}
    canonical = json.dumps(
        protocol_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(canonical)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "N/A"


def git_dirty_state() -> str:
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return "dirty" if status else "clean"


def version_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": TORCHVISION_VERSION,
        "pillow": PIL_VERSION,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def parse_roles(raw_roles: list[str] | None, configured: dict[str, Any]) -> list[str]:
    if raw_roles is None:
        roles = list(configured)
    else:
        roles = []
        for item in raw_roles:
            roles.extend(part.strip() for part in item.split(",") if part.strip())
    if not roles:
        raise ValueError("At least one classifier role must be selected")
    if len(set(roles)) != len(roles):
        raise ValueError(f"Duplicate --roles entries: {roles}")
    unknown = [role for role in roles if role not in configured]
    if unknown:
        raise KeyError(f"Unknown classifier roles {unknown}; configured roles are {list(configured)}")
    return roles


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "split", "training", "scratch_classifiers", "outputs"):
        if section not in config or not isinstance(config[section], dict):
            raise KeyError(f"Missing required config mapping: {section}")

    ratios = config["split"].get("ratios", {})
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"split.ratios must have exactly the keys {SPLIT_NAMES}; got {tuple(ratios)}")
    ratio_values = [float(ratios[name]) for name in SPLIT_NAMES]
    if any(value <= 0.0 for value in ratio_values) or not math.isclose(sum(ratio_values), 1.0, abs_tol=1e-9):
        raise ValueError(f"split.ratios must be positive and sum to one; got {ratios}")
    expected = [0.70, 0.10, 0.20]
    if any(not math.isclose(value, target, abs_tol=1e-12) for value, target in zip(ratio_values, expected)):
        raise ValueError(f"The preregistered split must remain 70/10/20; got {ratio_values}")

    training = config["training"]
    if str(training.get("optimizer", "")).lower() != "sgd":
        raise ValueError("training.optimizer must be sgd")
    if str(training.get("scheduler", "")).lower() != "cosine":
        raise ValueError("training.scheduler must be cosine")
    if str(training.get("checkpoint_selection", "cls_cal_macro_top1")) != "cls_cal_macro_top1":
        raise ValueError("Checkpoint selection must use cls_cal_macro_top1")
    if str(training.get("temperature_scaling_split", "cls_cal")) != "cls_cal":
        raise ValueError("Temperature scaling must use cls_cal")
    if int(training["epochs"]) <= 0 or int(training["batch_size"]) <= 0:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if int(training.get("warmup_epochs", 0)) < 0 or int(training.get("warmup_epochs", 0)) >= int(training["epochs"]):
        raise ValueError("training.warmup_epochs must be in [0, epochs)")

    if not config["scratch_classifiers"]:
        raise ValueError("scratch_classifiers cannot be empty")
    for role, model_cfg in config["scratch_classifiers"].items():
        if not isinstance(model_cfg, dict):
            raise TypeError(f"scratch_classifiers.{role} must be a mapping")
        if model_cfg.get("weights") is not None:
            raise ValueError(f"{role} must have weights: null (random initialization)")
        if model_cfg.get("pretrained", False) is not False:
            raise ValueError(f"{role} must not set pretrained=true")
        for key in ("architecture", "seed", "min_cal_macro_top1", "output_dir"):
            if key not in model_cfg:
                raise KeyError(f"scratch_classifiers.{role}.{key} is required")


def resolve_train_root(config: dict[str, Any]) -> Path:
    data = config["data"]
    if data.get("train_dir"):
        train_root = resolve_project_path(data["train_dir"])
    else:
        train_root = resolve_project_path(data["root"]) / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Official Imagenette train directory is missing: {train_root}")
    return train_root.resolve()


def validate_source_archive(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    archive = resolve_project_path(data["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"Official Imagenette archive is missing: {archive}")
    actual_size = archive.stat().st_size
    expected_size = int(data["archive_size_bytes"])
    if actual_size != expected_size:
        raise RuntimeError(f"Imagenette archive size mismatch: actual={actual_size}, expected={expected_size}")
    actual_md5 = md5_file(archive)
    expected_md5 = str(data["archive_md5"]).lower()
    if actual_md5 != expected_md5:
        raise RuntimeError(f"Imagenette archive MD5 mismatch: actual={actual_md5}, expected={expected_md5}")
    return {
        "path": project_relative(archive),
        "size_bytes": actual_size,
        "md5": actual_md5,
        "sha256": sha256_file(archive),
    }


def assert_outputs_outside_dataset(config: dict[str, Any], train_root: Path) -> None:
    dataset_root = resolve_project_path(config["data"]["root"]).resolve()
    output_paths = [resolve_project_path(config["outputs"]["split_manifest"])]
    output_paths.extend(resolve_project_path(item["output_dir"]) for item in config["scratch_classifiers"].values())
    for output_path in output_paths:
        resolved = output_path.resolve()
        if resolved == dataset_root or dataset_root in resolved.parents:
            raise ValueError(f"Output path must not be inside the official dataset: {resolved}")
        if resolved == train_root or train_root in resolved.parents:
            raise ValueError(f"Output path must not be inside official train: {resolved}")


def largest_remainder_counts(count: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: count * float(ratios[name]) for name in SPLIT_NAMES}
    allocated = {name: int(math.floor(raw[name])) for name in SPLIT_NAMES}
    remainder = count - sum(allocated.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (-(raw[name] - allocated[name]), SPLIT_NAMES.index(name)))
    for name in order[:remainder]:
        allocated[name] += 1
    if sum(allocated.values()) != count:
        raise AssertionError("Split allocation did not preserve the class count")
    return allocated


def build_split_manifest(config: dict[str, Any], train_root: Path) -> dict[str, Any]:
    # ImageFolder is used only on the official train directory. Official val is never instantiated or scanned.
    image_folder = ImageFolder(str(train_root))
    configured_classes = [str(item) for item in config["data"].get("classes", [])]
    if configured_classes and image_folder.classes != configured_classes:
        raise RuntimeError(
            "Imagenette class mapping differs from the preregistered WNIDs:\n"
            f"found={image_folder.classes}\nexpected={configured_classes}"
        )
    if len(image_folder.classes) != 10:
        raise RuntimeError(f"Expected official Imagenette's 10 WNID folders; found {len(image_folder.classes)}")

    split_seed = int(config["split"]["seed"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_paths: set[str] = set()
    for raw_path, class_idx in image_folder.samples:
        path = Path(raw_path).resolve()
        try:
            relative_path = path.relative_to(train_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"ImageFolder returned a sample outside official train: {path}") from exc
        parts = Path(relative_path).parts
        if len(parts) < 2:
            raise RuntimeError(f"Expected WNID/file relative path, got: {relative_path}")
        wnid = parts[0]
        if image_folder.class_to_idx.get(wnid) != int(class_idx):
            raise RuntimeError(f"Class index mismatch for {relative_path}")
        if relative_path in seen_paths:
            raise RuntimeError(f"Duplicate ImageFolder path: {relative_path}")
        seen_paths.add(relative_path)
        sample_id = relative_path
        split_score_payload = f"{split_seed}:{sample_id}".encode("utf-8")
        grouped[wnid].append(
            {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "wnid": wnid,
                "class_idx": int(class_idx),
                "path_sha256": sha256_bytes(split_score_payload),
                "content_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    ratios = {name: float(config["split"]["ratios"][name]) for name in SPLIT_NAMES}
    samples: list[dict[str, Any]] = []
    per_class_counts: dict[str, dict[str, int]] = {}
    for wnid in image_folder.classes:
        class_rows = sorted(grouped[wnid], key=lambda row: (row["path_sha256"], row["sample_id"]))
        allocation = largest_remainder_counts(len(class_rows), ratios)
        per_class_counts[wnid] = dict(allocation)
        cursor = 0
        for split_name in SPLIT_NAMES:
            stop = cursor + allocation[split_name]
            for row in class_rows[cursor:stop]:
                samples.append({**row, "split": split_name})
            cursor = stop
        if cursor != len(class_rows):
            raise AssertionError(f"Failed to assign every sample for {wnid}")

    samples.sort(key=lambda row: row["sample_id"])
    split_counts = {name: sum(row["split"] == name for row in samples) for name in SPLIT_NAMES}
    validate_split_manifest_samples(samples)

    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "algorithm": "per_wnid_sort_sha256_seed_colon_relative_path_then_largest_remainder_v1",
        "source_train_root": project_relative(train_root),
        "official_val_accessed": False,
        "split_seed": split_seed,
        "ratios": ratios,
        "classes": image_folder.classes,
        "class_to_idx": image_folder.class_to_idx,
        "sample_count": len(samples),
        "split_counts": split_counts,
        "per_class_split_counts": per_class_counts,
        "content_sha256_required": bool(config["split"].get("require_content_sha256", True)),
        "cross_split_exact_content_duplicates_rejected": bool(
            config["split"].get("reject_exact_content_duplicates", True)
        ),
        "samples": samples,
    }


def validate_split_manifest_samples(samples: list[dict[str, Any]]) -> None:
    sample_ids: set[str] = set()
    relative_paths: set[str] = set()
    split_members: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    content_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in samples:
        split = str(row["split"])
        if split not in split_members:
            raise RuntimeError(f"Unknown split in manifest: {split}")
        sample_id = str(row["sample_id"])
        relative_path = str(row["relative_path"])
        if sample_id in sample_ids:
            raise RuntimeError(f"A sample appears more than once in the split manifest: {sample_id}")
        if relative_path in relative_paths:
            raise RuntimeError(f"A relative path appears more than once in the split manifest: {relative_path}")
        sample_ids.add(sample_id)
        relative_paths.add(relative_path)
        split_members[split].add(sample_id)
        content_rows[str(row["content_sha256"])].append((sample_id, split))

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = split_members[left].intersection(split_members[right])
            if overlap:
                raise RuntimeError(f"Split overlap between {left} and {right}: {sorted(overlap)[:5]}")
    duplicate_content = {digest: rows for digest, rows in content_rows.items() if len(rows) > 1}
    if duplicate_content:
        preview = list(duplicate_content.items())[:5]
        raise RuntimeError(f"Exact image content is duplicated in the official train source: {preview}")


def manifest_csv_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in manifest["samples"]]


def materialize_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    overwrite: bool,
) -> tuple[str, Path]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_path.with_suffix(".csv")
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise FileExistsError(
                f"Existing split manifest differs from the deterministic current manifest: {manifest_path}. "
                "Inspect the data/config or pass --overwrite deliberately."
            )
    else:
        save_json(manifest_path, manifest)
    if not manifest_csv.exists() or overwrite:
        write_csv(manifest_csv, manifest_csv_rows(manifest))
    return sha256_file(manifest_path), manifest_csv


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, train_root: Path, rows: list[dict[str, Any]], transform: Callable[[Image.Image], torch.Tensor]):
        self.train_root = train_root
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = (self.train_root / row["relative_path"]).resolve()
        if self.train_root != path and self.train_root not in path.parents:
            raise RuntimeError(f"Manifest path escaped official train: {path}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["class_idx"])


def build_transforms(config: dict[str, Any]) -> tuple[Callable[[Image.Image], torch.Tensor], Callable[[Image.Image], torch.Tensor]]:
    data_cfg = config["data"]
    training = config["training"]
    image_size = int(data_cfg["image_size"])
    mean = [float(value) for value in training.get("normalization_mean", [0.485, 0.456, 0.406])]
    std = [float(value) for value in training.get("normalization_std", [0.229, 0.224, 0.225])]
    crop_scale = tuple(float(value) for value in training.get("train_crop_scale", [0.6, 1.0]))

    train_steps: list[Any] = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale, interpolation=transforms.InterpolationMode.BILINEAR)
    ]
    if bool(training.get("random_horizontal_flip", True)):
        train_steps.append(transforms.RandomHorizontalFlip())
    randaugment_ops = int(training.get("randaugment_num_ops", 0))
    if randaugment_ops > 0:
        train_steps.append(
            transforms.RandAugment(
                num_ops=randaugment_ops,
                magnitude=int(training.get("randaugment_magnitude", 9)),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    erase_probability = float(training.get("random_erasing_probability", 0.0))
    if erase_probability > 0.0:
        train_steps.append(transforms.RandomErasing(p=erase_probability))

    eval_transform = transforms.Compose(
        [
            # Match the exact geometry sent through the formal DeepJSCC pipeline.
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, int]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(config["training"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


MODEL_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    # Every builder passes weights=None explicitly. Add future scratch architectures here.
    "efficientnet_b0": lambda num_classes: models.efficientnet_b0(
        weights=None,
        num_classes=num_classes,
    ),
    "mobilenet_v3_small": lambda num_classes: models.mobilenet_v3_small(
        weights=None,
        num_classes=num_classes,
    ),
    "resnet18": lambda num_classes: models.resnet18(
        weights=None,
        num_classes=num_classes,
    ),
}


def build_scratch_model(role: str, model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    weights = model_cfg.get("weights")
    if weights is not None:
        raise ValueError(f"{role}: pretrained weights are forbidden; expected weights=None")
    if model_cfg.get("pretrained", False) is not False:
        raise ValueError(f"{role}: pretrained=True is forbidden")
    architecture = str(model_cfg["architecture"])
    if architecture not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported scratch architecture {architecture!r}; available={list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[architecture](num_classes)
    return model


def macro_accuracy(correct: torch.Tensor, count: torch.Tensor) -> float:
    if torch.any(count <= 0):
        missing = torch.nonzero(count <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Macro accuracy is undefined because classes are absent: {missing}")
    return float((correct.double() / count.double()).mean().item())


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    num_classes: int,
    collect_logits: bool = False,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = torch.zeros(num_classes, dtype=torch.long)
    count = torch.zeros(num_classes, dtype=torch.long)
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        batch_loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        predictions = logits.argmax(dim=1)
        loss_sum += float(batch_loss.item())
        total += int(labels.numel())
        for class_idx in range(num_classes):
            mask = labels == class_idx
            count[class_idx] += int(mask.sum().item())
            correct[class_idx] += int((predictions[mask] == class_idx).sum().item())
        if collect_logits:
            logits_parts.append(logits.detach().float().cpu())
            label_parts.append(labels.detach().cpu())
    if total == 0:
        raise RuntimeError("Evaluation loader is empty")
    output: dict[str, Any] = {
        "loss": loss_sum / total,
        "top1": int(correct.sum().item()) / total,
        "macro_top1": macro_accuracy(correct, count),
        "per_class_top1": [float(c / n) for c, n in zip(correct.double().tolist(), count.double().tolist())],
        "per_class_count": count.tolist(),
        "sample_count": total,
    }
    if collect_logits:
        output["logits"] = torch.cat(logits_parts, dim=0)
        output["labels"] = torch.cat(label_parts, dim=0)
    return output


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int,
    label_smoothing: float,
    use_amp: bool,
    grad_clip_norm: float,
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    total = 0
    correct = torch.zeros(num_classes, dtype=torch.long)
    count = torch.zeros(num_classes, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        if grad_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        loss_sum += float(loss.detach().item()) * batch_size
        total += batch_size
        for class_idx in range(num_classes):
            mask = labels == class_idx
            count[class_idx] += int(mask.sum().item())
            correct[class_idx] += int((predictions[mask] == class_idx).sum().item())
    if total == 0:
        raise RuntimeError("Training loader is empty")
    return {
        "loss": loss_sum / total,
        "top1": int(correct.sum().item()) / total,
        "macro_top1": macro_accuracy(correct, count),
        "sample_count": total,
    }


def cosine_with_warmup_lambda(epoch: int, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    cosine_epochs = max(epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    ece = torch.zeros((), dtype=torch.float64)
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=confidence.dtype)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper) if index else (confidence >= lower) & (confidence <= upper)
        if mask.any():
            weight = mask.double().mean()
            gap = confidence[mask].double().mean() - correct[mask].double().mean()
            ece += weight * gap.abs()
    return float(ece.item())


def fit_scalar_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    # ``evaluate`` intentionally runs under inference_mode; copy through NumPy so
    # LBFGS receives ordinary tensors that autograd may save for backward.
    logits64 = torch.from_numpy(logits.detach().cpu().numpy().copy()).double()
    labels_cpu = torch.from_numpy(labels.detach().cpu().numpy().copy()).long()
    before_nll = float(nn.functional.cross_entropy(logits64, labels_cpu).item())
    before_ece = expected_calibration_error(logits64, labels_cpu)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(logits64 / temperature, labels_cpu)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    after_logits = logits64 / temperature
    after_nll = float(nn.functional.cross_entropy(after_logits, labels_cpu).item())
    if not math.isfinite(after_nll) or after_nll > before_nll + 1e-10:
        temperature = 1.0
        after_logits = logits64
        after_nll = before_nll
    after_ece = expected_calibration_error(after_logits, labels_cpu)
    return {
        "temperature": temperature,
        "nll_before": before_nll,
        "nll_after": after_nll,
        "ece15_before": before_ece,
        "ece15_after": after_ece,
        "sample_count": int(labels_cpu.numel()),
    }


def prepare_role_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        meaningful = [path for path in output_dir.iterdir() if path.name not in {".DS_Store"}]
        if meaningful and not overwrite:
            raise FileExistsError(f"Classifier output already exists; pass --overwrite deliberately: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def require_recovery_value(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(f"Recovery validation failed for {label}: actual={actual!r}, expected={expected!r}")


def validate_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Completed training history CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(history):
        raise RuntimeError(f"history.csv row count differs from progress JSON: {len(rows)} != {len(history)}")
    for row_index, (csv_row, expected_row) in enumerate(zip(rows, history), start=1):
        if set(csv_row) != set(expected_row):
            raise RuntimeError(f"history.csv columns differ at row {row_index}")
        for key, expected in expected_row.items():
            raw = csv_row[key]
            if isinstance(expected, bool):
                actual: Any = raw == "True" if raw in {"True", "False"} else raw
            elif isinstance(expected, int):
                actual = int(raw)
            elif isinstance(expected, float):
                actual = float(raw)
            else:
                actual = raw
            require_recovery_value(actual, expected, f"history.csv row {row_index} field {key}")


def validate_completed_history(
    role: str,
    history: list[dict[str, Any]],
    expected_epochs: int,
) -> tuple[int, float, float]:
    if len(history) != expected_epochs:
        raise RuntimeError(
            f"Recovery requires exactly {expected_epochs} completed epochs for {role}; found {len(history)}"
        )
    best_epoch = -1
    best_macro = -math.inf
    best_loss = math.inf
    required_numeric = (
        "lr",
        "train_loss",
        "train_top1",
        "train_macro_top1",
        "cls_cal_loss",
        "cls_cal_top1",
        "cls_cal_macro_top1",
        "epoch_seconds",
    )
    for expected_epoch, row in enumerate(history, start=1):
        require_recovery_value(row.get("role"), role, f"history epoch {expected_epoch} role")
        require_recovery_value(row.get("epoch"), expected_epoch, f"history epoch sequence {expected_epoch}")
        for key in required_numeric:
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {key} in {role} epoch {expected_epoch}")
        macro = float(row["cls_cal_macro_top1"])
        loss = float(row["cls_cal_loss"])
        improved = macro > best_macro + 1e-12 or (
            math.isclose(macro, best_macro, abs_tol=1e-12) and loss < best_loss - 1e-12
        )
        require_recovery_value(
            row.get("selected_best"), improved, f"history epoch {expected_epoch} selected_best"
        )
        if improved:
            best_epoch = expected_epoch
            best_macro = macro
            best_loss = loss
    if best_epoch <= 0:
        raise RuntimeError(f"No best epoch can be reconstructed from completed history for {role}")
    return best_epoch, best_macro, best_loss


def validate_calibration_files(train_root: Path, cal_rows: list[dict[str, Any]]) -> None:
    """Validate only cls_cal image bytes; policy_dev and official val remain unopened."""
    if not cal_rows:
        raise RuntimeError("cls_cal is empty")
    for row in cal_rows:
        path = (train_root / str(row["relative_path"])).resolve()
        if train_root != path and train_root not in path.parents:
            raise RuntimeError(f"cls_cal manifest path escaped official train: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"cls_cal image is missing: {path}")
        require_recovery_value(path.stat().st_size, int(row["size_bytes"]), f"cls_cal size {row['sample_id']}")
        require_recovery_value(sha256_file(path), str(row["content_sha256"]), f"cls_cal hash {row['sample_id']}")


def finalize_role(
    *,
    role: str,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    protocol_hash: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    device: torch.device,
    model: nn.Module,
    cal_loader: DataLoader[tuple[torch.Tensor, int]],
    history: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_macro: float,
    best_loss: float,
    script_snapshot: Path,
    script_sha256: str,
    run_command: str,
    started: float,
    use_amp: bool,
) -> dict[str, Any]:
    """Calibrate and materialize a role identically after normal or recovered training."""
    model_cfg = config["scratch_classifiers"][role]
    output_dir = resolve_project_path(model_cfg["output_dir"])
    classes = [str(item) for item in manifest["classes"]]
    class_to_idx = {str(key): int(value) for key, value in manifest["class_to_idx"].items()}
    num_classes = len(classes)
    train_sample_count = int(manifest["split_counts"]["cls_train"])
    cal_sample_count = int(manifest["split_counts"]["cls_cal"])

    model.load_state_dict(best_state, strict=True)
    calibrated_eval = evaluate(model, cal_loader, device, num_classes, collect_logits=True)
    calibration = fit_scalar_temperature(calibrated_eval.pop("logits"), calibrated_eval.pop("labels"))
    temperature = float(calibration["temperature"])
    min_macro = float(model_cfg["min_cal_macro_top1"])
    quality_gate_passed = best_macro + 1e-12 >= min_macro

    versions = version_metadata()
    checkpoint_name = "best.pt" if quality_gate_passed else "rejected_best.pt"
    checkpoint_path = output_dir / "checkpoints" / checkpoint_name
    checkpoint_payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "role": role,
        "role_description": str(model_cfg.get("role", role)),
        "architecture": str(model_cfg["architecture"]),
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": classes,
        "state_dict": best_state,
        "temperature": temperature,
        "calibration": calibration,
        "seed": int(model_cfg["seed"]),
        "weights": None,
        "pretrained": False,
        "random_initialization": True,
        "training_split": "cls_train",
        "selection_split": "cls_cal",
        "temperature_scaling_split": "cls_cal",
        "policy_dev_manifest_hash_only": True,
        "policy_dev_used_for_training_selection_or_calibration": False,
        "official_val_accessed": False,
        "best_epoch": best_epoch,
        "best_cls_cal_macro_top1": best_macro,
        "best_cls_cal_loss": best_loss,
        "min_cal_macro_top1": min_macro,
        "quality_gate_passed": quality_gate_passed,
        "split_manifest_path": project_relative(manifest_path),
        "split_manifest_sha256": manifest_sha256,
        "config_path": project_relative(config_path),
        "config_sha256": config_sha256,
        "config_hash": config_sha256,
        "protocol_sha256": protocol_hash,
        "training_script": project_relative(script_snapshot),
        "training_script_sha256": script_sha256,
        "run_command": run_command,
        "git_commit": git_commit(),
        "git_state": git_dirty_state(),
        "versions": versions,
    }
    atomic_torch_save(checkpoint_payload, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    with (checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")).open("w", encoding="utf-8") as handle:
        handle.write(f"{checkpoint_sha256}  {checkpoint_path.name}\n")

    history_payload = {
        "format_version": 1,
        "role": role,
        "architecture": str(model_cfg["architecture"]),
        "seed": int(model_cfg["seed"]),
        "weights": None,
        "pretrained": False,
        "training_split": "cls_train",
        "selection_split": "cls_cal",
        "policy_dev_manifest_hash_only": True,
        "policy_dev_used_for_training_selection_or_calibration": False,
        "official_val_accessed": False,
        "split_manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_hash,
        "training_script": project_relative(script_snapshot),
        "training_script_sha256": script_sha256,
        "run_command": run_command,
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": best_epoch,
        "best_cls_cal_macro_top1": best_macro,
        "calibration": calibration,
        "epochs": history,
    }
    save_json(output_dir / "history.json", history_payload)
    write_csv(output_dir / "history.csv", history)

    summary = {
        "role": role,
        "role_description": str(model_cfg.get("role", role)),
        "architecture": str(model_cfg["architecture"]),
        "seed": int(model_cfg["seed"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_sample_count": train_sample_count,
        "cls_cal_sample_count": cal_sample_count,
        "best_epoch": best_epoch,
        "best_cls_cal_macro_top1": best_macro,
        "best_cls_cal_top1": float(calibrated_eval["top1"]),
        "best_cls_cal_loss": best_loss,
        "min_cal_macro_top1": min_macro,
        "quality_gate_passed": quality_gate_passed,
        "temperature": temperature,
        "calibration": calibration,
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest": project_relative(manifest_path),
        "split_manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_hash,
        "training_script": project_relative(script_snapshot),
        "training_script_sha256": script_sha256,
        "run_command": run_command,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "amp_enabled": use_amp,
        "versions": versions,
    }
    save_json(output_dir / "summary.json", summary)
    # Progress artifacts are retained until every durable final artifact above exists.
    (output_dir / "checkpoints" / "best_in_progress.pt").unlink(missing_ok=True)
    (output_dir / "history_in_progress.json").unlink(missing_ok=True)
    return summary


def train_role(
    role: str,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    protocol_hash: str,
    train_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    model_cfg = config["scratch_classifiers"][role]
    training = config["training"]
    output_dir = resolve_project_path(model_cfg["output_dir"])
    prepare_role_output(output_dir, overwrite)
    shutil.copy2(config_path, output_dir / "config.yaml")
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    script_snapshot = output_dir / "script_snapshot.py"
    shutil.copy2(script_path, script_snapshot)
    if sha256_file(script_snapshot) != script_sha256:
        raise RuntimeError("Training script snapshot hash mismatch")
    run_command = " ".join(shlex.quote(item) for item in sys.argv)

    seed = int(model_cfg["seed"])
    set_global_seed(seed)
    classes = [str(item) for item in manifest["classes"]]
    class_to_idx = {str(key): int(value) for key, value in manifest["class_to_idx"].items()}
    num_classes = len(classes)
    train_transform, eval_transform = build_transforms(config)
    train_rows = [row for row in manifest["samples"] if row["split"] == "cls_train"]
    cal_rows = [row for row in manifest["samples"] if row["split"] == "cls_cal"]
    # policy_dev is intentionally neither loaded nor evaluated by this training process.
    train_dataset = ManifestImageDataset(train_root, train_rows, train_transform)
    cal_dataset = ManifestImageDataset(train_root, cal_rows, eval_transform)
    train_loader = make_loader(train_dataset, config, device, seed, shuffle=True)
    cal_loader = make_loader(cal_dataset, config, device, seed + 1, shuffle=False)

    model = build_scratch_model(role, model_cfg, num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["lr"]),
        momentum=float(training.get("momentum", 0.9)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        nesterov=bool(training.get("nesterov", False)),
    )
    epochs = int(training["epochs"])
    warmup_epochs = int(training.get("warmup_epochs", 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_with_warmup_lambda(epoch, epochs, warmup_epochs),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_macro = -math.inf
    best_loss = math.inf
    started = time.time()
    for epoch_index in range(epochs):
        epoch_started = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            num_classes=num_classes,
            label_smoothing=float(training.get("label_smoothing", 0.0)),
            use_amp=use_amp,
            grad_clip_norm=float(training.get("grad_clip_norm", 0.0)),
        )
        cal_metrics = evaluate(model, cal_loader, device, num_classes)
        improved = cal_metrics["macro_top1"] > best_macro + 1e-12 or (
            math.isclose(cal_metrics["macro_top1"], best_macro, abs_tol=1e-12)
            and cal_metrics["loss"] < best_loss - 1e-12
        )
        if improved:
            best_macro = float(cal_metrics["macro_top1"])
            best_loss = float(cal_metrics["loss"])
            best_epoch = epoch_index + 1
            best_state = cpu_state_dict(model)
            atomic_torch_save(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "status": "training_in_progress",
                    "role": role,
                    "architecture": str(model_cfg["architecture"]),
                    "state_dict": best_state,
                    "epoch": best_epoch,
                    "cls_cal_macro_top1": best_macro,
                    "cls_cal_loss": best_loss,
                    "split_manifest_sha256": manifest_sha256,
                    "config_sha256": config_sha256,
                    "protocol_sha256": protocol_hash,
                    "official_val_accessed": False,
                },
                output_dir / "checkpoints" / "best_in_progress.pt",
            )
        row = {
            "role": role,
            "epoch": epoch_index + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_macro_top1": train_metrics["macro_top1"],
            "cls_cal_loss": cal_metrics["loss"],
            "cls_cal_top1": cal_metrics["top1"],
            "cls_cal_macro_top1": cal_metrics["macro_top1"],
            "selected_best": improved,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        # Preserve progress from long scratch runs even if the process is interrupted.
        write_csv(output_dir / "history.csv", history)
        save_json(
            output_dir / "history_in_progress.json",
            {
                "status": "training",
                "role": role,
                "architecture": str(model_cfg["architecture"]),
                "seed": seed,
                "weights": None,
                "pretrained": False,
                "training_split": "cls_train",
                "selection_split": "cls_cal",
                "policy_dev_manifest_hash_only": True,
                "policy_dev_used_for_training_selection_or_calibration": False,
                "official_val_accessed": False,
                "split_manifest_sha256": manifest_sha256,
                "config_sha256": config_sha256,
                "protocol_sha256": protocol_hash,
                "training_script_sha256": script_sha256,
                "run_command": run_command,
                "best_epoch_so_far": best_epoch,
                "best_cls_cal_macro_top1_so_far": best_macro,
                "epochs": history,
            },
        )
        print(
            f"[{role}] epoch {epoch_index + 1:03d}/{epochs:03d} "
            f"lr={current_lr:.6g} train_loss={train_metrics['loss']:.4f} "
            f"cal_macro_top1={cal_metrics['macro_top1']:.4f} "
            f"cal_top1={cal_metrics['top1']:.4f} best_epoch={best_epoch}",
            flush=True,
        )
        scheduler.step()

    if best_state is None or best_epoch <= 0:
        raise RuntimeError(f"No best checkpoint was selected for {role}")
    return finalize_role(
        role=role,
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        protocol_hash=protocol_hash,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        device=device,
        model=model,
        cal_loader=cal_loader,
        history=history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_macro=best_macro,
        best_loss=best_loss,
        script_snapshot=script_snapshot,
        script_sha256=script_sha256,
        run_command=run_command,
        started=started,
        use_amp=use_amp,
    )


def load_recovery_manifest(
    config: dict[str, Any],
    train_root: Path,
) -> tuple[dict[str, Any], Path, str]:
    """Load the already-frozen manifest without rescanning official train or official val."""
    manifest_path = resolve_project_path(config["outputs"]["split_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Recovery split manifest is missing: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise TypeError(f"Invalid recovery split manifest: {manifest_path}")
    require_recovery_value(manifest.get("format_version"), MANIFEST_FORMAT_VERSION, "manifest format_version")
    require_recovery_value(manifest.get("official_val_accessed"), False, "manifest official_val_accessed")
    require_recovery_value(manifest.get("source_train_root"), project_relative(train_root), "manifest train root")
    require_recovery_value(
        manifest.get("sample_count"), int(config["data"]["train_image_count"]), "manifest sample_count"
    )
    require_recovery_value(manifest.get("split_seed"), int(config["split"]["seed"]), "manifest split_seed")
    require_recovery_value(
        manifest.get("classes"), [str(item) for item in config["data"]["classes"]], "manifest classes"
    )
    expected_counts = {name: sum(row.get("split") == name for row in manifest["samples"]) for name in SPLIT_NAMES}
    require_recovery_value(manifest.get("split_counts"), expected_counts, "manifest split_counts")
    return manifest, manifest_path, manifest_sha256


def finalize_completed_role(
    *,
    role: str,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    protocol_hash: str,
    train_root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Safely finish calibration after all preregistered training epochs completed."""
    model_cfg = config["scratch_classifiers"][role]
    output_dir = resolve_project_path(model_cfg["output_dir"])
    progress_path = output_dir / "history_in_progress.json"
    history_csv_path = output_dir / "history.csv"
    checkpoint_path = output_dir / "checkpoints" / "best_in_progress.pt"
    script_snapshot = output_dir / "script_snapshot.py"
    config_snapshot = output_dir / "config.yaml"
    final_paths = [
        output_dir / "checkpoints" / "best.pt",
        output_dir / "checkpoints" / "rejected_best.pt",
        output_dir / "history.json",
        output_dir / "summary.json",
    ]
    existing_final = [str(path) for path in final_paths if path.exists()]
    if existing_final:
        raise FileExistsError(f"Refusing to overwrite existing final role artifacts: {existing_final}")
    for required in (progress_path, history_csv_path, checkpoint_path, script_snapshot, config_snapshot):
        if not required.is_file():
            raise FileNotFoundError(f"Required recovery artifact is missing: {required}")

    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if not isinstance(progress, dict) or not isinstance(progress.get("epochs"), list):
        raise TypeError(f"Invalid training progress JSON: {progress_path}")
    expected_progress = {
        "status": "training",
        "role": role,
        "architecture": str(model_cfg["architecture"]),
        "seed": int(model_cfg["seed"]),
        "weights": None,
        "pretrained": False,
        "training_split": "cls_train",
        "selection_split": "cls_cal",
        "policy_dev_manifest_hash_only": True,
        "policy_dev_used_for_training_selection_or_calibration": False,
        "official_val_accessed": False,
        "split_manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_hash,
    }
    for key, expected in expected_progress.items():
        require_recovery_value(progress.get(key), expected, f"progress {key}")
    require_recovery_value(sha256_file(config_snapshot), config_sha256, "config snapshot SHA-256")
    script_sha256 = sha256_file(script_snapshot)
    require_recovery_value(
        progress.get("training_script_sha256"), script_sha256, "training script snapshot SHA-256"
    )

    history = progress["epochs"]
    best_epoch, best_macro, best_loss = validate_completed_history(
        role, history, int(config["training"]["epochs"])
    )
    validate_history_csv(history_csv_path, history)
    require_recovery_value(progress.get("best_epoch_so_far"), best_epoch, "progress best epoch")
    require_recovery_value(
        progress.get("best_cls_cal_macro_top1_so_far"), best_macro, "progress best cls_cal macro top1"
    )

    in_progress = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(in_progress, dict) or not isinstance(in_progress.get("state_dict"), dict):
        raise TypeError(f"Invalid in-progress checkpoint: {checkpoint_path}")
    expected_checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "status": "training_in_progress",
        "role": role,
        "architecture": str(model_cfg["architecture"]),
        "epoch": best_epoch,
        "cls_cal_macro_top1": best_macro,
        "cls_cal_loss": best_loss,
        "split_manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_hash,
        "official_val_accessed": False,
    }
    for key, expected in expected_checkpoint.items():
        require_recovery_value(in_progress.get(key), expected, f"best-in-progress {key}")
    best_state = in_progress["state_dict"]
    for name, tensor in best_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid tensor in best-in-progress state_dict: {name!r}")

    cal_rows = [row for row in manifest["samples"] if row.get("split") == "cls_cal"]
    require_recovery_value(len(cal_rows), int(manifest["split_counts"]["cls_cal"]), "cls_cal row count")
    validate_calibration_files(train_root, cal_rows)
    _, eval_transform = build_transforms(config)
    cal_dataset = ManifestImageDataset(train_root, cal_rows, eval_transform)
    cal_loader = make_loader(
        cal_dataset,
        config,
        device,
        int(model_cfg["seed"]) + 1,
        shuffle=False,
    )
    set_global_seed(int(model_cfg["seed"]))
    model = build_scratch_model(role, model_cfg, len(manifest["classes"])).to(device)
    model.load_state_dict(best_state, strict=True)
    completed_epoch_seconds = sum(float(row["epoch_seconds"]) for row in history)
    started = time.time() - completed_epoch_seconds
    return finalize_role(
        role=role,
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        protocol_hash=protocol_hash,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        device=device,
        model=model,
        cal_loader=cal_loader,
        history=history,
        best_state=best_state,
        best_epoch=best_epoch,
        best_macro=best_macro,
        best_loss=best_loss,
        script_snapshot=script_snapshot,
        script_sha256=script_sha256,
        run_command=str(progress["run_command"]),
        started=started,
        use_amp=bool(config["training"].get("amp", True)) and device.type == "cuda",
    )


def dry_run_payload(
    config: dict[str, Any],
    config_path: Path,
    train_root: Path,
    manifest: dict[str, Any],
    roles: list[str],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "status": "dry_run_ok",
        "config": project_relative(config_path),
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": protocol_sha256(config),
        "official_train_root": project_relative(train_root),
        "official_val_accessed": False,
        "manifest_target": project_relative(resolve_project_path(config["outputs"]["split_manifest"])),
        "manifest_sample_count": manifest["sample_count"],
        "split_counts": manifest["split_counts"],
        "per_class_split_counts": manifest["per_class_split_counts"],
        "roles": {
            role: {
                "architecture": config["scratch_classifiers"][role]["architecture"],
                "seed": int(config["scratch_classifiers"][role]["seed"]),
                "weights": None,
                "output_dir": config["scratch_classifiers"][role]["output_dir"],
            }
            for role in roles
        },
        "device": str(device),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config is missing: {config_path}")
    config = load_yaml(config_path)
    validate_config(config)
    device = resolve_device(args.device)
    train_root = resolve_train_root(config)
    assert_outputs_outside_dataset(config, train_root)

    if args.finalize_completed is not None:
        if args.roles is not None or args.overwrite or args.dry_run:
            raise ValueError("--finalize-completed cannot be combined with --roles, --overwrite, or --dry-run")
        roles = parse_roles(args.finalize_completed, config["scratch_classifiers"])
        manifest, manifest_path, manifest_sha256 = load_recovery_manifest(config, train_root)
        config_sha256 = sha256_file(config_path)
        protocol_hash = protocol_sha256(config)
        summaries = [
            finalize_completed_role(
                role=role,
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                protocol_hash=protocol_hash,
                train_root=train_root,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                device=device,
            )
            for role in roles
        ]
        print(json.dumps({"status": "finalized_completed_training", "roles": summaries}, indent=2))
        failures = [item["role"] for item in summaries if not item["quality_gate_passed"]]
        if failures:
            raise RuntimeError(f"Classifier quality gate failed for roles: {failures}")
        return

    roles = parse_roles(args.roles, config["scratch_classifiers"])
    archive_integrity = validate_source_archive(config)
    manifest = build_split_manifest(config, train_root)
    expected_train_count = int(config["data"]["train_image_count"])
    if int(manifest["sample_count"]) != expected_train_count:
        raise RuntimeError(
            f"Official train count mismatch: actual={manifest['sample_count']}, expected={expected_train_count}"
        )

    if args.dry_run:
        print(json.dumps(dry_run_payload(config, config_path, train_root, manifest, roles, device), indent=2))
        return

    manifest_path = resolve_project_path(config["outputs"]["split_manifest"])
    manifest_sha256, manifest_csv = materialize_manifest(manifest_path, manifest, args.overwrite)
    config_sha256 = sha256_file(config_path)
    protocol_hash = protocol_sha256(config)
    summaries: list[dict[str, Any]] = []
    for role in roles:
        summaries.append(
            train_role(
                role=role,
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                protocol_hash=protocol_hash,
                train_root=train_root,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                device=device,
                overwrite=args.overwrite,
            )
        )

    all_role_summaries: list[dict[str, Any]] = []
    missing_roles: list[str] = []
    for configured_role, model_cfg in config["scratch_classifiers"].items():
        role_summary_path = resolve_project_path(model_cfg["output_dir"]) / "summary.json"
        if not role_summary_path.is_file():
            missing_roles.append(configured_role)
            continue
        with role_summary_path.open("r", encoding="utf-8") as handle:
            role_summary = json.load(handle)
        expected_manifest_sha256 = manifest_sha256
        mismatches = []
        if role_summary.get("role") != configured_role:
            mismatches.append(f"role={role_summary.get('role')!r}")
        if role_summary.get("protocol_sha256") != protocol_hash:
            mismatches.append("protocol_sha256")
        if role_summary.get("split_manifest_sha256") != expected_manifest_sha256:
            mismatches.append("split_manifest_sha256")
        checkpoint_path = resolve_project_path(role_summary.get("checkpoint", ""))
        if not checkpoint_path.is_file():
            mismatches.append("checkpoint_missing")
        elif sha256_file(checkpoint_path) != role_summary.get("checkpoint_sha256"):
            mismatches.append("checkpoint_sha256")
        if mismatches:
            raise RuntimeError(
                f"Refusing to combine stale or incompatible classifier summary for {configured_role}: {mismatches}"
            )
        all_role_summaries.append(role_summary)
    all_quality_passed = bool(all_role_summaries) and all(
        bool(item["quality_gate_passed"]) for item in all_role_summaries
    )
    if any(not item["quality_gate_passed"] for item in summaries):
        root_status = "quality_gate_failed"
    elif missing_roles:
        root_status = "partial"
    elif all_quality_passed:
        root_status = "complete"
    else:
        root_status = "quality_gate_failed"
    root_summary = {
        "status": root_status,
        "config": project_relative(config_path),
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_hash,
        "manifest": project_relative(manifest_path),
        "manifest_csv": project_relative(manifest_csv),
        "manifest_sha256": manifest_sha256,
        "official_val_accessed": False,
        "archive_integrity": archive_integrity,
        "roles_trained_this_invocation": [item["role"] for item in summaries],
        "missing_roles": missing_roles,
        "roles": all_role_summaries,
    }
    root_output = resolve_project_path(config["outputs"].get("classifier_root", manifest_path.parent))
    root_output.mkdir(parents=True, exist_ok=True)
    save_json(root_output / "training_summary.json", root_summary)
    print(json.dumps(root_summary, indent=2))
    failures = [item["role"] for item in summaries if not item["quality_gate_passed"]]
    if failures:
        raise RuntimeError(f"Classifier quality gate failed for roles: {failures}")


if __name__ == "__main__":
    main()
