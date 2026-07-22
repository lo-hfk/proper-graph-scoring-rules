"""
Step 2: Run inference with pretrained classifiers on the ImageNet-1k validation set.

Supports two backends:
  - torchvision  (original models: resnet18, resnet50, efficientnet_b0, etc.)
  - timm         (modern models: convnext_tiny, vit_small_patch16_224, etc.)

Both backends save outputs in the same format:
    model_validation_outputs/probs_<model>.npy        (50000, 1000) float32
    model_validation_outputs/true_labels_<model>.npy  (50000,) int64

Usage:
    python run_inference.py --model resnet50
    python run_inference.py --model convnext_tiny
    python run_inference.py --all
    python run_inference.py --all-timm      # only the timm models
    python run_inference.py --all-torchvision

PREREQUISITES:
  HF parquet files in DATA_DIR (see original script header for download instructions).
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# torchvision models
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
)

DATA_DIR = Path("C:/Users/ma1201204/Documents/Data/imagenet-1k")
OUT_DIR = Path("./model_validation_outputs")
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# torchvision model registry (unchanged from original)
# ---------------------------------------------------------------------------
TORCHVISION_REGISTRY = {
    "resnet50": {
        "backend": "torchvision",
        "builder": resnet50,
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "documented_top1": 0.80858,
        "params_millions": 25.6,
        "gflops": 4.09,
    },
    "resnet18": {
        "backend": "torchvision",
        "builder": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "documented_top1": 0.69758,
        "params_millions": 11.7,
        "gflops": 1.81,
    },
    "efficientnet_b0": {
        "backend": "torchvision",
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "documented_top1": 0.77692,
        "params_millions": 5.3,
        "gflops": 0.39,
    },
    "mobilenet_v3_large": {
        "backend": "torchvision",
        "builder": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "documented_top1": 0.75274,
        "params_millions": 5.5,
        "gflops": 0.22,
    },
    "mobilenet_v3_small": {
        "backend": "torchvision",
        "builder": mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "documented_top1": 0.67668,
        "params_millions": 2.5,
        "gflops": 0.06,
    },
    "shufflenet_v2_x1_0": {
        "backend": "torchvision",
        "builder": shufflenet_v2_x1_0,
        "weights": ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        "documented_top1": 0.69362,
        "params_millions": 2.3,
        "gflops": 0.14,
    },
}

# ---------------------------------------------------------------------------
# timm model registry
#
# All timm entries use "backend": "timm" and a "timm_name" string passed to
# timm.create_model().  The "documented_top1" values come from the timm model
# cards / papers, and are used only for the post-hoc sanity check.
#
# Model selection rationale (for the hierarchy-aware scoring rule paper):
#   - Similar Top-1 accuracy band (~80-84%) so flat metrics struggle to
#     separate them, making hierarchy-aware differences more visible.
#   - Four distinct architectural families spanning the major paradigms:
#     classic CNN, modern CNN, pure ViT, hierarchical ViT.
#   - All run comfortably on a laptop CPU/MPS (no training, inference only).
#
# timm model names can be verified via: timm.list_models(pretrained=True)
# The documented_top1 values below are from the timm model cards as of 2024.
# ---------------------------------------------------------------------------
TIMM_REGISTRY = {
    # Classic CNN baseline -- same as Bertinetto et al., good continuity
    "resnet50_timm": {
        "backend": "timm",
        "timm_name": "resnet50.a1_in1k",
        "documented_top1": 0.8013,
        "params_millions": 25.6,
        "gflops": 4.09,
        "notes": "ResNet-50 trained with timm's A1 recipe; slightly higher than torchvision V2",
    },
    # Modern CNN: ConvNeXt-Tiny, transformer-inspired training but pure convolutions
    "convnext_tiny": {
        "backend": "timm",
        "timm_name": "convnext_tiny.fb_in1k",
        "documented_top1": 0.8218,
        "params_millions": 28.6,
        "gflops": 4.47,
        "notes": "ConvNeXt-Tiny (Liu et al. 2022); CNN with transformer-inspired design",
    },
    # Pure Vision Transformer: shape-biased, qualitatively different errors
    "vit_small_patch16_224": {
        "backend": "timm",
        "timm_name": "vit_small_patch16_224.augreg_in1k",
        "documented_top1": 0.8099,
        "params_millions": 22.1,
        "gflops": 4.61,
        "notes": "ViT-S/16 (Dosovitskiy et al. 2020); pure transformer, shape-biased",
    },
    # Hierarchical ViT: local attention windows, hybrid inductive bias
    "swin_tiny": {
        "backend": "timm",
        "timm_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "documented_top1": 0.8119,
        "params_millions": 28.3,
        "gflops": 4.51,
        "notes": "Swin-Tiny (Liu et al. 2021); hierarchical transformer with local attention",
    },
    # EfficientNet-B2: compound-scaling CNN, different from ViT/Swin family
    "efficientnet_b2": {
        "backend": "timm",
        "timm_name": "efficientnet_b2.ra_in1k",
        "documented_top1": 0.8021,
        "params_millions": 9.1,
        "gflops": 1.00,
        "notes": "EfficientNet-B2 (Tan & Le 2019); compound-scaled CNN",
    },
}

MODEL_REGISTRY = {**TORCHVISION_REGISTRY, **TIMM_REGISTRY}


# ---------------------------------------------------------------------------
# Dataset (unchanged from original)
# ---------------------------------------------------------------------------
class ImageNetValParquetDataset(Dataset):
    def __init__(self, parquet_dir, transform):
        import pandas as pd

        parquet_dir = Path(parquet_dir)
        files = sorted(parquet_dir.glob("*.parquet"))
        assert len(files) > 0, (
            f"No .parquet files found in {parquet_dir}. "
            "Check the download path."
        )
        print(f"Found {len(files)} parquet shard(s); loading metadata...")

        dfs = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)

        assert "label" in self.df.columns
        assert "image" in self.df.columns
        assert len(self.df) == 50000, (
            f"Expected 50000 rows, found {len(self.df)}."
        )
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_field = row["image"]
        img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = int(row["label"])
        return self.transform(img), idx, label


# ---------------------------------------------------------------------------
# Model loading -- handles both backends
# ---------------------------------------------------------------------------
def load_model_and_transform(model_name, cfg):
    """
    Returns (model, transform) for either backend.
    For timm models, we use timm's own data_config to get the correct
    preprocessing -- equivalent to torchvision's weights.transforms().
    """
    if cfg["backend"] == "torchvision":
        weights = cfg["weights"]
        model = cfg["builder"](weights=weights)
        transform = weights.transforms()
        return model, transform

    elif cfg["backend"] == "timm":
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for timm-backend models. "
                "Install with: pip install timm"
            )
        model = timm.create_model(cfg["timm_name"], pretrained=True)
        # timm's resolve_data_config + create_transform gives the exact
        # preprocessing the model was trained with, equivalent to torchvision's
        # weights.transforms() -- handles different input sizes, normalisation
        # stats, interpolation modes, etc. automatically.
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform

    else:
        raise ValueError(f"Unknown backend: {cfg['backend']}")


# ---------------------------------------------------------------------------
# Inference loop (same for both backends)
# ---------------------------------------------------------------------------
def run_one_model(model_name, device):
    assert model_name in MODEL_REGISTRY, (
        f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
    )
    cfg = MODEL_REGISTRY[model_name]
    backend_label = f"[{cfg['backend']}]"
    notes = cfg.get("notes", "")
    print(f"\n{'='*70}\n{backend_label} {model_name}  "
          f"({cfg['params_millions']}M params, {cfg['gflops']} GFLOPs)\n"
          f"{notes}\n{'='*70}")

    model, transform = load_model_and_transform(model_name, cfg)
    model.eval()
    model.to(device)

    dataset = ImageNetValParquetDataset(DATA_DIR, transform)
    # num_workers=0 on Windows to avoid multiprocessing issues with DataLoader
    num_workers = 0 if torch.backends.mps.is_available() is False and device == "cpu" else 4
    # Actually: on Windows, num_workers > 0 requires the if __name__ == '__main__' guard.
    # Set to 0 to be safe; change to 4 if you're on Linux/Mac and want speed.
    num_workers = 0
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )

    all_probs = np.zeros((len(dataset), 1000), dtype=np.float32)
    true_labels = np.zeros(len(dataset), dtype=np.int64)

    print("Running inference...")
    with torch.no_grad():
        for imgs, idxs, labels in tqdm(loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            idxs_np = idxs.numpy()
            all_probs[idxs_np] = probs
            true_labels[idxs_np] = labels.numpy()

    np.save(OUT_DIR / f"probs_{model_name}.npy", all_probs)
    np.save(OUT_DIR / f"true_labels_{model_name}.npy", true_labels)

    pred_labels = all_probs.argmax(axis=1)
    top1_acc = (pred_labels == true_labels).mean()
    expected = cfg["documented_top1"]
    print(f"\nTop-1 accuracy: {top1_acc:.4f}  (expected ~{expected:.4f})")
    if abs(top1_acc - expected) > 0.03:
        print(f"WARNING: top-1 accuracy is far from documented value. "
              f"Check label alignment before trusting downstream results.")
    print(f"Saved: {OUT_DIR}/probs_{model_name}.npy  shape={all_probs.shape}")

    return top1_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Run inference for a single model.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all registered models (torchvision + timm).",
    )
    parser.add_argument(
        "--all-timm", action="store_true",
        help="Run only the timm models.",
    )
    parser.add_argument(
        "--all-torchvision", action="store_true",
        help="Run only the torchvision models.",
    )
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    if args.all:
        models_to_run = list(MODEL_REGISTRY.keys())
    elif args.all_timm:
        models_to_run = [k for k, v in MODEL_REGISTRY.items() if v["backend"] == "timm"]
    elif args.all_torchvision:
        models_to_run = [k for k, v in MODEL_REGISTRY.items() if v["backend"] == "torchvision"]
    elif args.model:
        models_to_run = [args.model]
    else:
        parser.print_help()
        return

    summary = {}
    for model_name in models_to_run:
        acc = run_one_model(model_name, device)
        summary[model_name] = acc

    if len(summary) > 1:
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        for name, acc in summary.items():
            backend = MODEL_REGISTRY[name]["backend"]
            print(f"  [{backend:12s}] {name:30s}  top-1 = {acc:.4f}")


if __name__ == "__main__":
    main()