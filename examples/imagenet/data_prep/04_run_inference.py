"""
Run inference with pretrained models on the ImageNet-1k validation set.

Saves softmax probabilities as data/probs/<model_name>.npy (shape: 50000 x 1000).
True labels are already saved by 02_check_imagenet.py and not repeated here.

Usage
-----
    python data_prep/04_run_inference.py --model convnext_tiny
    python data_prep/04_run_inference.py --all
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # so config.py can be found
from config import IMAGENET_VAL_DIR

PROBS_DIR = Path(__file__).parent.parent / "data" / "probs"
PROBS_DIR.mkdir(parents=True, exist_ok=True)

# Models:
# documented_top1 values for torchvision models are from
# https://pytorch.org/vision/stable/models.html
MODELS = {
    # torchvision models
    "resnet18": {
        "backend": "torchvision",
        "documented_top1": 0.69758,
    },
    "efficientnet_b0": {
        "backend": "torchvision",
        "documented_top1": 0.77692,
    },
    "mobilenet_v3_large": {
        "backend": "torchvision",
        "documented_top1": 0.75274,
    },
    "mobilenet_v3_small": {
        "backend": "torchvision",
        "documented_top1": 0.67668,
    },
    "shufflenet_v2_x1_0": {
        "backend": "torchvision",
        "documented_top1": 0.69362,
    },
    # timm models
    "resnet50_timm": {
        "backend": "timm",
        "timm_name": "resnet50.a1_in1k",
    },
    "convnext_tiny": {
        "backend": "timm",
        "timm_name": "convnext_tiny.fb_in1k",
    },
    "vit_small_patch16_224": {
        "backend": "timm",
        "timm_name": "vit_small_patch16_224.augreg_in1k",
    },
    "swin_tiny": {
        "backend": "timm",
        "timm_name": "swin_tiny_patch4_window7_224.ms_in1k",
    },
    "efficientnet_b2": {
        "backend": "timm",
        "timm_name": "efficientnet_b2.ra_in1k",
    },
}

# torchvision builder/weights lookup
TORCHVISION_BUILDERS = {
    "resnet18":           (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "efficientnet_b0":    (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (tvm.shufflenet_v2_x1_0, tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
}


class ImageNetValDataset(Dataset):
    """Reads ImageNet-1k validation parquet files, serves images on demand."""

    def __init__(self, parquet_dir, transform):
        files = sorted(Path(parquet_dir).glob("validation-*.parquet"))
        assert len(files) > 0, f"No parquet files found in {parquet_dir}"
        dfs = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        assert len(self.df) == 50_000
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_field = row["image"]
        img_bytes = img_field["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.transform(img), idx


def load_model_and_transform(name, cfg):
    # Load model and get corresponding transform of PIL image to tensor
    if cfg["backend"] == "torchvision":
        builder, weights = TORCHVISION_BUILDERS[name]
        model = builder(weights=weights)
        transform = weights.transforms()
    else:
        model = timm.create_model(cfg["timm_name"], pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
    model.eval()  # inference mode: no dropout, fixed batchnorm
    return model, transform


def run_inference(name, device):
    out_path = PROBS_DIR / f"{name}.npy"
    if out_path.exists():
        print(f"  {name}: already exists, skipping.")
        return

    cfg = MODELS[name]
    print(f"\nLoading {name}...")
    model, transform = load_model_and_transform(name, cfg)
    model.to(device)

    dataset = ImageNetValDataset(IMAGENET_VAL_DIR, transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    probs = np.zeros((50_000, 1000), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=name):
            logits = model(imgs.to(device))
            probs[idxs.numpy()] = F.softmax(logits, dim=1).cpu().numpy()

    np.save(out_path, probs)

    # sanity check
    true_labels = np.load(PROBS_DIR / "true_labels.npy")
    top1 = (probs.argmax(axis=1) == true_labels).mean()
    if "documented_top1" in cfg:
        expected = cfg["documented_top1"]
        print(f"  top-1: {top1:.4f} (expected ~{expected:.4f})")
        if abs(top1 - expected) > 0.03:
            print(f"  WARNING: top-1 accuracy far from expected — check label alignment.")
    else:
        print(f"  top-1: {top1:.4f}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if args.all:
        for name in MODELS:
            run_inference(name, device)
    elif args.model:
        run_inference(args.model, device)
    else:
        parser.print_usage()
