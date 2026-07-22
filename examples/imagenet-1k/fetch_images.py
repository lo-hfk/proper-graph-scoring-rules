"""
Fetch one or more validation images by global index and save them as PNG.

Usage:
    python fetch_images.py 12345
    python fetch_images.py 12345 7890 42001
    python fetch_images.py 12345 --out-dir my_images
    python fetch_images.py 12345 --show      # also open in default viewer
"""

import argparse
import io
import sys
from pathlib import Path

from PIL import Image

DATA_DIR = Path("C:/Users/ma1201204/Documents/Data/imagenet-1k")
OUT_DIR  = Path("./model_validation_outputs/qualitative_examples")


def load_image(image_idx):
    import pyarrow.parquet as pq
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    shard_sizes = [pq.read_metadata(f).num_rows for f in files]
    cumulative  = [0]
    for s in shard_sizes:
        cumulative.append(cumulative[-1] + s)

    total = cumulative[-1]
    if image_idx >= total or image_idx < 0:
        raise IndexError(f"image_idx {image_idx} out of range (0..{total-1})")

    shard  = next(i for i in range(len(files))
                  if cumulative[i] <= image_idx < cumulative[i + 1])
    within = image_idx - cumulative[shard]

    row       = pq.read_table(files[shard]).slice(within, 1).to_pydict()
    img_field = row["image"][0]
    img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch validation images by index from HF parquet shards.")
    parser.add_argument("indices", type=int, nargs="+",
                        help="One or more image indices (0-based global index).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"Output directory. Default: {OUT_DIR}")
    parser.add_argument("--show", action="store_true",
                        help="Open each image in the default viewer after saving.")
    args = parser.parse_args()

    out_dir = args.out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        print(f"Fetching image {idx}...", end=" ", flush=True)
        try:
            img = load_image(idx)
            out_path = out_dir / f"image_{idx:05d}.png"
            img.save(out_path)
            print(f"saved to {out_path}  ({img.size[0]}x{img.size[1]})")
            if args.show:
                img.show()
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()