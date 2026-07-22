"""
Step 0: Quick diagnostic: run this FIRST, after downloading just one parquet shard,
to confirm the file structure matches what run_inference.py expects,
before committing to downloading all shards / running full inference.

Usage:
    python3 check_parquet_format.py data/imagenet_hf/<some_file>.parquet
"""
import io
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

def main(parquet_path):
    parquet_path = Path(parquet_path)
    assert parquet_path.exists(), f"File not found: {parquet_path}"

    print(f"Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    assert "label" in df.columns, f"No 'label' column! Found: {list(df.columns)}"
    assert "image" in df.columns, f"No 'image' column! Found: {list(df.columns)}"

    print(f"\nLabel range: min={df['label'].min()}, max={df['label'].max()}")
    print(f"Unique labels in this shard: {df['label'].nunique()}")
    assert df["label"].min() >= 0 and df["label"].max() <= 999, (
        "Labels outside expected 0..999 range!"
    )

    # Try decoding the first image
    first_img_field = df.iloc[0]["image"]
    print(f"\nFirst row 'image' field type: {type(first_img_field)}")
    if isinstance(first_img_field, dict):
        print(f"  keys: {list(first_img_field.keys())}")
        img_bytes = first_img_field["bytes"]
    else:
        img_bytes = first_img_field

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"  Decoded image size: {img.size}, mode: {img.mode}")
    print(f"  First row label: {df.iloc[0]['label']}")

    print("\nFORMAT CHECK PASSED. This file is compatible with "
          "ImageNetValParquetDataset in 02_run_inference.py.")
    print(f"\nThis shard alone has {len(df)} rows. You'll need all shards "
          f"to total exactly 50000 validation rows before running full inference.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 check_parquet_format.py <path_to_one_parquet_file>")
        sys.exit(1)
    main(sys.argv[1])