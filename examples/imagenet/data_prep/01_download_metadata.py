"""
Download freely available metadata needed for the ImageNet evaluation.

Downloads:
  - imagenet_class_index.json  : ImageNet-1k class index (WNID + label per class)
  - WordNet data via NLTK (downloads to ~/nltk_data/ by default)

Usage
-----
    python data_prep/01_download_metadata.py
"""

import json
import urllib.request
from pathlib import Path

METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

CLASS_INDEX_PATH = METADATA_DIR / "imagenet_class_index.json"


def download_class_index():
    if CLASS_INDEX_PATH.exists():
        print("  imagenet_class_index.json already exists, skipping.")
        return
    print("  Downloading imagenet class index...")
    # JSON format: {"0": ["n01440764", "tench"], ..., "999": [...]}
    url = (
        "https://s3.amazonaws.com/deep-learning-models/image-models/"
        "imagenet_class_index.json"
    )
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {CLASS_INDEX_PATH}")


def validate_class_index():
    with open(CLASS_INDEX_PATH) as f:
        data = json.load(f)
    assert len(data) == 1000, f"Expected 1000 classes, got {len(data)}"
    assert "0" in data and "999" in data
    assert len(data["0"]) == 2
    print("  Validated: 1000 classes, format [wnid, label] per entry.")


def download_wordnet():
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("dog")
        print("  WordNet already downloaded.")
    except LookupError:
        print("  Downloading WordNet via NLTK...")
        nltk.download("wordnet")
        print("  WordNet downloaded.")


if __name__ == "__main__":
    print("\n=== Downloading ImageNet class index ===")
    download_class_index()
    validate_class_index()

    print("\n=== Downloading WordNet ===")
    download_wordnet()

    print("\nAll metadata ready.\n")