"""
Prepare the LAG dataset for use with RG-ICL.

Downloads/LAG layout:
  train/      -> g.XXXX.jpg (glaucoma), ng.XXXX.jpg (non_glaucoma)
  validation/ -> same
  test/       -> same

RG-ICL expected layout:
  data/lag/reference/glaucoma/
  data/lag/reference/non_glaucoma/
  data/lag/test/glaucoma/
  data/lag/test/non_glaucoma/

Strategy: create symlinks (no disk copy).
train + validation -> reference
test               -> test
"""

import json
import os
import sys
from pathlib import Path

LAG_SRC = Path.home() / "Downloads" / "LAG"
LAG_DST = Path(__file__).resolve().parent / "data" / "lag"

SPLITS = {
    "reference": ["train", "validation"],
    "test": ["test"],
}

LABEL_MAP = {
    "g.": "glaucoma",
    "ng.": "non_glaucoma",
}


def get_label(filename: str):
    for prefix, label in LABEL_MAP.items():
        if filename.startswith(prefix):
            return label
    return None


def prepare():
    if not LAG_SRC.exists():
        print(f"ERROR: LAG source not found at {LAG_SRC}")
        sys.exit(1)

    counts = {"reference": {"glaucoma": 0, "non_glaucoma": 0},
              "test": {"glaucoma": 0, "non_glaucoma": 0}}

    for dst_split, src_splits in SPLITS.items():
        for label in ["glaucoma", "non_glaucoma"]:
            dst_dir = LAG_DST / dst_split / label
            dst_dir.mkdir(parents=True, exist_ok=True)

        for src_split_name in src_splits:
            src_dir = LAG_SRC / src_split_name
            if not src_dir.exists():
                print(f"WARNING: source dir not found: {src_dir}")
                continue

            for fname in sorted(src_dir.iterdir()):
                if fname.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = get_label(fname.name)
                if label is None:
                    continue

                dst_path = LAG_DST / dst_split / label / fname.name
                if not dst_path.exists():
                    os.symlink(fname.resolve(), dst_path)
                counts[dst_split][label] += 1

    # Generate manifest.json
    LABEL_IDX = {"glaucoma": 1, "non_glaucoma": 0}
    samples = []
    for dst_split in ["reference", "test"]:
        for label_name in ["non_glaucoma", "glaucoma"]:
            d = LAG_DST / dst_split / label_name
            if not d.exists():
                continue
            for fpath in sorted(d.iterdir()):
                if fpath.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                rel = fpath.relative_to(LAG_DST)
                samples.append({
                    "id": f"lag_{dst_split}_{fpath.stem}",
                    "image_path": str(rel),
                    "split": dst_split,
                    "label": LABEL_IDX[label_name],
                    "metadata": {},
                })

    manifest = {"name": "lag", "task_type": "classification", "n_samples": len(samples), "samples": samples}
    manifest_path = LAG_DST / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("LAG data prepared:")
    for split, lc in counts.items():
        for label, n in lc.items():
            print(f"  data/lag/{split}/{label}: {n} images")
    print(f"  manifest.json: {len(samples)} total samples")


if __name__ == "__main__":
    prepare()
