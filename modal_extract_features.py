"""
Modal GPU-accelerated feature extraction for the LAG dataset using DINOv2-large.

Uploads local LAG images to Modal, runs DINOv2 on an A10G GPU,
and downloads global_embeddings.npy + spatial_features.npz + metadata.json
into outputs/features/lag/dinov3/.

Usage:
    modal run modal_extract_features.py
"""

import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

app = modal.App("rg-icl-lag-extract")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.0",
        "Pillow==10.2.0",
        "numpy==1.26.3",
        "tqdm==4.66.1",
        "huggingface-hub==0.22.0",
    )
)

# Volume to cache the DINOv2 model weights between runs
model_cache = modal.Volume.from_name("dinov2-model-cache", create_if_missing=True)

LAG_DATA_DIR = Path(__file__).resolve().parent / "data" / "lag"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "features" / "lag" / "dinov3"

SPLITS = ["reference", "test"]
LABEL_NAMES = ["non_glaucoma", "glaucoma"]
IMAGE_SIZE = 518
BATCH_SIZE = 32
CHUNK_SIZE = 200  # images per Modal call


# ---------------------------------------------------------------------------
# Helper: collect local image paths
# ---------------------------------------------------------------------------

def collect_local_images():
    """Returns list of (abs_path, split, label_idx, sample_id)."""
    samples = []
    for split in SPLITS:
        for label_idx, label_name in enumerate(LABEL_NAMES):
            d = LAG_DATA_DIR / split / label_name
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    sid = f"lag_{split}_{p.stem}"
                    samples.append((str(p.resolve()), split, label_idx, sid))
    return samples


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/model-cache": model_cache},
    memory=16384,
)
def extract_features(image_bytes_list: list, meta_list: list):
    """
    Receives batches of (image_bytes, split, label, sample_id) and returns
    {id, global_embedding, spatial_features, split, label}.
    """
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["HF_HOME"] = "/model-cache"

    device = torch.device("cuda")

    print("Loading DINOv2-large...")
    model = AutoModel.from_pretrained(
        "facebook/dinov2-large",
        cache_dir="/model-cache",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    results = []
    batch_size = BATCH_SIZE

    for start in range(0, len(image_bytes_list), batch_size):
        batch_bytes = image_bytes_list[start:start + batch_size]
        batch_meta = meta_list[start:start + batch_size]

        tensors = []
        valid_meta = []
        for raw_bytes, m in zip(batch_bytes, batch_meta):
            try:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                tensors.append(transform(img))
                valid_meta.append(m)
            except Exception as e:
                print(f"Skip {m['id']}: {e}")

        if not tensors:
            continue

        pixel_values = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=True)

        cls_tokens = outputs.last_hidden_state[:, 0, :].cpu().numpy()     # (B, 1024)
        patch_tokens = outputs.last_hidden_state[:, 1:, :].cpu().numpy()  # (B, N, 1024)

        for i, m in enumerate(valid_meta):
            g = cls_tokens[i]
            norm = np.linalg.norm(g)
            if norm > 0:
                g = g / norm
            results.append({
                "id": m["id"],
                "split": m["split"],
                "label": m["label"],
                "global_embedding": g.tolist(),
                "spatial_features": patch_tokens[i].tolist(),
            })

        if (start // batch_size) % 5 == 0:
            print(f"  Processed {min(start + batch_size, len(image_bytes_list))}"
                  f"/{len(image_bytes_list)}")

    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import numpy as np

    print("Collecting local LAG images...")
    samples = collect_local_images()
    print(f"  Found {len(samples)} images")

    # Read image bytes locally
    image_bytes_list = []
    meta_list = []
    for abs_path, split, label_idx, sid in samples:
        try:
            with open(abs_path, "rb") as f:
                image_bytes_list.append(f.read())
            meta_list.append({"id": sid, "split": split, "label": label_idx})
        except Exception as e:
            print(f"  Cannot read {abs_path}: {e}")

    print(f"  Sending {len(image_bytes_list)} images to Modal GPU (parallel chunks)...")

    # Build chunks
    chunks = []
    for start in range(0, len(image_bytes_list), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(image_bytes_list))
        chunks.append((image_bytes_list[start:end], meta_list[start:end]))

    print(f"  {len(chunks)} chunks of ~{CHUNK_SIZE} images each → running in parallel")

    # Run all chunks in parallel via .map()
    all_chunk_results = list(extract_features.map(
        [c[0] for c in chunks],
        [c[1] for c in chunks],
    ))

    results = []
    for chunk_result in all_chunk_results:
        results.extend(chunk_result)

    print(f"  Received {len(results)} embeddings back")

    # Assemble outputs
    ids, labels, splits_ = [], [], []
    global_embs = []
    spatial_feats = {}

    for i, r in enumerate(results):
        ids.append(r["id"])
        labels.append(r["label"])
        splits_.append(r["split"])
        global_embs.append(r["global_embedding"])
        spatial_feats[str(i)] = np.array(r["spatial_features"], dtype=np.float32)

    global_embs = np.array(global_embs, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "global_embeddings.npy", global_embs)
    np.savez_compressed(OUTPUT_DIR / "spatial_features.npz", **spatial_feats)

    metadata = {
        "ids": ids,
        "labels": labels,
        "splits": splits_,
        "encoder_name": "dinov3",
        "encoder_version": "dinov3-large-v1",
        "preprocessing_hash": "",
        "n_samples": len(ids),
        "embedding_dim": int(global_embs.shape[1]),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved to {OUTPUT_DIR}")
    print(f"  global_embeddings.npy : {global_embs.shape}")
    print(f"  spatial_features.npz  : {len(spatial_feats)} arrays")
    print(f"  metadata.json         : {len(ids)} samples")
