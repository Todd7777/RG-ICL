import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset, CLASSIFICATION_DATASETS, VQA_DATASETS
from encoders import get_encoder


def extract_for_dataset(dataset_name, data_root, encoder_name, encoder_cfg, output_dir):
    ds_ref = get_dataset(dataset_name, data_root, split="reference")
    ds_test = get_dataset(dataset_name, data_root, split="test")
    all_samples = ds_ref.samples + ds_test.samples

    if len(all_samples) == 0:
        print(f"No samples found for {dataset_name}, skipping.")
        return

    encoder = get_encoder(
        name=encoder_name,
        model_id=encoder_cfg.get("model_id", None),
        device=encoder_cfg.get("device", "cuda"),
        image_size=encoder_cfg.get("image_size", 518),
    )

    out_path = Path(output_dir) / dataset_name / encoder_name
    out_path.mkdir(parents=True, exist_ok=True)

    ids = []
    global_embeddings = []
    spatial_features = []
    labels = []
    splits = []

    for sample in tqdm(all_samples, desc=f"{dataset_name}/{encoder_name}"):
        try:
            img = sample.load_image()
            output = encoder.encode_image(img)
            ids.append(sample.id)
            global_embeddings.append(output.global_embedding)
            if output.spatial_features is not None:
                spatial_features.append(output.spatial_features)
            if hasattr(sample, 'label'):
                labels.append(sample.label)
            else:
                labels.append(-1)
            splits.append(sample.split)
        except Exception as e:
            print(f"Error processing {sample.id}: {e}")
            continue

    global_embeddings = np.array(global_embeddings)
    np.save(out_path / "global_embeddings.npy", global_embeddings)

    if spatial_features:
        np.savez_compressed(
            out_path / "spatial_features.npz",
            **{str(i): sf for i, sf in enumerate(spatial_features)}
        )

    metadata = {
        "ids": ids,
        "labels": labels,
        "splits": splits,
        "encoder_name": encoder_name,
        "encoder_version": encoder.encoder_version if hasattr(encoder, 'encoder_version') else "",
        "preprocessing_hash": encoder.preprocessing_hash() if hasattr(encoder, 'preprocessing_hash') else "",
        "n_samples": len(ids),
        "embedding_dim": int(global_embeddings.shape[1]) if len(global_embeddings.shape) > 1 else 0,
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(ids)} embeddings to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--encoders", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_names = args.datasets or cfg.datasets
    encoder_names = args.encoders or [cfg.encoder.name]

    encoder_configs = {
        "dinov3": {"model_id": "facebook/dinov2-large", "image_size": 518, "device": cfg.encoder.device},
        "clip": {"model_id": "openai/clip-vit-large-patch14", "image_size": 224, "device": cfg.encoder.device},
        "mae": {"model_id": "facebook/vit-mae-large", "image_size": 224, "device": cfg.encoder.device},
    }

    output_dir = Path(cfg.features_root) if hasattr(cfg, 'features_root') else Path(cfg.output_root) / "features"

    for ds_name in dataset_names:
        for enc_name in encoder_names:
            enc_cfg = encoder_configs.get(enc_name, {"device": cfg.encoder.device})
            extract_for_dataset(ds_name, cfg.data_root, enc_name, enc_cfg, output_dir)


if __name__ == "__main__":
    main()
