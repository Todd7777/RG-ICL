import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, save_config
from datasets import get_dataset, CLASSIFICATION_DATASETS
from encoders import get_encoder
from retrieval import get_retriever
from prompting import get_prompter
from inference import MLLMClient, OutputParser
from metrics import ClassificationMetrics


def load_features(output_root, dataset_name, encoder_name):
    feat_path = Path(output_root) / "features" / dataset_name / encoder_name

    with open(feat_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    global_emb = np.load(feat_path / "global_embeddings.npy")

    spatial_feats = None
    spatial_path = feat_path / "spatial_features.npz"
    if spatial_path.exists():
        spatial_data = np.load(spatial_path)
        spatial_feats = [spatial_data[str(i)] for i in range(len(metadata["ids"]))]

    return metadata, global_emb, spatial_feats


def run_zero_shot(cfg, dataset, client, parser, output_dir):
    prompter = get_prompter("zero_shot")
    test_samples = dataset.get_test_samples()
    results = []

    for sample in tqdm(test_samples, desc="zero_shot"):
        prompt_record = prompter.build_classification_prompt(
            query_sample=sample,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=dataset.name,
        )
        inference_record = client.infer(
            messages=prompt_record.messages,
            query_id=sample.id,
            method="zero_shot",
        )
        parsed = parser.parse_classification(
            raw_response=inference_record.raw_response,
            query_id=sample.id,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
        )
        results.append({
            "query_id": sample.id,
            "ground_truth_label": sample.label,
            "ground_truth_name": sample.label_name,
            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
            "prompt": prompt_record.to_dict(),
            "inference": inference_record.to_dict(),
            "parsed": parsed.to_dict(),
        })

    save_results(results, output_dir / "zero_shot")
    return results


def run_naive_icl(cfg, dataset, client, parser, output_dir, k=6):
    prompter = get_prompter("naive_icl", k=k, seed=cfg.seed)
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()
    results = []

    for sample in tqdm(test_samples, desc=f"naive_icl_k{k}"):
        prompt_record = prompter.build_classification_prompt(
            query_sample=sample,
            reference_pool=ref_pool,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=dataset.name,
            k=k,
        )
        inference_record = client.infer(
            messages=prompt_record.messages,
            query_id=sample.id,
            method="naive_icl",
        )
        parsed = parser.parse_classification(
            raw_response=inference_record.raw_response,
            query_id=sample.id,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
        )
        results.append({
            "query_id": sample.id,
            "ground_truth_label": sample.label,
            "ground_truth_name": sample.label_name,
            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
            "prompt": prompt_record.to_dict(),
            "inference": inference_record.to_dict(),
            "parsed": parsed.to_dict(),
        })

    save_results(results, output_dir / f"naive_icl_k{k}")
    return results


def run_rg_icl(cfg, dataset, client, parser, output_dir, metadata, global_emb,
               spatial_feats, method="rg_icl_global_spatial", k=6):
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()

    id_to_sample = {}
    for s in ref_pool + test_samples:
        id_to_sample[s.id] = s

    ids = metadata["ids"]
    labels = metadata["labels"]
    splits = metadata["splits"]

    if method == "rg_icl_global":
        retriever = get_retriever("global",
                                   similarity_metric=cfg.retrieval.similarity_metric,
                                   exclude_query=cfg.retrieval.exclude_query,
                                   exclude_test_set=cfg.retrieval.exclude_test_set)
        retriever.build_index(ids, global_emb, labels, splits)
    else:
        retriever = get_retriever("global_spatial",
                                   alpha=cfg.retrieval.alpha,
                                   similarity_metric=cfg.retrieval.similarity_metric,
                                   exclude_query=cfg.retrieval.exclude_query,
                                   exclude_test_set=cfg.retrieval.exclude_test_set)
        retriever.build_index(ids, global_emb, spatial_feats, labels, splits)

    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    prompter = get_prompter("rg_icl_global_spatial", k=k)
    results = []

    for sample in tqdm(test_samples, desc=method):
        idx = id_to_idx.get(sample.id)
        if idx is None:
            continue

        if method == "rg_icl_global":
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_embedding=global_emb[idx],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )
        else:
            retrieval_result = retriever.retrieve(
                query_id=sample.id,
                query_global=global_emb[idx],
                query_spatial=spatial_feats[idx],
                k=k,
                encoder_name=metadata.get("encoder_name", ""),
                encoder_version=metadata.get("encoder_version", ""),
                preprocessing_hash=metadata.get("preprocessing_hash", ""),
            )

        retrieved_refs = [id_to_sample[rid] for rid in retrieval_result.neighbor_ids
                          if rid in id_to_sample]

        prompt_record = prompter.build_classification_prompt(
            query_sample=sample,
            retrieved_refs=retrieved_refs,
            retrieval_result=retrieval_result,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
            dataset_name=dataset.name,
        )
        inference_record = client.infer(
            messages=prompt_record.messages,
            query_id=sample.id,
            method=method,
        )
        parsed = parser.parse_classification(
            raw_response=inference_record.raw_response,
            query_id=sample.id,
            label_names=dataset.label_names,
            is_multi_label=dataset.is_multi_label,
        )
        results.append({
            "query_id": sample.id,
            "ground_truth_label": sample.label,
            "ground_truth_name": sample.label_name,
            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
            "retrieval": {
                "neighbor_ids": retrieval_result.neighbor_ids,
                "neighbor_scores": retrieval_result.neighbor_scores,
                "neighbor_labels": retrieval_result.neighbor_labels,
                "query_embedding_hash": retrieval_result.query_embedding_hash,
                "encoder_name": retrieval_result.encoder_name,
                "encoder_version": retrieval_result.encoder_version,
                "preprocessing_hash": retrieval_result.preprocessing_hash,
            },
            "prompt": prompt_record.to_dict(),
            "inference": inference_record.to_dict(),
            "parsed": parsed.to_dict(),
        })

    save_results(results, output_dir / f"{method}_k{k}")
    return results


def compute_metrics(results, dataset, method_name, metrics_engine):
    if dataset.is_multi_label:
        y_true = np.array([r["ground_truth_multi_label"] for r in results])
        y_pred = np.array([r["parsed"]["multi_label_predictions"] for r in results])
        y_prob = np.array([r["parsed"]["multi_label_confidences"] for r in results])
        return metrics_engine.compute_multilabel(y_true, y_pred, y_prob, dataset.n_classes)
    elif dataset.n_classes == 2:
        y_true = np.array([r["ground_truth_label"] for r in results])
        y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in results])
        y_prob = np.array([r["parsed"]["confidence"] for r in results])
        return metrics_engine.compute_binary(y_true, y_pred, y_prob)
    else:
        y_true = np.array([r["ground_truth_label"] for r in results])
        y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in results])
        n_classes = dataset.n_classes
        y_prob = np.zeros((len(results), n_classes))
        for i, r in enumerate(results):
            pred_idx = r["parsed"]["predicted_label_idx"]
            conf = r["parsed"]["confidence"]
            if 0 <= pred_idx < n_classes:
                y_prob[i, pred_idx] = conf
                remaining = (1.0 - conf) / max(n_classes - 1, 1)
                for j in range(n_classes):
                    if j != pred_idx:
                        y_prob[i, j] = remaining
        return metrics_engine.compute_multiclass(y_true, y_pred, y_prob, n_classes)


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments/classification.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_names = args.datasets or cfg.datasets
    methods = args.methods or cfg.methods
    k = args.k or cfg.retrieval.k

    client = MLLMClient(
        model=cfg.inference.model,
        temperature=cfg.inference.temperature,
        max_tokens=cfg.inference.max_tokens,
        seed=cfg.inference.seed,
        top_p=cfg.inference.top_p,
        api_key_env=cfg.inference.api_key_env,
    )
    output_parser = OutputParser()
    metrics_engine = ClassificationMetrics()

    all_metrics = {}

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = get_dataset(ds_name, cfg.data_root, split="all")
        output_dir = Path(cfg.output_root) / ds_name

        ds_metrics = {}

        need_features = any(m.startswith("rg_icl") for m in methods)
        metadata, global_emb, spatial_feats = None, None, None
        if need_features:
            metadata, global_emb, spatial_feats = load_features(
                cfg.output_root, ds_name, cfg.encoder.name
            )

        if "zero_shot" in methods:
            results = run_zero_shot(cfg, dataset, client, output_parser, output_dir)
            m = compute_metrics(results, dataset, "zero_shot", metrics_engine)
            ds_metrics["zero_shot"] = m.to_dict()

        if "naive_icl" in methods:
            results = run_naive_icl(cfg, dataset, client, output_parser, output_dir, k=k)
            m = compute_metrics(results, dataset, "naive_icl", metrics_engine)
            ds_metrics["naive_icl"] = m.to_dict()

        if "rg_icl_global" in methods:
            results = run_rg_icl(cfg, dataset, client, output_parser, output_dir,
                                  metadata, global_emb, spatial_feats,
                                  method="rg_icl_global", k=k)
            m = compute_metrics(results, dataset, "rg_icl_global", metrics_engine)
            ds_metrics["rg_icl_global"] = m.to_dict()

        if "rg_icl_global_spatial" in methods:
            results = run_rg_icl(cfg, dataset, client, output_parser, output_dir,
                                  metadata, global_emb, spatial_feats,
                                  method="rg_icl_global_spatial", k=k)
            m = compute_metrics(results, dataset, "rg_icl_global_spatial", metrics_engine)
            ds_metrics["rg_icl_global_spatial"] = m.to_dict()

        all_metrics[ds_name] = ds_metrics

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(ds_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    summary_path = Path(cfg.output_root) / "classification_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll classification metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
