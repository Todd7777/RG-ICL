import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset
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


def run_single_k(cfg, dataset, client, parser, output_dir, metadata, global_emb,
                 spatial_feats, method, k):
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()

    if k == 0:
        prompter = get_prompter("zero_shot")
        results = []
        for sample in tqdm(test_samples, desc=f"{method}_k{k}"):
            prompt_record = prompter.build_classification_prompt(
                query_sample=sample,
                label_names=dataset.label_names,
                is_multi_label=dataset.is_multi_label,
                dataset_name=dataset.name,
            )
            inference_record = client.infer(
                messages=prompt_record.messages,
                query_id=sample.id,
                method=f"{method}_k0",
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
                "inference": inference_record.to_dict(),
                "parsed": parsed.to_dict(),
            })
        return results

    id_to_sample = {}
    for s in ref_pool + test_samples:
        id_to_sample[s.id] = s

    ids = metadata["ids"]
    labels = metadata["labels"]
    splits = metadata["splits"]

    if method == "naive_icl":
        prompter_icl = get_prompter("naive_icl", k=k, seed=cfg.seed)
        results = []
        for sample in tqdm(test_samples, desc=f"naive_icl_k{k}"):
            prompt_record = prompter_icl.build_classification_prompt(
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
                method=f"naive_icl_k{k}",
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
        return results

    if method in ("rg_icl_global", "rg_icl_global_spatial"):
        if method == "rg_icl_global":
            retriever = get_retriever("global",
                                       similarity_metric=cfg.retrieval.similarity_metric,
                                       exclude_query=True, exclude_test_set=True)
            retriever.build_index(ids, global_emb, labels, splits)
        else:
            retriever = get_retriever("global_spatial",
                                       alpha=cfg.retrieval.alpha,
                                       similarity_metric=cfg.retrieval.similarity_metric,
                                       exclude_query=True, exclude_test_set=True)
            retriever.build_index(ids, global_emb, spatial_feats, labels, splits)

        id_to_idx = {sid: i for i, sid in enumerate(ids)}
        prompter = get_prompter("rg_icl_global_spatial", k=k)
        results = []

        for sample in tqdm(test_samples, desc=f"{method}_k{k}"):
            idx = id_to_idx.get(sample.id)
            if idx is None:
                continue

            if method == "rg_icl_global":
                retrieval_result = retriever.retrieve(
                    query_id=sample.id, query_embedding=global_emb[idx], k=k)
            else:
                retrieval_result = retriever.retrieve(
                    query_id=sample.id, query_global=global_emb[idx],
                    query_spatial=spatial_feats[idx], k=k)

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
                method=f"{method}_k{k}",
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
                },
                "prompt": prompt_record.to_dict(),
                "inference": inference_record.to_dict(),
                "parsed": parsed.to_dict(),
            })
        return results

    return []


def compute_metrics_for_results(results, dataset, metrics_engine):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/experiments/k_sweep.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    raw_cfg_path = args.config
    with open(raw_cfg_path, "r") as f:
        import yaml
        raw = yaml.safe_load(f)
    k_values = raw.get("k_values", [0, 1, 2, 4, 6, 8])

    dataset_names = cfg.datasets
    methods = cfg.methods
    encoder_name = cfg.encoder.name

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

    sweep_results = {}

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"K-Sweep: {ds_name}")
        print(f"{'='*60}")

        dataset = get_dataset(ds_name, cfg.data_root, split="all")
        metadata, global_emb, spatial_feats = load_features(
            cfg.output_root, ds_name, encoder_name)

        ds_sweep = {}
        for method in methods:
            method_results = {}
            for k in k_values:
                print(f"  Running {method} k={k}...")
                output_dir = Path(cfg.output_root) / ds_name / "k_sweep"
                output_dir.mkdir(parents=True, exist_ok=True)

                results = run_single_k(
                    cfg, dataset, client, output_parser, output_dir,
                    metadata, global_emb, spatial_feats, method, k)

                pred_path = output_dir / f"{method}_k{k}_predictions.json"
                with open(pred_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)

                m = compute_metrics_for_results(results, dataset, metrics_engine)
                method_results[str(k)] = m.to_dict()
                print(f"    AUC={m.auc:.4f} Acc={m.accuracy:.4f}")

            ds_sweep[method] = method_results
        sweep_results[ds_name] = ds_sweep

    summary_path = Path(cfg.output_root) / "k_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nK-sweep summary saved to {summary_path}")


if __name__ == "__main__":
    main()
