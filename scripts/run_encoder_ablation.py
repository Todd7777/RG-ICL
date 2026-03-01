import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset, CLASSIFICATION_DATASETS, VQA_DATASETS
from retrieval import get_retriever
from prompting import get_prompter
from inference import MLLMClient, OutputParser
from metrics import ClassificationMetrics, VQAMetrics


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


def run_classification_ablation(cfg, raw_cfg, client, parser, output_dir):
    cls_metrics = ClassificationMetrics()
    encoders = raw_cfg.get("encoders", [])
    cls_datasets = raw_cfg.get("classification_datasets", [])
    methods = raw_cfg.get("methods", ["zero_shot", "naive_icl", "rg_icl_global_spatial"])
    k = cfg.retrieval.k

    results = {}

    for ds_name in cls_datasets:
        print(f"\n  Classification ablation: {ds_name}")
        dataset = get_dataset(ds_name, cfg.data_root, split="all")
        ref_pool = dataset.get_reference_pool()
        test_samples = dataset.get_test_samples()

        ds_results = {}

        for enc_info in encoders:
            enc_name = enc_info["name"]
            print(f"    Encoder: {enc_name}")

            try:
                metadata, global_emb, spatial_feats = load_features(
                    cfg.output_root, ds_name, enc_name)
            except FileNotFoundError:
                print(f"    Features not found for {ds_name}/{enc_name}, skipping.")
                continue

            ids = metadata["ids"]
            labels = metadata["labels"]
            splits = metadata["splits"]
            id_to_sample = {}
            for s in ref_pool + test_samples:
                id_to_sample[s.id] = s
            id_to_idx = {sid: i for i, sid in enumerate(ids)}

            enc_results = {}

            for method in methods:
                print(f"      Method: {method}")
                method_preds = []

                if method == "zero_shot":
                    prompter = get_prompter("zero_shot")
                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        pr = prompter.build_classification_prompt(
                            query_sample=sample, label_names=dataset.label_names,
                            is_multi_label=dataset.is_multi_label, dataset_name=ds_name)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_classification(
                            ir.raw_response, sample.id, dataset.label_names, dataset.is_multi_label)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_label": sample.label,
                            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
                            "parsed": parsed.to_dict(),
                        })

                elif method == "naive_icl":
                    prompter = get_prompter("naive_icl", k=k, seed=cfg.seed)
                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        pr = prompter.build_classification_prompt(
                            query_sample=sample, reference_pool=ref_pool,
                            label_names=dataset.label_names,
                            is_multi_label=dataset.is_multi_label, dataset_name=ds_name, k=k)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_classification(
                            ir.raw_response, sample.id, dataset.label_names, dataset.is_multi_label)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_label": sample.label,
                            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
                            "parsed": parsed.to_dict(),
                        })

                elif method == "rg_icl_global_spatial":
                    retriever = get_retriever("global_spatial",
                                               alpha=cfg.retrieval.alpha,
                                               similarity_metric=cfg.retrieval.similarity_metric,
                                               exclude_query=True, exclude_test_set=True)
                    retriever.build_index(ids, global_emb, spatial_feats, labels, splits)
                    prompter = get_prompter("rg_icl_global_spatial", k=k)

                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        idx = id_to_idx.get(sample.id)
                        if idx is None:
                            continue
                        ret = retriever.retrieve(
                            query_id=sample.id, query_global=global_emb[idx],
                            query_spatial=spatial_feats[idx], k=k)
                        refs = [id_to_sample[rid] for rid in ret.neighbor_ids if rid in id_to_sample]
                        pr = prompter.build_classification_prompt(
                            query_sample=sample, retrieved_refs=refs, retrieval_result=ret,
                            label_names=dataset.label_names,
                            is_multi_label=dataset.is_multi_label, dataset_name=ds_name)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_classification(
                            ir.raw_response, sample.id, dataset.label_names, dataset.is_multi_label)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_label": sample.label,
                            "ground_truth_multi_label": getattr(sample, 'multi_label', None),
                            "parsed": parsed.to_dict(),
                        })

                pred_dir = output_dir / ds_name / enc_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                with open(pred_dir / f"{method}_predictions.json", "w") as f:
                    json.dump(method_preds, f, indent=2, default=str)

                if dataset.is_multi_label:
                    y_true = np.array([r["ground_truth_multi_label"] for r in method_preds])
                    y_pred = np.array([r["parsed"]["multi_label_predictions"] for r in method_preds])
                    y_prob = np.array([r["parsed"]["multi_label_confidences"] for r in method_preds])
                    m = cls_metrics.compute_multilabel(y_true, y_pred, y_prob, dataset.n_classes)
                elif dataset.n_classes == 2:
                    y_true = np.array([r["ground_truth_label"] for r in method_preds])
                    y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in method_preds])
                    y_prob = np.array([r["parsed"]["confidence"] for r in method_preds])
                    m = cls_metrics.compute_binary(y_true, y_pred, y_prob)
                else:
                    y_true = np.array([r["ground_truth_label"] for r in method_preds])
                    y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in method_preds])
                    nc = dataset.n_classes
                    y_prob = np.zeros((len(method_preds), nc))
                    for i, r in enumerate(method_preds):
                        pi = r["parsed"]["predicted_label_idx"]
                        c = r["parsed"]["confidence"]
                        if 0 <= pi < nc:
                            y_prob[i, pi] = c
                            rem = (1.0 - c) / max(nc - 1, 1)
                            for j in range(nc):
                                if j != pi:
                                    y_prob[i, j] = rem
                    m = cls_metrics.compute_multiclass(y_true, y_pred, y_prob, nc)

                enc_results[method] = m.to_dict()
                print(f"        AUC={m.auc:.4f}")

            ds_results[enc_name] = enc_results
        results[ds_name] = ds_results

    return results


def run_vqa_ablation(cfg, raw_cfg, client, parser, output_dir):
    vqa_metrics = VQAMetrics()
    encoders = raw_cfg.get("encoders", [])
    vqa_datasets = raw_cfg.get("vqa_datasets", [])
    methods = raw_cfg.get("methods", ["zero_shot", "naive_icl", "rg_icl_global_spatial"])
    k = cfg.retrieval.k

    results = {}

    for ds_name in vqa_datasets:
        print(f"\n  VQA ablation: {ds_name}")
        dataset = get_dataset(ds_name, cfg.data_root, split="all")
        ref_pool = dataset.get_reference_pool()
        test_samples = dataset.get_test_samples()

        ds_results = {}

        for enc_info in encoders:
            enc_name = enc_info["name"]
            print(f"    Encoder: {enc_name}")

            enc_results = {}

            for method in methods:
                print(f"      Method: {method}")
                method_preds = []

                if method == "zero_shot":
                    prompter = get_prompter("zero_shot")
                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        pr = prompter.build_vqa_prompt(query_sample=sample)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_vqa(ir.raw_response, sample.id)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_answer": sample.answer,
                            "parsed": parsed.to_dict(),
                        })

                elif method == "naive_icl":
                    prompter = get_prompter("naive_icl", k=k, seed=cfg.seed)
                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        pr = prompter.build_vqa_prompt(
                            query_sample=sample, reference_pool=ref_pool, k=k)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_vqa(ir.raw_response, sample.id)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_answer": sample.answer,
                            "parsed": parsed.to_dict(),
                        })

                elif method == "rg_icl_global_spatial":
                    try:
                        metadata, global_emb, spatial_feats = load_features(
                            cfg.output_root, ds_name, enc_name)
                    except FileNotFoundError:
                        print(f"    Features not found for {ds_name}/{enc_name}, skipping.")
                        continue

                    ids = metadata["ids"]
                    labels_meta = metadata["labels"]
                    splits_meta = metadata["splits"]
                    id_to_sample = {}
                    for s in ref_pool + test_samples:
                        id_to_sample[s.id] = s
                    id_to_idx = {sid: i for i, sid in enumerate(ids)}

                    retriever = get_retriever("global_spatial",
                                               alpha=cfg.retrieval.alpha,
                                               similarity_metric=cfg.retrieval.similarity_metric,
                                               exclude_query=True, exclude_test_set=True)
                    retriever.build_index(ids, global_emb, spatial_feats, labels_meta, splits_meta)
                    prompter = get_prompter("rg_icl_global_spatial", k=k)

                    for sample in tqdm(test_samples, desc=f"{enc_name}/{method}"):
                        idx = id_to_idx.get(sample.id)
                        if idx is None:
                            continue
                        ret = retriever.retrieve(
                            query_id=sample.id, query_global=global_emb[idx],
                            query_spatial=spatial_feats[idx], k=k)
                        refs = [id_to_sample[rid] for rid in ret.neighbor_ids if rid in id_to_sample]
                        pr = prompter.build_vqa_prompt(
                            query_sample=sample, retrieved_refs=refs, retrieval_result=ret)
                        ir = client.infer(messages=pr.messages, query_id=sample.id, method=method)
                        parsed = parser.parse_vqa(ir.raw_response, sample.id)
                        method_preds.append({
                            "query_id": sample.id,
                            "ground_truth_answer": sample.answer,
                            "parsed": parsed.to_dict(),
                        })

                pred_dir = output_dir / ds_name / enc_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                with open(pred_dir / f"{method}_predictions.json", "w") as f:
                    json.dump(method_preds, f, indent=2, default=str)

                refs_list = [r["ground_truth_answer"] for r in method_preds]
                hyps_list = [r["parsed"]["answer"] for r in method_preds]
                m = vqa_metrics.compute(refs_list, hyps_list)
                enc_results[method] = m.to_dict()
                print(f"        BLEU4={m.bleu4:.4f} ROUGE-L={m.rouge_l:.4f} METEOR={m.meteor:.4f}")

            ds_results[enc_name] = enc_results
        results[ds_name] = ds_results

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/experiments/encoder_ablation.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    client = MLLMClient(
        model=cfg.inference.model,
        temperature=cfg.inference.temperature,
        max_tokens=cfg.inference.max_tokens,
        seed=cfg.inference.seed,
        top_p=cfg.inference.top_p,
        api_key_env=cfg.inference.api_key_env,
    )
    parser = OutputParser()
    output_dir = Path(cfg.output_root)

    print("=" * 60)
    print("Encoder Ablation Study")
    print("=" * 60)

    cls_results = run_classification_ablation(cfg, raw_cfg, client, parser, output_dir)
    vqa_results = run_vqa_ablation(cfg, raw_cfg, client, parser, output_dir)

    summary = {
        "classification": cls_results,
        "vqa": vqa_results,
    }

    summary_path = output_dir / "encoder_ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEncoder ablation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
