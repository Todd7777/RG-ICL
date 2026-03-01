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
from robustness import ImbalanceExperiment, OrderingExperiment, LabelInconsistencyExperiment


def load_features(features_root, dataset_name, encoder_name):
    feat_path = Path(features_root) / dataset_name / encoder_name
    with open(feat_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    global_emb = np.load(feat_path / "global_embeddings.npy")
    spatial_feats = None
    spatial_path = feat_path / "spatial_features.npz"
    if spatial_path.exists():
        spatial_data = np.load(spatial_path)
        spatial_feats = [spatial_data[str(i)] for i in range(len(metadata["ids"]))]
    return metadata, global_emb, spatial_feats


def run_imbalance_experiment(cfg, dataset, client, parser, output_dir, metadata,
                              global_emb, spatial_feats):
    imbalance_exp = ImbalanceExperiment(k=cfg.retrieval.k, seed=cfg.seed)
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()
    metrics_engine = ClassificationMetrics()

    ratios = cfg.robustness.imbalance_ratios
    all_results = {}

    for method in cfg.methods:
        method_results = {}

        for ratio in ratios:
            ratio_name = f"{ratio['neg']}N:{ratio['pos']}P"
            print(f"  Imbalance: {method} {ratio_name}")

            imbalanced_refs = imbalance_exp.construct_imbalanced_set(
                ref_pool, neg_ratio=ratio["neg"], pos_ratio=ratio["pos"])

            results = []
            for sample in tqdm(test_samples, desc=f"{method}_{ratio_name}"):
                if method == "naive_icl":
                    prompter = get_prompter("naive_icl", k=len(imbalanced_refs), seed=cfg.seed)
                    prompt_record = prompter.build_classification_prompt(
                        query_sample=sample,
                        reference_pool=imbalanced_refs,
                        label_names=dataset.label_names,
                        is_multi_label=dataset.is_multi_label,
                        dataset_name=dataset.name,
                        k=len(imbalanced_refs),
                    )
                else:
                    prompter = get_prompter("rg_icl_global_spatial", k=len(imbalanced_refs))
                    prompt_record = prompter.build_classification_prompt(
                        query_sample=sample,
                        retrieved_refs=imbalanced_refs,
                        label_names=dataset.label_names,
                        is_multi_label=dataset.is_multi_label,
                        dataset_name=dataset.name,
                    )

                inference_record = client.infer(
                    messages=prompt_record.messages,
                    query_id=sample.id,
                    method=f"{method}_{ratio_name}",
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
                    "reference_ids": [r.id for r in imbalanced_refs],
                    "reference_labels": [r.label_name for r in imbalanced_refs],
                    "inference": inference_record.to_dict(),
                    "parsed": parsed.to_dict(),
                })

            pred_path = output_dir / "imbalance" / f"{method}_{ratio_name}_predictions.json"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            if dataset.is_multi_label:
                y_true = np.array([r["ground_truth_multi_label"] for r in results])
                y_pred = np.array([r["parsed"]["multi_label_predictions"] for r in results])
                y_prob = np.array([r["parsed"]["multi_label_confidences"] for r in results])
                m = metrics_engine.compute_multilabel(y_true, y_pred, y_prob, dataset.n_classes)
            elif dataset.n_classes == 2:
                y_true = np.array([r["ground_truth_label"] for r in results])
                y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in results])
                y_prob = np.array([r["parsed"]["confidence"] for r in results])
                m = metrics_engine.compute_binary(y_true, y_pred, y_prob)
            else:
                y_true = np.array([r["ground_truth_label"] for r in results])
                y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in results])
                y_prob = np.array([r["parsed"]["confidence"] for r in results])
                m = metrics_engine.compute_binary(y_true, y_pred, y_prob)

            method_results[ratio_name] = m.to_dict()

        all_results[method] = method_results

    return all_results


def run_ordering_experiment(cfg, dataset, client, parser, output_dir, metadata,
                             global_emb, spatial_feats):
    ordering_exp = OrderingExperiment(
        n_permutations=cfg.robustness.ordering_permutations,
        seed=cfg.robustness.ordering_seed,
    )
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()
    metrics_engine = ClassificationMetrics()

    ids = metadata["ids"]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_sample = {}
    for s in ref_pool + test_samples:
        id_to_sample[s.id] = s
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    all_results = {}

    for method in cfg.methods:
        if method == "naive_icl":
            base_refs = ref_pool[:cfg.retrieval.k]
        else:
            retriever = get_retriever("global_spatial",
                                       alpha=cfg.retrieval.alpha,
                                       similarity_metric=cfg.retrieval.similarity_metric,
                                       exclude_query=True, exclude_test_set=True)
            retriever.build_index(ids, global_emb, spatial_feats, labels, splits)

            first_test = test_samples[0]
            first_idx = id_to_idx.get(first_test.id)
            if first_idx is not None:
                ret = retriever.retrieve(
                    query_id=first_test.id,
                    query_global=global_emb[first_idx],
                    query_spatial=spatial_feats[first_idx],
                    k=cfg.retrieval.k,
                )
                base_refs = [id_to_sample[rid] for rid in ret.neighbor_ids if rid in id_to_sample]
            else:
                base_refs = ref_pool[:cfg.retrieval.k]

        ref_ids_list = [r.id for r in base_refs]
        permutations = ordering_exp.generate_permutations(ref_ids_list)
        perm_aucs = []

        for perm_idx, perm in enumerate(permutations):
            print(f"  Ordering: {method} perm {perm_idx+1}/{len(permutations)}")
            reordered_refs = ordering_exp.reorder_references(base_refs, perm)

            results = []
            for sample in tqdm(test_samples, desc=f"perm_{perm_idx}"):
                prompter = get_prompter("rg_icl_global_spatial", k=len(reordered_refs))
                prompt_record = prompter.build_classification_prompt(
                    query_sample=sample,
                    retrieved_refs=reordered_refs,
                    label_names=dataset.label_names,
                    is_multi_label=dataset.is_multi_label,
                    dataset_name=dataset.name,
                )
                inference_record = client.infer(
                    messages=prompt_record.messages,
                    query_id=sample.id,
                    method=f"{method}_perm{perm_idx}",
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
                    "permutation_idx": perm_idx,
                    "permutation": perm,
                    "reference_ids": [r.id for r in reordered_refs],
                    "inference": inference_record.to_dict(),
                    "parsed": parsed.to_dict(),
                })

            pred_path = output_dir / "ordering" / f"{method}_perm{perm_idx}_predictions.json"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            y_true = np.array([r["ground_truth_label"] for r in results])
            y_pred = np.array([r["parsed"]["predicted_label_idx"] for r in results])
            y_prob = np.array([r["parsed"]["confidence"] for r in results])
            m = metrics_engine.compute_binary(y_true, y_pred, y_prob)
            perm_aucs.append(m.auc)

        summary = ordering_exp.summarize(
            [type('R', (), {"auc": a})() for a in perm_aucs],
            method=method,
        )
        all_results[method] = summary.to_dict()

    return all_results


def run_label_inconsistency_experiment(cfg, dataset, client, parser, output_dir,
                                        metadata, global_emb, spatial_feats):
    inconsistency_exp = LabelInconsistencyExperiment(seed=cfg.seed)
    ref_pool = dataset.get_reference_pool()
    test_samples = dataset.get_test_samples()
    metrics_engine = ClassificationMetrics()

    ids = metadata["ids"]
    labels = metadata["labels"]
    splits = metadata["splits"]
    id_to_sample = {}
    for s in ref_pool + test_samples:
        id_to_sample[s.id] = s
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    global_emb_dict = {}
    for i, sid in enumerate(ids):
        global_emb_dict[sid] = global_emb[i]

    all_results = {}

    for method in cfg.methods:
        if method == "naive_icl":
            rng = np.random.RandomState(cfg.seed)
            idx = rng.choice(len(ref_pool), size=cfg.retrieval.k, replace=False)
            base_refs = [ref_pool[i] for i in idx]
        else:
            retriever = get_retriever("global_spatial",
                                       alpha=cfg.retrieval.alpha,
                                       similarity_metric=cfg.retrieval.similarity_metric,
                                       exclude_query=True, exclude_test_set=True)
            retriever.build_index(ids, global_emb, spatial_feats, labels, splits)
            first_test = test_samples[0]
            first_idx = id_to_idx.get(first_test.id)
            if first_idx is not None:
                ret = retriever.retrieve(
                    query_id=first_test.id,
                    query_global=global_emb[first_idx],
                    query_spatial=spatial_feats[first_idx],
                    k=cfg.retrieval.k,
                )
                base_refs = [id_to_sample[rid] for rid in ret.neighbor_ids if rid in id_to_sample]
            else:
                base_refs = ref_pool[:cfg.retrieval.k]

        y_true_base = np.array([s.label for s in test_samples])
        base_results = _run_with_refs(test_samples, base_refs, dataset, client, parser, method)
        y_prob_base = np.array([r["parsed"]["confidence"] for r in base_results])
        y_pred_base = np.array([r["parsed"]["predicted_label_idx"] for r in base_results])
        m_base = metrics_engine.compute_binary(y_true_base, y_pred_base, y_prob_base)
        auc_before = m_base.auc

        injection_results = inconsistency_exp.run_stress_test(
            base_refs, ref_pool, global_emb_dict)

        method_inj_results = []
        for inj in injection_results:
            modified_refs = inj["modified_references"]
            inj_results = _run_with_refs(test_samples, modified_refs, dataset, client, parser, method)
            y_prob_inj = np.array([r["parsed"]["confidence"] for r in inj_results])
            y_pred_inj = np.array([r["parsed"]["predicted_label_idx"] for r in inj_results])
            m_inj = metrics_engine.compute_binary(y_true_base, y_pred_inj, y_prob_inj)
            auc_after = m_inj.auc

            method_inj_results.append({
                "position": inj["position"],
                "original_id": inj["original_id"],
                "original_label": inj["original_label"],
                "replacement_id": inj["replacement_id"],
                "replacement_label": inj["replacement_label"],
                "similarity": inj["similarity"],
                "auc_before": auc_before,
                "auc_after": auc_after,
                "auc_drop": auc_before - auc_after,
            })

        all_results[method] = method_inj_results

    out_path = output_dir / "label_inconsistency"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def _run_with_refs(test_samples, refs, dataset, client, parser, method):
    prompter = get_prompter("rg_icl_global_spatial", k=len(refs))
    results = []
    for sample in test_samples:
        prompt_record = prompter.build_classification_prompt(
            query_sample=sample,
            retrieved_refs=refs,
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
            "inference": inference_record.to_dict(),
            "parsed": parsed.to_dict(),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/experiments/robustness.yaml")
    ap.add_argument("--experiments", nargs="+",
                    default=["imbalance", "ordering", "label_inconsistency"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    client = MLLMClient(
        model=cfg.inference.model,
        temperature=cfg.inference.temperature,
        max_tokens=cfg.inference.max_tokens,
        seed=cfg.inference.seed,
        top_p=cfg.inference.top_p,
        api_key_env=cfg.inference.api_key_env,
    )
    output_parser = OutputParser()

    for ds_name in cfg.datasets:
        print(f"\n{'='*60}")
        print(f"Robustness: {ds_name}")
        print(f"{'='*60}")

        dataset = get_dataset(ds_name, cfg.data_root, split="all")
        output_dir = Path(cfg.output_root) / ds_name
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata, global_emb, spatial_feats = load_features(
            cfg.features_root, ds_name, cfg.encoder.name)

        robustness_summary = {}

        if "imbalance" in args.experiments:
            print("\n--- Imbalance Experiment ---")
            imb = run_imbalance_experiment(
                cfg, dataset, client, output_parser, output_dir,
                metadata, global_emb, spatial_feats)
            robustness_summary["imbalance"] = imb

        if "ordering" in args.experiments:
            print("\n--- Ordering Experiment ---")
            ord_res = run_ordering_experiment(
                cfg, dataset, client, output_parser, output_dir,
                metadata, global_emb, spatial_feats)
            robustness_summary["ordering"] = ord_res

        if "label_inconsistency" in args.experiments:
            print("\n--- Label Inconsistency Experiment ---")
            li = run_label_inconsistency_experiment(
                cfg, dataset, client, output_parser, output_dir,
                metadata, global_emb, spatial_feats)
            robustness_summary["label_inconsistency"] = li

        summary_path = output_dir / "robustness_summary.json"
        with open(summary_path, "w") as f:
            json.dump(robustness_summary, f, indent=2, default=str)
        print(f"Robustness summary saved to {summary_path}")


if __name__ == "__main__":
    main()
