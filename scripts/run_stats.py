import argparse
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from stats import PairedBootstrap, DeLongTest
from sklearn.metrics import roc_auc_score


def load_predictions(pred_path):
    with open(pred_path, "r") as f:
        return json.load(f)


def extract_binary_arrays(preds, is_multi_label=False):
    if is_multi_label:
        y_true = np.array([r["ground_truth_multi_label"] for r in preds])
        y_prob = np.array([r["parsed"]["multi_label_confidences"] for r in preds])
        return y_true, y_prob
    y_true = np.array([r["ground_truth_label"] for r in preds])
    y_prob = np.array([r["parsed"]["confidence"] for r in preds])
    return y_true, y_prob


def run_pairwise_stats(preds_a, preds_b, method_a_name, method_b_name,
                        bootstrap, delong, is_multi_label=False):
    id_to_a = {r["query_id"]: r for r in preds_a}
    id_to_b = {r["query_id"]: r for r in preds_b}
    common_ids = sorted(set(id_to_a.keys()) & set(id_to_b.keys()))

    paired_a = [id_to_a[qid] for qid in common_ids]
    paired_b = [id_to_b[qid] for qid in common_ids]

    if is_multi_label:
        y_true = np.array([r["ground_truth_multi_label"] for r in paired_a])
        scores_a = np.array([r["parsed"]["multi_label_confidences"] for r in paired_a])
        scores_b = np.array([r["parsed"]["multi_label_confidences"] for r in paired_b])

        n_labels = y_true.shape[1]
        bootstrap_results = []
        delong_results = []

        for c in range(n_labels):
            def auc_fn(y, s):
                if len(np.unique(y)) < 2:
                    return 0.5
                return roc_auc_score(y, s)

            br = bootstrap.test(
                y_true[:, c], scores_a[:, c], scores_b[:, c],
                metric_fn=auc_fn, metric_name=f"auc_label_{c}",
                method_a_name=method_a_name, method_b_name=method_b_name,
            )
            bootstrap_results.append(br.to_dict())

            dl = delong.test(
                y_true[:, c], scores_a[:, c], scores_b[:, c],
                method_a_name=method_a_name, method_b_name=method_b_name,
            )
            delong_results.append(dl.to_dict())

        return {
            "n_paired_samples": len(common_ids),
            "bootstrap": bootstrap_results,
            "delong": delong_results,
        }

    y_true = np.array([r["ground_truth_label"] for r in paired_a])
    scores_a = np.array([r["parsed"]["confidence"] for r in paired_a])
    scores_b = np.array([r["parsed"]["confidence"] for r in paired_b])

    boot_auc = bootstrap.test_auc(
        y_true, scores_a, scores_b,
        method_a_name=method_a_name, method_b_name=method_b_name,
    )

    preds_a_labels = np.array([r["parsed"]["predicted_label_idx"] for r in paired_a])
    preds_b_labels = np.array([r["parsed"]["predicted_label_idx"] for r in paired_b])
    boot_acc = bootstrap.test_accuracy(
        y_true, preds_a_labels, preds_b_labels,
        method_a_name=method_a_name, method_b_name=method_b_name,
    )

    def brier_fn(y, s):
        return float(np.mean((s - y) ** 2))

    boot_brier = bootstrap.test(
        y_true, scores_a, scores_b,
        metric_fn=brier_fn, metric_name="brier",
        method_a_name=method_a_name, method_b_name=method_b_name,
    )

    dl = delong.test(
        y_true, scores_a, scores_b,
        method_a_name=method_a_name, method_b_name=method_b_name,
    )

    return {
        "n_paired_samples": len(common_ids),
        "bootstrap_auc": boot_auc.to_dict(),
        "bootstrap_accuracy": boot_acc.to_dict(),
        "bootstrap_brier": boot_brier.to_dict(),
        "delong": dl.to_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--pred-root", type=str, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--task", type=str, choices=["classification", "vqa", "both"], default="both")
    args = ap.parse_args()

    cfg = load_config(args.config)

    bootstrap = PairedBootstrap(
        n_resamples=cfg.metrics.bootstrap_n,
        seed=cfg.metrics.bootstrap_seed,
        ci_level=cfg.metrics.ci_level,
        store_indices=True,
    )
    delong = DeLongTest()

    cls_datasets = ["lag", "ddr", "chexpert", "breakhis"]
    vqa_datasets = ["medical_cxr_vqa", "vqa_rad", "pathvqa", "pmc_vqa"]

    if args.datasets:
        cls_datasets = [d for d in args.datasets if d in cls_datasets]
        vqa_datasets = [d for d in args.datasets if d in vqa_datasets]

    method_pairs = [
        ("zero_shot", "naive_icl"),
        ("zero_shot", "rg_icl_global"),
        ("zero_shot", "rg_icl_global_spatial"),
        ("naive_icl", "rg_icl_global"),
        ("naive_icl", "rg_icl_global_spatial"),
        ("rg_icl_global", "rg_icl_global_spatial"),
    ]

    all_stats = {}

    if args.task in ("classification", "both"):
        cls_root = Path(args.pred_root or (cfg.output_root + "/classification"))

        for ds_name in cls_datasets:
            print(f"\n{'='*60}")
            print(f"Stats: {ds_name}")
            print(f"{'='*60}")

            is_ml = ds_name == "chexpert"
            ds_stats = {}

            for method_a, method_b in method_pairs:
                path_a = cls_root / ds_name / f"{method_a}_k6" / "predictions.json"
                if not path_a.exists():
                    path_a = cls_root / ds_name / method_a / "predictions.json"
                path_b = cls_root / ds_name / f"{method_b}_k6" / "predictions.json"
                if not path_b.exists():
                    path_b = cls_root / ds_name / method_b / "predictions.json"

                if not path_a.exists() or not path_b.exists():
                    print(f"  Skipping {method_a} vs {method_b}: predictions not found")
                    continue

                print(f"  {method_a} vs {method_b}...")
                preds_a = load_predictions(path_a)
                preds_b = load_predictions(path_b)

                result = run_pairwise_stats(
                    preds_a, preds_b, method_a, method_b,
                    bootstrap, delong, is_multi_label=is_ml,
                )
                comp_name = f"{method_a}_vs_{method_b}"
                ds_stats[comp_name] = result

            all_stats[ds_name] = ds_stats

    if args.task in ("vqa", "both"):
        vqa_root = Path(args.pred_root or (cfg.output_root + "/vqa"))

        for ds_name in vqa_datasets:
            print(f"\n{'='*60}")
            print(f"VQA Stats: {ds_name}")
            print(f"{'='*60}")

            ds_stats = {}

            for method_a, method_b in method_pairs:
                path_a = vqa_root / ds_name / f"{method_a}_k6" / "predictions.json"
                if not path_a.exists():
                    path_a = vqa_root / ds_name / method_a / "predictions.json"
                path_b = vqa_root / ds_name / f"{method_b}_k6" / "predictions.json"
                if not path_b.exists():
                    path_b = vqa_root / ds_name / method_b / "predictions.json"

                if not path_a.exists() or not path_b.exists():
                    print(f"  Skipping {method_a} vs {method_b}: predictions not found")
                    continue

                print(f"  {method_a} vs {method_b}...")
                preds_a = load_predictions(path_a)
                preds_b = load_predictions(path_b)

                id_to_a = {r["query_id"]: r for r in preds_a}
                id_to_b = {r["query_id"]: r for r in preds_b}
                common = sorted(set(id_to_a.keys()) & set(id_to_b.keys()))

                from metrics import VQAMetrics
                vqa_m = VQAMetrics()

                refs = [id_to_a[q]["ground_truth_answer"] for q in common]
                hyps_a = [id_to_a[q]["parsed"]["answer"] for q in common]
                hyps_b = [id_to_b[q]["parsed"]["answer"] for q in common]

                res_a = vqa_m.compute(refs, hyps_a)
                res_b = vqa_m.compute(refs, hyps_b)

                scores_a_bleu = np.array(res_a.per_sample_bleu4)
                scores_b_bleu = np.array(res_b.per_sample_bleu4)
                scores_a_rouge = np.array(res_a.per_sample_rouge_l)
                scores_b_rouge = np.array(res_b.per_sample_rouge_l)
                scores_a_meteor = np.array(res_a.per_sample_meteor)
                scores_b_meteor = np.array(res_b.per_sample_meteor)

                dummy_true = np.ones(len(common))

                def mean_fn(y, s):
                    return float(np.mean(s))

                boot_bleu = bootstrap.test(
                    dummy_true, scores_a_bleu, scores_b_bleu,
                    metric_fn=mean_fn, metric_name="bleu4",
                    method_a_name=method_a, method_b_name=method_b,
                )
                boot_rouge = bootstrap.test(
                    dummy_true, scores_a_rouge, scores_b_rouge,
                    metric_fn=mean_fn, metric_name="rouge_l",
                    method_a_name=method_a, method_b_name=method_b,
                )
                boot_meteor = bootstrap.test(
                    dummy_true, scores_a_meteor, scores_b_meteor,
                    metric_fn=mean_fn, metric_name="meteor",
                    method_a_name=method_a, method_b_name=method_b,
                )

                comp_name = f"{method_a}_vs_{method_b}"
                ds_stats[comp_name] = {
                    "n_paired_samples": len(common),
                    "bootstrap_bleu4": boot_bleu.to_dict(),
                    "bootstrap_rouge_l": boot_rouge.to_dict(),
                    "bootstrap_meteor": boot_meteor.to_dict(),
                }

            all_stats[ds_name] = ds_stats

    stats_path = Path(cfg.output_root) / "statistical_tests.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nAll statistical tests saved to {stats_path}")


if __name__ == "__main__":
    main()
