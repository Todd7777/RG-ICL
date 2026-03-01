import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_reference_tables():
    ref_path = Path(__file__).resolve().parent.parent / "reference_results" / "supplementary_tables.json"
    with open(ref_path, "r") as f:
        return json.load(f)


def load_baselines():
    ref_path = Path(__file__).resolve().parent.parent / "reference_results" / "finetuned_baselines.json"
    with open(ref_path, "r") as f:
        return json.load(f)


def verify_classification(pred_root, reference, tolerance=0.02):
    pred_root = Path(pred_root)
    mismatches = []
    matches = 0
    total = 0

    for dataset_name, methods in reference["classification"].items():
        metrics_path = pred_root / dataset_name / "metrics.json"
        if not metrics_path.exists():
            print(f"  SKIP {dataset_name} (no metrics.json)")
            continue

        with open(metrics_path, "r") as f:
            ds_metrics = json.load(f)

        for method_name, expected in methods.items():
            actual = ds_metrics.get(method_name)
            if actual is None:
                print(f"  SKIP {dataset_name}/{method_name} (not in metrics.json)")
                continue

            for metric, expected_val in expected.items():
                total += 1
                actual_val = actual.get(metric)
                if actual_val is None:
                    mismatches.append(f"{dataset_name}/{method_name}/{metric}: MISSING")
                    continue
                diff = abs(actual_val - expected_val)
                if diff <= tolerance:
                    matches += 1
                else:
                    mismatches.append(
                        f"{dataset_name}/{method_name}/{metric}: "
                        f"expected={expected_val:.4f} actual={actual_val:.4f} diff={diff:.4f}"
                    )

    return matches, total, mismatches


def verify_k_sweep(pred_root, reference, tolerance=0.02):
    pred_root = Path(pred_root)
    mismatches = []
    matches = 0
    total = 0

    summary_path = pred_root / "k_sweep_summary.json"
    if not summary_path.exists():
        print(f"  SKIP k_sweep (no k_sweep_summary.json)")
        return matches, total, mismatches

    with open(summary_path, "r") as f:
        sweep_data = json.load(f)

    dataset = reference["k_sweep"]["dataset"]
    ds_sweep = sweep_data.get(dataset, {})

    for method_name, k_results in reference["k_sweep"]["results"].items():
        method_sweep = ds_sweep.get(method_name, {})
        for k_key, expected_val in k_results.items():
            k = k_key.replace("k", "")
            actual_k = method_sweep.get(k, {})
            if not actual_k:
                print(f"  SKIP k_sweep/{method_name}/k={k} (not in summary)")
                continue

            total += 1
            actual_val = actual_k.get("auc")
            if actual_val is None:
                mismatches.append(f"k_sweep/{method_name}/k={k}/auc: MISSING")
                continue
            diff = abs(actual_val - expected_val)
            if diff <= tolerance:
                matches += 1
            else:
                mismatches.append(
                    f"k_sweep/{method_name}/k={k}/auc: "
                    f"expected={expected_val:.4f} actual={actual_val:.4f} diff={diff:.4f}"
                )

    return matches, total, mismatches


def verify_robustness_imbalance(pred_root, reference, tolerance=0.03):
    pred_root = Path(pred_root)
    mismatches = []
    matches = 0
    total = 0

    dataset = reference["robustness_imbalance"].get("dataset", "chexpert")
    summary_path = pred_root / dataset / "robustness_summary.json"
    if not summary_path.exists():
        print(f"  SKIP robustness imbalance (no robustness_summary.json for {dataset})")
        return matches, total, mismatches

    with open(summary_path, "r") as f:
        summary = json.load(f)

    imbalance_data = summary.get("imbalance", {})

    ref_to_code_method = {"naive_icl": "naive_icl", "rg_icl": "rg_icl_global_spatial"}
    ref_to_code_ratio = {"5N_1P": "5N:1P", "5P_1N": "1N:5P", "3N_3P": "3N:3P"}
    ref_to_code_metric = {"sensitivity": "sens_at_spec90", "specificity": "spec_at_sens90"}

    for ratio_key, methods in reference["robustness_imbalance"].items():
        if ratio_key == "dataset":
            continue
        code_ratio = ref_to_code_ratio.get(ratio_key, ratio_key)
        for method_name, expected in methods.items():
            code_method = ref_to_code_method.get(method_name, method_name)
            method_data = imbalance_data.get(code_method, {})
            actual = method_data.get(code_ratio, {})
            if not actual:
                print(f"  SKIP imbalance/{ratio_key}/{method_name} (not in summary)")
                continue

            for metric, expected_val in expected.items():
                code_metric = ref_to_code_metric.get(metric, metric)
                total += 1
                actual_val = actual.get(code_metric)
                if actual_val is None:
                    mismatches.append(f"imbalance/{ratio_key}/{method_name}/{metric}: MISSING")
                    continue
                diff = abs(actual_val - expected_val)
                if diff <= tolerance:
                    matches += 1
                else:
                    mismatches.append(
                        f"imbalance/{ratio_key}/{method_name}/{metric}: "
                        f"expected={expected_val:.4f} actual={actual_val:.4f}"
                    )

    return matches, total, mismatches


def verify_judge(pred_root, reference, tolerance=0.03):
    pred_root = Path(pred_root)
    mismatches = []
    matches = 0
    total = 0

    for dataset_name, comparisons in reference["judge_win_rates"].items():
        for comp_name, expected in comparisons.items():
            result_path = pred_root / dataset_name / "judge" / f"{comp_name}_summary.json"
            if not result_path.exists():
                print(f"  SKIP judge/{dataset_name}/{comp_name} (no result)")
                continue

            with open(result_path, "r") as f:
                actual = json.load(f)

            total += 1
            actual_wr = actual.get("win_rate_a")
            if actual_wr is None:
                mismatches.append(f"judge/{dataset_name}/{comp_name}/win_rate: MISSING")
                continue
            diff = abs(actual_wr - expected["win_rate"])
            if diff <= tolerance:
                matches += 1
            else:
                mismatches.append(
                    f"judge/{dataset_name}/{comp_name}/win_rate: "
                    f"expected={expected['win_rate']:.4f} actual={actual_wr:.4f}"
                )

    return matches, total, mismatches


def main():
    ap = argparse.ArgumentParser(description="Verify experiment results against supplementary tables")
    ap.add_argument("--config", type=str, default=None,
                    help="Config file (used to derive pred-root if --pred-root not set)")
    ap.add_argument("--pred-root", type=str, default=None,
                    help="Root directory containing experiment outputs")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="Acceptable absolute difference for metric comparison")
    ap.add_argument("--task", type=str, default="all",
                    choices=["all", "classification", "k_sweep", "robustness", "judge"])
    args = ap.parse_args()

    if args.pred_root:
        pred_root = Path(args.pred_root)
    elif args.config:
        from config import load_config
        cfg = load_config(args.config)
        pred_root = Path(cfg.output_root)
    else:
        pred_root = Path("outputs")

    reference = load_reference_tables()

    total_matches = 0
    total_checks = 0
    all_mismatches = []

    if args.task in ("all", "classification"):
        print("\n=== Classification Results (Tables 3-6) ===")
        m, t, mm = verify_classification(pred_root / "classification", reference, args.tolerance)
        total_matches += m
        total_checks += t
        all_mismatches.extend(mm)
        print(f"  {m}/{t} metrics within tolerance ({args.tolerance})")

    if args.task in ("all", "k_sweep"):
        print("\n=== K-Sweep Results (Table 15) ===")
        m, t, mm = verify_k_sweep(pred_root / "k_sweep", reference, args.tolerance)
        total_matches += m
        total_checks += t
        all_mismatches.extend(mm)
        print(f"  {m}/{t} metrics within tolerance ({args.tolerance})")

    if args.task in ("all", "robustness"):
        print("\n=== Robustness Imbalance Results (Table 16) ===")
        m, t, mm = verify_robustness_imbalance(pred_root / "robustness", reference, args.tolerance + 0.01)
        total_matches += m
        total_checks += t
        all_mismatches.extend(mm)
        print(f"  {m}/{t} metrics within tolerance ({args.tolerance + 0.01})")

    if args.task in ("all", "judge"):
        print("\n=== Judge Win Rates (Table 21) ===")
        m, t, mm = verify_judge(pred_root / "vqa", reference, args.tolerance + 0.01)
        total_matches += m
        total_checks += t
        all_mismatches.extend(mm)
        print(f"  {m}/{t} metrics within tolerance ({args.tolerance + 0.01})")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_matches}/{total_checks} metrics verified")

    if all_mismatches:
        print(f"\nMISMATCHES ({len(all_mismatches)}):")
        for mm in all_mismatches:
            print(f"  ✗ {mm}")
    else:
        if total_checks > 0:
            print("\n✓ All checked metrics match reference tables.")
        else:
            print("\nNo output files found to verify. Run experiments first.")


if __name__ == "__main__":
    main()
