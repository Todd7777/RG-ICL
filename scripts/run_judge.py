import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from judge import LLMJudge


def load_predictions(output_root, dataset_name, method_name):
    pred_path = Path(output_root) / dataset_name / method_name / "predictions.json"
    if not pred_path.exists():
        alt_path = Path(output_root) / dataset_name / f"{method_name}_k6" / "predictions.json"
        if alt_path.exists():
            pred_path = alt_path
        else:
            raise FileNotFoundError(f"Predictions not found: {pred_path}")
    with open(pred_path, "r") as f:
        return json.load(f)


def sample_pairs(preds_a, preds_b, n_samples, seed):
    rng = np.random.RandomState(seed)

    id_to_a = {r["query_id"]: r for r in preds_a}
    id_to_b = {r["query_id"]: r for r in preds_b}
    common_ids = sorted(set(id_to_a.keys()) & set(id_to_b.keys()))

    if len(common_ids) <= n_samples:
        selected_ids = common_ids
    else:
        idx = rng.choice(len(common_ids), size=n_samples, replace=False)
        selected_ids = [common_ids[i] for i in sorted(idx)]

    pairs = []
    for qid in selected_ids:
        a = id_to_a[qid]
        b = id_to_b[qid]
        pairs.append({
            "query_id": qid,
            "question": a.get("question", ""),
            "answer_a": a.get("parsed", {}).get("answer", a.get("inference", {}).get("raw_response", "")),
            "answer_b": b.get("parsed", {}).get("answer", b.get("inference", {}).get("raw_response", "")),
        })

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--output-root", type=str, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--comparisons", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_root = args.output_root or (cfg.output_root + "/vqa")

    vqa_datasets = args.datasets or ["medical_cxr_vqa", "vqa_rad", "pathvqa", "pmc_vqa"]

    comparisons = args.comparisons or [
        "naive_icl:zero_shot",
        "rg_icl_global_spatial:zero_shot",
        "rg_icl_global_spatial:naive_icl",
    ]

    judge = LLMJudge(
        model=cfg.judge.model,
        temperature=cfg.judge.temperature,
        seed=cfg.judge.seed,
        api_key_env=cfg.inference.api_key_env,
    )

    all_judge_results = {}

    for ds_name in vqa_datasets:
        print(f"\n{'='*60}")
        print(f"Judge Evaluation: {ds_name}")
        print(f"{'='*60}")

        ds_judge_results = {}

        for comp in comparisons:
            parts = comp.split(":")
            method_a = parts[0]
            method_b = parts[1]
            comp_name = f"{method_a}_vs_{method_b}"
            print(f"\n  Comparison: {method_a} vs {method_b}")

            try:
                preds_a = load_predictions(output_root, ds_name, method_a)
                preds_b = load_predictions(output_root, ds_name, method_b)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            pairs = sample_pairs(
                preds_a, preds_b,
                n_samples=cfg.judge.n_samples_per_dataset,
                seed=cfg.judge.seed,
            )

            print(f"  Sampled {len(pairs)} pairs (target: {cfg.judge.n_samples_per_dataset})")

            judge_items = []
            for p in pairs:
                judge_items.append({
                    "query_id": p["query_id"],
                    "question": p["question"],
                    "answer_a": p["answer_a"],
                    "answer_b": p["answer_b"],
                    "method_a": method_a,
                    "method_b": method_b,
                })

            results = []
            for i, item in enumerate(tqdm(judge_items, desc=f"{comp_name}")):
                result = judge.evaluate(
                    query_id=item["query_id"],
                    question=item["question"],
                    answer_a=item["answer_a"],
                    answer_b=item["answer_b"],
                    method_a=item["method_a"],
                    method_b=item["method_b"],
                    rng_seed=cfg.judge.seed + i,
                )
                results.append(result)

            aggregated = LLMJudge.aggregate_results(results)

            judge_out_dir = Path(output_root) / ds_name / "judge"
            judge_out_dir.mkdir(parents=True, exist_ok=True)

            raw_results = [r.to_dict() for r in results]
            with open(judge_out_dir / f"{comp_name}_raw.json", "w") as f:
                json.dump(raw_results, f, indent=2, default=str)

            with open(judge_out_dir / f"{comp_name}_summary.json", "w") as f:
                json.dump(aggregated, f, indent=2, default=str)

            ds_judge_results[comp_name] = aggregated
            print(f"  Win A: {aggregated.get('wins_a', 0)}, "
                  f"Win B: {aggregated.get('wins_b', 0)}, "
                  f"Tie: {aggregated.get('ties', 0)}")

        all_judge_results[ds_name] = ds_judge_results

    summary_path = Path(output_root) / "judge_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_judge_results, f, indent=2, default=str)
    print(f"\nJudge summary saved to {summary_path}")


if __name__ == "__main__":
    main()
