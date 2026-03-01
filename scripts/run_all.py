import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    {
        "name": "extract_features",
        "script": "scripts/extract_features.py",
        "config": "configs/default.yaml",
        "description": "Extract frozen encoder features for all datasets",
    },
    {
        "name": "classification",
        "script": "scripts/run_classification.py",
        "config": "configs/experiments/classification.yaml",
        "description": "Run classification experiments (zero-shot, naive ICL, RG-ICL)",
    },
    {
        "name": "vqa",
        "script": "scripts/run_vqa.py",
        "config": "configs/experiments/vqa.yaml",
        "description": "Run VQA experiments (zero-shot, naive ICL, RG-ICL)",
    },
    {
        "name": "k_sweep",
        "script": "scripts/run_k_sweep.py",
        "config": "configs/experiments/k_sweep.yaml",
        "description": "Run k-sweep analysis on CheXpert",
    },
    {
        "name": "robustness",
        "script": "scripts/run_robustness.py",
        "config": "configs/experiments/robustness.yaml",
        "description": "Run robustness experiments (imbalance, ordering, label inconsistency)",
    },
    {
        "name": "encoder_ablation",
        "script": "scripts/run_encoder_ablation.py",
        "config": "configs/experiments/encoder_ablation.yaml",
        "description": "Run encoder ablation (DINOv2 vs CLIP vs MAE)",
    },
    {
        "name": "judge",
        "script": "scripts/run_judge.py",
        "config": "configs/default.yaml",
        "description": "Run LLM-as-a-judge evaluation for VQA",
    },
    {
        "name": "stats",
        "script": "scripts/run_stats.py",
        "config": "configs/default.yaml",
        "description": "Compute paired bootstrap and DeLong statistical tests",
    },
]


def run_step(step, project_root, extra_args=None):
    script_path = project_root / step["script"]
    config_path = project_root / step["config"]

    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"STEP: {step['name']}")
    print(f"  {step['description']}")
    print(f"  Script: {step['script']}")
    print(f"  Config: {step['config']}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        print(f"\nERROR: Step '{step['name']}' failed with return code {result.returncode}")
        return False

    print(f"\nStep '{step['name']}' completed successfully.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=None,
                    help="Specific steps to run. Default: all steps.")
    ap.add_argument("--skip", nargs="+", default=None,
                    help="Steps to skip.")
    ap.add_argument("--start-from", type=str, default=None,
                    help="Start from this step (inclusive).")
    ap.add_argument("--stop-after", type=str, default=None,
                    help="Stop after this step (inclusive).")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    steps_to_run = STEPS

    if args.steps:
        step_names = set(args.steps)
        steps_to_run = [s for s in STEPS if s["name"] in step_names]

    if args.skip:
        skip_names = set(args.skip)
        steps_to_run = [s for s in steps_to_run if s["name"] not in skip_names]

    if args.start_from:
        found = False
        filtered = []
        for s in steps_to_run:
            if s["name"] == args.start_from:
                found = True
            if found:
                filtered.append(s)
        steps_to_run = filtered

    if args.stop_after:
        filtered = []
        for s in steps_to_run:
            filtered.append(s)
            if s["name"] == args.stop_after:
                break
        steps_to_run = filtered

    print("RG-ICL Full Reproduction Pipeline")
    print("=" * 70)
    print(f"Steps to run: {[s['name'] for s in steps_to_run]}")
    print()

    completed = []
    failed = []

    for step in steps_to_run:
        success = run_step(step, project_root)
        if success:
            completed.append(step["name"])
        else:
            failed.append(step["name"])
            print(f"\nPipeline halted at step: {step['name']}")
            break

    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {completed}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")


if __name__ == "__main__":
    main()
