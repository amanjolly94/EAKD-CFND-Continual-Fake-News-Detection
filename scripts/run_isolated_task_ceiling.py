"""Isolated per-task accuracy ceiling: train a fresh (non-continual) FT model
on EACH task by itself -- no prior tasks, no teacher, no sequence -- and
evaluate on that task's own test set. No forgetting is possible by
construction (there is nothing before it to forget), so this measures each
task's intrinsic achievable accuracy independent of any CL mechanism.

Distinguishes two competing explanations for FNN-Gossip-Time's flat BWT
(Section subsec:interpretation, Table tab:task_similarity):
  - low real forgetting pressure (tasks too similar to induce forgetting)
  - low intrinsic ceiling (headline-only text is just hard to classify,
    regardless of forgetting)
If the isolated ceiling is comparable to the ~29% seen in the full CIL run,
the flatness is mostly a low-ceiling story. If it is much higher, low
forgetting pressure remains the better-supported explanation.

Usage: python -m scripts.run_isolated_task_ceiling
"""
from __future__ import annotations

from eakd_cfnd.config import DEFAULT_SEEDS, RunConfig
from eakd_cfnd.data import load_dataset
from eakd_cfnd.train import run_cil_experiment
from scripts.common import aggregate_seeds, run_with_checkpoint, save_json

SEEDS = DEFAULT_SEEDS[:3]  # 3 seeds is enough for a ceiling estimate; keep this cheap


def main():
    all_results = {}
    for dataset_name in ["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"]:
        tasks = load_dataset(dataset_name, root="data")
        all_results[dataset_name] = {}
        for task in tasks:
            per_seed = []
            for seed in SEEDS:
                def _run(task=task, seed=seed):
                    config = RunConfig(dataset=dataset_name, method="FT", seed=seed)
                    # single-task list: no prior classes, no continual dynamics,
                    # this IS the isolated ceiling by construction.
                    return run_cil_experiment([task], config)
                ckpt_dir = f"runs/checkpoints/isolated_ceiling/{dataset_name}"
                summary = run_with_checkpoint(ckpt_dir, f"task{task.task_id}_seed{seed}", _run)
                per_seed.append(summary)
            agg = aggregate_seeds(per_seed)
            all_results[dataset_name][f"task{task.task_id}"] = {
                "n_train": len(task.train), "n_test": len(task.test),
                "aggregate": agg, "per_seed": per_seed,
            }
            print(f"{dataset_name} task{task.task_id} "
                  f"(n_train={len(task.train)}): "
                  f"acc={agg['avg_accuracy']['mean']:.2f}±{agg['avg_accuracy']['sd']:.2f}")
    save_json(all_results, "runs/isolated_task_ceiling.json")


if __name__ == "__main__":
    main()
