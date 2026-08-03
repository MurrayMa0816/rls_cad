from __future__ import annotations

import argparse
from pathlib import Path

from experiment.pipeline import (
    build_env_by_method,
    build_total_config,
    evaluate_policy_or_heuristic,
    load_trained_model,
    extract_resource_budget,
    read_run_config,
    summarize_episode_rows,
    write_rows_to_csv,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 run_dir 和保存模型做离线回放评估，导出所有区域与 step 的预算和状态量。"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--artifact", choices=["best", "final"], default="best")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[101, 202, 303, 404, 505])
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_config = read_run_config(run_dir)
    method_name = str(run_config["method"])
    mode = str(run_config["mode"])
    train_seed = int(run_config["seed"])

    eval_config = build_total_config(
        mode=mode,
        map_rows=int(run_config["map_rows"]),
        map_cols=int(run_config["map_cols"]),
        horizon_steps=int(run_config["horizon_steps"]),
        resource_budget=extract_resource_budget(run_config),
        scenario_id=str(run_config["scenario_id"]),
        seed_value=train_seed + 1000,
    )

    model_path = run_dir / "evaluation" / "best_policy_model.zip"
    if args.artifact == "final" or not model_path.exists():
        model_path = run_dir / "models" / "final_policy_model.zip"

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / f"offline_replay_{args.artifact}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(model_path, device=args.device)

    def make_env():
        return build_env_by_method(method_name, eval_config)

    replay_results = evaluate_policy_or_heuristic(
        method_name=method_name,
        mode=mode,
        env_factory=make_env,
        action_fn=lambda obs, _env, _info: model.predict(obs, deterministic=True)[0],
        eval_seeds=args.eval_seeds,
        train_seed=train_seed,
        record_cell_traces_for_first_episode=True,
    )
    replay_summary = summarize_episode_rows(
        method_name=method_name,
        mode=mode,
        episode_rows=replay_results["episode_rows"],
        train_seed=train_seed,
    )

    write_rows_to_csv(output_dir / "offline_replay_episode_metrics.csv", replay_results["episode_rows"])
    write_rows_to_csv(output_dir / "offline_replay_step_metrics.csv", replay_results["step_rows"])
    write_rows_to_csv(output_dir / "offline_replay_cell_traces.csv", replay_results["cell_rows"])
    save_json(output_dir / "offline_replay_summary.json", replay_summary)


if __name__ == "__main__":
    main()
