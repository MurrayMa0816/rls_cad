from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from algorithms.graph_features import synchronize_typed_graph_extractors
from experiment.pipeline import (
    build_env_by_method,
    build_heuristic_action_fn,
    build_total_config,
    discover_run_directories,
    ensure_directory,
    evaluate_policy_or_heuristic,
    extract_resource_budget,
    load_trained_model,
    map_kwargs_from_config,
    method_is_trainable,
    read_run_config,
    resolve_evaluation_seed_suite,
    save_compressed_traces,
    save_json,
    summarize_episode_rows,
    write_rows_to_csv,
)


DEFAULT_HEURISTICS = (
    "uniform",
    "roi_proportional",
    "roi_topk",
    "service_deficit_greedy",
    "greedy_bottleneck_relief",
    "one_step_marginal_greedy",
)

DEFAULT_SEED_MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "seed_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RLS-CAD models.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--suite-root", type=str, default=None)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Diagnostic override. Omit to use the frozen seed-manifest suite.",
    )
    parser.add_argument("--seed-manifest", type=str, default=str(DEFAULT_SEED_MANIFEST))
    parser.add_argument("--test-suite", type=str, default="in_distribution")
    parser.add_argument("--include-heuristics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heuristics", type=str, nargs="*", default=list(DEFAULT_HEURISTICS))
    parser.add_argument("--model-artifact", choices=["best", "final"], default="final")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--roi-topk-ratio", type=float, default=0.10)
    parser.add_argument(
        "--eval-profile",
        type=str,
        default="mixed_balanced",
        help="Evaluation profile. The default maps frozen seeds 20001..20100 to five balanced profiles.",
    )
    parser.add_argument(
        "--trace-mode",
        choices=("none", "compressed", "csv", "both"),
        default="compressed",
        help="Full-evaluation trace export. Compressed NPZ is the storage-safe default.",
    )
    parser.add_argument(
        "--exact-zero-tolerance",
        type=float,
        default=0.0,
        help="Allocation share tolerance used only to classify numerical exact zeros.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="New empty output directory. Defaults to a timestamped directory under the suite root.",
    )
    return parser.parse_args()


def aggregate_summary_by_method(episode_rows, default_mode: str = "restore"):
    rows = []
    methods = sorted({str(row.get("method", "")) for row in episode_rows if row.get("method")})
    for method_name in methods:
        rows_for_method = [row for row in episode_rows if row.get("method") == method_name]
        summary = summarize_episode_rows(method_name, default_mode, rows_for_method, train_seed=-1)
        summary["train_seed"] = "all"
        summary["run_dir"] = "aggregated"
        rows.append(summary)
    return rows


def aggregate_summary_by_method_and_profile(episode_rows, default_mode: str = "restore"):
    rows = []
    keys = sorted(
        {
            (str(row.get("method", "")), str(row.get("profile", "")))
            for row in episode_rows
            if row.get("method") and row.get("profile")
        }
    )
    for method_name, profile_name in keys:
        group = [
            row
            for row in episode_rows
            if row.get("method") == method_name and row.get("profile") == profile_name
        ]
        summary = summarize_episode_rows(method_name, default_mode, group, train_seed=-1)
        summary.update({"train_seed": "all", "run_dir": "aggregated", "profile": profile_name})
        rows.append(summary)
    return rows


def _new_output_directory(root: Path, requested: str | None) -> Path:
    if requested:
        output_dir = Path(requested)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = root / f"paper_eval_{stamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    return ensure_directory(output_dir)


def _trace_file_stem(method_name: str, train_seed: int | str, index: int) -> str:
    safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method_name)).strip("_")
    return f"{index:04d}_{safe_method}_trainseed_{train_seed}"


def main():
    args = parse_args()
    args.eval_seeds, seed_suite_metadata = resolve_evaluation_seed_suite(
        args.seed_manifest,
        args.test_suite,
        args.eval_seeds,
    )
    root_arg = args.suite_root or args.run_dir
    if not root_arg:
        raise ValueError("Need --run-dir or --suite-root.")
    root = Path(root_arg)
    run_dirs = discover_run_directories(root)
    if not run_dirs:
        raise RuntimeError(f"No run_config.json found under {root}")

    output_dir = _new_output_directory(root, args.output_dir)
    all_episode_rows = []
    all_step_rows = []
    all_cell_rows = []
    all_summary_rows = []
    reference_configs = []
    seen_config_keys = set()
    trace_manifest = []
    evaluated_policy_runs = []
    trace_index = 0
    reference_profile_counts = None
    record_compressed = args.trace_mode in {"compressed", "both"}
    record_cell_csv = args.trace_mode in {"csv", "both"}
    record_first_episode_cell_csv = args.trace_mode == "compressed"

    for run_dir in run_dirs:
        run_config = read_run_config(run_dir)
        method_name = str(run_config["method"])
        if method_name != "graph_tmsbd_ppo":
            continue
        if not method_is_trainable(method_name):
            continue
        eval_config = build_total_config(
            mode=str(run_config.get("mode", "restore")),
            map_rows=int(run_config["map_rows"]),
            map_cols=int(run_config["map_cols"]),
            horizon_steps=int(run_config["horizon_steps"]),
            resource_budget=extract_resource_budget(run_config),
            scenario_id=str(run_config.get("scenario_id", "regional_coupled_resource_allocation_v1")),
            seed_value=int(run_config.get("seed", 0)) + 5000,
            profile=str(args.eval_profile or run_config.get("profile", "mixed")),
            **map_kwargs_from_config(run_config),
        )
        def env_factory(cfg=eval_config, method=method_name):
            return build_env_by_method(method, copy.deepcopy(cfg))
        if args.model_artifact == "best":
            model_path = run_dir / "evaluation" / "best_policy_model.zip"
            if not model_path.exists():
                model_path = run_dir / "models" / "final_policy_model.zip"
        else:
            model_path = run_dir / "models" / "final_policy_model.zip"
        if not model_path.exists():
            raise RuntimeError(f"Missing model artifact: {model_path}")
        model = load_trained_model(model_path, device=args.device, method_name=method_name)
        probe_env = env_factory()
        synchronized_extractors = synchronize_typed_graph_extractors(model, probe_env, require=False)
        evaluated_policy_runs.append(
            {
                "method": method_name,
                "train_seed": int(run_config.get("seed", 0)),
                "run_dir": str(run_dir),
                "model_path": str(model_path),
                "num_regions": int(probe_env.n),
                "observation_dim": int(probe_env.observation_space.shape[0]),
                "action_dim": int(probe_env.action_space.shape[0]),
                "model_parameter_count": int(sum(parameter.numel() for parameter in model.policy.parameters())),
                "synchronized_typed_graph_extractors": int(synchronized_extractors),
            }
        )
        results = evaluate_policy_or_heuristic(
            method_name=method_name,
            mode=str(run_config.get("mode", "restore")),
            env_factory=env_factory,
            action_fn=lambda obs, _env, _info, model_obj=model: model_obj.predict(obs, deterministic=True)[0],
            eval_seeds=args.eval_seeds,
            train_seed=int(run_config.get("seed", 0)),
            record_all_cells_for_first_episode=record_first_episode_cell_csv,
            record_full_cell_traces=record_cell_csv,
            record_compressed_traces=record_compressed,
            exact_zero_tolerance=args.exact_zero_tolerance,
        )
        if reference_profile_counts is None:
            reference_profile_counts = {
                profile: sum(1 for row in results["episode_rows"] if row.get("profile") == profile)
                for profile in sorted({str(row.get("profile", "")) for row in results["episode_rows"]})
            }
        if record_compressed:
            trace_path = output_dir / "compressed_traces" / f"{_trace_file_stem(method_name, run_config.get('seed', 0), trace_index)}.npz"
            save_compressed_traces(trace_path, results["trace_arrays"])
            trace_manifest.append(
                {
                    "method": method_name,
                    "train_seed": int(run_config.get("seed", 0)),
                    "run_dir": str(run_dir),
                    "trace_path": str(trace_path.relative_to(output_dir)),
                    "records": int(results["trace_arrays"]["step"].size),
                    "allocation_shape": list(results["trace_arrays"]["allocation"].shape),
                    "raw_action_shape": list(results["trace_arrays"]["raw_action"].shape),
                }
            )
            trace_index += 1
        summary = summarize_episode_rows(
            method_name,
            str(run_config.get("mode", "restore")),
            results["episode_rows"],
            train_seed=int(run_config.get("seed", 0)),
        )
        summary["run_dir"] = str(run_dir)
        all_summary_rows.append(summary)
        all_episode_rows.extend(results["episode_rows"])
        all_step_rows.extend(results["step_rows"])
        all_cell_rows.extend(results["cell_rows"])

        key = (
            int(run_config["map_rows"]),
            int(run_config["map_cols"]),
            int(run_config["horizon_steps"]),
            float(extract_resource_budget(run_config)),
            str(run_config.get("scenario_id", "emergency_wireless_access_backhaul_v1")),
            str(args.eval_profile or run_config.get("profile", "mixed")),
            json.dumps(map_kwargs_from_config(run_config), ensure_ascii=False, sort_keys=True),
        )
        if key not in seen_config_keys:
            seen_config_keys.add(key)
            reference_configs.append(run_config)

    if args.include_heuristics:
        for run_config in reference_configs:
            for heuristic_name in args.heuristics:
                eval_config = build_total_config(
                    mode=str(run_config.get("mode", "restore")),
                    map_rows=int(run_config["map_rows"]),
                    map_cols=int(run_config["map_cols"]),
                    horizon_steps=int(run_config["horizon_steps"]),
                    resource_budget=extract_resource_budget(run_config),
                    scenario_id=str(run_config.get("scenario_id", "regional_coupled_resource_allocation_v1")),
                    seed_value=int(run_config.get("seed", 0)) + 7000,
                    profile=str(args.eval_profile or run_config.get("profile", "mixed")),
                    **map_kwargs_from_config(run_config),
                )
                def env_factory(cfg=eval_config):
                    return build_env_by_method("graph_tmsbd_ppo", copy.deepcopy(cfg))
                results = evaluate_policy_or_heuristic(
                    method_name=heuristic_name,
                    mode=str(run_config.get("mode", "restore")),
                    env_factory=env_factory,
                    action_fn=build_heuristic_action_fn(heuristic_name, topk_ratio=args.roi_topk_ratio),
                    eval_seeds=args.eval_seeds,
                    train_seed=int(run_config.get("seed", 0)),
                    record_all_cells_for_first_episode=record_first_episode_cell_csv,
                    record_full_cell_traces=record_cell_csv,
                    record_compressed_traces=record_compressed,
                    exact_zero_tolerance=args.exact_zero_tolerance,
                )
                if record_compressed:
                    trace_path = output_dir / "compressed_traces" / f"{_trace_file_stem(heuristic_name, run_config.get('seed', 0), trace_index)}.npz"
                    save_compressed_traces(trace_path, results["trace_arrays"])
                    trace_manifest.append(
                        {
                            "method": heuristic_name,
                            "train_seed": int(run_config.get("seed", 0)),
                            "run_dir": "heuristic",
                            "trace_path": str(trace_path.relative_to(output_dir)),
                            "records": int(results["trace_arrays"]["step"].size),
                            "allocation_shape": list(results["trace_arrays"]["allocation"].shape),
                            "raw_action_shape": list(results["trace_arrays"]["raw_action"].shape),
                        }
                    )
                    trace_index += 1
                summary = summarize_episode_rows(
                    heuristic_name,
                    str(run_config.get("mode", "restore")),
                    results["episode_rows"],
                    train_seed=int(run_config.get("seed", 0)),
                )
                summary["run_dir"] = "heuristic"
                all_summary_rows.append(summary)
                all_episode_rows.extend(results["episode_rows"])
                all_step_rows.extend(results["step_rows"])
                all_cell_rows.extend(results["cell_rows"])

    aggregated_summary_rows = aggregate_summary_by_method(all_episode_rows)
    profile_summary_rows = aggregate_summary_by_method_and_profile(all_episode_rows)
    write_rows_to_csv(output_dir / "summary_metrics_by_run.csv", all_summary_rows)
    write_rows_to_csv(output_dir / "summary_metrics.csv", aggregated_summary_rows)
    write_rows_to_csv(output_dir / "summary_metrics_by_profile.csv", profile_summary_rows)
    write_rows_to_csv(output_dir / "episode_metrics.csv", all_episode_rows)
    write_rows_to_csv(output_dir / "step_metrics.csv", all_step_rows)
    if record_cell_csv or record_first_episode_cell_csv:
        write_rows_to_csv(output_dir / "cell_traces.csv", all_cell_rows)
    save_json(output_dir / "summary_metrics_by_run.json", all_summary_rows)
    save_json(output_dir / "summary_metrics.json", aggregated_summary_rows)
    save_json(output_dir / "summary_metrics_by_profile.json", profile_summary_rows)
    save_json(
        output_dir / "evaluation_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root.resolve()),
            "model_artifact": args.model_artifact,
            "eval_profile": args.eval_profile,
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
            "seed_suite": seed_suite_metadata,
            "paired_instances": len(args.eval_seeds),
            "reference_profile_counts": reference_profile_counts or {},
            "balanced_five_profiles": bool(
                reference_profile_counts
                and len(reference_profile_counts) == 5
                and set(reference_profile_counts.values()) == {20}
            ),
            "formal_balanced_100_protocol": bool(
                str(args.eval_profile) == "mixed_balanced"
                and args.test_suite == "in_distribution"
                and seed_suite_metadata["matches_frozen_suite"]
                and len(args.eval_seeds) == 100
            ),
            "trace_mode": args.trace_mode,
            "cell_trace_csv_scope": (
                "all_evaluation_episodes"
                if record_cell_csv
                else "first_evaluation_episode_per_run"
                if record_first_episode_cell_csv
                else "not_exported"
            ),
            "exact_zero_tolerance": float(args.exact_zero_tolerance),
            "initial_scenario_fingerprint": {
                "field": "initial_scenario_sha256",
                "algorithm": "sha256",
                "scope": (
                    "reset-time topology, role assignment, dynamic state, objective/decoder weights, "
                    "profile, and evaluation seed"
                ),
            },
            "trace_schema": {
                "format": "numpy_npz_compressed",
                "allow_pickle_required": False,
                "fields": [
                    "allocation",
                    "raw_action",
                    "decode_rho",
                    "decode_lambda",
                    "decode_tau",
                    "decode_score_calibration",
                    "decode_score_center",
                    "decode_score_scale",
                    "decode_score_scale_used",
                    "decode_score_scale_floor",
                    "decode_score_calibration_degenerate",
                    "decode_temperature_parameter",
                    "decode_temperature_effective",
                    "decode_temperature_normalization",
                    "gates",
                    "gains",
                    "residual",
                    "fused_score",
                    "raw_net_score",
                    "preprojection_score",
                    "net_score",
                ],
            },
            "evaluated_policy_runs": evaluated_policy_runs,
            "traces": trace_manifest,
        },
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "num_methods": len(aggregated_summary_rows), "trace_files": len(trace_manifest)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
