from __future__ import annotations

import datetime as dt
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .pipeline import (
    evaluate_policy_or_heuristic,
    maybe_import_wandb,
    summarize_episode_rows,
)


def append_row_to_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


CORE_TRAIN_ENV_METRICS = {
    "train/last_scaled_reward",
    "train/last_objective_improvement",
    "train/episode_cum_reward",
    "train/episode_cum_improvement",
    "train/episode_cum_total_improvement",
    "train/roi_mean",
    "train/intercept_loss_mean",
    "train/intercept_loss_reduction_ratio",
    "train/corridor_control_mean",
    "train/corridor_exposure_mean",
    "train/corridor_topk_exposure_mean",
    "train/corridor_breach_ratio",
    "train/total_intercept_objective_mean",
    "train/readiness_mean",
    "train/restore_gap_mean",
    "train/network_stability_mean",
    "train/network_stability_gap_mean",
}

CORE_TRAIN_ACTION_METRICS = {
    "train/global_gamma",
    "train/global_tau",
    "train/global_residual_mix",
    "train/global_softmax_alpha",
    "train/global_softmax_mix",
    "train/global_expert_mix",
    "train/global_coverage_budget_share",
    "train/global_focus_budget_share",
    "train/global_coverage_gamma",
    "train/global_coverage_tau",
    "train/global_focus_gamma",
    "train/global_focus_tau",
    "train/selector_entropy",
    "train/selector_residual_cap",
    "train/selector_top_weight",
    "train/selector_roi_param_weight",
    "train/selector_roi_softmax_hybrid_weight",
    "train/selector_roi_dual_expert_weight",
    "train/selector_residual_weight",
    "train/adaptive_roi_gamma",
    "train/adaptive_roi_tau",
    "train/adaptive_hybrid_gamma",
    "train/adaptive_hybrid_tau",
    "train/adaptive_hybrid_softmax_alpha",
    "train/adaptive_hybrid_softmax_mix",
    "train/adaptive_dual_coverage_budget_share",
    "train/adaptive_dual_focus_budget_share",
    "train/adaptive_dual_coverage_gamma",
    "train/adaptive_dual_coverage_tau",
    "train/adaptive_dual_focus_gamma",
    "train/adaptive_dual_focus_tau",
    "train/adaptive_residual_alpha",
    "train/adaptive_residual_support_mix",
    "train/adaptive_residual_neighbor_mix",
}

CORE_SB3_METRICS = (
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "time/fps",
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "train/explained_variance",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/std",
)

CORE_EVAL_SUMMARY_KEYS = {
    "episode_cum_reward_mean",
    "mean_step_reward_mean",
    "episode_cum_total_improvement_mean",
    "final_intercept_loss_mean_mean",
    "final_intercept_loss_reduction_ratio_mean",
    "final_corridor_control_mean_mean",
    "final_corridor_exposure_mean_mean",
    "final_corridor_topk_exposure_mean_mean",
    "final_corridor_breach_ratio_mean",
    "final_total_intercept_objective_mean_mean",
    "final_readiness_mean_mean",
    "final_restore_gap_mean_mean",
    "final_network_stability_mean_mean",
    "final_network_stability_gap_mean_mean",
    "final_selector_entropy_mean",
    "final_selector_residual_cap_mean",
    "final_selector_top_weight_mean",
    "final_selector_roi_param_weight_mean",
    "final_selector_roi_softmax_hybrid_weight_mean",
    "final_selector_roi_dual_expert_weight_mean",
    "final_selector_residual_weight_mean",
    "final_selector_morphology_gate_mean",
    "final_selector_learned_softmax_weight_mean",
    "final_selector_morphology_prior_weight_mean",
    "final_selector_critical_peak_weight_mean",
    "final_selector_corridor_support_weight_mean",
    "final_selector_residual_backlog_weight_mean",
    "final_selector_chain_marginal_weight_mean",
}


def _abbreviate_mode(mode: str) -> str:
    mode_alias = {
        "intercept": "int",
        "restore": "res",
    }
    return mode_alias.get(mode, mode)


def _abbreviate_scenario_id(scenario_id: str) -> str:
    scenario_alias = {
        "paper_priority_v18": "pp18",
        "paper_priority_v17": "pp17",
        "paper_priority_v16": "pp16",
        "paper_priority_v15": "pp15",
        "paper_priority_v14": "pp14",
        "paper_priority_v12": "pp12",
        "paper_priority_v13": "pp13",
        "paper_priority_v11": "pp11",
        "paper_priority_v10": "pp10",
        "paper_priority_v9": "pp9",
        "paper_priority_v8": "pp8",
        "paper_priority_v6": "pp6",
        "paper_priority_v5": "pp5",
        "paper_priority_v4": "pp4",
        "paper_priority_v3": "pp3",
        "paper_priority_v2": "pp2",
        "paper_complex_v1": "pc1",
        "baseline_hex_v3": "bh3",
        "mountain_barrier_v1": "mb1",
    }
    if scenario_id in scenario_alias:
        return scenario_alias[scenario_id]

    parts = [part for part in scenario_id.lower().split("_") if part]
    abbreviated_parts: list[str] = []
    for part in parts:
        if part.startswith("v") and part[1:].isdigit():
            abbreviated_parts.append(part)
        elif part.isdigit():
            abbreviated_parts.append(part)
        elif len(part) <= 3:
            abbreviated_parts.append(part)
        else:
            abbreviated_parts.append(part[0])
    return "".join(abbreviated_parts)


def build_wandb_group_name(
    method_name: str,
    mode: str,
    scenario_id: str,
    map_rows: int,
    map_cols: int,
    horizon_steps: int,
    resource_budget: float,
) -> str:
    base = (
        f"p3"
        f"__{method_name}"
        f"__{_abbreviate_mode(mode)}"
        f"__{_abbreviate_scenario_id(scenario_id)}"
        f"__{map_rows}x{map_cols}"
        f"__h{horizon_steps}"
        f"__b{resource_budget:g}"
    )
    suffix = str(os.environ.get("WANDB_GROUP_SUFFIX", "")).strip()
    if suffix:
        safe_suffix = suffix.replace(" ", "_").replace("/", "-")
        return f"{base}__{safe_suffix}"
    return base


def build_wandb_run_name(group_name: str, seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [part for part in str(group_name).split("__") if part]
    method_name = parts[1] if len(parts) > 1 else "run"
    return f"seed{seed}__{method_name}__{timestamp}"


def init_wandb_run(
    project: str,
    group_name: str,
    run_name: str,
    config_dict: Dict[str, Any],
    entity: str | None = None,
    tags: List[str] | None = None,
    notes: str | None = None,
    save_dir: str | None = None,
):
    wandb = maybe_import_wandb()
    if wandb is None:
        raise RuntimeError("当前环境未安装 wandb，请先执行 `pip install -r requirements.txt`。")
    run = wandb.init(
        project=project,
        entity=entity,
        group=group_name,
        name=run_name,
        config=config_dict,
        tags=tags,
        notes=notes,
        dir=save_dir,
        sync_tensorboard=False,
        save_code=False,
        reinit=True,
    )
    try:
        run.define_metric("time/num_timesteps")
        run.define_metric("train/*", step_metric="time/num_timesteps")
        run.define_metric("eval/*", step_metric="time/num_timesteps")
    except Exception:
        pass
    return run


def wandb_is_active(wandb_module) -> bool:
    if wandb_module is None:
        return False
    return getattr(wandb_module, "run", None) is not None


def filter_eval_summary_for_wandb(summary_row: Dict[str, Any], log_profile: str) -> Dict[str, float]:
    if log_profile == "full":
        return {
            f"eval/{key}": float(value)
            for key, value in summary_row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    return {
        f"eval/{key}": float(value)
        for key, value in summary_row.items()
        if key in CORE_EVAL_SUMMARY_KEYS and isinstance(value, (int, float, np.integer, np.floating))
    }


class WandbHexMetricsCallback(BaseCallback):
    def __init__(self, log_every_n_steps: int = 100, log_profile: str = "compact", verbose: int = 0):
        super().__init__(verbose)
        self.log_every_n_steps = int(log_every_n_steps)
        self.log_profile = str(log_profile)
        self.wandb = maybe_import_wandb()

    @staticmethod
    def _unwrap_env(env):
        current = env
        seen = set()
        while hasattr(current, "env") and id(current) not in seen:
            seen.add(id(current))
            current = current.env
        if hasattr(current, "unwrapped"):
            return current.unwrapped
        return current

    def _collect_scalar_metrics_from_env(self, env) -> Dict[str, float]:
        env = self._unwrap_env(env)
        last_info = getattr(env, "last_info_dict", {}) or {}

        def _read_last_info(key: str, fallback: float = 0.0) -> float:
            value = last_info.get(key, fallback)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            return float(fallback)

        scalar_metrics: Dict[str, float] = {}
        scalar_metrics["train/step_index"] = _read_last_info("t", getattr(env, "current_step_index", 0))
        scalar_metrics["train/resource_budget"] = _read_last_info("resource_budget", getattr(env, "current_budget", 0.0))
        scalar_metrics["train/current_budget"] = _read_last_info("current_budget", getattr(env, "current_budget", 0.0))
        scalar_metrics["train/last_scaled_reward"] = _read_last_info("last_scaled_reward", getattr(env, "last_scaled_reward", 0.0))
        scalar_metrics["train/last_objective_improvement"] = _read_last_info(
            "last_objective_improvement",
            getattr(env, "last_objective_improvement", 0.0),
        )
        scalar_metrics["train/episode_cum_reward"] = _read_last_info(
            "episode_cum_reward",
            getattr(env, "episode_cum_reward", 0.0),
        )
        scalar_metrics["train/episode_cum_improvement"] = _read_last_info(
            "episode_cum_improvement",
            getattr(env, "episode_cum_improvement", 0.0),
        )
        scalar_metrics["train/episode_cum_total_improvement"] = _read_last_info(
            "episode_cum_total_improvement",
            getattr(env, "episode_cum_improvement", 0.0) * getattr(env, "num_hex_cells", 1),
        )
        scalar_metrics["train/roi_mean"] = _read_last_info("roi_mean")
        scalar_metrics["train/roi_max"] = _read_last_info("roi_max")
        scalar_metrics["train/action_mean"] = _read_last_info("action_mean")
        scalar_metrics["train/action_max"] = _read_last_info("action_max")
        scalar_metrics["train/action_std"] = _read_last_info("action_std")

        last_action_info = getattr(env, "last_decoded_action_info", {})
        for metric_key, metric_value in last_action_info.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                scalar_metrics[f"train/{metric_key}"] = float(metric_value)

        current_mode = getattr(env, "current_mode", None)
        if current_mode == "intercept":
            scalar_metrics["train/intercept_loss_mean"] = _read_last_info("intercept_loss_mean")
            scalar_metrics["train/initial_intercept_loss_mean"] = _read_last_info(
                "initial_intercept_loss_mean",
                getattr(env, "initial_intercept_loss_mean", 0.0),
            )
            scalar_metrics["train/corridor_control_mean"] = _read_last_info("corridor_control_mean")
            scalar_metrics["train/corridor_exposure_mean"] = _read_last_info("corridor_exposure_mean")
            scalar_metrics["train/corridor_topk_exposure_mean"] = _read_last_info("corridor_topk_exposure_mean")
            scalar_metrics["train/corridor_breach_ratio"] = _read_last_info("corridor_breach_ratio")
            scalar_metrics["train/total_intercept_objective_mean"] = _read_last_info("total_intercept_objective_mean")
            if float(getattr(env, "initial_intercept_loss_mean", 0.0)) > 0.0:
                current_loss = scalar_metrics["train/intercept_loss_mean"]
                initial_loss = scalar_metrics["train/initial_intercept_loss_mean"]
                scalar_metrics["train/intercept_loss_reduction_ratio"] = float(
                    (initial_loss - current_loss) / max(initial_loss, 1e-12)
                )
                scalar_metrics["train/initial_to_current_loss_drop"] = float(initial_loss - current_loss)
            scalar_metrics["train/threat_mean"] = _read_last_info("threat_mean")
            scalar_metrics["train/protection_mean"] = _read_last_info("protection_mean")
            scalar_metrics["train/capacity_mean"] = _read_last_info("capacity_mean")
        elif current_mode == "restore":
            scalar_metrics["train/readiness_mean"] = _read_last_info("readiness_mean")
            scalar_metrics["train/repair_need_mean"] = _read_last_info("repair_need_mean")
            scalar_metrics["train/stock_mean"] = _read_last_info("stock_mean")
            scalar_metrics["train/demand_pressure_mean"] = _read_last_info("demand_pressure_mean")
            scalar_metrics["train/network_stability_mean"] = _read_last_info("network_stability_mean")
            scalar_metrics["train/network_stability_gap_mean"] = _read_last_info("network_stability_gap_mean")

        if self.log_profile != "full":
            scalar_metrics = {
                key: value
                for key, value in scalar_metrics.items()
                if key in CORE_TRAIN_ENV_METRICS or key in CORE_TRAIN_ACTION_METRICS
            }
        return scalar_metrics

    def _collect_sb3_logger_metrics(self) -> Dict[str, float]:
        logger_obj = getattr(self.model, "logger", None)
        name_to_value = getattr(logger_obj, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return {}

        if self.log_profile == "full":
            return {
                key: float(value)
                for key, value in name_to_value.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }

        output: Dict[str, float] = {}
        for key in CORE_SB3_METRICS:
            value = name_to_value.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                output[key] = float(value)
        return output

    def _on_step(self) -> bool:
        if not wandb_is_active(self.wandb):
            return True
        if (self.n_calls % self.log_every_n_steps) != 0:
            return True
        if not hasattr(self.training_env, "envs"):
            return True

        metric_list = []
        for single_env in self.training_env.envs:
            try:
                metric_list.append(self._collect_scalar_metrics_from_env(single_env))
            except Exception:
                continue

        if not metric_list:
            return True

        merged_metrics: Dict[str, float] = {}
        metric_keys = metric_list[0].keys()
        for metric_key in metric_keys:
            merged_metrics[metric_key] = float(np.mean([metric_dict[metric_key] for metric_dict in metric_list]))
        merged_metrics.update(self._collect_sb3_logger_metrics())
        merged_metrics["time/num_timesteps"] = float(self.num_timesteps)
        self.wandb.log(merged_metrics)
        return True


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        method_name: str,
        mode: str,
        train_seed: int,
        eval_env_factory,
        eval_seeds: Sequence[int],
        eval_freq: int,
        output_dir: str | Path,
        deterministic: bool = True,
        minimum_eval_timesteps: int | None = None,
        log_profile: str = "compact",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.method_name = method_name
        self.mode = mode
        self.train_seed = int(train_seed)
        self.eval_env_factory = eval_env_factory
        self.eval_seeds = [int(seed_value) for seed_value in eval_seeds]
        self.eval_freq = int(eval_freq)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deterministic = bool(deterministic)
        self.minimum_eval_timesteps = int(
            self.eval_freq if minimum_eval_timesteps is None else minimum_eval_timesteps
        )
        if self.minimum_eval_timesteps <= 0:
            raise ValueError("minimum_eval_timesteps must be positive for periodic validation")
        self.log_profile = str(log_profile)
        self.best_metric = -float("inf")
        self.best_num_timesteps = 0
        self.wandb = maybe_import_wandb()

    def _evaluate_model(self) -> Dict[str, Any]:
        eval_results = evaluate_policy_or_heuristic(
            method_name=self.method_name,
            mode=self.mode,
            env_factory=self.eval_env_factory,
            action_fn=lambda obs, _env, _info: self.model.predict(obs, deterministic=self.deterministic)[0],
            eval_seeds=self.eval_seeds,
            train_seed=self.train_seed,
            record_cell_traces_for_first_episode=False,
        )
        summary_row = summarize_episode_rows(
            method_name=self.method_name,
            mode=self.mode,
            episode_rows=eval_results["episode_rows"],
            train_seed=self.train_seed,
        )
        return summary_row

    def _run_eval_and_maybe_update_best(self) -> Dict[str, Any]:
        summary_row = self._evaluate_model()
        summary_row["num_timesteps"] = int(self.num_timesteps)

        current_metric = float(
            summary_row.get(
                "cti_mean",
                summary_row.get("final_capability_mean", -float("inf")),
            )
        )
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_num_timesteps = int(self.num_timesteps)
            best_model_path = self.output_dir / "best_policy_model"
            self.model.save(best_model_path)
            append_row_to_csv(
                self.output_dir / "best_model_events.csv",
                {
                    "num_timesteps": int(self.num_timesteps),
                    "cti_mean": current_metric,
                    "best_model_path": str(best_model_path),
                },
            )

        summary_row["best_so_far_cti_mean"] = float(self.best_metric)
        summary_row["drawdown_from_best_cti_mean"] = float(self.best_metric - current_metric)
        summary_row["best_so_far_num_timesteps"] = int(self.best_num_timesteps)

        append_row_to_csv(self.output_dir / "evaluation_history.csv", summary_row)
        if wandb_is_active(self.wandb):
            wandb_payload = {"time/num_timesteps": float(self.num_timesteps)}
            wandb_payload.update(filter_eval_summary_for_wandb(summary_row, self.log_profile))
            self.wandb.log(wandb_payload)
        return summary_row

    def _on_training_start(self) -> None:
        # A randomly initialized (0-step) policy is not an admissible checkpoint.
        # Validation starts only after the predeclared minimum training budget.
        return None

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps < self.minimum_eval_timesteps:
            return True
        if (self.n_calls % self.eval_freq) != 0:
            return True

        self._run_eval_and_maybe_update_best()
        return True
