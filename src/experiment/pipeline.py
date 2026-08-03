from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from algorithms import (
    DirichletActorCriticPolicy,
    NodeWiseActorCriticPolicy,
    NodeWiseTypedGraphFeatureExtractor,
)
from algorithms.graph_features import TypedGraphFeatureExtractor
from env.action_decoders import (
    TheaterHexSignalMatchedSoftmaxEnv,
    TheaterHexSignalMatchedSparseProjectionEnv,
    TheaterHexDirectSimplexActionEnv,
    TheaterHexDualExpertROIParamActionEnv,
    TheaterHexLTSSCPLatentDecoderEnv,
    TheaterHexROIParamActionEnv,
    TheaterHexSoftmaxActionEnv,
    TheaterHexSparseProjectionActionEnv,
    TheaterHexTMSBDEnv,
    TheaterHexTMSBDFixedMorphologyEnv,
    TheaterHexTMSBDFixedTopKEnv,
    TheaterHexTMSBDNoChainEnv,
    TheaterHexTMSBDNoGateEnv,
    TheaterHexTMSBDSoftmaxBudgetEnv,
    TheaterHexTMSBDSoftmaxGateEnv,
    TheaterHexTMSBDSingleBacklogEnv,
    TheaterHexTMSBDSingleCriticalEnv,
    TheaterHexTMSBDSingleE2EEnv,
    TheaterHexTMSBDSingleSupportEnv,
)
from env.environment import TheaterHexEnvConfig, TheaterHexResourceEnv


DEFAULT_MAIN_SCENARIO_ID = "emergency_wireless_access_backhaul_v1"
DEFAULT_MAIN_ROWS = 12
DEFAULT_MAIN_COLS = 15
DEFAULT_MAIN_HORIZON = 40
DEFAULT_MAIN_RESOURCE_BUDGET = 1.0
DEFAULT_MAIN_BUDGET = DEFAULT_MAIN_RESOURCE_BUDGET
DEFAULT_EVAL_SEEDS = tuple(range(20001, 20101))

NODE_WISE_GRAPH_PPO_METHODS = frozenset(
    {
        "direct_softmax_ppo",
        "signal_matched_softmax_ppo",
        "signal_matched_sparse_projection_ppo",
    }
)

BOTTLENECK_EXPORT_NAMES = {
    "key": "critical",
    "route": "support",
    "backlog": "backlog",
    "chain": "e2e",
}

PROFILE_EXPORT_NAMES = {
    "key_damage": "critical_service",
    "route_fracture": "backhaul_bottleneck",
    "backlog": "traffic_backlog",
    "distributed": "distributed_access_service",
    "compound": "compound_access_backhaul",
    "key_damage_hard": "critical_service_hard",
    "route_fracture_hard": "backhaul_bottleneck_hard",
    "backlog_hard": "traffic_backlog_hard",
    "distributed_hard": "distributed_access_service_hard",
    "compound_hard": "compound_access_backhaul_hard",
}


def export_profile_name(profile: str) -> str:
    return PROFILE_EXPORT_NAMES.get(str(profile), str(profile))


TRAINABLE_METHOD_SPECS: Dict[str, Dict[str, str]] = {
    "graph_tmsbd_ppo": {
        "paper_label": "RLS-CAD",
        "family": "tmsbd",
        "encoder": "typed_graph",
    },
    "direct_softmax_ppo": {
        "paper_label": "GNN-PPO",
        "family": "direct_softmax",
        "encoder": "typed_graph",
    },
    "mlp_softmax_ppo": {
        "paper_label": "PPO-Softmax",
        "family": "direct_softmax",
        "encoder": "flat_mlp",
    },
    "mlp_dirichlet_ppo": {
        "paper_label": "PPO-Dirichlet",
        "family": "direct_simplex",
        "encoder": "typed_graph",
    },
    "direct_softmax_sac": {
        "paper_label": "Direct SAC-Graph",
        "family": "direct_softmax",
        "encoder": "typed_graph",
    },
    "direct_softmax_td3": {
        "paper_label": "Direct TD3-Graph",
        "family": "direct_softmax",
        "encoder": "typed_graph",
    },
    "direct_sparse_projection_ppo": {
        "paper_label": "Direct Sparse-Projection PPO",
        "family": "direct_sparse_projection",
        "encoder": "typed_graph",
    },
    "direct_sparse_projection_sac": {
        "paper_label": "SAC-Projection",
        "family": "direct_sparse_projection",
        "encoder": "typed_graph",
    },
    "direct_sparse_projection_td3": {
        "paper_label": "TD3-Projection",
        "family": "direct_sparse_projection",
        "encoder": "typed_graph",
    },
    "roi_param_ppo": {
        "paper_label": "Priority Param-Graph",
        "family": "roi_param",
        "encoder": "typed_graph",
    },
    "coverage_focus_dual_ppo": {
        "paper_label": "Coverage-Focus Dual",
        "family": "coverage_focus_dual",
        "encoder": "flat_graph",
    },
    "tmsbd_no_gate_ppo": {
        "paper_label": "RLS-CAD w/o Gate",
        "family": "ablation",
        "encoder": "typed_graph",
    },
    "tmsbd_no_chain_ppo": {
        "paper_label": "RLS-CAD w/o Chain",
        "family": "ablation",
        "encoder": "typed_graph",
    },
    "tmsbd_softmax_budget_ppo": {
        "paper_label": "RLS-CAD Softmax Budget",
        "family": "ablation",
        "encoder": "typed_graph",
    },
    "tmsbd_softmax_gate_ppo": {
        "paper_label": "RLS-CAD Temperature-Softmax Gate",
        "family": "matched_control",
        "encoder": "typed_graph",
    },
    "signal_matched_softmax_ppo": {
        "paper_label": "Signal-Matched Softmax PPO",
        "family": "matched_control",
        "encoder": "typed_graph",
    },
    "signal_matched_sparse_projection_ppo": {
        "paper_label": "Signal-Matched Sparse-Projection PPO",
        "family": "matched_control",
        "encoder": "typed_graph",
    },
    "tmsbd_fixed_topk_ppo": {
        "paper_label": "RLS-CAD Fixed Top-K",
        "family": "ablation",
        "encoder": "typed_graph",
    },
    "tmsbd_fixed_morphology_ppo": {
        "paper_label": "Fixed-Morphology Weights",
        "family": "ablation",
        "encoder": "typed_graph",
    },
    "lts_scp_latent_ppo": {
        "paper_label": "LTS-SCP-style Latent Decoder",
        "family": "latent_decoder",
        "encoder": "typed_graph",
    },
    "tmsbd_single_critical_ppo": {
        "paper_label": "Single Critical-Bottleneck Decoder",
        "family": "single_bottleneck",
        "encoder": "typed_graph",
    },
    "tmsbd_single_support_ppo": {
        "paper_label": "Single Support-Bottleneck Decoder",
        "family": "single_bottleneck",
        "encoder": "typed_graph",
    },
    "tmsbd_single_backlog_ppo": {
        "paper_label": "Single Backlog-Bottleneck Decoder",
        "family": "single_bottleneck",
        "encoder": "typed_graph",
    },
    "tmsbd_single_e2e_ppo": {
        "paper_label": "Single End-to-End-Bottleneck Decoder",
        "family": "single_bottleneck",
        "encoder": "typed_graph",
    },
    # Backward-compatible aliases for old scripts.
    "graph_adaptive_decoder_selection_residual_ppo": {
        "paper_label": "RLS-CAD",
        "family": "tmsbd",
        "encoder": "typed_graph",
    },
    "graph_softmax_ppo": {
        "paper_label": "GNN-PPO",
        "family": "direct_softmax",
        "encoder": "flat_graph",
    },
    "graph_roi_param_ppo": {
        "paper_label": "Priority Param-Graph",
        "family": "roi_param",
        "encoder": "flat_graph",
    },
    "graph_roi_dual_expert_ppo": {
        "paper_label": "Coverage-Focus Dual",
        "family": "coverage_focus_dual",
        "encoder": "flat_graph",
    },
    "graph_roi_softmax_hybrid_ppo": {
        "paper_label": "RLS-CAD Softmax Budget",
        "family": "ablation",
        "encoder": "typed_graph",
    },
}

HEURISTIC_METHOD_SPECS: Dict[str, Dict[str, str]] = {
    "uniform": {"paper_label": "Uniform", "family": "heuristic", "encoder": "none"},
    "roi_proportional": {"paper_label": "Priority-Proportional", "family": "heuristic", "encoder": "none"},
    "roi_topk": {"paper_label": "Priority-TopK", "family": "heuristic", "encoder": "none"},
    "service_deficit_greedy": {"paper_label": "Service-Deficit Greedy", "family": "heuristic", "encoder": "none"},
    "repair_gap_greedy": {"paper_label": "Service-Deficit Greedy", "family": "heuristic", "encoder": "none"},
    "greedy_bottleneck_relief": {"paper_label": "Greedy Bottleneck Relief", "family": "heuristic", "encoder": "none"},
    "one_step_marginal_greedy": {"paper_label": "One-step Marginal Greedy", "family": "heuristic", "encoder": "none"},
}

ALL_METHOD_SPECS = {**TRAINABLE_METHOD_SPECS, **HEURISTIC_METHOD_SPECS}
ALL_METHODS = tuple(ALL_METHOD_SPECS.keys())


def require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `{module_name}`. Run `{install_hint}` first.") from exc


def maybe_import_wandb():
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


def method_is_trainable(method_name: str) -> bool:
    return method_name in TRAINABLE_METHOD_SPECS


def method_label(method_name: str) -> str:
    return ALL_METHOD_SPECS[method_name]["paper_label"]


def canonical_method(method_name: str) -> str:
    alias = {
        "graph_adaptive_decoder_selection_residual_ppo": "graph_tmsbd_ppo",
        "graph_softmax_ppo": "direct_softmax_ppo",
        "ppo_softmax": "mlp_softmax_ppo",
        "graph_softmax_sac": "direct_softmax_sac",
        "graph_softmax_td3": "direct_softmax_td3",
        "graph_dirichlet_ppo": "mlp_dirichlet_ppo",
        "graph_roi_param_ppo": "roi_param_ppo",
        "graph_roi_dual_expert_ppo": "coverage_focus_dual_ppo",
        "graph_roi_softmax_hybrid_ppo": "tmsbd_softmax_budget_ppo",
    }
    return alias.get(method_name, method_name)


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: Any):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def resolve_evaluation_seed_suite(
    manifest_path: str | Path,
    suite_name: str,
    explicit_seeds: Sequence[int] | None = None,
) -> tuple[List[int], Dict[str, Any]]:
    """Resolve and fingerprint one frozen evaluation seed suite.

    Explicit seeds remain available for smoke/diagnostic calls, but the
    returned metadata makes a non-matching override impossible to mistake for
    the fixed evaluation protocol.
    """
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds_payload = payload.get("seeds", payload)
    test_suites = seeds_payload.get("test_suites", {})
    if suite_name not in test_suites:
        raise KeyError(f"Seed suite {suite_name!r} is absent from {path}; available={sorted(test_suites)}")
    frozen_seeds = [int(value) for value in test_suites[suite_name]]
    selected_seeds = frozen_seeds if explicit_seeds is None else [int(value) for value in explicit_seeds]
    if not selected_seeds:
        raise ValueError("Evaluation seed selection must not be empty")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError(f"Evaluation seeds contain duplicates: {selected_seeds}")
    return selected_seeds, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "suite_name": str(suite_name),
        "source": "frozen_manifest" if explicit_seeds is None else "explicit_override",
        "matches_frozen_suite": selected_seeds == frozen_seeds,
        "selected_count": len(selected_seeds),
        "frozen_count": len(frozen_seeds),
    }


def write_rows_to_csv(path: str | Path, rows: Sequence[Dict[str, Any]]):
    path = Path(path)
    ensure_directory(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8") as file_obj:
            file_obj.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_resource_budget(config_dict: Dict[str, Any], default_value: float = DEFAULT_MAIN_RESOURCE_BUDGET) -> float:
    for key in ("resource_budget", "annual_budget", "budget"):
        if key in config_dict:
            return float(config_dict[key])
    environment = config_dict.get("environment", {})
    if isinstance(environment, dict):
        for key in ("resource_budget", "annual_budget", "budget"):
            if key in environment:
                return float(environment[key])
    return float(default_value)


def build_total_config(
    mode: str = "restore",
    map_rows: int = DEFAULT_MAIN_ROWS,
    map_cols: int = DEFAULT_MAIN_COLS,
    horizon_steps: int = DEFAULT_MAIN_HORIZON,
    resource_budget: float | None = None,
    scenario_id: str = DEFAULT_MAIN_SCENARIO_ID,
    seed_value: int = 0,
    annual_budget: float | None = None,
    profile: str = "mixed",
    max_sparse_support_fraction: float | None = None,
    tmsbd_action_dim: int | None = None,
    tmsbd_tau_multiplier: float | None = None,
    tmsbd_gamma_base: float | None = None,
    tmsbd_gamma_scale: float | None = None,
    tmsbd_gamma_bias: float | None = None,
    tmsbd_gate_residual_scale: float | None = None,
    tmsbd_descriptor_mix: float | None = None,
    tmsbd_channel_prior_strength: float | None = None,
    tmsbd_value_gain_floor: float | None = None,
    tmsbd_value_gain_span: float | None = None,
    tmsbd_value_gate_bias: float | None = None,
    tmsbd_value_residual_scale: float | None = None,
    tmsbd_mu_scale: float | None = None,
    tmsbd_mu_bias: float | None = None,
    tmsbd_projection_min_contrast: float | None = None,
    switch_penalty_weight: float | None = None,
    route_damage_fraction: float | None = None,
    stochastic_disturbance: bool | None = None,
    disturbance_noise: float | None = None,
    initial_readiness_low: float | None = None,
    initial_readiness_high: float | None = None,
    budget_dilution_threshold: float | None = None,
    diffuse_budget_efficiency: float | None = None,
    active_support_penalty_weight: float | None = None,
    lambda_zeta: float | None = None,
    decoder_lambda_zeta: float | None = None,
    reference_lambda_zeta: float | None = None,
    arrival_coefficient_of_variation: float | None = None,
    arrival_base_rate: float | None = None,
    objective_weight_override: Sequence[float] | None = None,
    decoder_weight_override: Sequence[float] | None = None,
    tmsbd_gate_softmax_temperature: float | None = None,
    min_effective_share: float | None = None,
    signal_matched_score_calibration: str | None = None,
    signal_matched_score_scale_floor: float | None = None,
    signal_matched_sparse_tau_normalization: str | None = None,
    direct_softmax_temperature: float | None = None,
    direct_sparse_projection_tau: float | None = None,
    topology_seed: int | None = None,
    spatial_edge_drop_fraction: float | None = None,
    edge_rewire_fraction: float | None = None,
    route_edge_drop_fraction: float | None = None,
    support_edge_drop_fraction: float | None = None,
    role_relocation_fraction: float | None = None,
    support_providers_per_target: int | None = None,
) -> TheaterHexEnvConfig:
    del mode
    cfg = TheaterHexEnvConfig()
    cfg.map_config.map_rows = int(map_rows)
    cfg.map_config.map_cols = int(map_cols)
    cfg.map_config.horizon_steps = int(horizon_steps)
    cfg.map_config.resource_budget = float(
        DEFAULT_MAIN_RESOURCE_BUDGET if resource_budget is None and annual_budget is None else
        (resource_budget if resource_budget is not None else annual_budget)
    )
    cfg.map_config.random_seed = int(seed_value)
    cfg.map_config.scenario_id = str(scenario_id)
    cfg.map_config.profile = str(profile)
    if max_sparse_support_fraction is not None:
        cfg.map_config.max_sparse_support_fraction = float(max_sparse_support_fraction)
    if tmsbd_action_dim is not None:
        cfg.map_config.tmsbd_action_dim = int(tmsbd_action_dim)
    if tmsbd_tau_multiplier is not None:
        cfg.map_config.tmsbd_tau_multiplier = float(tmsbd_tau_multiplier)
    if tmsbd_gamma_base is not None:
        cfg.map_config.tmsbd_gamma_base = float(tmsbd_gamma_base)
    if tmsbd_gamma_scale is not None:
        cfg.map_config.tmsbd_gamma_scale = float(tmsbd_gamma_scale)
    if tmsbd_gamma_bias is not None:
        cfg.map_config.tmsbd_gamma_bias = float(tmsbd_gamma_bias)
    if tmsbd_gate_residual_scale is not None:
        cfg.map_config.tmsbd_gate_residual_scale = float(tmsbd_gate_residual_scale)
    if tmsbd_descriptor_mix is not None:
        cfg.map_config.tmsbd_descriptor_mix = float(tmsbd_descriptor_mix)
    if tmsbd_channel_prior_strength is not None:
        cfg.map_config.tmsbd_channel_prior_strength = float(tmsbd_channel_prior_strength)
    if tmsbd_value_gain_floor is not None:
        cfg.map_config.tmsbd_value_gain_floor = float(tmsbd_value_gain_floor)
    if tmsbd_value_gain_span is not None:
        cfg.map_config.tmsbd_value_gain_span = float(tmsbd_value_gain_span)
    if tmsbd_value_gate_bias is not None:
        cfg.map_config.tmsbd_value_gate_bias = float(tmsbd_value_gate_bias)
    if tmsbd_value_residual_scale is not None:
        cfg.map_config.tmsbd_value_residual_scale = float(tmsbd_value_residual_scale)
    if tmsbd_mu_scale is not None:
        cfg.map_config.tmsbd_mu_scale = float(tmsbd_mu_scale)
    if tmsbd_mu_bias is not None:
        cfg.map_config.tmsbd_mu_bias = float(tmsbd_mu_bias)
    if tmsbd_projection_min_contrast is not None:
        cfg.map_config.tmsbd_projection_min_contrast = float(tmsbd_projection_min_contrast)
    if switch_penalty_weight is not None:
        cfg.map_config.switch_penalty_weight = float(switch_penalty_weight)
    if route_damage_fraction is not None:
        cfg.map_config.route_damage_fraction = float(route_damage_fraction)
    if stochastic_disturbance is not None:
        cfg.map_config.stochastic_disturbance = bool(stochastic_disturbance)
    if disturbance_noise is not None:
        cfg.map_config.disturbance_noise = float(disturbance_noise)
    if initial_readiness_low is not None:
        cfg.map_config.initial_readiness_low = float(initial_readiness_low)
    if initial_readiness_high is not None:
        cfg.map_config.initial_readiness_high = float(initial_readiness_high)
    if budget_dilution_threshold is not None:
        cfg.map_config.budget_dilution_threshold = float(budget_dilution_threshold)
    if diffuse_budget_efficiency is not None:
        cfg.map_config.diffuse_budget_efficiency = float(diffuse_budget_efficiency)
    if active_support_penalty_weight is not None:
        cfg.map_config.active_support_penalty_weight = float(active_support_penalty_weight)
    if lambda_zeta is not None:
        cfg.map_config.lambda_zeta = float(lambda_zeta)
    if decoder_lambda_zeta is not None:
        cfg.map_config.decoder_lambda_zeta = float(decoder_lambda_zeta)
    if reference_lambda_zeta is not None:
        cfg.map_config.reference_lambda_zeta = float(reference_lambda_zeta)
    if arrival_coefficient_of_variation is not None:
        cfg.map_config.arrival_coefficient_of_variation = float(arrival_coefficient_of_variation)
    if arrival_base_rate is not None:
        cfg.map_config.arrival_base_rate = float(arrival_base_rate)
    if objective_weight_override is not None:
        cfg.map_config.objective_weight_override = tuple(float(value) for value in objective_weight_override)
    if decoder_weight_override is not None:
        cfg.map_config.decoder_weight_override = tuple(float(value) for value in decoder_weight_override)
    if tmsbd_gate_softmax_temperature is not None:
        cfg.map_config.tmsbd_gate_softmax_temperature = float(tmsbd_gate_softmax_temperature)
    if min_effective_share is not None:
        cfg.map_config.min_effective_share = float(min_effective_share)
    if signal_matched_score_calibration is not None:
        cfg.map_config.signal_matched_score_calibration = str(signal_matched_score_calibration)
    if signal_matched_score_scale_floor is not None:
        cfg.map_config.signal_matched_score_scale_floor = float(signal_matched_score_scale_floor)
    if signal_matched_sparse_tau_normalization is not None:
        cfg.map_config.signal_matched_sparse_tau_normalization = str(
            signal_matched_sparse_tau_normalization
        )
    if direct_softmax_temperature is not None:
        cfg.map_config.direct_softmax_temperature = float(direct_softmax_temperature)
    if direct_sparse_projection_tau is not None:
        cfg.map_config.direct_sparse_projection_tau = float(direct_sparse_projection_tau)
    if topology_seed is not None:
        cfg.map_config.topology_seed = int(topology_seed)
    if spatial_edge_drop_fraction is not None:
        cfg.map_config.spatial_edge_drop_fraction = float(spatial_edge_drop_fraction)
    if edge_rewire_fraction is not None:
        cfg.map_config.edge_rewire_fraction = float(edge_rewire_fraction)
    if route_edge_drop_fraction is not None:
        cfg.map_config.route_edge_drop_fraction = float(route_edge_drop_fraction)
    if support_edge_drop_fraction is not None:
        cfg.map_config.support_edge_drop_fraction = float(support_edge_drop_fraction)
    if role_relocation_fraction is not None:
        cfg.map_config.role_relocation_fraction = float(role_relocation_fraction)
    if support_providers_per_target is not None:
        cfg.map_config.support_providers_per_target = int(support_providers_per_target)
    return cfg


def map_kwargs_from_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "max_sparse_support_fraction",
        "tmsbd_action_dim",
        "tmsbd_tau_multiplier",
        "tmsbd_gamma_base",
        "tmsbd_gamma_scale",
        "tmsbd_gamma_bias",
        "tmsbd_gate_residual_scale",
        "tmsbd_descriptor_mix",
        "tmsbd_channel_prior_strength",
        "tmsbd_value_gain_floor",
        "tmsbd_value_gain_span",
        "tmsbd_value_gate_bias",
        "tmsbd_value_residual_scale",
        "tmsbd_mu_scale",
        "tmsbd_mu_bias",
        "tmsbd_projection_min_contrast",
        "switch_penalty_weight",
        "route_damage_fraction",
        "stochastic_disturbance",
        "disturbance_noise",
        "initial_readiness_low",
        "initial_readiness_high",
        "budget_dilution_threshold",
        "diffuse_budget_efficiency",
        "active_support_penalty_weight",
        "lambda_zeta",
        "decoder_lambda_zeta",
        "reference_lambda_zeta",
        "arrival_coefficient_of_variation",
        "arrival_base_rate",
        "objective_weight_override",
        "decoder_weight_override",
        "tmsbd_gate_softmax_temperature",
        "min_effective_share",
        "signal_matched_score_calibration",
        "signal_matched_score_scale_floor",
        "signal_matched_sparse_tau_normalization",
        "direct_softmax_temperature",
        "direct_sparse_projection_tau",
        "topology_seed",
        "spatial_edge_drop_fraction",
        "edge_rewire_fraction",
        "route_edge_drop_fraction",
        "support_edge_drop_fraction",
        "role_relocation_fraction",
        "support_providers_per_target",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key in config_dict:
            out[key] = config_dict[key]
    environment = config_dict.get("environment", {})
    if isinstance(environment, dict):
        for key in keys:
            if key in environment and key not in out:
                out[key] = environment[key]
    return out


def build_env_by_method(method_name: str, total_config: TheaterHexEnvConfig) -> TheaterHexResourceEnv:
    method = canonical_method(method_name)
    if method in {"direct_softmax_ppo", "mlp_softmax_ppo", "direct_softmax_sac", "direct_softmax_td3"}:
        return TheaterHexSoftmaxActionEnv(total_config)
    if method == "mlp_dirichlet_ppo":
        return TheaterHexDirectSimplexActionEnv(total_config)
    if method in {"direct_sparse_projection_ppo", "direct_sparse_projection_sac", "direct_sparse_projection_td3"}:
        return TheaterHexSparseProjectionActionEnv(total_config)
    if method == "roi_param_ppo":
        return TheaterHexROIParamActionEnv(total_config)
    if method == "coverage_focus_dual_ppo":
        return TheaterHexDualExpertROIParamActionEnv(total_config)
    if method == "tmsbd_no_gate_ppo":
        return TheaterHexTMSBDNoGateEnv(total_config)
    if method == "tmsbd_no_chain_ppo":
        return TheaterHexTMSBDNoChainEnv(total_config)
    if method == "tmsbd_softmax_budget_ppo":
        return TheaterHexTMSBDSoftmaxBudgetEnv(total_config)
    if method == "tmsbd_softmax_gate_ppo":
        return TheaterHexTMSBDSoftmaxGateEnv(total_config)
    if method == "signal_matched_softmax_ppo":
        return TheaterHexSignalMatchedSoftmaxEnv(total_config)
    if method == "signal_matched_sparse_projection_ppo":
        return TheaterHexSignalMatchedSparseProjectionEnv(total_config)
    if method == "tmsbd_fixed_topk_ppo":
        return TheaterHexTMSBDFixedTopKEnv(total_config)
    if method == "tmsbd_fixed_morphology_ppo":
        return TheaterHexTMSBDFixedMorphologyEnv(total_config)
    if method == "lts_scp_latent_ppo":
        return TheaterHexLTSSCPLatentDecoderEnv(total_config)
    if method == "tmsbd_single_critical_ppo":
        return TheaterHexTMSBDSingleCriticalEnv(total_config)
    if method == "tmsbd_single_support_ppo":
        return TheaterHexTMSBDSingleSupportEnv(total_config)
    if method == "tmsbd_single_backlog_ppo":
        return TheaterHexTMSBDSingleBacklogEnv(total_config)
    if method == "tmsbd_single_e2e_ppo":
        return TheaterHexTMSBDSingleE2EEnv(total_config)
    return TheaterHexTMSBDEnv(total_config)


def build_policy_and_kwargs(method_name: str, env: TheaterHexResourceEnv | None = None):
    net_arch = [128, 128]
    method = canonical_method(method_name)
    method_spec = TRAINABLE_METHOD_SPECS.get(method_name, TRAINABLE_METHOD_SPECS.get(method, {}))
    # Capacity-matched graph policies share the same trunk.  Their action-head
    # dimensions may differ and are reported separately in run manifests.
    if method_spec.get("encoder") == "typed_graph":
        net_arch = [256, 256, 128]
    elif method in {
        "mlp_softmax_ppo",
    }:
        net_arch = [256, 128]

    if method in NODE_WISE_GRAPH_PPO_METHODS:
        if env is None or getattr(env, "n", 0) <= 0:
            raise ValueError(
                f"Method {method!r} requires an environment to build its node-wise graph policy"
            )
        dep = np.maximum(
            np.asarray(env.a_dep, dtype=np.float32),
            np.asarray(env.a_dep, dtype=np.float32).T,
        )
        return NodeWiseActorCriticPolicy, {
            "features_extractor_class": NodeWiseTypedGraphFeatureExtractor,
            "features_extractor_kwargs": {
                "num_cells": int(env.n),
                "cell_feature_dim": int(env.node_feature_dim),
                "global_feature_dim": int(env.global_feature_dim),
                "adjacency_matrices": {
                    "adj": np.asarray(env.a_adj, dtype=np.float32),
                    "route": np.asarray(env.a_route, dtype=np.float32),
                    "dep": dep.astype(np.float32),
                },
                "hidden_dim": 128,
                "message_passing_steps": 2,
            },
            # Use the same depth/width on the shared regional score head and
            # pooled critic as the historical typed-graph PPO trunk.  Learned
            # actor parameters do not grow with the number of regions.
            "node_actor_hidden_dims": (256, 256, 128),
            "critic_hidden_dims": (256, 256, 128),
        }

    policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}
    if env is not None and getattr(env, "n", 0) > 0 and method_spec.get("encoder") == "typed_graph":
        dep = np.maximum(np.asarray(env.a_dep, dtype=np.float32), np.asarray(env.a_dep, dtype=np.float32).T)
        policy_kwargs.update(
            {
                "features_extractor_class": TypedGraphFeatureExtractor,
                "features_extractor_kwargs": {
                    "num_cells": int(env.n),
                    "cell_feature_dim": int(env.node_feature_dim),
                    "global_feature_dim": int(env.global_feature_dim),
                    "adjacency_matrices": {
                        # Keep these legacy module keys stable so trained
                        # checkpoints remain loadable; exported artifacts use
                        # the paper-facing sp/svc/sup terminology below.
                        "adj": np.asarray(env.a_adj, dtype=np.float32),
                        "route": np.asarray(env.a_route, dtype=np.float32),
                        "dep": dep.astype(np.float32),
                    },
                    "hidden_dim": 128,
                    "message_passing_steps": 2,
                    "features_dim": 256,
                },
            }
        )
    if method == "mlp_dirichlet_ppo":
        return DirichletActorCriticPolicy, policy_kwargs
    return "MlpPolicy", policy_kwargs


def algorithm_name_for_method(method_name: str) -> str:
    method = canonical_method(method_name)
    if method == "direct_softmax_sac":
        return "SAC"
    if method == "direct_softmax_td3":
        return "TD3"
    if method == "direct_sparse_projection_sac":
        return "SAC"
    if method == "direct_sparse_projection_td3":
        return "TD3"
    return "PPO"


def algorithm_class_for_method(method_name: str):
    sb3 = require_dependency("stable_baselines3", "pip install -r requirements.txt")
    algorithm_name = algorithm_name_for_method(method_name)
    return getattr(sb3, algorithm_name)


def infer_method_from_model_path(model_path: str | Path) -> str | None:
    path = Path(model_path).resolve()
    for parent in [path.parent, *path.parents]:
        config_path = parent / "run_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            method = payload.get("method")
            return str(method) if method else None
    return None


def load_trained_model(model_path: str | Path, device: str = "cpu", method_name: str | None = None):
    method = method_name or infer_method_from_model_path(model_path)
    if method:
        model_cls = algorithm_class_for_method(method)
        return model_cls.load(str(model_path), device=device)
    sb3 = require_dependency("stable_baselines3", "pip install -r requirements.txt")
    last_error: Exception | None = None
    for model_cls in (sb3.PPO, sb3.SAC, sb3.TD3):
        try:
            return model_cls.load(str(model_path), device=device)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load model artifact {model_path}") from last_error


def read_run_config(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir) / "run_config.json"
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def discover_run_directories(root: str | Path) -> List[Path]:
    root = Path(root)
    if (root / "run_config.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("run_config.json"))


def export_environment_static_artifacts(env: TheaterHexResourceEnv, outdir: str | Path):
    outdir = ensure_directory(outdir)
    save_json(
        outdir / "environment_static.json",
        {
            "rows": env.rows,
            "cols": env.cols,
            "num_regions": env.n,
            "node_types": env.type_id.astype(int).tolist(),
            "value": env.value.astype(float).tolist(),
            "criticality": env.criticality.astype(float).tolist(),
            "exec_cost": env.exec_cost.astype(float).tolist(),
            "sp_edges": int(env.a_adj.sum()),
            "svc_edges": int(env.a_route.sum()),
            "sup_edges": int(env.a_dep.sum()),
        },
    )


def build_heuristic_action_fn(method_name: str, **kwargs: Any) -> Callable[[np.ndarray, TheaterHexResourceEnv, Dict[str, Any]], np.ndarray]:
    def action_fn(_obs: np.ndarray, env: TheaterHexResourceEnv, _info: Dict[str, Any]) -> np.ndarray:
        return env.heuristic_action(method_name, **kwargs)

    return action_fn


def _scalarize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    return value


def _unit_standardize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (arr - float(np.mean(arr))) / (float(np.std(arr)) + 1e-6)


def _fixed_vector(value: Any, size: int, fill_value: float = np.nan) -> np.ndarray:
    """Return a fixed-width float vector without creating object arrays."""
    out = np.full(int(size), fill_value, dtype=np.float32)
    if value is None:
        return out
    values = np.asarray(value, dtype=np.float32).reshape(-1)
    count = min(out.size, values.size)
    if count:
        out[:count] = values[:count]
    return out


def _allocation_diagnostics(
    allocation: np.ndarray,
    previous_allocation: np.ndarray,
    env: TheaterHexResourceEnv,
    *,
    is_initial_placement: bool,
    exact_zero_tolerance: float,
) -> Dict[str, Any]:
    allocation = np.asarray(allocation, dtype=np.float32).reshape(-1)
    previous = np.asarray(previous_allocation, dtype=np.float32).reshape(-1)
    total_budget = max(float(env.budget), 1e-12)
    zero_threshold = max(float(exact_zero_tolerance) * total_budget, 0.0)
    effective_threshold = max(float(env.config.min_effective_share) * total_budget, 0.0)
    positive = allocation > zero_threshold
    exact_zero = ~positive
    effective = allocation >= effective_threshold if effective_threshold > zero_threshold else positive
    subthreshold = positive & ~effective
    migration = float(np.sum(np.abs(allocation - previous)) / total_budget)
    current_support = positive
    previous_support = previous > zero_threshold
    support_flip_count = int(np.count_nonzero(current_support != previous_support))
    previous_effective = previous >= effective_threshold if effective_threshold > zero_threshold else previous_support
    effective_union = int(np.count_nonzero(effective | previous_effective))
    effective_intersection = int(np.count_nonzero(effective & previous_effective))
    effective_jaccard_change = (
        np.nan
        if is_initial_placement
        else float(1.0 - effective_intersection / effective_union) if effective_union else 0.0
    )
    return {
        "exact_zero_tolerance": float(zero_threshold),
        "effective_allocation_threshold": float(effective_threshold),
        "positive_count": int(np.count_nonzero(positive)),
        "positive_ratio": float(np.mean(positive)),
        "exact_zero_count": int(np.count_nonzero(exact_zero)),
        "exact_zero_ratio": float(np.mean(exact_zero)),
        "effective_count": int(np.count_nonzero(effective)),
        "effective_ratio": float(np.mean(effective)),
        "subthreshold_positive_count": int(np.count_nonzero(subthreshold)),
        "subthreshold_positive_ratio": float(np.mean(subthreshold)),
        "subthreshold_capacity": float(np.sum(allocation[subthreshold])),
        "subthreshold_capacity_share": float(np.sum(allocation[subthreshold]) / total_budget),
        "allocation_l1_migration": migration,
        "is_initial_placement": bool(is_initial_placement),
        "initial_placement_l1": migration if is_initial_placement else np.nan,
        "switch_migration_l1": np.nan if is_initial_placement else migration,
        "switch_moved_capacity_fraction": np.nan if is_initial_placement else 0.5 * migration,
        "support_flip_count": support_flip_count,
        "support_flip_ratio": float(support_flip_count / max(allocation.size, 1)),
        "effective_support_jaccard_change": effective_jaccard_change,
        "positive_mask": positive,
        "effective_mask": effective,
        "subthreshold_mask": subthreshold,
    }


def _decoder_snapshot(
    method_name: str,
    env: TheaterHexResourceEnv,
    raw_action: np.ndarray,
    decode: Dict[str, Any],
) -> Dict[str, Any]:
    """Capture decoder inputs/scores before the environment transition.

    New decoders can export ``fused_score`` and ``net_score`` directly.  The
    reconstruction below keeps historical checkpoints auditable without
    changing their decoder implementation.
    """
    n = int(env.n)
    method = canonical_method(method_name)
    maps: Dict[str, np.ndarray] = {}
    if hasattr(env, "value_maps"):
        no_chain = method == "tmsbd_no_chain_ppo"
        maps = {
            name: np.asarray(values, dtype=np.float32).copy()
            for name, values in env.value_maps(no_chain=no_chain).items()
        }

    exported_fused = decode.get("fused_score")
    exported_raw_net = decode.get("raw_net_score")
    exported_preprojection = decode.get("preprojection_score")
    exported_net = decode.get("net_score")
    fused_score = _fixed_vector(exported_fused, n) if exported_fused is not None else np.full(n, np.nan, dtype=np.float32)
    raw_net_score = (
        _fixed_vector(exported_raw_net, n)
        if exported_raw_net is not None
        else np.full(n, np.nan, dtype=np.float32)
    )
    preprojection_score = (
        _fixed_vector(exported_preprojection, n)
        if exported_preprojection is not None
        else np.full(n, np.nan, dtype=np.float32)
    )
    net_score = _fixed_vector(exported_net, n) if exported_net is not None else np.full(n, np.nan, dtype=np.float32)

    alpha = _fixed_vector(decode.get("alpha"), 4)
    gains = _fixed_vector(decode.get("channel_gain"), 4, fill_value=1.0)
    residual = _fixed_vector(decode.get("residual_coeff"), 4, fill_value=0.0)
    gamma = float(decode.get("gamma", np.nan))
    mu = float(decode.get("mu", 0.0))

    if not np.all(np.isfinite(net_score)) and maps and np.all(np.isfinite(alpha)) and np.isfinite(gamma):
        map_names = ("key", "route", "backlog", "chain")
        standardized = [
            gamma * float(gains[idx]) * _unit_standardize(np.asarray(maps.get(name, np.zeros(n)), dtype=np.float32))
            for idx, name in enumerate(map_names)
        ]
        fused_score = sum(float(alpha[idx]) * standardized[idx] for idx in range(4))
        backlog_ratio = np.asarray(env.backlog, dtype=np.float32) / (
            np.asarray(env.backlog_cap, dtype=np.float32) + 1e-8
        )
        residual_basis = np.stack(
            [
                _unit_standardize(np.asarray(env.delta, dtype=np.float32)),
                _unit_standardize(1.0 - np.asarray(env.kappa, dtype=np.float32)),
                _unit_standardize(backlog_ratio),
                _unit_standardize(1.0 - np.maximum(np.asarray(env.sigma, dtype=np.float32), 0.0)),
            ],
            axis=0,
        )
        fused_score = fused_score + gamma * np.sum(residual[:, None] * residual_basis, axis=0)
        net_score = fused_score - mu * np.asarray(env.exec_cost, dtype=np.float32)
    elif not np.all(np.isfinite(net_score)):
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if method in {"direct_softmax_ppo", "mlp_softmax_ppo", "direct_softmax_sac", "direct_softmax_td3"}:
            if raw.size != n:
                raise ValueError(f"Direct action has {raw.size} entries; expected {n}")
            net_score = raw.astype(np.float32)
            fused_score = net_score.copy()
        elif method in {
            "direct_sparse_projection_ppo",
            "direct_sparse_projection_sac",
            "direct_sparse_projection_td3",
        }:
            if raw.size != n:
                raise ValueError(f"Direct projection action has {raw.size} entries; expected {n}")
            net_score = _unit_standardize(raw)
            fused_score = net_score.copy()

    return {
        "maps": maps,
        "fused_score": np.asarray(fused_score, dtype=np.float32),
        "raw_net_score": np.asarray(raw_net_score, dtype=np.float32),
        "preprojection_score": np.asarray(preprojection_score, dtype=np.float32),
        "net_score": np.asarray(net_score, dtype=np.float32),
        "gates": alpha,
        "gains": gains,
        "residual": residual,
        "service_interruption": np.asarray(env.delta, dtype=np.float32).copy(),
        "upstream_support": np.asarray(env.kappa, dtype=np.float32).copy(),
        "local_service": np.asarray(env.sigma, dtype=np.float32).copy(),
        "backlog_ratio": (
            np.asarray(env.backlog, dtype=np.float32)
            / (np.asarray(env.backlog_cap, dtype=np.float32) + 1e-8)
        ).copy(),
    }


def _pad_trace_vectors(records: Sequence[Dict[str, Any]], key: str, width: int | None = None) -> np.ndarray:
    if not records:
        return np.empty((0, int(width or 0)), dtype=np.float32)
    if width is None:
        width = max(np.asarray(record.get(key, []), dtype=np.float32).size for record in records)
    return np.stack([_fixed_vector(record.get(key), int(width)) for record in records], axis=0)


def _pack_trace_records(records: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pack full-evaluation decoder traces into a non-pickle NPZ schema."""
    scalar_float_keys = (
        "decode_lambda",
        "decode_tau",
        "decode_gamma",
        "decode_mu",
        "decode_score_center",
        "decode_score_scale",
        "decode_score_scale_used",
        "decode_score_scale_floor",
        "decode_temperature_parameter",
        "decode_temperature_effective",
        "exact_zero_ratio",
        "effective_ratio",
        "subthreshold_positive_ratio",
        "subthreshold_capacity_share",
        "allocation_l1_migration",
        "initial_placement_l1",
        "switch_migration_l1",
        "switch_moved_capacity_fraction",
        "effective_support_jaccard_change",
    )
    scalar_int_keys = ("train_seed", "eval_seed", "episode", "step", "node_count", "raw_action_dim", "decode_rho")
    packed: Dict[str, np.ndarray] = {
        "schema_version": np.asarray([2], dtype=np.int16),
        "method": np.asarray([str(record.get("method", "")) for record in records], dtype=str),
        "profile": np.asarray([str(record.get("profile", "")) for record in records], dtype=str),
        "decode_score_calibration": np.asarray(
            [str(record.get("decode_score_calibration", "")) for record in records], dtype=str
        ),
        "decode_temperature_normalization": np.asarray(
            [str(record.get("decode_temperature_normalization", "")) for record in records], dtype=str
        ),
        "decode_score_calibration_degenerate": np.asarray(
            [bool(record.get("decode_score_calibration_degenerate", False)) for record in records],
            dtype=np.bool_,
        ),
        "is_initial_placement": np.asarray(
            [bool(record.get("is_initial_placement", False)) for record in records], dtype=np.bool_
        ),
        "allocation": _pad_trace_vectors(records, "allocation"),
        "raw_action": _pad_trace_vectors(records, "raw_action"),
        "valid_region_mask": np.empty((0, 0), dtype=np.bool_),
        "net_score": _pad_trace_vectors(records, "net_score"),
        "raw_net_score": _pad_trace_vectors(records, "raw_net_score"),
        "preprojection_score": _pad_trace_vectors(records, "preprojection_score"),
        "fused_score": _pad_trace_vectors(records, "fused_score"),
        "gates": _pad_trace_vectors(records, "gates", width=4),
        "gains": _pad_trace_vectors(records, "gains", width=4),
        "residual": _pad_trace_vectors(records, "residual", width=4),
    }
    if records:
        max_regions = packed["allocation"].shape[1]
        packed["valid_region_mask"] = np.stack(
            [
                np.arange(max_regions, dtype=np.int32) < int(record.get("node_count", 0))
                for record in records
            ],
            axis=0,
        )
    for key in scalar_float_keys:
        packed[key] = np.asarray([float(record.get(key, np.nan)) for record in records], dtype=np.float64)
    for key in scalar_int_keys:
        packed[key] = np.asarray([int(record.get(key, -1)) for record in records], dtype=np.int64)
    return packed


def save_compressed_traces(path: str | Path, trace_arrays: Dict[str, np.ndarray]):
    """Write trace arrays with compression and without Python object payloads."""
    path = Path(path)
    ensure_directory(path.parent)
    np.savez_compressed(path, **trace_arrays)


def initial_scenario_fingerprint(env: TheaterHexResourceEnv) -> str:
    """Hash the policy-independent simulator state immediately after reset."""
    digest = hashlib.sha256()
    digest.update(str(env.current_profile).encode("utf-8"))
    digest.update(str(int(env._last_reset_seed)).encode("ascii"))
    for name in (
        "type_id",
        "a_adj",
        "a_route",
        "a_dep",
        "delta",
        "kappa",
        "sigma",
        "backlog",
        "readiness",
        "demand",
        "route_edge_health",
        "loss_weights",
        "decoder_loss_weights",
    ):
        array = np.ascontiguousarray(np.asarray(getattr(env, name)))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_policy_or_heuristic(
    method_name: str,
    mode: str,
    env_factory: Callable[[], TheaterHexResourceEnv],
    action_fn: Callable[[np.ndarray, TheaterHexResourceEnv, Dict[str, Any]], np.ndarray],
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    train_seed: int = 0,
    record_cell_traces_for_first_episode: bool = False,
    record_all_cells_for_first_episode: bool = False,
    record_full_cell_traces: bool = False,
    record_compressed_traces: bool = False,
    collect_step_rows: bool = True,
    exact_zero_tolerance: float = 0.0,
) -> Dict[str, Any]:
    del mode
    episode_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    cell_rows: List[Dict[str, Any]] = []
    trace_records: List[Dict[str, Any]] = []
    trainable = method_is_trainable(method_name)

    for episode_idx, seed in enumerate(eval_seeds):
        env = env_factory()
        obs, info = env.reset(seed=int(seed))
        scenario_fingerprint = initial_scenario_fingerprint(env)
        first_capability = float(info["capability"])
        first_reference_capability = float(info.get("reference_capability", np.nan))
        total_reward = 0.0
        done = False
        episode_step_rows: List[Dict[str, Any]] = []
        previous_rho = np.nan
        previous_lambda = np.nan
        while not done:
            raw_action = np.asarray(action_fn(obs, env, info), dtype=np.float32).reshape(-1)
            previous_allocation = np.asarray(env.last_budget, dtype=np.float32).copy()
            is_initial_placement = int(env.t) == 0
            should_record_cells = bool(record_full_cell_traces) or (
                bool(record_cell_traces_for_first_episode or record_all_cells_for_first_episode)
                and episode_idx == 0
            )
            preview_required = should_record_cells or bool(record_compressed_traces)
            preview_decode: Dict[str, Any] = {}
            decoder_snapshot: Dict[str, Any] = {}
            if preview_required:
                if trainable:
                    _, preview_decode = env.decode_action(raw_action)
                else:
                    preview_decode = {"mode": method_name}
                decoder_snapshot = _decoder_snapshot(method_name, env, raw_action, preview_decode)
            if trainable:
                obs, reward, terminated, truncated, info = env.step(raw_action)
            else:
                obs, reward, terminated, truncated, info = env.step_budget(raw_action, {"mode": method_name})
            done = bool(terminated or truncated)
            total_reward += float(reward)
            profile_name = export_profile_name(str(info.get("profile", "")))
            allocation = np.asarray(info.get("budget", np.zeros(env.n)), dtype=np.float32)
            diagnostics = _allocation_diagnostics(
                allocation,
                previous_allocation,
                env,
                is_initial_placement=is_initial_placement,
                exact_zero_tolerance=exact_zero_tolerance,
            )
            decode = dict(getattr(env, "last_decode", {}))
            decode_rho = int(decode.get("rho", -1)) if decode.get("rho") is not None else -1
            decode_lambda = float(decode.get("lambda", np.nan)) if decode.get("lambda") is not None else np.nan
            decode_tau = float(decode.get("tau", np.nan)) if decode.get("tau") is not None else np.nan
            decode_gamma = float(decode.get("gamma", np.nan)) if decode.get("gamma") is not None else np.nan
            decode_mu = float(decode.get("mu", np.nan)) if decode.get("mu") is not None else np.nan
            alpha_values = _fixed_vector(decode.get("alpha", info.get("morphology_alpha")), 4)
            gain_values = _fixed_vector(decode.get("channel_gain"), 4)
            residual_values = _fixed_vector(decode.get("residual_coeff"), 4)
            switch_penalty_weight = float(env.config.switch_penalty_weight)
            switch_penalty = switch_penalty_weight * float(diagnostics["allocation_l1_migration"])
            row = {
                "method": method_name,
                "paper_label": method_label(method_name),
                "train_seed": train_seed,
                "eval_seed": int(seed),
                "episode": episode_idx,
                "step": int(info["t"]),
                "decision_step": int(info["t"]) - 1,
                "reward": float(reward),
                "capability": float(info["capability"]),
                "reference_capability": float(info.get("reference_capability", np.nan)),
                "total_loss": float(info["total_loss"]),
                "loss_critical": float(info["loss_key"]),
                "loss_support": float(info["loss_route"]),
                "loss_backlog": float(info["loss_backlog"]),
                "loss_e2e": float(info["loss_chain"]),
                "profile": profile_name,
                "loss_weight_critical": float(info.get("loss_weight_key", np.nan)),
                "loss_weight_support": float(info.get("loss_weight_route", np.nan)),
                "loss_weight_backlog": float(info.get("loss_weight_backlog", np.nan)),
                "loss_weight_e2e": float(info.get("loss_weight_chain", np.nan)),
                "decoder_weight_critical": float(info.get("decoder_weight_key", np.nan)),
                "decoder_weight_support": float(info.get("decoder_weight_route", np.nan)),
                "decoder_weight_backlog": float(info.get("decoder_weight_backlog", np.nan)),
                "decoder_weight_e2e": float(info.get("decoder_weight_chain", np.nan)),
                "lambda_zeta": float(info.get("lambda_zeta", np.nan)),
                "decoder_lambda_zeta": float(info.get("decoder_lambda_zeta", np.nan)),
                "reference_lambda_zeta": float(info.get("reference_lambda_zeta", np.nan)),
                "arrival_total": float(info.get("arrival_total", np.nan)),
                "arrival_mean": float(info.get("arrival_mean", np.nan)),
                "arrival_multiplier_mean": float(info.get("arrival_multiplier_mean", np.nan)),
                "arrival_multiplier_std": float(info.get("arrival_multiplier_std", np.nan)),
                "active_count": int(info["active_count"]),
                "active_ratio": float(info["active_ratio"]),
                "zero_ratio": float(info["zero_ratio"]),
                "budget_entropy": float(info["budget_entropy"]),
                "top_budget_share": float(info["top_budget_share"]),
                "switch_cost": float(info.get("switch_cost", 0.0)),
                "switch_penalty_weight": switch_penalty_weight,
                "switch_penalty_in_reward": switch_penalty,
                "initial_placement_switch_penalty": switch_penalty if is_initial_placement else 0.0,
                "reconfiguration_switch_penalty": 0.0 if is_initial_placement else switch_penalty,
                "positive_count": diagnostics["positive_count"],
                "positive_ratio": diagnostics["positive_ratio"],
                "exact_zero_count": diagnostics["exact_zero_count"],
                "exact_zero_ratio": diagnostics["exact_zero_ratio"],
                "effective_count": diagnostics["effective_count"],
                "effective_ratio": diagnostics["effective_ratio"],
                "subthreshold_positive_count": diagnostics["subthreshold_positive_count"],
                "subthreshold_positive_ratio": diagnostics["subthreshold_positive_ratio"],
                "subthreshold_capacity": diagnostics["subthreshold_capacity"],
                "subthreshold_capacity_share": diagnostics["subthreshold_capacity_share"],
                "exact_zero_tolerance": diagnostics["exact_zero_tolerance"],
                "effective_allocation_threshold": diagnostics["effective_allocation_threshold"],
                "is_initial_placement": diagnostics["is_initial_placement"],
                "initial_placement_l1": diagnostics["initial_placement_l1"],
                "switch_migration_l1": diagnostics["switch_migration_l1"],
                "switch_moved_capacity_fraction": diagnostics["switch_moved_capacity_fraction"],
                "allocation_l1_migration": diagnostics["allocation_l1_migration"],
                "support_flip_count": diagnostics["support_flip_count"],
                "support_flip_ratio": diagnostics["support_flip_ratio"],
                "effective_support_jaccard_change": diagnostics["effective_support_jaccard_change"],
                "decode_rho": decode_rho,
                "decode_rho_delta": float(decode_rho - previous_rho) if np.isfinite(previous_rho) and decode_rho >= 0 else np.nan,
                "decode_lambda": decode_lambda,
                "decode_lambda_delta": float(decode_lambda - previous_lambda)
                if np.isfinite(previous_lambda) and np.isfinite(decode_lambda)
                else np.nan,
                "decode_tau": decode_tau,
                "decode_gamma": decode_gamma,
                "decode_mu": decode_mu,
                "decode_projection": str(decode.get("projection", "")),
                "decode_gate_normalizer": str(decode.get("gate_normalizer", "")),
                "decode_score_calibration": str(decode.get("score_calibration", "")),
                "decode_score_center": float(decode.get("score_center", np.nan)),
                "decode_score_scale": float(decode.get("score_scale", np.nan)),
                "decode_score_scale_used": float(decode.get("score_scale_used", np.nan)),
                "decode_score_scale_floor": float(decode.get("score_scale_floor", np.nan)),
                "decode_score_calibration_degenerate": bool(
                    decode.get("score_calibration_degenerate", False)
                ),
                "decode_temperature_parameter": float(
                    decode.get("temperature_parameter", np.nan)
                ),
                "decode_temperature_effective": float(
                    decode.get("temperature_effective", np.nan)
                ),
                "decode_temperature_normalization": str(
                    decode.get("temperature_normalization", "")
                ),
                "decode_value_contrast": float(decode.get("value_contrast", np.nan)),
                "decode_low_contrast": bool(decode.get("low_contrast", False)),
            }
            alpha = info.get("morphology_alpha", [np.nan] * 4)
            desc = info.get("morphology_descriptors", [np.nan] * 4)
            for idx, old_name in enumerate(("key", "route", "backlog", "chain")):
                name = BOTTLENECK_EXPORT_NAMES[old_name]
                row[f"alpha_{name}"] = float(alpha[idx]) if idx < len(alpha) else np.nan
                row[f"morph_{name}"] = float(desc[idx]) if idx < len(desc) else np.nan
                row[f"gain_{name}"] = float(gain_values[idx])
                row[f"residual_{name}"] = float(residual_values[idx])
            if decoder_snapshot:
                finite_net = np.asarray(decoder_snapshot["net_score"], dtype=np.float32)
                finite_net = finite_net[np.isfinite(finite_net)]
                row["net_score_min"] = float(np.min(finite_net)) if finite_net.size else np.nan
                row["net_score_max"] = float(np.max(finite_net)) if finite_net.size else np.nan
                row["net_score_mean"] = float(np.mean(finite_net)) if finite_net.size else np.nan
                row["net_score_std"] = float(np.std(finite_net)) if finite_net.size else np.nan
            episode_step_rows.append(row)
            if collect_step_rows:
                step_rows.append(row)
            previous_rho = float(decode_rho) if decode_rho >= 0 else np.nan
            previous_lambda = decode_lambda

            if record_compressed_traces:
                trace_records.append(
                    {
                        "method": method_name,
                        "train_seed": train_seed,
                        "eval_seed": int(seed),
                        "episode": episode_idx,
                        "step": int(info["t"]),
                        "profile": profile_name,
                        "node_count": int(env.n),
                        "raw_action_dim": int(raw_action.size),
                        "raw_action": raw_action.copy(),
                        "allocation": allocation.copy(),
                        "decode_rho": decode_rho,
                        "decode_lambda": decode_lambda,
                        "decode_tau": decode_tau,
                        "decode_gamma": decode_gamma,
                        "decode_mu": decode_mu,
                        "decode_score_calibration": str(
                            decode.get("score_calibration", "")
                        ),
                        "decode_score_center": float(decode.get("score_center", np.nan)),
                        "decode_score_scale": float(decode.get("score_scale", np.nan)),
                        "decode_score_scale_used": float(
                            decode.get("score_scale_used", np.nan)
                        ),
                        "decode_score_scale_floor": float(
                            decode.get("score_scale_floor", np.nan)
                        ),
                        "decode_score_calibration_degenerate": bool(
                            decode.get("score_calibration_degenerate", False)
                        ),
                        "decode_temperature_parameter": float(
                            decode.get("temperature_parameter", np.nan)
                        ),
                        "decode_temperature_effective": float(
                            decode.get("temperature_effective", np.nan)
                        ),
                        "decode_temperature_normalization": str(
                            decode.get("temperature_normalization", "")
                        ),
                        "gates": alpha_values,
                        "gains": gain_values,
                        "residual": residual_values,
                        "fused_score": decoder_snapshot.get("fused_score", np.full(env.n, np.nan)),
                        "raw_net_score": decoder_snapshot.get(
                            "raw_net_score", np.full(env.n, np.nan)
                        ),
                        "preprojection_score": decoder_snapshot.get(
                            "preprojection_score", np.full(env.n, np.nan)
                        ),
                        "net_score": decoder_snapshot.get("net_score", np.full(env.n, np.nan)),
                        "exact_zero_ratio": diagnostics["exact_zero_ratio"],
                        "effective_ratio": diagnostics["effective_ratio"],
                        "subthreshold_positive_ratio": diagnostics["subthreshold_positive_ratio"],
                        "subthreshold_capacity_share": diagnostics["subthreshold_capacity_share"],
                        "allocation_l1_migration": diagnostics["allocation_l1_migration"],
                        "is_initial_placement": diagnostics["is_initial_placement"],
                        "initial_placement_l1": diagnostics["initial_placement_l1"],
                        "switch_migration_l1": diagnostics["switch_migration_l1"],
                        "switch_moved_capacity_fraction": diagnostics["switch_moved_capacity_fraction"],
                        "effective_support_jaccard_change": diagnostics["effective_support_jaccard_change"],
                    }
                )

            if should_record_cells:
                maps = decoder_snapshot.get("maps", {})
                fused_value = np.asarray(decoder_snapshot.get("fused_score", np.full(env.n, np.nan)), dtype=np.float32)
                net_value = np.asarray(decoder_snapshot.get("net_score", np.full(env.n, np.nan)), dtype=np.float32)
                for cell_idx in range(env.n):
                    if (
                        not record_full_cell_traces
                        and not record_all_cells_for_first_episode
                        and allocation[cell_idx] <= diagnostics["exact_zero_tolerance"]
                        and cell_idx % max(env.n // 80, 1) != 0
                    ):
                        continue
                    cell_rows.append(
                        {
                            "method": method_name,
                            "train_seed": train_seed,
                            "eval_seed": int(seed),
                            "profile": profile_name,
                            "step": int(info["t"]),
                            "decision_step": int(info["t"]) - 1,
                            "cell": int(cell_idx),
                            "node_type": int(env.type_id[cell_idx]),
                            "budget": float(allocation[cell_idx]),
                            "allocation": float(allocation[cell_idx]),
                            "is_positive_allocation": bool(diagnostics["positive_mask"][cell_idx]),
                            "is_exact_zero": bool(not diagnostics["positive_mask"][cell_idx]),
                            "is_effective_allocation": bool(diagnostics["effective_mask"][cell_idx]),
                            "is_subthreshold_positive": bool(diagnostics["subthreshold_mask"][cell_idx]),
                            "service_interruption": float(decoder_snapshot["service_interruption"][cell_idx]),
                            "upstream_support": float(decoder_snapshot["upstream_support"][cell_idx]),
                            "local_service": float(decoder_snapshot["local_service"][cell_idx]),
                            "backlog": float(decoder_snapshot["backlog_ratio"][cell_idx]),
                            "value_critical": float(np.asarray(maps.get("key", np.zeros(env.n)))[cell_idx]),
                            "value_support": float(np.asarray(maps.get("route", np.zeros(env.n)))[cell_idx]),
                            "value_backlog": float(np.asarray(maps.get("backlog", np.zeros(env.n)))[cell_idx]),
                            "value_e2e": float(np.asarray(maps.get("chain", np.zeros(env.n)))[cell_idx]),
                            "fused_value": float(fused_value[cell_idx]),
                            "net_value": float(net_value[cell_idx]),
                        }
                    )

        episode_rows.append(
            {
                "method": method_name,
                "paper_label": method_label(method_name),
                "train_seed": train_seed,
                "eval_seed": int(seed),
                "episode": episode_idx,
                "profile": export_profile_name(str(info.get("profile", ""))),
                "initial_scenario_sha256": scenario_fingerprint,
                "total_reward": total_reward,
                "initial_capability": first_capability,
                "initial_reference_capability": first_reference_capability,
                "final_capability": float(info["capability"]),
                "final_reference_capability": float(info.get("reference_capability", np.nan)),
                "cti": float(info["capability"] - first_capability),
                "reference_cti": float(info.get("reference_capability", np.nan) - first_reference_capability),
                # Retain the terminal objective at episode granularity so
                # episode-only evaluation artifacts remain sufficient for
                # summary statistics without a potentially very large step
                # CSV.  This is an output-only addition; it does not alter the
                # reward, transition, or policy execution path.
                "final_total_loss": float(info["total_loss"]),
                "task_utility": float(1.0 - float(info["total_loss"])),
                "critical_pressure": float(episode_step_rows[-1]["morph_critical"] if episode_step_rows else np.nan),
                "support_pressure": float(episode_step_rows[-1]["morph_support"] if episode_step_rows else np.nan),
                "mean_active_ratio": float(np.mean([r["active_ratio"] for r in episode_step_rows])),
                "mean_zero_ratio": float(np.mean([r["zero_ratio"] for r in episode_step_rows])),
                "mean_exact_zero_ratio": float(np.mean([r["exact_zero_ratio"] for r in episode_step_rows])),
                "mean_effective_ratio": float(np.mean([r["effective_ratio"] for r in episode_step_rows])),
                "mean_subthreshold_positive_ratio": float(
                    np.mean([r["subthreshold_positive_ratio"] for r in episode_step_rows])
                ),
                "mean_subthreshold_capacity_share": float(
                    np.mean([r["subthreshold_capacity_share"] for r in episode_step_rows])
                ),
                "mean_budget_entropy": float(np.mean([r["budget_entropy"] for r in episode_step_rows])),
                "mean_switch_cost": float(np.mean([r["switch_cost"] for r in episode_step_rows])),
                "mean_reconfiguration_switch_penalty": float(
                    np.mean([r["reconfiguration_switch_penalty"] for r in episode_step_rows[1:]])
                )
                if len(episode_step_rows) > 1
                else np.nan,
                "initial_placement_l1": float(episode_step_rows[0]["initial_placement_l1"]),
                "mean_switch_migration_l1": float(
                    np.nanmean([r["switch_migration_l1"] for r in episode_step_rows])
                )
                if len(episode_step_rows) > 1
                else np.nan,
                "mean_reconfiguration_l1": float(
                    np.nanmean([r["switch_migration_l1"] for r in episode_step_rows])
                )
                if len(episode_step_rows) > 1
                else np.nan,
                "mean_switch_moved_capacity_fraction": float(
                    np.nanmean([r["switch_moved_capacity_fraction"] for r in episode_step_rows])
                )
                if len(episode_step_rows) > 1
                else np.nan,
                "mean_effective_support_jaccard_change": float(
                    np.nanmean([r["effective_support_jaccard_change"] for r in episode_step_rows])
                )
                if len(episode_step_rows) > 1
                else np.nan,
                "rho_jump_rate": float(
                    np.mean(
                        [
                            abs(float(r["decode_rho_delta"])) > 0.0
                            for r in episode_step_rows[1:]
                            if np.isfinite(r["decode_rho_delta"])
                        ]
                    )
                )
                if any(np.isfinite(r["decode_rho_delta"]) for r in episode_step_rows[1:])
                else np.nan,
                "mean_abs_rho_delta": float(
                    np.mean(
                        [
                            abs(float(r["decode_rho_delta"]))
                            for r in episode_step_rows[1:]
                            if np.isfinite(r["decode_rho_delta"])
                        ]
                    )
                )
                if any(np.isfinite(r["decode_rho_delta"]) for r in episode_step_rows[1:])
                else np.nan,
                "lambda_total_variation": float(
                    np.sum(
                        [
                            abs(float(r["decode_lambda_delta"]))
                            for r in episode_step_rows[1:]
                            if np.isfinite(r["decode_lambda_delta"])
                        ]
                    )
                )
                if any(np.isfinite(r["decode_lambda_delta"]) for r in episode_step_rows[1:])
                else np.nan,
                "mean_abs_lambda_delta": float(
                    np.mean(
                        [
                            abs(float(r["decode_lambda_delta"]))
                            for r in episode_step_rows[1:]
                            if np.isfinite(r["decode_lambda_delta"])
                        ]
                    )
                )
                if any(np.isfinite(r["decode_lambda_delta"]) for r in episode_step_rows[1:])
                else np.nan,
                "mean_support_flip_ratio": float(
                    np.mean([r["support_flip_ratio"] for r in episode_step_rows[1:]])
                )
                if len(episode_step_rows) > 1
                else np.nan,
            }
        )
    return {
        "episode_rows": episode_rows,
        "step_rows": step_rows,
        "cell_rows": cell_rows,
        "trace_arrays": _pack_trace_records(trace_records),
    }


def summarize_episode_rows(
    method_name: str,
    mode: str,
    episode_rows: Sequence[Dict[str, Any]],
    train_seed: int = 0,
) -> Dict[str, Any]:
    del mode
    metrics = [
        "total_reward",
        "final_capability",
        "final_reference_capability",
        "cti",
        "reference_cti",
        "final_total_loss",
        "task_utility",
        "mean_active_ratio",
        "mean_zero_ratio",
        "mean_exact_zero_ratio",
        "mean_effective_ratio",
        "mean_subthreshold_positive_ratio",
        "mean_subthreshold_capacity_share",
        "mean_budget_entropy",
        "mean_switch_cost",
        "mean_reconfiguration_switch_penalty",
        "initial_placement_l1",
        "mean_switch_migration_l1",
        "mean_reconfiguration_l1",
        "mean_switch_moved_capacity_fraction",
        "mean_effective_support_jaccard_change",
        "rho_jump_rate",
        "mean_abs_rho_delta",
        "lambda_total_variation",
        "mean_abs_lambda_delta",
        "mean_support_flip_ratio",
    ]
    row: Dict[str, Any] = {
        "method": method_name,
        "paper_label": method_label(method_name),
        "train_seed": train_seed,
        "episodes": len(episode_rows),
    }
    for metric in metrics:
        vals = np.array([float(ep.get(metric, np.nan)) for ep in episode_rows], dtype=np.float64)
        finite_values = vals[np.isfinite(vals)]
        row[f"{metric}_mean"] = float(np.mean(finite_values)) if finite_values.size else np.nan
        row[f"{metric}_std"] = float(np.std(finite_values)) if finite_values.size else np.nan
    return row


def save_pickle(path: str | Path, payload: Any):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
