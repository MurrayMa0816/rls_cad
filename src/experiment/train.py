from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment.pipeline import (
    DEFAULT_EVAL_SEEDS,
    DEFAULT_MAIN_COLS,
    DEFAULT_MAIN_HORIZON,
    DEFAULT_MAIN_RESOURCE_BUDGET,
    DEFAULT_MAIN_ROWS,
    DEFAULT_MAIN_SCENARIO_ID,
    TRAINABLE_METHOD_SPECS,
    algorithm_class_for_method,
    algorithm_name_for_method,
    build_env_by_method,
    build_policy_and_kwargs,
    build_total_config,
    ensure_directory,
    evaluate_policy_or_heuristic,
    export_environment_static_artifacts,
    map_kwargs_from_config,
    maybe_import_wandb,
    method_is_trainable,
    require_dependency,
    save_json,
    summarize_episode_rows,
    write_rows_to_csv,
)
from experiment.callbacks import PeriodicEvalCallback


DEFAULT_DEBUG_METHOD = "graph_tmsbd_ppo"
MANIFEST_SCHEMA_VERSION = 1
PPO_ENTROPY_PROTOCOL_SCHEMA_VERSION = 1


def default_paper_runs_root() -> Path:
    env_root = os.environ.get("RLS_CAD_RUNS_ROOT")
    if env_root:
        return Path(env_root)
    return Path("runs")


def load_config_file(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    require_dependency("yaml", "pip install -r requirements.txt")
    import yaml

    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def lookup(config: dict[str, Any], key: str, default: Any = None, sections: tuple[str, ...] = ()) -> Any:
    if key in config:
        return config[key]
    for section_name in sections:
        section = config.get(section_name, {})
        if isinstance(section, dict) and key in section:
            return section[key]
    return default


def _integer_seed_list(values: Any, label: str) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{label} must be a YAML/JSON list of integers")
    seeds = [int(value) for value in values]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{label} contains duplicate seeds: {seeds}")
    return seeds


def _optional_four_weights(values: Any, label: str) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or len(values) != 4:
        raise TypeError(f"{label} must be null or a four-element YAML/JSON list")
    weights = [float(value) for value in values]
    if any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(f"{label} must contain finite, non-negative weights: {weights}")
    if sum(weights) <= 0.0:
        raise ValueError(f"{label} must contain at least one positive weight")
    return weights


def validate_seed_partitions(training_seeds: list[int], validation_seeds: list[int], test_seeds: list[int]) -> None:
    partitions = {
        "training_seeds": set(training_seeds),
        "validation_seeds": set(validation_seeds),
        "test_seeds": set(test_seeds),
    }
    for left_name, right_name in (
        ("training_seeds", "validation_seeds"),
        ("training_seeds", "test_seeds"),
        ("validation_seeds", "test_seeds"),
    ):
        overlap = sorted(partitions[left_name] & partitions[right_name])
        if overlap:
            raise ValueError(f"Seed leakage between {left_name} and {right_name}: {overlap}")


def resolve_ppo_entropy_protocol(
    *,
    algorithm_name: str,
    configured_ent_coef: float,
    dimension_normalization: bool,
    reference_action_dim: int,
    action_shape: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    """Resolve the coefficient applied to SB3's action-summed PPO entropy.

    Stable-Baselines3 sums diagonal-Gaussian entropy over action coordinates.
    With normalization enabled, ``configured_ent_coef`` is therefore the
    coefficient at ``reference_action_dim`` and the executed coefficient is
    scaled by ``reference_action_dim / action_dim``.  The action dimension is
    read from the constructed environment so map-size or interface changes
    cannot silently retain a coefficient calibrated for another action space.
    """

    configured_ent_coef = float(configured_ent_coef)
    reference_action_dim = int(reference_action_dim)
    dimensions = tuple(int(value) for value in action_shape)
    if not np.isfinite(configured_ent_coef) or configured_ent_coef < 0.0:
        raise ValueError("configured_ent_coef must be finite and non-negative")
    if reference_action_dim <= 0:
        raise ValueError("entropy_reference_action_dim must be positive")
    if not dimensions or any(value <= 0 for value in dimensions):
        raise ValueError(f"Action shape must contain positive dimensions: {dimensions}")
    action_dim = int(np.prod(dimensions, dtype=np.int64))
    is_ppo = str(algorithm_name).upper() == "PPO"
    applied = bool(is_ppo and dimension_normalization)
    scale_factor = float(reference_action_dim / action_dim) if applied else 1.0
    resolved_ent_coef = float(configured_ent_coef * scale_factor)
    return {
        "schema_version": PPO_ENTROPY_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": (
            "ppo_action_dim_normalized_entropy_v1"
            if applied
            else "fixed_entropy_coefficient_v1"
        ),
        "algorithm": str(algorithm_name),
        "distribution_entropy_reduction": "sum_over_action_dimensions",
        "dimension_normalization_configured": bool(dimension_normalization),
        "dimension_normalization_applied": applied,
        "configured_reference_ent_coef": configured_ent_coef,
        "ent_coef_base": configured_ent_coef,
        "reference_action_dim": reference_action_dim,
        "resolved_action_shape": list(dimensions),
        "resolved_action_dim": action_dim,
        "scale_factor": scale_factor,
        "resolved_ent_coef": resolved_ent_coef,
        "ent_coef_effective": resolved_ent_coef,
        "formula": (
            "configured_reference_ent_coef * reference_action_dim / resolved_action_dim"
            if applied
            else "configured_reference_ent_coef"
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "kind": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): manifest_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [manifest_safe(item) for item in value]
    if isinstance(value, type):
        return {"kind": "class", "qualified_name": f"{value.__module__}.{value.__qualname__}"}
    return {"kind": type(value).__name__, "repr": repr(value)}


def atomic_save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_json(temporary_path, payload)
    temporary_path.replace(path)


def build_source_snapshot(journal_root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for relative_root in ("src", "configs", "scripts", "docs/figures"):
        source_root = journal_root / relative_root
        if not source_root.exists():
            continue
        candidates.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".sh"}
        )
    files = [
        {"path": str(path.relative_to(journal_root)), "sha256": sha256_file(path)}
        for path in sorted(set(candidates))
    ]
    return {"aggregate_sha256": sha256_json(files), "files": files}


def collect_dependency_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package_name in ("numpy", "torch", "gymnasium", "stable-baselines3", "PyYAML"):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def collect_hardware_manifest() -> dict[str, Any]:
    import torch

    hardware: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        hardware["logical_cuda_0_name"] = torch.cuda.get_device_name(0)
        hardware["logical_cuda_0_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        hardware["nvidia_smi_physical_gpus"] = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        hardware["nvidia_smi_error"] = str(exc)
    return hardware


def enforce_physical_gpu0(device: str, required: bool) -> None:
    if not required:
        return
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible_devices != "0":
        raise RuntimeError(
            "Formal Paper03 training requires physical GPU 0: set CUDA_VISIBLE_DEVICES=0 "
            f"(received {visible_devices!r})."
        )
    if device != "cuda:0":
        raise RuntimeError(f"Formal Paper03 training requires --device cuda:0 (received {device!r}).")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "CUDA preflight failed after masking to physical GPU 0: expected exactly one visible CUDA device."
        )


def nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def require_free_disk(path: Path, minimum_free_disk_gb: float) -> dict[str, float | str]:
    existing_parent = nearest_existing_parent(path)
    usage = shutil.disk_usage(existing_parent)
    free_gib = usage.free / float(1024**3)
    if free_gib < minimum_free_disk_gb:
        raise RuntimeError(
            f"Insufficient free disk for formal training at {existing_parent}: "
            f"{free_gib:.2f} GiB available, {minimum_free_disk_gb:.2f} GiB required."
        )
    return {
        "checked_path": str(existing_parent),
        "free_gib_at_start": free_gib,
        "required_free_gib": float(minimum_free_disk_gb),
    }


def create_new_run_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(
            f"Refusing to reuse run directory {path}. Use a new timestamped directory; old artifacts are immutable."
        )
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_frozen_seed_manifest(
    manifest_path: str | None,
    test_suite_name: str,
    training_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds_payload = payload.get("seeds", payload)
    test_suites = seeds_payload.get("test_suites", {})
    frozen = {
        "training": [int(value) for value in seeds_payload.get("training", [])],
        "validation": [int(value) for value in seeds_payload.get("validation", [])],
        "test": [int(value) for value in test_suites.get(test_suite_name, [])],
    }
    requested = {
        "training": training_seeds,
        "validation": validation_seeds,
        "test": test_seeds,
    }
    if frozen != requested:
        raise ValueError(
            f"Run seed partitions do not match frozen manifest suite {test_suite_name!r}: "
            f"frozen={frozen}, requested={requested}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "test_suite_name": test_suite_name,
    }


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config_file(args.config)
    experiment_name = str(lookup(config, "experiment_name", "debug_tmsbd"))
    outdir = args.outdir or args.save_dir or lookup(config, "outdir", None) or lookup(config, "save_dir", None)
    if outdir is None:
        outdir = str(default_paper_runs_root() / experiment_name)

    args.method = args.method or str(lookup(config, "method", DEFAULT_DEBUG_METHOD, sections=("training",)))
    if not method_is_trainable(args.method):
        raise ValueError(f"Unsupported trainable method `{args.method}`. Choices: {sorted(TRAINABLE_METHOD_SPECS)}")
    args.mode = args.mode or str(lookup(config, "mode", "restore", sections=("environment",)))
    args.total_timesteps = int(args.total_timesteps or lookup(config, "total_timesteps", 1024, sections=("training",)))
    args.map_rows = int(args.map_rows or lookup(config, "map_rows", DEFAULT_MAIN_ROWS, sections=("environment",)))
    args.map_cols = int(args.map_cols or lookup(config, "map_cols", DEFAULT_MAIN_COLS, sections=("environment",)))
    args.horizon_steps = int(args.horizon_steps or lookup(config, "horizon_steps", DEFAULT_MAIN_HORIZON, sections=("environment",)))
    args.resource_budget = float(
        args.resource_budget or lookup(config, "resource_budget", DEFAULT_MAIN_RESOURCE_BUDGET, sections=("environment",))
    )
    args.scenario_id = args.scenario_id or str(
        lookup(config, "scenario_id", DEFAULT_MAIN_SCENARIO_ID, sections=("environment",))
    )
    args.profile = args.profile or str(lookup(config, "profile", "mixed", sections=("environment",)))
    args.max_sparse_support_fraction = float(
        args.max_sparse_support_fraction
        if args.max_sparse_support_fraction is not None
        else lookup(config, "max_sparse_support_fraction", 0.12, sections=("environment",))
    )
    args.tmsbd_action_dim = int(lookup(config, "tmsbd_action_dim", 15, sections=("environment",)))
    args.tmsbd_tau_multiplier = float(
        args.tmsbd_tau_multiplier
        if args.tmsbd_tau_multiplier is not None
        else lookup(config, "tmsbd_tau_multiplier", 1.25, sections=("environment",))
    )
    args.tmsbd_gamma_base = float(lookup(config, "tmsbd_gamma_base", 0.0, sections=("environment",)))
    args.tmsbd_gamma_scale = float(lookup(config, "tmsbd_gamma_scale", 1.0, sections=("environment",)))
    args.tmsbd_gamma_bias = float(lookup(config, "tmsbd_gamma_bias", 0.75, sections=("environment",)))
    args.tmsbd_gate_residual_scale = float(
        lookup(config, "tmsbd_gate_residual_scale", 0.25, sections=("environment",))
    )
    args.tmsbd_descriptor_mix = float(lookup(config, "tmsbd_descriptor_mix", 0.20, sections=("environment",)))
    args.tmsbd_channel_prior_strength = float(
        lookup(config, "tmsbd_channel_prior_strength", 0.18, sections=("environment",))
    )
    args.tmsbd_value_gain_floor = float(lookup(config, "tmsbd_value_gain_floor", 0.15, sections=("environment",)))
    args.tmsbd_value_gain_span = float(lookup(config, "tmsbd_value_gain_span", 1.30, sections=("environment",)))
    args.tmsbd_value_gate_bias = float(lookup(config, "tmsbd_value_gate_bias", 0.0, sections=("environment",)))
    args.tmsbd_value_residual_scale = float(
        lookup(config, "tmsbd_value_residual_scale", 0.25, sections=("environment",))
    )
    args.tmsbd_mu_scale = float(lookup(config, "tmsbd_mu_scale", 0.20, sections=("environment",)))
    args.tmsbd_mu_bias = float(lookup(config, "tmsbd_mu_bias", 2.50, sections=("environment",)))
    args.tmsbd_projection_min_contrast = float(
        lookup(config, "tmsbd_projection_min_contrast", 0.03, sections=("environment",))
    )
    args.switch_penalty_weight = float(
        args.switch_penalty_weight
        if args.switch_penalty_weight is not None
        else lookup(config, "switch_penalty_weight", 0.002, sections=("environment",))
    )
    args.route_damage_fraction = float(
        args.route_damage_fraction
        if args.route_damage_fraction is not None
        else lookup(config, "route_damage_fraction", 0.20, sections=("environment",))
    )
    args.stochastic_disturbance = bool(lookup(config, "stochastic_disturbance", False, sections=("environment",)))
    args.disturbance_noise = float(lookup(config, "disturbance_noise", 0.01, sections=("environment",)))
    args.initial_readiness_low = float(lookup(config, "initial_readiness_low", 0.75, sections=("environment",)))
    args.initial_readiness_high = float(lookup(config, "initial_readiness_high", 1.0, sections=("environment",)))
    args.budget_dilution_threshold = float(
        lookup(config, "budget_dilution_threshold", 0.0, sections=("environment",))
    )
    args.diffuse_budget_efficiency = float(
        lookup(config, "diffuse_budget_efficiency", 1.0, sections=("environment",))
    )
    args.active_support_penalty_weight = float(
        lookup(config, "active_support_penalty_weight", 0.0, sections=("environment",))
    )
    args.lambda_zeta = float(lookup(config, "lambda_zeta", 1.50, sections=("environment",)))
    args.decoder_lambda_zeta = float(
        lookup(config, "decoder_lambda_zeta", 1.50, sections=("environment",))
    )
    args.reference_lambda_zeta = float(
        lookup(config, "reference_lambda_zeta", 1.50, sections=("environment",))
    )
    args.arrival_base_rate = float(lookup(config, "arrival_base_rate", 0.06, sections=("environment",)))
    args.arrival_coefficient_of_variation = float(
        lookup(config, "arrival_coefficient_of_variation", 0.0, sections=("environment",))
    )
    args.objective_weight_override = _optional_four_weights(
        lookup(config, "objective_weight_override", None, sections=("environment",)),
        "objective_weight_override",
    )
    args.decoder_weight_override = _optional_four_weights(
        lookup(config, "decoder_weight_override", None, sections=("environment",)),
        "decoder_weight_override",
    )
    args.tmsbd_gate_softmax_temperature = float(
        lookup(config, "tmsbd_gate_softmax_temperature", 1.0, sections=("environment",))
    )
    args.min_effective_share = float(
        lookup(config, "min_effective_share", 1e-3, sections=("environment",))
    )
    args.signal_matched_score_calibration = str(
        lookup(
            config,
            "signal_matched_score_calibration",
            "center_unit_population_std_v1",
            sections=("environment",),
        )
    )
    args.signal_matched_score_scale_floor = float(
        lookup(
            config,
            "signal_matched_score_scale_floor",
            1.0e-6,
            sections=("environment",),
        )
    )
    args.signal_matched_sparse_tau_normalization = str(
        lookup(
            config,
            "signal_matched_sparse_tau_normalization",
            "per_region_quadratic_v1",
            sections=("environment",),
        )
    )
    if (
        not np.isfinite(args.signal_matched_score_scale_floor)
        or args.signal_matched_score_scale_floor <= 0.0
    ):
        raise ValueError("signal_matched_score_scale_floor must be finite and positive")
    args.direct_softmax_temperature = float(
        lookup(config, "direct_softmax_temperature", 1.0, sections=("environment",))
    )
    args.direct_sparse_projection_tau = float(
        lookup(config, "direct_sparse_projection_tau", 0.18, sections=("environment",))
    )
    args.topology_seed = int(lookup(config, "topology_seed", 0, sections=("environment",)))
    args.spatial_edge_drop_fraction = float(
        lookup(config, "spatial_edge_drop_fraction", 0.0, sections=("environment",))
    )
    args.edge_rewire_fraction = float(
        lookup(config, "edge_rewire_fraction", 0.0, sections=("environment",))
    )
    args.route_edge_drop_fraction = float(
        lookup(config, "route_edge_drop_fraction", 0.0, sections=("environment",))
    )
    args.support_edge_drop_fraction = float(
        lookup(config, "support_edge_drop_fraction", 0.0, sections=("environment",))
    )
    args.role_relocation_fraction = float(
        lookup(config, "role_relocation_fraction", 0.0, sections=("environment",))
    )
    args.support_providers_per_target = int(
        lookup(config, "support_providers_per_target", 3, sections=("environment",))
    )
    args.environment_kwargs = {
        key: getattr(args, key)
        for key in (
            "lambda_zeta",
            "decoder_lambda_zeta",
            "reference_lambda_zeta",
            "arrival_base_rate",
            "arrival_coefficient_of_variation",
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
    }
    args.seed = int(args.seed if args.seed is not None else lookup(config, "seed", 0, sections=("training",)))
    args.device = args.device or str(lookup(config, "device", "cpu", sections=("training",)))
    args.init_model = args.init_model or lookup(config, "init_model", None, sections=("training",))
    args.learning_rate = float(lookup(config, "learning_rate", 3e-4, sections=("training",)))
    args.n_steps = int(lookup(config, "n_steps", 128, sections=("training",)))
    args.batch_size = int(lookup(config, "batch_size", 64, sections=("training",)))
    args.gamma = float(lookup(config, "gamma", 0.99, sections=("training",)))
    args.gae_lambda = float(lookup(config, "gae_lambda", 0.95, sections=("training",)))
    args.clip_range = float(lookup(config, "clip_range", 0.2, sections=("training",)))
    cli_ent_coef = getattr(args, "ent_coef", None)
    args.ent_coef = float(
        cli_ent_coef
        if cli_ent_coef is not None
        else lookup(config, "ent_coef", 1e-3, sections=("training",))
    )
    configured_entropy_normalization = lookup(
        config,
        "entropy_dimension_normalization",
        True,
        sections=("training",),
    )
    if not isinstance(configured_entropy_normalization, bool):
        raise TypeError("entropy_dimension_normalization must be a YAML/JSON boolean")
    cli_entropy_normalization = getattr(args, "entropy_dimension_normalization", None)
    args.entropy_dimension_normalization = bool(
        configured_entropy_normalization
        if cli_entropy_normalization is None
        else cli_entropy_normalization
    )
    cli_reference_action_dim = getattr(args, "entropy_reference_action_dim", None)
    args.entropy_reference_action_dim = int(
        cli_reference_action_dim
        if cli_reference_action_dim is not None
        else lookup(config, "entropy_reference_action_dim", 15, sections=("training",))
    )
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0.0:
        raise ValueError("ent_coef must be finite and non-negative")
    if args.entropy_reference_action_dim <= 0:
        raise ValueError("entropy_reference_action_dim must be positive")
    args.periodic_eval_freq = int(
        args.periodic_eval_freq
        if args.periodic_eval_freq is not None
        else lookup(config, "periodic_eval_freq", 0, sections=("training", "evaluation"))
    )
    configured_training_seeds = lookup(config, "training_seeds", None, sections=("training",))
    args.training_seeds = _integer_seed_list(
        args.training_seeds if args.training_seeds is not None else configured_training_seeds,
        "training_seeds",
    )
    if not args.training_seeds:
        args.training_seeds = [args.seed]
    if configured_training_seeds is not None and args.seed not in args.training_seeds:
        raise ValueError(f"Training seed {args.seed} is not declared in training_seeds={args.training_seeds}")
    legacy_eval_seeds = lookup(config, "eval_seeds", None, sections=("evaluation",))
    configured_validation_seeds = lookup(config, "validation_seeds", None, sections=("evaluation",))
    configured_test_seeds = lookup(config, "test_seeds", legacy_eval_seeds, sections=("evaluation",))
    args.validation_seeds = _integer_seed_list(
        args.validation_seeds if args.validation_seeds is not None else configured_validation_seeds,
        "validation_seeds",
    )
    args.test_seeds = _integer_seed_list(
        args.test_seeds
        if args.test_seeds is not None
        else (configured_test_seeds if configured_test_seeds is not None else list(DEFAULT_EVAL_SEEDS)),
        "test_seeds",
    )
    validate_seed_partitions(args.training_seeds, args.validation_seeds, args.test_seeds)
    if args.periodic_eval_freq > 0 and not args.validation_seeds:
        raise ValueError("periodic_eval_freq > 0 requires a non-empty, held-out validation_seeds list")
    args.checkpoint_selection = str(
        lookup(config, "checkpoint_selection", "final", sections=("training", "evaluation"))
    )
    if args.checkpoint_selection not in {"final", "best_validation"}:
        raise ValueError("checkpoint_selection must be `final` or `best_validation`")
    args.minimum_free_disk_gb = float(
        lookup(config, "minimum_free_disk_gb", 0.0, sections=("training", "outputs"))
    )
    args.require_physical_gpu0 = bool(
        lookup(config, "require_physical_gpu0", False, sections=("training",))
    )
    if args.require_physical_gpu0 and args.minimum_free_disk_gb < 10.0:
        raise ValueError("Formal GPU0 training requires minimum_free_disk_gb >= 10.0")
    args.training_progress_csv = lookup(
        config,
        "training_progress_csv",
        False,
        sections=("training", "outputs"),
    )
    if not isinstance(args.training_progress_csv, bool):
        raise TypeError("training_progress_csv must be a YAML/JSON boolean")
    args.export_static_env = bool(lookup(config, "export_static_env", False, sections=("outputs",)))
    args.use_wandb = bool(lookup(config, "use_wandb", False, sections=("wandb",)))
    args.outdir = str(outdir)
    args.config_payload = config
    args.experiment_name = experiment_name
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLS-CAD.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--map-rows", type=int, default=None)
    parser.add_argument("--map-cols", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--resource-budget", type=float, default=None)
    parser.add_argument("--scenario-id", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--max-sparse-support-fraction", type=float, default=None)
    parser.add_argument("--tmsbd-tau-multiplier", type=float, default=None)
    parser.add_argument("--switch-penalty-weight", type=float, default=None)
    parser.add_argument("--route-damage-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init-model", type=str, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument(
        "--entropy-dimension-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Scale PPO ent_coef by reference_action_dim / resolved_action_dim.",
    )
    parser.add_argument("--entropy-reference-action-dim", type=int, default=None)
    parser.add_argument("--periodic-eval-freq", type=int, default=None)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=None)
    return parser.parse_args()


def make_run_config(args: argparse.Namespace, ppo_entropy_protocol: dict[str, Any]) -> dict[str, Any]:
    run_config = {
        "paper": "paper03",
        "workspace": "journal",
        "experiment_name": args.experiment_name,
        "method": args.method,
        "mode": args.mode,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "map_rows": args.map_rows,
        "map_cols": args.map_cols,
        "horizon_steps": args.horizon_steps,
        "resource_budget": args.resource_budget,
        "scenario_id": args.scenario_id,
        "profile": args.profile,
        "max_sparse_support_fraction": args.max_sparse_support_fraction,
        "tmsbd_action_dim": args.tmsbd_action_dim,
        "tmsbd_tau_multiplier": args.tmsbd_tau_multiplier,
        "tmsbd_gamma_base": args.tmsbd_gamma_base,
        "tmsbd_gamma_scale": args.tmsbd_gamma_scale,
        "tmsbd_gamma_bias": args.tmsbd_gamma_bias,
        "tmsbd_gate_residual_scale": args.tmsbd_gate_residual_scale,
        "tmsbd_descriptor_mix": args.tmsbd_descriptor_mix,
        "tmsbd_channel_prior_strength": args.tmsbd_channel_prior_strength,
        "tmsbd_value_gain_floor": args.tmsbd_value_gain_floor,
        "tmsbd_value_gain_span": args.tmsbd_value_gain_span,
        "tmsbd_value_gate_bias": args.tmsbd_value_gate_bias,
        "tmsbd_value_residual_scale": args.tmsbd_value_residual_scale,
        "tmsbd_mu_scale": args.tmsbd_mu_scale,
        "tmsbd_mu_bias": args.tmsbd_mu_bias,
        "tmsbd_projection_min_contrast": args.tmsbd_projection_min_contrast,
        "switch_penalty_weight": args.switch_penalty_weight,
        "route_damage_fraction": args.route_damage_fraction,
        "stochastic_disturbance": args.stochastic_disturbance,
        "disturbance_noise": args.disturbance_noise,
        "initial_readiness_low": args.initial_readiness_low,
        "initial_readiness_high": args.initial_readiness_high,
        "budget_dilution_threshold": args.budget_dilution_threshold,
        "diffuse_budget_efficiency": args.diffuse_budget_efficiency,
        "active_support_penalty_weight": args.active_support_penalty_weight,
        "run_purpose": os.environ.get("PAPER03_RUN_PURPOSE", "formal_main"),
        "transfer_protocol": os.environ.get("PAPER03_TRANSFER_PROTOCOL", "not_applicable"),
        "condition_id": os.environ.get("PAPER03_CONDITION_ID", "baseline"),
        "device": args.device,
        "init_model": args.init_model,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        # ``ent_coef`` is the coefficient actually passed to PPO.  The
        # configured 15-dimensional reference value and the full resolution
        # rule are retained separately for auditability.
        "ent_coef": ppo_entropy_protocol["resolved_ent_coef"],
        "ent_coef_base": ppo_entropy_protocol["ent_coef_base"],
        "ent_coef_effective": ppo_entropy_protocol["ent_coef_effective"],
        "configured_reference_ent_coef": args.ent_coef,
        "entropy_dimension_normalization": args.entropy_dimension_normalization,
        "entropy_reference_action_dim": args.entropy_reference_action_dim,
        "ppo_entropy_protocol": ppo_entropy_protocol,
        "periodic_eval_freq": args.periodic_eval_freq,
        "training_seeds": args.training_seeds,
        "validation_seeds": args.validation_seeds,
        "test_seeds": args.test_seeds,
        "checkpoint_selection": args.checkpoint_selection,
        "minimum_free_disk_gb": args.minimum_free_disk_gb,
        "require_physical_gpu0": args.require_physical_gpu0,
        "training_progress_csv": args.training_progress_csv,
    }
    run_config.update(args.environment_kwargs)
    return run_config


def main():
    args = resolve_args(parse_args())
    if args.method != "graph_tmsbd_ppo":
        raise ValueError("This release trains only the RLS-CAD method.")
    enforce_physical_gpu0(args.device, args.require_physical_gpu0)
    disk_preflight = require_free_disk(Path(args.outdir), args.minimum_free_disk_gb)
    sb3 = require_dependency("stable_baselines3", "pip install -r requirements.txt")
    model_cls = algorithm_class_for_method(args.method)
    algorithm_name = algorithm_name_for_method(args.method)
    journal_root = Path(__file__).resolve().parents[2]
    source_snapshot = build_source_snapshot(journal_root)
    expected_source_hash = os.environ.get("RLS_CAD_EXPECTED_SOURCE_SHA256")
    if expected_source_hash and source_snapshot["aggregate_sha256"] != expected_source_hash:
        raise RuntimeError(
            "The source or configuration changed after the run was prepared."
        )
    frozen_seed_manifest = verify_frozen_seed_manifest(
        os.environ.get("RLS_CAD_SEED_MANIFEST"),
        os.environ.get("RLS_CAD_TEST_SUITE_NAME", "in_distribution"),
        args.training_seeds,
        args.validation_seeds,
        args.test_seeds,
    )

    run_dir = create_new_run_directory(Path(args.outdir))
    model_dir = ensure_directory(run_dir / "models")
    eval_dir = ensure_directory(run_dir / "evaluation")
    training_log_dir = run_dir / "training"
    if args.training_progress_csv:
        training_log_dir = ensure_directory(training_log_dir)

    total_config = build_total_config(
        mode=args.mode,
        map_rows=args.map_rows,
        map_cols=args.map_cols,
        horizon_steps=args.horizon_steps,
        resource_budget=args.resource_budget,
        scenario_id=args.scenario_id,
        seed_value=args.seed,
        profile=args.profile,
        max_sparse_support_fraction=args.max_sparse_support_fraction,
        tmsbd_action_dim=args.tmsbd_action_dim,
        tmsbd_tau_multiplier=args.tmsbd_tau_multiplier,
        tmsbd_gamma_base=args.tmsbd_gamma_base,
        tmsbd_gamma_scale=args.tmsbd_gamma_scale,
        tmsbd_gamma_bias=args.tmsbd_gamma_bias,
        tmsbd_gate_residual_scale=args.tmsbd_gate_residual_scale,
        tmsbd_descriptor_mix=args.tmsbd_descriptor_mix,
        tmsbd_channel_prior_strength=args.tmsbd_channel_prior_strength,
        tmsbd_value_gain_floor=args.tmsbd_value_gain_floor,
        tmsbd_value_gain_span=args.tmsbd_value_gain_span,
        tmsbd_value_gate_bias=args.tmsbd_value_gate_bias,
        tmsbd_value_residual_scale=args.tmsbd_value_residual_scale,
        tmsbd_mu_scale=args.tmsbd_mu_scale,
        tmsbd_mu_bias=args.tmsbd_mu_bias,
        tmsbd_projection_min_contrast=args.tmsbd_projection_min_contrast,
        switch_penalty_weight=args.switch_penalty_weight,
        route_damage_fraction=args.route_damage_fraction,
        stochastic_disturbance=args.stochastic_disturbance,
        disturbance_noise=args.disturbance_noise,
        initial_readiness_low=args.initial_readiness_low,
        initial_readiness_high=args.initial_readiness_high,
        budget_dilution_threshold=args.budget_dilution_threshold,
        diffuse_budget_efficiency=args.diffuse_budget_efficiency,
        active_support_penalty_weight=args.active_support_penalty_weight,
        **args.environment_kwargs,
    )
    env = build_env_by_method(args.method, copy.deepcopy(total_config))
    ppo_entropy_protocol = resolve_ppo_entropy_protocol(
        algorithm_name=algorithm_name,
        configured_ent_coef=args.ent_coef,
        dimension_normalization=args.entropy_dimension_normalization,
        reference_action_dim=args.entropy_reference_action_dim,
        action_shape=env.action_space.shape,
    )
    if args.export_static_env:
        export_environment_static_artifacts(env, run_dir)

    run_config = make_run_config(args, ppo_entropy_protocol)
    save_json(run_dir / "run_config.json", run_config)
    save_json(run_dir / "config_resolved.json", {"run_config": run_config, "config_payload": args.config_payload})
    manifest_path = run_dir / "run_manifest.json"
    dependency_manifest = collect_dependency_manifest()
    hardware_manifest = collect_hardware_manifest()
    run_manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_identity": {
            "paper": "paper03",
            "experiment_name": args.experiment_name,
            "method": args.method,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "mode": args.mode,
            "run_purpose": run_config["run_purpose"],
            "condition_id": run_config["condition_id"],
            "switch_penalty_weight": args.switch_penalty_weight,
        },
        "seed_partitions": {
            "training": args.training_seeds,
            "validation": args.validation_seeds,
            "test": args.test_seeds,
        },
        "frozen_seed_manifest": frozen_seed_manifest,
        "checkpoint_selection": args.checkpoint_selection,
        "ppo_entropy_protocol": ppo_entropy_protocol,
        "training_progress": {
            "enabled": args.training_progress_csv,
            "path": "training/progress.csv" if args.training_progress_csv else None,
            "logger_outputs": ["stdout", "csv"] if args.training_progress_csv else ["stdout"],
        },
        "disk_preflight": disk_preflight,
        "source_snapshot": source_snapshot,
        "suite_expected_source_sha256": expected_source_hash,
        "resolved_config_sha256": sha256_file(run_dir / "config_resolved.json"),
        "dependencies": dependency_manifest,
        "dependencies_sha256": sha256_json(dependency_manifest),
        "hardware": hardware_manifest,
        "hardware_sha256": sha256_json(hardware_manifest),
    }
    atomic_save_json(manifest_path, run_manifest)

    wandb_run = None
    if args.use_wandb:
        wandb = maybe_import_wandb()
        if wandb is not None:
            wandb_run = wandb.init(project="rls_cad", config=run_config, tags=["rls-cad"])

    policy, policy_kwargs = build_policy_and_kwargs(args.method, env)
    if args.init_model:
        model = model_cls.load(str(args.init_model), env=env, device=args.device)
        model.verbose = 1
        if algorithm_name == "PPO":
            model.ent_coef = ppo_entropy_protocol["resolved_ent_coef"]
    else:
        common_kwargs = {
            "policy": policy,
            "env": env,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "policy_kwargs": policy_kwargs,
            "seed": args.seed,
            "verbose": 1,
            "device": args.device,
        }
        if algorithm_name == "PPO":
            model = model_cls(
                **common_kwargs,
                n_steps=args.n_steps,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=ppo_entropy_protocol["resolved_ent_coef"],
            )
        elif algorithm_name == "SAC":
            model = model_cls(
                **common_kwargs,
                ent_coef="auto",
                learning_starts=max(1000, args.n_steps * 4),
                train_freq=1,
                gradient_steps=1,
                buffer_size=200000,
            )
        elif algorithm_name == "TD3":
            action_dim = int(np.prod(env.action_space.shape))
            action_noise = sb3.common.noise.NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=0.12 * np.ones(action_dim),
            )
            model = model_cls(
                **common_kwargs,
                action_noise=action_noise,
                learning_starts=max(1000, args.n_steps * 4),
                train_freq=1,
                gradient_steps=1,
                buffer_size=200000,
            )
        else:
            raise ValueError(f"Unsupported algorithm `{algorithm_name}` for method `{args.method}`")
    if args.training_progress_csv:
        model.set_logger(sb3.common.logger.configure(str(training_log_dir), ["stdout", "csv"]))
    total_parameters = int(sum(parameter.numel() for parameter in model.policy.parameters()))
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad)
    )
    safe_policy_kwargs = manifest_safe(policy_kwargs)
    run_manifest["model"] = {
        "algorithm": algorithm_name,
        "policy_class": f"{model.policy.__class__.__module__}.{model.policy.__class__.__qualname__}",
        "policy_kwargs": safe_policy_kwargs,
        "policy_kwargs_sha256": sha256_json(safe_policy_kwargs),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "action_shape": list(env.action_space.shape),
        "observation_shape": list(env.observation_space.shape),
    }
    atomic_save_json(manifest_path, run_manifest)
    callback = None
    if args.periodic_eval_freq > 0:
        eval_history_config = build_total_config(
            mode=args.mode,
            map_rows=args.map_rows,
            map_cols=args.map_cols,
            horizon_steps=args.horizon_steps,
            resource_budget=args.resource_budget,
            scenario_id=args.scenario_id,
            seed_value=args.seed + 1000,
            profile=args.profile,
            **map_kwargs_from_config(run_config),
        )
        callback = PeriodicEvalCallback(
            method_name=args.method,
            mode=args.mode,
            train_seed=args.seed,
            eval_env_factory=lambda: build_env_by_method(args.method, copy.deepcopy(eval_history_config)),
            eval_seeds=args.validation_seeds,
            eval_freq=args.periodic_eval_freq,
            output_dir=eval_dir,
            deterministic=True,
            minimum_eval_timesteps=args.periodic_eval_freq,
        )

    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=callback)
    final_model_path = model_dir / "final_policy_model.zip"
    model.save(str(final_model_path))
    best_model_path = eval_dir / "best_policy_model.zip"
    if args.checkpoint_selection == "best_validation":
        if not best_model_path.exists():
            raise RuntimeError("No eligible validation checkpoint was produced; refusing 0-step/fallback selection")
        selected_model = model_cls.load(str(best_model_path), env=env, device=args.device)
        selected_model_path = best_model_path
    else:
        selected_model = model
        selected_model_path = final_model_path
    save_json(
        run_dir / "selected_model.json",
        {
            "selection_rule": args.checkpoint_selection,
            "path": str(selected_model_path.relative_to(run_dir)),
            "sha256": sha256_file(selected_model_path),
        },
    )

    eval_config = build_total_config(
        mode=args.mode,
        map_rows=args.map_rows,
        map_cols=args.map_cols,
        horizon_steps=args.horizon_steps,
        resource_budget=args.resource_budget,
        scenario_id=args.scenario_id,
        seed_value=args.seed + 1000,
        profile=args.profile,
        **map_kwargs_from_config(run_config),
    )
    def env_factory():
        return build_env_by_method(args.method, copy.deepcopy(eval_config))
    results = evaluate_policy_or_heuristic(
        method_name=args.method,
        mode=args.mode,
        env_factory=env_factory,
        action_fn=lambda obs, _env, _info: selected_model.predict(obs, deterministic=True)[0],
        # This per-run report is a post-training diagnostic, not the formal
        # paper test.  Keeping it on the frozen validation partition ensures
        # that the 100-instance test suite is opened only by evaluate.py after
        # every method/seed has finished and the protocol is frozen.
        eval_seeds=args.validation_seeds,
        train_seed=args.seed,
        record_cell_traces_for_first_episode=True,
    )
    summary = summarize_episode_rows(args.method, args.mode, results["episode_rows"], train_seed=args.seed)
    summary["evaluation_partition"] = "validation_post_training_diagnostic"
    summary["evaluation_seeds"] = [int(seed) for seed in args.validation_seeds]
    save_json(eval_dir / "evaluation_summary.json", summary)
    save_json(eval_dir / "evaluation_episodes.json", results["episode_rows"])
    write_rows_to_csv(eval_dir / "episode_metrics.csv", results["episode_rows"])
    write_rows_to_csv(eval_dir / "step_metrics.csv", results["step_rows"])
    write_rows_to_csv(eval_dir / "cell_traces.csv", results["cell_rows"])

    with (run_dir / "evaluation_report.txt").open("w", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(summary, ensure_ascii=False, indent=2))
        file_obj.write("\n")
    if wandb_run is not None:
        wandb_run.finish()
    run_manifest["status"] = "completed"
    run_manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    run_manifest["selected_model"] = {
        "selection_rule": args.checkpoint_selection,
        "path": str(selected_model_path.relative_to(run_dir)),
        "sha256": sha256_file(selected_model_path),
    }
    run_manifest["post_training_diagnostic"] = {
        "partition": "validation",
        "seeds": [int(seed) for seed in args.validation_seeds],
        "note": "The frozen formal test suite is evaluated only by experiment/evaluate.py.",
    }
    artifacts = {
        "final_model": {
            "path": str(final_model_path.relative_to(run_dir)),
            "sha256": sha256_file(final_model_path),
        },
        "evaluation_summary": {
            "path": "evaluation/evaluation_summary.json",
            "sha256": sha256_file(eval_dir / "evaluation_summary.json"),
        },
        "run_config": {"path": "run_config.json", "sha256": sha256_file(run_dir / "run_config.json")},
    }
    training_progress_path = training_log_dir / "progress.csv"
    if args.training_progress_csv:
        if not training_progress_path.is_file() or training_progress_path.stat().st_size <= 0:
            raise RuntimeError("training_progress_csv was enabled but training/progress.csv is missing or empty")
        artifacts["training_progress"] = {
            "path": str(training_progress_path.relative_to(run_dir)),
            "sha256": sha256_file(training_progress_path),
        }
    run_manifest["artifacts"] = artifacts
    atomic_save_json(manifest_path, run_manifest)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
