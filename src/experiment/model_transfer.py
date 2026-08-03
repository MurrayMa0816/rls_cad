"""Evaluation-only graph synchronization and fixed-action policy transfer.

This module intentionally does not implement training resume.  Optimizer,
rollout-buffer, and environment RNG state are never migrated.  It supports:

* same-size topology synchronization for ``TypedGraphFeatureExtractor``;
* cross-size reconstruction of the fixed 15-dimensional RLS decoder policy.

Policies with region-dependent ``N`` action heads are rejected
before any state is copied.  In particular, this module never truncates,
repeats, pads, or resizes an action to make incompatible policies appear to
work on another graph size.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from algorithms.graph_features import TypedGraphFeatureExtractor


FIXED_15D_TYPED_GRAPH_METHODS = frozenset(
    {
        "graph_tmsbd_ppo",
        "tmsbd_no_gate_ppo",
        "tmsbd_no_chain_ppo",
        "tmsbd_softmax_gate_ppo",
        "tmsbd_softmax_budget_ppo",
        "tmsbd_fixed_topk_ppo",
        "tmsbd_fixed_morphology_ppo",
        "tmsbd_single_critical_ppo",
        "tmsbd_single_support_ppo",
        "tmsbd_single_backlog_ppo",
        "tmsbd_single_e2e_ppo",
        "lts_scp_latent_ppo",
    }
)

REGION_DEPENDENT_HEAD_METHODS = frozenset(
    {
        "direct_softmax_ppo",
        "mlp_softmax_ppo",
        "mlp_dirichlet_ppo",
        "direct_softmax_sac",
        "direct_softmax_td3",
        "direct_sparse_projection_ppo",
        "direct_sparse_projection_sac",
        "direct_sparse_projection_td3",
        "signal_matched_softmax_ppo",
        "signal_matched_sparse_projection_ppo",
    }
)

GRAPH_EDGE_TYPES = ("adj", "route", "dep")
FIXED_ACTION_DIMENSION = 15


class StructuralTransferError(RuntimeError):
    """Raised when a policy cannot be transferred without changing its meaning."""


@dataclass(frozen=True)
class GraphSyncReport:
    source_num_cells: int
    target_num_cells: int
    unique_extractors_updated: int
    edge_types: tuple[str, ...]
    before_hashes: dict[str, tuple[str, ...]]
    target_hashes: dict[str, str]
    after_hashes: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyTransferReport:
    method: str
    source_num_cells: int
    target_num_cells: int
    source_observation_shape: tuple[int, ...]
    target_observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    copied_tensor_keys: tuple[str, ...]
    target_graph_buffer_keys: tuple[str, ...]
    target_graph_hashes: dict[str, str]
    source_checkpoint_sha256: str | None = None
    evaluation_only: bool = True
    optimizer_transferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_array(values: np.ndarray | th.Tensor) -> str:
    if isinstance(values, th.Tensor):
        array = values.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def typed_graph_adjacencies_from_env(env: Any) -> dict[str, np.ndarray]:
    """Return the exact three graph views used by the typed graph encoder."""

    required = ("a_adj", "a_route", "a_dep", "n")
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise StructuralTransferError(f"Target environment lacks graph fields: {missing}")
    num_cells = int(env.n)
    spatial = np.asarray(env.a_adj, dtype=np.float32)
    route = np.asarray(env.a_route, dtype=np.float32)
    dependency_directed = np.asarray(env.a_dep, dtype=np.float32)
    matrices = {
        "adj": spatial,
        "route": route,
        # This matches build_policy_and_kwargs: message passing uses the
        # undirected support-provider view while the environment retains the
        # directed access-backhaul dependency relation.
        "dep": np.maximum(dependency_directed, dependency_directed.T).astype(np.float32),
    }
    for edge_type, matrix in matrices.items():
        if matrix.shape != (num_cells, num_cells):
            raise StructuralTransferError(
                f"Environment graph {edge_type!r} has shape {matrix.shape}; "
                f"expected {(num_cells, num_cells)}"
            )
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise StructuralTransferError(
                f"Environment graph {edge_type!r} must be finite and nonnegative"
            )
    return matrices


def _as_policy(model_or_policy: Any) -> Any:
    return getattr(model_or_policy, "policy", model_or_policy)


def iter_unique_typed_graph_extractors(model_or_policy: Any) -> tuple[TypedGraphFeatureExtractor, ...]:
    """Find shared or separate actor/critic graph extractors exactly once."""

    policy = _as_policy(model_or_policy)
    extractors: list[TypedGraphFeatureExtractor] = []
    seen: set[int] = set()
    for attribute in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(policy, attribute, None)
        if not isinstance(extractor, TypedGraphFeatureExtractor) or id(extractor) in seen:
            continue
        seen.add(id(extractor))
        extractors.append(extractor)
    return tuple(extractors)


def synchronize_same_size_typed_graph(model_or_policy: Any, target_env: Any) -> GraphSyncReport:
    """Synchronize every unique typed graph extractor with a same-size env.

    Validation is completed for every extractor before the first buffer is
    committed, preventing a shared/separate actor-critic policy from ending in
    a partially synchronized state.
    """

    policy = _as_policy(model_or_policy)
    extractors = iter_unique_typed_graph_extractors(policy)
    if not extractors:
        raise StructuralTransferError("Policy does not contain a TypedGraphFeatureExtractor")
    if _space_shape(policy.observation_space) != _space_shape(target_env.observation_space):
        raise StructuralTransferError(
            "Same-size graph synchronization requires identical observation-space shapes"
        )
    if _space_shape(policy.action_space) != _space_shape(target_env.action_space):
        raise StructuralTransferError(
            "Same-size graph synchronization requires identical action-space shapes"
        )
    matrices = typed_graph_adjacencies_from_env(target_env)
    target_num_cells = int(target_env.n)

    # Prevalidate every extractor before mutating any of them.
    for extractor in extractors:
        if extractor.num_cells != target_num_cells:
            raise StructuralTransferError(
                f"Same-size synchronization cannot change N: extractor has {extractor.num_cells}, "
                f"target environment has {target_num_cells}"
            )
        if int(extractor.cell_feature_dim) != int(target_env.node_feature_dim):
            raise StructuralTransferError(
                "Cell feature dimensions differ between checkpoint and target environment"
            )
        if int(extractor.global_feature_dim) != int(target_env.global_feature_dim):
            raise StructuralTransferError(
                "Global feature dimensions differ between checkpoint and target environment"
            )
        if set(extractor.edge_types) != set(matrices):
            raise StructuralTransferError(
                f"Edge types differ: extractor={sorted(extractor.edge_types)}, "
                f"target={sorted(matrices)}"
            )
        for edge_type in extractor.edge_types:
            normalized = extractor._normalize_adjacency(matrices[edge_type])
            if normalized.shape != (target_num_cells, target_num_cells):
                raise StructuralTransferError(
                    f"Graph {edge_type!r} cannot be synchronized at N={target_num_cells}"
                )

    before_hashes = {
        edge_type: tuple(
            _sha256_array(getattr(extractor, f"adjacency_{edge_type}"))
            for extractor in extractors
        )
        for edge_type in GRAPH_EDGE_TYPES
    }
    for extractor in extractors:
        extractor.set_adjacency_matrices(matrices)
    target_hashes = {
        edge_type: _sha256_array(extractors[0]._normalize_adjacency(matrices[edge_type]))
        for edge_type in GRAPH_EDGE_TYPES
    }
    after_hashes = {
        edge_type: tuple(
            _sha256_array(getattr(extractor, f"adjacency_{edge_type}"))
            for extractor in extractors
        )
        for edge_type in GRAPH_EDGE_TYPES
    }
    for edge_type in GRAPH_EDGE_TYPES:
        if any(value != target_hashes[edge_type] for value in after_hashes[edge_type]):
            raise StructuralTransferError(f"Graph synchronization verification failed for {edge_type!r}")

    return GraphSyncReport(
        source_num_cells=int(extractors[0].num_cells),
        target_num_cells=target_num_cells,
        unique_extractors_updated=len(extractors),
        edge_types=tuple(extractors[0].edge_types),
        before_hashes=before_hashes,
        target_hashes=target_hashes,
        after_hashes=after_hashes,
    )


def _space_shape(space: Any) -> tuple[int, ...]:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise StructuralTransferError(f"Expected a fixed-shape space, received {space!r}")
    return tuple(int(value) for value in shape)


def _is_graph_buffer_key(key: str) -> bool:
    return key.rsplit(".", 1)[-1].startswith("adjacency_")


def _validate_fixed_15d_method(method_name: str, source_model: Any, target_env: Any) -> None:
    if method_name in REGION_DEPENDENT_HEAD_METHODS:
        raise StructuralTransferError(
            f"Method {method_name!r} has a region-dependent N or N+6 action head and cannot be "
            "zero-shot transferred across graph sizes; retrain it for the target size."
        )
    if method_name not in FIXED_15D_TYPED_GRAPH_METHODS:
        raise StructuralTransferError(
            f"Method {method_name!r} is not an approved fixed-15D typed-graph transfer method"
        )
    source_shape = _space_shape(source_model.action_space)
    target_shape = _space_shape(target_env.action_space)
    expected = (FIXED_ACTION_DIMENSION,)
    if source_shape != expected or target_shape != expected:
        raise StructuralTransferError(
            f"Cross-size RLS transfer requires action shape {expected}; "
            f"source={source_shape}, target={target_shape}. No action resizing is permitted."
        )


def transfer_fixed_15d_policy_for_evaluation(
    source_model: Any,
    target_env: Any,
    *,
    method_name: str,
    device: str | th.device = "cpu",
    source_checkpoint_sha256: str | None = None,
) -> tuple[Any, PolicyTransferReport]:
    """Rebuild an RLS policy for another N and copy only safe tensors.

    The returned object is a Stable-Baselines3 policy (and therefore exposes
    ``predict``), not an optimizer-bearing training-resume model.
    """

    _validate_fixed_15d_method(method_name, source_model, target_env)
    source_policy = _as_policy(source_model)
    source_extractors = iter_unique_typed_graph_extractors(source_policy)
    if not source_extractors:
        raise StructuralTransferError("Source policy is not backed by a typed graph extractor")
    source_extractor = source_extractors[0]
    source_num_cells = int(source_extractor.num_cells)
    target_num_cells = int(target_env.n)
    if source_num_cells == target_num_cells:
        raise StructuralTransferError(
            "Cross-size transfer requires different source and target N; use "
            "synchronize_same_size_typed_graph for a same-size topology shift"
        )

    policy_kwargs = copy.deepcopy(getattr(source_model, "policy_kwargs", None))
    if not isinstance(policy_kwargs, dict):
        raise StructuralTransferError("Source model does not expose reconstructible policy_kwargs")
    extractor_class = policy_kwargs.get("features_extractor_class")
    if not isinstance(extractor_class, type) or not issubclass(
        extractor_class, TypedGraphFeatureExtractor
    ):
        raise StructuralTransferError("Source policy_kwargs do not specify a typed graph extractor")
    extractor_kwargs = policy_kwargs.get("features_extractor_kwargs")
    if not isinstance(extractor_kwargs, dict):
        raise StructuralTransferError("Source policy_kwargs lack features_extractor_kwargs")
    if int(extractor_kwargs.get("cell_feature_dim", -1)) != int(target_env.node_feature_dim):
        raise StructuralTransferError("Target cell feature dimension differs from the source policy")
    if int(extractor_kwargs.get("global_feature_dim", -1)) != int(target_env.global_feature_dim):
        raise StructuralTransferError("Target global feature dimension differs from the source policy")

    target_matrices = typed_graph_adjacencies_from_env(target_env)
    source_edge_types = tuple(extractor_kwargs.get("adjacency_matrices", {}).keys())
    if set(source_edge_types) != set(target_matrices):
        raise StructuralTransferError(
            f"Target edge types {sorted(target_matrices)} differ from source {sorted(source_edge_types)}"
        )
    extractor_kwargs["num_cells"] = target_num_cells
    extractor_kwargs["cell_feature_dim"] = int(target_env.node_feature_dim)
    extractor_kwargs["global_feature_dim"] = int(target_env.global_feature_dim)
    extractor_kwargs["adjacency_matrices"] = target_matrices

    policy_class = getattr(source_model, "policy_class", source_policy.__class__)
    target_policy = policy_class(
        target_env.observation_space,
        target_env.action_space,
        lambda _progress: 0.0,
        **policy_kwargs,
    )
    target_policy.to(th.device(device))

    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    if set(source_state) != set(target_state):
        missing_from_source = sorted(set(target_state) - set(source_state))
        extra_in_source = sorted(set(source_state) - set(target_state))
        raise StructuralTransferError(
            "Source/target policy tensor keys differ: "
            f"missing_from_source={missing_from_source}, extra_in_source={extra_in_source}"
        )

    migrated_state: dict[str, th.Tensor] = {}
    copied_keys: list[str] = []
    graph_buffer_keys: list[str] = []
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if _is_graph_buffer_key(key):
            # Keep the target graph constructed from target_env.  Never copy a
            # source-size or source-topology adjacency buffer.
            migrated_state[key] = target_tensor
            graph_buffer_keys.append(key)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise StructuralTransferError(
                f"Non-graph tensor {key!r} changes shape: "
                f"source={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise StructuralTransferError(
                f"Non-graph tensor {key!r} changes dtype: "
                f"source={source_tensor.dtype}, target={target_tensor.dtype}"
            )
        migrated_state[key] = source_tensor.detach().to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        )
        copied_keys.append(key)

    target_policy.load_state_dict(migrated_state, strict=True)
    target_policy.set_training_mode(False)

    loaded_state = target_policy.state_dict()
    for key in copied_keys:
        expected = source_state[key].detach().to(
            device=loaded_state[key].device,
            dtype=loaded_state[key].dtype,
        )
        if not th.equal(loaded_state[key], expected):
            raise StructuralTransferError(f"Transferred tensor verification failed for {key!r}")

    target_extractors = iter_unique_typed_graph_extractors(target_policy)
    if not target_extractors:
        raise StructuralTransferError("Reconstructed policy lost its typed graph extractor")
    graph_hashes = {
        edge_type: _sha256_array(
            getattr(target_extractors[0], f"adjacency_{edge_type}")
        )
        for edge_type in GRAPH_EDGE_TYPES
    }
    expected_hashes = {
        edge_type: _sha256_array(
            target_extractors[0]._normalize_adjacency(target_matrices[edge_type])
        )
        for edge_type in GRAPH_EDGE_TYPES
    }
    if graph_hashes != expected_hashes:
        raise StructuralTransferError("Reconstructed policy does not contain the target graph buffers")

    report = PolicyTransferReport(
        method=str(method_name),
        source_num_cells=source_num_cells,
        target_num_cells=target_num_cells,
        source_observation_shape=_space_shape(source_model.observation_space),
        target_observation_shape=_space_shape(target_env.observation_space),
        action_shape=_space_shape(target_env.action_space),
        copied_tensor_keys=tuple(sorted(copied_keys)),
        target_graph_buffer_keys=tuple(sorted(graph_buffer_keys)),
        target_graph_hashes=graph_hashes,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    return target_policy, report


def load_fixed_15d_policy_for_cross_size_evaluation(
    checkpoint_path: str | Path,
    target_env: Any,
    *,
    method_name: str,
    device: str | th.device = "cpu",
) -> tuple[Any, PolicyTransferReport]:
    """Load a source checkpoint and return an evaluation-only target policy."""

    from experiment.pipeline import load_trained_model

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    source_model = load_trained_model(checkpoint, device=str(device), method_name=method_name)
    return transfer_fixed_15d_policy_for_evaluation(
        source_model,
        target_env,
        method_name=method_name,
        device=device,
        source_checkpoint_sha256=_sha256_file(checkpoint),
    )


def assert_policy_action_matches_environment(action: Any, env: Any) -> np.ndarray:
    """Fail before decoding if a policy emits the wrong action dimension."""

    values = np.asarray(action)
    expected = _space_shape(env.action_space)
    actual = tuple(int(value) for value in values.shape)
    if actual != expected:
        raise StructuralTransferError(
            f"Policy action shape {actual} does not match target environment {expected}; "
            "action resizing is forbidden."
        )
    if not np.all(np.isfinite(values)):
        raise StructuralTransferError("Policy action contains NaN or infinity")
    return values


__all__ = [
    "FIXED_15D_TYPED_GRAPH_METHODS",
    "REGION_DEPENDENT_HEAD_METHODS",
    "StructuralTransferError",
    "GraphSyncReport",
    "PolicyTransferReport",
    "typed_graph_adjacencies_from_env",
    "iter_unique_typed_graph_extractors",
    "synchronize_same_size_typed_graph",
    "transfer_fixed_15d_policy_for_evaluation",
    "load_fixed_15d_policy_for_cross_size_evaluation",
    "assert_policy_action_matches_environment",
]
