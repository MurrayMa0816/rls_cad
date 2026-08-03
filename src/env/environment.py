from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Sequence, Tuple
import math
import heapq

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback used by the smoke test
    try:
        import gym
        from gym import spaces
    except Exception:  # pragma: no cover
        class _Env:
            metadata = {}

        class _Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _Spaces:
            Box = _Box

        class _Gym:
            Env = _Env

        gym = _Gym()
        spaces = _Spaces()


EPS = 1e-8

SIGNAL_MATCHED_SCORE_CALIBRATION = "center_unit_population_std_v1"
SIGNAL_MATCHED_SCORE_SCALE_FLOOR = 1e-6
SIGNAL_MATCHED_SPARSE_TAU_NORMALIZATION = "per_region_quadratic_v1"

NODE_TYPES = (
    "airfield",
    "access",
    "transfer",
    "fuel",
    "power",
    "comm",
    "repair",
    "assembly",
    "buffer",
)
TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPES)}


@dataclass
class MapConfig:
    map_rows: int = 12
    map_cols: int = 15
    horizon_steps: int = 40
    random_seed: int = 0
    resource_budget: float = 1.0
    scenario_id: str = "emergency_wireless_access_backhaul_v1"
    profile: str = "mixed"
    allocation_mode: str = "tmsbd"
    observation_mode: str = "flat_node_features"

    min_effective_share: float = 1e-3
    switch_penalty_weight: float = 0.002
    route_damage_fraction: float = 0.20
    stochastic_disturbance: bool = False
    disturbance_noise: float = 0.01
    # Mean-preserving, cell-wise traffic-arrival uncertainty.  A value of
    # zero reproduces the deterministic base-arrival path.
    arrival_base_rate: float = 0.06
    arrival_coefficient_of_variation: float = 0.0
    initial_readiness_low: float = 0.75
    initial_readiness_high: float = 1.00
    budget_dilution_threshold: float = 0.0
    diffuse_budget_efficiency: float = 1.0
    active_support_penalty_weight: float = 0.0
    probe_budget: float = 0.05
    topk_ratio: float = 0.10
    fixed_topk_ratio: float = 0.10
    max_sparse_support_fraction: float = 0.12
    tmsbd_action_dim: int = 15
    tmsbd_tau_multiplier: float = 1.25
    tmsbd_gamma_base: float = 0.0
    tmsbd_gamma_scale: float = 1.0
    tmsbd_gamma_bias: float = 0.75
    tmsbd_gate_residual_scale: float = 0.25
    tmsbd_gate_softmax_temperature: float = 1.0
    tmsbd_descriptor_mix: float = 0.20
    tmsbd_channel_prior_strength: float = 0.18
    tmsbd_value_gain_floor: float = 0.15
    tmsbd_value_gain_span: float = 1.30
    tmsbd_value_gate_bias: float = 0.0
    tmsbd_value_residual_scale: float = 0.25
    tmsbd_mu_scale: float = 0.20
    tmsbd_mu_bias: float = 2.50
    tmsbd_projection_min_contrast: float = 0.03
    # Non-tunable declarations for the corrected matched-control protocol.
    # Their values are also emitted by every decode for artifact provenance.
    signal_matched_score_calibration: str = SIGNAL_MATCHED_SCORE_CALIBRATION
    signal_matched_score_scale_floor: float = SIGNAL_MATCHED_SCORE_SCALE_FLOOR
    signal_matched_sparse_tau_normalization: str = SIGNAL_MATCHED_SPARSE_TAU_NORMALIZATION
    direct_softmax_temperature: float = 1.0
    direct_sparse_projection_tau: float = 0.18
    roi_temperature: float = 0.35

    # Keep the physical/reference capability coefficient separate from the
    # decoder coefficient so sensitivity experiments do not move the metric
    # ruler and the policy action at the same time.
    lambda_zeta: float = 1.50
    decoder_lambda_zeta: float = 1.50
    reference_lambda_zeta: float = 1.50

    # Optional four-channel overrides.  The objective weights determine the
    # scalar loss/reward, whereas decoder weights condition morphology only.
    objective_weight_override: Tuple[float, float, float, float] | None = None
    decoder_weight_override: Tuple[float, float, float, float] | None = None

    # Structural-shift controls.  Defaults exactly preserve the formal graph.
    topology_seed: int = 0
    spatial_edge_drop_fraction: float = 0.0
    edge_rewire_fraction: float = 0.0
    route_edge_drop_fraction: float = 0.0
    support_edge_drop_fraction: float = 0.0
    role_relocation_fraction: float = 0.0
    support_providers_per_target: int = 3

    @property
    def annual_budget(self) -> float:
        return float(self.resource_budget)

    @annual_budget.setter
    def annual_budget(self, value: float):
        self.resource_budget = float(value)


@dataclass
class TheaterHexEnvConfig:
    map_config: MapConfig = field(default_factory=MapConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"map_config": asdict(self.map_config)}


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def softplus_scalar(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return float(np.log1p(np.exp(x)))


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), EPS)
    z = np.asarray(logits, dtype=np.float64).reshape(-1) / temp
    z = np.clip(z - np.max(z), -60.0, 60.0)
    exp_z = np.exp(z)
    return (exp_z / np.maximum(exp_z.sum(), EPS)).astype(np.float32)


def sparsemax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size == 1:
        return np.ones(1, dtype=np.float32)
    z_sorted = np.sort(z)[::-1]
    cssv = np.cumsum(z_sorted)
    k = np.arange(1, z.size + 1)
    support = z_sorted > (cssv - 1.0) / k
    if not np.any(support):
        return np.full_like(z, 1.0 / z.size, dtype=np.float32)
    rho = int(k[support][-1])
    tau = (cssv[rho - 1] - 1.0) / rho
    p = np.maximum(z - tau, 0.0)
    return (p / np.maximum(p.sum(), EPS)).astype(np.float32)


def bounded_simplex_np(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Project a nonnegative 4-channel gate onto a simplex with floor and cap."""
    p = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if p.size == 0:
        return p.astype(np.float32)
    lower = float(np.clip(lower, 0.0, 1.0 / p.size))
    upper = float(np.clip(upper, 1.0 / p.size, 1.0))
    if float(p.sum()) <= EPS:
        p = np.full(p.size, 1.0 / p.size, dtype=np.float64)
    else:
        p = p / float(p.sum())

    out = np.full(p.size, lower, dtype=np.float64)
    remaining = 1.0 - lower * p.size
    free = np.ones(p.size, dtype=bool)
    cap = upper - lower
    while np.any(free) and remaining > EPS:
        free_idx = np.where(free)[0]
        score = p[free_idx]
        if float(score.sum()) <= EPS:
            alloc = np.full(free_idx.size, remaining / free_idx.size, dtype=np.float64)
        else:
            alloc = remaining * score / float(score.sum())
        over = alloc > cap + 1e-10
        if not np.any(over):
            out[free_idx] += alloc
            remaining = 0.0
            break
        over_idx = free_idx[over]
        out[over_idx] = upper
        remaining -= cap * over_idx.size
        free[over_idx] = False
        p[over_idx] = 0.0
    return (out / np.maximum(out.sum(), EPS)).astype(np.float32)


def sparse_budget_projection(values: np.ndarray, tau: float, max_support: int | None = None) -> Tuple[np.ndarray, float, int]:
    """Solve argmax_w q'w - tau/2 ||w||^2 over the simplex."""
    q = np.asarray(values, dtype=np.float64).reshape(-1)
    tau = max(float(tau), 1e-4)
    order = np.argsort(q)[::-1]
    q_sorted = q[order]
    cssv = np.cumsum(q_sorted)
    k = np.arange(1, q.size + 1)
    thresholds = (cssv - tau) / k
    support = q_sorted > thresholds
    if not np.any(support):
        w = np.full(q.size, 1.0 / q.size, dtype=np.float32)
        return w, float(np.mean(q) - tau / q.size), int(q.size)
    rho = int(k[support][-1])
    if max_support is not None:
        rho = min(rho, max(1, int(max_support)))
        idx = order[:rho]
        lam = float((np.sum(q[idx]) - tau) / rho)
        w = np.zeros(q.size, dtype=np.float64)
        w[idx] = np.maximum((q[idx] - lam) / tau, 0.0)
        if float(w.sum()) <= EPS:
            w[idx] = 1.0 / rho
        else:
            w = w / float(w.sum())
        return w.astype(np.float32), lam, int(np.count_nonzero(w > EPS))
    lam = float((cssv[rho - 1] - tau) / rho)
    w = np.maximum((q - lam) / tau, 0.0)
    w = w / np.maximum(w.sum(), EPS)
    return w.astype(np.float32), lam, rho


def budget_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float(-np.sum(w * np.log(w + EPS)))


def calibrate_signal_matched_score(
    values: np.ndarray,
    scale_floor: float = SIGNAL_MATCHED_SCORE_SCALE_FLOOR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Put the shared matched-control score on a dimensionless scale.

    Both signal-matched allocation controls call this function on the same
    learned regional score vector. Centering makes the protocol shift
    invariant, and population-standard-deviation scaling keeps the output
    operators on a fixed numerical scale. The constant-score fallback is
    deterministic and finite.
    """

    raw_score = np.asarray(values, dtype=np.float64).reshape(-1)
    if raw_score.size == 0:
        raise ValueError("Signal-matched score calibration requires at least one region.")
    if not np.all(np.isfinite(raw_score)):
        raise ValueError("Signal-matched score calibration requires finite scores.")

    floor = max(float(scale_floor), np.finfo(np.float64).eps)
    center = float(np.mean(raw_score))
    centered = raw_score - center
    raw_scale = float(np.sqrt(np.mean(np.square(centered))))
    degenerate = bool(raw_scale < floor)
    scale_used = floor if degenerate else raw_scale
    calibrated = np.zeros_like(centered) if degenerate else centered / scale_used
    metadata: Dict[str, Any] = {
        "score_calibration": SIGNAL_MATCHED_SCORE_CALIBRATION,
        "score_center": center,
        "score_scale": raw_scale,
        "score_scale_used": float(scale_used),
        "score_scale_floor": floor,
        "score_calibration_degenerate": degenerate,
    }
    return calibrated.astype(np.float32), metadata


class JointMobilitySupportEnv(gym.Env):
    """Regional joint mobility-support hub recovery environment.

    The environment follows the paper specification: typed regional graph,
    fixed per-step budget, sparse executable allocation, four loss channels,
    and task-morphology sparse budget decoding.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: TheaterHexEnvConfig | MapConfig | None = None):
        if config is None:
            map_config = MapConfig()
        elif isinstance(config, TheaterHexEnvConfig):
            map_config = config.map_config
        elif isinstance(config, MapConfig):
            map_config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)!r}")

        self.config = map_config
        self.rows = int(map_config.map_rows)
        self.cols = int(map_config.map_cols)
        self.n = self.rows * self.cols
        self.horizon = int(map_config.horizon_steps)
        self.budget = float(map_config.resource_budget)
        self._reset_seed_rng = np.random.default_rng(int(map_config.random_seed))
        self.rng = np.random.default_rng(int(map_config.random_seed))
        self.arrival_rng = np.random.default_rng(int(map_config.random_seed) + 1)
        self.disturbance_rng = np.random.default_rng(int(map_config.random_seed) + 2)
        self.topology_rng = np.random.default_rng(int(map_config.topology_seed))

        self._build_static_graph()
        self.base_node_feature_dim = len(NODE_TYPES) + 2 + 6 + 6 + 1
        self.signal_feature_dim = 6 if map_config.allocation_mode in (
            "signal_matched_softmax",
            "signal_matched_sparse_projection",
        ) else 0
        self.node_feature_dim = self.base_node_feature_dim + self.signal_feature_dim
        self.global_feature_dim = 10
        obs_dim = self.n * self.node_feature_dim + self.global_feature_dim
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
        action_dim = self.action_dim_for_mode(map_config.allocation_mode, self.n)
        if map_config.allocation_mode in (
            "tmsbd",
            "tmsbd_no_gate",
            "tmsbd_no_chain",
            "tmsbd_softmax_gate",
            "tmsbd_softmax_budget",
            "tmsbd_fixed_topk",
            "tmsbd_fixed_morphology",
            "tmsbd_single_critical",
            "tmsbd_single_support",
            "tmsbd_single_backlog",
            "tmsbd_single_e2e",
            "lts_scp_latent",
        ):
            configured_action_dim = int(map_config.tmsbd_action_dim)
            if configured_action_dim != 15:
                raise ValueError(
                    "The structured decoder has an exact, reproducible 15-dimensional "
                    f"actor mapping; received tmsbd_action_dim={configured_action_dim}."
                )
            action_dim = 15
        self.action_space = spaces.Box(
            low=-8.0,
            high=8.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        self.t = 0
        self.current_profile = str(map_config.profile)
        self._last_reset_seed = int(map_config.random_seed)
        self.loss_weights = np.full(4, 0.25, dtype=np.float32)
        self.decoder_loss_weights = self.loss_weights.copy()
        self.last_budget = np.zeros(self.n, dtype=np.float32)
        self.delta = np.zeros(self.n, dtype=np.float32)
        self.kappa = np.ones(self.n, dtype=np.float32)
        self.sigma = np.ones(self.n, dtype=np.float32)
        self.backlog = np.zeros(self.n, dtype=np.float32)
        self.readiness = np.ones(self.n, dtype=np.float32)
        self.demand = np.ones(self.n, dtype=np.float32)
        self.last_arrival = np.zeros(self.n, dtype=np.float32)
        self.last_arrival_multiplier = np.ones(self.n, dtype=np.float32)
        self.route_edge_health = np.ones((self.n, self.n), dtype=np.float32)
        self.capability = 1.0
        self.losses = self._compute_losses()
        self.initial_loss_vector = self._loss_vector()
        self.previous_loss_vector = self.initial_loss_vector.copy()
        self.last_decode: Dict[str, Any] = {}

    @staticmethod
    def action_dim_for_mode(mode: str, n: int | None = None) -> int:
        if mode in (
            "tmsbd",
            "tmsbd_no_gate",
            "tmsbd_no_chain",
            "tmsbd_softmax_gate",
            "tmsbd_softmax_budget",
            "tmsbd_fixed_topk",
            "tmsbd_fixed_morphology",
            "tmsbd_single_critical",
            "tmsbd_single_support",
            "tmsbd_single_backlog",
            "tmsbd_single_e2e",
            "lts_scp_latent",
        ):
            return 15
        if mode == "roi_param":
            return 2
        if mode in ("coverage_focus_dual", "fixed_dual"):
            return 3
        if mode in (
            "direct_softmax",
            "full_action",
            "direct_sparse_projection",
            "direct_simplex",
        ):
            if n is None:
                return 1
            return int(n)
        if mode in ("signal_matched_softmax", "signal_matched_sparse_projection"):
            if n is None:
                return 1
            # This control receives the matched analytical features
            # in its observation and directly outputs one score per region.
            return int(n)
        return 7

    @staticmethod
    def _fraction(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _drop_undirected_edges(self, matrix: np.ndarray, fraction: float) -> np.ndarray:
        """Drop a deterministic fraction of undirected edges symmetrically."""
        out = np.asarray(matrix, dtype=np.float32).copy()
        frac = self._fraction(fraction)
        edges = np.argwhere(np.triu(out, k=1) > 0)
        count = min(edges.shape[0], int(round(frac * edges.shape[0])))
        if count <= 0:
            return out
        chosen = self.topology_rng.choice(edges.shape[0], size=count, replace=False)
        for edge_idx in np.asarray(chosen).reshape(-1):
            src, dst = edges[int(edge_idx)]
            out[src, dst] = 0.0
            out[dst, src] = 0.0
        return out

    def _rewire_undirected_edges(self, matrix: np.ndarray, fraction: float) -> np.ndarray:
        """Perform degree-preserving double-edge swaps on a symmetric graph."""
        out = np.asarray(matrix, dtype=np.float32).copy()
        frac = self._fraction(fraction)
        initial_edges = np.argwhere(np.triu(out, k=1) > 0)
        requested = int(round(frac * initial_edges.shape[0] / 2.0))
        completed = 0
        attempts = 0
        max_attempts = max(100, requested * 50)
        while completed < requested and attempts < max_attempts:
            attempts += 1
            edges = np.argwhere(np.triu(out, k=1) > 0)
            if edges.shape[0] < 2:
                break
            selected = self.topology_rng.choice(edges.shape[0], size=2, replace=False)
            a, b = (int(v) for v in edges[int(selected[0])])
            c, d = (int(v) for v in edges[int(selected[1])])
            if len({a, b, c, d}) < 4:
                continue
            if bool(self.topology_rng.integers(0, 2)):
                c, d = d, c
            if a == d or c == b or out[a, d] > 0 or out[c, b] > 0:
                continue
            out[a, b] = out[b, a] = 0.0
            out[c, d] = out[d, c] = 0.0
            out[a, d] = out[d, a] = 1.0
            out[c, b] = out[b, c] = 1.0
            completed += 1
        return out

    def _drop_directed_edges(self, matrix: np.ndarray, fraction: float) -> np.ndarray:
        out = np.asarray(matrix, dtype=np.float32).copy()
        frac = self._fraction(fraction)
        edges = np.argwhere(out > 0)
        count = min(edges.shape[0], int(round(frac * edges.shape[0])))
        if count <= 0:
            return out
        chosen = self.topology_rng.choice(edges.shape[0], size=count, replace=False)
        selected_edges = edges[np.asarray(chosen).reshape(-1)]
        out[selected_edges[:, 0], selected_edges[:, 1]] = 0.0
        return out

    def _build_static_graph(self):
        n = self.n
        rr, cc = np.divmod(np.arange(n), self.cols)
        # Preserve the historical planning benchmark exactly: the rendered
        # cells are hexagons, while the computational spatial graph is a
        # four-neighbor row/column lattice.  Changing this geometry would also
        # change roles, route edges, nearest support providers, value/cost
        # fields, and graph messages, confounding the reset correction.
        norm_r = rr.astype(np.float64) / max(self.rows - 1, 1)
        norm_c = cc.astype(np.float64) / max(self.cols - 1, 1)
        self.pos = np.stack([norm_r, norm_c], axis=1).astype(np.float32)
        center_r = (self.rows - 1) / 2.0
        center_c = (self.cols - 1) / 2.0
        dist_center = np.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
        max_dist = np.maximum(dist_center.max(), EPS)

        type_id = np.full(n, TYPE_TO_ID["buffer"], dtype=np.int64)
        core = (np.abs(rr - center_r) <= max(1.0, self.rows * 0.08)) & (
            np.abs(cc - center_c) <= max(1.0, self.cols * 0.10)
        )
        type_id[core] = TYPE_TO_ID["airfield"]

        horizontal_route = np.abs(rr - center_r) <= 1
        vertical_route = np.abs(cc - center_c) <= 1
        route_mask = horizontal_route | vertical_route
        type_id[route_mask & ~core] = TYPE_TO_ID["access"]

        transfer_mask = ((cc < self.cols * 0.18) | (cc > self.cols * 0.82)) & horizontal_route
        type_id[transfer_mask] = TYPE_TO_ID["transfer"]

        assembly_mask = ((rr > self.rows * 0.72) & (cc > self.cols * 0.65)) | (
            (rr < self.rows * 0.25) & (cc < self.cols * 0.22)
        )
        type_id[assembly_mask & ~route_mask] = TYPE_TO_ID["assembly"]

        support_candidates = np.where((dist_center <= max_dist * 0.55) & (~core) & (~route_mask))[0]
        support_types = ["fuel", "power", "comm", "repair"]
        for idx, support_type in enumerate(support_types):
            if support_candidates.size == 0:
                break
            stride = max(1, support_candidates.size // (len(support_types) + 1))
            selected = support_candidates[idx * stride :: max(stride * 3, 1)][: max(2, n // 80)]
            type_id[selected] = TYPE_TO_ID[support_type]

        relocation_fraction = self._fraction(self.config.role_relocation_fraction)
        relocation_count = min(n, int(round(relocation_fraction * n)))
        if relocation_count >= 2:
            relocation_idx = self.topology_rng.choice(n, size=relocation_count, replace=False)
            relocated_types = type_id[relocation_idx].copy()
            self.topology_rng.shuffle(relocated_types)
            type_id[relocation_idx] = relocated_types

        self.type_id = type_id
        self.type_onehot = np.eye(len(NODE_TYPES), dtype=np.float32)[type_id]

        self.a_adj = np.zeros((n, n), dtype=np.float32)
        self.a_route = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            r, c = divmod(i, self.cols)
            neighbors = (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            )
            for nr, nc in neighbors:
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    j = nr * self.cols + nc
                    self.a_adj[i, j] = 1.0
                    if type_id[i] in (
                        TYPE_TO_ID["airfield"],
                        TYPE_TO_ID["access"],
                        TYPE_TO_ID["transfer"],
                        TYPE_TO_ID["assembly"],
                    ) and type_id[j] in (
                        TYPE_TO_ID["airfield"],
                        TYPE_TO_ID["access"],
                        TYPE_TO_ID["transfer"],
                        TYPE_TO_ID["assembly"],
                    ):
                        self.a_route[i, j] = 1.0

        self.a_adj = self._drop_undirected_edges(
            self.a_adj,
            self.config.spatial_edge_drop_fraction,
        )
        self.a_adj = self._rewire_undirected_edges(
            self.a_adj,
            self.config.edge_rewire_fraction,
        )
        # A functional route is always also a current spatial edge.
        self.a_route = self.a_route * self.a_adj
        self.a_route = self._drop_undirected_edges(
            self.a_route,
            self.config.route_edge_drop_fraction,
        )

        self.a_dep = np.zeros((n, n), dtype=np.float32)
        dependent = np.where(
            np.isin(type_id, [TYPE_TO_ID["airfield"], TYPE_TO_ID["assembly"], TYPE_TO_ID["transfer"]])
        )[0]
        for support_type in ("fuel", "power", "comm", "repair", "access"):
            sources = np.where(type_id == TYPE_TO_ID[support_type])[0]
            if sources.size == 0:
                continue
            for target in dependent:
                d = np.linalg.norm(self.pos[sources] - self.pos[target], axis=1)
                providers = max(1, int(self.config.support_providers_per_target))
                nearest = sources[np.argsort(d)[: min(providers, sources.size)]]
                self.a_dep[nearest, target] = 1.0
        self.a_dep = self._drop_directed_edges(
            self.a_dep,
            self.config.support_edge_drop_fraction,
        )

        route_degree = self.a_route.sum(axis=1)
        self.route_break_base = np.clip(route_degree / np.maximum(route_degree.max(), 1.0), 0.0, 1.0)
        self.betweenness = self._approx_betweenness()
        self.entry_nodes = np.where((cc == 0) | (cc == self.cols - 1) | (rr == 0) | (rr == self.rows - 1))[0]

        type_value = np.array([1.0, 0.62, 0.75, 0.58, 0.60, 0.57, 0.52, 0.70, 0.22], dtype=np.float32)
        type_criticality = np.array([1.0, 0.58, 0.72, 0.70, 0.72, 0.68, 0.64, 0.62, 0.12], dtype=np.float32)
        self.value = type_value[type_id] * (0.75 + 0.25 * (1.0 - dist_center / max_dist))
        self.criticality = type_criticality[type_id]
        self.repair_efficiency = np.clip(0.65 + 0.25 * self.type_onehot[:, TYPE_TO_ID["repair"]] + 0.10 * self.route_break_base, 0.2, 1.0)
        self.dep_strength = np.clip(self.a_dep.sum(axis=0) / 4.0, 0.0, 1.0)
        self.backlog_cap = np.clip(0.5 + 0.7 * self.value + 0.3 * self.criticality, 0.4, 1.6).astype(np.float32)
        self.repair_scale = np.clip(0.12 + 0.30 * (1.0 - self.repair_efficiency), 0.08, 0.5).astype(np.float32)
        self.exec_cost = np.clip(0.15 + 0.40 * dist_center / max_dist + 0.25 * (1.0 - self.route_break_base), 0.05, 1.0).astype(np.float32)
        self.static_features = np.concatenate(
            [
                self.type_onehot,
                self.pos,
                self.value[:, None],
                self.criticality[:, None],
                self.repair_efficiency[:, None],
                self.dep_strength[:, None],
                self.backlog_cap[:, None],
                self.repair_scale[:, None],
            ],
            axis=1,
        ).astype(np.float32)

    def _approx_betweenness(self) -> np.ndarray:
        r, c = np.divmod(np.arange(self.n), self.cols)
        center_r = (self.rows - 1) / 2.0
        center_c = (self.cols - 1) / 2.0
        corridor = np.exp(-np.abs(r - center_r) / max(self.rows * 0.15, 1.0))
        cross = np.exp(-np.abs(c - center_c) / max(self.cols * 0.15, 1.0))
        score = np.maximum(corridor, cross) * (0.4 + 0.6 * (self.a_route.sum(axis=1) > 0))
        return (score / np.maximum(score.max(), EPS)).astype(np.float32)

    @staticmethod
    def _normalized_channel_weights(
        values: Sequence[float] | np.ndarray,
        *,
        label: str,
    ) -> np.ndarray:
        weights = np.asarray(values, dtype=np.float32).reshape(-1)
        if weights.size != 4 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(f"{label} must contain four finite nonnegative values, got {values!r}")
        total = float(weights.sum())
        if total <= EPS:
            raise ValueError(f"{label} must have a positive sum")
        return (weights / total).astype(np.float32)

    def _seed_episode_rngs(self, episode_seed: int) -> None:
        seed_sequence = np.random.SeedSequence(int(episode_seed))
        scenario_seed, arrival_seed, disturbance_seed = seed_sequence.spawn(3)
        self.rng = np.random.default_rng(scenario_seed)
        self.arrival_rng = np.random.default_rng(arrival_seed)
        self.disturbance_rng = np.random.default_rng(disturbance_seed)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover - compatibility fallback
            pass
        if seed is not None:
            episode_seed = int(seed)
            self._reset_seed_rng = np.random.default_rng(episode_seed)
        else:
            episode_seed = int(self._reset_seed_rng.integers(0, np.iinfo(np.int32).max))
        self._seed_episode_rngs(episode_seed)
        self._last_reset_seed = episode_seed
        self.t = 0
        self.last_budget = np.zeros(self.n, dtype=np.float32)
        self.delta = np.zeros(self.n, dtype=np.float32)
        self.backlog = np.zeros(self.n, dtype=np.float32)
        # These derived support states must not retain the preceding episode.
        self.sigma = np.ones(self.n, dtype=np.float32)
        self.kappa = np.ones(self.n, dtype=np.float32)
        self.last_arrival = np.zeros(self.n, dtype=np.float32)
        self.last_arrival_multiplier = np.ones(self.n, dtype=np.float32)
        readiness_low = float(np.clip(self.config.initial_readiness_low, 0.0, 1.0))
        readiness_high = float(np.clip(self.config.initial_readiness_high, readiness_low, 1.0))
        self.readiness = np.clip(
            readiness_low + (readiness_high - readiness_low) * self.rng.random(self.n),
            0.0,
            1.0,
        ).astype(np.float32)
        self.demand = np.clip(0.45 + 0.55 * self.value + 0.15 * self.rng.normal(size=self.n), 0.0, 1.0).astype(np.float32)
        self.route_edge_health = np.ones((self.n, self.n), dtype=np.float32)
        self._apply_profile(options or {})
        self._update_support_and_connectivity()
        self.capability = self._compute_capability()
        self.losses = self._compute_losses()
        self.initial_loss_vector = self._loss_vector()
        self.previous_loss_vector = self.initial_loss_vector.copy()
        self.last_decode = {}
        return self._get_obs(), self._info()

    def _apply_profile(self, options: Dict[str, Any]):
        profile = str(options.get("profile", self.config.profile))
        if profile == "mixed_balanced":
            profiles = ["key_damage", "route_fracture", "backlog", "distributed", "compound"]
            profile = profiles[int(getattr(self, "_last_reset_seed", 0)) % len(profiles)]
        elif profile == "mixed_complex_balanced":
            profiles = ["key_damage_hard", "route_fracture_hard", "backlog_hard", "distributed_hard", "compound_hard"]
            profile = profiles[int(getattr(self, "_last_reset_seed", 0)) % len(profiles)]
        elif profile == "mixed":
            profiles = ["key_damage", "route_fracture", "distributed", "backlog", "compound"]
            probs = np.array([0.20, 0.22, 0.16, 0.20, 0.22], dtype=np.float64)
            profile = str(self.rng.choice(profiles, p=probs / probs.sum()))
        elif profile == "mixed_complex":
            profiles = ["key_damage_hard", "route_fracture_hard", "distributed_hard", "backlog_hard", "compound_hard"]
            probs = np.array([0.18, 0.24, 0.16, 0.18, 0.24], dtype=np.float64)
            profile = str(self.rng.choice(profiles, p=probs / probs.sum()))
        self.current_profile = profile
        hard_profile = profile.endswith("_hard")
        base_profile = profile[:-5] if hard_profile else profile
        objective_weights = {
            "key_damage": (0.48, 0.12, 0.08, 0.32),
            "route_fracture": (0.08, 0.50, 0.08, 0.34),
            "distributed": (0.14, 0.14, 0.20, 0.52),
            "backlog": (0.08, 0.08, 0.50, 0.34),
            "compound": (0.14, 0.18, 0.12, 0.56),
        }
        profile_weights = objective_weights.get(base_profile, (0.25, 0.25, 0.25, 0.25))
        objective_override = options.get("objective_weight_override", self.config.objective_weight_override)
        decoder_override = options.get("decoder_weight_override", self.config.decoder_weight_override)
        self.loss_weights = self._normalized_channel_weights(
            profile_weights if objective_override is None else objective_override,
            label="objective_weight_override",
        )
        self.decoder_loss_weights = self._normalized_channel_weights(
            self.loss_weights if decoder_override is None else decoder_override,
            label="decoder_weight_override",
        )

        key_nodes = np.where(self.criticality >= 0.70)[0]
        route_nodes = np.where(self.a_route.sum(axis=1) > 0)[0]
        repair_nodes = np.where(
            np.isin(self.type_id, [TYPE_TO_ID["repair"], TYPE_TO_ID["airfield"], TYPE_TO_ID["assembly"]])
        )[0]

        def choose(nodes: np.ndarray, count: int) -> np.ndarray:
            if nodes.size == 0:
                return nodes
            count = min(max(1, count), nodes.size)
            return self.rng.choice(nodes, size=count, replace=False)

        key_count = max(10, self.n // 60)
        route_count = max(12, self.n // 45)
        distributed_fraction = 0.30
        backlog_count = max(14, self.n // 40)
        compound_count = max(12, self.n // 35)
        route_multiplier = 1.35 if base_profile == "route_fracture" else 1.10
        route_health_low, route_health_high = 0.10, 0.50
        if hard_profile:
            key_count = max(16, self.n // 38)
            route_count = max(18, self.n // 32)
            distributed_fraction = 0.46
            backlog_count = max(20, self.n // 28)
            compound_count = max(20, self.n // 26)
            route_multiplier = 1.85 if base_profile == "route_fracture" else 1.55
            route_health_low, route_health_high = 0.03, 0.34

        if base_profile in ("key_damage", "compound"):
            selected = choose(key_nodes, key_count)
            low = 0.72 if hard_profile else 0.65
            self.delta[selected] = self.rng.uniform(low, 1.0, size=selected.size)
        if base_profile in ("route_fracture", "compound"):
            selected = choose(route_nodes, route_count)
            low, high = (0.58, 0.96) if hard_profile else (0.45, 0.85)
            self.delta[selected] = np.maximum(self.delta[selected], self.rng.uniform(low, high, size=selected.size))
            route_edges = np.argwhere(self.a_route > 0)
            if route_edges.size > 0:
                m = max(1, int(route_edges.shape[0] * self.config.route_damage_fraction * route_multiplier))
                chosen_edges = route_edges[self.rng.choice(route_edges.shape[0], size=m, replace=False)]
                self.route_edge_health[chosen_edges[:, 0], chosen_edges[:, 1]] = self.rng.uniform(
                    route_health_low, route_health_high, size=m
                )
        if base_profile in ("distributed", "compound"):
            count = max(5, int(self.n * distributed_fraction))
            selected = choose(np.arange(self.n), count)
            low, high = (0.30, 0.70) if hard_profile else (0.20, 0.50)
            self.delta[selected] = np.maximum(self.delta[selected], self.rng.uniform(low, high, size=selected.size))
        if base_profile in ("backlog", "compound"):
            selected = choose(repair_nodes, backlog_count)
            low = 0.76 if hard_profile else 0.65
            self.backlog[selected] = self.backlog_cap[selected] * self.rng.uniform(low, 1.0, size=selected.size)
            high = 0.26 if hard_profile else 0.35
            self.readiness[selected] = self.rng.uniform(0.02, high, size=selected.size)
        if base_profile == "compound":
            support_sources = np.where(self.a_dep.sum(axis=1) > 0)[0]
            selected = choose(support_sources, compound_count)
            low, high = (0.55, 0.92) if hard_profile else (0.40, 0.78)
            self.delta[selected] = np.maximum(self.delta[selected], self.rng.uniform(low, high, size=selected.size))
            blog_low, blog_high = (0.48, 0.92) if hard_profile else (0.35, 0.78)
            self.backlog[selected] = np.maximum(
                self.backlog[selected],
                self.backlog_cap[selected] * self.rng.uniform(blog_low, blog_high, size=selected.size),
            )
            ready_high = 0.34 if hard_profile else 0.45
            self.readiness[selected] = np.minimum(
                self.readiness[selected], self.rng.uniform(0.05, ready_high, size=selected.size)
            )
        if hard_profile:
            surge_nodes = choose(np.arange(self.n), max(8, int(self.n * 0.18)))
            self.demand[surge_nodes] = np.clip(self.demand[surge_nodes] + self.rng.uniform(0.10, 0.30, surge_nodes.size), 0.0, 1.0)

        self.delta = np.clip(self.delta, 0.0, 1.0).astype(np.float32)
        self.backlog = np.clip(self.backlog, 0.0, self.backlog_cap).astype(np.float32)

    def step(self, action: np.ndarray):
        prev_loss = float(self.total_loss)
        budget, decode = self.decode_action(action)
        return self._advance_with_budget(budget, decode, prev_loss)

    def step_budget(self, budget: np.ndarray, decode: Dict[str, Any] | None = None):
        prev_loss = float(self.total_loss)
        budget = np.asarray(budget, dtype=np.float32).reshape(-1)
        if budget.size != self.n:
            raise ValueError(f"Budget action has {budget.size} entries; expected {self.n}")
        budget = np.maximum(budget, 0.0)
        if float(budget.sum()) <= EPS:
            budget = np.full(self.n, self.budget / self.n, dtype=np.float32)
        else:
            budget = (self.budget * budget / float(budget.sum())).astype(np.float32)
        return self._advance_with_budget(budget, decode or {"mode": "heuristic"}, prev_loss)

    def _advance_with_budget(self, budget: np.ndarray, decode: Dict[str, Any], prev_loss: float):
        prev_budget = self.last_budget.copy()
        is_initial_placement = self.t == 0
        previous_loss_vector = self._loss_vector()
        self._transition(budget)
        self.t += 1
        self._update_support_and_connectivity()
        self.capability = self._compute_capability()
        self.losses = self._compute_losses()
        self.previous_loss_vector = previous_loss_vector
        switch_cost = float(np.sum(np.abs(budget - prev_budget)) / max(self.budget, EPS))
        migration_fraction = 0.0 if is_initial_placement else 0.5 * switch_cost
        initial_placement_cost = switch_cost if is_initial_placement else 0.0
        effective_threshold = float(self.config.min_effective_share) * self.budget
        active_ratio = float(np.mean(budget >= effective_threshold))
        reward = (
            prev_loss
            - float(self.total_loss)
            - float(self.config.switch_penalty_weight) * switch_cost
            - float(self.config.active_support_penalty_weight) * active_ratio
        )
        self.last_budget = budget.astype(np.float32)
        self.last_decode = decode
        terminated = False
        truncated = self.t >= self.horizon
        info = self._info()
        info["reward_raw_improvement"] = prev_loss - float(self.total_loss)
        info["switch_cost"] = switch_cost
        info["migration_fraction"] = float(migration_fraction)
        info["initial_placement_cost"] = float(initial_placement_cost)
        info["is_initial_placement"] = bool(is_initial_placement)
        info["active_support_penalty"] = float(self.config.active_support_penalty_weight) * active_ratio
        return self._get_obs(), float(reward), terminated, truncated, info

    def _transition(self, budget: np.ndarray):
        effective_budget = self._effective_intervention_budget(budget)
        repair = np.minimum(effective_budget / (self.repair_scale + EPS), 1.0)
        repair = repair * self.repair_efficiency * (0.5 + 0.5 * self.readiness)
        key_node = np.isin(
            self.type_id,
            [TYPE_TO_ID["airfield"], TYPE_TO_ID["assembly"], TYPE_TO_ID["transfer"]],
        ).astype(np.float32)
        route_node = (
            np.isin(self.type_id, [TYPE_TO_ID["airfield"], TYPE_TO_ID["access"], TYPE_TO_ID["transfer"]]).astype(
                np.float32
            )
            * (0.45 + 0.55 * self.route_break_base)
        )
        support_node = np.isin(
            self.type_id,
            [TYPE_TO_ID["fuel"], TYPE_TO_ID["power"], TYPE_TO_ID["comm"], TYPE_TO_ID["repair"]],
        ).astype(np.float32)
        source_node = (self.a_dep.sum(axis=1) > 0).astype(np.float32)
        damage_gain = 0.35 + 0.45 * key_node + 0.20 * self.criticality
        backlog_gain = 0.24 + 0.52 * support_node + 0.24 * self.type_onehot[:, TYPE_TO_ID["repair"]]
        support_gain = 0.20 + 0.45 * support_node + 0.25 * source_node + 0.10 * self.dep_strength
        route_gain = 0.18 + 0.58 * route_node + 0.24 * self.betweenness
        if self.config.stochastic_disturbance:
            noise_delta = self.disturbance_rng.normal(0.0, self.config.disturbance_noise, size=self.n)
            noise_backlog = self.disturbance_rng.normal(0.0, self.config.disturbance_noise, size=self.n)
        else:
            noise_delta = 0.0
            noise_backlog = 0.0
        arrival_cv = max(float(self.config.arrival_coefficient_of_variation), 0.0)
        if arrival_cv > 0.0:
            log_sigma = math.sqrt(math.log1p(arrival_cv * arrival_cv))
            arrival_multiplier = np.exp(
                log_sigma * self.arrival_rng.normal(size=self.n) - 0.5 * log_sigma * log_sigma
            )
        else:
            arrival_multiplier = np.ones(self.n, dtype=np.float64)
        self.delta = np.clip(self.delta + noise_delta - 0.44 * damage_gain * repair, 0.0, 1.0).astype(np.float32)
        arrival = float(self.config.arrival_base_rate) * self.delta * arrival_multiplier
        self.last_arrival_multiplier = np.asarray(arrival_multiplier, dtype=np.float32)
        self.last_arrival = np.asarray(arrival, dtype=np.float32)
        self.backlog = np.clip(
            self.backlog + self.last_arrival + noise_backlog - 0.32 * backlog_gain * repair * self.readiness,
            0.0,
            self.backlog_cap,
        ).astype(np.float32)
        self.readiness = np.clip(
            self.readiness + 0.16 * support_gain * effective_budget + 0.08 * support_gain * repair
            - 0.04 * self.backlog / (self.backlog_cap + EPS),
            0.0,
            1.0,
        ).astype(np.float32)
        route_repair = route_gain * effective_budget
        self.route_edge_health = np.clip(
            self.route_edge_health + 0.10 * (route_repair[:, None] + route_repair[None, :]),
            0.0,
            1.0,
        )

    def _effective_intervention_budget(self, budget: np.ndarray) -> np.ndarray:
        threshold = float(self.config.budget_dilution_threshold)
        if threshold <= 0.0:
            return np.asarray(budget, dtype=np.float32)
        min_efficiency = float(np.clip(self.config.diffuse_budget_efficiency, 0.0, 1.0))
        scale = max(0.25 * threshold, 1e-6)
        budget_arr = np.asarray(budget, dtype=np.float32)
        gate = sigmoid((budget_arr - threshold) / scale)
        efficiency = min_efficiency + (1.0 - min_efficiency) * gate
        return (budget_arr * efficiency).astype(np.float32)

    def _update_support_and_connectivity(self):
        o = 1.0 - self.delta
        backlog_factor = 1.0 - self.backlog / (self.backlog_cap + EPS)
        source_service = o * np.maximum(self.sigma, 0.0) * np.maximum(self.readiness, 0.0) * backlog_factor
        support_num = self.a_dep.T @ source_service
        support_den = self.a_dep.sum(axis=0) + EPS
        support_bar = support_num / support_den
        no_dep = self.a_dep.sum(axis=0) <= 0
        support_bar[no_dep] = 1.0
        self.sigma = (o * ((1.0 - self.dep_strength) + self.dep_strength * support_bar)).astype(np.float32)
        route = self.a_route * self.route_edge_health * np.minimum.outer(o * self.sigma, o * self.sigma)
        self.kappa = self._widest_path(route).astype(np.float32)

    def _widest_path(self, route_weight: np.ndarray) -> np.ndarray:
        best = np.zeros(self.n, dtype=np.float64)
        heap: list[Tuple[float, int]] = []
        for node in self.entry_nodes:
            best[node] = 1.0
            heapq.heappush(heap, (-1.0, int(node)))
        while heap:
            neg_cap, node = heapq.heappop(heap)
            cap = -neg_cap
            if cap < best[node] - 1e-12:
                continue
            neigh = np.where(route_weight[node] > 0)[0]
            for nb in neigh:
                new_cap = min(cap, float(route_weight[node, nb]))
                if new_cap > best[nb] + 1e-12:
                    best[nb] = new_cap
                    heapq.heappush(heap, (-new_cap, int(nb)))
        return np.clip(best, 0.0, 1.0)

    def _compute_capability(self, lambda_zeta: float | None = None) -> float:
        dependency_coefficient = (
            float(self.config.lambda_zeta) if lambda_zeta is None else float(lambda_zeta)
        )
        backlog_ratio = np.clip(self.backlog / (self.backlog_cap + EPS), 0.0, 1.0)
        backlog_factor = 1.0 - backlog_ratio
        support_factor = np.power(
            np.maximum(self.sigma, 0.0),
            1.0 + dependency_coefficient * self.dep_strength,
        )
        cap = np.sum(self.value * (1.0 - self.delta) * support_factor * self.kappa * backlog_factor)
        return float(np.clip(cap / (np.sum(self.value) + EPS), 0.0, 1.0))

    def _compute_losses(self) -> Dict[str, float]:
        key = self.criticality >= 0.70
        target = np.isin(self.type_id, [TYPE_TO_ID["airfield"], TYPE_TO_ID["assembly"], TYPE_TO_ID["transfer"]])
        backlog_ratio = np.clip(self.backlog / (self.backlog_cap + EPS), 0.0, 1.0)
        l_key = float(np.sum(self.criticality[key] * self.delta[key]) / (np.sum(self.criticality[key]) + EPS))
        l_route = float(1.0 - np.sum(self.value[target] * self.kappa[target]) / (np.sum(self.value[target]) + EPS))
        l_backlog = float(np.sum(self.value * backlog_ratio) / (np.sum(self.value) + EPS))
        l_chain = float(1.0 - self._compute_capability())
        return {"key": l_key, "route": l_route, "backlog": l_backlog, "chain": l_chain}

    @property
    def total_loss(self) -> float:
        loss_vec = self._loss_vector()
        # Reward/objective scalarization must remain independent of decoder
        # conditioning.  ``decoder_loss_weights`` is used only by morphology
        # descriptors/gating so matched-vs-mismatched evaluations do not move
        # the metric ruler at the same time as the decoder prior.
        weights = getattr(self, "loss_weights", np.full(4, 0.25, dtype=np.float32))
        return float(np.dot(weights, loss_vec))

    def _loss_vector(self) -> np.ndarray:
        return np.array(
            [self.losses["key"], self.losses["route"], self.losses["backlog"], self.losses["chain"]],
            dtype=np.float32,
        )

    def morphology_descriptors(self) -> np.ndarray:
        current = self._loss_vector()
        initial = np.maximum(getattr(self, "initial_loss_vector", current), 0.03)
        previous = getattr(self, "previous_loss_vector", current)
        weights = getattr(self, "decoder_loss_weights", np.full(4, 0.25, dtype=np.float32))
        priority = weights / np.maximum(float(np.mean(weights)), EPS)
        relative_pressure = priority * current / initial
        worsening_pressure = priority * np.maximum(current - previous, 0.0) / initial
        channel_scale = np.array([0.28, 0.18, 0.10, 0.34], dtype=np.float32)
        absolute_pressure = np.sqrt(priority * np.maximum(current, 0.0) / channel_scale)
        descriptors = 0.55 * relative_pressure + 0.30 * absolute_pressure + 0.15 * worsening_pressure
        return np.clip(descriptors, 0.0, 3.0).astype(np.float32)

    def route_break_strength(self) -> np.ndarray:
        route = self.a_route * self.route_edge_health
        broken = self.a_route * (1.0 - route)
        return np.clip(broken.sum(axis=1) / np.maximum(self.a_route.sum(axis=1), 1.0), 0.0, 1.0).astype(np.float32)

    def value_maps(self, no_chain: bool = False) -> Dict[str, np.ndarray]:
        backlog_ratio = np.clip(self.backlog / (self.backlog_cap + EPS), 0.0, 1.0)
        key = self.criticality * self.value * self.delta * (0.5 + 0.5 * self.sigma)
        route = (1.0 - self.kappa) * self.value + 0.6 * self.route_break_strength() + 0.35 * self.betweenness
        backlog = 0.8 * backlog_ratio + 0.45 * (1.0 - self.readiness) + 0.25 * self.demand
        local_chain_health = (
            (1.0 - self.delta)
            * np.maximum(self.sigma, 0.0)
            * np.maximum(self.kappa, 0.0)
            * (1.0 - backlog_ratio)
        )
        exponent = 1.0 + float(self.config.decoder_lambda_zeta) * self.dep_strength
        sigma_safe = np.maximum(self.sigma, EPS)
        cap_sensitivity = (
            self.value
            * (1.0 - self.delta)
            * np.maximum(self.kappa, 0.0)
            * (1.0 - backlog_ratio)
            * exponent
            * np.power(sigma_safe, exponent - 1.0)
        )
        dep_den = self.a_dep.sum(axis=0) + EPS
        downstream_sensitivity = self.a_dep @ (cap_sensitivity * self.dep_strength / dep_den)
        repair_sensitivity = self.repair_efficiency * (0.5 + 0.5 * self.readiness) / (self.repair_scale + EPS)
        source_restore_gap = (
            0.45 * self.delta
            + 0.25 * backlog_ratio
            + 0.20 * (1.0 - self.readiness)
            + 0.10 * (1.0 - np.maximum(self.sigma, 0.0))
        )
        provider_marginal = downstream_sensitivity * repair_sensitivity * source_restore_gap
        route_bridge_marginal = (
            self.betweenness
            * self.route_break_strength()
            * (1.0 - np.maximum(self.kappa, 0.0))
            * (0.35 + 0.65 * self.route_break_base)
        )
        local_chain_gap = self.value * self.dep_strength * (1.0 - local_chain_health) * repair_sensitivity
        direct_anchor = 0.40 * key + 0.32 * route + 0.28 * backlog

        def unit(values: np.ndarray) -> np.ndarray:
            v = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
            return v / (float(np.max(v)) + EPS)

        chain = (
            0.46 * unit(provider_marginal)
            + 0.20 * unit(route_bridge_marginal)
            + 0.22 * unit(local_chain_gap)
            + 0.12 * unit(direct_anchor)
        )
        if no_chain:
            chain = np.zeros_like(chain)
        return {
            "key": key.astype(np.float32),
            "route": route.astype(np.float32),
            "backlog": backlog.astype(np.float32),
            "chain": chain.astype(np.float32),
            "e2e_marginal": unit(provider_marginal).astype(np.float32),
        }

    @staticmethod
    def _unit_nonnegative(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
        return (values / (float(np.max(values)) + EPS)).astype(np.float32)

    def signal_matched_features(self) -> np.ndarray:
        """Task-information-matched node features for controlled baselines."""
        maps = self.value_maps(no_chain=False)
        columns = (
            maps["key"],
            maps["route"],
            maps["backlog"],
            maps["chain"],
            maps["e2e_marginal"],
            self.exec_cost,
        )
        return np.stack([self._unit_nonnegative(values) for values in columns], axis=1)

    def static_roi_score(self) -> np.ndarray:
        route_priority = 0.40 * self.route_break_base + 0.25 * self.betweenness
        type_priority = (
            0.36 * np.isin(self.type_id, [TYPE_TO_ID["airfield"], TYPE_TO_ID["transfer"]]).astype(np.float32)
            + 0.24 * np.isin(self.type_id, [TYPE_TO_ID["access"], TYPE_TO_ID["assembly"]]).astype(np.float32)
            + 0.16 * (self.a_dep.sum(axis=1) > 0).astype(np.float32)
        )
        score = 0.50 * self.value + 0.35 * self.criticality + route_priority + type_priority
        return score.astype(np.float32)

    def roi_score(self, dynamic: bool = True) -> np.ndarray:
        if not dynamic:
            return self.static_roi_score()
        maps = self.value_maps(no_chain=False)
        score = 0.30 * maps["key"] + 0.25 * maps["route"] + 0.20 * maps["backlog"] + 0.25 * maps["chain"]
        return score.astype(np.float32)

    def latent_basis_maps(self) -> Dict[str, np.ndarray]:
        """Generic latent basis maps for the LTS-SCP-style baseline.

        These maps use role, topology, demand, and readiness cues rather than
        the explicit four-channel task-morphology loss descriptors.
        """
        backlog_ratio = self.backlog / (self.backlog_cap + EPS)
        access_like = np.isin(
            self.type_id,
            [TYPE_TO_ID["airfield"], TYPE_TO_ID["access"], TYPE_TO_ID["transfer"], TYPE_TO_ID["comm"]],
        ).astype(np.float32)
        remote_like = np.isin(
            self.type_id,
            [TYPE_TO_ID["repair"], TYPE_TO_ID["assembly"], TYPE_TO_ID["buffer"]],
        ).astype(np.float32)
        role_value = (
            0.34 * self.value
            + 0.24 * self.criticality
            + 0.18 * self.repair_efficiency
            + 0.14 * access_like
            + 0.10 * remote_like
        )
        topology_value = (
            0.38 * self.betweenness
            + 0.24 * self.route_break_base
            + 0.20 * self.dep_strength
            + 0.18 * (self.a_dep.sum(axis=1) > 0).astype(np.float32)
        )
        demand_state = (
            0.34 * self.demand
            + 0.26 * backlog_ratio
            + 0.24 * self.delta
            + 0.16 * (1.0 - self.readiness)
        )
        availability_response = (
            0.30 * (1.0 - self.sigma)
            + 0.26 * (1.0 - self.kappa)
            + 0.22 * self.repair_efficiency
            + 0.22 * (1.0 - self.exec_cost / (float(np.max(self.exec_cost)) + EPS))
        )
        return {
            "latent_role": np.maximum(role_value, 0.0).astype(np.float32),
            "latent_topology": np.maximum(topology_value, 0.0).astype(np.float32),
            "latent_demand": np.maximum(demand_state, 0.0).astype(np.float32),
            "latent_response": np.maximum(availability_response, 0.0).astype(np.float32),
        }

    def decode_action(self, raw_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        mode = self.config.allocation_mode
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        expected_action_dim = int(np.prod(self.action_space.shape))
        if raw.size != expected_action_dim:
            raise ValueError(
                f"Action for mode {mode!r} has {raw.size} entries; expected {expected_action_dim}. "
                "Cross-size actions must never be resized silently."
            )
        if mode in ("direct_softmax", "full_action"):
            logits = raw
            w = softmax_np(logits, self.config.direct_softmax_temperature)
            return (self.budget * w).astype(np.float32), {"weights": w, "mode": mode}
        if mode == "direct_simplex":
            w = np.maximum(raw, 0.0)
            if float(w.sum()) <= EPS:
                w = np.full(self.n, 1.0 / self.n, dtype=np.float32)
            else:
                w = (w / float(w.sum())).astype(np.float32)
            return (self.budget * w).astype(np.float32), {"weights": w, "mode": mode}
        if mode == "direct_sparse_projection":
            scores = self._standardize(raw)
            w, lam, rho = sparse_budget_projection(scores, self.config.direct_sparse_projection_tau, max_support=None)
            return (
                self.budget * w
            ).astype(np.float32), {"weights": w, "lambda": lam, "rho": rho, "mode": mode}
        if mode in ("signal_matched_softmax", "signal_matched_sparse_projection"):
            if self.config.signal_matched_score_calibration != SIGNAL_MATCHED_SCORE_CALIBRATION:
                raise ValueError(
                    "Unsupported signal-matched score calibration: "
                    f"{self.config.signal_matched_score_calibration!r}."
                )
            if self.config.signal_matched_sparse_tau_normalization != SIGNAL_MATCHED_SPARSE_TAU_NORMALIZATION:
                raise ValueError(
                    "Unsupported signal-matched sparse temperature normalization: "
                    f"{self.config.signal_matched_sparse_tau_normalization!r}."
                )
            # Recovery signals, the local marginal-capability estimate, and
            # execution cost are already present in the per-region actor
            # observation.  This controlled policy therefore adds
            # no second analytical fusion stage and learns only N scores.
            actor_scores = raw
            raw_net_q = actor_scores
            net_q, score_calibration = calibrate_signal_matched_score(
                raw_net_q,
                scale_floor=float(self.config.signal_matched_score_scale_floor),
            )
            if mode == "signal_matched_softmax":
                temperature_parameter = max(float(self.config.direct_softmax_temperature), 1e-4)
                temperature = temperature_parameter
                w = softmax_np(net_q, temperature=temperature)
                lam = None
                rho = self.n
                projection = "softmax"
                temperature_normalization = "identity"
            else:
                # ``direct_sparse_projection_tau`` is retained as the frozen
                # scalar configuration knob, but for this matched control it
                # denotes a *per-region* quadratic concentration coefficient.
                # The sparse objective therefore uses
                #   q'w - (N * tau_bar / 2) ||w||^2,
                # which stays comparable when the number of regions changes
                # and avoids the old fixed-global-tau scale collapse.
                temperature_parameter = max(float(self.config.direct_sparse_projection_tau), 1e-4)
                temperature = float(self.n) * temperature_parameter
                w, lam_value, rho = sparse_budget_projection(net_q, temperature, max_support=None)
                lam = float(lam_value)
                projection = "simplex_threshold"
                temperature_normalization = self.config.signal_matched_sparse_tau_normalization
            decode = {
                "weights": w,
                "actor_score": np.asarray(actor_scores, dtype=np.float32),
                "fused_score": np.asarray(raw_net_q, dtype=np.float32),
                "raw_net_score": np.asarray(raw_net_q, dtype=np.float32),
                "preprojection_score": np.asarray(net_q, dtype=np.float32),
                "net_score": np.asarray(net_q, dtype=np.float32),
                "basis_names": [],
                "basis_coeff": [],
                "mu": 0.0,
                "tau": float(temperature),
                "temperature_parameter": float(temperature_parameter),
                "temperature_effective": float(temperature),
                "temperature_normalization": temperature_normalization,
                "lambda": lam,
                "rho": int(rho),
                "projection": projection,
                "information_control": "matched_features_direct_regional_scores",
                "mode": mode,
                **score_calibration,
            }
            return (self.budget * w).astype(np.float32), decode
        if mode == "roi_param":
            gamma = 1.0 + softplus_scalar(raw[0] if raw.size else 0.0)
            tau = 0.08 + softplus_scalar(raw[1] if raw.size > 1 else 0.0)
            simple_gap = 0.45 * self.delta + 0.30 * (1.0 - self.kappa) + 0.25 * self.backlog / (self.backlog_cap + EPS)
            q = gamma * self._standardize(0.75 * self.roi_score(dynamic=False) + 0.25 * simple_gap)
            w, lam, rho = sparse_budget_projection(q, tau, max_support=self._max_sparse_support())
            return (self.budget * w).astype(np.float32), {"weights": w, "lambda": lam, "rho": rho, "mode": mode}
        if mode in ("coverage_focus_dual", "fixed_dual"):
            share = float(sigmoid(raw[0] if raw.size else 0.0))
            route_w, _, _ = sparse_budget_projection(
                self._standardize(self.value_maps()["route"]),
                0.25,
                max_support=self._max_sparse_support(),
            )
            focus_w, _, _ = sparse_budget_projection(
                self._standardize(self.value_maps()["key"] + self.value_maps()["backlog"]),
                0.20,
                max_support=self._max_sparse_support(),
            )
            w = share * route_w + (1.0 - share) * focus_w
            w = w / np.maximum(w.sum(), EPS)
            return (self.budget * w).astype(np.float32), {"weights": w, "share": share, "mode": mode}
        if mode in (
            "tmsbd",
            "tmsbd_no_gate",
            "tmsbd_no_chain",
            "tmsbd_softmax_gate",
            "tmsbd_softmax_budget",
            "tmsbd_fixed_topk",
            "tmsbd_fixed_morphology",
            "tmsbd_single_critical",
            "tmsbd_single_support",
            "tmsbd_single_backlog",
            "tmsbd_single_e2e",
        ):
            return self._decode_tmsbd(raw, mode)
        if mode == "lts_scp_latent":
            return self._decode_lts_scp_latent(raw)
        return self._decode_tmsbd(raw, "tmsbd")

    def _decode_tmsbd(self, raw: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw = np.pad(raw, (0, max(0, 15 - raw.size)), mode="constant")
        alpha_logits = raw[:4]
        tau = 0.06 + 0.44 * sigmoid(raw[4])
        gamma = float(self.config.tmsbd_gamma_base) + float(self.config.tmsbd_gamma_scale) * softplus_scalar(
            raw[5] - float(self.config.tmsbd_gamma_bias)
        )
        mu = float(self.config.tmsbd_mu_scale) * softplus_scalar(raw[6] - float(self.config.tmsbd_mu_bias))
        descriptors = self.morphology_descriptors()
        single_index = {
            "tmsbd_single_critical": 0,
            "tmsbd_single_support": 1,
            "tmsbd_single_backlog": 2,
            "tmsbd_single_e2e": 3,
        }.get(mode)
        if single_index is not None:
            alpha = np.zeros(4, dtype=np.float32)
            alpha[single_index] = 1.0
        elif mode == "tmsbd_no_gate":
            alpha = np.full(4, 0.25, dtype=np.float32)
        elif mode == "tmsbd_fixed_morphology":
            alpha = np.maximum(
                getattr(self, "decoder_loss_weights", np.full(4, 0.25, dtype=np.float32)),
                0.0,
            )
            alpha = (alpha / max(float(alpha.sum()), EPS)).astype(np.float32)
        else:
            weights = getattr(self, "decoder_loss_weights", np.full(4, 0.25, dtype=np.float32))
            weight_logits = np.log(np.maximum(weights, 0.03) / 0.25)
            residual = float(self.config.tmsbd_gate_residual_scale) * alpha_logits
            morphology_coupling = np.array(
                [
                    [1.00, 0.08, 0.04, 0.18],
                    [0.08, 1.00, 0.06, 0.18],
                    [0.05, 0.05, 0.45, 0.20],
                    [0.18, 0.22, 0.55, 1.00],
                ],
                dtype=np.float32,
            )
            coupled_descriptors = morphology_coupling @ descriptors
            descriptor_mix = float(np.clip(self.config.tmsbd_descriptor_mix, 0.0, 1.0))
            shape_logits = (
                residual
                + descriptor_mix * coupled_descriptors
                + float(self.config.tmsbd_channel_prior_strength) * weight_logits
            )
            if mode == "tmsbd_softmax_gate":
                alpha = softmax_np(shape_logits, self.config.tmsbd_gate_softmax_temperature)
            else:
                alpha = sparsemax_np(shape_logits)
        maps = self.value_maps(no_chain=(mode == "tmsbd_no_chain"))
        names = ("key", "route", "backlog", "chain")
        gain_span = max(0.0, float(self.config.tmsbd_value_gain_span))
        gain_bias = max(0.0, float(self.config.tmsbd_value_gate_bias))
        gain_floor = max(0.0, float(self.config.tmsbd_value_gain_floor))
        if gain_bias > 0.0:
            channel_gain = gain_floor + gain_span * sigmoid(raw[7:11] - gain_bias)
        else:
            channel_gain = 1.0 + gain_span * (sigmoid(raw[7:11]) - 0.5)
        channel_gain = np.clip(channel_gain, max(0.0, gain_floor), 2.25).astype(np.float32)
        if mode in ("tmsbd_no_gate", "tmsbd_fixed_morphology") or single_index is not None:
            channel_gain = np.ones(4, dtype=np.float32)
        standardized = [gamma * float(channel_gain[idx]) * self._standardize(maps[name]) for idx, name in enumerate(names)]
        q = sum(float(alpha[idx]) * standardized[idx] for idx in range(4))
        backlog_ratio = self.backlog / (self.backlog_cap + EPS)
        residual_basis = np.stack(
            [
                self._standardize(self.delta),
                self._standardize(1.0 - self.kappa),
                self._standardize(backlog_ratio),
                self._standardize(1.0 - np.maximum(self.sigma, 0.0)),
            ],
            axis=0,
        )
        residual_coeff = float(self.config.tmsbd_value_residual_scale) * np.tanh(raw[11:15])
        if mode in ("tmsbd_no_gate", "tmsbd_fixed_morphology") or single_index is not None:
            residual_coeff = np.zeros(4, dtype=np.float32)
        q = q + gamma * np.sum(residual_coeff[:, None] * residual_basis, axis=0)
        net_q = q - mu * self.exec_cost
        tau = float(self.config.tmsbd_tau_multiplier) * tau
        value_contrast = float(np.std(net_q))
        low_contrast = value_contrast < float(self.config.tmsbd_projection_min_contrast)
        if mode == "tmsbd_softmax_budget":
            w = softmax_np(net_q, temperature=max(tau, 0.05))
            lam = float("nan")
            rho = self.n
        elif mode == "tmsbd_fixed_topk":
            k = max(1, int(math.ceil(self.config.fixed_topk_ratio * self.n)))
            idx = np.argsort(net_q)[::-1][:k]
            w = np.zeros(self.n, dtype=np.float32)
            local = softmax_np(net_q[idx], temperature=max(tau, 0.05))
            w[idx] = local
            lam = float(np.min(net_q[idx]))
            rho = k
        else:
            w, lam, rho = sparse_budget_projection(net_q, tau, max_support=None)
        gate_normalizer = "softmax" if mode == "tmsbd_softmax_gate" else "sparsemax"
        decode = {
            "weights": w,
            "fused_score": np.asarray(q, dtype=np.float32),
            "net_score": np.asarray(net_q, dtype=np.float32),
            "alpha": alpha.astype(float).tolist(),
            "descriptors": descriptors.astype(float).tolist(),
            "tau": float(tau),
            "gamma": float(gamma),
            "mu": float(mu),
            "gate_normalizer": gate_normalizer,
            "channel_gain": channel_gain.astype(float).tolist(),
            "residual_coeff": residual_coeff.astype(float).tolist(),
            "value_contrast": value_contrast,
            "low_contrast": bool(low_contrast),
            "lambda": float(lam) if np.isfinite(lam) else None,
            "rho": int(rho),
            "projection": "simplex_threshold" if mode != "tmsbd_softmax_budget" else "softmax",
            "mode": mode,
        }
        return (self.budget * w).astype(np.float32), decode

    def _decode_lts_scp_latent(self, raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw = np.pad(raw, (0, max(0, 15 - raw.size)), mode="constant")
        latent_logits = raw[:4]
        tau = 0.06 + 0.44 * sigmoid(raw[4])
        gamma = 0.35 + softplus_scalar(raw[5] - 0.50)
        mu = float(self.config.tmsbd_mu_scale) * softplus_scalar(raw[6] - float(self.config.tmsbd_mu_bias))
        latent_weights = sparsemax_np(latent_logits)
        basis = self.latent_basis_maps()
        names = ("latent_role", "latent_topology", "latent_demand", "latent_response")
        gains = 0.35 + 1.30 * sigmoid(raw[7:11])
        standardized = [
            gamma * float(gains[idx]) * self._standardize(basis[name])
            for idx, name in enumerate(names)
        ]
        q = sum(float(latent_weights[idx]) * standardized[idx] for idx in range(4))
        residual_basis = np.stack(
            [
                self._standardize(self.value),
                self._standardize(self.betweenness),
                self._standardize(self.demand),
                self._standardize(self.readiness),
            ],
            axis=0,
        )
        residual_coeff = 0.18 * np.tanh(raw[11:15])
        net_q = q + gamma * np.sum(residual_coeff[:, None] * residual_basis, axis=0) - mu * self.exec_cost
        w, lam, rho = sparse_budget_projection(net_q, tau, max_support=None)
        descriptors = np.array(
            [float(np.mean(basis[name])) for name in names],
            dtype=np.float32,
        )
        decode = {
            "weights": w,
            "alpha": latent_weights.astype(float).tolist(),
            "descriptors": descriptors.astype(float).tolist(),
            "tau": float(tau),
            "gamma": float(gamma),
            "mu": float(mu),
            "gate_normalizer": "latent_sparsemax",
            "channel_gain": gains.astype(float).tolist(),
            "residual_coeff": residual_coeff.astype(float).tolist(),
            "value_contrast": float(np.std(net_q)),
            "low_contrast": False,
            "lambda": float(lam) if np.isfinite(lam) else None,
            "rho": int(rho),
            "projection": "latent_simplex_threshold",
            "mode": "lts_scp_latent",
            "latent_basis_names": list(names),
        }
        return (self.budget * w).astype(np.float32), decode

    def _max_sparse_support(self) -> int:
        fraction = float(np.clip(self.config.max_sparse_support_fraction, 1.0 / max(self.n, 1), 1.0))
        return max(1, int(math.ceil(fraction * self.n)))

    @staticmethod
    def _standardize(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        return (v - float(np.mean(v))) / (float(np.std(v)) + 1e-6)

    def heuristic_action(self, name: str, **kwargs: Any) -> np.ndarray:
        if name == "uniform":
            return np.full(self.n, self.budget / self.n, dtype=np.float32)
        if name == "roi_proportional":
            score = np.maximum(self.roi_score(dynamic=False), 0.0) + EPS
            return (self.budget * score / score.sum()).astype(np.float32)
        if name == "roi_topk":
            ratio = float(kwargs.get("topk_ratio", self.config.topk_ratio))
            k = max(1, int(math.ceil(ratio * self.n)))
            score = self.roi_score(dynamic=False)
            idx = np.argsort(score)[::-1][:k]
            w = np.zeros(self.n, dtype=np.float32)
            local = np.maximum(score[idx], 0.0) + EPS
            w[idx] = local / local.sum()
            return (self.budget * w).astype(np.float32)
        if name in ("repair_gap_greedy", "service_deficit_greedy"):
            maps = self.value_maps()
            score = 0.45 * maps["key"] + 0.30 * maps["route"] + 0.25 * maps["backlog"]
            w, _, _ = sparse_budget_projection(score, 0.18, max_support=self._max_sparse_support())
            return (self.budget * w).astype(np.float32)
        if name == "one_step_marginal_greedy":
            score = self.value_maps()["chain"] + 0.4 * self.value_maps()["route"]
            w, _, _ = sparse_budget_projection(score, 0.12, max_support=self._max_sparse_support())
            return (self.budget * w).astype(np.float32)
        if name == "greedy_bottleneck_relief":
            maps = self.value_maps()
            losses = self._loss_vector()
            names = ("key", "route", "backlog", "chain")
            dominant = names[int(np.argmax(losses))]
            score = maps[dominant]
            w, _, _ = sparse_budget_projection(score, 0.14, max_support=self._max_sparse_support())
            return (self.budget * w).astype(np.float32)
        raise ValueError(f"Unknown heuristic: {name}")

    def _get_obs(self) -> np.ndarray:
        dynamic = np.stack(
            [
                self.delta,
                self.kappa,
                self.sigma,
                self.backlog / (self.backlog_cap + EPS),
                self.readiness,
                self.demand,
                self.last_budget / max(self.budget, EPS),
            ],
            axis=1,
        )
        node = np.concatenate([self.static_features, dynamic], axis=1)
        if self.signal_feature_dim:
            node = np.concatenate([node, self.signal_matched_features()], axis=1)
        global_features = np.array(
            [
                self.t / max(self.horizon, 1),
                self.capability,
                self.losses["key"],
                self.losses["route"],
                self.losses["backlog"],
                self.losses["chain"],
                self.loss_weights[0],
                self.loss_weights[1],
                self.loss_weights[2],
                self.loss_weights[3],
            ],
            dtype=np.float32,
        )
        return np.concatenate([node.reshape(-1), global_features], axis=0).astype(np.float32)

    def _info(self) -> Dict[str, Any]:
        w = self.last_budget / max(self.budget, EPS)
        numerical_zero_epsilon = 1e-12 * max(abs(self.budget), 1.0)
        effective_threshold = float(self.config.min_effective_share) * self.budget
        # Projection writes exact IEEE zeros; keep this distinct from both a
        # numerical-near-zero diagnostic and the operational threshold.
        positive_mask = self.last_budget > 0.0
        effective_mask = positive_mask & (self.last_budget >= effective_threshold)
        subthreshold_mask = positive_mask & ~effective_mask
        exact_zero_count = int(np.count_nonzero(self.last_budget == 0.0))
        numerical_zero_count = int(np.count_nonzero(self.last_budget <= numerical_zero_epsilon))
        positive_count = int(np.count_nonzero(positive_mask))
        effective_count = int(np.count_nonzero(effective_mask))
        subthreshold_count = int(np.count_nonzero(subthreshold_mask))
        info = {
            "t": int(self.t),
            "episode_seed": int(self._last_reset_seed),
            "capability": float(self.capability),
            "reference_capability": float(self._compute_capability(self.config.reference_lambda_zeta)),
            "total_loss": float(self.total_loss),
            "loss_key": float(self.losses["key"]),
            "loss_route": float(self.losses["route"]),
            "loss_backlog": float(self.losses["backlog"]),
            "loss_chain": float(self.losses["chain"]),
            "profile": self.current_profile,
            "loss_weight_key": float(self.loss_weights[0]),
            "loss_weight_route": float(self.loss_weights[1]),
            "loss_weight_backlog": float(self.loss_weights[2]),
            "loss_weight_chain": float(self.loss_weights[3]),
            "decoder_weight_key": float(self.decoder_loss_weights[0]),
            "decoder_weight_route": float(self.decoder_loss_weights[1]),
            "decoder_weight_backlog": float(self.decoder_loss_weights[2]),
            "decoder_weight_chain": float(self.decoder_loss_weights[3]),
            "lambda_zeta": float(self.config.lambda_zeta),
            "decoder_lambda_zeta": float(self.config.decoder_lambda_zeta),
            "reference_lambda_zeta": float(self.config.reference_lambda_zeta),
            "positive_count": positive_count,
            "positive_ratio": float(positive_count / self.n),
            "effective_count": effective_count,
            "effective_ratio": float(effective_count / self.n),
            # Backward-compatible aliases: active means operationally effective.
            "active_count": effective_count,
            "active_ratio": float(effective_count / self.n),
            "exact_zero_count": exact_zero_count,
            "exact_zero_ratio": float(exact_zero_count / self.n),
            "zero_ratio": float(exact_zero_count / self.n),
            "numerical_zero_count": numerical_zero_count,
            "numerical_zero_ratio": float(numerical_zero_count / self.n),
            "numerical_zero_epsilon": float(numerical_zero_epsilon),
            "subthreshold_positive_count": subthreshold_count,
            "subthreshold_positive_ratio": float(subthreshold_count / self.n),
            "subthreshold_capacity_mass": float(np.sum(self.last_budget[subthreshold_mask]) / max(self.budget, EPS)),
            "ineffective_ratio": float(1.0 - effective_count / self.n),
            "effective_allocation_threshold": float(effective_threshold),
            "budget_entropy": budget_entropy(w),
            "top_budget_share": float(np.max(w) if w.size else 0.0),
            "arrival_total": float(np.sum(self.last_arrival)),
            "arrival_mean": float(np.mean(self.last_arrival)),
            "arrival_multiplier_mean": float(np.mean(self.last_arrival_multiplier)),
            "arrival_multiplier_std": float(np.std(self.last_arrival_multiplier)),
            "budget": self.last_budget.copy(),
            "morphology_alpha": self.last_decode.get("alpha", [np.nan] * 4),
            "morphology_descriptors": self.last_decode.get("descriptors", self.morphology_descriptors().tolist()),
        }
        large_decode_fields = {
            "weights",
            "actor_score",
            "fused_score",
            "raw_net_score",
            "preprojection_score",
            "net_score",
            "residual_score",
        }
        info.update({f"decode_{k}": v for k, v in self.last_decode.items() if k not in large_decode_fields})
        return info


TheaterHexResourceEnv = JointMobilitySupportEnv


def make_env(config: TheaterHexEnvConfig | MapConfig | None = None) -> JointMobilitySupportEnv:
    return JointMobilitySupportEnv(config)
