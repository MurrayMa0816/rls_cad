
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def build_four_neighbor_adjacency_matrix(map_rows: int, map_cols: int) -> np.ndarray:
    """Build the historical row/column spatial graph used by the benchmark."""
    num_cells = map_rows * map_cols
    adjacency_matrix = np.zeros((num_cells, num_cells), dtype=np.float32)

    def idx_fn(row_index: int, col_index: int) -> int:
        return row_index * map_cols + col_index

    for row_index in range(map_rows):
        for col_index in range(map_cols):
            current_idx = idx_fn(row_index, col_index)
            candidate_neighbors = [
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ]
            for neighbor_row, neighbor_col in candidate_neighbors:
                if 0 <= neighbor_row < map_rows and 0 <= neighbor_col < map_cols:
                    adjacency_matrix[current_idx, idx_fn(neighbor_row, neighbor_col)] = 1.0
    return adjacency_matrix


def build_hex_adjacency_matrix(map_rows: int, map_cols: int) -> np.ndarray:
    """Backward-compatible name for the benchmark's rendered-cell lattice.

    The manuscript renders planning cells as hexagons, but the frozen
    computational benchmark uses four-neighbor row/column adjacency.  Keep the
    legacy helper name so old imports do not silently select a different graph.
    """
    return build_four_neighbor_adjacency_matrix(map_rows, map_cols)


class GraphCellFeatureExtractor(BaseFeaturesExtractor):
    """
    原始全局图特征提取器：
    保留兼容，服务 graph_dirichlet_ppo 等已有方法。
    """
    def __init__(
        self,
        observation_space,
        num_cells: int,
        cell_feature_dim: int,
        global_feature_dim: int,
        adjacency_matrix: np.ndarray,
        hidden_dim: int = 128,
        message_passing_steps: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.num_cells = int(num_cells)
        self.cell_feature_dim = int(cell_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)

        adjacency_matrix = adjacency_matrix.astype(np.float32)
        adjacency_matrix = adjacency_matrix + np.eye(self.num_cells, dtype=np.float32)
        degree_vector = adjacency_matrix.sum(axis=1, keepdims=True)
        normalized_adjacency = adjacency_matrix / np.maximum(degree_vector, 1e-6)
        self.register_buffer("normalized_adjacency", th.tensor(normalized_adjacency, dtype=th.float32))

        self.cell_embed_net = nn.Sequential(
            nn.Linear(self.cell_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_linear_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)])
        self.neighbor_linear_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)])

        self.global_feature_net = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        cell_part_dim = self.num_cells * self.cell_feature_dim
        cell_part = observations[:, :cell_part_dim]
        global_part = observations[:, cell_part_dim:]
        cell_feature_tensor = cell_part.reshape(batch_size, self.num_cells, self.cell_feature_dim)

        hidden_tensor = self.cell_embed_net(cell_feature_tensor)
        adjacency_batch = self.normalized_adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        for self_linear, neighbor_linear in zip(self.self_linear_list, self.neighbor_linear_list):
            neighbor_tensor = th.bmm(adjacency_batch, hidden_tensor)
            hidden_tensor = th.relu(self_linear(hidden_tensor) + neighbor_linear(neighbor_tensor))

        pooled_mean = hidden_tensor.mean(dim=1)
        pooled_max = hidden_tensor.max(dim=1).values
        pooled_graph_feature = 0.5 * (pooled_mean + pooled_max)

        global_feature = self.global_feature_net(global_part)
        final_feature = self.output_net(th.cat([pooled_graph_feature, global_feature], dim=1))
        return final_feature


class TypedGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with separate message passing for spatial, route, and
    dependency edges. This matches the RLS-CAD paper mechanism more closely than
    a flat MLP over the concatenated regional state.
    """

    def __init__(
        self,
        observation_space,
        num_cells: int,
        cell_feature_dim: int,
        global_feature_dim: int,
        adjacency_matrices: Dict[str, np.ndarray],
        hidden_dim: int = 128,
        message_passing_steps: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.num_cells = int(num_cells)
        self.cell_feature_dim = int(cell_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.edge_types = tuple(adjacency_matrices.keys())

        for edge_type, matrix in adjacency_matrices.items():
            normalized = self._normalize_adjacency(matrix)
            self.register_buffer(
                f"adjacency_{edge_type}",
                th.tensor(normalized, dtype=th.float32),
            )

        self.cell_embed_net = nn.Sequential(
            nn.Linear(self.cell_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_linear_list = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.edge_linear = nn.ModuleDict(
            {
                edge_type: nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(message_passing_steps)]
                )
                for edge_type in self.edge_types
            }
        )
        self.global_feature_net = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, features_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _normalize_adjacency(matrix: np.ndarray) -> np.ndarray:
        adjacency = np.asarray(matrix, dtype=np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}")
        if not np.all(np.isfinite(adjacency)):
            raise ValueError("Adjacency matrix must contain only finite values")
        if np.any(adjacency < 0.0):
            raise ValueError("Adjacency matrix must be nonnegative")
        degree = adjacency.sum(axis=1, keepdims=True)
        return adjacency / np.maximum(degree, 1.0)

    def _adjacency(self, edge_type: str, batch_size: int) -> th.Tensor:
        adjacency = getattr(self, f"adjacency_{edge_type}")
        return adjacency.unsqueeze(0).expand(batch_size, -1, -1)

    def set_adjacency_matrices(self, adjacency_matrices: Dict[str, np.ndarray]) -> None:
        """Replace graph buffers for a same-size structural-shift evaluation.

        Learned weights remain frozen; only the environment-provided graph is
        synchronized.  Shape/key checks are deliberately strict so a shifted
        environment can never be evaluated with stale topology silently.
        """
        provided = set(adjacency_matrices)
        expected = set(self.edge_types)
        if provided != expected:
            raise ValueError(
                f"Adjacency edge types differ: expected {sorted(expected)}, got {sorted(provided)}"
            )
        replacements: Dict[str, th.Tensor] = {}
        for edge_type in self.edge_types:
            normalized = self._normalize_adjacency(adjacency_matrices[edge_type])
            if normalized.shape != (self.num_cells, self.num_cells):
                raise ValueError(
                    f"Adjacency {edge_type!r} has shape {normalized.shape}; "
                    f"expected {(self.num_cells, self.num_cells)}"
                )
            buffer = getattr(self, f"adjacency_{edge_type}")
            replacements[edge_type] = th.as_tensor(
                normalized,
                dtype=buffer.dtype,
                device=buffer.device,
            )
        # Commit only after every edge type has passed validation.
        for edge_type, replacement in replacements.items():
            buffer = getattr(self, f"adjacency_{edge_type}")
            with th.no_grad():
                buffer.copy_(replacement)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        cell_part_dim = self.num_cells * self.cell_feature_dim
        cell_part = observations[:, :cell_part_dim]
        global_part = observations[:, cell_part_dim:]
        cell_feature_tensor = cell_part.reshape(batch_size, self.num_cells, self.cell_feature_dim)

        hidden_tensor = self.cell_embed_net(cell_feature_tensor)
        for layer_idx, self_linear in enumerate(self.self_linear_list):
            message_tensor = self_linear(hidden_tensor)
            for edge_type in self.edge_types:
                adjacency_batch = self._adjacency(edge_type, batch_size)
                neighbor_tensor = th.bmm(adjacency_batch, hidden_tensor)
                message_tensor = message_tensor + self.edge_linear[edge_type][layer_idx](neighbor_tensor)
            hidden_tensor = th.relu(message_tensor)

        pooled_mean = hidden_tensor.mean(dim=1)
        pooled_max = hidden_tensor.max(dim=1).values
        global_feature = self.global_feature_net(global_part)
        return self.output_net(th.cat([pooled_mean, pooled_max, global_feature], dim=1))


def typed_adjacency_matrices_from_env(env: Any) -> Dict[str, np.ndarray]:
    """Build the canonical typed graph passed to trained policy extractors."""
    dependency = np.maximum(
        np.asarray(env.a_dep, dtype=np.float32),
        np.asarray(env.a_dep, dtype=np.float32).T,
    )
    return {
        "adj": np.asarray(env.a_adj, dtype=np.float32),
        "route": np.asarray(env.a_route, dtype=np.float32),
        "dep": dependency.astype(np.float32),
    }


def synchronize_typed_graph_extractors(model_or_policy: Any, env: Any, *, require: bool = True) -> int:
    """Synchronize every unique typed extractor with a same-size environment.

    This supports structural-shift evaluation for both shared and separate
    actor/value feature extractors.  It intentionally rejects cross-size
    replacement; cross-size policies must be rebuilt before weights are moved.
    """
    policy = getattr(model_or_policy, "policy", model_or_policy)
    if not isinstance(policy, nn.Module):
        raise TypeError(f"Expected a torch policy/module, got {type(policy)!r}")
    matrices = typed_adjacency_matrices_from_env(env)
    extractors = [module for module in policy.modules() if isinstance(module, TypedGraphFeatureExtractor)]
    if require and not extractors:
        raise RuntimeError("Policy contains no TypedGraphFeatureExtractor to synchronize")
    # Validate every extractor before mutating any buffer.
    for extractor in extractors:
        if extractor.num_cells != int(env.n):
            raise ValueError(
                f"Cannot synchronize {extractor.num_cells}-cell extractor with {int(env.n)}-cell environment; "
                "rebuild the policy for cross-size transfer"
            )
        if extractor.cell_feature_dim != int(env.node_feature_dim):
            raise ValueError(
                f"Node feature dimension differs: policy={extractor.cell_feature_dim}, "
                f"environment={int(env.node_feature_dim)}"
            )
        if extractor.global_feature_dim != int(env.global_feature_dim):
            raise ValueError(
                f"Global feature dimension differs: policy={extractor.global_feature_dim}, "
                f"environment={int(env.global_feature_dim)}"
            )
        if set(extractor.edge_types) != set(matrices):
            raise ValueError(
                f"Policy edge types differ: expected {sorted(matrices)}, "
                f"got {sorted(extractor.edge_types)}"
            )
        for matrix in matrices.values():
            extractor._normalize_adjacency(matrix)
    for extractor in extractors:
        extractor.set_adjacency_matrices(matrices)
    return len(extractors)


class GroupAwareGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    新的 group-aware extractor：
    除了全图 pooled feature，还会按 frontline / core / rear 分组做 pooling。
    这样更适合新的 contextual priority allocator。
    """
    def __init__(
        self,
        observation_space,
        num_cells: int,
        cell_feature_dim: int,
        global_feature_dim: int,
        adjacency_matrix: np.ndarray,
        zone_group_id_vector: np.ndarray,
        hidden_dim: int = 128,
        message_passing_steps: int = 2,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.num_cells = int(num_cells)
        self.cell_feature_dim = int(cell_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.num_groups = int(np.max(zone_group_id_vector)) + 1

        adjacency_matrix = adjacency_matrix.astype(np.float32)
        adjacency_matrix = adjacency_matrix + np.eye(self.num_cells, dtype=np.float32)
        degree_vector = adjacency_matrix.sum(axis=1, keepdims=True)
        normalized_adjacency = adjacency_matrix / np.maximum(degree_vector, 1e-6)
        self.register_buffer("normalized_adjacency", th.tensor(normalized_adjacency, dtype=th.float32))

        self.register_buffer("zone_group_id_vector", th.tensor(zone_group_id_vector.astype(np.int64), dtype=th.long))

        self.cell_embed_net = nn.Sequential(
            nn.Linear(self.cell_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_linear_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)])
        self.neighbor_linear_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps)])

        self.global_feature_net = nn.Sequential(
            nn.Linear(self.global_feature_dim, hidden_dim),
            nn.ReLU(),
        )

        input_concat_dim = hidden_dim * (3 + self.num_groups)  # global mean + global max + 3 group means
        self.output_net = nn.Sequential(
            nn.Linear(input_concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        cell_part_dim = self.num_cells * self.cell_feature_dim
        cell_part = observations[:, :cell_part_dim]
        global_part = observations[:, cell_part_dim:]
        cell_feature_tensor = cell_part.reshape(batch_size, self.num_cells, self.cell_feature_dim)

        hidden_tensor = self.cell_embed_net(cell_feature_tensor)
        adjacency_batch = self.normalized_adjacency.unsqueeze(0).expand(batch_size, -1, -1)

        for self_linear, neighbor_linear in zip(self.self_linear_list, self.neighbor_linear_list):
            neighbor_tensor = th.bmm(adjacency_batch, hidden_tensor)
            hidden_tensor = th.relu(self_linear(hidden_tensor) + neighbor_linear(neighbor_tensor))

        pooled_global_mean = hidden_tensor.mean(dim=1)
        pooled_global_max = hidden_tensor.max(dim=1).values

        pooled_group_feature_list = []
        for group_id in range(self.num_groups):
            group_mask = (self.zone_group_id_vector == group_id).float().view(1, self.num_cells, 1)
            group_mask = group_mask.expand(batch_size, -1, -1)
            group_count = th.clamp(group_mask.sum(dim=1), min=1.0)
            pooled_group_mean = (hidden_tensor * group_mask).sum(dim=1) / group_count
            pooled_group_feature_list.append(pooled_group_mean)

        global_feature = self.global_feature_net(global_part)
        concat_feature = th.cat([pooled_global_mean, pooled_global_max] + pooled_group_feature_list + [global_feature], dim=1)
        final_feature = self.output_net(concat_feature)
        return final_feature
