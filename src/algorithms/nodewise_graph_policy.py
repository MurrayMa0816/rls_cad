"""Node-wise typed-graph actor with a pooled graph critic.

The high-dimensional graph controls in Paper 03 execute one regional score per
node (and, for the signal-matched controls, six additional global controls).
A graph encoder that pools all nodes before the policy head removes the direct
node-to-action correspondence.  This module keeps the regional embeddings
through the actor path and applies one shared score head to every node.  Only
the critic and the optional global controls use the pooled graph context.

The fixed 15-dimensional RLS-CAD policy deliberately does not use these
classes; it retains the historical ``TypedGraphFeatureExtractor`` path.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Sequence

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .graph_features import TypedGraphFeatureExtractor


class NodeWiseTypedGraphFeatureExtractor(TypedGraphFeatureExtractor):
    """Typed message passing that preserves every node embedding.

    Subclassing :class:`TypedGraphFeatureExtractor` keeps the existing graph
    synchronization and structural-audit utilities compatible.  The parent
    constructor is intentionally not called because its final operation is a
    global pooling projection.  The message-passing module names and graph
    buffer contract remain the same.
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
    ):
        self.num_cells = int(num_cells)
        self.cell_feature_dim = int(cell_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        if self.num_cells <= 0:
            raise ValueError(f"num_cells must be positive, got {self.num_cells}")
        if self.cell_feature_dim <= 0 or self.global_feature_dim <= 0:
            raise ValueError("Cell and global feature dimensions must be positive")
        if self.hidden_dim <= 0 or self.message_passing_steps <= 0:
            raise ValueError("hidden_dim and message_passing_steps must be positive")
        if not adjacency_matrices:
            raise ValueError("At least one typed adjacency matrix is required")

        self.edge_types = tuple(adjacency_matrices.keys())
        # Packed output = N node embeddings + mean/max/global context.
        packed_features_dim = (self.num_cells + 3) * self.hidden_dim
        BaseFeaturesExtractor.__init__(self, observation_space, packed_features_dim)

        for edge_type, matrix in adjacency_matrices.items():
            normalized = self._normalize_adjacency(matrix)
            if normalized.shape != (self.num_cells, self.num_cells):
                raise ValueError(
                    f"Adjacency {edge_type!r} has shape {normalized.shape}; "
                    f"expected {(self.num_cells, self.num_cells)}"
                )
            self.register_buffer(
                f"adjacency_{edge_type}",
                th.tensor(normalized, dtype=th.float32),
            )

        self.cell_embed_net = nn.Sequential(
            nn.Linear(self.cell_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.self_linear_list = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.message_passing_steps)]
        )
        self.edge_linear = nn.ModuleDict(
            {
                edge_type: nn.ModuleList(
                    [
                        nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                        for _ in range(self.message_passing_steps)
                    ]
                )
                for edge_type in self.edge_types
            }
        )
        self.global_feature_net = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.hidden_dim),
            nn.ReLU(),
        )

    @property
    def context_dim(self) -> int:
        """Dimension of the pooled mean/max/global critic context."""

        return 3 * self.hidden_dim

    def encode_nodes_and_context(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Return ``[batch, N, hidden]`` nodes and invariant graph context."""

        if observations.ndim != 2:
            raise ValueError(
                f"Expected a batched rank-2 observation tensor, got {tuple(observations.shape)}"
            )
        batch_size = observations.shape[0]
        cell_part_dim = self.num_cells * self.cell_feature_dim
        expected_dim = cell_part_dim + self.global_feature_dim
        if observations.shape[1] != expected_dim:
            raise ValueError(
                f"Observation width is {observations.shape[1]}; expected {expected_dim}"
            )
        cell_part = observations[:, :cell_part_dim]
        global_part = observations[:, cell_part_dim:]
        cell_features = cell_part.reshape(batch_size, self.num_cells, self.cell_feature_dim)

        node_embeddings = self.cell_embed_net(cell_features)
        for layer_idx, self_linear in enumerate(self.self_linear_list):
            messages = self_linear(node_embeddings)
            for edge_type in self.edge_types:
                neighbor_embeddings = th.bmm(
                    self._adjacency(edge_type, batch_size),
                    node_embeddings,
                )
                messages = messages + self.edge_linear[edge_type][layer_idx](neighbor_embeddings)
            node_embeddings = th.relu(messages)

        pooled_mean = node_embeddings.mean(dim=1)
        pooled_max = node_embeddings.max(dim=1).values
        global_embedding = self.global_feature_net(global_part)
        context = th.cat([pooled_mean, pooled_max, global_embedding], dim=1)
        return node_embeddings, context

    def forward(self, observations: th.Tensor) -> th.Tensor:
        node_embeddings, context = self.encode_nodes_and_context(observations)
        return th.cat([node_embeddings.flatten(start_dim=1), context], dim=1)


def _hidden_body(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation_fn: type[nn.Module],
) -> tuple[nn.Module, int]:
    modules: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimensions must be positive, got {hidden_dim}")
        modules.extend([nn.Linear(current_dim, hidden_dim), activation_fn()])
        current_dim = hidden_dim
    return (nn.Sequential(*modules) if modules else nn.Identity()), current_dim


class NodeWiseActorCriticNetwork(nn.Module):
    """Shared regional score head and pooled-context critic network."""

    def __init__(
        self,
        *,
        features_dim: int,
        num_cells: int,
        node_embedding_dim: int,
        action_dim: int,
        node_actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        activation_fn: type[nn.Module],
    ):
        super().__init__()
        self.features_dim = int(features_dim)
        self.num_cells = int(num_cells)
        self.node_embedding_dim = int(node_embedding_dim)
        self.action_dim = int(action_dim)
        self.context_dim = 3 * self.node_embedding_dim
        expected_features_dim = self.num_cells * self.node_embedding_dim + self.context_dim
        if self.features_dim != expected_features_dim:
            raise ValueError(
                f"Packed feature width is {self.features_dim}; expected {expected_features_dim}"
            )
        self.num_global_actions = self.action_dim - self.num_cells
        if self.num_global_actions not in (0, 6):
            raise ValueError(
                "Node-wise Paper 03 policies require N regional actions or N+6 "
                f"regional/global actions, got N={self.num_cells}, action_dim={self.action_dim}"
            )

        node_input_dim = self.node_embedding_dim + self.context_dim
        self.node_actor_body, node_latent_dim = _hidden_body(
            node_input_dim,
            node_actor_hidden_dims,
            activation_fn,
        )
        self.node_score_head = nn.Linear(node_latent_dim, 1)

        if self.num_global_actions:
            self.global_actor_body, global_latent_dim = _hidden_body(
                self.context_dim,
                node_actor_hidden_dims,
                activation_fn,
            )
            self.global_action_head: nn.Module | None = nn.Linear(
                global_latent_dim,
                self.num_global_actions,
            )
        else:
            self.global_actor_body = None
            self.global_action_head = None

        self.critic_body, critic_latent_dim = _hidden_body(
            self.context_dim,
            critic_hidden_dims,
            activation_fn,
        )
        # These attributes are part of the interface expected by SB3 policies.
        # The actor latent is already the complete Gaussian mean vector.
        self.latent_dim_pi = self.action_dim
        self.latent_dim_vf = critic_latent_dim

    def _unpack(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.features_dim:
            raise ValueError(
                f"Expected packed features [batch, {self.features_dim}], got {tuple(features.shape)}"
            )
        node_width = self.num_cells * self.node_embedding_dim
        node_embeddings = features[:, :node_width].reshape(
            features.shape[0],
            self.num_cells,
            self.node_embedding_dim,
        )
        context = features[:, node_width:]
        return node_embeddings, context

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        node_embeddings, context = self._unpack(features)
        repeated_context = context.unsqueeze(1).expand(-1, self.num_cells, -1)
        node_inputs = th.cat([node_embeddings, repeated_context], dim=-1)
        regional_scores = self.node_score_head(self.node_actor_body(node_inputs)).squeeze(-1)
        if self.global_action_head is None or self.global_actor_body is None:
            return regional_scores
        global_controls = self.global_action_head(self.global_actor_body(context))
        return th.cat([regional_scores, global_controls], dim=1)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        _node_embeddings, context = self._unpack(features)
        return self.critic_body(context)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class NodeWiseActorCriticPolicy(ActorCriticPolicy):
    """PPO policy for N or N+6 actions with explicit node correspondence."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        node_actor_hidden_dims: Sequence[int] = (256, 256, 128),
        critic_hidden_dims: Sequence[int] = (256, 256, 128),
        **kwargs,
    ):
        self.node_actor_hidden_dims = tuple(int(value) for value in node_actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(value) for value in critic_hidden_dims)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        extractor = self.features_extractor
        if not isinstance(extractor, NodeWiseTypedGraphFeatureExtractor):
            raise TypeError(
                "NodeWiseActorCriticPolicy requires NodeWiseTypedGraphFeatureExtractor, "
                f"received {type(extractor)!r}"
            )
        self.mlp_extractor = NodeWiseActorCriticNetwork(
            features_dim=self.features_dim,
            num_cells=extractor.num_cells,
            node_embedding_dim=extractor.hidden_dim,
            action_dim=get_action_dim(self.action_space),
            node_actor_hidden_dims=self.node_actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            activation_fn=self.activation_fn,
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError(
                "NodeWiseActorCriticPolicy supports continuous Box actions with a "
                f"diagonal Gaussian distribution, received {type(self.action_dist)!r}"
            )

        action_dim = get_action_dim(self.action_space)
        # ``mlp_extractor.forward_actor`` already emits the structured action
        # mean.  Keeping this layer as Identity prevents a dense N-by-N matrix
        # from reintroducing cross-node mixing after the shared score head.
        self.action_net = nn.Identity()
        self.log_std = nn.Parameter(th.ones(action_dim) * self.log_std_init)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            self.features_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            # Gaussian action means use the same small output scale as SB3's
            # default ActorCriticPolicy.
            self.mlp_extractor.node_score_head.apply(
                partial(self.init_weights, gain=0.01)
            )
            if self.mlp_extractor.global_action_head is not None:
                self.mlp_extractor.global_action_head.apply(
                    partial(self.init_weights, gain=0.01)
                )
            self.value_net.apply(partial(self.init_weights, gain=1.0))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            node_actor_hidden_dims=self.node_actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
        )
        return data


__all__ = [
    "NodeWiseActorCriticNetwork",
    "NodeWiseActorCriticPolicy",
    "NodeWiseTypedGraphFeatureExtractor",
]
