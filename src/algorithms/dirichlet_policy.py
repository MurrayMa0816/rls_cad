from __future__ import annotations

from typing import Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim


class DirichletSimplexDistribution(Distribution):
    """
    定义在 simplex 上的 Dirichlet 动作分布：
    动作天然满足 a_i >= 0, sum_i a_i = 1。
    """

    def __init__(self, action_dim: int, min_concentration: float = 0.05):
        super().__init__()
        self.action_dim = int(action_dim)
        self.min_concentration = float(min_concentration)
        self.distribution: Optional[th.distributions.Dirichlet] = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, alpha_logits: th.Tensor):
        concentration = F.softplus(alpha_logits) + self.min_concentration
        self.distribution = th.distributions.Dirichlet(concentration)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        actions = th.clamp(actions, 1e-8, 1.0)
        actions = actions / th.clamp(actions.sum(dim=1, keepdim=True), min=1e-8)
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        concentration = self.distribution.concentration
        if th.all(concentration > 1.0):
            numerator = concentration - 1.0
            return numerator / th.clamp(numerator.sum(dim=1, keepdim=True), min=1e-8)
        return concentration / th.clamp(concentration.sum(dim=1, keepdim=True), min=1e-8)

    def actions_from_params(
        self,
        alpha_logits: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        self.proba_distribution(alpha_logits)
        actions = self.get_actions(deterministic=deterministic)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob_from_params(self, alpha_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.actions_from_params(alpha_logits)


class DirichletActorCriticPolicy(ActorCriticPolicy):
    """
    SB3 PPO 的自定义 Dirichlet policy。
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        self._lr_schedule = lr_schedule

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        action_dim = get_action_dim(self.action_space)
        self.action_dist = DirichletSimplexDistribution(action_dim=action_dim)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=self._lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        alpha_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(alpha_logits)
