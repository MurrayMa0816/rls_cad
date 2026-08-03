from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment.pipeline import build_env_by_method, build_total_config  # noqa: E402


def main() -> None:
    config = build_total_config(
        mode="restore",
        map_rows=12,
        map_cols=15,
        horizon_steps=40,
        resource_budget=1.0,
        scenario_id="emergency_wireless_access_backhaul_v1",
        seed_value=0,
        profile="mixed",
        tmsbd_tau_multiplier=1.25,
        switch_penalty_weight=0.002,
        lambda_zeta=1.50,
        decoder_lambda_zeta=1.50,
        reference_lambda_zeta=1.50,
    )
    env = build_env_by_method("graph_tmsbd_ppo", copy.deepcopy(config))
    observation, _ = env.reset(seed=0)
    latent_action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, truncated, info = env.step(latent_action)
    allocation = np.asarray(env.last_budget, dtype=np.float64)

    if observation.shape != env.observation_space.shape:
        raise AssertionError("Observation shape does not match the declared space")
    if allocation.shape != (env.n,):
        raise AssertionError(f"Unexpected allocation shape: {allocation.shape}")
    if np.min(allocation) < -1e-10:
        raise AssertionError("Allocation contains a negative value")
    if not np.isclose(np.sum(allocation), env.budget, atol=1e-8):
        raise AssertionError("Allocation does not conserve the fixed total budget")
    if terminated or truncated:
        raise AssertionError("The environment ended after the first action")

    print(
        {
            "status": "ok",
            "regions": env.n,
            "observation_dim": int(observation.size),
            "action_dim": int(latent_action.size),
            "allocation_sum": float(np.sum(allocation)),
            "positive_regions": int(np.count_nonzero(allocation > 0.0)),
        }
    )


if __name__ == "__main__":
    main()
