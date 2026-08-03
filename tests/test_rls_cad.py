from __future__ import annotations

import copy

import numpy as np

from experiment.pipeline import build_env_by_method, build_total_config


def make_environment():
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
    return build_env_by_method("graph_tmsbd_ppo", copy.deepcopy(config))


def test_action_interface_has_fifteen_dimensions():
    environment = make_environment()
    assert environment.action_space.shape == (15,)


def test_decoder_conserves_the_capacity_budget():
    environment = make_environment()
    environment.reset(seed=0)
    action = np.zeros(environment.action_space.shape, dtype=np.float32)
    environment.step(action)
    allocation = np.asarray(environment.last_budget, dtype=np.float64)

    assert allocation.shape == (environment.n,)
    assert np.all(allocation >= 0.0)
    assert np.isclose(allocation.sum(), environment.budget, atol=1e-8)


def test_seeded_transition_is_reproducible():
    first = make_environment()
    second = make_environment()
    first_observation, _ = first.reset(seed=17)
    second_observation, _ = second.reset(seed=17)
    action = np.zeros(first.action_space.shape, dtype=np.float32)
    first_step = first.step(action)
    second_step = second.step(action)

    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_step[0], second_step[0])
    assert first_step[1] == second_step[1]
    assert first_step[2] == second_step[2]
    assert first_step[3] == second_step[3]
    first_info = first_step[4]
    second_info = second_step[4]
    assert first_info.keys() == second_info.keys()
    for key in first_info:
        first_value = np.asarray(first_info[key])
        second_value = np.asarray(second_info[key])
        assert np.array_equal(first_value, second_value)
