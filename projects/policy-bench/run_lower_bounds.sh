#!/usr/bin/env bash

uv run policy-bench --dp.iterations=10_000 --data.dataset=lower_bound/stable/1 --dp.model=mlp --dp.action_horizon=1 --data.action_length=1 --dp.relative_action=False --dp.prediction_type=sample
uv run policy-bench --dp.iterations=10_000 --data.dataset=lower_bound/stable/1 --dp.model=mlp --dp.action_horizon=4 --data.action_length=4 --dp.relative_action=False --dp.prediction_type=sample
uv run policy-bench --dp.iterations=10_000 --data.dataset=lower_bound/stable/1 --dp.model=mlp --dp.action_horizon=8 --data.action_length=8 --dp.relative_action=False --dp.prediction_type=sample
