# RLS-CAD

This repository provides the implementation of Recovery-Limiting Signal-Conditioned Action Decoding (RLS-CAD) used in *Recovery-Limiting Signal-Conditioned Action Decoding for Emergency Wireless Access--Backhaul Capacity Allocation*.

It contains the normalized planning-level simulator, the 15-dimensional policy interface, recovery-signal decoding, finite-support simplex projection, the experimental configuration, fixed data partitions, and the training and evaluation programs.

## Scope

The code implements the planning-level benchmark described in the paper. Physical-channel simulation and PHY/MAC scheduling are outside its scope.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Smoke test

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

The test constructs the environment, decodes an RLS-CAD action, and checks nonnegativity and fixed-budget conservation.

## Training check

The following command runs a short CPU training session. Each run requires a new output directory.

```bash
PYTHONPATH=src python -m experiment.train \
  --config configs/paper.yaml \
  --seed 0 \
  --device cpu \
  --total-timesteps 1024 \
  --outdir runs/smoke_seed0
```

## Experimental protocol

The reported experiments use five training seeds, 80,000 steps per seed, final checkpoints, and 100 shared test scenarios. On a CUDA machine, run each seed in a separate directory:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m experiment.train \
    --config configs/paper.yaml \
    --seed "$seed" \
    --device cuda:0 \
    --outdir "runs/rls_cad_seed${seed}"
done
```

Evaluate the completed runs with the frozen test suite:

```bash
PYTHONPATH=src python -m experiment.evaluate \
  --suite-root runs \
  --seed-manifest configs/seed_manifest.json \
  --test-suite in_distribution \
  --model-artifact final \
  --device cpu \
  --no-include-heuristics \
  --trace-mode none \
  --output-dir evaluation/main
```

## Main entry points

- `src/env/environment.py`: normalized access--backhaul recovery simulator.
- `src/env/action_decoders.py`: RLS-CAD action interface.
- `src/algorithms/graph_features.py`: typed graph feature extractor.
- `src/experiment/train.py`: training and run-manifest generation.
- `src/experiment/evaluate.py`: held-out evaluation.
- `configs/paper.yaml`: paper hyperparameters.
- `configs/seed_manifest.json`: frozen training, validation, and test partitions.

## Reproducibility notes

- A run directory cannot overwrite an existing run.
- The training process records the resolved configuration, dependency versions, hardware information, and a source snapshot hash.
- A release tag can be used to identify the version associated with the reported results.
