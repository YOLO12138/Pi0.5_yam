# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenPI** is a Vision-Language-Action (VLA) model repository from Physical Intelligence (π) for robotics. It supports three model families:
- **π₀ (Pi0)**: Flow-based VLA model
- **π₀-FAST (Pi0-FAST)**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅ (Pi05)**: Upgraded π₀ with better open-world generalization via knowledge insulation

This fork includes a **YAM bimanual robot** integration (14D joint space: `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]`).

## Setup

Python 3.11+ is required. The project uses `uv` for dependency management (not pip/conda):

```bash
# Install uv: https://docs.astral.sh/uv/
pip install uv

# Install project
uv sync

# Or with CUDA 12 JAX support
uv sync --extra cuda12
```

## Common Commands

```bash
# Run tests
uv run pytest

# Linting and formatting
uv run ruff check .
uv run ruff format .

# Pre-commit (runs ruff + uv lock check)
pre-commit run --all-files
```

### Training

```bash
# Compute normalization stats (required before JAX training)
uv run scripts/compute_norm_stats.py --config-name <config>

# JAX training (single node)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name>

# PyTorch training (single GPU)
uv run scripts/train_pytorch.py <config> --exp_name <name>

# PyTorch training (multi-GPU)
uv run torchrun --nproc_per_node=<N> scripts/train_pytorch.py <config> --exp_name <name>
```

### Inference

```bash
# Serve a policy via WebSocket
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=<config> \
  --policy.dir=<checkpoint_path>
```

## Architecture

### Package Layout

- **`src/openpi/`** — main package
- **`packages/openpi-client/`** — lightweight client package with minimal deps (numpy, websockets, msgpack); used by robots to communicate with the policy server
- **`examples/`** — integration examples for ALOHA, DROID, LIBERO, UR5, YAM
- **`deployment/`** — YAM robot deployment bridge (`openpi_bridge.py`)
- **`scripts/`** — training, serving, and utility scripts

### Core Modules (`src/openpi/`)

| Module | Purpose |
|---|---|
| `models/` | JAX/Flax model implementations (Pi0, Pi0-FAST, Pi05, Gemma, SigLIP, LoRA) |
| `models_pytorch/` | PyTorch equivalents (Pi0, Gemma, preprocessing) |
| `policies/` | Robot-specific policy wrappers (ALOHA, DROID, LIBERO, YAM) |
| `training/` | Training infrastructure (data loading, checkpoints, optimizer, FSDP sharding) |
| `serving/` | WebSocket-based policy server for remote inference |
| `shared/` | Utilities — GCS download, normalization, image tools, NNX helpers |
| `transforms.py` | Data transformation pipeline (normalization, image resize, tokenization, action padding) |

### Data & Transform Pipeline

The transform pipeline is layered:
1. **Repack transforms** — reshape raw dataset observations into canonical keys
2. **Data transforms** — normalize state/actions, resize images
3. **Model transforms** — tokenize language, pad/chunk actions for the specific model

Normalization supports two modes: `z_score` (mean/std) and `quantile` (robust). Stats are precomputed by `scripts/compute_norm_stats.py` and stored as assets alongside checkpoints.

### Configuration

Training configs are dataclasses managed via **tyro** (CLI) and **ml_collections**. The main config class is `TrainConfig` in `src/openpi/training/config.py`. Policy configs in `src/openpi/policies/policy_config.py` act as a factory that maps config names to model + transform combinations.

### Checkpoint Storage

Model checkpoints are hosted on GCS and auto-downloaded to `~/.cache/openpi` via `src/openpi/shared/download.py`. The `Orbax` library handles checkpoint serialization.

### Dual Framework Support

JAX and PyTorch implementations coexist. JAX is the primary framework for training (supports LoRA, EMA, FSDP). PyTorch models in `models_pytorch/` mirror the JAX equivalents and are used for inference or alternative training workflows. `examples/convert_jax_model_to_pytorch.py` converts between formats.

### YAM Robot Integration

- **Policy**: `src/openpi/policies/yam_policy.py` — defines obs/action space for the 14-DOF YAM bimanual arm
- **Deployment bridge**: `deployment/openpi_bridge.py` — adapts between OpenPI's canonical observation format and the raiden motor control loop; handles head/wrist camera images and joint order remapping

## Code Style

- Line length: 120 characters (enforced by ruff)
- Ruff rules: `E`, `F`, `I`, `B`, `TCH`, `RUF`, `UP`, `SIM` with select ignores
- Third-party code in `third_party/` is excluded from linting

## Hardware Requirements

- Inference: 8GB+ VRAM GPU
- Full fine-tuning: 22.5–70GB VRAM (scales with model size and parallelism)
