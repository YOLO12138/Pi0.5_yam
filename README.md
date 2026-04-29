# YAM OpenPI

This repository is developed on top of [OpenPI](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence, with additions for the **YAM bimanual robot** platform. The original codebase provides Vision-Language-Action (VLA) models (π₀, π₀-FAST, π₀.₅); this fork adds a YAM-specific data config, policy transforms, and a deployment bridge for the raiden robot loop.

---

## Training π₀.₅ on YAM

### 0. Installation

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Python 3.11+ is required.

### 1. Dataset format

Training uses the [LeRobot](https://github.com/huggingface/lerobot) dataset format. Your dataset must contain the following keys:

| LeRobot dataset key | Type | Description |
|---|---|---|
| `observation.images.head` | `uint8 [H, W, 3]` | Head (base) camera, RGB |
| `observation.images.left_wrist` | `uint8 [H, W, 3]` | Left wrist camera, RGB |
| `observation.images.right_wrist` | `uint8 [H, W, 3]` | Right wrist camera, RGB |
| `observation.state` | `float32 [14]` | Joint positions: `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]` |
| `action` | `float32 [T, 14]` | Absolute joint targets in the same 14D order |
| `task` (or `prompt`) | `str` | Language instruction per episode |

Images can be any resolution — they are resized to 224×224 during training. Actions are expected as **absolute joint positions**; the data pipeline converts joints to delta actions and keeps grippers absolute automatically (controlled by `use_delta_joint_actions=True` in `LeRobotYAMDataConfig`).

#### Converting raw raiden data to LeRobot format

[yam_dataset_builder/convert_yam_data_to_lerobot.py](yam_dataset_builder/convert_yam_data_to_lerobot.py) converts a raw raiden-exported dataset into the LeRobot format expected by this training pipeline.

The script assumes the following raw data layout on disk:

```
<raw_dir>/
  0000000001/
    rgb/
      head/          0000000000.jpg  0000000001.jpg  ...
      left_wrist/    0000000000.jpg  ...
      right_wrist/   0000000000.jpg  ...
    lowdim/
      head/          0000000000.npz  ...   # each file has a "joints" key → float32 [14]
    metadata.json                          # must have "num_frames" and "language.prompt"
  0000000002/
    ...
```

Episode directories must be named with digits. Each `.npz` lowdim file must expose a `joints` key containing a 14D array in OpenPI order: `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]`. Actions are derived as `action[t] = joints[t+1]`, so the last frame is dropped. The script encodes images to H.264 MP4 via ffmpeg and writes the dataset to `$HF_LEROBOT_HOME/<repo_id>`.

```bash
uv run yam_dataset_builder/convert_yam_data_to_lerobot.py \
    --args.raw-dir /path/to/raw_dataset \
    --args.repo-id your_hf_username/my_yam_dataset

# Push to HuggingFace Hub after conversion
uv run yam_dataset_builder/convert_yam_data_to_lerobot.py \
    --args.raw-dir /path/to/raw_dataset \
    --args.repo-id your_hf_username/my_yam_dataset \
    --args.push-to-hub True
```

> **Note:** This script was written for a specific raiden export layout. If your raw data has a different directory structure, joint ordering, or lowdim format, you will need to adjust the script accordingly — particularly `load_lowdim_bulk` (which reads the `.npz` files) and the `RepackTransform` mappings in `LeRobotYAMDataConfig`.

To host your dataset on HuggingFace Hub and reference it by `repo_id`, follow the [LeRobot dataset format guide](https://github.com/huggingface/lerobot).

### 2. Modifying the default configuration

The YAM training configs are defined in [src/openpi/training/config.py](src/openpi/training/config.py) (search for `pi05_yam`). Two configs are provided:

**`pi05_yam`** — full fine-tuning of π₀.₅:
```python
TrainConfig(
    name="pi05_yam",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotYAMDataConfig(
        repo_id="YOLO2431/place_lock_simple_v2",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    batch_size=64,
    num_workers=8,
)
```

**`pi05_yam_low_mem`** — LoRA fine-tuning (lower VRAM requirement):
```python
TrainConfig(
    name="pi05_yam_low_mem",
    model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
    ...
    freeze_filter=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora").get_freeze_filter(),
    ema_decay=None,
)
```

To train on your own dataset, change `repo_id` to your HuggingFace dataset. To add a new config variant, copy an existing `TrainConfig(name="pi05_yam", ...)` block, give it a new unique `name`, and add it to the `_CONFIGS` list.

Key fields in `TrainConfig` you may want to adjust:

| Field | Default | Description |
|---|---|---|
| `num_train_steps` | `30_000` | Total training steps |
| `batch_size` | `64` | Global batch size |
| `num_workers` | `8` | Data loader workers |
| `assets_base_dir` | `./assets` | Where norm stats are stored |
| `checkpoint_base_dir` | `./checkpoints` | Where checkpoints are saved |
| `wandb_enabled` | `True` | Toggle W&B logging |
| `data.default_prompt` | `None` | Inject a fixed prompt (overrides dataset task) |
| `data.use_delta_joint_actions` | `True` | Convert absolute joints to deltas |

### 3. Computing normalization statistics

Before training, compute normalization statistics for your dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_yam
```

Stats are saved under `./assets/pi05_yam/`. If your dataset changes, re-run this step.

### 4. Training

**Full fine-tuning** (requires ~45GB+ VRAM for `pi05_yam`):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam --exp-name my_run
```

**LoRA fine-tuning** (lower VRAM, `pi05_yam_low_mem`):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_yam_low_mem --exp-name my_run
```

Checkpoints are saved to `./checkpoints/pi05_yam/my_run/`.

### 5. Overriding config values from the CLI

All `TrainConfig` fields can be overridden at the command line using `tyro` syntax — append `--field value` after the config name:

```bash
# Change the dataset
uv run scripts/train.py pi05_yam --exp-name my_run \
  --data.repo_id your_hf_username/my_yam_dataset

# Override batch size and number of steps
uv run scripts/train.py pi05_yam --exp-name my_run \
  --batch-size 32 \
  --num-train-steps 50000

# Disable W&B logging
uv run scripts/train.py pi05_yam --exp-name my_run \
  --wandb-enabled False

# Resume from a previous run
uv run scripts/train.py pi05_yam --exp-name my_run --resume

# Override checkpoint and assets directories
uv run scripts/train.py pi05_yam --exp-name my_run \
  --assets-base-dir /path/to/assets \
  --checkpoint-base-dir /path/to/checkpoints
```

To see all available CLI flags for a config:
```bash
uv run scripts/train.py pi05_yam --help
```

---

## Deployment (YAM + raiden)

After training, serve the policy and connect the deployment bridge:

```bash
# Terminal 1: start the OpenPI policy server
uv run scripts/serve_policy.py \
  --policy.config pi05_yam \
  --policy.dir ./checkpoints/pi05_yam/my_run \
  --port 8000

# Terminal 2: run raiden inference
rd infer \
  --bridge deployment.openpi_bridge:OpenPiBridge \
  --ckpt_path unused \
  --action_hz 50.0 \
  --host localhost --port 8000 \
  --action_horizon 50 \
  --prompt "<my prompt>"
```

The bridge ([deployment/openpi_bridge.py](deployment/openpi_bridge.py)) handles the joint-order difference between raiden and OpenPI:
- **OpenPI 14D**: `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]`
- **Raiden 14D**: `[right_joint(6), right_grip(1), left_joint(6), left_grip(1)]`

---
