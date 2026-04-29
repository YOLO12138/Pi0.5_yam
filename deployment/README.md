# YAM OpenPI Deployment

Bridge for deploying OpenPI Pi0.5 policies on the YAM bimanual robot via [raiden](https://github.com/your-org/raiden).

## Architecture

```
┌──────────────┐   websocket   ┌──────────────────┐
│  raiden loop  │◄────────────►│  OpenPI server    │
│  + bridge     │   (actions)  │  (serve_policy.py)│
└──────────────┘               └──────────────────┘
```

The OpenPI server loads the model and handles all inference. The bridge (`openpi_bridge.py`) is a thin adapter that:
1. Reformats raiden observations into OpenPI's expected input dict
2. Sends observations to the server, receives action chunks
3. Remaps 14D actions from OpenPI order to raiden motor command order

## Quick Start

### 1. Start the OpenPI policy server

```bash
cd ~/projects/yam_openpi
python scripts/serve_policy.py \
    --policy.config pi05_yam \
    --policy.dir /path/to/checkpoint \
    --port 8000
```

### 2. Run the bridge

```bash
cd ~/projects/YAM_robot
source .venv/bin/activate

# Via rd infer
PYTHONPATH=~/projects/yam_openpi:$PYTHONPATH \
rd infer \
    --bridge deployment.openpi_bridge:OpenPiBridge \
    --ckpt_path unused \
    --action_hz 50.0 \
    --host localhost --port 8000 \
    --action_horizon 10 \
    --prompt "pick up the lock and put it into the box"

# Or standalone
python -m deployment.openpi_bridge \
    --host localhost --port 8000 \
    --action_horizon 10 \
    --action_hz 50.0 \
    --prompt "pick up the lock and put it into the box"
```

## Dimension Mapping

| Format | Layout |
|--------|--------|
| OpenPI 14D | `[left_joint(6), left_grip(1), right_joint(6), right_grip(1)]` |
| Raiden 14D | `[right_joint(6), right_grip(1), left_joint(6), left_grip(1)]` |

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `localhost` | OpenPI server host |
| `--port` | `8000` | OpenPI server port |
| `--action_horizon` | `10` | Actions to execute per inference call |
| `--action_hz` | `50.0` | Robot control frequency |
| `--prompt` | `""` | Language instruction for the task |
