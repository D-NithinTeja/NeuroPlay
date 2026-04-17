# Neuroplay

Neuroplay is a browser-based tag game with an RL training/export pipeline.
Runner and chaser policies can be trained with PPO and exported to ONNX for browser inference.

## Project Layout

```text
Neuroplay/
├── frontend/
│   ├── index.html              # UI, controls, and game-page wiring
│   ├── game.js                 # Main loop, rendering, visibility mask, control state
│   ├── agent.js                # ONNX loading, normalization, inference, path helpers
│   ├── style.css               # Theme and responsive layout
│   └── assets/
│       ├── user.png            # Player sprite
│       └── AI.png              # AI sprite
├── training/
│   ├── env.py                  # Gymnasium env + reward shaping
│   ├── train.py                # PPO training/eval entrypoint
│   ├── export_onnx.py          # SB3 -> ONNX export + norm stats
│   └── playtest_loops.py       # Behavior diagnostics
├── model/
│   ├── tag_agent.onnx
│   ├── norm_stats.json
│   ├── tag_chaser.onnx
│   └── norm_stats_chaser.json
├── pyproject.toml
└── README.md
```

## Setup (uv)

Run from repository root:

```bash
uv sync
```

If you need extra dependencies:

```bash
uv add stable-baselines3[extra] gymnasium torch onnx onnxruntime tensorboard
```

## Run Frontend

Serve the repo root (ES modules require HTTP):

```bash
uv run python -m http.server 8080
```

Open:

- http://localhost:8080/frontend/

## Frontend Controls

The current page controls are:

- `Restart Quest`: resets the current match.
- `Mind`: toggles deterministic vs stochastic runner-model action selection.
- `Sight`: toggles `FULL MAP` vs `TORCHLIGHT` visibility.
- `Stone`: cycles texture presets (`CRYPT`, `VOLCANIC`, `SANDSTONE`).
- `Pace` `- / +`: slows down or speeds up player movement.

Keyboard controls:

- Movement: `WASD` or arrow keys.
- Visibility quick toggle: `V`.

## Frontend Runtime Notes (Important)

- Player default speed is `110 ms` per move.
- AI move interval is `170 ms` per move.
- Torchlight visibility is strict radial fog with a 4-tile pixel-space radius.
- AI evasion uses the runner ONNX policy (`model/tag_agent.onnx`).
- AI chaser behavior currently uses robust BFS chase logic in `frontend/game.js`.
- A chaser ONNX inference path exists in `frontend/agent.js`, but default match flow is BFS chaser at runtime.

## URL Query Parameters

Optional frontend URL flags:

- `view=partial` starts in torchlight mode.
- `texture=crypt|volcanic|sandstone` sets initial floor/wall texture preset.
- `chaser=bfs` is accepted by code path checks (currently behavior-equivalent to default runtime).

Example:

```text
http://localhost:8080/frontend/?view=partial&texture=volcanic
```

## Training

Training supports two roles:

- `runner`
- `chaser`

Train runner:

```bash
uv run python training/train.py --role runner --timesteps 300000 --device cuda
```

Train chaser:

```bash
uv run python training/train.py --role chaser --timesteps 300000 --device cuda
```

Useful flags:

```bash
uv run python training/train.py --role runner --resume
uv run python training/train.py --role chaser --eval
uv run python training/train.py --role runner --eval-only
```

## Export to ONNX

Runner export:

```bash
uv run python training/export_onnx.py --role runner --verify
```

Creates:

- `model/tag_agent.onnx`
- `model/norm_stats.json`

Chaser export:

```bash
uv run python training/export_onnx.py --role chaser --verify
```

Creates:

- `model/tag_chaser.onnx`
- `model/norm_stats_chaser.json`

Custom export example:

```bash
uv run python training/export_onnx.py \
  --role chaser \
  --model training/models/best_model_chaser/best_model.zip \
  --vecnorm training/models/vecnormalize_chaser.pkl \
  --output model/tag_chaser.onnx \
  --norm-output model/norm_stats_chaser.json \
  --verify
```

## Evaluation and Diagnostics

Role eval:

```bash
uv run python training/train.py --role runner --eval-only
uv run python training/train.py --role chaser --eval-only
```

Loop/oscillation playtest:

```bash
uv run python training/playtest_loops.py --episodes 30
```

TensorBoard:

```bash
uv run tensorboard --logdir training/logs
```

## Updating This README

If gameplay or controls change, update this file in the same change set.
In particular, keep these aligned with code:

- `frontend/index.html` control labels and handlers.
- `frontend/game.js` runtime behavior and defaults.
- `training/env.py` reward logic.