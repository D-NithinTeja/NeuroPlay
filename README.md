# Neuroplay

Neuroplay is a browser-based tag game where both runner and chaser behaviors can be learned with PPO, exported to ONNX, and executed in-browser with onnxruntime-web.

## Project Layout

```text
Neuroplay/
├── frontend/                     ← Browser game client (UI + gameplay loop)
│   ├── index.html               ← HUD, controls, and app wiring
│   ├── game.js                  ← Game state machine, movement, role switching, rendering
│   ├── agent.js                 ← ONNX runtime loader/inference and fallback logic
│   └── style.css                ← Visual theme and responsive layout styles
├── training/                     ← RL training and model export pipeline
│   ├── env.py                   ← Gymnasium environment, observations, and reward shaping
│   ├── train.py                 ← PPO training/evaluation entrypoint (runner/chaser roles)
│   ├── export_onnx.py           ← SB3 checkpoint to ONNX exporter (+ norm stats)
│   └── playtest_loops.py        ← Loop/oscillation diagnostics for trained behavior
├── model/                        ← Frontend-consumed ONNX artifacts
│   ├── tag_agent.onnx           ← Runner policy model used when AI is evader
│   ├── norm_stats.json          ← Runner observation normalization stats
│   ├── tag_chaser.onnx          ← Chaser policy model used when AI is IT/chaser
│   └── norm_stats_chaser.json   ← Chaser observation normalization stats
├── pyproject.toml                ← Project metadata, dependencies, and uv indexes/sources
└── README.md                     ← Usage, training/export guide, and design notes
```

## Setup (uv)

From repository root:

```bash
uv sync
```

If you need to add missing packages:

```bash
uv add stable-baselines3[extra] gymnasium torch onnx onnxruntime tensorboard
```

## Training

Training supports two roles:

- runner: policy learns to evade.
- chaser: policy learns to tag.

### Reward shaping

These tables mirror the current implementation in `training/env.py` and are intended to be easy to extend.

Runner role (`--role runner`)

| Condition | Reward Formula | Intent |
|---|---|---|
| Every step | `+0.05` | Encourage survival over time |
| Distance change | `+(dist_t - dist_{t-1}) * 0.3` | Reward increasing separation from chaser |
| Distance threshold | `+0.2 if dist > 10 else +0.1 if dist > 6` | Favor staying in safer spacing zones |
| Hit wall | `-0.1` | Discourage invalid/tight movement |
| Danger zone | `-0.4 if dist <= 3` | Penalize getting too close |
| Critical danger | `-0.8 if dist <= 2` | Strongly penalize near-capture states |
| Oscillation detected | `-0.15` | Reduce 2-cycle loop behavior |
| Tagged (terminal) | `-20.0` | Strong failure signal |
| Episode timeout without tag | `+5.0` | Reward full-episode survival |

Chaser role (`--role chaser`)

| Condition | Reward Formula | Intent |
|---|---|---|
| Every step | `-0.02` | Add urgency; discourage stalling |
| Distance change | `+(dist_{t-1} - dist_t) * 0.5` | Reward closing distance to target |
| In range | `+0.1 if dist <= 6` | Encourage sustained pressure |
| Close range | `+0.2 if dist <= 3` | Encourage finishing approach |
| Hit wall | `-0.1` | Discourage poor pathing |
| Oscillation detected | `-0.1` | Reduce looping in obstacle pockets |
| Tagged target (terminal) | `+20.0` | Strong success signal |
| Episode timeout without tag | `-5.0` | Penalize failure to finish |

Notes for contributors

- Keep formulas in this section synchronized with `TagEnv.step()` in `training/env.py`.
- Prefer adding new terms as new rows instead of overloading existing rows.
- If a term is role-specific, add it only to the relevant table.
- When changing reward weights, update both code and this table in the same PR/commit.

### Train runner policy

```bash
uv run python training/train.py --role runner --timesteps 300000 --device cuda
```

### Train chaser policy

```bash
uv run python training/train.py --role chaser --timesteps 300000 --device cuda
```

### Useful flags

```bash
uv run python training/train.py --role runner --resume
uv run python training/train.py --role chaser --eval
uv run python training/train.py --role runner --eval-only
```

### Role-specific outputs

Runner:

- training/models/best_model/best_model.zip
- training/models/tag_agent_final.zip
- training/models/vecnormalize.pkl

Chaser:

- training/models/best_model_chaser/best_model.zip
- training/models/tag_chaser_final.zip
- training/models/vecnormalize_chaser.pkl

## Export to ONNX

### Export runner ONNX

```bash
uv run python training/export_onnx.py --role runner --verify
```

Creates:

- model/tag_agent.onnx
- model/norm_stats.json

### Export chaser ONNX

```bash
uv run python training/export_onnx.py --role chaser --verify
```

Creates:

- model/tag_chaser.onnx
- model/norm_stats_chaser.json

### Custom export paths

```bash
uv run python training/export_onnx.py \
  --role chaser \
  --model training/models/best_model_chaser/best_model.zip \
  --vecnorm training/models/vecnormalize_chaser.pkl \
  --output model/tag_chaser.onnx \
  --norm-output model/norm_stats_chaser.json \
  --verify
```

## Run Frontend

The app uses ES modules, so serve over HTTP from repository root:

```bash
uv run python -m http.server 8080
```

Open:

- http://localhost:8080/frontend/

## Runtime Behavior

- When AI is runner, frontend uses model/tag_agent.onnx.
- When AI is chaser, frontend tries model/tag_chaser.onnx.
- If chaser ONNX is missing, frontend falls back to heuristic/BFS chase logic.

## Evaluation and Playtests

Evaluate a trained role policy:

```bash
uv run python training/train.py --role runner --eval-only
uv run python training/train.py --role chaser --eval-only
```

Loop and oscillation playtest:

```bash
uv run python training/playtest_loops.py --episodes 30
```

TensorBoard:

```bash
uv run tensorboard --logdir training/logs
```