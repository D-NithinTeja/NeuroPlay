# 🏃 Game of Tags

A browser-based tag game where the AI opponent is a real RL agent — trained with **Stable Baselines3 / PPO**, exported via **ONNX**, and run in-browser via **onnxruntime-web (WASM)**.

---

## File Structure

```
game-of-tags/
├── frontend/
│   ├── index.html          ← game UI + event wiring
│   ├── game.js             ← game loop, rendering, tag logic
│   └── agent.js            ← ONNX Runtime Web inference
│
├── training/
│   ├── env.py              ← Custom Gymnasium environment
│   ├── train.py            ← PPO training (Stable Baselines3)
│   └── export_onnx.py      ← Export .zip → .onnx + norm_stats.json
│
└── model/
    ├── tag_agent.onnx      ← exported model (created by export step)
    └── norm_stats.json     ← VecNormalize stats (created by export step)
```

---

## Quick Start

### 1. Install Python dependencies

```bash
uv add stable-baselines3[extra] gymnasium torch onnx onnxruntime
```

### 2. Train the agent

```bash
cd training

# Full training run (~1.5M steps, ~10-20 min on CPU / ~3-5 min on GPU)
python train.py

# Quick smoke-test run (fewer steps, weaker agent)
python train.py --timesteps 300000

# Resume from checkpoint
python train.py --resume

# Train + evaluate afterwards
python train.py --eval
```

Training outputs:
```
training/models/best_model/best_model.zip   ← best checkpoint (use this)
training/models/tag_agent_final.zip         ← final weights
training/models/vecnormalize.pkl            ← normalizer stats
training/logs/                              ← TensorBoard logs
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir training/logs
```

### 3. Export to ONNX

```bash
cd training

# Export best model (recommended)
python export_onnx.py

# Export + verify outputs match PyTorch
python export_onnx.py --verify

# Custom paths
python export_onnx.py \
  --model  models/best_model/best_model.zip \
  --output ../model/tag_agent.onnx \
  --vecnorm models/vecnormalize.pkl
```

This creates:
```
model/tag_agent.onnx       ← neural net (input: [1,11] → output: [1,4] logits)
model/norm_stats.json      ← obs normalization params for the browser
```

### 4. Serve the frontend

The game uses ES modules, so it **must be served** (not opened as file://):

```bash
# Python
cd game-of-tags
python -m http.server 8080

# Node
npx serve .

# VS Code: use Live Server extension
```

Open: http://localhost:8080/frontend/

---

## How It Works

### Observation Space (11 floats)

| Index | Feature             | Range     |
|-------|---------------------|-----------|
| 0     | agent_x (norm)      | [0, 1]    |
| 1     | agent_y (norm)      | [0, 1]    |
| 2     | player_x (norm)     | [0, 1]    |
| 3     | player_y (norm)     | [0, 1]    |
| 4     | rel_dx              | [-1, 1]   |
| 5     | rel_dy              | [-1, 1]   |
| 6     | manhattan dist      | [0, 1]    |
| 7–10  | obstacle flags UDLR | {0, 1}    |

### Action Space

`Discrete(4)` → Up / Down / Left / Right

### Reward Shaping

| Event                  | Reward        |
|------------------------|---------------|
| Survival per step      | +0.05         |
| Distance increased     | +delta × 0.3  |
| Distance > 10 tiles    | +0.2          |
| Distance > 6 tiles     | +0.1          |
| Hit wall               | −0.1          |
| Distance ≤ 3           | −0.4          |
| Distance ≤ 2           | −0.8          |
| Tagged (terminal)      | −20.0         |
| Survived full episode  | +5.0          |

### ONNX Export

```
SB3 PPO .zip
  └─ policy.mlp_extractor  (pi branch, 128×128 Tanh)
  └─ policy.action_net     (128 → 4)
       ↓ torch.onnx.export (opset 17, dynamic batch)
model/tag_agent.onnx       (input: obs[B,11] → logits[B,4])
```

VecNormalize stats are extracted separately and applied in `agent.js` before inference.

### In-Browser Inference (agent.js)

```
buildObs(aiPos, playerPos)   →  Float32Array[11]
ObsNormalizer.normalize()    →  Float32Array[11]  (clip((x-mean)/std, ±10))
ort.InferenceSession.run()   →  Float32Array[4]   (logits)
argmax()                     →  action index 0–3
```

---

## Gameplay

- **You start as IT** (cyan, dashed ring) — chase and tag the AI
- On tag, **roles flip** — now the AI hunts you
- 90-second rounds, score tracked for both
- WASD or Arrow Keys to move

### When AI is Runner (default)
The trained ONNX model drives the AI — it actively evades you using learned policy.

### When AI is IT (chaser)
A greedy heuristic chases the player (moves to minimise Manhattan distance).
You can extend this by training a second "chaser" policy and exporting it separately.

---

## Extending

**Train a dedicated chaser policy:**
Set `is_IT=True` in `TagEnv` and invert the reward (reward closeness instead of distance).
Export as `tag_chaser.onnx` and load it in `agent.js` when `!state.itIsPlayer`.

**Harder AI:**
- Reduce `chaser_noise` in `train.py` toward 0 for a more relentless simulated chaser
- Increase `TOTAL_TIMESTEPS` to 3–5M for a stronger policy
- Tune `net_arch` to `[256, 256]` for more capacity

**Stochastic play:**
Toggle "Mode: STOCHASTIC" in the UI to sample from the softmax distribution
instead of taking argmax — makes the AI less predictable.