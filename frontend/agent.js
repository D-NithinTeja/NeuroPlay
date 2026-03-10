const MODEL_PATH     = './model/tag_agent.onnx';
const NORM_STATS_PATH= './model/norm_stats.json';
const OBS_DIM        = 11;

export const COLS = 32;
export const ROWS = 24;

export const ACTION_DIRS = [
  [ 0, -1],  // 0: up
  [ 0,  1],  // 1: down
  [-1,  0],  // 2: left
  [ 1,  0],  // 3: right
];

export const ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT'];


const OBSTACLE_RECTS = [
  [4,4,3,3],  [10,2,2,4],  [18,3,3,2],  [26,2,2,3],
  [2,10,2,4], [8,8,4,2],   [16,9,3,3],  [24,8,2,5],
  [4,16,3,2], [12,15,2,4], [20,14,4,2], [28,15,2,4],
  [6,20,4,2], [14,19,3,3], [22,20,4,2],
  [10,12,2,2],[20,11,2,2],
];

const _obstacleSet = new Set();
for (const [cx, cy, w, h] of OBSTACLE_RECTS) {
  for (let r = cy; r < cy + h; r++)
    for (let c = cx; c < cx + w; c++)
      _obstacleSet.add(`${c},${r}`);
}

export function isBlocked(x, y) {
  return x < 0 || y < 0 || x >= COLS || y >= ROWS || _obstacleSet.has(`${x},${y}`);
}

export const OBSTACLES = _obstacleSet;


class ObsNormalizer {
  constructor() {
    this.mean    = null;
    this.std     = null;
    this.clipObs = 10.0;
    this.enabled = false;
  }

  load(stats) {
    if (!stats || !stats.obs_mean || !stats.obs_var) return;
    const eps    = 1e-8;
    this.mean    = new Float32Array(stats.obs_mean);
    this.std     = new Float32Array(stats.obs_var.map(v => Math.sqrt(v + eps)));
    this.clipObs = stats.clip_obs ?? 10.0;
    this.enabled = true;
    console.log('[AgentRunner] VecNormalize stats loaded — normalization active');
  }

  normalize(obs) {
    if (!this.enabled) return obs;
    const out = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) {
      let v = (obs[i] - this.mean[i]) / this.std[i];
      v = Math.max(-this.clipObs, Math.min(this.clipObs, v));
      out[i] = v;
    }
    return out;
  }
}


export class AgentRunner {
  constructor() {
    this.session    = null;
    this.normalizer = new ObsNormalizer();
    this.ready      = false;
    this._loadError = null;

    this.inferenceCount = 0;
    this.totalInferenceMs = 0;
  }

  async load() {
    try {
      console.log('[AgentRunner] Loading ONNX model...');

      ort.env.wasm.numThreads = navigator.hardwareConcurrency
        ? Math.min(navigator.hardwareConcurrency, 4)
        : 2;

      try {
        const res = await fetch(NORM_STATS_PATH);
        if (res.ok) {
          const stats = await res.json();
          this.normalizer.load(stats);
        } else {
          console.warn('[AgentRunner] norm_stats.json not found — raw obs will be used');
        }
      } catch (e) {
        console.warn('[AgentRunner] Could not load norm_stats.json:', e.message);
      }

      this.session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });

      const inputName  = this.session.inputNames[0];
      const outputName = this.session.outputNames[0];
      console.log(`[AgentRunner] Input : "${inputName}"`);
      console.log(`[AgentRunner] Output: "${outputName}"`);

      await this._warmup();

      this.ready = true;
      console.log('[AgentRunner] ✓ Ready');
    } catch (err) {
      this._loadError = err;
      console.error('[AgentRunner] Failed to load model:', err);
      throw err;
    }
  }

  async _warmup() {
    const dummyObs = new Float32Array(OBS_DIM).fill(0);
    await this._runInference(dummyObs);
    console.log('[AgentRunner] Warm-up pass complete');
  }

  async _runInference(rawObs) {
    const normalized = this.normalizer.normalize(rawObs);
    const tensor     = new ort.Tensor('float32', normalized, [1, OBS_DIM]);
    const feeds      = { [this.session.inputNames[0]]: tensor };
    const results    = await this.session.run(feeds);
    const logits     = results[this.session.outputNames[0]].data; // Float32Array[4]
    return logits;
  }

  async act(obs, deterministic = true) {
    if (!this.ready) {
      console.warn('[AgentRunner] Not ready — returning random action');
      return Math.floor(Math.random() * 4);
    }

    const t0     = performance.now();
    const logits = await this._runInference(
      obs instanceof Float32Array ? obs : new Float32Array(obs)
    );
    const dt = performance.now() - t0;

    this.inferenceCount++;
    this.totalInferenceMs += dt;

    if (deterministic) {
      return argmax(logits);
    } else {
      return sampleSoftmax(logits);
    }
  }

  async actFromState(agentPos, playerPos, deterministic = true) {
    const obs = buildObs(agentPos, playerPos);
    return this.act(obs, deterministic);
  }

  get avgInferenceMs() {
    if (this.inferenceCount === 0) return 0;
    return this.totalInferenceMs / this.inferenceCount;
  }

  resetStats() {
    this.inferenceCount   = 0;
    this.totalInferenceMs = 0;
  }

  get isReady() { return this.ready; }
  get error()   { return this._loadError; }
}

export function buildObs(agentPos, playerPos) {
  const ax = agentPos.x, ay = agentPos.y;
  const px = playerPos.x, py = playerPos.y;

  const relDx   = (px - ax) / COLS;
  const relDy   = (py - ay) / ROWS;
  const dist    = (Math.abs(px - ax) + Math.abs(py - ay)) / (COLS + ROWS);

  const obsUp    = isBlocked(ax,     ay - 1) ? 1.0 : 0.0;
  const obsDown  = isBlocked(ax,     ay + 1) ? 1.0 : 0.0;
  const obsLeft  = isBlocked(ax - 1, ay    ) ? 1.0 : 0.0;
  const obsRight = isBlocked(ax + 1, ay    ) ? 1.0 : 0.0;

  return new Float32Array([
    ax / (COLS - 1),
    ay / (ROWS - 1),
    px / (COLS - 1),
    py / (ROWS - 1),
    relDx,
    relDy,
    dist,
    obsUp,
    obsDown,
    obsLeft,
    obsRight,
  ]);
}


function argmax(arr) {
  let best = -Infinity, idx = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > best) { best = arr[i]; idx = i; }
  }
  return idx;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function sampleSoftmax(logits) {
  const probs = softmax(Array.from(logits));
  const r = Math.random();
  let cumul = 0;
  for (let i = 0; i < probs.length; i++) {
    cumul += probs[i];
    if (r < cumul) return i;
  }
  return probs.length - 1;
}

export class GreedyFallbackAgent {
  constructor() {
    this.ready = true;
    console.warn('[GreedyFallbackAgent] ONNX model not available — using rule-based fallback');
  }

  async load() { return; }

  async actFromState(agentPos, playerPos) {
    const ax = agentPos.x, ay = agentPos.y;
    const px = playerPos.x, py = playerPos.y;

    const dx = ax - px;
    const dy = ay - py;

    const scores = ACTION_DIRS.map(([ddx, ddy], i) => {
      const nx = ax + ddx, ny = ay + ddy;
      if (isBlocked(nx, ny)) return -Infinity;
      const newDist = Math.abs(nx - px) + Math.abs(ny - py);
      return newDist + Math.random() * 0.5;
    });

    return argmax(scores);
  }

  get avgInferenceMs() { return 0; }
  get error() { return null; }
}

export async function createAgent() {
  const agent = new AgentRunner();
  try {
    await agent.load();
    return agent;
  } catch (err) {
    console.warn('[createAgent] Falling back to GreedyFallbackAgent:', err.message);
    return new GreedyFallbackAgent();
  }
}