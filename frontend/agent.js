const MODEL_PATH     = '../model/tag_agent.onnx';
const NORM_STATS_PATH= '../model/norm_stats.json';
const CHASER_MODEL_PATH = '../model/tag_chaser.onnx';
const CHASER_NORM_STATS_PATH = '../model/norm_stats_chaser.json';
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

// True shortest-path grid distance with obstacle awareness.
export function bfsDistance(x1, y1, x2, y2) {
  if (x1 === x2 && y1 === y2) return 0;
  const MAX_DIST = COLS + ROWS;

  const visited = new Uint8Array(COLS * ROWS);
  const queue   = new Int32Array(COLS * ROWS * 2);
  let head = 0, tail = 0;

  queue[tail++] = x1;
  queue[tail++] = y1;
  visited[y1 * COLS + x1] = 1;

  let dist = 0;
  let layerEnd = tail;

  while (head < tail) {
    if (head === layerEnd) {
      dist++;
      layerEnd = tail;
      if (dist > MAX_DIST) break;
    }

    const cx = queue[head++];
    const cy = queue[head++];

    for (const [dx, dy] of ACTION_DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (isBlocked(nx, ny)) continue;
      if (nx === x2 && ny === y2) return dist + 1;
      const idx = ny * COLS + nx;
      if (visited[idx]) continue;
      visited[idx] = 1;
      queue[tail++] = nx;
      queue[tail++] = ny;
    }
  }

  return MAX_DIST;
}

// Uncapped BFS distance.
// Returns Number.POSITIVE_INFINITY if the target is unreachable.
// Use this for reachability checks, spawning, and action scoring.
export function bfsDistanceExact(x1, y1, x2, y2) {
  if (x1 === x2 && y1 === y2) return 0;

  const visited = new Uint8Array(COLS * ROWS);
  const queue   = new Int32Array(COLS * ROWS * 2);
  let head = 0, tail = 0;

  queue[tail++] = x1;
  queue[tail++] = y1;
  visited[y1 * COLS + x1] = 1;

  let dist = 0;
  let layerEnd = tail;

  while (head < tail) {
    if (head === layerEnd) {
      dist++;
      layerEnd = tail;
    }

    const cx = queue[head++];
    const cy = queue[head++];

    for (const [dx, dy] of ACTION_DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (isBlocked(nx, ny)) continue;
      if (nx === x2 && ny === y2) return dist + 1;
      const idx = ny * COLS + nx;
      if (visited[idx]) continue;
      visited[idx] = 1;
      queue[tail++] = nx;
      queue[tail++] = ny;
    }
  }

  return Number.POSITIVE_INFINITY;
}

// Returns the first action index along a shortest path from source to target.
export function bfsNextStep(from, to) {
  const sx = from.x;
  const sy = from.y;
  const tx = to.x;
  const ty = to.y;
  if (sx === tx && sy === ty) return 0;

  const parent = new Int16Array(COLS * ROWS).fill(-1);
  const queue  = new Int32Array(COLS * ROWS * 2);
  let head = 0, tail = 0;

  queue[tail++] = sx;
  queue[tail++] = sy;
  parent[sy * COLS + sx] = sy * COLS + sx;

  let found = false;
  outer: while (head < tail) {
    const cx = queue[head++];
    const cy = queue[head++];

    for (const [dx, dy] of ACTION_DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (isBlocked(nx, ny)) continue;
      const idx = ny * COLS + nx;
      if (parent[idx] !== -1) continue;
      parent[idx] = cy * COLS + cx;
      if (nx === tx && ny === ty) {
        found = true;
        break outer;
      }
      queue[tail++] = nx;
      queue[tail++] = ny;
    }
  }

  if (!found) {
    // Fallback to shortest immediate BFS distance among valid neighbors.
    let bestAction = 0;
    let bestDist = Infinity;
    for (let a = 0; a < ACTION_DIRS.length; a++) {
      const [dx, dy] = ACTION_DIRS[a];
      const nx = sx + dx;
      const ny = sy + dy;
      if (isBlocked(nx, ny)) continue;
      const d = bfsDistanceExact(nx, ny, tx, ty);
      if (d < bestDist) {
        bestDist = d;
        bestAction = a;
      }
    }
    return bestAction;
  }

  let cx = tx;
  let cy = ty;
  while (true) {
    const pidx = parent[cy * COLS + cx];
    const px = pidx % COLS;
    const py = Math.floor(pidx / COLS);
    if (px === sx && py === sy) {
      for (let a = 0; a < ACTION_DIRS.length; a++) {
        const [dx, dy] = ACTION_DIRS[a];
        if (sx + dx === cx && sy + dy === cy) return a;
      }
      return 0;
    }
    cx = px;
    cy = py;
  }
}


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
    this.chaserSession = null;
    this.chaserNormalizer = new ObsNormalizer();
    this.ready      = false;
    this.chaserReady = false;
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

      // Optional: load a dedicated chaser model if present.
      await this._loadChaserModelIfAvailable();

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

  async _warmupChaser() {
    if (!this.chaserSession) return;
    const dummyObs = new Float32Array(OBS_DIM).fill(0);
    await this._runInferenceOnSession(this.chaserSession, this.chaserNormalizer, dummyObs);
    console.log('[AgentRunner] Chaser warm-up pass complete');
  }

  async _loadChaserModelIfAvailable() {
    try {
      try {
        const res = await fetch(CHASER_NORM_STATS_PATH);
        if (res.ok) {
          const stats = await res.json();
          this.chaserNormalizer.load(stats);
        } else {
          console.warn('[AgentRunner] norm_stats_chaser.json not found — chaser raw obs will be used');
        }
      } catch (e) {
        console.warn('[AgentRunner] Could not load norm_stats_chaser.json:', e.message);
      }

      this.chaserSession = await ort.InferenceSession.create(CHASER_MODEL_PATH, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      await this._warmupChaser();
      this.chaserReady = true;
      console.log('[AgentRunner] ✓ Chaser model active');
    } catch (e) {
      console.warn('[AgentRunner] Chaser model unavailable, using heuristic fallback:', e.message);
      this.chaserSession = null;
      this.chaserReady = false;
    }
  }

  async _runInferenceOnSession(session, normalizer, rawObs) {
    const normalized = normalizer.normalize(rawObs);
    const tensor     = new ort.Tensor('float32', normalized, [1, OBS_DIM]);
    const feeds      = { [session.inputNames[0]]: tensor };
    const results    = await session.run(feeds);
    const logits     = results[session.outputNames[0]].data;
    return logits;
  }

  async _runInference(rawObs) {
    return this._runInferenceOnSession(this.session, this.normalizer, rawObs);
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

  async actChaserFromState(chaserPos, targetPos, deterministic = true) {
    if (!this.chaserReady || !this.chaserSession) {
      return bfsNextStep(chaserPos, targetPos);
    }

    const obs = buildObs(chaserPos, targetPos);
    const t0 = performance.now();
    const logits = await this._runInferenceOnSession(
      this.chaserSession,
      this.chaserNormalizer,
      obs instanceof Float32Array ? obs : new Float32Array(obs)
    );
    const dt = performance.now() - t0;

    this.inferenceCount++;
    this.totalInferenceMs += dt;

    return deterministic ? argmax(logits) : sampleSoftmax(logits);
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
  get isChaserReady() { return this.chaserReady; }
  get error()   { return this._loadError; }
}

export function buildObs(agentPos, playerPos) {
  const ax = agentPos.x, ay = agentPos.y;
  const px = playerPos.x, py = playerPos.y;

  const relDx   = (px - ax) / COLS;
  const relDy   = (py - ay) / ROWS;
  const dist    = bfsDistance(ax, ay, px, py) / (COLS + ROWS);

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
    this._agentPosHistory = [];
    console.warn('[GreedyFallbackAgent] ONNX model not available — using rule-based fallback');
  }

  async load() { return; }

  _pushPos(pos) {
    const last = this._agentPosHistory[this._agentPosHistory.length - 1];
    if (!last || last.x !== pos.x || last.y !== pos.y) {
      this._agentPosHistory.push({ x: pos.x, y: pos.y });
      if (this._agentPosHistory.length > 8) this._agentPosHistory.shift();
    }
  }

  _isOscillating2Cycle() {
    const h = this._agentPosHistory;
    if (h.length < 4) return false;
    const a = h[h.length - 4];
    const b = h[h.length - 3];
    const c = h[h.length - 2];
    const d = h[h.length - 1];
    return (a.x === c.x && a.y === c.y && b.x === d.x && b.y === d.y);
  }

  _scoreEvade(actionIdx, agentPos, playerPos, prevPos) {
    const [dx, dy] = ACTION_DIRS[actionIdx];
    const nx = agentPos.x + dx;
    const ny = agentPos.y + dy;
    if (isBlocked(nx, ny)) return -Infinity;

    const currentDist = bfsDistance(agentPos.x, agentPos.y, playerPos.x, playerPos.y);
    const nextDist = bfsDistance(nx, ny, playerPos.x, playerPos.y);
    const distanceScore = (nextDist - currentDist) * 5.0; // reward moving away

    let oscillationPenalty = 0.0;
    if (prevPos && nx === prevPos.x && ny === prevPos.y) {
      oscillationPenalty = -3.0;
      if (this._isOscillating2Cycle()) oscillationPenalty -= 2.0;
    }

    let adjacentObstacles = 0;
    for (const [tx, ty] of ACTION_DIRS) {
      if (isBlocked(nx + tx, ny + ty)) adjacentObstacles++;
    }
    const wallProximityPenalty = -adjacentObstacles * 0.3;

    let continuityBonus = 0.0;
    if (prevPos) {
      const prevDx = agentPos.x - prevPos.x;
      const prevDy = agentPos.y - prevPos.y;
      if (prevDx * dx + prevDy * dy > 0) continuityBonus = 0.5;
    }

    return distanceScore + oscillationPenalty + wallProximityPenalty + continuityBonus;
  }

  async actFromState(agentPos, playerPos) {
    this._pushPos(agentPos);

    const prevPos = this._agentPosHistory.length >= 2
      ? this._agentPosHistory[this._agentPosHistory.length - 2]
      : null;

    // Avoid immediate reversals when alternatives exist.
    const legal = [];
    for (let a = 0; a < ACTION_DIRS.length; a++) {
      const [dx, dy] = ACTION_DIRS[a];
      const nx = agentPos.x + dx;
      const ny = agentPos.y + dy;
      if (!isBlocked(nx, ny)) legal.push(a);
    }
    if (legal.length === 0) return 0;

    const nonReverse = prevPos
      ? legal.filter(a => {
          const [dx, dy] = ACTION_DIRS[a];
          return !(agentPos.x + dx === prevPos.x && agentPos.y + dy === prevPos.y);
        })
      : legal;
    const candidates = nonReverse.length > 0 ? nonReverse : legal;

    let bestA = candidates[0];
    let bestS = -Infinity;
    for (const a of candidates) {
      const s = this._scoreEvade(a, agentPos, playerPos, prevPos) + Math.random() * 0.05;
      if (s > bestS) { bestS = s; bestA = a; }
    }
    return bestA;
  }

  async actChaserFromState(chaserPos, targetPos) {
    return bfsNextStep(chaserPos, targetPos);
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