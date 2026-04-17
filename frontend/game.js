import {
  createAgent,
  isBlocked,
  OBSTACLES,
  ACTION_DIRS,
  COLS,
  ROWS,
  bfsDistance,
  bfsDistanceExact,
  bfsNextStep,
} from './agent.js';

const canvas = document.getElementById('game-canvas');
const ctx    = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;
const CELL   = 24;
canvas.width  = COLS * CELL;
canvas.height = ROWS * CELL;

const USER_SPRITE_PATH = './assets/user.png';
const AI_SPRITE_PATH   = './assets/AI.png';
const CHARACTER_SCALE  = 1.34;

let characterSprites = {
  player: null,
  ai: null,
};

const DEFAULT_PLAYER_MOVE_MS = 110;
const MIN_PLAYER_MOVE_MS = 60;
const MAX_PLAYER_MOVE_MS = 260;
const PLAYER_SPEED_STEP_MS = 10;
const AI_MOVE_MS     = 170;
const ROUND_SECS     = 90;
const TAG_DISTANCE   = 1;   
const FORCE_BFS_CHASER = new URLSearchParams(window.location.search).get('chaser') === 'bfs';
const PARTIAL_VIEW_NEIGHBORHOOD_CELLS = 4; // Radial brightness radius in partial view.

const VIEW_MODE = {
  FULL: 'full',
  PARTIAL: 'partial',
};

const TEXTURE_PRESET = {
  CRYPT: 'crypt',
  VOLCANIC: 'volcanic',
  SANDSTONE: 'sandstone',
};

const TEXTURE_PRESET_ORDER = [
  TEXTURE_PRESET.CRYPT,
  TEXTURE_PRESET.VOLCANIC,
  TEXTURE_PRESET.SANDSTONE,
];

let visibilityMode = new URLSearchParams(window.location.search).get('view') === VIEW_MODE.PARTIAL
  ? VIEW_MODE.PARTIAL
  : VIEW_MODE.FULL;

let texturePreset = (() => {
  const q = new URLSearchParams(window.location.search).get('texture');
  return TEXTURE_PRESET_ORDER.includes(q) ? q : TEXTURE_PRESET.CRYPT;
})();

let deterministicMode = true;

let state = {
  running:     false,
  playerScore: 0,
  aiScore:     0,
  timeLeft:    ROUND_SECS,
  itIsPlayer:  true,      // true = player is IT (chaser), false = AI is IT
  phase:       'idle',    // 'idle' | 'playing' | 'tagged' | 'gameover'
};

let player = { x: 2,        y: 2,        trail: [] };
let ai     = { x: COLS - 3, y: ROWS - 3, trail: [] };

const keys = {};
let lastPlayerMove = 0;
let lastAiMove     = 0;
let timerInterval  = null;
let animFrameId    = null;
let agentRunner    = null;
let aiActionPending = false;
let aiPosHistory = [];
let aiActionHistory = [];
let chaseRecoveryTicks = 0;
let chaseCommitAction = null;
let chaseCommitTicks = 0;
let lastChaseAction = null;
let chaseDirectionLock = null;
let chaseDirectionLockTicks = 0;
let playerMoveMs = DEFAULT_PLAYER_MOVE_MS;

export const Events = new EventTarget();

function emit(name, detail = {}) {
  Events.dispatchEvent(new CustomEvent(name, { detail }));
}

window.addEventListener('keydown', e => {
  if (e.key === 'v' || e.key === 'V') {
    toggleVisibilityMode();
    return;
  }
  keys[e.key] = true;
  if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key))
    e.preventDefault();
});
window.addEventListener('keyup', e => { keys[e.key] = false; });

export function pressKey(key)   { keys[key] = true;  }
export function releaseKey(key) { keys[key] = false; }

function getPlayerInput() {
  if (keys['ArrowUp']    || keys['w'] || keys['W']) return [0, -1];
  if (keys['ArrowDown']  || keys['s'] || keys['S']) return [0,  1];
  if (keys['ArrowLeft']  || keys['a'] || keys['A']) return [-1, 0];
  if (keys['ArrowRight'] || keys['d'] || keys['D']) return [ 1, 0];
  return null;
}

function randomFreeCell(excludes = []) {
  while (true) {
    const x = Math.floor(Math.random() * COLS);
    const y = Math.floor(Math.random() * ROWS);
    if (isBlocked(x, y)) continue;
    if (excludes.some(p => p.x === x && p.y === y)) continue;
    return { x, y };
  }
}

function spawnFarApart(minDist = 12) {
  let p, a;
  do {
    p = randomFreeCell();
    a = randomFreeCell([p]);
    // Ensure the two spawns are actually connected; otherwise the chaser can
    // look “stuck” near corners/obstacles trying to reach an unreachable target.
    const d = bfsDistanceExact(p.x, p.y, a.x, a.y);
    if (Number.isFinite(d) && d >= minDist) return { p, a };
  } while (true);
}

function respawn() {
  const { p, a } = spawnFarApart(12);
  player = { ...p, trail: [] };
  ai     = { ...a, trail: [] };
  aiPosHistory = [{ x: ai.x, y: ai.y }];
  aiActionHistory = [];
  chaseRecoveryTicks = 0;
  chaseCommitAction = null;
  chaseCommitTicks = 0;
  lastChaseAction = null;
  chaseDirectionLock = null;
  chaseDirectionLockTicks = 0;
}

const MAX_TRAIL = 10;

function pushTrail(entity) {
  entity.trail.push({ x: entity.x, y: entity.y });
  if (entity.trail.length > MAX_TRAIL) entity.trail.shift();
}

function tryMove(entity, dx, dy) {
  const nx = entity.x + dx;
  const ny = entity.y + dy;
  if (!isBlocked(nx, ny)) {
    pushTrail(entity);
    entity.x = nx;
    entity.y = ny;
    return true;
  }
  return false;
}

function checkTag(ts) {
  if (bfsDistance(player.x, player.y, ai.x, ai.y) <= TAG_DISTANCE) {
    onTag(ts);
  }
}

function recordAiPos() {
  aiPosHistory.push({ x: ai.x, y: ai.y });
  if (aiPosHistory.length > 8) aiPosHistory.shift();
}

function recordAiAction(actionIdx) {
  aiActionHistory.push(actionIdx);
  if (aiActionHistory.length > 10) aiActionHistory.shift();
}

function oppositeAction(actionIdx) {
  if (actionIdx === 0) return 1;
  if (actionIdx === 1) return 0;
  if (actionIdx === 2) return 3;
  if (actionIdx === 3) return 2;
  return actionIdx;
}

function isActionOscillating2Cycle(actionHistory) {
  // Detect strict A,B,A,B pattern in recent actions.
  if (!actionHistory || actionHistory.length < 4) return false;
  const a = actionHistory[actionHistory.length - 4];
  const b = actionHistory[actionHistory.length - 3];
  const c = actionHistory[actionHistory.length - 2];
  const d = actionHistory[actionHistory.length - 1];
  if (a !== c || b !== d || a === b) return false;

  // Require opposite direction pair (UP<->DOWN or LEFT<->RIGHT).
  return oppositeAction(a) === b;
}

function breakActionOscillation(baseAction, mode) {
  if (mode !== 'chase') return baseAction;
  if (!isActionOscillating2Cycle(aiActionHistory)) return baseAction;

  const legal = [];
  for (let a = 0; a < ACTION_DIRS.length; a++) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = ai.x + dx;
    const ny = ai.y + dy;
    if (!isBlocked(nx, ny)) legal.push(a);
  }
  if (legal.length === 0) return baseAction;

  const last = aiActionHistory[aiActionHistory.length - 1];
  const avoid = new Set([last, oppositeAction(last)]);

  const candidates = legal.filter(a => !avoid.has(a));
  const pool = candidates.length > 0 ? candidates : legal;

  const prevPos = aiPosHistory.length >= 2 ? aiPosHistory[aiPosHistory.length - 2] : null;
  let bestAction = pool[0];
  let bestScore = -Infinity;
  for (const a of pool) {
    const s = scoreAction(a, ai, player, prevPos, aiPosHistory, 'chase');
    if (s > bestScore) {
      bestScore = s;
      bestAction = a;
    }
  }

  return bestAction;
}

function getLegalActionsAt(pos) {
  const legal = [];
  for (let a = 0; a < ACTION_DIRS.length; a++) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = pos.x + dx;
    const ny = pos.y + dy;
    if (!isBlocked(nx, ny)) legal.push(a);
  }
  return legal;
}

function actionDistanceToTarget(fromPos, targetPos, actionIdx) {
  const [dx, dy] = ACTION_DIRS[actionIdx];
  const nx = fromPos.x + dx;
  const ny = fromPos.y + dy;
  if (isBlocked(nx, ny)) return Number.POSITIVE_INFINITY;
  return bfsDistanceExact(nx, ny, targetPos.x, targetPos.y);
}

function enforceBfsProgress(actionIdx, fromPos, targetPos) {
  const bfsAction = chaserAction(fromPos, targetPos);
  const chosenDist = actionDistanceToTarget(fromPos, targetPos, actionIdx);
  const bfsDist = actionDistanceToTarget(fromPos, targetPos, bfsAction);
  if (!Number.isFinite(chosenDist)) return bfsAction;
  // If model/filter chose an action that is strictly worse than shortest-path,
  // snap back to BFS to avoid local dithering.
  if (chosenDist > bfsDist) return bfsAction;
  return actionIdx;
}

function pickDetourCommitAction(fromPos, targetPos) {
  const legal = getLegalActionsAt(fromPos);
  if (legal.length === 0) return null;

  const last = aiActionHistory.length > 0 ? aiActionHistory[aiActionHistory.length - 1] : null;
  const avoid = new Set();
  if (last !== null) {
    avoid.add(last);
    avoid.add(oppositeAction(last));
  }

  const candidates = legal.filter(a => !avoid.has(a));
  const pool = candidates.length > 0 ? candidates : legal;

  let bestAction = pool[0];
  let bestScore = -Infinity;
  for (const a of pool) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = fromPos.x + dx;
    const ny = fromPos.y + dy;
    const openness = 4 - blockedNeighborCount(nx, ny);
    const progress = bfsDistanceExact(fromPos.x, fromPos.y, targetPos.x, targetPos.y)
      - bfsDistanceExact(nx, ny, targetPos.x, targetPos.y);
    const score = openness * 2.0 + progress * 1.0;
    if (score > bestScore) {
      bestScore = score;
      bestAction = a;
    }
  }

  return bestAction;
}

function isOscillating2Cycle(posHistory) {
  // Detect strict A,B,A,B pattern over the last 4 positions.
  if (!posHistory || posHistory.length < 4) return false;
  const a = posHistory[posHistory.length - 4];
  const b = posHistory[posHistory.length - 3];
  const c = posHistory[posHistory.length - 2];
  const d = posHistory[posHistory.length - 1];
  return (a.x === c.x && a.y === c.y && b.x === d.x && b.y === d.y);
}

function blockedNeighborCount(x, y) {
  let blocked = 0;
  for (const [dx, dy] of ACTION_DIRS) {
    if (isBlocked(x + dx, y + dy)) blocked++;
  }
  return blocked;
}

function isChaserCornerStuck(posHistory, currentPos, targetPos) {
  // Detect local trap behavior: little positional diversity near constrained cells
  // while not improving chase distance.
  if (!posHistory || posHistory.length < 6) return false;

  const recent = posHistory.slice(-6);
  const unique = new Set(recent.map(p => `${p.x},${p.y}`));
  const lowDiversity = unique.size <= 3;

  const cornerLike = blockedNeighborCount(currentPos.x, currentPos.y) >= 2;

  const nowDist = bfsDistanceExact(currentPos.x, currentPos.y, targetPos.x, targetPos.y);
  const start = recent[0];
  const oldDist = bfsDistanceExact(start.x, start.y, targetPos.x, targetPos.y);
  const noProgress = nowDist >= oldDist - 1;

  return lowDiversity && cornerLike && noProgress;
}

function isChaserStalling(posHistory, currentPos, targetPos) {
  // Broader stall detector: repeated local positions without distance improvement.
  if (!posHistory || posHistory.length < 8) return false;

  const recent = posHistory.slice(-8);
  const unique = new Set(recent.map(p => `${p.x},${p.y}`));
  const lowDiversity = unique.size <= 4;

  const dists = recent.map(p => bfsDistanceExact(p.x, p.y, targetPos.x, targetPos.y));
  const bestRecent = Math.min(...dists);
  const nowDist = bfsDistanceExact(currentPos.x, currentPos.y, targetPos.x, targetPos.y);
  const noProgress = nowDist >= bestRecent;

  return lowDiversity && noProgress;
}

function cornerEscapeAction(legalActions, currentPos, targetPos, prevPos) {
  // Escape heuristic: prefer moves into cells with more exits while still
  // keeping target distance reasonable.
  const nonReverse = prevPos
    ? legalActions.filter(a => {
        const [dx, dy] = ACTION_DIRS[a];
        return !(currentPos.x + dx === prevPos.x && currentPos.y + dy === prevPos.y);
      })
    : legalActions;

  const candidates = nonReverse.length > 0 ? nonReverse : legalActions;

  const currentDist = bfsDistanceExact(currentPos.x, currentPos.y, targetPos.x, targetPos.y);
  let bestAction = candidates[0];
  let bestScore = -Infinity;

  for (const a of candidates) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = currentPos.x + dx;
    const ny = currentPos.y + dy;

    const nextDist = bfsDistanceExact(nx, ny, targetPos.x, targetPos.y);
    const progress = currentDist - nextDist;
    const openness = 4 - blockedNeighborCount(nx, ny);

    // Strongly prefer escaping constrained cells, then preserve chase pressure.
    const score = openness * 2.0 + progress * 1.5 + Math.random() * 0.01;
    if (score > bestScore) {
      bestScore = score;
      bestAction = a;
    }
  }

  return bestAction;
}

function scoreAction(actionIdx, currentPos, targetPos, prevPos, posHistory, mode) {
  /**
   * Score-based action evaluation combining multiple factors:
   * - Distance to target (weighted heavily)
   * - Oscillation penalty (discourages reversing direction)
   * - Wall proximity penalty (prefer open areas)
   * - Movement continuity bonus (prefer smooth movement)
   */
  const [dx, dy] = ACTION_DIRS[actionIdx];
  const nx = currentPos.x + dx;
  const ny = currentPos.y + dy;
  
  // 1. Distance score
  const currentDist = bfsDistanceExact(currentPos.x, currentPos.y, targetPos.x, targetPos.y);
  const nextDist = bfsDistanceExact(nx, ny, targetPos.x, targetPos.y);
  // chase: reward getting closer; evade: reward getting farther
  const signedImprovement = (mode === 'evade') ? (nextDist - currentDist) : (currentDist - nextDist);
  const distanceScore = signedImprovement * 5.0;  // Weight distance heavily
  
  // 2. Oscillation penalty: discourage reversing to previous position
  let oscillationPenalty = 0.0;
  if (prevPos && nx === prevPos.x && ny === prevPos.y) {
    oscillationPenalty = -3.0;  // Strong penalty for immediate reversal
  }
 
  // Extra penalty if we're already stuck in a 2-cycle and would keep reversing.
  if (oscillationPenalty < 0 && isOscillating2Cycle(posHistory)) {
    oscillationPenalty -= 2.0;
  }
  
  // 4. Wall proximity score: prefer moves away from walls/obstacles
  let adjacentObstacles = 0;
  for (const [tx, ty] of ACTION_DIRS) {
    if (isBlocked(nx + tx, ny + ty)) {
      adjacentObstacles++;
    }
  }
  const wallProximityPenalty = -adjacentObstacles * 0.3;
  
  // 5. Movement continuity: prefer continuing in similar direction
  let continuityBonus = 0.0;
  if (prevPos) {
    const prevDx = currentPos.x - prevPos.x;
    const prevDy = currentPos.y - prevPos.y;
    // Bonus if moving in similar direction (dot product > 0)
    if (prevDx * dx + prevDy * dy > 0) {
      continuityBonus = 0.5;
    }
  }
  
  // Combine all factors
  const totalScore = distanceScore + oscillationPenalty + wallProximityPenalty + continuityBonus;
  return totalScore;
}

function antiCycleAction(baseAction, mode) {
  const legal = [];
  for (let a = 0; a < ACTION_DIRS.length; a++) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = ai.x + dx;
    const ny = ai.y + dy;
    if (!isBlocked(nx, ny)) legal.push(a);
  }
  if (legal.length === 0) return baseAction;

  // True previous position (one step ago), if available.
  const prevPos = aiPosHistory.length >= 2 ? aiPosHistory[aiPosHistory.length - 2] : null;
  const targetPos = player;

  // Chaser-specific stuck breaker for corner/obstacle pockets.
  if (mode === 'chase' && isChaserCornerStuck(aiPosHistory, ai, targetPos)) {
    return cornerEscapeAction(legal, ai, targetPos, prevPos);
  }

  // If the model/planner action is legal and doesn't immediately reverse, prefer it.
  if (legal.includes(baseAction)) {
    const [dx, dy] = ACTION_DIRS[baseAction];
    const bx = ai.x + dx;
    const by = ai.y + dy;
    const wouldReverse = prevPos && bx === prevPos.x && by === prevPos.y;
    if (!wouldReverse) return baseAction;
  }

  // Otherwise, choose the best scored action, avoiding immediate reversal when alternatives exist.
  const nonReverse = prevPos
    ? legal.filter(a => {
        const [dx, dy] = ACTION_DIRS[a];
        return !(ai.x + dx === prevPos.x && ai.y + dy === prevPos.y);
      })
    : legal;

  const candidates = nonReverse.length > 0 ? nonReverse : legal;

  let bestAction = candidates[0];
  let bestScore = -Infinity;

  for (const actionIdx of candidates) {
    let score = scoreAction(actionIdx, ai, targetPos, prevPos, aiPosHistory, mode);
    // Small tie-breaker toward the base action (when it's not an immediate reverse).
    if (actionIdx === baseAction) score += 0.25;
    if (score > bestScore) {
      bestScore = score;
      bestAction = actionIdx;
    }
  }

  return bestAction;
}

function onTag(ts) {
  const wasPlayerIT = state.itIsPlayer;

  if (wasPlayerIT) {
    state.playerScore++;
    emit('score', { player: state.playerScore, ai: state.aiScore });
    emit('tagged', { by: 'player' });
  } else {
    state.aiScore++;
    emit('score', { player: state.playerScore, ai: state.aiScore });
    emit('tagged', { by: 'ai' });
  }

  state.itIsPlayer = !state.itIsPlayer;
  emit('roleChange', { itIsPlayer: state.itIsPlayer });

  respawn();
}

async function loop(ts) {
  if (!state.running) return;

  if (ts - lastPlayerMove >= playerMoveMs) {
    const dir = getPlayerInput();
    if (dir) {
      tryMove(player, dir[0], dir[1]);
      lastPlayerMove = ts;
    }
  }

  if (ts - lastAiMove >= AI_MOVE_MS && !aiActionPending) {
    aiActionPending = true;
    lastAiMove = ts;

    try {
      let actionIdx;
      if (!state.itIsPlayer) {
        // Permanent pure BFS chaser mode for AI-as-IT.
        actionIdx = chaserAction(ai, player);
      } else {
        actionIdx = await agentRunner.actFromState(ai, player, deterministicMode);
        actionIdx = antiCycleAction(actionIdx, 'evade');
      }

      const [dx, dy] = ACTION_DIRS[actionIdx];
      tryMove(ai, dx, dy);
      recordAiAction(actionIdx);
      if (!state.itIsPlayer) lastChaseAction = actionIdx;
      recordAiPos();
    } finally {
      aiActionPending = false;
    }
  }

  checkTag(ts);
  draw(ts);

  animFrameId = requestAnimationFrame(loop);
}

function robustBfsChaserAction(aiPos, playerPos) {
  const legal = getLegalActionsAt(aiPos);
  if (legal.length === 0) return 0;

  const currentDist = bfsDistanceExact(aiPos.x, aiPos.y, playerPos.x, playerPos.y);
  const scored = legal.map(a => ({ a, d: actionDistanceToTarget(aiPos, playerPos, a) }));
  const bestDist = Math.min(...scored.map(s => s.d));

  // Keep a short lock on direction when recently oscillating.
  if (chaseDirectionLock !== null && chaseDirectionLockTicks > 0) {
    const locked = scored.find(s => s.a === chaseDirectionLock);
    if (locked && Number.isFinite(locked.d) && locked.d <= bestDist + 1) {
      chaseDirectionLockTicks--;
      return locked.a;
    }
    chaseDirectionLock = null;
    chaseDirectionLockTicks = 0;
  }

  // Hysteresis: keep previous chase action if still near-optimal.
  if (lastChaseAction !== null) {
    const prev = scored.find(s => s.a === lastChaseAction);
    if (prev && Number.isFinite(prev.d) && prev.d <= bestDist + 1) {
      return prev.a;
    }
  }

  let bestActions = scored.filter(s => s.d === bestDist).map(s => s.a);

  // Avoid immediate reverse unless there is no other best action.
  if (lastChaseAction !== null) {
    const reverse = oppositeAction(lastChaseAction);
    const nonReverse = bestActions.filter(a => a !== reverse);
    if (nonReverse.length > 0) bestActions = nonReverse;
  }

  // Prefer actions that lead to more open neighboring cells.
  let chosen = bestActions[0];
  let bestOpen = -Infinity;
  for (const a of bestActions) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = aiPos.x + dx;
    const ny = aiPos.y + dy;
    const openness = 4 - blockedNeighborCount(nx, ny);
    if (openness > bestOpen) {
      bestOpen = openness;
      chosen = a;
    }
  }

  if (isActionOscillating2Cycle(aiActionHistory)) {
    chaseDirectionLock = chosen;
    chaseDirectionLockTicks = 2;
  }

  // Hard safety: if chosen action is clearly regressive, defer to canonical BFS step.
  const chosenDist = actionDistanceToTarget(aiPos, playerPos, chosen);
  if (Number.isFinite(currentDist) && Number.isFinite(chosenDist) && chosenDist > currentDist + 1) {
    return bfsNextStep(aiPos, playerPos);
  }

  return chosen;
}

function chaserAction(aiPos, playerPos) {
  return robustBfsChaserAction(aiPos, playerPos);
}

const PALETTE = {
  void:        '#120d1c',
  floorA:      '#8f8473',
  floorB:      '#7d7364',
  mossA:       '#737858',
  mossB:       '#60684a',
  dustA:       '#8e775f',
  dustB:       '#7a634d',
  grout:       '#4e463d',
  moss:        '#5f7b43',
  wall:        '#5c4c3f',
  wallTop:     '#9b8a75',
  wallEdge:    '#3a3028',
  wallMortar:  'rgba(34, 28, 22, 0.55)',
  player:      '#54ff7b',
  playerGlow:  'rgba(84,255,123,',
  ai:          '#c86cff',
  aiGlow:      'rgba(200,108,255,',
  itRing:      '#ffe2a3',
  trailPlayer: 'rgba(84,255,123,',
  trailAi:     'rgba(200,108,255,',
};

const _scanlines = document.createElement('canvas');
_scanlines.width  = COLS * CELL;
_scanlines.height = ROWS * CELL;
(function buildScanlines() {
  const c = _scanlines.getContext('2d');
  for (let y = 0; y < _scanlines.height; y += 2) {
    c.fillStyle = 'rgba(24,18,13,0.035)';
    c.fillRect(0, y, _scanlines.width, 1);
  }
  for (let y = 0; y < _scanlines.height; y += 8) {
    for (let x = (y % 16) === 0 ? 2 : 6; x < _scanlines.width; x += 12) {
      c.fillStyle = 'rgba(255, 235, 190, 0.03)';
      c.fillRect(x, y, 1, 1);
    }
  }
})();

function tileNoise(x, y, seed = 0) {
  // Deterministic per-cell pseudo-noise used for tile variation.
  const n = (((x + 1) * 73856093) ^ ((y + 1) * 19349663) ^ (seed * 83492791)) >>> 0;
  return (n % 1024) / 1023;
}

function isNearWhite(r, g, b, a) {
  if (a <= 8) return false;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return max >= 220 && min >= 190 && (max - min) <= 35;
}

function makeWhiteBackgroundTransparent(canvas) {
  const w = canvas.width;
  const h = canvas.height;
  const cctx = canvas.getContext('2d', { willReadFrequently: true });
  const image = cctx.getImageData(0, 0, w, h);
  const data = image.data;

  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;

  function enqueueIfBg(x, y) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const idx = y * w + x;
    if (visited[idx]) return;
    const p = idx * 4;
    if (!isNearWhite(data[p], data[p + 1], data[p + 2], data[p + 3])) return;
    visited[idx] = 1;
    queue[tail++] = idx;
  }

  for (let x = 0; x < w; x++) {
    enqueueIfBg(x, 0);
    enqueueIfBg(x, h - 1);
  }
  for (let y = 1; y < h - 1; y++) {
    enqueueIfBg(0, y);
    enqueueIfBg(w - 1, y);
  }

  while (head < tail) {
    const idx = queue[head++];
    const x = idx % w;
    const y = Math.floor(idx / w);
    const p = idx * 4;

    // Remove only background-connected near-white pixels.
    data[p + 3] = 0;

    enqueueIfBg(x - 1, y);
    enqueueIfBg(x + 1, y);
    enqueueIfBg(x, y - 1);
    enqueueIfBg(x, y + 1);
  }

  cctx.putImageData(image, 0, 0);
}

function getOpaqueBounds(canvas) {
  const cctx = canvas.getContext('2d', { willReadFrequently: true });
  const pixels = cctx.getImageData(0, 0, canvas.width, canvas.height).data;

  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const a = pixels[(y * canvas.width + x) * 4 + 3];
      if (a <= 4) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < minX || maxY < minY) {
    return { sx: 0, sy: 0, sw: canvas.width, sh: canvas.height };
  }

  return {
    sx: minX,
    sy: minY,
    sw: maxX - minX + 1,
    sh: maxY - minY + 1,
  };
}

function loadSpriteAsset(path) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const source = document.createElement('canvas');
      source.width = img.width;
      source.height = img.height;
      const sctx = source.getContext('2d', { willReadFrequently: true });
      sctx.drawImage(img, 0, 0);

      makeWhiteBackgroundTransparent(source);
      const bounds = getOpaqueBounds(source);
      resolve({ source, ...bounds });
    };
    img.onerror = () => {
      console.warn(`[Sprites] Failed to load: ${path}`);
      resolve(null);
    };
    img.src = path;
  });
}

async function preloadCharacterSprites() {
  const [playerSprite, aiSprite] = await Promise.all([
    loadSpriteAsset(USER_SPRITE_PATH),
    loadSpriteAsset(AI_SPRITE_PATH),
  ]);

  characterSprites = {
    player: playerSprite,
    ai: aiSprite,
  };
}

function drawFloorTile(gx, gy) {
  const x = gx * CELL;
  const y = gy * CELL;

  const preset = texturePreset;
  const zone = tileNoise(Math.floor(gx / 4), Math.floor(gy / 4), 211);
  let baseA;
  let baseB;
  let dustA;
  let dustB;
  let mossA;
  let mossB;

  if (preset === TEXTURE_PRESET.VOLCANIC) {
    baseA = '#51484f';
    baseB = '#453c43';
    dustA = '#635452';
    dustB = '#554745';
    mossA = '#5b4f47';
    mossB = '#4f443f';
  } else if (preset === TEXTURE_PRESET.SANDSTONE) {
    baseA = '#a69274';
    baseB = '#937f64';
    dustA = '#b49a78';
    dustB = '#9e8667';
    mossA = '#8f8867';
    mossB = '#7a7458';
  } else {
    baseA = PALETTE.floorA;
    baseB = PALETTE.floorB;
    dustA = PALETTE.dustA;
    dustB = PALETTE.dustB;
    mossA = PALETTE.mossA;
    mossB = PALETTE.mossB;
  }

  if (zone < 0.32) {
    baseA = dustA;
    baseB = dustB;
  } else if (zone > 0.72) {
    baseA = mossA;
    baseB = mossB;
  }

  const n = tileNoise(gx, gy, 17);
  const base = n > 0.47 ? baseA : baseB;
  ctx.fillStyle = base;
  ctx.fillRect(x, y, CELL, CELL);

  // Cobble-like quarter shading pattern for a different floor texture profile.
  if (((gx + gy) & 1) === 0) {
    ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC
      ? 'rgba(255, 210, 180, 0.04)'
      : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(255, 243, 210, 0.09)' : 'rgba(255, 242, 212, 0.07)');
    ctx.fillRect(x, y, CELL / 2, CELL / 2);
    ctx.fillRect(x + CELL / 2, y + CELL / 2, CELL / 2, CELL / 2);
  } else {
    ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC ? 'rgba(0, 0, 0, 0.12)' : 'rgba(0, 0, 0, 0.08)';
    ctx.fillRect(x + CELL / 2, y, CELL / 2, CELL / 2);
    ctx.fillRect(x, y + CELL / 2, CELL / 2, CELL / 2);
  }

  // Soft bevel for depth.
  ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC
    ? 'rgba(255, 176, 118, 0.06)'
    : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(255, 228, 182, 0.10)' : 'rgba(255, 228, 182, 0.08)');
  ctx.fillRect(x, y, CELL, 2);
  ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC ? 'rgba(0, 0, 0, 0.16)' : 'rgba(0, 0, 0, 0.12)';
  ctx.fillRect(x, y + CELL - 2, CELL, 2);

  ctx.strokeStyle = preset === TEXTURE_PRESET.VOLCANIC ? '#3f353d'
    : (preset === TEXTURE_PRESET.SANDSTONE ? '#645746' : PALETTE.grout);
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);

  const crack = tileNoise(gx, gy, 31);
  const specks = tileNoise(gx, gy, 67);
  if (crack > 0.78) {
    ctx.strokeStyle = preset === TEXTURE_PRESET.VOLCANIC
      ? 'rgba(56, 35, 33, 0.55)'
      : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(82, 67, 47, 0.46)' : 'rgba(52, 43, 34, 0.52)');
    ctx.beginPath();
    ctx.moveTo(x + 4, y + CELL - 5);
    ctx.lineTo(x + 8, y + CELL - 10);
    ctx.lineTo(x + 12, y + CELL - 8);
    ctx.lineTo(x + CELL - 4, y + 4);
    ctx.stroke();
  }

  if (specks > 0.52) {
    if (preset === TEXTURE_PRESET.VOLCANIC) {
      ctx.fillStyle = 'rgba(214, 114, 78, 0.34)';
    } else if (preset === TEXTURE_PRESET.SANDSTONE) {
      ctx.fillStyle = 'rgba(151, 134, 92, 0.36)';
    } else {
      ctx.fillStyle = zone > 0.72 ? 'rgba(95, 123, 67, 0.62)' : 'rgba(40, 33, 27, 0.34)';
    }
    ctx.fillRect(x + 3, y + 6, 1, 1);
    ctx.fillRect(x + 8, y + 4, 1, 1);
    ctx.fillRect(x + 13, y + 11, 1, 1);
    ctx.fillRect(x + 6, y + 13, 1, 1);
  }
}

function drawWallTile(gx, gy) {
  const x = gx * CELL;
  const y = gy * CELL;
  const preset = texturePreset;
  const northBlocked = isBlocked(gx, gy - 1);
  const southBlocked = isBlocked(gx, gy + 1);
  const westBlocked = isBlocked(gx - 1, gy);
  const eastBlocked = isBlocked(gx + 1, gy);

  const wallBase = preset === TEXTURE_PRESET.VOLCANIC ? '#4e3d3e'
    : (preset === TEXTURE_PRESET.SANDSTONE ? '#876c51' : PALETTE.wall);
  const wallTop = preset === TEXTURE_PRESET.VOLCANIC ? '#7a5f5a'
    : (preset === TEXTURE_PRESET.SANDSTONE ? '#b8956d' : PALETTE.wallTop);
  const wallMortar = preset === TEXTURE_PRESET.VOLCANIC ? 'rgba(42, 26, 26, 0.55)'
    : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(72, 58, 41, 0.45)' : PALETTE.wallMortar);

  ctx.fillStyle = wallBase;
  ctx.fillRect(x, y, CELL, CELL);

  // Inner carved block gives a new obstacle texture profile.
  ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC
    ? 'rgba(64, 48, 47, 0.88)'
    : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(138, 110, 80, 0.84)' : 'rgba(74, 61, 51, 0.86)');
  ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);

  ctx.strokeStyle = preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(60, 44, 31, 0.48)' : 'rgba(26, 20, 16, 0.55)';
  ctx.strokeRect(x + 2.5, y + 2.5, CELL - 5, CELL - 5);

  if (!northBlocked) {
    ctx.fillStyle = wallTop;
    ctx.fillRect(x, y, CELL, 3);
    ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC
      ? 'rgba(255, 194, 162, 0.18)'
      : 'rgba(255, 232, 190, 0.22)';
    ctx.fillRect(x, y, CELL, 1.5);
  }

  if (!westBlocked) {
    ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC
      ? 'rgba(170, 128, 112, 0.26)'
      : 'rgba(177, 151, 125, 0.30)';
    ctx.fillRect(x, y, 1.5, CELL);
  }

  if (!eastBlocked) {
    ctx.fillStyle = preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(52, 40, 30, 0.34)' : 'rgba(31, 24, 20, 0.38)';
    ctx.fillRect(x + CELL - 1.5, y, 1.5, CELL);
  }

  if (!southBlocked) {
    ctx.fillStyle = preset === TEXTURE_PRESET.VOLCANIC ? 'rgba(22, 14, 14, 0.56)' : 'rgba(23, 18, 14, 0.52)';
    ctx.fillRect(x, y + CELL - 2, CELL, 2);
  }

  ctx.strokeStyle = wallMortar;
  ctx.lineWidth = 1;
  for (let ly = 5; ly < CELL; ly += 5) {
    ctx.beginPath();
    ctx.moveTo(x + 1, y + ly + 0.5);
    ctx.lineTo(x + CELL - 1, y + ly + 0.5);
    ctx.stroke();
  }

  const rowOffset = (gy % 2 === 0) ? 3 : 0;
  for (let lx = rowOffset; lx < CELL; lx += 6) {
    ctx.beginPath();
    ctx.moveTo(x + lx + 0.5, y + 1);
    ctx.lineTo(x + lx + 0.5, y + CELL - 1);
    ctx.stroke();
  }

  // Rune-like nicks so walls are not repetitive.
  if (tileNoise(gx, gy, 181) > 0.62) {
    ctx.strokeStyle = preset === TEXTURE_PRESET.VOLCANIC
      ? 'rgba(225, 128, 82, 0.24)'
      : (preset === TEXTURE_PRESET.SANDSTONE ? 'rgba(224, 198, 135, 0.20)' : 'rgba(200, 171, 130, 0.22)');
    ctx.beginPath();
    ctx.moveTo(x + 6, y + 7);
    ctx.lineTo(x + 9, y + 9);
    ctx.lineTo(x + 7, y + 12);
    ctx.stroke();
  }
}

function draw(ts) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawBackground();
  drawObstacles();
  drawTrail(player.trail, PALETTE.trailPlayer, PALETTE.player);
  drawTrail(ai.trail,     PALETTE.trailAi,     PALETTE.ai);
  drawCharacter(player, PALETTE.player,  PALETTE.playerGlow, state.itIsPlayer,  'YOU');
  drawCharacter(ai,     PALETTE.ai,      PALETTE.aiGlow,    !state.itIsPlayer,  'AI');
  drawVisibilityMask();
  drawScanlines();
}

function drawVisibilityMask() {
  if (visibilityMode !== VIEW_MODE.PARTIAL) return;

  const px = player.x * CELL + CELL / 2;
  const py = player.y * CELL + CELL / 2;
  const brightRadiusTiles = PARTIAL_VIEW_NEIGHBORHOOD_CELLS;
  const brightRadiusPx = brightRadiusTiles * CELL;

  // Strict pixel-space radial fog: bright center, smooth falloff, fully dark beyond 4 tiles.
  ctx.save();
  const fog = ctx.createRadialGradient(px, py, 0, px, py, brightRadiusPx);
  fog.addColorStop(0.00, 'rgba(0, 0, 0, 0.02)');
  fog.addColorStop(0.40, 'rgba(0, 0, 0, 0.10)');
  fog.addColorStop(0.70, 'rgba(0, 0, 0, 0.40)');
  fog.addColorStop(0.92, 'rgba(0, 0, 0, 0.78)');
  fog.addColorStop(1.00, 'rgba(0, 0, 0, 1.00)');
  ctx.fillStyle = fog;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function normalizeViewMode(mode) {
  return mode === VIEW_MODE.PARTIAL ? VIEW_MODE.PARTIAL : VIEW_MODE.FULL;
}

export function setVisibilityMode(mode) {
  const next = normalizeViewMode(mode);
  if (next === visibilityMode) return visibilityMode;
  visibilityMode = next;
  emit('visibilityChange', { mode: visibilityMode });
  return visibilityMode;
}

export function toggleVisibilityMode() {
  return setVisibilityMode(
    visibilityMode === VIEW_MODE.FULL ? VIEW_MODE.PARTIAL : VIEW_MODE.FULL
  );
}

export function getVisibilityMode() {
  return visibilityMode;
}

function normalizeTexturePreset(preset) {
  return TEXTURE_PRESET_ORDER.includes(preset) ? preset : TEXTURE_PRESET.CRYPT;
}

export function setTexturePreset(preset) {
  const next = normalizeTexturePreset(preset);
  if (next === texturePreset) return texturePreset;
  texturePreset = next;
  emit('texturePresetChange', { preset: texturePreset });
  return texturePreset;
}

export function toggleTexturePreset() {
  const idx = TEXTURE_PRESET_ORDER.indexOf(texturePreset);
  const next = TEXTURE_PRESET_ORDER[(idx + 1) % TEXTURE_PRESET_ORDER.length];
  return setTexturePreset(next);
}

export function getTexturePreset() {
  return texturePreset;
}

export function setDeterministicMode(enabled) {
  deterministicMode = Boolean(enabled);
  emit('deterministicChange', { deterministic: deterministicMode });
  return deterministicMode;
}

export function getDeterministicMode() {
  return deterministicMode;
}

function clampPlayerMoveMs(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return playerMoveMs;
  return Math.max(MIN_PLAYER_MOVE_MS, Math.min(MAX_PLAYER_MOVE_MS, Math.round(n)));
}

export function setPlayerMoveMs(value) {
  const next = clampPlayerMoveMs(value);
  if (next === playerMoveMs) return playerMoveMs;
  playerMoveMs = next;
  emit('playerSpeedChange', {
    moveMs: playerMoveMs,
    tilesPerSecond: 1000 / playerMoveMs,
  });
  return playerMoveMs;
}

export function adjustPlayerMoveMs(stepCount) {
  const delta = Number(stepCount) * PLAYER_SPEED_STEP_MS;
  if (!Number.isFinite(delta) || delta === 0) return playerMoveMs;
  return setPlayerMoveMs(playerMoveMs + delta);
}

export function getPlayerMoveMs() {
  return playerMoveMs;
}

function drawBackground() {
  ctx.fillStyle = PALETTE.void;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let gy = 0; gy < ROWS; gy++) {
    for (let gx = 0; gx < COLS; gx++) {
      if (isBlocked(gx, gy)) continue;
      drawFloorTile(gx, gy);
    }
  }
}

function drawObstacles() {
  for (const key of OBSTACLES) {
    const [cx, cy] = key.split(',').map(Number);
    drawWallTile(cx, cy);
  }
}

function drawTrail(trail, colorFn, solidColor) {
  for (let i = 0; i < trail.length; i++) {
    const t     = trail[i];
    const alpha = ((i + 1) / trail.length) * 0.22;
    const size  = CELL * 0.28 * ((i + 1) / trail.length);
    ctx.fillStyle = colorFn + alpha + ')';
    ctx.fillRect(
      t.x * CELL + CELL / 2 - size / 2,
      t.y * CELL + CELL / 2 - size / 2,
      size, size
    );
  }
}

function drawCharacterSprite(sprite, cx, cy) {
  if (!sprite || !sprite.source) return false;

  const destH = CELL * CHARACTER_SCALE;
  const aspect = sprite.sw / sprite.sh;
  const destW = destH * aspect;

  const x = cx - destW / 2;
  const y = cy - destH / 2 + CELL * 0.05;

  // Grounded shadow helps sprite readability on noisy floor tiles.
  ctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
  const shadowW = Math.max(10, destW * 0.56);
  ctx.fillRect(cx - shadowW / 2, cy + destH * 0.25, shadowW, 2.6);

  ctx.drawImage(
    sprite.source,
    sprite.sx,
    sprite.sy,
    sprite.sw,
    sprite.sh,
    x,
    y,
    destW,
    destH
  );

  return true;
}

function drawHeroSprite(cx, cy, baseColor) {
  const left = Math.round(cx - 6);
  const top = Math.round(cy - 7);
  ctx.fillStyle = 'rgba(20, 12, 8, 0.55)';
  ctx.fillRect(left + 1, top + 13, 10, 3);

  ctx.fillStyle = '#3e2d20';
  ctx.fillRect(left + 2, top + 10, 3, 3);
  ctx.fillRect(left + 7, top + 10, 3, 3);

  ctx.fillStyle = baseColor;
  ctx.fillRect(left + 1, top + 5, 10, 7);
  ctx.fillStyle = 'rgba(255,255,255,0.22)';
  ctx.fillRect(left + 2, top + 6, 2, 5);

  ctx.fillStyle = '#f4d9b2';
  ctx.fillRect(left + 4, top + 2, 4, 3);
  ctx.fillStyle = '#2d1d13';
  ctx.fillRect(left + 5, top + 3, 1, 1);
  ctx.fillRect(left + 7, top + 3, 1, 1);
  ctx.fillStyle = '#7f633f';
  ctx.fillRect(left + 3, top + 1, 6, 1);
}

function drawEnemySprite(cx, cy, baseColor) {
  const left = Math.round(cx - 6);
  const top = Math.round(cy - 7);
  ctx.fillStyle = 'rgba(20, 12, 8, 0.55)';
  ctx.fillRect(left + 1, top + 13, 10, 3);

  ctx.fillStyle = baseColor;
  ctx.fillRect(left + 1, top + 4, 10, 8);
  ctx.fillStyle = 'rgba(0,0,0,0.18)';
  ctx.fillRect(left + 2, top + 9, 8, 2);

  ctx.fillStyle = '#6f2f24';
  ctx.beginPath();
  ctx.moveTo(left + 3, top + 4);
  ctx.lineTo(left + 4, top + 1);
  ctx.lineTo(left + 6, top + 4);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(left + 9, top + 4);
  ctx.lineTo(left + 8, top + 1);
  ctx.lineTo(left + 6, top + 4);
  ctx.fill();

  ctx.fillStyle = '#ffe5a8';
  ctx.fillRect(left + 4, top + 6, 1, 1);
  ctx.fillRect(left + 7, top + 6, 1, 1);
}

function drawCharacter(entity, color, glowFn, isIT, label) {
  const cx = entity.x * CELL + CELL / 2;
  const cy = entity.y * CELL + CELL / 2;
  const r  = CELL * 0.46;

  const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
  grd.addColorStop(0,   glowFn + '0.30)');
  grd.addColorStop(0.5, glowFn + '0.10)');
  grd.addColorStop(1,   glowFn + '0.00)');
  ctx.fillStyle = grd;
  ctx.beginPath(); ctx.arc(cx, cy, r * 3, 0, Math.PI * 2); ctx.fill();

  const sprite = label === 'YOU' ? characterSprites.player : characterSprites.ai;
  const drewSprite = drawCharacterSprite(sprite, cx, cy);
  if (!drewSprite) {
    if (label === 'YOU') {
      drawHeroSprite(cx, cy, color);
    } else {
      drawEnemySprite(cx, cy, color);
    }
  }

  if (isIT) {
    const ty = entity.y * CELL - 3;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(cx - 8, ty - 8, 16, 10);

    ctx.fillStyle = PALETTE.itRing;
    ctx.font = 'bold 8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('IT', cx, ty);
  }

  if (!isIT) {
    ctx.fillStyle = '#f1e4c7';
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    ctx.globalAlpha = 0.7;
    ctx.fillText(label, cx, entity.y * CELL - 2);
    ctx.globalAlpha = 1;
  }
}

function drawScanlines() {
  ctx.globalAlpha = 1;
  ctx.drawImage(_scanlines, 0, 0);
}

function startTimer() {
  clearInterval(timerInterval);
  state.timeLeft = ROUND_SECS;
  emit('tick', { timeLeft: state.timeLeft });

  timerInterval = setInterval(() => {
    state.timeLeft--;
    emit('tick', { timeLeft: state.timeLeft });
    if (state.timeLeft <= 0) endGame();
  }, 1000);
}

export async function init() {
  emit('loading', { message: 'Loading AI model…' });
  const [loadedAgent] = await Promise.all([
    createAgent(),
    preloadCharacterSprites(),
  ]);
  agentRunner = loadedAgent;
  emit('visibilityChange', { mode: visibilityMode });
  emit('texturePresetChange', { preset: texturePreset });
  emit('deterministicChange', { deterministic: deterministicMode });
  emit('playerSpeedChange', {
    moveMs: playerMoveMs,
    tilesPerSecond: 1000 / playerMoveMs,
  });
  emit('loading', {
    message: agentRunner.constructor.name === 'AgentRunner'
      ? '✓ ONNX model loaded'
      : '⚠ Using fallback agent (train model first)',
    ready: true,
    usingOnnx: agentRunner.constructor.name === 'AgentRunner',
  });
}

export function startGame() {
  if (!agentRunner) { console.warn('Agent not loaded yet'); return; }

  cancelAnimationFrame(animFrameId);
  clearInterval(timerInterval);

  state = {
    running:     true,
    playerScore: 0,
    aiScore:     0,
    timeLeft:    ROUND_SECS,
    itIsPlayer:  true,
    phase:       'playing',
  };

  respawn();
  Object.keys(keys).forEach(k => delete keys[k]);
  aiActionPending = false;
  aiPosHistory = [{ x: ai.x, y: ai.y }];
  aiActionHistory = [];
  chaseRecoveryTicks = 0;
  chaseCommitAction = null;
  chaseCommitTicks = 0;
  lastChaseAction = null;
  chaseDirectionLock = null;
  chaseDirectionLockTicks = 0;

  emit('start');
  emit('score',      { player: 0, ai: 0 });
  emit('roleChange', { itIsPlayer: true });
  emit('visibilityChange', { mode: visibilityMode });
  startTimer();

  animFrameId = requestAnimationFrame(loop);
}

export function endGame() {
  state.running = false;
  state.phase   = 'gameover';
  cancelAnimationFrame(animFrameId);
  clearInterval(timerInterval);

  draw(performance.now());

  emit('gameover', {
    playerScore: state.playerScore,
    aiScore:     state.aiScore,
    winner: state.playerScore > state.aiScore ? 'player'
          : state.playerScore < state.aiScore ? 'ai'
          : 'draw',
    avgInferenceMs: agentRunner?.avgInferenceMs?.toFixed(2) ?? '—',
  });
}

export function getCanvas() { return canvas; }
export function getState()  { return state; }