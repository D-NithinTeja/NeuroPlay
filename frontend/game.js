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
const CELL   = 20;
canvas.width  = COLS * CELL;
canvas.height = ROWS * CELL;

const PLAYER_MOVE_MS = 110;
const AI_MOVE_MS     = 170;
const ROUND_SECS     = 90;
const TAG_DISTANCE   = 1;   
const FORCE_BFS_CHASER = new URLSearchParams(window.location.search).get('chaser') === 'bfs';
const PARTIAL_VIEW_NEIGHBORHOOD_CELLS = 3; // Reveals a 7x7 local grid around the player.

const VIEW_MODE = {
  FULL: 'full',
  PARTIAL: 'partial',
};

let visibilityMode = new URLSearchParams(window.location.search).get('view') === VIEW_MODE.PARTIAL
  ? VIEW_MODE.PARTIAL
  : VIEW_MODE.FULL;

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

  if (ts - lastPlayerMove >= PLAYER_MOVE_MS) {
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
  bg:          '#02030a',
  gridLine:    '#080a14',
  obstacle:    '#0d1124',
  obstacleEdge:'#1a2144',
  player:      '#00f5c4',
  playerGlow:  'rgba(0,245,196,',
  ai:          '#ff2d6f',
  aiGlow:      'rgba(255,45,111,',
  itRing:      '#ffd600',
  trailPlayer: 'rgba(0,245,196,',
  trailAi:     'rgba(255,45,111,',
};

const _scanlines = document.createElement('canvas');
_scanlines.width  = COLS * CELL;
_scanlines.height = ROWS * CELL;
(function buildScanlines() {
  const c = _scanlines.getContext('2d');
  for (let y = 0; y < _scanlines.height; y += 3) {
    c.fillStyle = 'rgba(0,0,0,0.08)';
    c.fillRect(0, y, _scanlines.width, 1);
  }
})();

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

  // Hard fog-of-war: keep local neighborhood visible and fully black out all distant cells.
  ctx.save();
  for (let gy = 0; gy < ROWS; gy++) {
    for (let gx = 0; gx < COLS; gx++) {
      const chebyshev = Math.max(Math.abs(gx - player.x), Math.abs(gy - player.y));

      let alpha;
      if (chebyshev <= 1) {
        alpha = 0.0; // fully visible core around player
      } else if (chebyshev <= PARTIAL_VIEW_NEIGHBORHOOD_CELLS) {
        alpha = 0.03; // bright visibility up to 3 tiles
      } else {
        alpha = 1.0; // completely dark and not visible outside neighborhood
      }

      ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
      ctx.fillRect(gx * CELL, gy * CELL, CELL, CELL);
    }
  }

  // Small soft glow confined to local neighborhood for readability.
  ctx.globalCompositeOperation = 'lighter';
  const glowRadius = CELL * PARTIAL_VIEW_NEIGHBORHOOD_CELLS;
  const glow = ctx.createRadialGradient(px, py, 0, px, py, glowRadius);
  glow.addColorStop(0.0, 'rgba(0, 245, 196, 0.16)');
  glow.addColorStop(0.6, 'rgba(0, 245, 196, 0.05)');
  glow.addColorStop(1.0, 'rgba(0, 245, 196, 0.00)');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(px, py, glowRadius, 0, Math.PI * 2);
  ctx.fill();
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

export function setDeterministicMode(enabled) {
  deterministicMode = Boolean(enabled);
  emit('deterministicChange', { deterministic: deterministicMode });
  return deterministicMode;
}

export function getDeterministicMode() {
  return deterministicMode;
}

function drawBackground() {
  ctx.fillStyle = PALETTE.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = PALETTE.gridLine;
  for (let r = 0; r <= ROWS; r++)
    for (let c = 0; c <= COLS; c++) {
      ctx.fillRect(c * CELL - 0.5, r * CELL - 0.5, 1, 1);
    }
}

function drawObstacles() {
  for (const key of OBSTACLES) {
    const [cx, cy] = key.split(',').map(Number);
    const x = cx * CELL, y = cy * CELL;

    ctx.fillStyle = PALETTE.obstacle;
    ctx.fillRect(x, y, CELL, CELL);

    ctx.strokeStyle = PALETTE.obstacleEdge;
    ctx.lineWidth   = 1;
    ctx.strokeRect(x + 1.5, y + 1.5, CELL - 3, CELL - 3);

    ctx.fillStyle = PALETTE.obstacleEdge;
    for (const [ox, oy] of [[3,3],[CELL-5,3],[3,CELL-5],[CELL-5,CELL-5]]) {
      ctx.beginPath();
      ctx.arc(x + ox, y + oy, 1.2, 0, Math.PI * 2);
      ctx.fill();
    }
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

function drawCharacter(entity, color, glowFn, isIT, label) {
  const cx = entity.x * CELL + CELL / 2;
  const cy = entity.y * CELL + CELL / 2;
  const r  = CELL * 0.36;

  const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 3);
  grd.addColorStop(0,   glowFn + '0.30)');
  grd.addColorStop(0.5, glowFn + '0.10)');
  grd.addColorStop(1,   glowFn + '0.00)');
  ctx.fillStyle = grd;
  ctx.beginPath(); ctx.arc(cx, cy, r * 3, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();

  if (isIT) {
    ctx.strokeStyle = PALETTE.itRing;
    ctx.lineWidth   = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(cx, cy, r + 5, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = PALETTE.itRing;
    ctx.font = 'bold 8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('◆ IT', cx, entity.y * CELL - 2);
  }

  ctx.fillStyle = '#07080f';
  ctx.beginPath(); ctx.arc(cx - 3, cy - 2, 2.2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(cx + 3, cy - 2, 2.2, 0, Math.PI * 2); ctx.fill();

  if (!isIT) {
    ctx.fillStyle = color;
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
  agentRunner = await createAgent();
  emit('visibilityChange', { mode: visibilityMode });
  emit('deterministicChange', { deterministic: deterministicMode });
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