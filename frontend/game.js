import {
  createAgent,
  buildObs,
  isBlocked,
  OBSTACLES,
  ACTION_DIRS,
  COLS,
  ROWS,
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

export const Events = new EventTarget();

function emit(name, detail = {}) {
  Events.dispatchEvent(new CustomEvent(name, { detail }));
}

window.addEventListener('keydown', e => {
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
  } while (Math.abs(p.x - a.x) + Math.abs(p.y - a.y) < minDist);
  return { p, a };
}

function respawn() {
  const { p, a } = spawnFarApart(12);
  player = { ...p, trail: [] };
  ai     = { ...a, trail: [] };
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

function manhattan() {
  return Math.abs(player.x - ai.x) + Math.abs(player.y - ai.y);
}

function checkTag(ts) {
  if (manhattan() <= TAG_DISTANCE) {
    onTag(ts);
  }
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

    let actionIdx;
    if (!state.itIsPlayer) {
      actionIdx = await agentRunner.actFromState(ai, player, true);
      actionIdx = await chaserAction(ai, player);
    } else {
      actionIdx = await agentRunner.actFromState(ai, player, true);
    }

    const [dx, dy] = ACTION_DIRS[actionIdx];
    tryMove(ai, dx, dy);
    aiActionPending = false;
  }

  checkTag(ts);
  draw(ts);

  animFrameId = requestAnimationFrame(loop);
}

async function chaserAction(aiPos, playerPos) {
  const ax = aiPos.x,  ay = aiPos.y;
  const px = playerPos.x, py = playerPos.y;

  let bestAction = 0, bestDist = Infinity;
  for (let a = 0; a < 4; a++) {
    const [dx, dy] = ACTION_DIRS[a];
    const nx = ax + dx, ny = ay + dy;
    if (isBlocked(nx, ny)) continue;
    const d = Math.abs(nx - px) + Math.abs(ny - py);
    if (d < bestDist) { bestDist = d; bestAction = a; }
  }
  return bestAction;
}

const PALETTE = {
  bg:          '#07080f',
  gridLine:    '#0d0e18',
  obstacle:    '#12152a',
  obstacleEdge:'#1c2040',
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
  drawScanlines();
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

  emit('start');
  emit('score',      { player: 0, ai: 0 });
  emit('roleChange', { itIsPlayer: true });
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