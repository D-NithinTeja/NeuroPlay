"""
Observation space (11 floats, all normalized to [-1, 1] or [0, 1]):
  [0]  agent_x          normalized: agent_x / (COLS-1)
  [1]  agent_y          normalized: agent_y / (ROWS-1)
  [2]  player_x         normalized
  [3]  player_y         normalized
  [4]  rel_dx           (player_x - agent_x) / COLS  — direction to player
  [5]  rel_dy           (player_y - agent_y) / ROWS
  [6]  bfs_path_dist    normalized: dist / (COLS + ROWS)
  [7]  obs_up           1.0 if cell above is blocked, else 0.0
  [8]  obs_down         1.0 if cell below is blocked
  [9]  obs_left         1.0 if cell left  is blocked
  [10] obs_right        1.0 if cell right is blocked

Action space: Discrete(4)
  0 = up    (-y)
  1 = down  (+y)
  2 = left  (-x)
  3 = right (+x)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque


COLS = 32
ROWS = 24

OBSTACLE_RECTS = [
    (4,4,3,3),  (10,2,2,4),  (18,3,3,2),  (26,2,2,3),
    (2,10,2,4), (8,8,4,2),   (16,9,3,3),  (24,8,2,5),
    (4,16,3,2), (12,15,2,4), (20,14,4,2), (28,15,2,4),
    (6,20,4,2), (14,19,3,3), (22,20,4,2),
    (10,12,2,2),(20,11,2,2),
]

def _build_obstacle_set():
    obs = set()
    for (cx, cy, w, h) in OBSTACLE_RECTS:
        for r in range(cy, cy + h):
            for c in range(cx, cx + w):
                obs.add((c, r))
    return obs

OBSTACLES = _build_obstacle_set()

DIRS = {
    0: (0, -1),  # up
    1: (0,  1),  # down
    2: (-1, 0),  # left
    3: ( 1, 0),  # right
}


def is_blocked(x, y):
    return x < 0 or y < 0 or x >= COLS or y >= ROWS or (x, y) in OBSTACLES


def bfs_distance(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return 0

    # Use a large sentinel for unreachable so callers can detect it.
    # (Max reachable shortest-path distance is always < COLS*ROWS.)
    unreachable_dist = COLS * ROWS
    q = deque([(x1, y1, 0)])
    visited = {(x1, y1)}

    while q:
        cx, cy, d = q.popleft()
        for dx, dy in DIRS.values():
            nx, ny = cx + dx, cy + dy
            if is_blocked(nx, ny) or (nx, ny) in visited:
                continue
            if nx == x2 and ny == y2:
                return d + 1
            visited.add((nx, ny))
            q.append((nx, ny, d + 1))

    return unreachable_dist


def random_free_pos(rng: np.random.Generator, exclude=None):
    while True:
        x = int(rng.integers(0, COLS))
        y = int(rng.integers(0, ROWS))
        if not is_blocked(x, y):
            if exclude is None or (x, y) != exclude:
                return x, y


def score_action(action, current_pos, target_pos, prev_pos, pos_history, mode="chase"):
    """
    Score-based action evaluation combining multiple factors:
    - Distance to target (weighted heavily)
    - Oscillation penalty (discourages reversing direction)
    - Wall proximity penalty (prefer open areas)
    - Movement continuity bonus (prefer smooth movement)
    """
    cx, cy = current_pos
    tx, ty = target_pos
    dx, dy = DIRS[action]
    nx, ny = cx + dx, cy + dy
    
    # 1. Distance score
    current_dist = bfs_distance(cx, cy, tx, ty)
    next_dist = bfs_distance(nx, ny, tx, ty)
    signed_improvement = (next_dist - current_dist) if mode == "evade" else (current_dist - next_dist)
    distance_score = signed_improvement * 5.0  # Weight distance heavily
    
    # 2. Oscillation penalty: discourage reversing to previous position
    oscillation_penalty = 0.0
    if prev_pos and (nx, ny) == prev_pos:
        oscillation_penalty = -3.0  # Strong penalty for immediate reversal
    
    # 3. Two-cycle detection penalty: check if this would cause 2-position bouncing
    if len(pos_history) >= 4:
        h = pos_history
        # Check if we're about to enter a 2-cycle pattern
        if (nx, ny) == h[-3]:
            oscillation_penalty -= 2.0
    
    # 4. Wall proximity score: prefer moves away from walls/obstacles
    wall_proximity_penalty = 0.0
    adjacent_obstacles = 0
    for test_action in range(4):
        tx_adj, ty_adj = DIRS[test_action]
        if is_blocked(nx + tx_adj, ny + ty_adj):
            adjacent_obstacles += 1
    wall_proximity_penalty = -adjacent_obstacles * 0.3  # Slight penalty for constrained positions
    
    # 5. Movement continuity: prefer continuing in similar direction
    continuity_bonus = 0.0
    if prev_pos:
        prev_dx = cx - prev_pos[0]
        prev_dy = cy - prev_pos[1]
        # Bonus if moving in similar direction (dot product > 0)
        if prev_dx * dx + prev_dy * dy > 0:
            continuity_bonus = 0.5
    
    # Combine all factors
    total_score = distance_score + oscillation_penalty + wall_proximity_penalty + continuity_bonus
    return total_score


class SimpleChaser:

    def __init__(self, noise_prob=0.15):
        self.noise_prob = noise_prob
        self._history = []

    def reset(self, start_pos=None):
        self._history = []
        if start_pos is not None:
            self._history.append(start_pos)

    def _push_pos(self, pos):
        self._history.append(pos)
        if len(self._history) > 8:
            self._history.pop(0)

    def act(self, player_pos, agent_pos, rng: np.random.Generator):
        if not self._history or self._history[-1] != player_pos:
            self._push_pos(player_pos)

        px, py = player_pos
        ax, ay = agent_pos

        valid = [a for a in range(4) if not is_blocked(px + DIRS[a][0], py + DIRS[a][1])]
        if not valid:
            return 0

        if rng.random() < self.noise_prob:
            action = int(rng.choice(valid))
            ddx, ddy = DIRS[action]
            self._push_pos((px + ddx, py + ddy))
            return action

        prev = self._history[-2] if len(self._history) >= 2 else None

        # Use scoring-based selection instead of greedy distance minimization
        best_action = valid[0]
        best_score = float('-inf')
        
        for action in valid:
            score = score_action(action, player_pos, agent_pos, prev, self._history, mode="chase")
            if score > best_score:
                best_score = score
                best_action = action
        
        action = best_action
        ddx, ddy = DIRS[action]
        self._push_pos((px + ddx, py + ddy))
        return action


class SimpleEvader:

    def __init__(self, noise_prob=0.10):
        self.noise_prob = noise_prob
        self._history = []

    def reset(self, start_pos=None):
        self._history = []
        if start_pos is not None:
            self._history.append(start_pos)

    def _push_pos(self, pos):
        self._history.append(pos)
        if len(self._history) > 8:
            self._history.pop(0)

    def act(self, evader_pos, chaser_pos, rng: np.random.Generator):
        if not self._history or self._history[-1] != evader_pos:
            self._push_pos(evader_pos)

        ex, ey = evader_pos
        valid = [a for a in range(4) if not is_blocked(ex + DIRS[a][0], ey + DIRS[a][1])]
        if not valid:
            return 0

        if rng.random() < self.noise_prob:
            action = int(rng.choice(valid))
            ddx, ddy = DIRS[action]
            self._push_pos((ex + ddx, ey + ddy))
            return action

        prev = self._history[-2] if len(self._history) >= 2 else None

        best_action = valid[0]
        best_score = float("-inf")
        for action in valid:
            score = score_action(action, evader_pos, chaser_pos, prev, self._history, mode="evade")
            if score > best_score:
                best_score = score
                best_action = action

        action = best_action
        ddx, ddy = DIRS[action]
        self._push_pos((ex + ddx, ey + ddy))
        return action


class TagEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, max_steps=300, render_mode=None, chaser_noise=0.15, role="runner"):
        super().__init__()

        if role not in ("runner", "chaser"):
            raise ValueError(f"Invalid role '{role}'. Expected 'runner' or 'chaser'.")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.role = role
        self.chaser = SimpleChaser(noise_prob=chaser_noise)
        self.evader = SimpleEvader(noise_prob=0.10)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng()
        self._agent_pos = (0, 0)
        self._player_pos = (0, 0)
        self._step_count = 0
        self._prev_dist = 0
        self._agent_pos_history = deque(maxlen=8)  # Track longer history for loop/stall detection
        self._agent_action_history = deque(maxlen=8)
        self._oscillation_penalty = -0.15  # Penalty for oscillating between 2 positions

    def _blocked_neighbor_count(self, pos):
        x, y = pos
        blocked = 0
        for dx, dy in DIRS.values():
            if is_blocked(x + dx, y + dy):
                blocked += 1
        return blocked


    def _get_obs(self):
        ax, ay = self._agent_pos
        px, py = self._player_pos

        rel_dx = (px - ax) / COLS
        rel_dy = (py - ay) / ROWS
        d = bfs_distance(ax, ay, px, py)
        # Keep this feature within [0, 1] as documented.
        dist = min(d, COLS + ROWS) / (COLS + ROWS)

        obs_up    = 1.0 if is_blocked(ax,     ay - 1) else 0.0
        obs_down  = 1.0 if is_blocked(ax,     ay + 1) else 0.0
        obs_left  = 1.0 if is_blocked(ax - 1, ay    ) else 0.0
        obs_right = 1.0 if is_blocked(ax + 1, ay    ) else 0.0

        return np.array([
            ax / (COLS - 1),
            ay / (ROWS - 1),
            px / (COLS - 1),
            py / (ROWS - 1),
            rel_dx,
            rel_dy,
            dist,
            obs_up,
            obs_down,
            obs_left,
            obs_right,
        ], dtype=np.float32)

    def _distance(self):
        ax, ay = self._agent_pos
        px, py = self._player_pos
        return bfs_distance(ax, ay, px, py)

    def _is_oscillating(self):
        """
        Detect if agent is oscillating between 2 positions.
        Returns True if: position[i] == position[i-2] and position[i] != position[i-1]
        """
        if len(self._agent_pos_history) < 4:
            return False
        
        # Check if oscillating: current == 2-steps-ago AND current != 1-step-ago
        current = self._agent_pos_history[-1]
        one_step_ago = self._agent_pos_history[-2]
        two_steps_ago = self._agent_pos_history[-3]
        
        is_oscillating = (current == two_steps_ago and current != one_step_ago)
        return is_oscillating

    def _is_repeating_local_loop(self):
        # Detect low-diversity loops over recent positions (e.g., A-B-C-A-B-C).
        if len(self._agent_pos_history) < 6:
            return False
        recent = list(self._agent_pos_history)[-6:]
        unique = len(set(recent))
        return unique <= 3

    def _is_action_flipflop(self):
        # Detect ABAB action alternation pattern.
        if len(self._agent_action_history) < 4:
            return False
        a, b, c, d = list(self._agent_action_history)[-4:]
        if a != c or b != d or a == b:
            return False
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        return opposite.get(a) == b


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        unreachable_dist = COLS * ROWS

        while True:
            self._player_pos = random_free_pos(self._rng)
            self._agent_pos  = random_free_pos(self._rng, exclude=self._player_pos)
            d = self._distance()
            # Avoid disconnected spawns; they can make the chaser look "stuck".
            if d < unreachable_dist and d >= 10:
                break

        if self.role == "runner":
            self.chaser.reset(self._player_pos)
        else:
            self.evader.reset(self._player_pos)

        self._step_count = 0
        self._prev_dist  = self._distance()

        # Reset oscillation tracking each episode and seed with the start pos
        self._agent_pos_history.clear()
        self._agent_pos_history.append(self._agent_pos)
        self._agent_action_history.clear()

        return self._get_obs(), {}

    def step(self, action):
        ax, ay = self._agent_pos
        dx, dy = DIRS[int(action)]
        nx, ny = ax + dx, ay + dy

        hit_wall = False
        self._agent_action_history.append(int(action))
        if not is_blocked(nx, ny):
            self._agent_pos = (nx, ny)
            self._agent_pos_history.append(self._agent_pos)
        else:
            hit_wall = True  
        
        if self.role == "runner":
            p_action = self.chaser.act(self._player_pos, self._agent_pos, self._rng)
        else:
            # Opponent is now an evader while the policy controls the chaser.
            p_action = self.evader.act(self._player_pos, self._agent_pos, self._rng)
        pdx, pdy = DIRS[p_action]
        pnx = self._player_pos[0] + pdx
        pny = self._player_pos[1] + pdy
        if not is_blocked(pnx, pny):
            self._player_pos = (pnx, pny)

        self._step_count += 1
        dist = self._distance()
        # Keep reward terms numerically stable even if pathing ever becomes unreachable.
        dist_eff = min(float(dist), float(COLS + ROWS))
        prev_dist_eff = min(float(self._prev_dist), float(COLS + ROWS))
        delta = dist_eff - prev_dist_eff
        local_congestion = self._blocked_neighbor_count(self._agent_pos)
        is_2cycle = self._is_oscillating()
        is_looping = self._is_repeating_local_loop()
        is_flipflop = self._is_action_flipflop()

        tagged = dist <= 1
        truncated = self._step_count >= self.max_steps

        reward = 0.0
        if self.role == "runner":
            # Small survival incentive.
            reward += 0.03

            # Potential-based shaping: reward increasing distance from chaser.
            reward += delta * 0.35

            # Smooth comfort bonus once distance is reasonably safe.
            comfort = max(0.0, min(1.0, (dist_eff - 4.0) / 8.0))
            reward += 0.12 * comfort

            # Smooth danger penalties as chaser gets close.
            if dist_eff < 4.0:
                reward -= (4.0 - dist_eff) * 0.18
            if dist_eff < 2.0:
                reward -= (2.0 - dist_eff) * 0.35

            if hit_wall:
                reward -= 0.12

            # Loop penalties (stronger in congested cells).
            if is_2cycle:
                reward -= 0.12
            if is_looping:
                reward -= 0.12
            if is_flipflop:
                reward -= 0.18
            if local_congestion >= 2 and abs(delta) < 0.25 and (is_looping or is_flipflop):
                reward -= 0.10

            if tagged:
                reward -= 15.0
            if truncated and not tagged:
                reward += 3.0
        else:
            # Chaser objective: close distance quickly and tag.
            reward -= 0.01  # light time pressure

            reward += (-delta) * 0.45

            # Smooth proximity bonus to encourage finishing behavior.
            close_bonus = max(0.0, 6.0 - dist_eff)
            reward += close_bonus * 0.06

            if hit_wall:
                reward -= 0.08

            # Chaser gets stronger anti-dither penalties to prevent obstacle bounce loops.
            if is_2cycle:
                reward -= 0.12
            if is_looping:
                reward -= 0.18
            if is_flipflop:
                reward -= 0.30
            if local_congestion >= 2 and abs(delta) < 0.25 and (is_looping or is_flipflop):
                reward -= 0.18

            if tagged:
                reward += 15.0
            if truncated and not tagged:
                reward -= 3.0

        self._prev_dist = dist

        terminated = tagged

        if self.render_mode == "human":
            self._render_text()

        return self._get_obs(), reward, terminated, truncated, {}


    def _render_text(self):
        grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]
        for (cx, cy) in OBSTACLES:
            grid[cy][cx] = '#'
        px, py = self._player_pos
        ax, ay = self._agent_pos
        grid[py][px] = 'P'
        grid[ay][ax] = 'A'
        print(f"\n--- Step {self._step_count} | dist={self._distance()} ---")
        for row in grid:
            print(''.join(row))

    def render(self):
        if self.render_mode == "human":
            self._render_text()

    def close(self):
        pass


if __name__ == "__main__":
    import time

    print("=== TagEnv Smoke Test ===\n")
    env = TagEnv(max_steps=50, render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"Obs shape : {obs.shape}")
    print(f"Obs sample: {obs}")
    print(f"Action space: {env.action_space}")

    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"\n[TAGGED] at step {step+1} | total reward: {total_reward:.2f}")
            break
        if truncated:
            print(f"\n[SURVIVED] full episode | total reward: {total_reward:.2f}")
            break

    env.close()
    print("\n✓ Smoke test passed.")