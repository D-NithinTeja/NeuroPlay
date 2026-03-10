"""
Observation space (11 floats, all normalized to [-1, 1] or [0, 1]):
  [0]  agent_x          normalized: agent_x / (COLS-1)
  [1]  agent_y          normalized: agent_y / (ROWS-1)
  [2]  player_x         normalized
  [3]  player_y         normalized
  [4]  rel_dx           (player_x - agent_x) / COLS  — direction to player
  [5]  rel_dy           (player_y - agent_y) / ROWS
  [6]  manhattan_dist   normalized: dist / (COLS + ROWS)
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


def random_free_pos(rng: np.random.Generator, exclude=None):
    while True:
        x = int(rng.integers(0, COLS))
        y = int(rng.integers(0, ROWS))
        if not is_blocked(x, y):
            if exclude is None or (x, y) != exclude:
                return x, y


class SimpleChaser:

    def __init__(self, noise_prob=0.15):
        self.noise_prob = noise_prob

    def act(self, player_pos, agent_pos, rng: np.random.Generator):
        if rng.random() < self.noise_prob:
            return int(rng.integers(0, 4)) 

        px, py = player_pos
        ax, ay = agent_pos
        dx = ax - px
        dy = ay - py

        candidates = []
        if abs(dx) >= abs(dy):
            if dx > 0: candidates.append(3)
            elif dx < 0: candidates.append(2) 
            if dy > 0: candidates.append(1)
            elif dy < 0: candidates.append(0)
        else:
            if dy > 0: candidates.append(1)
            elif dy < 0: candidates.append(0)
            if dx > 0: candidates.append(3)
            elif dx < 0: candidates.append(2)

        for action in candidates:
            ddx, ddy = DIRS[action]
            nx, ny = px + ddx, py + ddy
            if not is_blocked(nx, ny):
                return action

        valid = [a for a in range(4) if not is_blocked(px + DIRS[a][0], py + DIRS[a][1])]
        if valid:
            return int(rng.choice(valid))
        return 0  


class TagEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, max_steps=300, render_mode=None, chaser_noise=0.15):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.chaser = SimpleChaser(noise_prob=chaser_noise)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng()
        self._agent_pos = (0, 0)
        self._player_pos = (0, 0)
        self._step_count = 0
        self._prev_dist = 0


    def _get_obs(self):
        ax, ay = self._agent_pos
        px, py = self._player_pos

        rel_dx = (px - ax) / COLS
        rel_dy = (py - ay) / ROWS
        dist   = (abs(px - ax) + abs(py - ay)) / (COLS + ROWS)

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

    def _manhattan(self):
        ax, ay = self._agent_pos
        px, py = self._player_pos
        return abs(px - ax) + abs(py - ay)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        while True:
            self._player_pos = random_free_pos(self._rng)
            self._agent_pos  = random_free_pos(self._rng, exclude=self._player_pos)
            if self._manhattan() >= 10:
                break

        self._step_count = 0
        self._prev_dist  = self._manhattan()

        return self._get_obs(), {}

    def step(self, action):
        ax, ay = self._agent_pos
        dx, dy = DIRS[int(action)]
        nx, ny = ax + dx, ay + dy

        hit_wall = False
        if not is_blocked(nx, ny):
            self._agent_pos = (nx, ny)
        else:
            hit_wall = True  
        p_action = self.chaser.act(self._player_pos, self._agent_pos, self._rng)
        pdx, pdy = DIRS[p_action]
        pnx = self._player_pos[0] + pdx
        pny = self._player_pos[1] + pdy
        if not is_blocked(pnx, pny):
            self._player_pos = (pnx, pny)

        self._step_count += 1
        dist = self._manhattan()

        reward = 0.0

        reward += 0.05

        delta = dist - self._prev_dist
        reward += delta * 0.3

        if dist > 10:
            reward += 0.2
        elif dist > 6:
            reward += 0.1

        if hit_wall:
            reward -= 0.1

        if dist <= 3:
            reward -= 0.4
        if dist <= 2:
            reward -= 0.8

        self._prev_dist = dist

        tagged = dist <= 1
        truncated = self._step_count >= self.max_steps

        if tagged:
            reward -= 20.0  

        if truncated and not tagged:
            reward += 5.0 

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
        print(f"\n--- Step {self._step_count} | dist={self._manhattan()} ---")
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