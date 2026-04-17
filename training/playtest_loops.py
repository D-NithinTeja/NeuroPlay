import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import TagEnv, is_blocked


SCRIPT_DIR = Path(__file__).resolve().parent


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def is_near_obstacle(pos):
    x, y = pos
    return (
        is_blocked(x, y - 1)
        or is_blocked(x, y + 1)
        or is_blocked(x - 1, y)
        or is_blocked(x + 1, y)
    )


def two_cycle_counts(history):
    cycles = 0
    near_obstacle_cycles = 0
    for i in range(2, len(history)):
        a = history[i]
        b = history[i - 1]
        c = history[i - 2]
        if a == c and a != b:
            cycles += 1
            if is_near_obstacle(a):
                near_obstacle_cycles += 1
    return cycles, near_obstacle_cycles


def run_playtest(model_path, vecnorm_path, episodes, max_steps, chaser_noise):
    base_env = DummyVecEnv([
        lambda: Monitor(TagEnv(max_steps=max_steps, chaser_noise=chaser_noise))
    ])

    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, base_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = base_env

    model = PPO.load(model_path, env=vec_env)

    runner_cycles = 0
    runner_near_obstacle_cycles = 0
    chaser_cycles = 0
    chaser_near_obstacle_cycles = 0
    total_steps = 0
    survived = 0

    for _ in range(episodes):
        obs = vec_env.reset()
        done = False

        env_for_state = vec_env.venv.envs[0] if hasattr(vec_env, "venv") else vec_env.envs[0]
        raw_env = unwrap_env(env_for_state)

        agent_history = [raw_env._agent_pos]
        chaser_history = [raw_env._player_pos]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            done = bool(done_arr[0])

            agent_history.append(raw_env._agent_pos)
            chaser_history.append(raw_env._player_pos)
            total_steps += 1

        tagged = done and not info[0].get("TimeLimit.truncated", False)
        survived += 0 if tagged else 1

        c1, c1o = two_cycle_counts(agent_history)
        c2, c2o = two_cycle_counts(chaser_history)
        runner_cycles += c1
        runner_near_obstacle_cycles += c1o
        chaser_cycles += c2
        chaser_near_obstacle_cycles += c2o

    vec_env.close()

    step_denom = max(total_steps, 1)
    print("\n=== Loop/Oscillation Playtest ===")
    print(f"episodes={episodes} steps={total_steps} chaser_noise={chaser_noise}")
    print("\nRunner (policy-controlled AI):")
    print(f"  two-cycle count: {runner_cycles}")
    print(f"  two-cycle rate : {runner_cycles / step_denom:.4f} per step")
    print(f"  near-obstacle two-cycles: {runner_near_obstacle_cycles}")
    print(f"  near-obstacle cycle rate: {runner_near_obstacle_cycles / step_denom:.4f} per step")

    print("\nChaser (env heuristic pursuer):")
    print(f"  two-cycle count: {chaser_cycles}")
    print(f"  two-cycle rate : {chaser_cycles / step_denom:.4f} per step")
    print(f"  near-obstacle two-cycles: {chaser_near_obstacle_cycles}")
    print(f"  near-obstacle cycle rate: {chaser_near_obstacle_cycles / step_denom:.4f} per step")

    print("\nOutcomes:")
    print(f"  survival rate: {survived / max(episodes, 1) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playtest loop/limit-cycle behavior")
    parser.add_argument("--model", type=str, default=str(SCRIPT_DIR / "models" / "best_model" / "best_model"))
    parser.add_argument("--vecnorm", type=str, default=str(SCRIPT_DIR / "models" / "vecnormalize.pkl"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--chaser-noise", type=float, default=0.0)
    args = parser.parse_args()

    run_playtest(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        episodes=args.episodes,
        max_steps=args.max_steps,
        chaser_noise=args.chaser_noise,
    )
