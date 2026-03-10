import os
import argparse
import numpy as np
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import TagEnv

MODELS_DIR      = "models"
LOGS_DIR        = "logs"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model")
FINAL_MODEL_PATH= os.path.join(MODELS_DIR, "tag_agent_final")
VECNORM_PATH    = os.path.join(MODELS_DIR, "vecnormalize.pkl")
CHECKPOINT_DIR  = os.path.join(MODELS_DIR, "checkpoints")

os.makedirs(MODELS_DIR,     exist_ok=True)
os.makedirs(LOGS_DIR,       exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


PPO_HYPERPARAMS = dict(
    learning_rate        = 3e-4,
    n_steps              = 2048,       
    batch_size           = 256,
    n_epochs             = 10,
    gamma                = 0.995,     
    gae_lambda           = 0.95,
    clip_range           = 0.2,
    clip_range_vf        = None,
    ent_coef             = 0.01,     
    vf_coef              = 0.5,
    max_grad_norm        = 0.5,

    policy_kwargs = dict(
        net_arch          = dict(pi=[128, 128], vf=[128, 128]),
        activation_fn     = torch.nn.Tanh,
    ),

    verbose              = 1,
    tensorboard_log      = LOGS_DIR,
)

N_ENVS           = 8        
TOTAL_TIMESTEPS  = 1_500_000
EVAL_FREQ        = 20_000   
EVAL_EPISODES    = 20
CHECKPOINT_FREQ  = 100_000


class TrainingProgressCallback(BaseCallback):

    def __init__(self, total_timesteps, log_interval=10_000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval    = log_interval
        self._last_log_step  = 0
        self._ep_rewards     = []
        self._ep_lengths     = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])

        if self.num_timesteps - self._last_log_step >= self.log_interval:
            self._last_log_step = self.num_timesteps
            pct = self.num_timesteps / self.total_timesteps * 100

            if self._ep_rewards:
                mean_r = np.mean(self._ep_rewards[-100:])
                mean_l = np.mean(self._ep_lengths[-100:])
                print(
                    f"  [{pct:5.1f}%] steps={self.num_timesteps:>9,} | "
                    f"mean_reward={mean_r:+.2f} | mean_ep_len={mean_l:.0f}"
                )
            else:
                print(f"  [{pct:5.1f}%] steps={self.num_timesteps:>9,}")

        return True  


def make_env(rank: int, seed: int = 0, chaser_noise: float = 0.15):
    def _init():
        env = TagEnv(max_steps=300, chaser_noise=chaser_noise)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_training_envs(n_envs: int, seed: int = 42):
    env_fns = [make_env(rank=i, seed=seed, chaser_noise=0.15) for i in range(n_envs)]
    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception:
        from stable_baselines3.common.vec_env import DummyVecEnv
        print("  [warn] SubprocVecEnv failed, using DummyVecEnv")
        vec_env = make_vec_env(
            lambda: TagEnv(max_steps=300), n_envs=n_envs, seed=seed
        )
    return vec_env


def make_eval_env(seed: int = 999, gamma: float = 0.995):
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: Monitor(TagEnv(max_steps=300, chaser_noise=0.05))])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,   
        clip_obs=10.0,
        gamma=gamma,
    )
    return env


def train(total_timesteps: int, resume: bool = False):
    print("\n" + "═"*60)
    print("  GAME OF TAGS — PPO TRAINING")
    print("═"*60)
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Parallel envs: {N_ENVS}")
    print(f"  Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*60 + "\n")

    vec_env  = make_training_envs(N_ENVS)
    eval_env = make_eval_env(gamma=PPO_HYPERPARAMS["gamma"])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=PPO_HYPERPARAMS["gamma"],
    )

    if resume and os.path.exists(FINAL_MODEL_PATH + ".zip"):
        print(f"  Resuming from {FINAL_MODEL_PATH}.zip ...\n")
        model = PPO.load(
            FINAL_MODEL_PATH,
            env=vec_env,
            **{k: v for k, v in PPO_HYPERPARAMS.items()
               if k not in ("verbose", "tensorboard_log", "policy_kwargs")},
        )
    else:
        model = PPO(
            policy         = "MlpPolicy",
            env            = vec_env,
            **PPO_HYPERPARAMS,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOGS_DIR,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "tag_agent",
        verbose     = 0,
    )

    progress_callback = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        log_interval=10_000,
    )

    model.learn(
        total_timesteps   = total_timesteps,
        callback          = [eval_callback, checkpoint_callback, progress_callback],
        tb_log_name       = "ppo_tag",
        reset_num_timesteps = not resume,
    )

    model.save(FINAL_MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"\n  ✓ Final model saved  → {FINAL_MODEL_PATH}.zip")
    print(f"  ✓ VecNormalize stats → {VECNORM_PATH}")

    vec_env.close()
    eval_env.close()
    return model


def evaluate(model_path: str = FINAL_MODEL_PATH, n_episodes: int = 20):
    print(f"\n  Evaluating {model_path} over {n_episodes} episodes...\n")

    from stable_baselines3.common.vec_env import DummyVecEnv

    # Always use VecNormalize — load saved stats if available, else create fresh
    vec_env = DummyVecEnv([lambda: Monitor(TagEnv(max_steps=300, chaser_noise=0.05))])
    if os.path.exists(VECNORM_PATH):
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    rewards, lengths, survived = [], [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward, ep_len = 0.0, 0
        done = False
        tagged = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_len    += 1
            done   = bool(done_arr[0])
            tagged = done and not info[0].get("TimeLimit.truncated", False)

        rewards.append(ep_reward)
        lengths.append(ep_len)
        survived.append(0 if tagged else 1)
        print(f"    ep {ep+1:>3}: reward={ep_reward:+7.2f}  len={ep_len:>4}  "
              f"{'SURVIVED ✓' if survived[-1] else 'TAGGED   ✗'}")

    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Mean reward  : {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean ep len  : {np.mean(lengths):.1f}")
    print(f"  Survival rate: {np.mean(survived)*100:.1f}%")
    print(f"  ─────────────────────────────────────────────\n")

    if use_vec:
        vec_env.close()
    else:
        env.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate the Tag runner agent")
    parser.add_argument("--timesteps", type=int,   default=TOTAL_TIMESTEPS,
                        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from saved final model")
    parser.add_argument("--eval",      action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing model")
    parser.add_argument("--model",     type=str, default=FINAL_MODEL_PATH,
                        help="Model path for --eval-only")
    args = parser.parse_args()

    if not args.eval_only:
        model = train(
            total_timesteps = args.timesteps,
            resume          = args.resume,
        )

    if args.eval or args.eval_only:
        evaluate(
            model_path = args.model if args.eval_only else FINAL_MODEL_PATH,
            n_episodes = 20,
        )