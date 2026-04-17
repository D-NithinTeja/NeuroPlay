import os
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

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


def get_device(force_cpu: bool = False) -> str:
    """
    Determine the best device for training.
    Returns 'cuda' if GPU is available, 'cpu' otherwise.
    
    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability
    
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    if force_cpu:
        return 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
        device_info = f"{device} ({torch.cuda.get_device_name(0)})"
    else:
        device = 'cpu'
        device_info = device
    
    return device

SCRIPT_DIR      = Path(__file__).resolve().parent
MODELS_DIR      = SCRIPT_DIR / "models"
LOGS_DIR        = SCRIPT_DIR / "logs"

ROLE_ARTIFACTS = {
    "runner": {
        "best_model_path": MODELS_DIR / "best_model",
        "final_model_path": MODELS_DIR / "tag_agent_final",
        "vecnorm_path": MODELS_DIR / "vecnormalize.pkl",
        "checkpoint_dir": MODELS_DIR / "checkpoints",
        "checkpoint_prefix": "tag_agent",
        "tb_log_name": "ppo_tag_runner",
    },
    "chaser": {
        "best_model_path": MODELS_DIR / "best_model_chaser",
        "final_model_path": MODELS_DIR / "tag_chaser_final",
        "vecnorm_path": MODELS_DIR / "vecnormalize_chaser.pkl",
        "checkpoint_dir": MODELS_DIR / "checkpoints_chaser",
        "checkpoint_prefix": "tag_chaser",
        "tb_log_name": "ppo_tag_chaser",
    },
}

os.makedirs(str(MODELS_DIR),     exist_ok=True)
os.makedirs(str(LOGS_DIR),       exist_ok=True)
for _role_artifacts in ROLE_ARTIFACTS.values():
    os.makedirs(str(_role_artifacts["checkpoint_dir"]), exist_ok=True)
    os.makedirs(str(_role_artifacts["best_model_path"]), exist_ok=True)


PPO_HYPERPARAMS = dict(
    learning_rate        = 2e-4,
    n_steps              = 1024,
    batch_size           = 512,
    n_epochs             = 10,
    gamma                = 0.99,
    gae_lambda           = 0.95,
    clip_range           = 0.2,
    clip_range_vf        = None,
    ent_coef             = 0.005,
    vf_coef              = 0.5,
    max_grad_norm        = 0.5,

    policy_kwargs = dict(
        net_arch          = dict(pi=[128, 128], vf=[128, 128]),
        activation_fn     = torch.nn.Tanh,
    ),

    verbose              = 1,
    tensorboard_log      = str(LOGS_DIR),
)

N_ENVS           = 8        
TOTAL_TIMESTEPS  = 5_000_000
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


def make_env(rank: int, seed: int = 0, chaser_noise: float = 0.15, role: str = "runner"):
    def _init():
        env = TagEnv(max_steps=300, chaser_noise=chaser_noise, role=role)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_training_envs(n_envs: int, seed: int = 42, role: str = "runner"):
    env_fns = [make_env(rank=i, seed=seed, chaser_noise=0.15, role=role) for i in range(n_envs)]
    try:
        vec_env = SubprocVecEnv(env_fns)
    except Exception:
        from stable_baselines3.common.vec_env import DummyVecEnv
        print("  [warn] SubprocVecEnv failed, using DummyVecEnv")
        vec_env = make_vec_env(
            lambda: TagEnv(max_steps=300, role=role), n_envs=n_envs, seed=seed
        )
    return vec_env


def make_eval_env(seed: int = 999, gamma: float = 0.995, role: str = "runner"):
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: Monitor(TagEnv(max_steps=300, chaser_noise=0.05, role=role))])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,   
        clip_obs=10.0,
        gamma=gamma,
    )
    return env


def get_role_artifacts(role: str):
    if role not in ROLE_ARTIFACTS:
        raise ValueError(f"Unsupported role: {role}")
    return ROLE_ARTIFACTS[role]


def train(total_timesteps: int, resume: bool = False, device: str = None, role: str = "runner"):
    if device is None:
        device = get_device()

    artifacts = get_role_artifacts(role)
    best_model_path = artifacts["best_model_path"]
    final_model_path = artifacts["final_model_path"]
    vecnorm_path = artifacts["vecnorm_path"]
    checkpoint_dir = artifacts["checkpoint_dir"]
    checkpoint_prefix = artifacts["checkpoint_prefix"]
    tb_log_name = artifacts["tb_log_name"]

    # Guard against a CPU-only PyTorch build when the user explicitly requests CUDA.
    # (This commonly happens when installing the default torch wheel on Windows.)
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n  [warn] --device cuda was requested, but torch.cuda.is_available() is False")
        print("         Your installed PyTorch build likely has no CUDA support.")
        print("         Falling back to CPU. To use GPU, install a CUDA-enabled PyTorch build.")
        device = 'cpu'
    
    print("\n" + "═"*60)
    print("  GAME OF TAGS — PPO TRAINING")
    print("═"*60)
    print(f"  Role       : {role}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Parallel envs: {N_ENVS}")
    print(f"  Device     : {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM avail : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*60 + "\n")

    vec_env  = make_training_envs(N_ENVS, role=role)
    eval_env = make_eval_env(gamma=PPO_HYPERPARAMS["gamma"], role=role)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=PPO_HYPERPARAMS["gamma"],
    )

    if resume and os.path.exists(str(final_model_path) + ".zip"):
        print(f"  Resuming from {final_model_path}.zip ...\n")
        model = PPO.load(
            str(final_model_path),
            env=vec_env,
            device=device,
            **{k: v for k, v in PPO_HYPERPARAMS.items()
               if k not in ("verbose", "tensorboard_log", "policy_kwargs")},
        )
    else:
        model = PPO(
            policy         = "MlpPolicy",
            env            = vec_env,
            device         = device,
            **PPO_HYPERPARAMS,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = str(best_model_path),
        log_path             = str(LOGS_DIR),
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = str(checkpoint_dir),
        name_prefix = checkpoint_prefix,
        verbose     = 0,
    )

    progress_callback = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        log_interval=10_000,
    )

    model.learn(
        total_timesteps   = total_timesteps,
        callback          = [eval_callback, checkpoint_callback, progress_callback],
        tb_log_name       = tb_log_name,
        reset_num_timesteps = not resume,
    )

    model.save(str(final_model_path))
    vec_env.save(str(vecnorm_path))
    print(f"\n  ✓ Final model saved  → {final_model_path}.zip")
    print(f"  ✓ VecNormalize stats → {vecnorm_path}")

    vec_env.close()
    eval_env.close()
    return model


def evaluate(model_path: str | None = None, n_episodes: int = 20, device: str = None, role: str = "runner"):
    if device is None:
        device = get_device()

    artifacts = get_role_artifacts(role)
    if model_path is None:
        model_path = str(artifacts["final_model_path"])
    vecnorm_path = artifacts["vecnorm_path"]
    
    print(f"\n  Evaluating {model_path} over {n_episodes} episodes...")
    print(f"  Device: {device}\n")

    from stable_baselines3.common.vec_env import DummyVecEnv

    # Always use VecNormalize — load saved stats if available, else create fresh
    vec_env = DummyVecEnv([lambda: Monitor(TagEnv(max_steps=300, chaser_noise=0.05, role=role))])
    if os.path.exists(str(vecnorm_path)):
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env, device=device)
    rewards, lengths, outcomes = [], [], []

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
        if role == "runner":
            outcomes.append(0 if tagged else 1)
            outcome_label = 'SURVIVED ✓' if outcomes[-1] else 'TAGGED   ✗'
        else:
            outcomes.append(1 if tagged else 0)
            outcome_label = 'TAGGED ✓' if outcomes[-1] else 'MISSED   ✗'

        print(f"    ep {ep+1:>3}: reward={ep_reward:+7.2f}  len={ep_len:>4}  {outcome_label}")

    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Mean reward  : {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean ep len  : {np.mean(lengths):.1f}")
    metric_name = "Survival rate" if role == "runner" else "Tag rate"
    print(f"  {metric_name}: {np.mean(outcomes)*100:.1f}%")
    print(f"  ─────────────────────────────────────────────\n")

    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / evaluate Tag agents (runner or chaser)")
    parser.add_argument("--timesteps", type=int,   default=TOTAL_TIMESTEPS,
                        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from saved final model")
    parser.add_argument("--eval",      action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate existing model")
    parser.add_argument("--model",     type=str, default=None,
                        help="Model path for --eval-only (default depends on --role)")
    parser.add_argument("--role",      type=str, choices=['runner', 'chaser'], default='runner',
                        help="Train/eval role: runner (evade) or chaser (tag)")
    parser.add_argument("--device",    type=str, choices=['cuda', 'cpu'], default=None,
                        help="Device for training (default: cuda if available, else cpu)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    args = parser.parse_args()

    device = 'cpu' if args.force_cpu else (args.device or get_device())
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n[warn] CUDA requested but not available in this PyTorch install.")
        print("       Use --device cpu, or install a CUDA-enabled PyTorch build.")
        device = 'cpu'

    if not args.eval_only:
        model = train(
            total_timesteps = args.timesteps,
            resume          = args.resume,
            device          = device,
            role            = args.role,
        )

    if args.eval or args.eval_only:
        evaluate(
            model_path = args.model,
            n_episodes = 20,
            device     = device,
            role       = args.role,
        )