import os
import argparse
import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL_PATH  = os.path.join("models", "best_model", "best_model.zip")
DEFAULT_VECNORM     = os.path.join("models", "vecnormalize.pkl")
DEFAULT_OUTPUT_DIR  = os.path.join("..", "model")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "tag_agent.onnx")
NORM_STATS_PATH     = os.path.join(DEFAULT_OUTPUT_DIR, "norm_stats.json")

OBS_DIM = 11


class PolicyWrapper(nn.Module):

    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor   
        self.action_net    = policy.action_net       

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features_pi, _ = self.mlp_extractor(obs)
        logits          = self.action_net(features_pi)
        return logits


def extract_norm_stats(vecnorm_path: str) -> dict | None:

    if not os.path.exists(vecnorm_path):
        print(f"  [warn] VecNormalize not found at {vecnorm_path} — skipping norm stats")
        return None

    try:
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        import sys, importlib
        sys.path.insert(0, os.path.dirname(__file__))
        from env import TagEnv

        dummy = DummyVecEnv([lambda: TagEnv()])
        venv  = VecNormalize.load(vecnorm_path, dummy)

        stats = {
            "obs_mean": venv.obs_rms.mean.tolist(),
            "obs_var":  venv.obs_rms.var.tolist(),
            "clip_obs": float(venv.clip_obs),
            "obs_dim":  OBS_DIM,
        }
        dummy.close()
        return stats
    except Exception as e:
        print(f"  [warn] Could not extract norm stats: {e}")
        return None


def export(model_path: str, output_path: str, vecnorm_path: str, verify: bool):
    print("\n" + "═"*60)
    print("  GAME OF TAGS — ONNX EXPORT")
    print("═"*60)
    print(f"  Source model : {model_path}")
    print(f"  Output path  : {output_path}")
    print(f"  VecNormalize : {vecnorm_path}")
    print("═"*60 + "\n")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "  → Run train.py first, or point --model to the correct .zip path."
        )

    from stable_baselines3 import PPO
    print("  Loading SB3 model...")
    model = PPO.load(model_path, device="cpu")
    policy = model.policy
    policy.eval()
    print(f"  ✓ Policy loaded  ({sum(p.numel() for p in policy.parameters()):,} params)")

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    dummy_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)

    with torch.no_grad():
        test_out = wrapper(dummy_obs)
    assert test_out.shape == (1, 4), f"Unexpected output shape: {test_out.shape}"
    print(f"  ✓ Wrapper test pass  output shape: {test_out.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("  Exporting to ONNX...")

    import inspect as _inspect
    _use_dynamo_kwarg = "dynamo" in _inspect.signature(torch.onnx.export).parameters
    _dynamo_kw = {"dynamo": False} if _use_dynamo_kwarg else {}

    torch.onnx.export(
        wrapper,
        dummy_obs,
        output_path,

        export_params        = True,
        opset_version        = 12,         
        do_constant_folding  = True,

        input_names          = ["obs"],
        output_names         = ["action_logits"],

        dynamic_axes = {
            "obs":            {0: "batch_size"},
            "action_logits":  {0: "batch_size"},
        },
        **_dynamo_kw,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✓ ONNX saved → {output_path}  ({size_kb:.1f} KB)")

    stats = extract_norm_stats(vecnorm_path)
    if stats:
        import json
        os.makedirs(os.path.dirname(NORM_STATS_PATH) or ".", exist_ok=True)
        with open(NORM_STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Norm stats saved → {NORM_STATS_PATH}")
    else:
        print("  [info] No norm stats saved — browser will use raw observations.")
        print("         This is fine if VecNormalize was not used during training.")

    if verify:
        _verify_onnx(output_path, policy, stats)

    print("\n  Done! Files ready for the frontend:\n")
    print(f"    model/tag_agent.onnx")
    if stats:
        print(f"    model/norm_stats.json")
    print()


def _verify_onnx(onnx_path: str, original_policy, norm_stats: dict | None):

    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping verification")
        return

    print("\n  Verifying ONNX output against PyTorch...")

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    n_tests  = 50
    max_diff = 0.0

    for _ in range(n_tests):
        raw_obs = np.random.randn(1, OBS_DIM).astype(np.float32)

        # PyTorch forward
        with torch.no_grad():
            wrapper = PolicyWrapper(original_policy)
            wrapper.eval()
            pt_out = wrapper(torch.tensor(raw_obs)).numpy()

        # ONNX forward
        ort_out = sess.run(
            ["action_logits"],
            {"obs": raw_obs},
        )[0]

        diff = float(np.max(np.abs(pt_out - ort_out)))
        max_diff = max(max_diff, diff)

    print(f"  ✓ Max abs difference (PyTorch vs ONNX): {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  ✓ Verification PASSED — outputs match within tolerance")
    else:
        print("  ⚠ Verification WARNING — outputs diverge slightly")
        print("    This may be acceptable; check opset compatibility.")

    print("\n  Action consistency check (10 obs):")
    for i in range(10):
        obs = np.random.randn(1, OBS_DIM).astype(np.float32)
        with torch.no_grad():
            wrapper_tmp = PolicyWrapper(original_policy)
            pt_logits   = wrapper_tmp(torch.tensor(obs)).numpy()[0]
        ort_logits  = sess.run(["action_logits"], {"obs": obs})[0][0]

        pt_action   = int(np.argmax(pt_logits))
        ort_action  = int(np.argmax(ort_logits))
        match       = "✓" if pt_action == ort_action else "✗"
        action_name = ["UP", "DOWN", "LEFT", "RIGHT"][pt_action]
        print(f"    [{match}] obs[{i}]: PT={pt_action}({action_name}) ONNX={ort_action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to ONNX")

    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to SB3 .zip model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Output .onnx path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=DEFAULT_VECNORM,
        help=f"Path to VecNormalize .pkl (default: {DEFAULT_VECNORM})"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run onnxruntime verification after export"
    )

    args = parser.parse_args()

    export(
        model_path  = args.model,
        output_path = args.output,
        vecnorm_path= args.vecnorm,
        verify      = args.verify,
    )