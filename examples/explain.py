"""Explainability research script for trained policies.

Usage:
    # With a trained checkpoint:
    python examples/explain.py \
        --rollout logs/rollouts/can_video_0.npz \
        --index 5 \
        --task can_mh_image \
        --network sudeepdit \
        --model_path checkpoints/image_transformers/can_mh_image_flow_sudeepdit_256_seed0_success100.pt

    # Smoke test (random weights, no checkpoint needed):
    python examples/explain.py \
        --rollout logs/rollouts/can_video_0.npz \
        --index 0 \
        --task can_mh_image_patch \
        --network patchchitransformer
"""

import argparse
import os

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from mip.agent import TrainingAgent  # noqa: E402


def load_config(task, network, loss_type, model_path=None):
    GlobalHydra.instance().clear()
    overrides = [f"task={task}", f"network={network}", f"optimization.loss_type={loss_type}", "mode=eval",
                 "optimization.use_compile=false"]
    if model_path:
        overrides.append(f"optimization.model_path={model_path}")
    with initialize_config_dir(config_dir=os.path.abspath("examples/configs"), version_base=None):
        cfg = compose("main", overrides=overrides)
    obs_dim = cfg.network.emb_dim if cfg.task.obs_type == "image" else -1
    OmegaConf.update(cfg, "task.obs_dim", obs_dim)
    return cfg


def load_agent(config):
    agent = TrainingAgent(config)
    if config.optimization.model_path and config.optimization.model_path != "None":
        agent.load(config.optimization.model_path, load_optimizer=False)
    agent.eval()
    return agent


def visualize_obs(obs):
    """Save unnormalized observation to explanations/debug.png.

    obs: dict of {key: (obs_steps, ...) array} for image obs, or (obs_steps, obs_dim) for state.
    """
    os.makedirs("explanations", exist_ok=True)
    if isinstance(obs, dict):
        frames = []
        for k, v in obs.items():
            if "image" not in k:
                continue
            img = v[-1]  # last obs step
            if img.shape[0] in (1, 3):  # CHW -> HWC
                img = img.transpose(1, 2, 0)
            img_f = img.astype(np.float32)
            if img_f.max() > 1.0:
                img_f = img_f / 255.0
            frames.append((img_f * 255).clip(0, 255).astype(np.uint8))
        Image.fromarray(np.concatenate(frames, axis=1)).save("explanations/debug.png")
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(len(obs.flatten())), obs.flatten())
        fig.savefig("explanations/debug.png", bbox_inches="tight")
        plt.close(fig)


def visualize_attn(agent, obs_unnorm, alpha=0.5):
    """Save per-camera cross-attention heatmaps overlaid on the observation images.

    Reads cross-attention maps from PatchChiTransformer decoder layers (must be called
    after a forward pass). Averages over decoder layers, attention heads, and action steps
    to produce a spatial heatmap for each (camera, obs_step) pair.

    Saves:
        explanations/attn_{key}_step{s}.png  — for each image key and obs step
    """
    # Mirror the same model selection as agent.sample (use_ema=True by default)
    if agent.config.optimization.ema_rate < 1:
        flow_map = agent.flow_map_ema
        enc = agent.encoder_ema
    else:
        flow_map = agent.flow_map
        enc = agent.encoder
    network = flow_map.net

    if not hasattr(network, "get_cross_attn_maps"):
        print("Network does not support get_cross_attn_maps(); skipping attention visualization.")
        return

    maps = network.get_cross_attn_maps()  # list of (B, nhead, Ta, N_obs_tokens) or None
    maps = [m for m in maps if m is not None]
    if not maps:
        print("No cross-attention maps captured; skipping.")
        return

    # Average over decoder layers: (B, nhead, Ta, N_obs_tokens)
    attn = torch.stack(maps, dim=0).mean(0)
    # Average over heads and action steps: (B, N_obs_tokens)
    attn = attn.mean(dim=1).mean(dim=1)[0]  # (N_obs_tokens,)
    attn = attn.float().cpu().numpy()
    if not hasattr(enc, "n_patches"):
        print("Encoder is not a PatchEncoder; skipping attention visualization.")
        return

    n_h, n_w = enc.n_h, enc.n_w
    n_patches = enc.n_patches
    obs_steps = enc.obs_steps
    num_cameras = enc.num_cameras
    image_keys = enc.image_keys  # sorted list, same order as in forward

    os.makedirs("explanations", exist_ok=True)

    for step_idx in range(obs_steps):
        for cam_idx, key in enumerate(image_keys):
            token_start = (step_idx * num_cameras + cam_idx) * n_patches
            token_end = token_start + n_patches
            patch_attn = attn[token_start:token_end]  # (n_patches,)

            # Normalize
            patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)
            heatmap = patch_attn.reshape(n_h, n_w)  # (n_h, n_w)

            # Get the corresponding image (unnormalized, CHW uint8)
            if isinstance(obs_unnorm, dict):
                img = obs_unnorm[key][step_idx]  # (C, H, W) or (obs_steps, C, H, W)
            else:
                continue

            if img.shape[0] in (1, 3):  # CHW -> HWC
                img = img.transpose(1, 2, 0)
            H, W = img.shape[:2]
            img_f = img.astype(np.float32)
            if img_f.max() > 1.0:  # [0, 255] range
                img_f = img_f / 255.0
            img_f = np.clip(img_f, 0, 1)

            # Upsample heatmap to image size
            heatmap_up = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_up), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            overlay = (alpha * heatmap_color + (1 - alpha) * img_f)
            overlay = (overlay * 255).clip(0, 255).astype(np.uint8)

            fname = f"explanations/attn_{key}_step{step_idx}.png"
            Image.fromarray(overlay).save(fname)
            print(f"Saved {fname}  (patch grid: {n_h}x{n_w}, heatmap range: {patch_attn.min():.3f}–{patch_attn.max():.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True, help="Path to .npz rollout file")
    parser.add_argument("--index", type=int, required=True, help="Step index into rollout")
    parser.add_argument("--task", default="can_mh_image")
    parser.add_argument("--network", default="sudeepdit")
    parser.add_argument("--loss_type", default="flow")
    parser.add_argument("--model_path", default=None, help="Checkpoint path (omit for random-weight smoke test)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity (0=image only, 1=heatmap only)")
    args = parser.parse_args()

    data = np.load(args.rollout, allow_pickle=True)
    idx = args.index

    # Load all obs keys (image + low-dim) stored with obs_unnorm_/obs_norm_ prefix
    obs_keys = [k.replace("obs_unnorm_", "") for k in data.files if k.startswith("obs_unnorm_")]
    if obs_keys:
        obs_unnorm = {k: data[f"obs_unnorm_{k}"][idx, 0] for k in obs_keys}  # (obs_steps, ...)
        obs_norm   = {k: data[f"obs_norm_{k}"][idx, 0]   for k in obs_keys}
    else:
        obs_unnorm = data["obs_unnorm"][idx, 0]
        obs_norm   = data["obs_norm"][idx, 0]

    act_norm   = data["act_norm"][idx, 0]
    act_unnorm = data["act_unnorm"][idx, 0]

    config = load_config(args.task, args.network, args.loss_type, args.model_path)
    agent  = load_agent(config)

    # Run a forward pass to verify the architecture end-to-end
    device = config.optimization.device
    if isinstance(obs_norm, dict):
        obs_tensor = {k: torch.tensor(v, device=device, dtype=torch.float32).unsqueeze(0) for k, v in obs_norm.items()}
    else:
        obs_tensor = torch.tensor(obs_norm, device=device, dtype=torch.float32).unsqueeze(0)

    act_0 = torch.randn(1, config.task.horizon, config.task.act_dim, device=device)
    with torch.no_grad():
        act_out = agent.sample(act_0, obs_tensor, num_steps=1)

    visualize_obs(obs_unnorm)
    visualize_attn(agent, obs_unnorm, alpha=args.alpha)
    print(f"Output action shape: {act_out.shape}")
    print(f"Saved explanations/debug.png | act_unnorm: {act_unnorm.shape} | act_norm: {act_norm.shape}")
