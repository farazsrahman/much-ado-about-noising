"""Single-episode data collection for robomimic.

Author: Faraz Rahman
Date: 2026-01-16
"""

import os
from collections import defaultdict

import h5py
import hydra
import loguru
import numpy as np
import torch

from mip.agent import TrainingAgent
from mip.config import Config
from mip.datasets.robomimic_dataset import make_dataset
from mip.envs.robomimic.robomimic_env import make_robomimic_env
from mip.logger import Logger
from mip.torch_utils import limit_threads, set_seed

os.environ["MUJOCO_GL"] = "egl"
torch.set_float32_matmul_precision("high")


def save_episodes(output_path, env_name, buffers):
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        f.attrs["env_name"] = env_name
        f.attrs["total_demos"] = len(buffers)

        for i, buffer in enumerate(buffers):
            demo_grp = data_grp.create_group(f"demo_{i}")

            def save_dict(grp, data_dict):
                for k, v in data_dict.items():
                    if "/" in k:
                        subgrp, subkey = k.split("/", 1)
                        if subgrp not in grp:
                            grp.create_group(subgrp)
                        save_dict(grp[subgrp], {subkey: v})
                    else:
                        try:
                            grp.create_dataset(k, data=np.array(v))
                        except Exception as e:
                            print(f"Error saving key {k}: {e}")
                            if isinstance(v, list) and len(v) > 0:
                                print(f"  List length: {len(v)}")
                                print(f"  First item shape: {np.array(v[0]).shape}")
                                # Try to find where it's inconsistent
                                for idx, item in enumerate(v):
                                    if np.array(item).shape != np.array(v[0]).shape:
                                        print(f"  Inconsistent item at index {idx}: shape {np.array(item).shape}")
                                        break
                            raise e

            save_dict(demo_grp, buffer)
            demo_grp.attrs["num_samples"] = len(buffer["actions"])


def collect_one_episode(config: Config, dataset, agent, idx=0):
    # If save_video is True, we save a video for every episode with a unique name
    render = config.task.save_video
    video_path = None
    if render:
        os.makedirs("results", exist_ok=True)
        video_path = f"results/video_{idx}.mp4"

    env_thunk = make_robomimic_env(
        config.task, idx=0, render=render, seed=config.optimization.seed + idx, video_path=video_path
    )
    env = env_thunk()
    obs, info = env.reset()

    buffer = defaultdict(list)

    def extract_info_all(info_batch, key):
        val = info_batch.get(key)
        if val is None:
            return []
        # MultiStepWrapper might return deques or numpy arrays (for numeric info)
        from collections import deque
        if isinstance(val, (list, deque, np.ndarray)):
            return list(val)
        return [val]


    # Initial observations from reset (MultiStepWrapper returns history_len repeated)
    curr_raw_obs = info["raw_obs"]
    if isinstance(curr_raw_obs, (list, np.ndarray)) and len(curr_raw_obs) > 0:
        curr_raw_obs = curr_raw_obs[0]
    curr_state = info.get("states")
    if curr_state is not None and isinstance(curr_state, (list, np.ndarray)) and len(curr_state) > 0:
        curr_state = curr_state[0]

    done = False
    step_count = 0
    max_steps = config.task.max_episode_steps

    while not done and step_count < max_steps:
        if config.task.obs_type == "state":
            obs_tensor = obs.astype(np.float32)[None, ...]
            obs_tensor = dataset.normalizer["obs"]["state"].normalize(obs_tensor)
            obs_input = {
                "state": torch.tensor(obs_tensor, device=config.optimization.device)
            }
        else:
            obs_input = {}
            for k in obs:
                val = obs[k].astype(np.float32)[None, ...]
                val = dataset.normalizer["obs"][k].normalize(val)
                obs_input[k] = torch.tensor(val, device=config.optimization.device)

        act_0 = torch.randn((1, config.task.horizon, config.task.act_dim), device=config.optimization.device)

        with torch.no_grad():
            # Use configurable num_steps instead of hardcoded 1
            act_normed = agent.sample(act_0=act_0, obs=obs_input, num_steps=config.optimization.num_steps, use_ema=True)

        act = dataset.normalizer["action"].unnormalize(act_normed.detach().cpu().numpy())

        start = config.task.obs_steps - 1
        end = start + config.task.act_steps
        act_exec = act[0, start:end, :]

        if config.task.abs_action and config.task.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
            # undo_transform_action expects (B, T, D)
            act_exec_unnorm = dataset.undo_transform_action(act_exec[None, ...])[0]
        else:
            act_exec_unnorm = act_exec

        # Execute in environment
        next_obs, reward, terminated, truncated, info = env.step(act_exec_unnorm)

        done = bool(terminated or truncated)

        # Get all sub-steps if MultiStepWrapper was used
        all_rewards = info["rewards"]
        all_dones = info["dones"]
        n_steps_done = len(all_rewards)

        all_next_raw_obs = extract_info_all(info, "raw_obs")[-n_steps_done:]
        all_next_states = extract_info_all(info, "states")[-n_steps_done:]

        # Record each sub-transition
        for i in range(len(all_next_raw_obs)):
            # Record current obs
            for k in config.task.obs_keys:
                v = curr_raw_obs[k]
                buffer[f"obs/{k}"].append(v)
            if curr_state is not None:
                buffer["states"].append(np.squeeze(curr_state))

            # Record action, reward, done, and next obs
            buffer["actions"].append(act_exec_unnorm[i])
            buffer["rewards"].append(all_rewards[i])
            buffer["dones"].append(all_dones[i])

            for k in config.task.obs_keys:
                v = all_next_raw_obs[i][k]
                buffer[f"next_obs/{k}"].append(v)

            # Update curr for next sub-step
            curr_raw_obs = all_next_raw_obs[i]
            if all_next_states:
                curr_state = all_next_states[i]

            step_count += 1
            if all_dones[i]:
                break

        obs = next_obs

    # Success heuristic: strictly positive total reward
    total_reward = sum(buffer["rewards"])
    success = total_reward > 0

    print(f"Episode {idx}: {step_count} steps, reward: {total_reward}, success: {success}")

    # Finalize video by calling reset (which triggers VideoRecordingWrapper to stop recording)
    env.reset()
    env.close()
    return buffer, success



@hydra.main(version_base=None, config_path="configs/", config_name="main")
def main(config):
    os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
    set_seed(config.optimization.seed)
    limit_threads(1)
    Logger(config)

    # post process config
    if config.network.network_type == "chiunet":
        # make sure config.task.horizon is a power of 2
        old_horizon = config.task.horizon
        config.task.horizon = int(2 ** np.ceil(np.log2(old_horizon)))
        print(f"ChiUNet requires horizon to be a power of 2, old horizon: {old_horizon}, new horizon: {config.task.horizon}")

    config.task.num_envs = 1
    config.task.save_video = True
    env_thunk = make_robomimic_env(
        config.task, idx=0, render=False, seed=config.optimization.seed
    )
    env = env_thunk()
    obs, _ = env.reset()
    config.task.obs_dim = (
        obs.shape[-1] if config.task.obs_type == "state" else config.network.emb_dim
    )
    env.close()

    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)

    if config.optimization.model_path is not None:
        agent.load(config.optimization.model_path, strict=False)
    elif config.optimization.auto_resume:
        checkpoint_base_name = (
            f"{config.task.env_name}_{config.task.env_type}_{config.task.obs_type}_"
            f"{config.optimization.loss_type}_{config.network.network_type}_"
            f"{config.network.emb_dim}_seed{config.optimization.seed}"
        )
        checkpoint_path = Logger(config).find_latest_checkpoint(checkpoint_base_name)
        if checkpoint_path:
            agent.load(str(checkpoint_path))

    agent.eval()

    num_demos = config.task.num_demos
    rejection_sample = config.task.rejection_sample

    if rejection_sample:
        print(f"Rejection sampling: collecting until {num_demos} successful episodes...")
    else:
        print(f"Collecting {num_demos} episodes...")

    buffers = []
    attempt = 0
    num_failures = 0

    while len(buffers) < num_demos:
        buffer, success = collect_one_episode(config, dataset, agent, idx=attempt)
        if rejection_sample:
            if success:
                buffers.append(buffer)
                print(f"  Accepted ({len(buffers)}/{num_demos} successful).")
            else:
                num_failures += 1
                print(f"  Rejected (failure {num_failures}).")
        else:
            buffers.append(buffer)
        attempt += 1

    if rejection_sample:
        print(f"Done: {num_demos} successful episodes after {attempt} rollouts ({num_failures} failures).")

    save_episodes(f"collected_{config.task.env_name}.hdf5", config.task.env_name, buffers)


if __name__ == "__main__":
    main()
