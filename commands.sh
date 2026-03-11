# eval command
# checkpoints/image_transformers/can_mh_image_flow_sudeepdit_256_seed0_success100.pt

uv run examples/eval_robomimic.py \
    mode=eval \
    task=can_mh_image \
    network=sudeepdit \
    optimization.loss_type=flow \
    log.eval_episodes=1 \
    task.num_envs=1 \
    optimization.model_path="/home/farazr/explain-mimic/checkpoints/image_transformers/can_mh_image_flow_sudeepdit_256_seed0_success100.pt"

# save_rollouts command
uv run examples/eval_robomimic.py \
    mode=save_rollouts \
    task=can_mh_image \
    network=sudeepdit \
    optimization.loss_type=flow \
    log.eval_episodes=1 \
    task.num_envs=1 \
    log.save_video=true \
    optimization.model_path="/home/farazr/explain-mimic/checkpoints/image_transformers/can_mh_image_flow_sudeepdit_256_seed0_success100.pt"

# explain command
uv run examples/explain.py \
    --rollout logs/rollouts/can_video_0.npz \
    --index 5 \
    --model_path "/home/farazr/explain-mimic/checkpoints/image_transformers/can_mh_image_flow_sudeepdit_256_seed0_success100.pt"

# explain PatchChiTransformer with trained checkpoint
uv run examples/explain.py \
    --rollout logs/rollouts/can_video_0.npz \
    --index 5 \
    --task can_mh_image \
    --network patchchitransformer \
    --model_path "/home/farazr/explain-mimic/checkpoints/image_transformers/can_mh_image_flow_patchchitransformer_obs2.pt"

# PatchChiTransformer smoke test (random weights, no checkpoint needed)
uv run examples/explain.py \
    --rollout logs/rollouts/can_video_0.npz \
    --index 0 \
    --task can_mh_image_patch \
    --network patchchitransformer

# train ChiTransformer baseline (lift_mh_image)
uv run examples/train_robomimic.py \
    task=lift_mh_image \
    network=chitransformer \
    optimization.loss_type=flow \
    optimization.seed=0 \
    log.log_freq=100 \
    log.eval_freq=5000 \
    optimization.gradient_steps=50000

# train PatchChiTransformer (lift_mh_image)
uv run examples/train_robomimic.py \
    task=lift_mh_image \
    network=patchchitransformer \
    optimization.loss_type=flow \
    optimization.seed=0 \
    log.log_freq=100 \
    log.eval_freq=5000 \
    log.eval_episodes=10 \
    optimization.gradient_steps=50000

# This is needed when selecting GPU
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=0 \
