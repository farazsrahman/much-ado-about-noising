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
