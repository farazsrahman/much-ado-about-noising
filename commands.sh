# SQUARE DEBUG FULL SWEEP ON OG DATA
  launcher=local \
```shell
CUDA_VISIBLE_DEVICES=2 uv run -- examples/train_robomimic.py \
  task=square_ph_state \
  network=chiunet \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  log.group=square_state_profiling
```

# SQUARE DEBUG FULL SWEEP ON OG DATA
uv run -- examples/train_robomimic.py \
  launcher=local \
  task=square_ph_image network=chiunet \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  log.group=square_state_profiling \

  log.eval_freq=2500 \
  optimization.gradient_steps=20001 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  optimization.seed=0,1,2,3
  hydra.launcher.gpus_per_node=4 \
  hydra.launcher.tasks_per_node=4 \
  hydra.sweeper.max_batch_size=4 \

  log.skip_eval_until=14990 \



# REAL
uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  task=square_ph_image network=chiunet \
  optimization.gradient_steps=20001 \
  log.eval_freq=2500 \
  log.skip_eval_until=14990 \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  log.group=can_ph_image_datasweep_full \
  hydra.launcher.gpus_per_node=4 \
  hydra.launcher.tasks_per_node=4 \
  hydra.sweeper.max_batch_size=4 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  optimization.seed=0,1,2,3


# DEBUG
uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  task=square_ph_image network=chiunet \
  optimization.gradient_steps=20001 \
  log.eval_freq=2500 \
  log.skip_eval_until=14990 \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  log.group=can_ph_image_datasweep_full \
  hydra.launcher.gpus_per_node=4 \
  hydra.launcher.tasks_per_node=4 \
  hydra.sweeper.max_batch_size=4 \
  task.train_subset_percentage=1.0 \
  optimization.seed=0


# REAL
SEED=0
CUDA_VISIBLE_DEVICES=$SEED uv run -- examples/train_robomimic.py \
  launcher=local \
  task=square_ph_image network=chiunet \
  log.group=can_ph_state \
  optimization.gradient_steps=50001 \
  log.eval_freq=10000 \
  log.eval_episodes=100 \
  task.train_subset_percentage=1.0 \
  log.wandb_mode=online \
  optimization.seed=$SEED


# REAL
uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  task=square_ph_state network=chiunet \
  optimization.gradient_steps=10001 \
  log.eval_freq=10000 \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  log.group=can_ph_state_datasweep \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=1 \
  hydra.sweeper.max_batch_size=1 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  optimization.seed=0,1,2,3




uv run -- examples/train_robomimic.py \
  launcher=local \
  task=can_ph_state network=chiunet \
  optimization.gradient_steps=10001 \
  log.eval_freq=10000 \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=1 \
  hydra.sweeper.max_batch_size=1 \
  task.train_subset_percentage=0.01




uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  task=can_ph_state network=chiunet \
  optimization.gradient_steps=501 \
  log.log_freq=500 \
  log.eval_freq=10000 \
  log.eval_episodes=100 \
  log.wandb_mode=online \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=1 \
  hydra.sweeper.max_batch_size=1 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  optimization.seed=0,1,2,3






uv run -- examples/train_robomimic.py \
  launcher=local \
  optimization.seed=0 \

# debug
uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  optimization.seed=0 \
  task=can_ph_state network=chiunet \
  log.eval_freq=10000 log.eval_episodes=100 \
  optimization.gradient_steps=5000 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=1 \
  task.train_subset_percentage=0.75 \
  log.wandb_mode=online




uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  optimization.seed=0,1 \
  task=can_ph_state network=chiunet \
  log.eval_freq=10000 log.eval_episodes=100 \
  optimization.gradient_steps=5000 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=2 \
  task.train_subset_percentage=0.75 \
  log.wandb_mode=online


uv run -- examples/train_robomimic.py --multirun \
  launcher=local \
  optimization.seed=0,1,2,3,4 \
  task=can_ph_state network=chiunet \
  log.eval_freq=10000 log.eval_episodes=100 \
  optimization.gradient_steps=5000 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.tasks_per_node=5 \
  task.train_subset_percentage=0.75 \
  log.wandb_mode=online


# COLLECTION: Collect 50 episodes for the 'lift' task
# PYTHONPATH=. .venv/bin/python examples/collect_robomimic.py \
#   task=lift_ph_state \
#   network=chiunet \
#   optimization.model_path=checkpoints/lift_ph_state_flow_chiunet_256_seed0_success100.pt \
#   optimization.seed=0 \
#   task.num_envs=1 \
#   ++task.num_demos=50 \
#   task.save_video=False

# TRAINING: Train a Flow Matching policy on the collected data
# PYTHONPATH=. .venv/bin/python examples/train_robomimic.py \
#   task=lift_ph_state \
#   network=chiunet \
#   ++task.dataset_path=$(pwd)/collected_lift.hdf5 \
#   optimization.model_path=null


# COLLECTION: Collect 50 episodes for the 'can' task
# PYTHONPATH=. .venv/bin/python examples/collect_robomimic.py \
#   task=can_ph_state \
#   network=chiunet \
#   optimization.model_path=checkpoints/can_ph_state_flow_chiunet_256_seed0_success100.pt \
#   optimization.seed=0 \
#   task.num_envs=1 \
#   ++task.num_demos=50 \
#   task.save_video=False

# COLLECTION: Collect 50 episodes for the 'square' task (using best available 93% checkpoint)
# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python examples/collect_robomimic.py \
#   task=square_ph_state \
#   network=chiunet \
#   optimization.model_path=checkpoints/square_ph_state_flow_chiunet_256_seed0_success93.pt \
#   optimization.seed=0 \
#   task.num_envs=1 \
#   ++task.num_demos=50 \
#   task.save_video=False \
#   optimization.num_steps=50 \
#   task.act_steps=1


# Commands for training
uv run examples/train_robomimic.py \
  task=square_ph_state \
  network=chiunet \
  optimization.seed=0 \
  log.eval_freq=5000 \
  log.log_freq=500 \
  log.wandb_mode=online

  \
  log.exp_name="profile_square_ph_state_w_data_collect"


uv run examples/train_robomimic.py \
  task=square_ph_state \
  network=chiunet \
  ++task.dataset_path=$(pwd)/collected_square.hdf5 \
  optimization.seed=0 \
  log.eval_freq=5000 \
  log.log_freq=500 \
  log.wandb_mode=online


# COLLECTION: Collect 50 episodes for the 'lift' task
uv run examples/collect_robomimic.py \
  task=lift_ph_state \
  network=chiunet \
  optimization.model_path=checkpoints/lift_ph_state_flow_chiunet_256_seed0_success100.pt \
  optimization.seed=0 \
  task.num_envs=1 \
  ++task.num_demos=20 \
  task.save_video=False


# DEBUG Eval
uv run examples/train_robomimic.py \
  task=square_ph_state \
  network=chiunet \
  optimization.seed=0 \
  log.eval_freq=100 \
  log.log_freq=50 \
  log.wandb_mode=online


uv run examples/train_robomimic.py \
  task=lift_ph_state \
  network=chiunet \
  optimization.seed=0 \
  log.eval_freq=1000 \
  log.log_freq=500 \
  log.wandb_mode=online \
  ++task.dataset_path=$(pwd)/collected_lift.hdf5


# COLLECTION: Collect 200 episodes for the 'square' task
uv run examples/collect_robomimic.py \
  task=square_ph_state \
  network=chiunet \
  optimization.model_path=checkpoints/square_ph_state_flow_chiunet_256_seed0_success95.pt \
  optimization.seed=0 \
  task.num_envs=1 \
  ++task.num_demos=200 \
  task.save_video=False

uv run examples/train_robomimic.py --multirun \
  launcher=basic \
  task=square_ph_state \
  network=chiunet \
  optimization.seed=0 \
  log.eval_freq=10000 \
  log.log_freq=5000 \
  optimization.gradient_steps=50000 \
  log.wandb_mode=online \
  ++task.dataset_path=$(pwd)/collected_square.hdf5 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0

uv run examples/train_robomimic.py --multirun \
  launcher=basic \
  task=square_ph_state \
  network=chiunet \
  optimization.seed=0 \
  log.eval_freq=4 \
  log.eval_episodes=1 \
  log.log_freq=1 \
  log.wandb_mode=online \
  optimization.gradient_steps=5 \
  ++task.dataset_path=$(pwd)/collected_square.hdf5 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0




uv run examples/train_robomimic.py --multirun \
  launcher=basic \
  task=square_ph_state \
  network=chiunet \
  log.group="sweep_robomimic_square_state_ph" \
  log.eval_freq=5000 \
  log.log_freq=2500 \
  optimization.gradient_steps=10000 \
  log.wandb_mode=online \
  'task.dataset_path="default","collected_square.hdf5"' \
  optimization.seed=0,1,2,3,4 \
  task.train_subset_percentage=0.01,0.05,0.1,0.2,0.3,0.5,0.75,1.0
