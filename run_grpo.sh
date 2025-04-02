export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
export EXPERIMENT_NAME=grpo
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
# set available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
# calculate the number of gpus
export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# dataset name
export DATASET_NAME='gsm8k'
# Sanitize the base model name by replacing forward slashes with underscores
export SANITIZED_MODEL_NAME=$(echo $BASE_MODEL | tr '/' '_')
export PROJECT_NAME=${DATASET_NAME}_${SANITIZED_MODEL_NAME}_grpo


python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/${DATASET_NAME}/train.parquet \
    data.val_files=./data/${DATASET_NAME}/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.max_start_length=256 \
    data.max_tool_response_length=2048 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    tool.env='python' $@