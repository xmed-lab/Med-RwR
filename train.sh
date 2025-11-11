export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8 # Total GPUs
export NPROC_PER_NODE=8 # GPUs per node
export MASTER_ADDR=127.0.0.1  # Master node address
export MASTER_PORT=29500 # Port for communication
export MAX_PIXELS=262144
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TIMEOUT=-1

export SWIFT_GRPO_TRAINER_TYPE="grpo_trainer_retrieve"
export SWIFT_ENGINE_TYPE="pt_engine_retrieve_train"

export WANDB_API_KEY=""

swift rlhf \
    --rlhf_type grpo \
    --model MODEL_INITIAL_PATH \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_func external_acc external_retrieve_format external_length external_retrieve_semantic external_retrieve_imagesim external_retrieve_logit \
    --use_vllm False \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset DATASET_PATH \
    --max_length 8192 \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --eval_strategy no \
    --save_steps 10 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1 \
    --top_p 1.0 \
    --top_k 0 \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --async_generate false \
    --attn_impl flash_attn \
    --report_to tensorboard wandb \
    --beta 0.001 \
    --gradient_accumulation_steps 2