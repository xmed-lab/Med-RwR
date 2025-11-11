export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8 # Total GPUs
export NPROC_PER_NODE=8 # GPUs per node
export MASTER_ADDR=127.0.0.1  # Master node address
export MASTER_PORT=29400 # Port for communication
export MAX_PIXELS=262144
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TIMEOUT=-1

export MODEL_PATH=""
DIR_PATH=$(dirname "${MODEL_PATH}")

export SWIFT_ENGINE_TYPE="pt_engine_retrieve_infer"
# export SWIFT_ENGINE_TYPE="pt_engine_retrieve_infer_img"  ## for confidence-driven image re-retrieval

swift infer \
    --model ${MODEL_PATH} \
    --infer_backend pt \
    --attn_impl flash_attn \
    --max_batch_size 2 \
    --max_new_tokens 4096 \
    --val_dataset VAL_DATASET_PATH \
    --result_path ${DIR_PATH}/result.jsonl
