#!/bin/bash

# Runs the "700M" parameter model

set -x

echo "MLP_WORKER_GPU=$MLP_WORKER_GPU"
echo "MLP_WORKER_NUM=$MLP_WORKER_NUM"
echo "MLP_ROLE_INDEX=$MLP_ROLE_INDEX"
echo "MLP_WORKER_0_HOST=$MLP_WORKER_0_HOST"
echo "MLP_WORKER_0_PORT=$MLP_WORKER_0_PORT"
echo "RUN_ROOT=$RUN_ROOT"

DISTRIBUTED_ARGS="
    --nproc_per_node $MLP_WORKER_GPU \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT
"

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=$RUN_ROOT/checkpoints
VOCAB_FILE=/share_nfs/process_data/openwebtext_gpt2_tokenized/gpt2-vocab.json
MERGE_FILE=/share_nfs/process_data/openwebtext_gpt2_tokenized/gpt2-merges.txt
DATA_PATH=/share_nfs/process_data/openwebtext_gpt2_tokenized/tokenized_text_document

OPTIM=adam
PEAK_LR=${PEAK_LR:-2.5e-4}
MIN_LR=${MIN_LR:-2.5e-5}
WD=0.1
B1=0.9
B2=0.95
EPS=1e-8

MICRO_BSZ=${MB_SIZE:-4}
GLOBAL_BSZ=${GLOBAL_BSZ:-512}

TRAIN_ITERS=${TRAIN_ITERS:-48000}  # ~50B tokens, 5epoch
DECAY_ITERS=${DECAY_ITERS:-46000}  # 48B tokens
WARMUP_FRAC=${WARMUP_FRAC:-0.01}  # 0.5B tokens


GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1536 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size $MICRO_BSZ \
    --global-batch-size $GLOBAL_BSZ \
    --lr $PEAK_LR \
    --min-lr $MIN_LR \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters $DECAY_ITERS \
    --lr-decay-style cosine \
    --lr-warmup-fraction $WARMUP_FRAC \
    --use-distributed-optimizer \
    --optimizer $OPTIM \
    --weight-decay $WD \
    --adam-beta1 $B1 \
    --adam-beta2 $B2 \
    --adam-eps $EPS \
    --clip-grad 1.0 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 3200 \
    --eval-interval 500 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

