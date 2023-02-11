#! /bin/bash
export PYTHONPATH=/home/qinyujia/ModelCenter
export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR=localhost
MASTER_PORT=12452
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

BASE_PATH="/home/qinyujia/ModelCenter"
DATASET="webgpt"
SAVE="webgpt_bs16_epoch10"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /home/qinyujia/pretrained_models/cpm1-large"
# the real train batch size is train-batch-size * accumulation_steps
OPTS+=" --train-batch-size 8"
OPTS+=" --accumulation_steps 1"
OPTS+=" --eval-batch-size 1"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 2048"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name ${SAVE}"
OPTS+=" --lr 0.02"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-ratio 0.08"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --log_interval 50"
OPTS+=" --epochs 10"
OPTS+=" --query_only 0"
OPTS+=" --abstract_only 0"

# OPTS+=" --debug"
# OPTS+=" --load ${BASE_PATH}/results/cpm1-new.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${BASE_PATH}/examples/cpm1/finetune_cpm1_webgpt.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/cpm1/${SAVE}.log
