#! /bin/bash
export PYTHONPATH=/home/qinyujia/ModelCenter
export CUDA_VISIBLE_DEVICES=2
MASTER_ADDR=localhost
MASTER_PORT=12323
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

export PYTHONPATH=/home/qinyujia/ModelCenter

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT "

BASE_PATH="/home/qinyujia/ModelCenter"
DATASET="webgpt"

# LOADS="webgpt_newdata_bs64_epoch5_2048.pt"
LOADS="webgpt_bs64_epoch5_newdata_shuffle_partialpastaction-best.pt"

for LOAD in $LOADS
do

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /home/qinyujia/pretrained_models/cpm1-large"
# the real train batch size is train-batch-size * accumulation_steps
OPTS+=" --train-batch-size 4"
OPTS+=" --accumulation_steps 4"
OPTS+=" --eval-batch-size 1"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 2048"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name webgpt_new_test"
OPTS+=" --lr 0.02"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-ratio 0.08"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --log_interval 50"
OPTS+=" --epochs 5"
OPTS+=" --load ${LOAD}"
OPTS+=" --debug"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${BASE_PATH}/examples/cpm1/finetune_cpm1_webgpt_eval_query.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/logs/cpm1/evaluate_${LOAD}.log

done