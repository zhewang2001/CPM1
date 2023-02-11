#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12347
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
DATASET="CMRC"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config /home/qinyujia/pretrained_models/cpm1-large"
OPTS+=" --train-batch-size 32"
OPTS+=" --eval-batch-size 1"
OPTS+=" --train-iters 3000"
OPTS+=" --save-iters 1000"
OPTS+=" --max-length 512"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-cpm1-ckpt"
OPTS+=" --lr 0.02"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-ratio 0.08"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-3"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --log_interval 50"
OPTS+=" --epochs 10"

OPTS+=" --load ${BASE_PATH}/results/finetune-cpm1-ckpt_new_test-best.pt"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/cpm1/infer_cpm1.py ${OPTS}"
echo ${CMD}

CUDA_VISIBLE_DEVICES=1 ${CMD} 2>&1 | tee ${BASE_PATH}/logs/cpm1/${DATASET}_infer.log