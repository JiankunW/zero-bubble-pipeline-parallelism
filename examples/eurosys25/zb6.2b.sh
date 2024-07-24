#!/bin/bash

set -euo pipefail

# you may config here
model_size="6.2b"  # "1.5b", "6.2b", "14.6b", "28.3b"
schedule_type="zbv"  # "1f1b", "1f1b-i", "zb1p", "zb2p", "zbv"
###

source ./examples/eurosys25/gpt3-$model_size

export MASTER_PORT=12353
export GPUS_PER_NODE=8

export TP_SIZE=1
# export PIPELINE_SIZE=1
# export MICRO_BATCH_SIZE=3
# export NUM_MICRO_BATCH=24
# export GLOBAL_BATCH_SIZE=$(( $NUM_MICRO_BATCH * $MICRO_BATCH_SIZE ))
# args=" \
#     --use-distributed-optimizer \
#     --overlap-grad-reduce \
#     --overlap-param-gather"
args="--fp16"

if [ $PIPELINE_SIZE -gt 1 ]; then
    if [ $schedule_type == "1f1b-i" ]; then
        export INTERLEAVED_1F1B=1
    elif [ $schedule_type == "zb1p" ]; then
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((1 * $PIPELINE_SIZE))
    elif [ $schedule_type == "zb2p" ]; then
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
    elif [ $schedule_type == "zbv" ]; then
        export ZERO_BUBBLE_V_SCHEDULE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))  # TODO: useless. Remove it.
    fi
    job_name="zb${model_size}_tp${TP_SIZE}pp${PIPELINE_SIZE}_${schedule_type}"
    echo "model gpt3-$model_size schedule $schedule_type"
else 
    job_name="zb${model_size}_tp${TP_SIZE}pp${PIPELINE_SIZE}"
fi

DIR=`pwd`
export JOB_DIR="$DIR/logs/$job_name"
mkdir -p $JOB_DIR

srun -p llm_s --nodes=1 --gres=gpu:8 --exclusive --ntasks-per-node=1 --cpus-per-task=24 \
 --job-name=$job_name bash examples/pretrain_zero_bubble_pjlab.sh $args