#!/bin/bash

set -euo pipefail

# you may config here
model_size="14.6b"  # "1.5b", "6.2b", "14.6b", "28.3b"
schedule_type="zbv"  # "1f1b", "1f1b-i", "zb1p", "zb2p", "zbv"
job_name="zb${model_size}_${schedule_type}"
###
echo "model gpt3-$model_size schedule $schedule_type"

DIR=`pwd`
export JOB_DIR="$DIR/logs/$job_name"
mkdir -p $JOB_DIR

source ./examples/eurosys25/gpt3-$model_size

export MASTER_PORT=12353
export GPUS_PER_NODE=8

export EXIT_INTERVAL=1000
export EVAL_INTERVAL=10000
export TP_SIZE=1

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

srun -p Intern5 --nodes=2 --gres=gpu:8 --exclusive --ntasks-per-node=1 --cpus-per-task=24 \
 --job-name=$job_name bash examples/pretrain_zero_bubble_pjlab.sh