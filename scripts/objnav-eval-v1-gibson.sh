#!/bin/bash
#SBATCH --job-name eval
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 1
#SBATCH --partition short
#SBATCH --constraint "a40"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

source activate zson

CKPT_DIR="data/checkpoints/zson_conf_A.pth"
DATA_PATH="data/datasets/objectnav/gibson/v1/{split}/{split}.json.gz"

set -x
srun python -u run.py \
--exp-config configs/experiments/objectnav_v1.yaml \
    --run-type eval \
    TASK_CONFIG.TASK.SENSORS '["OBJECTGOAL_PROMPT_SENSOR"]' \
    TASK_CONFIG.TASK.MEASUREMENTS '["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL", "AGENT_ROTATION", "AGENT_POSITION"]' \
    EVAL_CKPT_PATH_DIR $CKPT_DIR \
    EVAL.SPLIT "val" \
    NUM_ENVIRONMENTS 4 \
    TASK_CONFIG.DATASET.DATA_PATH $DATA_PATH \
    RL.POLICY.pretrained_encoder 'data/models/omnidata_DINO_02.pth' \
    RL.REWARD_MEASURE "distance_to_goal" \
    RL.POLICY.CLIP_MODEL "RN50" \



    


