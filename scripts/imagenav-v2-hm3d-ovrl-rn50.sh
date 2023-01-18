#!/bin/bash
#SBATCH --job-name zson
#SBATCH --output log.out
#SBATCH --error log.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 8
#SBATCH --partition short
#SBATCH --constraint a40
#SBATCH --signal USR1@600
#SBATCH --requeue

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

source activate zson

set -x
srun \
python -u run.py \
--exp-config configs/experiments/imagenav_hm3d.yaml \
--run-type train \
NUM_CHECKPOINTS 400 \
TOTAL_NUM_STEPS 1e9 \
TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 500 \
TASK_CONFIG.TASK.SENSORS '["CACHED_GOAL_SENSOR"]' \
TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.DATA_PATH 'data/goal_datasets/imagenav/hm3d/v2-rn50/{split}/content/{scene}.npy' \
TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.SINGLE_VIEW True \
TASK_CONFIG.TASK.CACHED_GOAL_SENSOR.SAMPLE_GOAL_ANGLE True \
TASK_CONFIG.TASK.SIMPLE_REWARD.SUCCESS_REWARD 5.0 \
TASK_CONFIG.TASK.SIMPLE_REWARD.ANGLE_SUCCESS_REWARD 5.0 \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD True \
TASK_CONFIG.TASK.SIMPLE_REWARD.USE_ATG_REWARD True \
RL.POLICY.use_data_aug True \
RL.POLICY.CLIP_MODEL "RN50" \
RL.POLICY.pretrained_encoder 'data/models/omnidata_DINO_02.pth' \
