#!/bin/bash

set -x

export WANDB_API_KEY="XXX"
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct  # replace it with your local file path
EXPERIMENT_NAME=Qwen3_vl_2b_nav_grpo_eqa+obj_1228
SAVE_PATH=/home/szx/project/NavEvolver_v2/DATA/results/${EXPERIMENT_NAME}

CUDA_VISIBLE_DEVICES=2,3 python3 -m verl.trainer.main \
    config=examples/config_low_mem.yaml \
    data.train_files=/home/szx/project/NavEvolver_v2/DATA/CG-DATA-5_qwen3vlplus \
    data.val_files=/home/szx/project/NavEvolver_v2/DATA/CG-DATA-5_qwen3vlplus \
    data.usage_ratio=1 \
    data.val_ratio=0.2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.n_gpus_per_node=2
