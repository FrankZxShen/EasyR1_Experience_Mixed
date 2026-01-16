#!/bin/bash

set -x

export WANDB_API_KEY="xxx"
MODEL_PATH=/workdir/szx/LLM_data/Qwen3-VL-2B-Instruct  # replace it with your local file path
EXPERIMENT_NAME=Qwen3_vl_2b_nav_grpo_eqa+obj_260115_mix_data
SAVE_PATH=/workdir/szx/nav_data/data/LLM_results/${EXPERIMENT_NAME}

CUDA_VISIBLE_DEVICES=2,3 python3 -m verl.trainer.main \
    config=examples/config_low_mem.yaml \
    data.train_files=/home/whj/szx_project/EasyR1/DATA/CG-DATA-InteriorGS:0.5,/home/whj/szx_project/EasyR1/DATA/CG-DATA-5_qwen3vlplus:0.5 \
    data.val_files=/home/whj/szx_project/EasyR1/DATA/CG-DATA-InteriorGS:0.5,/home/whj/szx_project/EasyR1/DATA/CG-DATA-5_qwen3vlplus:0.5 \
    data.usage_ratio=1 \
    data.val_ratio=0.2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.n_gpus_per_node=2
