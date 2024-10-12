#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate model
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=640
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/model/RMtrainer.py"

n_nodes=1
n_gpus_per_node=4
torch_num_workers=4
batch_size=1
pin_memory=False
#exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"
#    --exp_name=${exp_name} \
python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name="megadepth_train" \
    --gpus=1 \
    --num_nodes=$n_nodes \
    --batch_size=$batch_size \
    --num_workers=$torch_num_workers \
    --pin_memory=$pin_memory \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30
