#!/bin/bash -l


TRAIN_IMG_SIZE=640
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/model/RMtrainer.py"

configs/data/megadepth_trainval_640.py
configs/model/outdoor/RMtrainer.py
--exp_name
megadepth_train
--gpus
1
--num_nodes
1
--batch_size
1
--num_workers
8
--pin_memory
True
--check_val_every_n_epoch
1
--log_every_n_steps
1
--flush_logs_every_n_steps
1
--limit_val_batches
1.
--num_sanity_val_steps
10
--benchmark
True
--max_epochs
2
