#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 \
    /workspace/bert/run_pretraining.py \
    --seed=42 \
    --do_train \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=0.714 \
    --bert_config_path=/workspace/bert_data/phase1/bert_config.json \
    --skip_checkpoint \
    --output_dir=/results \
    --fp16 \
    --distributed_lamb \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --train_batch_size=4 \
    --learning_rate=4e-5 \
    --warmup_proportion=1.0 \
    --input_dir=/workspace/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --max_steps=100 \
    --init_checkpoint=/workspace/bert_data/phase1/model.ckpt-28252.pt
