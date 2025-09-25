#!/bin/bash

PACKING_FACTOR=3 \
INIT_LOSS_SCALE=1024.0 \
USE_FLASH_ATTENTION=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
    /workspace/bert/run_pretraining.py \
    --seed=42 \
    --train_batch_size=48 \
    --learning_rate=0.00096 \
    --opt_lamb_beta_1=0.60466 \
    --opt_lamb_beta_2=0.85437 \
    --warmup_proportion=1.0 \
    --warmup_steps=0.0 \
    --start_warmup_step=0 \
    --max_steps=200 \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/workspace/bert_data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=0.720 \
    --sustained_training_time=0 \
    --weight_decay_rate=0.1 \
    --max_samples_termination=14000000 \
    --eval_iter_start_samples=100000 \
    --eval_iter_samples=100000 \
    --eval_batch_size=16 \
    --eval_dir=/workspace/bert_data/hdf5/eval_varlength \
    --num_eval_examples=10000 \
    --cache_eval_data \
    --output_dir=/results \
    --fp16 \
    --distributed_lamb \
    --dwu-num-rs-pg=1 \
    --dwu-num-ar-pg=1 \
    --dwu-num-ag-pg=1 \
    --dwu-num-blocks=1 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --dense_seq_output \
    --pad_fmha \
    --fused_bias_fc \
    --fused_bias_mha \
    --fused_dropout_add \
    --fused_gemm_gelu \
    --packed_samples \
    --use_transformer_engine2 \
    --cuda_graph_mode 'segmented' \
    --use_cuda_graph \
    --eval_cuda_graph \
    --bert_config_path=/workspace/bert_data/phase1/bert_config.json \
    --init_checkpoint=/workspace/bert_data/phase1/model.ckpt-28252.pt
