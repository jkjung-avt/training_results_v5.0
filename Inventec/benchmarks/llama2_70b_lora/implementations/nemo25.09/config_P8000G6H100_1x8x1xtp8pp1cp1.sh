#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0006
export MINIBS=2
export TP=8
export CP=1
export FP8_ACT=1

#export MCORE_CUDA_GRAPH=1
#export NUM_WORKERS=4

# To avoid OOM
export NCCL_NVLS_ENABLE=0

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=60
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Inventec stuffs
export MLPERF_SUBMITTER="Inventec Corporation"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_SYSTEM_NAME="P8000G6H100"
export MLPERF_SUBMISSION_PLATFORM="Inventec P8000G6H100"
export MLPERF_STATUS="research"
export GPU_ARCH=h100
