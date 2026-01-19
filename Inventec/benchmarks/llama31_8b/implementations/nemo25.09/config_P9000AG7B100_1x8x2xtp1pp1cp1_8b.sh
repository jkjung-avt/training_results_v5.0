source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=2
export FULL_CUDA_GRAPH=0

export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768


export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=220
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export PLATFORM_SRUN_OPTIONS='--export=ALL,UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1"'

# Inventec stuffs
export MLPERF_SUBMITTER="Inventec Corporation"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_SYSTEM_NAME="P9000AG7B200"
export MLPERF_SUBMISSION_PLATFORM="Inventec P9000AG7B200"
export MLPERF_STATUS="research"
export GPU_ARCH=b200
