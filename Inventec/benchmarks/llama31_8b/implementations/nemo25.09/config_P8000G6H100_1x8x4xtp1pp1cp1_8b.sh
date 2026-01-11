source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=4
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1

export WARMUP_STEPS=20
export VAL_CHECK_INTERVAL=256

export LR=0.0004 #0.00008

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=720 #220
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# Inventec stuffs
export MLPERF_SUBMITTER="Inventec Corporation"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_SYSTEM_NAME="P8000G6H100"
export MLPERF_SUBMISSION_PLATFORM="Inventec P8000G6H100"
export MLPERF_STATUS="research"
export GPU_ARCH=h100
