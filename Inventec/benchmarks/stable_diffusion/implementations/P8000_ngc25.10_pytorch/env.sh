: "${RETINANET_DATA_DIR:?ERROR: RETINANET_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR="/hps/data/mlperf/stable_diffusion/datasets/laion-400m"
export COCODIR="/hps/data/mlperf/stable_diffusion/datasets/coco2014"
export CHECKPOINT_CLIP="/hps/data/mlperf/stable_diffusion/checkpoints/clip"
export CHECKPOINT_FID="/hps/data/mlperf/stable_diffusion/checkpoints/sd"
export CHECKPOINT_SD="/hps/data/mlperf/stable_diffusion/checkpoints/inception"
export LOGDIR=${PWD}/results
export CONT_FILE="${SQSH_DIR}/sd_ngc25.04_pyt.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
