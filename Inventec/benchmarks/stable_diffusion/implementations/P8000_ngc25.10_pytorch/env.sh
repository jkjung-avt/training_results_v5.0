: "${SD_DATA_DIR:?ERROR: SD_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR="${SD_DATA_DIR}/datasets"
export COCODIR="${SD_DATA_DIR}/datasets"
export CHECKPOINT_CLIP="${SD_DATA_DIR}/checkpoints/clip"
export CHECKPOINT_FID="${SD_DATA_DIR}/checkpoints/inception"
export CHECKPOINT_SD="${SD_DATA_DIR}/checkpoints/sd"
export LOGDIR=${PWD}/results
export CONT_FILE="${SQSH_DIR}/sd_ngc25.04_pyt.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
