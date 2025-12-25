: "${RETINANET_DATA_DIR:?ERROR: RETINANET_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR="${RETINANET_DATA_DIR}/open-images-v6"
export BACKBONE_DIR="${RETINANET_DATA_DIR}/torch-home/hub/checkpoints"
export LOGDIR=${PWD}/results
export CONT_FILE="${SQSH_DIR}/retinanet_ngc25.10_pyt.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
