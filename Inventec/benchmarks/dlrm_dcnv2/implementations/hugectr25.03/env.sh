: "${DLRM_DATA_DIR:?ERROR: DLRM_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR=${DLRM_DATA_DIR}/criteo_1tb_multihot_raw
export DATADIR_VAL=${DLRM_DATA_DIR}/criteo_1tb_multihot_raw
export LOGDIR=${PWD}/results
export CONT_FILE="${SQSH_DIR}/dlrm_hugectr25.03.sqsh"

export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
