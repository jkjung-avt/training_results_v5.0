: "${LLAMA31_8B_DATA_DIR:?ERROR: LLAMA31_8B_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR="${LLAMA31_8B_DATA_DIR}"
export LOGDIR=${PWD}/results
export CONT_FILE="${SQSH_DIR}/llama31_8b_nemo25.09.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
