: "${LLAMA2_DATA_DIR:?ERROR: LLAMA2_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export DATADIR="${LLAMA2_DATA_DIR}/gov_report"
export MODEL="${LLAMA2_DATA_DIR}/model"
export LOGDIR="./results"
export CONT_FILE="${SQSH_DIR}/llama2_70b_lora_ngc25.09_pyt.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
