: "${USER_DIR:?ERROR: USER_DIR not set}"
: "${BERT_DATA_DIR:?ERROR: BERT_DATA_DIR not set}"
: "${SQSH_DIR:?ERROR: SQSH_DIR not set}"

export EVALDIR="${BERT_DATA_DIR}/hdf5/eval_varlength"
export DATADIR_PHASE2="${BERT_DATA_DIR}/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="${BERT_DATA_DIR}/packed_data"
export CHECKPOINTDIR_PHASE1="${BERT_DATA_DIR}/phase1"
export LOGDIR=${PWD}/results
export CONT=mlperf-nvidia:bert_ngc23.09_pyt
export CONT_FILE="${SQSH_DIR}/bert_ngc23.09_pyt.sqsh"
export SLURM_MPI_TYPE=pmi2
export NEXP=1   # Number of experiments (training runs)
