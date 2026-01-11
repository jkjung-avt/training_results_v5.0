# MLPerf Llama 3.1 8B Training on Inventec P8000/P9000 GPU Servers

Table of contents
-----------------

* [Environment Setup](#setup)
* [Step-by-step](#steps)
* [Known Issues](#issues)

<a name="setup"></a>
Environment Setup
------------------

* Machines

  - Slurm head node * 1, e.g. "p8000-head-1"
  - Slurm compute node * 1 or more, e.g. "compute-h100-1", "compute-h100-2", ...
  - Also refer to [README_8b.md](README_8b.md) for hardware and software requirements

* Storage

  - Network storage (preferably a High Performance Storage) mounted on both the head and the compute nodes: `/hps` or `/mnt` on "head-p8000-1" and "compute-h100-1", "compute-h100-2", ...
  - Source code (`training_results_v5.0`) to be checked out in ${USER_DIR} (`/mnt/jkjung`)
  - Data (training data, validation data, checkpoints) in ${LLAMA31_8B_DATA_DIR} (`/hps/data/mlperf_training/llama31`)
  - Docker container SquashFS file in ${SQSH_DIR} (`/hps/sqsh`)

<a name="steps"></a>
Step-by-step
------------

1. Set environment variables.  Replace the paths below with your own if necessary.

   ```shell
   export USER_DIR=/mnt/jkjung
   export LLAMA31_8B_DATA_DIR=/hps/data/mlperf_training/llama31
   export SQSH_DIR=/hps/sqsh
   ```

   Then clone this repository on the compute node ("compute-h100-1").

   ```shell
   cd ${USER_DIR}
   git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Build the container on the compute node ("compute-h100-1").  You will have to use your NGC API key to pull the base PyTorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.

   ```shell
   cd training_results_v5.0/Inventec/benchmarks/dlrm_dcnv2/implementations/hugectr25.03/
   docker build -t mlperf-inventec:llama31_8b_nemo25.09 .
   ```

3. Prepare dataset on the compute node ("compute-h100-1").

   Set the directory for the data to be downloaded to.  Then download the dataset.  This takes a couple of hours.

   ```bash
   export DATADIR=<path/to/dataset>
   bash data_scripts/download_8b.sh
   ```

   At the end, the directory structure should look like:

   ```bash
   $ tree ${LLAMA31_8B_DATA_DIR}/8b
   8b/
   |-- LICENSE.txt
   |-- NOTICE.txt
   |-- c4-train.en_6_text_document.bin
   |-- c4-train.en_6_text_document.idx
   |-- c4-validation-91205-samples.en_text_document.bin
   |-- c4-validation-91205-samples.en_text_document.idx
   |-- llama-3-1-8b-preprocessed-c4-dataset.md5
   `-- tokenizer
       |-- LICENSE
       |-- README.md
       |-- USE_POLICY.md
       |-- llama-3-1-8b-tokenizer.md5
       |-- special_tokens_map.json
       |-- tokenizer.json
       `-- tokenizer_config.json

   2 directories, 14 files
   ```

4. Prepare the model and checkpoint.

   The model largely follows the paper titled [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).

   The LLama3.1 8B is trained from scratch and is not using a checkpoint.

5. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").

   ```bash
   enroot import -o ${SQSH_DIR}/llama31_8b_nemo25.09.sqsh dockerd://mlperf-inventec:llama31_8b_nemo25.09
   ```

6. Launch training with Slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000G6H100_1x8x4xtp1pp1cp1_8b.sh
   sbatch -w compute-h100-1 --time=${WALLTIME} run.sub
   ```

   Note:

   * Rename "compute-h100-1" if you are using a different Slurm compute node.
   * It's a good idea, before you start `run.sub`, to verify that there is no process occupying the CPUs/GPUs/memory on the compute node.  For example, do `docker ps -a` and `docker rm <CONTAINER ID>` to remove all running/pending containers.
   * You could adjust experiment setting in the `env.sh` script.

   The above `sbatch` command would output the Slurm batch job id.  You could track progress of the Slurm batch job by checking the corresponding log file.

   ```bash
   tail -fn +1 slurm-<SLURM JOB ID>.out
   ```

7. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `SLURM_MPI_TYPE`: `pmi2` seems be perform better than `pmix` in our experiments.
