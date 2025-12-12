# MLPerf DLRM-DCNv2 Training on Inventec P8000/P9000 GPU Servers

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
  - Also refer to [README-NVIDIA.md](README-NVIDIA.md) for hardware and software requirements

* Storage

  - Network storage (preferably a High Performance Storage) mounted on both the head and the compute nodes: `/hps` or `/mnt` on "head-p8000-1" and "compute-h100-1", "compute-h100-2", ...
  - Source code to be checked out in ${USER_DIR} (`/mnt/jkjung`), where the stable_diffusion benchmark code is found at `training_results_v5.0/Inventec/benchmarks/dlrm_dvnv2/implementations/hugectr25.03
  - Data (training data, validation data, checkpoints) in ${SD_DATA_DIR} (`/hps/data/mlperf/dlrm_dcnv2`)
  - Docker container SquashFS file in ${SQSH_DIR} (`/hps/sqsh`)

<a name="steps"></a>
Step-by-step
------------

1. Set environment variables.  Replace the paths below with your own if necessary.

   ```shell
   export USER_DIR=/mnt/jkjung
   export DLRM_DATA_DIR=/hps/data/mlperf/dlrm_dcnv2
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
   docker build -t mlperf-inventec:dlrm_hugectr25.03 .
   ```

3. Download preprocessed dataset on the compute node ("compute-h100-1").

   We use preprocessed dataset for this benchmark from MLCommons, which saves a lot of time!

   Start the container with the following command.

   ```bash
   docker run -it --rm --gpus=all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${DLRM_DATA_DIR}:/data mlperf-inventec:dlrm_hugectr25.03
   ```

   Then run the following within the container to download the preprocessed Criteo 1TB Click Logs dataset.  Total download size: ~3.7TB.  The script would verify md5 checksums in the end.

   ```bash
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d /data/criteo_1tb_multihot_raw https://training.mlcommons-storage.org/metadata/dlrmv2-preprocessed-criteo-click-logs.uri
   ```
   You could double check integrity of the train/val data files with the following commands.

   ```bash
   md5sum /data/criteo_1tb_multihot_raw/train_data.bin  # 4d48daf07cc244f6fa933b832d7fe5a3
   md5sum /data/criteo_1tb_multihot_raw/val_data.bin    # c7ca591ad3fd2b09b75d99fa4fc210e2
   ```

   Final dataset structure:

   ```
   /data/criteo_1tb_multihot_raw
   ├── train_data.bin
   └── val_data.bin
   ```

   This benchmark does not require downloading of checkpoints.  You should exit the Docker container for the rest of the steps.

4. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").

   ```bash
   enroot import -o ${SQSH_DIR}/dlrm_hugectro25.03.sqsh dockerd://mlperf-inventec:dlrm_hugectr25.03
   ```

5. Launch training with Slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000IG6H100_1x8x32.sh
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

8. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `SLURM_MPI_TYPE`: `pmi2` seems be perform better than `pmix` in our experiments.
