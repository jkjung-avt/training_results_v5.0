# MLPerf Retinanet Training on Inventec P8000/P9000 GPU Servers

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
  - Source code (`training_results_v5.0`) to be checked out in ${USER_DIR} (`/mnt/jkjung`)
  - Data (training data, validation data, checkpoints) in ${RETINANET_DATA_DIR} (`/hps/data/mlperf_training/retinanet`)
  - Docker container SquashFS file in ${SQSH_DIR} (`/hps/sqsh`)

<a name="steps"></a>
Step-by-step
------------

1. Set environment variables.  Replace the paths below with your own if necessary.

   ```shell
   export USER_DIR=/mnt/jkjung
   export RETINANET_DATA_DIR=/hps/data/mlperf_training/retinanet
   export SQSH_DIR=/hps/sqsh
   ```

   Then clone this repository on the compute node ("compute-h100-1").

   ```shell
   cd ${USER_DIR}
   git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Build the container on the compute node ("compute-h100-1").  You will have to use your NGC API key to pull the base PyTorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.  Note that the docker image name "bert_ngc23.09_pyt" is different from that in NVIDIA's original implementation.

   ```shell
   cd training_results_v5.0/Inventec/benchmarks/retinanet/implementations/pytorch25.10/
   docker build -t mlperf-inventec:retinanet_pytorch25.10 .
   ```

3. Prepare dataset on the compute node ("compute-h100-1").

   Start the container with the following command.

   ```bash
   docker run -it --rm --gpus=all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${RETINANET_DATA_DIR}:/workspace/ssd_dataset mlperf-inventec:retinanet_pytorch25.10
   ```

   Then run within the container:

   ```bash
   cd /workspace/ssd
   pip install --upgrade numpy==1.26.4
   pip install opencv_python_headless==4.11.0.86
   pip install fiftyone==1.2.0
   export FIFTYONE_DATASET_ZOO_DIR=/workspace/ssd_dataset
   ./public-scripts/download_openimages_mlperf.sh -d /workspace/ssd_dataset/open-images-v6
   ```

   This script may take around a whole day to complete (~4 hours for downloading coco dataset and ~18 hours for pre-processing images).  The expected folder structure after download and conversion is:

   ```
   /workspace/ssd_dataset/open-images-v6
   │
   └───info.json
   │
   └───train
   │   └─── data  # 1170301 files inside
   │   │      000002b66c9c498e.jpg
   │   │      000002b97e5471a0.jpg
   │   │      ...
   │   └─── metadata
   │   │      classes.csv
   │   │      hierarchy.json
   │   │      image_ids.csv
   │   └─── labels
   │          detections.csv
   │          openimages-mlperf.json  # conversion output
   │
   └───validation
       └─── data  # 24781 files inside
       │      0001eeaf4aed83f9.jpg
       │      00075905539074f2.jpg
       │      ...
       └─── metadata
       │      classes.csv
       │      hierarchy.json
       │      image_ids.csv
       └─── labels
              detections.csv
              openimages-mlperf.json  # conversion output
   ```

4. Download the pretrained backbone by executing the following within the container:

   ```bash
   ./public-scripts/download_backbone.sh
   ```

5. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").  The created `${SQSH_DIR}/retinanet_pytorch25.10.sqsh` file is needed for running the experiment with Slurm.

   ```bash
   enroot import -o ${SQSH_DIR}/retinanet_pytorch25.10.sqsh dockerd://mlperf-inventec:retinanet_pytorch25.10
   ```

6. Launch training with Slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000G6H100_1x8x32.sh
   sbatch -w compute-h100-1 --time=${WALLTIME} run.sub
   ```

   Note:

   * Rename "compute-h100-1" if you are using a different Slurm compute node.
   * It's a good idea, before you start `run.sub`, to verify that there is no process occupying the CPUs/GPUs/memory on the compute node.  For example, do `docker ps -a` and `docker rm <CONTAINER ID>` to remove all running/pending containers.
   * You could adjust experiment setting in the `env.sh` script.

   The above `sbatch` command would output the Slurm batch job id.  You could track progress of the Slurm batch job by checking the corresponding log file.

   ```bash
   tail -f slurm-<SLURM JOB ID>.out
   ```

7. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `SLURM_MPI_TYPE`: `pmi2` seems be perform better than `pmix` in our experiments.
