# MLPerf Stable_Diffusion Training on Inventec P8000/P9000 GPU Servers

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
  - Source code to be checked out in ${USER_DIR} (`/mnt/jkjung`), where the stable_diffusion benchmark code is found at `training_results_v5.0/Inventec/benchmarks/stable_diffusion/implementations/P8000_ngc25.04_pytorch`
  - Data (training data, validation data, checkpoints) in ${SD_DATA_DIR} (`/hps/data/mlperf/stable_diffusion`)
  - Docker container SquashFS file in ${SQSH_DIR} (`/hps/sqsh`)

<a name="steps"></a>
Step-by-step
------------

1. Set environment variables.  Replace the paths below with your own if necessary.

   ```shell
   export USER_DIR=/mnt/jkjung
   export SD_DATA_DIR=/hps/data/mlperf/stable_diffusion
   export SQSH_DIR=/hps/sqsh
   ```

   Then clone this repository on the compute node ("compute-h100-1").

   ```shell
   cd ${USER_DIR}
   git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Build the container on the compute node ("compute-h100-1").  You will have to use your NGC API key to pull the base PyTorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.

   ```shell
   cd training_results_v5.0/Inventec/benchmarks/stable_diffusion/implementations/P8000_ngc25.04_pytorch/
   docker build -t mlperf-inventec:sd_ngc25.04_pyt .
   ```

3. Prepare dataset on the compute node ("compute-h100-1").

   The benchmark employs two datasets, both of which will be downloaded by the commands below.

   * Training: a subset of [laion-400m](https://laion.ai/blog/laion-400-open-dataset)
   * Validation: a subset of [coco-2014 validation](https://cocodataset.org/#download)

   Start the container with the following command.

   ```bash
   docker run -it --rm --gpus=all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${SD_DATA_DIR}/datasets:/datasets -v ${SD_DATA_DIR}/checkpoints:/checkpoints mlperf-inventec:sd_ngc25.04_pyt
   ```

   Then run the following within the container to download the LAION 400M preprocessed moments dataset.  Total download size: ~831GB.  The script would verify md5 checksums of all downloaded files (00000.tar ~ 00831.tar) in the end.

   ```bash
   cd /workspace/sd
   bash scripts/datasets/laion400m-filtered-download-moments.sh
   ```

   Next, download the COCO-2014-validation dataset and do preprocessing.  Total size after preprocessing: ~15GB.

   ```bash
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d /datasets/coco2014 https://training.mlcommons-storage.org/metadata/stable-diffusion-coco2014-validation-prompts-dataset.uri
   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d datasets/coco2014 https://training.mlcommons-storage.org/metadata/stable-diffusion-coco2014-validation-stats-dataset.uri
   bash scripts/datasets/coco-2014-validation-download.sh
   bash scripts/datasets/coco-2014-validation-split-resize.sh
```

4. Download checkpoints on the compute node ("compute-h100-1").

   Download checkpoint of the Stable Diffusion model.  This component leverages StabilityAI's 512-base-ema.ckpt checkpoint from HuggingFace.  While the checkpoint includes weights for the UNet, VAE, and OpenCLIP text embedder, the UNet weights are not used and are discarded when loading the weights.  Download size: ~4.9GB.

   ```bash
   bash scripts/checkpoints/download_sd.sh --output-dir /checkpoints/sd
   ```

   Download checkpoint of the Inception network, which is employed during validation to compute the Fréchet Inception Distance (FID) score.  Download size: <100MB.

   ```bash
   bash scripts/checkpoints/download_inception.sh --output-dir /checkpoints/inception
   ```

   Download checkpoint of the OpenCLIP ViT-H-14 Model.  This model is utilized for the computation of the CLIP score.  Download size: ~3.7GB.

   ```bash
   python -c "from infer_and_eval_tools import CLIPEncoder; clip_model = CLIPEncoder(clip_version='ViT-H-14', cache_dir='/checkpoints/clip', device='cpu')"
   ```
5. Preprocess Laion 400m dataset

   This script creates a version of dataset that contains encoded CLIP captions.  It requires single node with preferably 8 GPUs and 2.7TB space for outputs.  There are 832 shards to process.  Splitting the work across 8 GPUs and assuming each fragment takes ~100 seconds to process, the total time needed is estimated at ~3 hours.

   ```bash
   bash scripts/datasets/laion400m-encode-captions.sh
   ```

   The final datasets and checkpoints structure should look like (showing only relevant files):

   ```
   /datasets/laion-400m/webdataset-moments-filtered-encoded  # 2.7T containing 832 tar files inside
   /datasets/coco2014/val2014_30k.tsv  # 2M
   /datasets/coco2014/val2014_512x512_30k_stats.npz  # 31M
   /checkpoints/clip/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/blobs/9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4  # 3.7G
   /checkpoints/inception/pt_inception-2015-12-05-6726825d.pth  # 92M
   /checkpoints/sd/512-base-ema.ckpt  # 4.9G
   ```

   Exit the container.

6. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").  The created `${SQSH_DIR}/sd_ngc25.04_pyt.sqsh` file is needed for running the experiment with Slurm.

   ```bash
   enroot import -o ${SQSH_DIR}/sd_ngc25.04_pyt.sqsh dockerd://mlperf-inventec:sd_ngc25.04_pyt
   ```

7. Launch training with Slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000H100_1x8x32.sh
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

8. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `SLURM_MPI_TYPE`: `pmi2` seems be perform better than `pmix` in our experiments.
