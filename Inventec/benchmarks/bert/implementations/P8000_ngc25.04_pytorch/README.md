# MLPerf BERT Training on Inventec P8000/P9000 GPU Servers

Table of contents
-----------------

* [Environment Setup](#setup)
* [Step-by-step](#steps)
* [Known Issues](#issues)

<a name="setup"></a>
Environment Setup
------------------

Single GPU server case

* Machines

  - Slurm head node * 1, e.g. "p8000-head-1"
  - Slurm compute node * 1, e.g. "compute-h100-1"
  - Also refer to [README-NVIDIA.md](README-NVIDIA.md) for hardware and software requirements

* Storage

  - NFS mounted on both the head and the compute nodes: `/mnt` on "head-p8000-1" and "compute-h100-1"
  - Data (training data, checkpoints) in /mnt/mlperf/bert_data
  - Docker container file in /mnt/sqsh
  - Source code in /mnt/jkjung/training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch

Multiple GPU server case (TO-DO)

* ......

<a name="steps"></a>
Step-by-step
------------

1. Clone this repository on the compute node ("compute-h100-1").  You might want to replace `/mnt/jkjung` with your own directory.

   ```shell
   cd /mnt/jkjung
   git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Build the container on the compute node ("compute-h100-1").  You will have to use your NGC API key to pull the base pytorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.  Note that the docker image name "bert_ngc23.09_pyt" is different from that in NVIDIA's original implementation.

   ```shell
   cd training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch/
   docker build --pull -t mlperf-nvidia:bert_ngc23.09_pyt .
   ```

3. Prepare dataset on the compute node ("compute-h100-1").  You could refer to [README-NVIDIA.md](README-NVIDIA.md) for more details about the `prepare_data.sh` script.

   Start the container with the following command.  Note that the container (without the `--rm` flag) is not removed automatically.  You'll need to do `docker rm <CONTAINER ID>` manually.

   ```bash
   docker run -it --gpus=all --runtime=nvidia --ipc=host -v /mnt/mlperf/bert_data:/workspace/bert_data mlperf-nvidia:bert_ngc23.09_pyt
   ```

   Then run within the container:

   ```bash
   cd /workspace/bert
   ./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data --packed-data
   ```

   This script may take around 12 hours to complete.  If you have already downloaded the dataset and checkpoint files but would like to re-create the training data, you could add `--skip-download` option to the `prepare_data.sh` script.  That would save both network bandwidth and time.

   Once the script finishes running, you should verify `bert_data` has been correctly generated as follows.  Pay special attention to `phase1/model.ckpt-28252.pt` (the PyTorch checkpoint file converted from TensorFlow v1 checkpoint) which is generated in the last step of the `prepare_data.sh` script.

   ```
   /workspace/bert_data/
   ├── download
   │   ├── bert_reference_results_text_md5.txt
   │   ├── results4  [502 entries]
   │   ├── results_text.tar.gz
   ├── hdf5
   │   ├── eval
   │   │   ├── eval_all.hdf5
   │   │   └── part_eval_10k.hdf5
   │   ├── eval_varlength
   │   │   └── part_eval_10k.hdf5
   │   ├── training  [500 entries]
   │   └── training-4320
   │       ├── hdf5_4320_shards_uncompressed  [4320 entries]
   │       ├── hdf5_4320_shards_varlength  [4320 entries]
   │       └── hdf5_4320_shards_varlength_shuffled  [8640 entries]
   ├── packed_data  [8640 entries]
   ├── per_seqlen  [512 entries]
   ├── per_seqlen_parts  [500 entries]
   └── phase1
       ├── bert_config.json
       ├── model.ckpt-28252.data-00000-of-00001
       ├── model.ckpt-28252.index
       ├── model.ckpt-28252.meta
       ├── model.ckpt-28252.pt
       └── vocab.txt
   ```

4. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").  The created `/mnt/sqsh/bert_ngc23.09_pyt.sqsh` file is needed for running the experiment with slurm.

   ```bash
   enroot import -o /mnt/sqsh/bert_ngc23.09_pyt.sqsh dockerd://mlperf-nvidia:bert_ngc23.09_pyt
   ```

5. Launch training with slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000H100_1x8x48x1_pack.sh
   sbatch -w compute-h100-1 --time=${WALLTIME} run.sub
   ```

   Note:

   * Rename "compute-h100-1" if you are using a different slurm compute node.
   * It's a good idea, before you start `run.sub`, to verify that there is no process occupying the CPUs/GPUs/memory on the compute node.  For example, do `docker ps -a` and `docker rm <CONTAINER ID>` to remove all running/pending containers.
   * You could adjust experiment setting in the `env.sh` script.

   The above `sbatch` command would output the slurm batch job id.  You could track progress of the slurm batch job by checking the corresponding log file.

   ```bash
   tail -f slurm-<SLURM JOB ID>.out
   ```

6. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `--gres=gpu:8` needs to be explicitly added to all `srun` commands.  This is done in `run.sub`.
* Dedicating CPUs for WekaIO causing numa "cpu argument out of range" errors, e.g. `<84-95,180-191> is invalid`.
* `SLURM_MPI_TYPE`: `pmi2` works but `pmix` doesn't.
