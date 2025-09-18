# MLPerf BERT Training on Inventec P8000/P9000 GPU Servers

Table of contents
-----------------

* [Environment Setup](#setup)
* [Step-by-step](#steps)

<a name="setup"></a>
Environment Setup
------------------

Single GPU server case

* Machines

  - Slurm head node * 1, e.g. "p8000-head-1"
  - Slurm compute node * 1, e.g. "compute-h100-1"
  - Also refer to [README-NVIDIA.md](README-NVIDIA.md) for hardware and software requirements

* Storage

  - NFS mounted on the compute node: `/mnt` on "compute-h100-1"
  - Data (training data, checkpoints) in /mnt/mlperf/bert_data
  - Source code in /mnt/jkjung/training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch

Multiple GPU server case (TO-DO)

......

<a name="steps"></a>
Step-by-step
------------

1. Clone this repository.  You might want to replace `/mnt/jkjung` with your own directory.

   ```shell
   root@compute-h100-1:~# cd /mnt/jkjung
   root@compute-h100-1:/mnt/jkjung# git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Builder the container.  You will have to use your NGC API key to pull the base pytorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.  Note that the docker image name "bert_ngc23.09_pyt" is different from that in NVIDIA's original inmplementation.

   ```shell
   root@compute-h100-1:/mnt/jkjung# cd training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch/
   root@compute-h100-1:/mnt/jkjung/training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch# docker build --pull -t mlperf-nvidia:bert_ngc23.09_pyt .
   ```

3. Prepare dataset.  You could refer to [README-NVIDIA.md](README-NVIDIA.md) for more details about the `prepare_data.sh` script.

   Start the container with the following command.  Note that the container (without the `--rm` flag) is not removed automatically.  You'll need to do `docker rm <CONTAINER ID>` manually.

   ```bash
   root@compute-h100-1:/mnt/jkjung/training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch# docker run -it --gpus=all --runtime=nvidia --ipc=host -v /mnt/mlperf/bert_data:/workspace/bert_data mlperf-nvidia:bert_ngc23.09_pyt
   ```

   Then run within the container:

   ```bash
   cd /workspace/bert
   ./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data --packed-data
   ```

   This script may take around 24 hours to complete.  Once the script finishes running, you should verify `bert_data` has been correctly generated as follows.  Pay special attention to `phase1/model.ckpt-28252.pt` (PyTorch checkpoint file converted from the original TensorFlow v1 checkpoint) which is generated in the last step of the `prepare_data.sh` script.

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

4. TODO: Launch training.

   Navigate to the directory where `run.sub` is stored.

   The launch command structure:

   ```bash
   export EVALDIR="/path/to/your/data/hdf5/eval_varlength"
   export DATADIR_PHASE2="/path/to/your/data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
   export DATADIR_PHASE2_PACKED="/path/to/your/data/packed_data"
   export CHECKPOINTDIR_PHASE1="/path/to/your/data/phase1"
   export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
   export CONT=<docker/registry>/mlperf-nvidia:language_model-pyt
   source config_DGXH100_1x8x48x1_pack.sh  # select config and source it
   sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
   ```
