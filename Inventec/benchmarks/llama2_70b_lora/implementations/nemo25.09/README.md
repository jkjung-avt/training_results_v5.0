# MLPerf LLama2-70B LoRA Training on Inventec P8000/P9000 GPU Servers

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
  - Data (training data, validation data, checkpoints) in ${LLAMA2_DATA_DIR} (`/hps/data/mlperf_training/llama2`)
  - Docker container SquashFS file in ${SQSH_DIR} (`/hps/sqsh`)

<a name="steps"></a>
Step-by-step
------------

1. Set environment variables.  Replace the paths below with your own if necessary.

   ```shell
   export USER_DIR=/mnt/jkjung
   export LLAMA2_DATA_DIR=/hps/data/mlperf_training/llama2
   export SQSH_DIR=/hps/sqsh
   ```

   Then clone this repository on the compute node ("compute-h100-1").

   ```shell
   cd ${USER_DIR}
   git clone https://github.com/jkjung-avt/training_results_v5.0.git
   ```

2. Build the container on the compute node ("compute-h100-1").  You will have to use your NGC API key to pull the base PyTorch docker image, e.g. `docker login nvcr.io` or use `~/.config/enroot/.credentials`.

   ```shell
   cd training_results_v5.0/Inventec/benchmarks/llama2_70b_lora/implementations/nemo25.09/
   docker build -t mlperf-inventec:llama2_70b_lora_nemo25.09 .
   ```

3. Download dataset, download model, and do preprocessing on the compute node ("compute-h100-1").

   Start the container with the following command.

   ```bash
   docker run -it --rm --gpus=all --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${LLAMA2_DATA_DIR}:/data mlperf-inventec:llama2_70b_lora_nemo25.09
   ```

   Then run within the container, under the /workspace/ft-llm directory:

   ```bash
   python scripts/download_dataset.py --data_dir /data/gov_report
   python scripts/download_model.py --model_dir /data/model
   ```

   This 1st script takes less than 1 minute.  The second script could take up to 30 minutes.  After both scripts finish, you should see the following files in the `/data` directory:

   ```
   /data
   ├── gov_report
   │   ├── train.npy
   │   └── validation.npy
   └── model
       ├── context
       │   ├── io.json
       │   ├── model.yaml
       │   └── nemo_tokenizer
       └── weights
           ├── common.pt
           ├── metadata.json
           ├── module.decoder.final_layernorm._extra_state
           ├── module.decoder.final_layernorm.weight
           ├── module.decoder.layers.mlp.linear_fc1._extra_state
           ├── module.decoder.layers.mlp.linear_fc1.layer_norm_weight
           ├── module.decoder.layers.mlp.linear_fc1.weight
           ├── module.decoder.layers.mlp.linear_fc2._extra_state
           ├── module.decoder.layers.mlp.linear_fc2.weight
           ├── module.decoder.layers.self_attention.core_attention._extra_state
           ├── module.decoder.layers.self_attention.linear_proj._extra_state
           ├── module.decoder.layers.self_attention.linear_proj.weight
           ├── module.decoder.layers.self_attention.linear_qkv._extra_state
           ├── module.decoder.layers.self_attention.linear_qkv.layer_norm_weight
           ├── module.decoder.layers.self_attention.linear_qkv.weight
           ├── module.embedding.word_embeddings.weight
           └── module.output_layer.weight
   ```

   Exit the container.

4. Create the SquashFS file from the docker image on the compute node ("compute-h100-1").

   ```bash
   enroot import -o ${SQSH_DIR}/llama2_70b_lora_nemo25.09.sqsh dockerd://mlperf-inventec:llama2_70b_lora_nemo25.09
   ```

5. Launch training with Slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000G6H100_1x8x1xtp8pp1cp1.sh
   sbatch -w compute-h100-1 -t ${WALLTIME} run.sub
   ```

   Note:

   * Rename "compute-h100-1" if you are using a different Slurm compute node.
   * It's a good idea, before you start `run.sub`, to verify that there is no process occupying the CPUs/GPUs/memory on the compute node.  For example, do `docker ps -a` and `docker rm <CONTAINER ID>` to remove all running/pending containers.
   * You could adjust experiment setting in the `env.sh` script.

   The above `sbatch` command would output the Slurm batch job id.  You could track progress of the Slurm batch job by checking the corresponding log file.

   ```bash
   tail -fn +1 slurm-<SLURM JOB ID>.out
   ```

6. Check experiment results in the `results` folder.

<a name="issues"></a>
Known Issues
------------

* `SLURM_MPI_TYPE`: `pmi2` seems be perform better than `pmix` in our experiments.
