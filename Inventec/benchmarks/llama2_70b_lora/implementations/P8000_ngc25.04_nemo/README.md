# MLPerf LLama2-70B LoRA PyTorch Training on Inventec P8000/P9000 GPU Servers

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
  - Data (training data, checkpoints) in /mnt/mlperf/llama2_70b_lora_data
  - Docker container file in /mnt/sqsh
  - Source code in /mnt/jkjung/training_results_v5.0/Inventec/benchmarks/llama2_70b_lora/implementations/P8000_ngc25.04_nemo

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
   cd training_results_v5.0/Inventec/benchmarks/llama2_70b_lora/implementations/P8000_ngc25.04_nemo/
   docker build -t mlperf-nvidia:llama2_70b_lora_pyt .
   ```

3. Download dataset, download model, and do preprocessing on the compute node ("compute-h100-1").

   Start the container with the following command.

   ```bash
   docker run -it --gpus=all --rm --gpus all --network=host --ipc=host -v /mnt/mlperf/llama2_70b_lora_data:/data mlperf-nvidia:llama2_70b_lora_pyt
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
   enroot import -o /mnt/sqsh/llama2_70b_lora_pyt.sqsh dockerd://mlperf-nvidia:llama2_70b_lora_pyt
   ```

5. Launch training with slurm on the *head* node ("head-p8000-1").  Navigate to the directory where `run.sub` is stored and execute the following.

   ```bash
   source env.sh
   source config_P8000_1x8x1xtp8pp1cp1.sh
   sbatch -w compute-h100-1 -t ${WALLTIME} run.sub
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
