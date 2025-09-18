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

2. Builder the container.  Note the docker image name "bert_ngc23.09_pyt" is different from that in NVIDIA's original inmplementation.

   ```shell
   root@compute-h100-1:/mnt/jkjung# cd training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch/
   root@compute-h100-1:/mnt/jkjung/training_results_v5.0/Inventec/benchmarks/bert/implementations/P8000_ngc23.09_pytorch# docker build --pull -t mlperf-nvidia:bert_ngc23.09_pyt .
   ```

3. Prepare dataset.

   To be continued...

