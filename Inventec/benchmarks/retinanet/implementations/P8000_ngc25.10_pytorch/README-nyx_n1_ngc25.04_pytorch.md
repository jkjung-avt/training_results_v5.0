## Steps to launch training

### nyx_n1_ngc25.04_pytorch

Launch configuration and system-specific hyperparameters for the
nyx_n1_ngc25.04_pytorch submission are in the
`benchmarks/retinanet/implementations/nyx_n1_ngc25.04_pytorch/config_DGXB200_001x08x032.sh` script.

Steps required to launch training for nyx_n1_ngc25.04_pytorch.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXB200_001x08x032.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
