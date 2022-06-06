#!/bin/bash

export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${HOME}/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO
export NCCL_ALGO=ring

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export LOCAL_RANK=${SLURM_LOCALID}
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

echo $MASTER_ADDR
echo $RANK
#conda activate pt
exit
python -u trainer.py --mode=fsdp --model=GPT13B --dtype="fp16" --vocab_size=50000 --block_size=2048 --batch_size=8 --activation="checkpoint" --cpu-offload=True --prefetch="noop" --version="pytorch"
