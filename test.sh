#!/bin/bash
export MASTER_PORT=29503
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "using master addr"$MASTER_ADDR
echo "using part "$SLURM_PARTITION
# TODO - minibatch size must be at least the no of local gpu?
echo "running python script"
echo "world size "$SLURM_NTASKS
echo "node name "$SLURMD_NODENAME
echo "rank? "$SLURM_PROCID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=info
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

if [[ $RANK -eq 0 ]]
then
#    export NCCL_SOCKET_IFNAME="veth69b725e"
    echo "hi"
fi

#export NCCL_SOCKET_IFNAME=
#torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=
python -u meta_test.py
#########python -u trainer.py --mode=fsdp --model=GPT13B --dtype="fp16" --vocab_size=50000 --block_size=2048 --batch_size=8 --activation="checkpoint" --cpu-offload=True --prefetch="noop" --version="pytorch"
#python dlrm_s_pytorch.py --use-gpu --print-time --dist-backend nccl --nepochs 2 --data-size 2048 --mini-batch-size 2 --debug-mode --arch-sparse-feature-size 2 --node-world-size ${SLURM_NTASKS} --rank ${SLURM_PROCID}
#README.md:41:torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_endpoint="localhost:5678" trainer.py
