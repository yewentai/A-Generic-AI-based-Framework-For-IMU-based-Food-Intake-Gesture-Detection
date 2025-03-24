#!/bin/bash
#SBATCH --account=lp-emedia
#SBATCH --cluster=wice
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yewentai126@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=2  # Request 2 GPUs on one node
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddp_train_%j.out

# Load Miniconda
export PATH=$VSC_DATA/miniconda3/bin:$PATH
cd $VSC_DATA/thesis
source ~/.bashrc
conda activate torch

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)   # Use the current node as master
export MASTER_PORT=29500         # Choose a free port

# Optional debugging flags
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Launch the training script with torchrun for DDP support
torchrun --nnodes=1 --nproc_per_node=2 train_hpc.py
