#!/bin/bash
#SBATCH --account=lp-emedia
#SBATCH --cluster=wice
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yewentai126@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gpus-per-node=1  # Adjust this if using multiple GPUs
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --output=logs/ddp_train_%j.out

# Load Miniconda
export PATH=$VSC_DATA/miniconda3/bin:$PATH
cd $VSC_DATA/thesis
source ~/.bashrc  # Ensures Conda is properly initialized
conda activate torch  # Activates the Conda environment

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)  # Use the current node as master
export MASTER_PORT=29500  # Choose an open port

# (Optional) Debugging flags
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1  # Helps debugging CUDA errors

# Launch the training script with torchrun for DDP (Distributed Data Parallel) support
torchrun --nnodes=1 --nproc_per_node=1 train_hpc.py
