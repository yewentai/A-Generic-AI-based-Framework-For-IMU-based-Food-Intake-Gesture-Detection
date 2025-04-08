#!/bin/bash
#SBATCH --account=lp-emedia                   # Project account to charge resource usage
#SBATCH --cluster=wice                        # Name of the cluster to submit the job to
#SBATCH --mail-type=BEGIN,END,FAIL            # Send email notifications when job begins, ends, or fails
#SBATCH --mail-user=yewentai126@gmail.com     # Email address to receive notifications
#SBATCH --nodes=1                             # Number of compute nodes to allocate (1 node)
#SBATCH --ntasks-per-node=16                  # Number of CPU tasks (or processes) to run on the node (max: 72)
#SBATCH --gpus-per-node=4                     # Request 2 GPUs per node (for parallel GPU training)
#SBATCH --partition=gpu_h100                  # Use the 'gpu_h100' partition (nodes with H100 GPUs)
#SBATCH --time=8:00:00                        # Maximum allowed runtime: 3 days
#SBATCH --output=logs/ddp_train_%j.out        # Standard output and error log file (%j is the job ID)


# Load Miniconda
export PATH=$VSC_DATA/miniconda3/bin:$PATH
cd $VSC_DATA/thesis
source ~/.bashrc
conda activate torch

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Optional debugging flags
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Launch the training script with torchrun for DDP support
torchrun --nnodes=1 --nproc_per_node=2 dl_train_hpc.py
