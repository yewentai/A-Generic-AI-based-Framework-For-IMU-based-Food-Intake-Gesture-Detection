#!/bin/bash
#SBATCH --account=lp-emedia                   # Project account to charge resource usage
#SBATCH --cluster=wice                        # Name of the cluster to submit the job to
#SBATCH --mail-type=BEGIN,END,FAIL            # Send email notifications on job start, end, or failure
#SBATCH --mail-user=yewentai126@gmail.com     # Email address for notifications
#SBATCH --nodes=1                             # Number of compute nodes to allocate (Maximum 4 nodes)
#SBATCH --ntasks-per-node=18                  # Number of CPU tasks (or processes) to run per node (Maximum 72 cores)
#SBATCH --gpus-per-node=4                     # Number of GPUs to request per node (Maximum 4 GPUs)
#SBATCH --partition=gpu_a100                  # Partition to use (A100 GPUs)
#SBATCH --time=24:00:00                       # Maximum wall time (Maximum 3 days)
#SBATCH --output=logs_hpc/ddp_train_%j.output     # Standard output and error log file (%j is the job ID)

# Load Miniconda
export PATH=$VSC_DATA/miniconda3/bin:$PATH
cd $VSC_DATA/thesis
source ~/.bashrc
conda activate torch

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# (Optional) Debugging flags
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Launch the training script with torchrun for DDP (Distributed Data Parallel) support
torchrun train.py --distributed