#!/bin/bash
#SBATCH --account=lp-emedia                   # Account to charge resources
#SBATCH --cluster=wice                        # Cluster name
#SBATCH --mail-type=BEGIN,END,FAIL            # Send email notifications on job start, end, or failure
#SBATCH --mail-user=yewentai126@gmail.com     # Email address for notifications
#SBATCH --nodes=1                             # Request 1 node (Maximum 4 nodes)
#SBATCH --ntasks-per-node=16                  # Number of CPU cores to use per node (Maximum 72 cores)
#SBATCH --gpus-per-node=1                     # Request 1 GPU per node (Maximum 4 GPUs)
#SBATCH --partition=gpu_h100                  # Partition to use (H100 GPUs)
#SBATCH --time=24:00:00                       # Maximum runtime for the job (24 hours)

# Update PATH for Miniconda and activate your Conda environment
export PATH=/$VSC_DATA/miniconda3/bin:$PATH   # Add Miniconda to PATH
cd $VSC_DATA/thesis                           # Navigate to your project directory
source activate torch                         # Activate your Conda environment for PyTorch

# (Optional) Set NCCL debugging level for distributed training troubleshooting
export NCCL_DEBUG=INFO

# Launch the training script with torchrun for DDP (Distributed Data Parallel) support
torchrun --nnodes=1 --nproc_per_node=1 train_mstcn_fd_kfold.py
