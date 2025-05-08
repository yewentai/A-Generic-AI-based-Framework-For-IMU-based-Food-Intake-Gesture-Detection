#!/usr/bin/env python3
import subprocess

# Define all smoothing types to try
smoothing_types = ["MSE", "HUBER", "KL", "JS", "TV", "SEC_DIFF", "EMD", "SPECTRAL"]

# Path to your training script
script_path = "dl_train.py"

# Whether to use distributed training
use_distributed = False  # Set to True if needed

for smoothing in smoothing_types:
    print(f"\n--- Running with smoothing: {smoothing} ---\n")
    cmd = ["python3", script_path, "--smoothing", smoothing]
    if use_distributed:
        cmd = ["torchrun", "--nproc_per_node=2"] + cmd  # Adjust for your setup

    # Launch the training script with the current smoothing method
    subprocess.run(cmd)
