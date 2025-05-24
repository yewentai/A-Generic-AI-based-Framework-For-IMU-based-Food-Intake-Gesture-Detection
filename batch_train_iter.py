#!/usr/bin/env python3
"""
===============================================================================
IMU Segmentation Batch Training Launcher
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-24
Description : This script automates batch training for IMU segmentation experiments,
              iterating over multiple smoothing types for a given dataset, model,
              and augmentation setting. Each configuration is executed by calling
              train.py with the appropriate arguments.
===============================================================================
"""

import subprocess

script_path = "train.py"

# === Settings ===
# options: "DXI", "DXII", "FDI", "FDII"
dataset = "DXI"
# options:"CNN_LSTM", "TCN", "MSTCN", "ResNet_MLP", "ResNet_BiLSTM", "ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"
model = "MSTCN"
# options: "None", "HandMirroring", "AxisPermutation", "PlanarRotation", "SpatialOrientation"
augmentation = "None"
# options: "MSE", "L1", "HUBER", "JS", "TV", "SEC_DIFF", "EMD"
smoothing_types = ["MSE", "L1", "HUBER", "JS", "TV", "SEC_DIFF", "EMD"]
# Path to your training script

for smoothing_type in smoothing_types:
    cmd = [
        "python3",
        script_path,
        "--dataset",
        dataset,
        "--model",
        model,
        "--augmentation",
        augmentation,
        "--smoothing",
        smoothing_type,
    ]
    print(f"\n=== Running: {smoothing_type} ===\n")
    subprocess.run(cmd)
