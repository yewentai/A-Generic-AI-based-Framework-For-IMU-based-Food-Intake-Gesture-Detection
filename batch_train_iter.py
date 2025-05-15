#!/usr/bin/env python3
import subprocess

script_path = "train.py"

# === Settings ===
# options: "DXI", "DXII", "FDI", "FDII"
dataset = "DXI"
# options:"CNN_LSTM", "TCN", "MSTCN", "AccNet", "ResNet_BiLSTM", "ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"
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
