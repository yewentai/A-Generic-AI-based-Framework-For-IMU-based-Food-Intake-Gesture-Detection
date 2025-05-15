#!/usr/bin/env python3
import subprocess


script_path = "train.py"

# === Settings ===
# options: "DXI", "DXII", "FDI", "FDII"
datasets = ["DXI"]
# options:"CNN_LSTM", "TCN", "MSTCN", "AccNet", "ResNet_BiLSTM", "ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"
models = ["ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"]
# options: "None", "HandMirroring", "AxisPermutation", "PlanarRotation", "SpatialOrientation"
augmentations = ["None"]
# options: "MSE", "L1", "HUBER", "JS", "TV", "SEC_DIFF", "EMD"
smoothing_types = ["L1"]
# Path to your training script

for dataset in datasets:
    for model in models:
        for augmentation in augmentations:
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
                subprocess.run(cmd)
