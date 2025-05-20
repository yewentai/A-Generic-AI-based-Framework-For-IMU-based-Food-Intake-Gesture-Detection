#!/usr/bin/env python3
import subprocess

script_path = "train.py"

# === Available Options ===
# Datasets:        "DXI", "DXII", "FDI", "FDII"
# Models:          "CNN_LSTM", "TCN", "MSTCN", "ResNet_MLP", "ResNet_BiLSTM", "ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"
# Augmentations:   "None", "AM", "AP", "AR", "AS", "DM"
# Smoothing types: "MSE", "L1", "HUBER", "JS", "TV", "SEC_DIFF", "EMD", "None"

# Define only the combinations you want to run
combinations = [
    {"dataset": "FDI", "model": "MSTCN", "augmentation": "None", "smoothing": "None"},
    # Add more specific combos here
]

for combo in combinations:
    cmd = [
        "python3",
        script_path,
        "--dataset",
        combo["dataset"],
        "--model",
        combo["model"],
        "--augmentation",
        combo["augmentation"],
        "--smoothing",
        combo["smoothing"],
    ]
    print(f"\n=== Running: {combo} ===\n")
    subprocess.run(cmd)
