#!/usr/bin/env python3
import os

ROOT = "result"  # or change this to any base directory you like
OLD_NAME = "train_stats.npy"
NEW_NAME = "training_stats.npy"


def rename_configs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if OLD_NAME in filenames:
            old_path = os.path.join(dirpath, OLD_NAME)
            new_path = os.path.join(dirpath, NEW_NAME)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} → {new_path}")
            except Exception as e:
                print(f"⚠️  Could not rename {old_path}: {e}")


if __name__ == "__main__":
    rename_configs(ROOT)
    print("Done.")
