#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project Directory Cleanup Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-14
Description : This script recursively scans target project directories to:
              - Remove empty subdirectories
              - Delete all 'analysis' folders and their contents
              - (Optional) Remove specific validation files
              - (Optional) Rename validation output files

              It helps maintain a clean directory structure for experiments.
===============================================================================
"""


import os
import json


def remove_empty_subdirs(root):
    """
    Recursively traverse the directory tree under `root` (bottom-up)
    and remove any directories that are empty, except for the root folder.
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Skip the root folder itself
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue

        # If no subdirectories and no files, delete the folder
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty directory: {dirpath}")
            except Exception as e:
                print(f"Failed to delete {dirpath}: {e}")


def remove_specified_files(root):
    """
    Traverse the directory tree under `root` and remove only the files
    'validation_stats_Clemson.json' and 'validation_stats_Clemson.npy'.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # Check if the specified files exist and delete them
        for filename in filenames:
            if filename in ["validation_stats_Clemson.json", "validation_stats_Clemson.npy"]:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete file {file_path}: {e}")


def remove_analysis_folders(root):
    """
    Traverse the directory tree under `root` and remove any folder named 'analysis' along with its contents.
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for dirname in dirnames:
            if dirname == "analysis":
                folder_path = os.path.join(dirpath, dirname)
                try:
                    # Remove the folder and its contents
                    for root_dir, subdirs, files in os.walk(folder_path, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root_dir, file))
                        for subdir in subdirs:
                            os.rmdir(os.path.join(root_dir, subdir))
                    os.rmdir(folder_path)
                    print(f"Deleted folder and its contents: {folder_path}")
                except Exception as e:
                    print(f"Failed to delete folder {folder_path}: {e}")


def rename_files(root):
    """
    Traverse the directory tree under `root` and rename files by removing postfixes like '_dxi'
    or '_fdi' in 'validation_stats_dxi.json', 'validation_stats_dxi.npy',
    'validation_stats_fdi.json', and 'validation_stats_fdi.npy'.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("validation_stats_") and ("_dxi" in filename or "_fdi" in filename):
                old_file_path = os.path.join(dirpath, filename)
                new_file_name = filename.replace("_dxi", "").replace("_fdi", "")
                new_file_path = os.path.join(dirpath, new_file_name)
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")


def remove_both_substring(root):
    """
    Traverse the directory tree under `root` and remove all occurrences
    of 'BOTH_' within directory names.
    """
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for dirname in dirnames:
            if "BOTH_" in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_name = dirname.replace("BOTH_", "")
                new_path = os.path.join(dirpath, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: {old_path} to {new_path}")
                except Exception as e:
                    print(f"Failed to rename directory {old_path}: {e}")


def update_training_configs(root):
    """
    Traverse the directory tree under `root` and ensure every training_config.json
    has a 'selected_channels' field. Models in CNN_LSTM, TCN, MSTCN get channels [0-5].
    Models in AccNet, ResNetBiLSTM get channels [0-2].
    """
    for dirpath, _, filenames in os.walk(root):
        if "training_config.json" in filenames:
            file_path = os.path.join(dirpath, "training_config.json")
            try:
                with open(file_path, "r") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue

            if "selected_channels" not in cfg:
                model_name = cfg.get("model", os.path.basename(dirpath))
                # Determine channels based on model category
                if any(key in model_name for key in ["CNN_LSTM", "TCN", "MSTCN"]):
                    cfg["selected_channels"] = list(range(6))
                elif any(
                    key in model_name
                    for key in ["AccNet", "ResNetBiLSTM", "ResNetBiLSTM_FTFull", "ResNetBiLSTM_FTHead"]
                ):
                    cfg["selected_channels"] = list(range(3))
                else:
                    # Unknown model type; skip
                    continue

                # Write updated config back
                try:
                    with open(file_path, "w") as f:
                        json.dump(cfg, f, indent=4)
                    print(f"Updated selected_channels in {file_path}")
                except Exception as e:
                    print(f"Failed to write {file_path}: {e}")


def main():
    # List the top-level directories to scan
    parent_dirs = ["results"]
    for parent in parent_dirs:
        if os.path.exists(parent):
            print(f"Scanning directory: {parent}")
            # Uncomment the operations you want to perform:
            # remove_analysis_folders(parent)
            remove_empty_subdirs(parent)
            # remove_specified_files(parent)
            # rename_files(parent)
            # remove_both_substring(parent)
            # update_training_configs(parent)
        else:
            print(f"Directory '{parent}' does not exist.")


if __name__ == "__main__":
    main()
