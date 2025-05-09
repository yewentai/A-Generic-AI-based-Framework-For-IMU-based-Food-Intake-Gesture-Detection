#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project Directory Cleanup Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-02
Description : This script recursively scans target project directories to:
              - Remove empty subdirectories
              - Delete all 'analysis' folders and their contents
              - (Optional) Remove specific validation files
              - (Optional) Rename validation output files

              It helps maintain a clean directory structure for experiments.
===============================================================================
"""


import os


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
    Traverse the directory tree under `root` and rename specific files based on patterns.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Rename validate_rotation.json and .npy
            if filename == "validation_rotation.npy":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_stats_rotation.npy")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")

            # Rename validate_mirror.json and .npy
            elif filename == "validation_mirroring.npy":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_stats_mirroring.npy")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")


def main():
    # List the top-level directories to scan
    parent_dirs = ["./result"]
    for parent in parent_dirs:
        if os.path.exists(parent):
            print(f"Scanning directory: {parent}")
            # remove_analysis_folders(parent)
            remove_empty_subdirs(parent)
            # remove_specified_files(parent)
            # rename_files(parent)
        else:
            print(f"Directory '{parent}' does not exist.")


if __name__ == "__main__":
    main()
