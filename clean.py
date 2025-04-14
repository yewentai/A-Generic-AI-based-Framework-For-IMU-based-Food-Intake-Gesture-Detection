#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project Directory Cleanup Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-14
Description : This script recursively traverses specified directories to find and
              remove empty subdirectories, 'analysis' folders, and specific files.
===============================================================================
"""

import os
import shutil


def remove_empty_subdirs(root):
    """
    Recursively traverse the directory tree under `root` (bottom-up)
    and remove any directories that are empty, except for the root folder.
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # Skip the root folder itself
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue

        # Remove 'analysis' folders, even if they are not empty
        if os.path.basename(dirpath) == "analysis":
            try:
                shutil.rmtree(dirpath)
                print(f"Deleted 'analysis' folder: {dirpath}")
            except Exception as e:
                print(f"Failed to delete 'analysis' folder {dirpath}: {e}")

        # If no subdirectories and no files, delete the folder
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty directory: {dirpath}")
            except Exception as e:
                print(f"Failed to delete {dirpath}: {e}")


def remove_specified_files(root):
    """
    Traverse the directory tree under `root` and remove the specific files.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # Check if the specified files exist and delete them
        for filename in filenames:
            if filename in ["validate_stats.npy", "validate_stats.json"] or filename.endswith(".log"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete file {file_path}: {e}")


def rename_files(root):
    """
    Traverse the directory tree under `root` and rename specific files based on patterns.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Rename validate_rotation.json and .npy
            if filename == "validate_rotation.json":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_rotation.json")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")
            elif filename == "validate_rotation.npy":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_rotation.npy")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")

            # Rename validate_mirror.json and .npy
            elif filename == "validate_mirror.json":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_mirroring.json")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")
            elif filename == "validate_mirror.npy":
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, "validation_mirroring.npy")
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Failed to rename file {old_file_path}: {e}")


def main():
    # List the top-level directories to scan
    parent_dirs = ["result"]
    for parent in parent_dirs:
        if os.path.exists(parent):
            print(f"Scanning directory: {parent}")
            remove_empty_subdirs(parent)
            remove_specified_files(parent)
            rename_files(parent)
        else:
            print(f"Directory '{parent}' does not exist.")


if __name__ == "__main__":
    main()
