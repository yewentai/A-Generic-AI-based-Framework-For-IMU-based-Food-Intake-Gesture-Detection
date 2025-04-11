#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Project Directory Cleanup Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-03
Description : This script recursively traverses specified directories to find and
              remove empty subdirectories, keeping the project workspace tidy.
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


def main():
    # List the top-level directories to scan
    parent_dirs = ["result"]
    for parent in parent_dirs:
        if os.path.exists(parent):
            print(f"Scanning directory: {parent}")
            remove_empty_subdirs(parent)
        else:
            print(f"Directory '{parent}' does not exist.")


if __name__ == "__main__":
    main()
