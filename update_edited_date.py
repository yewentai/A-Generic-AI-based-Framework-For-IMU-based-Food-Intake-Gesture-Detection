#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Update Script for Edited Date in Python Files
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-04-10
Description : This script automatically updates the "Edited" date in the header
              of Python files staged for commit in a Git repository. It scans
              for Python files, identifies the header, and modifies the date.
===============================================================================
"""

import subprocess
from datetime import date


def get_changed_python_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    files = result.stdout.strip().split("\n")
    return [f for f in files if f.endswith(".py")]


def update_header_date(file_path):
    today = date.today().strftime("%Y-%m-%d")
    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for line in lines:
            if line.strip().startswith("Edited"):
                line = f"Edited      : {today}\n"
            f.write(line)


if __name__ == "__main__":
    py_files = get_changed_python_files()
    for file in py_files:
        try:
            update_header_date(file)
            print(f"Updated: {file}")
        except Exception as e:
            print(f"Error updating {file}: {e}")
    print("Changed files:", py_files)
