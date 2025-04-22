#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Git Pre-Commit Hook: Auto Update 'Edited' Date in Python File Headers
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
P25-04-22
Description : This script is intended to be used as a Git pre-commit hook.
              It scans all staged Python files, finds the 'Edited' date field
              in the header, and updates it to the current date if present.
              It re-adds the modified files to the staging area to ensure the
              changes are included in the commit.
===============================================================================
"""

import os
import re
import subprocess
from datetime import datetime

# Get the list of staged files
staged_files = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()

# Regex pattern to match the Edited date line
edit_date_pattern = re.compile(r"(Edited\s+: )(20\d{2}-\d{2}-\d{2})")

# Files to update
py_files = [f for f in staged_files if f.endswith(".py") and os.path.isfile(f)]

today = datetime.now().strftime("%Y-%m-%d")

for filepath in py_files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if "Edited" in content:
        new_content, count = edit_date_pattern.subn(rf"\1{today}", content)
        if count > 0:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            subprocess.run(["git", "add", filepath])
