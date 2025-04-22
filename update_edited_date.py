#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_edited_date.py

When run in a pre-commit hook, this script will:
 1. Use `git diff --cached --name-only` to list staged files.
 2. Filter for `.py` files.
 3. Update the 'Edited' or 'Last Edited' header date to today.
 4. git-add any files that changed.

Usage (in pre-commit hook):
    python3 ./update_edited_date.py
"""

import subprocess
import re
import sys
from datetime import datetime

# Today’s date in YYYY-MM-DD
TODAY = datetime.now().strftime("%Y-%m-%d")

# Match lines like:
#   # Edited      : 2025-04-14
#   # Last Edited : 2025-04-14
PAT = re.compile(r"^(?P<prefix>\s*#\s*(?:Edited|Last Edited)\s*:\s*)(\d{4}-\d{2}-\d{2})(\s*)$")


def get_staged_py_files():
    """Return a list of staged .py files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], stdout=subprocess.PIPE, check=True, text=True
    )
    return [f for f in result.stdout.splitlines() if f.endswith(".py")]


def update_file(path):
    """Update the Edited date in the given file. Return True if modified."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return False

    changed = False
    for i, line in enumerate(lines):
        m = PAT.match(line)
        if m:
            new_line = f"{m.group('prefix')}{TODAY}{m.group(3)}\n"
            if new_line != line:
                lines[i] = new_line
                changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def main():
    files = get_staged_py_files()
    if not files:
        return

    modified = []
    for path in files:
        if update_file(path):
            modified.append(path)

    if modified:
        # Re-stage the updated files
        subprocess.run(["git", "add"] + modified, check=True)
        print(f"Updated Edited date in: {', '.join(modified)}")


if __name__ == "__main__":
    main()
