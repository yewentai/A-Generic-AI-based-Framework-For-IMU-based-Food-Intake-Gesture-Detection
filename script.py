#!/usr/bin/env python3
import os
import re

ROOT_DIR = "result"
TARGET_NAME = "training_config.json"
INSERT_KEY = '"hand_separation"'
INSERT_LINE = "true"  # lowercase for JSON boolean


def inject_line_after_dataset(path):
    """Reads a JSON file, inserts the hand_separation line after dataset, and overwrites it."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    pattern = re.compile(r'^\s*"dataset"\s*:')
    for line in lines:
        out_lines.append(line)
        if pattern.match(line):
            # capture indentation to align the new line
            indent = re.match(r"^(\s*)", line).group(1)
            # ensure comma after dataset value (if missing)
            if not line.rstrip().endswith(","):
                out_lines[-1] = line.rstrip() + ",\n"
            # insert the new line
            out_lines.append(f"{indent}{INSERT_KEY}: {INSERT_LINE},\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        if TARGET_NAME in filenames:
            cfg_path = os.path.join(dirpath, TARGET_NAME)
            print(f"Updating {cfg_path}...")
            inject_line_after_dataset(cfg_path)
    print("All done.")


if __name__ == "__main__":
    main()
