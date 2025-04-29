#!/usr/bin/env python3
import os
import re


def inject_downsample(path):
    """Read a JSON file as text, insert the downsample_factor line after num_classes, and overwrite it."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    pattern = re.compile(r'^\s*"num_classes"\s*:')
    for line in lines:
        out.append(line)
        if pattern.match(line):
            # capture indentation of the matched line:
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(f'{indent}"downsample_factor": 4,\n')
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


if __name__ == "__main__":
    root_dir = "result"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "config.json" in filenames:
            cfg_path = os.path.join(dirpath, "config.json")
            print(f"Updating {cfg_path}...")
            inject_downsample(cfg_path)
    print("Done.")
