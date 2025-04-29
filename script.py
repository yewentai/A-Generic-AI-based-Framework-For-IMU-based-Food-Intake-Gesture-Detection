#!/usr/bin/env python3
import os
import json
import re

# Path to your 'result' directory
ROOT = "result"

# A simple sanity regex for timestamp folder names (e.g. 202504211732)
TIMESTAMP_RE = re.compile(r"^\d{12}$")


def build_new_name(cfg):
    # base components
    parts = [cfg.get("dataset", "UNKNOWN"), cfg.get("hand", "UNKNOWN"), cfg.get("model", "UNKNOWN")]
    # all boolean keys we care about:
    #   any config entry whose value is exactly True
    for key, val in cfg.items():
        if isinstance(val, bool) and val:
            parts.append(key)
    # sanitize each part to avoid bad characters
    safe = [re.sub(r"\s+", "", p) for p in parts]
    return "_".join(safe)


def main():
    for entry in os.listdir(ROOT):
        old_path = os.path.join(ROOT, entry)
        if not os.path.isdir(old_path):
            continue
        if not TIMESTAMP_RE.match(entry):
            # skip non-timestamp directories
            continue

        cfg_path = os.path.join(old_path, "config.json")
        if not os.path.isfile(cfg_path):
            print(f"⚠️  Skipping {old_path}: no config.json found")
            continue

        # Load config.json
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Construct new directory name
        new_name = build_new_name(cfg)
        new_path = os.path.join(ROOT, new_name)

        # If target exists, append a counter
        counter = 1
        candidate = new_path
        while os.path.exists(candidate):
            candidate = f"{new_path}_{counter}"
            counter += 1
        new_path = candidate

        # Perform rename
        print(f"Renaming:\n  {old_path}\n→ {new_path}\n")
        os.rename(old_path, new_path)

    print("All done.")


if __name__ == "__main__":
    main()
