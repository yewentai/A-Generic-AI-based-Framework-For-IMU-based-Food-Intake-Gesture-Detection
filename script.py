#!/usr/bin/env python3
import os

ROOT_DIR = "result"


def remove_validation_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("validation"):
                path = os.path.join(dirpath, fname)
                try:
                    os.remove(path)
                    print(f"Deleted {path}")
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")


if __name__ == "__main__":
    remove_validation_files(ROOT_DIR)
    print("Done.")
