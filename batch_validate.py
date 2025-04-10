import os
import subprocess

# Get all result version folders
result_root = "result"
versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
versions.sort()

for version in versions:
    print(f"\n=== Validating Version: {version} (Mirror=False) ===\n")
    subprocess.run(["python", "dl_validate.py", version, "no_mirror"])

    print(f"\n=== Validating Version: {version} (Mirror=True) ===\n")
    subprocess.run(["python", "dl_validate.py", version, "mirror"])
