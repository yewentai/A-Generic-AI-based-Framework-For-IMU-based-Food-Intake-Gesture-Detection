import os
import subprocess

# Root directory for all result folders
result_root = "result"
versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
versions.sort()

for version in versions:
    version_dir = os.path.join(result_root, version)

    # --------------------------
    # Validate without mirror
    # --------------------------
    stats_no_mirror = os.path.join(version_dir, "validate_stats.npy")
    if os.path.exists(stats_no_mirror) and os.path.getsize(stats_no_mirror) > 2048:
        print(f"\n[SKIP] {version} (no_mirror) already validated.")
    else:
        print(f"\n=== Validating Version: {version} (Mirror=False) ===\n")
        subprocess.run(["python", "dl_validate.py", version, "no_mirror"])

    # --------------------------
    # Validate with mirror
    # --------------------------
    stats_mirror = os.path.join(version_dir, "validate_stats_mirrored.npy")
    if os.path.exists(stats_mirror) and os.path.getsize(stats_mirror) > 2048:
        print(f"\n[SKIP] {version} (mirror) already validated.")
    else:
        print(f"\n=== Validating Version: {version} (Mirror=True) ===\n")
        subprocess.run(["python", "dl_validate.py", version, "mirror"])
