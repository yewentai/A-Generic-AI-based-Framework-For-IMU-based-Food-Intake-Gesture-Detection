import os
import numpy as np
import scipy.io as sio


def convert_npy_to_mat(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_path = os.path.join(dirpath, filename)
                mat_path = os.path.join(dirpath, filename.replace(".npy", ".mat"))

                print(f"Converting: {npy_path} -> {mat_path}")

                try:
                    data = np.load(npy_path, allow_pickle=True)
                    if hasattr(data, "item"):
                        data = data.item()  # convert to dict if possible
                        sio.savemat(mat_path, {"data": data})
                    else:
                        sio.savemat(mat_path, {"data": data})
                except Exception as e:
                    print(f"Failed to convert {npy_path}: {e}")


result_root = "result"
# versions = ["202504100555"]  # Specify the version to analyze
versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
versions.sort()

for version in versions:
    result_dir = os.path.join("result", version)
    convert_npy_to_mat(result_dir)
