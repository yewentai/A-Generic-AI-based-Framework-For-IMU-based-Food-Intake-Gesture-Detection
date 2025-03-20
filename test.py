import os
import pickle
import numpy as np

# Base directory where FD datasets are located
base_dir = "/home/wsl/codes_linux/thesis/dataset/FD"
dataset_folders = ["FD-I", "FD-II", "FD-III"]

for folder in dataset_folders:
    folder_path = os.path.join(base_dir, folder)

    # Define input file paths
    x_l_path = os.path.join(folder_path, "X_L.pkl")
    x_r_path = os.path.join(folder_path, "X_R.pkl")
    y_l_path = os.path.join(folder_path, "Y_L.pkl")
    y_r_path = os.path.join(folder_path, "Y_R.pkl")

    # Define output file paths
    x_combined_path = os.path.join(folder_path, "X.pkl")
    y_combined_path = os.path.join(folder_path, "Y.pkl")

    # Load pickle files for X and Y
    with open(x_l_path, "rb") as f:
        x_l = pickle.load(f)
    with open(x_r_path, "rb") as f:
        x_r = pickle.load(f)
    with open(y_l_path, "rb") as f:
        y_l = pickle.load(f)
    with open(y_r_path, "rb") as f:
        y_r = pickle.load(f)

    # Check that both left and right files have the same number of subjects (should be 46)
    assert len(x_l) == len(x_r), "Mismatch in number of subjects for X"
    assert len(y_l) == len(y_r), "Mismatch in number of subjects for Y"

    # Concatenate subject-wise along the time dimension (axis=0)
    x_combined = []
    y_combined = []

    for i in range(len(x_l)):
        # Each subject's data: shape remains (n_rows_left + n_rows_right, 6)
        combined_x = np.concatenate([x_l[i], x_r[i]], axis=0)
        x_combined.append(combined_x)

        # For labels, assuming a similar structure (e.g., 1D or 2D arrays)
        combined_y = np.concatenate([y_l[i], y_r[i]], axis=0)
        y_combined.append(combined_y)

    # Save the combined pickle files
    with open(x_combined_path, "wb") as f:
        pickle.dump(x_combined, f)
    with open(y_combined_path, "wb") as f:
        pickle.dump(y_combined, f)

    print(f"Combined X and Y pickle files have been saved in {folder_path}")
