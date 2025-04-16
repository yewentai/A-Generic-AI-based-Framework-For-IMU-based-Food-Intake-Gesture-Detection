#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU Training Result Conversion Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Last Edited : 2025-04-16
Description : This script recursively converts .npy training result files into
              .mat files for MATLAB compatibility. It detects and handles both
              regular numpy arrays and lists of dictionaries (e.g., training
              statistics) by converting them to MATLAB struct arrays.
===============================================================================
"""

import os
import numpy as np
import scipy.io as sio


def convert_list_of_dicts_to_struct_array(list_of_dicts):
    """Convert a list of dicts to a NumPy structured array that MATLAB can understand."""
    keys = list_of_dicts[0].keys()
    structured_data = {k: [d[k] for d in list_of_dicts] for k in keys}
    return structured_data


def convert_npy_to_mat(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)

        if isinstance(data, np.ndarray) and data.dtype == object:
            # Might be a list of dicts
            data_list = data.tolist()
            if isinstance(data_list, list) and isinstance(data_list[0], dict):
                mat_data = convert_list_of_dicts_to_struct_array(data_list)
            else:
                mat_data = {"data": data}
        else:
            mat_data = {"data": data}

        mat_filepath = filepath.replace(".npy", ".mat")
        sio.savemat(mat_filepath, mat_data)
        print(f"Converted: {filepath} -> {mat_filepath}")
    except Exception as e:
        print(f"Failed to convert {filepath}: {e}")


def convert_all_npys_to_mat(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                convert_npy_to_mat(os.path.join(root, file))


result_root = "result"
# versions = ["202504100555"]  # Specify the version to analyze
versions = [d for d in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, d))]
versions.sort()

for version in versions:
    result_dir = os.path.join("result", version)
    convert_all_npys_to_mat(result_dir)
