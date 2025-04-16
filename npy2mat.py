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


def convert_list_of_dicts_to_struct_array(data_list):
    keys = data_list[0].keys()
    mat_struct = {}
    for key in keys:
        mat_struct[key] = [d[key] for d in data_list]
    return mat_struct


def convert_npy_to_mat(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_path = os.path.join(dirpath, filename)
                mat_path = os.path.join(dirpath, filename.replace(".npy", ".mat"))

                print(f"Converting: {npy_path} -> {mat_path}")

                try:
                    data = np.load(npy_path, allow_pickle=True)

                    # Handle list of dicts
                    if isinstance(data, np.ndarray) and isinstance(data[0], dict):
                        struct_array = convert_list_of_dicts_to_struct_array(data)
                        sio.savemat(mat_path, {"data": struct_array})
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
