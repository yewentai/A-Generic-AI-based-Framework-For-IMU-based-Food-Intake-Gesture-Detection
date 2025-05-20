#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX table from F1 score JSON summary (mean and std)
Only shows results at thresholds 0.1 and 0.5
"""

import json

# Input path (update this if needed)
json_path = "results/aug/FDI/compare_axis_permuted_FDI_table.json"
thresholds_to_show = ["0.1", "0.5"]

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Prepare LaTeX table lines
header = "Model & F1@0.1 & F1@0.5 \\\\ \\midrule"
rows = []
for model_name, metrics in data.items():
    cells = [model_name]
    for t in thresholds_to_show:
        if t in metrics:
            m = metrics[t]["mean"]
            s = metrics[t]["std"]
            cells.append(f"{m:.2f} $\\pm$ {s:.2f}")
        else:
            cells.append("–")
    rows.append(" & ".join(cells) + " \\\\")

# Combine and print full table
print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Segment-wise F1 score comparison at thresholds 0.1 and 0.5}")
print("\\label{tab:f1-threshold}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print(header)
print("\n".join(rows))
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
