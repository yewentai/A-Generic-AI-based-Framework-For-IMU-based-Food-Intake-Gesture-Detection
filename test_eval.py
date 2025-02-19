import numpy as np
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from evaluation import segment_evaluation


# Generate test sequences
def generate_test_sequence(length, num_segments, classes=[0, 1]):
    arr = np.zeros(length, dtype=int)
    for _ in range(num_segments):
        cls = np.random.choice(classes[1:])  # Exclude background class 0
        start = np.random.randint(0, length - 5)
        end = start + np.random.randint(1, 5)
        arr[start:end] = cls
    return arr


# Generate binary test case
np.random.seed(42)
binary_gt = generate_test_sequence(50, 8, [0, 1])
binary_pred = generate_test_sequence(50, 10, [0, 1])

# Generate 3-class test case
multiclass_gt = generate_test_sequence(50, 8, [0, 1, 2, 3])
multiclass_pred = generate_test_sequence(50, 10, [0, 1, 2, 3])

# # Test binary case
# print("Binary Case Evaluation:")
# binary_fn, binary_fp, binary_tp = segment_confusion_matrix_binary(
#     binary_pred, binary_gt, threshold=0.5, debug_plot=True
# )
# print(f"Binary Results (FN: {binary_fn}, FP: {binary_fp}, TP: {binary_tp})\n")

# # Test 3-class case
# print("Multi-class Case Evaluation:")
# multiclass_fn, multiclass_fp, multiclass_tp = segment_confusion_matrix(
#     multiclass_pred, multiclass_gt, threshold=0.5, debug_plot=True
# )
# print(
#     f"Multi-class Results (FN: {multiclass_fn}, FP: {multiclass_fp}, TP: {multiclass_tp})\n"
# )


# Test Hungarian algorithm version
print("\nHungarian Algorithm Evaluation:")
hungarian_binary = segment_evaluation(binary_pred, binary_gt, debug_plot=True)
hungarian_multiclass = segment_evaluation(
    multiclass_pred, multiclass_gt, debug_plot=True
)
print(f"Hungarian Binary Results: {hungarian_binary}")
print(f"Hungarian Multi-class Results: {hungarian_multiclass}")
