import numpy as np
from scipy.optimize import linear_sum_assignment


def match_segments(gt_segments, pred_segments, iou_threshold=0.5):
    """
    使用匈牙利算法匹配预测片段和真实片段
    :param gt_segments: List of (start, end) tuples
    :param pred_segments: List of (start, end) tuples
    :param iou_threshold: float
    :return: (tp, fp, fn)
    """
    # 计算 IoU 矩阵
    iou_matrix = np.zeros((len(gt_segments), len(pred_segments)))
    for i, (gt_start, gt_end) in enumerate(gt_segments):
        for j, (pred_start, pred_end) in enumerate(pred_segments):
            inter_start = max(gt_start, pred_start)
            inter_end = min(gt_end, pred_end)
            inter = max(0, inter_end - inter_start)
            union = (gt_end - gt_start) + (pred_end - pred_start) - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0

    # 匈牙利算法匹配
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    tp = 0
    matched_gt = set()
    matched_pred = set()

    for i, j in zip(gt_indices, pred_indices):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1
            matched_gt.add(i)
            matched_pred.add(j)

    fp = len(pred_segments) - len(matched_pred)
    fn = len(gt_segments) - len(matched_gt)

    return tp, fp, fn


# 示例数据
gt = [(5, 10), (20, 25), (30, 40)]  # 真实片段
pred = [(6, 9), (18, 22), (28, 35), (45, 50)]  # 预测片段

tp, fp, fn = match_segments(gt, pred, iou_threshold=0.7)
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
# 输出：TP: 2, FP: 2, FN: 1

# plot the sequences
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot ground truth
for start, end in gt:
    ax1.axvspan(start, end, color="blue", alpha=0.3)
    ax1.set_ylabel("State")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title("Ground Truth")

# Plot prediction
for start, end in pred:
    ax2.axvspan(start, end, color="red", alpha=0.3)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("State")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title("Prediction")

plt.suptitle(f"Ground Truth vs Prediction")
plt.tight_layout()
plt.show()
