import numpy as np
import matplotlib.pyplot as plt
import os

# 配置
DATASET = "DXI"
RESULT_VERSION = "02261202"  # 修改为实际版本号
result_dir = "result"

# 加载数据
testing_stats = np.load(
    os.path.join(result_dir, f"testing_stats_{DATASET.lower()}_{RESULT_VERSION}.npy"),
    allow_pickle=True,
)
training_stats = np.load(
    os.path.join(result_dir, f"training_stats_{DATASET.lower()}_{RESULT_VERSION}.npy"),
    allow_pickle=True,
)

# 统计分析
f1_scores = [e["average_f1"] for e in testing_stats]
tp_total = sum(e["class_metrics"]["class_1"]["tp"] for e in testing_stats)
fp_total = sum(e["class_metrics"]["class_1"]["fp"] for e in testing_stats)
fn_total = sum(e["class_metrics"]["class_1"]["fn"] for e in testing_stats)

print(f"\n{' Results Summary ':=^40}")
print(f"Average F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Total TP: {tp_total}  FP: {fp_total}  FN: {fn_total}")

# 可视化
plt.figure(figsize=(12, 6))

# F1分布
plt.subplot(121)
plt.boxplot(
    f1_scores,
    vert=False,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
)
plt.title("F1 Score Distribution")
plt.xlim(0, 1)

# Fold-wise表现
plt.subplot(122)
plt.bar(range(1, len(f1_scores) + 1), f1_scores, color="skyblue")
plt.xticks(range(1, len(f1_scores) + 1))
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.title("F1 Score per Fold")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
