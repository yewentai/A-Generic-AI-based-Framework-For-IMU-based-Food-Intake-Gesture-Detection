import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_block(ax, x, y, label, color="steelblue"):
    rect = patches.Rectangle((x, y), 0.5, 0.5, linewidth=1, edgecolor="black", facecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.25, y + 0.25, label, color="white", ha="center", va="center", fontsize=6)


def draw_column(ax, x, y, labels, color="steelblue", alpha=1.0):
    for i, label in enumerate(labels):
        draw_block(ax, x, y - i * 0.6, label, color)
    for p in ax.patches[-len(labels):]:
        p.set_alpha(alpha)


def draw_sequence(ax, x_start, y, labels, n_steps=4, color="steelblue", alpha=1.0, add_dots=False, add_final_gap=False):
    for i in range(n_steps):
        draw_column(ax, x_start + i * 0.6, y, labels, color=color, alpha=alpha)
    if add_dots:
        dot_x = x_start + n_steps * 0.6 + (0.3 if add_final_gap else -0.3)
        ax.text(dot_x, y - 1.8, "...", fontsize=14, ha="center")
    if add_final_gap:
        draw_column(ax, x_start + n_steps * 0.6 + 0.6, y, labels, color=color, alpha=alpha)


def draw_time_labels(ax, x_start, y, steps, add_final_gap=False):
    for i in range(steps):
        ax.text(x_start + i * 0.6 + 0.25, y - 3.6, f"$t_{{{i}}}$", fontsize=6, ha="center")
    offset = steps * 0.6 + (0.6 if add_final_gap else 0.0)
    ax.text(x_start + offset + 0.25, y - 3.6, "$t_n$", fontsize=6, ha="center")

def draw_time_labels_1(ax, x_start, y, steps, add_final_gap=False):
    for i in range(steps):
        ax.text(x_start + i * 0.6 + 0.25, y - 3.6, f"$t_{{{i}}}$", fontsize=6, ha="center")
    offset = steps * 0.6 + (0.6 if add_final_gap else 0.0)
    ax.text(x_start + offset + 0.25, y - 3.6, "$t_m$", fontsize=6, ha="center")

def draw_time_labels_2(ax, x_start, y, steps, add_final_gap=False):
    for i in range(steps):
        ax.text(x_start + i * 0.6 + 0.25, y - 3.6, f"$t_n+t_{{{i}}}$", fontsize=6, ha="center")
    offset = steps * 0.6 + (0.6 if add_final_gap else 0.0)
    ax.text(x_start + offset + 0.25, y - 3.6, "$t_n+t_m$", fontsize=6, ha="center")


def draw_arrow(ax, start, end, text=None, double=False, color="black"):
    if double:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->", color=color, lw=2))
    else:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.2, text, ha="center", fontsize=8)


fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(-1, 18)
ax.set_ylim(-6, 5)
ax.axis("off")

imu_labels = [r"$a_x$", r"$a_y$", r"$a_z$", r"$g_x$", r"$g_y$", r"$g_z$"]
mirrored_labels = [r"$-a_x$", r"$a_y$", r"$a_z$", r"$g_x$", r"$-g_y$", r"$-g_z$"]

# Original sequence
draw_sequence(ax, 0, 4, imu_labels, n_steps=4, color="steelblue", add_dots=True, add_final_gap=True)
draw_time_labels(ax, 0, 4.3, 4, add_final_gap=True)
ax.text(1.7, 4.7, "Original sequence", fontsize=8, ha="center")

# Arrow between original and sampled
draw_arrow(ax, (1.7, 0.5), (1.7, 0), color="dodgerblue")
ax.text(2.2, 0.2, "Sampling", fontsize=8, ha="center")

# Sampled sequence
draw_sequence(ax, 0.5, -0.5, imu_labels, n_steps=2, color="steelblue", add_dots=True, add_final_gap=True)
draw_time_labels_1(ax, 0.5, -0.2, 2, add_final_gap=True)
ax.text(1.7, -4.2, "Sampled subsequence", fontsize=8, ha="center")

# Sampled after augmentation
draw_sequence(ax, 6, -0.5, mirrored_labels, n_steps=2, color="skyblue", add_dots=True, add_final_gap=True)
draw_time_labels_1(ax, 6, -0.2, 2, add_final_gap=True)
ax.text(7.2, -4.2, "Augmented samples", fontsize=8, ha="center")

# Arrow between sampled and augmented
draw_arrow(ax, (3.1, -2), (5.7, -2), color="dodgerblue", text="Augmentation")
ax.text(4.4, -2.5, "Hand mirroring/", fontsize=7, ha="center")
ax.text(4.4, -2.9, "Axis permutation/", fontsize=7, ha="center")
ax.text(4.4, -3.3, "Planar rotation", fontsize=7, ha="center")

# Original sequence (full again)
draw_sequence(ax, 5.5, 4, imu_labels, n_steps=4, color="steelblue", add_dots=True, add_final_gap=True)
draw_time_labels(ax, 5.5, 4.3, 4, add_final_gap=True)
ax.text(7.2, 4.7, "Original sequence", fontsize=8, ha="center")

# Augmented sequence (concat sampled + mirrored)
draw_sequence(ax, 10.7, 1.7, imu_labels, n_steps=4, color="steelblue", add_dots=True, add_final_gap=True)
draw_sequence(ax, 14.3, 1.7, mirrored_labels, n_steps=2, color="skyblue", add_dots=True, add_final_gap=True)
draw_time_labels(ax, 10.7, 2, 4, add_final_gap=True)
draw_time_labels_2(ax, 14.3, 2, 2, add_final_gap=True)

# Arrow indicating concat
draw_arrow(ax, (9.3, 0.1), (10.7, 0.1), color="dodgerblue", text="Concatenate")

# Label
ax.text(13.5, 2.5, "Augmented sequence", fontsize=8, ha="center")

plt.tight_layout()
# plt.show()
plt.savefig("plots/augmentation.pdf", format="pdf", dpi=300, bbox_inches="tight")
