import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_tcn_stage_schematic(ax, center_x, base_y, stage_block_width,
                             nodes_per_layer_vis, layer_height_vis, kernel_size_vis):
    """
    Draws a schematic of one TCN stage with input, hidden, and output node layers,
    and simplified dilated connections.
    """
    num_layers = len(nodes_per_layer_vis)
    # compute vertical span
    total_height = (num_layers - 1) * layer_height_vis
    # collect node positions
    layer_positions = []
    for i, num_nodes in enumerate(nodes_per_layer_vis):
        y = base_y + i * layer_height_vis
        xs = np.linspace(center_x - stage_block_width/2,
                         center_x + stage_block_width/2, num_nodes)
        for x in xs:
            color = ['cornflowerblue'] + ['lightgrey']*(num_layers-2) + ['sandybrown']
            ax.add_patch(patches.Circle((x, y), 0.05, facecolor=color[i], edgecolor='black', lw=0.5))
        layer_positions.append(list(zip(xs, [y]*num_nodes)))
    # draw connections
    for i in range(num_layers - 1):
        for (x0, y0) in layer_positions[i]:
            for (x1, y1) in layer_positions[i+1]:
                ax.plot([x0, x1], [y0, y1], color='gray', lw=0.5, alpha=0.5)
    # draw rounded rectangle
    rect = patches.FancyBboxPatch(
        (center_x-stage_block_width/2-0.1, base_y-0.1),
        stage_block_width+0.2, total_height+0.2,
        boxstyle="round,pad=0.1", edgecolor='black', facecolor='none', lw=1.2
    )
    ax.add_patch(rect)
    return base_y + total_height

def generate_mstcn_matplotlib_schematic():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.axis('off')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 5)

    # 1) Input bar
    ax.add_patch(patches.Rectangle((-0.8, 0.0), 1.6, 0.1, facecolor='lightgray', edgecolor='black'))
    ax.text(0, -0.1, "Input: x", ha='center', va='top', fontsize=10)

    # 2) Stage 1
    y0_stage1 = 0.2
    ax.text(0, y0_stage1+0.1, "Stage 1", ha='center', va='bottom', fontsize=12, fontweight='bold')
    top1 = draw_tcn_stage_schematic(
        ax, 0, y0_stage1+0.2,
        stage_block_width=1.2,
        nodes_per_layer_vis=[8,6,4,6],
        layer_height_vis=0.4,
        kernel_size_vis=3
    )
    # L1 arrow
    ax.annotate('', xy=(0.6, (y0_stage1+0.2+top1)/2), xytext=(0.3, (y0_stage1+0.2+top1)/2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.65, (y0_stage1+0.2+top1)/2, r'$L_1$', color='red', va='center')

    # 3) connection to Stage N
    ax.annotate('', xy=(0, top1+0.1), xytext=(0, top1+0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--'))

    # 4) Stage N
    y0_stageN = top1+0.6
    ax.text(0, y0_stageN+0.1, "Stage N", ha='center', va='bottom', fontsize=12, fontweight='bold')
    topN = draw_tcn_stage_schematic(
        ax, 0, y0_stageN+0.2,
        stage_block_width=1.2,
        nodes_per_layer_vis=[8,6,4,6],
        layer_height_vis=0.4,
        kernel_size_vis=3
    )
    # LN arrow
    ax.annotate('', xy=(0.6, (y0_stageN+0.2+topN)/2), xytext=(0.3, (y0_stageN+0.2+topN)/2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.65, (y0_stageN+0.2+topN)/2, r'$L_N$', color='red', va='center')

    # 5) Predict bar
    y_pred = topN + 0.4
    colors = ['cornflowerblue','mediumseagreen','sandybrown','lightcoral']
    widths = [0.3,0.4,0.6,0.5]
    x_left = -0.8
    for w,c in zip(widths,colors):
        ax.add_patch(patches.Rectangle((x_left, y_pred), w, 0.1, facecolor=c, edgecolor='black'))
        x_left += w
    ax.text(-0.8, y_pred+0.15, "Predict: Y", ha='left', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("mstcn_refined.png", dpi=300, bbox_inches='tight', pad_inches=0)
    print("Saved figure as mstcn_refined.png")

if __name__ == '__main__':
    generate_mstcn_matplotlib_schematic()
