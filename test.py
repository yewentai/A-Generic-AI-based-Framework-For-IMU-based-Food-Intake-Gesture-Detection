import numpy as np
import matplotlib.pyplot as plt

def nonzero_intervals(x):
    """Extract start and end indices of nonzero intervals."""
    # Prepare a list to collect results
    results = []

    # Track when we are inside a nonzero segment
    in_segment = False
    start_idx = 0

    for i in range(len(x)):
        if x[i] != 0 and not in_segment:
            # Encounter the start of a nonzero segment
            in_segment = True
            start_idx = i
        elif x[i] == 0 and in_segment:
            # Encounter the end of a nonzero segment
            in_segment = False
            results.append([start_idx, i])

    # Handle the case where the sequence ends with a nonzero segment
    if in_segment:
        results.append([start_idx, len(x)])
    
    # Convert the results list to a NumPy array
    return np.array(results, dtype=int)
    

def segment_f1_binary(pred, gt, threshold=0.5, debug_plot=False):
    f_n, f_p, t_p = 0, 0, 0
    union = np.logical_or(pred, gt).astype(int)
    union_intervals = nonzero_intervals(union)

    if debug_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.step(range(len(gt)), gt, where='post', label='Ground Truth', color='blue')
        ax1.set_title('Ground Truth')
        ax2.step(range(len(pred)), pred, where='post', label='Prediction', color='red')
        ax2.set_title('Prediction')
        ax3.step(range(len(union)), union, where='post', label='Union', color='green')
        ax3.set_title('Union')

    for start_idx, end_idx in union_intervals:
        flag_t_p = 0
        gt_interval = nonzero_intervals(gt[start_idx:end_idx])
        pred_interval = nonzero_intervals(pred[start_idx:end_idx])
        
        gt_interval = gt_interval + start_idx
        pred_interval = pred_interval + start_idx

        if debug_plot:
            for gs, ge in gt_interval:
                ax1.axvspan(gs, ge, alpha=0.3, color='blue')
            for ps, pe in pred_interval:
                ax2.axvspan(ps, pe, alpha=0.3, color='red')

        if len(gt_interval) == 0 and len(pred_interval) == 1:
            f_p += 1
            if debug_plot:
                ax3.axvspan(pred_interval[0][0], pred_interval[0][1], alpha=0.5, color='red')
                ax3.text((pred_interval[0][0] + pred_interval[0][1]) / 2, 0.9, 'FP', color='red', ha='center')
        elif len(gt_interval) == 1 and len(pred_interval) == 0:
            f_n += 1
            if debug_plot:
                ax3.axvspan(gt_interval[0][0], gt_interval[0][1], alpha=0.5, color='blue')
                ax3.text((gt_interval[0][0] + gt_interval[0][1]) / 2, 0.9, 'FN', color='blue', ha='center')
        elif len(gt_interval) == 1 and len(pred_interval) == 1:
            gt_start, gt_end = gt_interval[0]
            pred_start, pred_end = pred_interval[0]
            intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
            union = max(gt_end, pred_end) - min(gt_start, pred_start)
            iou = intersection / union
            len_gt = gt_interval[0][1] - gt_interval[0][0] 
            len_pred = pred_interval[0][1] - pred_interval[0][0] 
            if iou >= threshold:
                t_p += 1
                if debug_plot:
                    if len_gt >= len_pred:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color='red')
                        ax3.text((pred_start + pred_end) / 2, 0.9, 'TP', color='red', ha='center')
                    else:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color='blue')
                        ax3.text((gt_start + gt_end) / 2, 0.9, 'TP', color='blue', ha='center')
            elif len_gt > len_pred:
                f_n += 1
                if debug_plot:
                    ax3.axvspan(gt_start, gt_end, alpha=0.5, color='blue')
                    ax3.text((gt_start + gt_end) / 2, 0.9, 'FN', color='blue', ha='center')
            else:
                f_p += 1
                if debug_plot:
                    ax3.axvspan(pred_start, pred_end, alpha=0.5, color='red')
                    ax3.text((pred_start + pred_end) / 2, 0.9, 'FP', color='red', ha='center')
        elif len(gt_interval) >= 2 and len(pred_interval) == 1:
            pred_start, pred_end = pred_interval[0]
            for gt_start, gt_end in gt_interval:
                intersection = abs(min(gt_end, pred_end) - max(gt_start, pred_start))
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start , gt_end, alpha=0.5, color='green')
                        ax3.text((gt_start + gt_end) / 2, 0.9, 'TP', color='green', ha='center')
                else:   
                    f_p += 1
                    if debug_plot:
                        ax3.axvspan(gt_start, gt_end, alpha=0.5, color='blue')
                        ax3.text((gt_start + gt_end) / 2, 0.9, 'FP', color='blue', ha='center')
        elif len(gt_interval) == 1 and len(pred_interval) >= 2:
            gt_start, gt_end = gt_interval[0]
            for pred_start, pred_end in pred_interval:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start)
                union = max(gt_end, pred_end) - min(gt_start, pred_start)
                iou = intersection / union
                if iou >= threshold and flag_t_p == 0:
                    flag_t_p += 1
                    t_p += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color='green')
                        ax3.text((pred_start + pred_end) / 2, 0.9, 'TP', color='green', ha='center')
                else:
                    f_n += 1
                    if debug_plot:
                        ax3.axvspan(pred_start, pred_end, alpha=0.5, color='blue')
                        ax3.text((pred_start + pred_end) / 2, 0.9, 'FN', color='blue', ha='center')
        else:
            print("Case not handled")
            

    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if debug_plot:
        for ax in (ax1, ax2, ax3):
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
        ax3.set_xlabel('Time')
        plt.tight_layout()
        plt.show()

    return f1

# Ground Truth sequence
gt = np.array([
    0, 0, 1, 1, 1, 0, 0,  # Normal gesture
    0, 0, 0, 0, 0, 0, 0,  # No gesture (for insertion test)
    1, 1, 1, 1, 0, 0, 0,  # Gesture for underfill and overfill tests
    0, 1, 1, 0, 1, 1, 0,  # Two gestures for merge test
    1, 1, 1, 1, 1, 1, 1,  # Long gesture for fragmentation test
    0, 0, 1, 1, 1, 0, 0   # Gesture for deletion test
])

# Prediction sequence
pred = np.array([
    0, 1, 1, 1, 1, 1, 0,  # Overfill
    0, 1, 1, 0, 0, 0, 0,  # Insertion
    1, 1, 0, 0, 0, 0, 0,  # Underfill
    0, 1, 1, 1, 1, 1, 0,  # Merge
    1, 0, 1, 0, 1, 0, 1,  # Fragmentation
    0, 0, 0, 0, 0, 0, 0   # Deletion
])

# Plot the sequences
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot ground truth
ax1.step(range(len(gt)), gt, where='post', label='Ground Truth', color='blue')
ax1.set_ylabel('State')
ax1.set_ylim(-0.1, 1.1)
ax1.legend()
ax1.set_title('Ground Truth')

# Plot prediction
ax2.step(range(len(pred)), pred, where='post', label='Prediction', color='red')
ax2.set_xlabel('Time')
ax2.set_ylabel('State')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.set_title('Prediction')

# Add labels for different situations
situations = ['Normal/Overfill', 'Insertion', 'Underfill', 'Merge', 'Fragmentation', 'Deletion']
for i, situation in enumerate(situations):
    ax1.text(i*7 + 3, 1.15, situation, ha='center', va='center', rotation=45)
    ax2.text(i*7 + 3, -0.15, situation, ha='center', va='center', rotation=45)

plt.suptitle(f"Ground Truth vs Prediction")
plt.tight_layout()
plt.show()

# Calculate F1 score segment-wise
f1_score_seg = segment_f1_binary(pred, gt, 0.2, debug_plot=True)
print(f"F1 Score: {f1_score_seg:.4f}")