import numpy as np
import matplotlib.pyplot as plt

def nonzero_intervals(x):
    """Extract start and end indices of nonzero intervals."""
    # Find the indices where the segments change
    idxs = np.array([0] + (np.nonzero(np.diff(x))[0] + 1).tolist() + [len(x)])
    
    # Prepare a list to collect results
    results = []
    
    # Iterate through segments and find those with non-zero labels
    for i in range(len(idxs) - 1):
        start_idx = idxs[i]
        end_idx = idxs[i + 1] - 1
        
        # Add the start index and end index
        results.append([start_idx, end_idx])
    
    # Convert the results list to a NumPy array
    return np.array(results, dtype=int)
    

def segment_f1_binary(pred, gt, threshold=0.5):
    f_n, f_p, t_p = 0, 0, 0
    union = np.logical_or(pred, gt).astype(int)
    union_intervals = nonzero_intervals(union)

    for start_idx, end_idx in union_intervals:
        # Find the corresponding intervals in the ground truth and prediction
        gt_interval = nonzero_intervals(gt[start_idx:end_idx + 1])
        pred_interval = nonzero_intervals(pred[start_idx:end_idx + 1])
        
        # Adjust intervals to be relative to the current union interval
        gt_interval = gt_interval + start_idx
        pred_interval = pred_interval + start_idx

        # Process ground truth intervals
        for gt_start, gt_end in gt_interval:
            matched = False
            for pred_start, pred_end in pred_interval:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start) + 1
                union = max(gt_end, pred_end) - min(gt_start, pred_start) + 1
                iou = intersection / union

                if iou >= threshold:
                    t_p += 1
                    matched = True
                    break
            
            if not matched:
                f_n += 1

        # Process prediction intervals
        for pred_start, pred_end in pred_interval:
            matched = False
            for gt_start, gt_end in gt_interval:
                intersection = min(gt_end, pred_end) - max(gt_start, pred_start) + 1
                union = max(gt_end, pred_end) - min(gt_start, pred_start) + 1
                iou = intersection / union

                if iou >= threshold:
                    matched = True
                    break
            
            if not matched:
                f_p += 1

    # Calculate F1 score
    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1
        

# Generate two random 0-1 sequences
np.random.seed(7)  # for reproducibility
length = 100
gt = np.random.randint(0, 2, length)
pred = np.random.randint(0, 2, length)

# Calculate segement F1 score
f1_score_seg_1 = segment_f1_binary(pred, gt, threshold=0.1)
print(f"F1 Score (Segment) with threshold 0.1: {f1_score_seg_1:.4f}")
f1_score_seg_2 = segment_f1_binary(pred, gt, threshold=0.5)
print(f"F1 Score (Segment) with threshold 0.5: {f1_score_seg_2:.4f}")
f1_score_seg_3 = segment_f1_binary(pred, gt, threshold=0.9)
print(f"F1 Score (Segment) with threshold 0.9: {f1_score_seg_3:.4f}")


# Calculate sample F1 score
tp = np.sum((pred == 1) & (gt == 1))
fp = np.sum((pred == 1) & (gt == 0))
fn = np.sum((pred == 0) & (gt == 1))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score_sample = 2 * precision * recall / (precision + recall)
print(f"F1 Score (Sample): {f1_score_sample:.4f}")

# Plot the sequences in separate plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot predictions
ax1.step(range(len(pred)), pred, where='post', label='Predictions', color='red')
ax1.set_ylabel('State')
ax1.set_ylim(-0.1, 1.1)
ax1.legend()
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot ground truth (labels)
ax2.step(range(len(gt)), gt, where='post', label='Labels', color='blue')
ax2.set_xlabel('Time')
ax2.set_ylabel('State')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
