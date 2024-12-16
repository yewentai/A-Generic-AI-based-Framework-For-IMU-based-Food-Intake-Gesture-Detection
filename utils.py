import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset class
class IMUDataset(Dataset):
    def __init__(self, X, Y, sequence_length=128):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length
        self.subject_indices = []  # Record which subject each sample belongs to

        # Processing data for each session
        for subject_idx, (imu_data, labels) in enumerate(zip(X, Y)):
            # imu_data = self.normalize(imu_data)
            num_samples = len(labels)

            for i in range(0, num_samples, sequence_length):
                imu_segment = imu_data[i : i + sequence_length]
                label_segment = labels[i : i + sequence_length]

                if len(imu_segment) == sequence_length:
                    self.data.append(imu_segment)
                    self.labels.append(label_segment)
                    self.subject_indices.append(subject_idx)

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
    

# Evaluation function
# Function to find the intervals (start and end) where changes occur in the sequence
def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

# Function to label each segment
def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

# Function to compute F1 score for drinking
def segment_f1_drinking(y_pre_list, y_gt_list):
    FN_d_thre = []
    FP_d_thre = []
    data_len = len(y_pre_list)
    thre = [0.1, 0.25, 0.5, 0.75]  # Set of overlap thresholds to evaluate

    for k in range(4):
        overlap = thre[k]
        f_n_drink = np.zeros(data_len)
        f_p_drink = np.zeros(data_len)
        num_true_id_drink = np.zeros(data_len)

        for i in range(data_len):
            y_pred, y_true = y_pre_list[i], y_gt_list[i]
            true_intervals = np.array(segment_intervals(y_true))
            true_labels = segment_labels(y_true)
            pred_intervals = np.array(segment_intervals(y_pred))
            pred_labels = segment_labels(y_pred)

            bg_class = 0  # Background class label
            # Remove background labels
            true_intervals = true_intervals[true_labels != bg_class]
            true_labels = true_labels[true_labels != bg_class]
            pred_intervals = pred_intervals[pred_labels != bg_class]
            pred_labels = pred_labels[pred_labels != bg_class]

            num_true_id_drink[i] = np.count_nonzero(true_labels == 1)  # For drinking (assuming label = 1)

            num_true_len = len(true_intervals)
            num_pred_len = len(pred_intervals)

            hit_pred_tp = np.zeros(num_pred_len)  # True positive hits
            hit_pred = np.zeros(num_pred_len)    # Prediction hits

            # Evaluate true positive segments for drinking
            for k in range(num_true_len):
                if true_labels[k] == 1:
                    overlap_id_list = []
                    key = 0
                    start_idx, end_idx = true_intervals[k]
                    len_true = end_idx - start_idx

                    for j in range(num_pred_len):
                        if pred_intervals[j][0] < end_idx and pred_intervals[j][1] > start_idx:
                            if pred_labels[j] == 1:
                                key += 1
                                intersection = min(pred_intervals[j][1], end_idx) - max(pred_intervals[j][0], start_idx)
                                union = max(pred_intervals[j][1], end_idx) - min(pred_intervals[j][0], start_idx)
                                IoU = intersection / union
                                len_pre_seg = pred_intervals[j][1] - pred_intervals[j][0]
                                overlap_id_list.append([j, IoU, len_pre_seg])
                        elif pred_intervals[j][0] > end_idx and key == 0:  # No detection
                            f_n_drink[i] += 1
                            break
                        elif j == num_pred_len - 1 and key == 0:
                            f_n_drink[i] += 1
                            break
                        elif pred_intervals[j][0] > end_idx and key == 1:  # One detection
                            idx = int(overlap_id_list[0][0])
                            if hit_pred[idx] == 0:
                                hit_pred[idx] = 1
                                if overlap_id_list[0][1] < overlap and len_true > overlap_id_list[0][2]:
                                    f_n_drink[i] += 1
                                elif overlap_id_list[0][1] < overlap and len_true < overlap_id_list[0][2]:
                                    f_p_drink[i] += 1
                                elif overlap_id_list[0][1] > overlap:
                                    hit_pred_tp[idx] = 1
                                break
                            elif hit_pred[idx] == 1:
                                f_n_drink[i] += 1
                                break
                        elif pred_intervals[j][0] > end_idx and key > 1:  # More than one detection
                            overlap_id_list = np.array(overlap_id_list)
                            idx = int(np.array(overlap_id_list)[:, 1].argmax())
                            hit_pred[idx] = 1
                            if overlap_id_list[idx, 1] < overlap:
                                f_n_drink[i] += 1
                            elif overlap_id_list[idx, 1] > overlap:
                                hit_pred_tp[int(overlap_id_list[idx][0])] = 1
                            f_p_drink[i] += key - 1
                            break

            # False positive predictions for drinking
            for k in range(num_pred_len):
                if pred_labels[k] == 1:
                    key = 0
                    start_idx, end_idx = pred_intervals[k]
                    for j in range(num_true_len):
                        if true_intervals[j][0] < end_idx and true_intervals[j][1] > start_idx and true_labels[j] == 1:
                            key += 1
                        elif j == num_true_len - 1 and key == 0:
                            f_p_drink[i] += 1
                            break
                        elif true_intervals[j][0] > end_idx and key == 0:
                            f_p_drink[i] += 1
                            break

        FN_d_thre.append(f_n_drink)
        FP_d_thre.append(f_p_drink)

    # Compute true positives and F1-score for drinking
    TP_d_thre = []
    F1_d_thre = []
    total_d_FN = []
    total_d_FP = []
    total_d_TP = []

    for i in range(len(FN_d_thre)):
        t_p_d = np.zeros(len(FN_d_thre[i]))
        if i == 0:
            t_p_d = num_true_id_drink - FN_d_thre[i]
        else:
            t_p_d = (num_true_id_drink - FN_d_thre[i-1]) - (FN_d_thre[i] - FN_d_thre[i-1]) - (FP_d_thre[i] - FP_d_thre[i-1])
        f1_d = 2 * t_p_d / (2 * t_p_d + FN_d_thre[i] + FP_d_thre[i])
        TP_d_thre.append(t_p_d)
        F1_d_thre.append(f1_d)
        total_d_FN.append(sum(FN_d_thre[i]))
        total_d_FP.append(sum(FP_d_thre[i]))
        total_d_TP.append(sum(TP_d_thre[i]))

    total_d_F1 = 2 * np.array(total_d_TP) / (2 * np.array(total_d_TP) + np.array(total_d_FN) + np.array(total_d_FP))

    return total_d_F1


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