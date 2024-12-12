import numpy as np
import matplotlib.pyplot as plt
from evaluation_function import segment_f1_drinking

def generate_test_data(num_samples=5):
    """
    Generate test data for segment_f1 function.
    
    Each sample is a sequence of labels where:
    0 = Background
    1 = Drinking
    
    Parameters:
    -----------
    num_samples : int, optional (default=5)
        Number of samples to generate
    
    Returns:
    --------
    y_pre_list : list of numpy arrays
        Predicted labels
    y_gt_list : list of numpy arrays
        Ground truth labels
    """
    y_pre_list = []
    y_gt_list = []
    
    for _ in range(num_samples):
        # Sample length between 50 and 200
        sample_length = np.random.randint(5000, 20001)
        
        # Ground truth labels
        y_true = np.zeros(sample_length, dtype=int)
        
        # Generate drinking segments
        num_drink_segments = np.random.randint(0, 2)
        for _ in range(num_drink_segments):
            drink_start = np.random.randint(0, sample_length - 10)
            drink_length = np.random.randint(5, 16)
            # Ensure no overlap with eating segments
            if np.all(y_true[drink_start:drink_start + drink_length] == 0):
                y_true[drink_start:drink_start + drink_length] = 2
        
        # Predicted labels (with some intentional errors)
        y_pred = y_true.copy()
        
        # Introduce some false positives and false negatives
        error_rate = 0.1
        noise_mask = np.random.random(sample_length) < error_rate
        y_pred[noise_mask] = np.random.choice([0, 1], size=np.sum(noise_mask))
        
        y_pre_list.append(y_pred)
        y_gt_list.append(y_true)
    
    return y_pre_list, y_gt_list

# Example usage
y_pre_list, y_gt_list = generate_test_data(num_samples=5)


def plot_test_data(y_pre_list, y_gt_list):
    fig, axs = plt.subplots(len(y_pre_list), 2, figsize=(12, 3*len(y_pre_list)))
    
    for i, (y_pred, y_true) in enumerate(zip(y_pre_list, y_gt_list)):
        axs[i, 0].imshow(y_true.reshape(1, -1), aspect='auto', cmap='viridis')
        axs[i, 0].set_title(f'Ground Truth Sample {i+1}')
        axs[i, 0].set_yticks([])
        
        axs[i, 1].imshow(y_pred.reshape(1, -1), aspect='auto', cmap='viridis')
        axs[i, 1].set_title(f'Predicted Sample {i+1}')
        axs[i, 1].set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Uncomment to visualize the test data
plot_test_data(y_pre_list, y_gt_list)

# Evaluate the test data
segment_f1_drinking(y_pre_list, y_gt_list)
