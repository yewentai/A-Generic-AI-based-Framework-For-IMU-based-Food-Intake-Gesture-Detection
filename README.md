# A Generic AI-Based Framework for IMU-Based Food Intake Gesture Detection

**Author**: Wentai Ye  
**Daily Supervisor**: Chunzhuo Wang  
**Promotor**: Bart Vanrumste  
**Institution**: KU Leuven  

## Project Overview

This repository contains the code and resources for the master's thesis project titled: _"A Generic AI-Based Framework for IMU-Based Food Intake Gesture Detection."_

### Research Background

Maintaining a balanced dietary intake is essential for a high-quality lifestyle and is instrumental in preventing numerous eating-related diseases affecting millions globally. Issues such as malnutrition, metabolic disorders, and obesity are common consequences of unbalanced diets, underlining the importance of tracking dietary intake.

Recent research has explored the use of novel sensors for food intake behavior monitoring. Various sensors, including acoustic, camera, Electromyography (EMG), and inertial sensors, have been tested for this purpose. Among these, inertial sensors, commonly embedded in smartwatches and other wearables, have gained popularity due to their accessibility. However, detecting eating gestures with IMU (Inertial Measurement Unit) sensors is challenging due to sensitivity to device orientation, often resulting in models that perform poorly on data from different orientations. Additionally, manual pre-processing of IMU data to mirror the left and right hands is cumbersome.

### Thesis Objective

The objective of this thesis is to design a generic AI-based framework to process IMU data for food intake gesture detection, minimizing the impact of sensor orientation differences and removing manual mirroring of left-hand data. This will involve:

1. Conducting a thorough literature review on food intake behavior monitoring with a focus on IMU sensors.
2. Designing and executing experiments to collect relevant data.
3. Developing a machine learning model capable of handling orientation variations and hand variations (left and right) in IMU data to accurately detect eating gestures.
4. Evaluating the model's performance on the collected data and comparing it with existing approaches using segment-wise methods.

## Repository Structure

```
.
├── LICENSE                     # License information
├── README.md                   # Project documentation (this file)
├── analyze_IMU.py              # Script for analyzing IMU data
├── augmentation.py             # Data augmentation techniques for IMU data
├── checkpoint.py               # Checkpoint handling for model training
├── checkpoints/                # Directory for storing model checkpoints
├── dataset/                    # Contains the datasets used in the project
│   ├── DX/                     # Dataset variant DX
│   │   ├── 00_ReadMe.txt       # Description of DX dataset
│   │   ├── DX-I/               # DX-I dataset
│   │   │   ├── X_L.pkl         # IMU data for left hand
│   │   │   ├── X_R.pkl         # IMU data for right hand
│   │   │   ├── Y_L.pkl         # Labels for left hand
│   │   │   ├── Y_R.pkl         # Labels for right hand
│   │   └── DX-II/              # DX-II dataset
│   ├── FD/                     # Dataset variant FD
│   │   ├── 00_ReadMe.txt       # Description of FD dataset
│   │   ├── FD-I/               # FD-I dataset
│   │   ├── FD-II/              # FD-II dataset
│   │   ├── MO/                 # Additional motion data
├── datasets.py                 # Dataset processing script
├── evaluation.py               # Evaluation metrics and functions
├── figs/                       # Figures and visualizations
│   ├── fig1.png                # Sample result visualization
│   ├── fig2.png
│   ├── fig3.png
├── logs/                       # Log files from model training runs
│   ├── slurm-*.out             # Logs from SLURM job submissions
├── model_cnnlstm.py            # CNN-LSTM model implementation
├── model_mstcn.py              # MS-TCN model implementation
├── post_processing.py          # Post-processing of model outputs
├── pre_processing.py           # Pre-processing of IMU data
├── project_a100.slurm          # SLURM script for training on A100 GPU
├── project_debug.slurm         # SLURM script for debugging
├── project_h100.slurm          # SLURM script for training on H100 GPU
├── result/                     # Directory to store results
├── sync.ffs_db                 # Synchronization metadata
├── test.py                     # Testing script
├── test_eval_bi.py             # Evaluation script for binary classification
├── test_eval_tri.py            # Evaluation script for ternary classification
├── train.py                    # Model training script
```

### Key Components

- **Data Processing**: Scripts for loading, augmenting, and preprocessing IMU data.
- **Model Training**: Implementation of CNN-LSTM and MS-TCN models to detect food intake gestures.
- **Evaluation**: Performance metrics and analysis scripts to compare the trained models.
- **SLURM Job Scripts**: Pre-configured job submission scripts for running experiments on high-performance computing clusters.
- **Visualization**: Tools for generating and saving figures related to the research.

## Installation and Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/your-repo-link.git
   cd your-repo-link
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Train the model:

   ```sh
   python train.py
   ```
