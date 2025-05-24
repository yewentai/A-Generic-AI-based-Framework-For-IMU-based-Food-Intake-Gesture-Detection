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

### Datasets

This project utilizes three distinct datasets for training and evaluation:

1. **DX Dataset**
   - A dataset containing IMU data from both left and right hands
   - Includes two variants:
     - **DX-I**: 8.9 hours of data from 13 participants in semi-controlled environments
       - Contains 410 drinking gestures (101 left hand, 266 right hand, 43 two hands)
       - Participants engaged in various activities while drinking
     - **DX-II**: 45.2 hours of data from 7 participants in free-living environments
       - Contains 304 drinking gestures (142 left hand, 152 right hand, 10 two hands)
       - 6.5±2.0 consecutive hours per participant
   - Uses Shimmer3 IMU sensors with 6 DoF (accelerometer and gyroscope)
   - Sampling frequency: 64 Hz
   - Binary classification: drinking gestures (1) vs. null activities (0)

   - The dataset is available at [DX Dataset](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/W0H2A4).

2. **FD Dataset**
   - A comprehensive dataset containing IMU data from both left and right hands
   - Includes three variants:
     - **FD-I**: 251.70 hours of data from 34 participants
       - Contains 4,568 eating and 1,100 drinking gestures
       - Includes four eating styles: forks & knives, chopsticks, spoon, and hands
       - Covers both individual and social eating scenarios
     - **FD-II**: 261.68 hours of data from 27 participants
       - Contains 2,723 eating gestures
       - Serves as a hold-out dataset
     - **MO (Meal-Only)**: 46 meal sessions from 46 participants
       - Contains 2,894 eating and 763 drinking activities
       - No participant overlap with FD-I and FD-II
   - Uses Shimmer3 IMU sensors with 6 DoF
   - Sampling frequency: 64 Hz
   - Ternary classification: eating gestures (1), drinking gestures (2), and others (0)

   - The dataset is available at [FD Dataset](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/CN8VBB).

3. **UK Biobank Pre-trained Model**

   - the pre-trained model is available at [UK Biobank Pre-trained Model](https://github.com/OxWearables/ssl-wearables/tree/main/model_check_point).

Each dataset contributes unique characteristics to the research, allowing for comprehensive evaluation of the proposed framework across different scenarios and use cases.

### Thesis Objective

The objective of this thesis is to design a generic AI-based framework to process IMU data for food intake gesture detection, minimizing the impact of sensor orientation differences and removing manual mirroring of left-hand data. This will involve:

1. Conducting a thorough literature review on food intake behavior monitoring with a focus on IMU sensors.
2. Designing and executing experiments to collect relevant data.
3. Developing a machine learning model capable of handling orientation variations and hand variations (left and right) in IMU data to accurately detect eating gestures.
4. Evaluating the model's performance on the collected data and comparing it with existing approaches using segment-wise methods.

## Repository Structure

.
├── LICENSE                     # License information
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── components/                 # Core components and utilities
├── analysis/                   # Analysis scripts and tools
├── dataset/                    # Contains the datasets used in the project
│   ├── DX/                     # Dataset variant DX
│   │   ├── 00_ReadMe.txt       # Description of DX dataset
│   │   ├── DX-I/               # DX-I dataset
│   │   │   ├── X_L.pkl         # IMU data for left hand
│   │   │   ├── X_R.pkl         # IMU data for right hand
│   │   │   ├── Y_L.pkl         # Labels for left hand
│   │   │   └── Y_R.pkl         # Labels for right hand
│   │   └── DX-II/              # DX-II dataset
│   ├── FD/                     # Dataset variant FD
│   │   ├── 00_ReadMe.txt       # Description of FD dataset
│   │   ├── FD-I/               # FD-I dataset
│   │   ├── FD-II/              # FD-II dataset
│   │   └── MO/                 # MO dataset
├── logs/                       # Log files from model training runs
├── result/                     # Directory to store results
├── slurm_a100.sh               # SLURM script for training on A100 GPU
├── slurm_h100.sh               # SLURM script for training on H100 GPU
├── sync.ffs_db                 # Synchronization metadata
├── analyze_predefined_fold.py  # Script for analyzing predefined folds
├── analyze_raw_data.py         # Script for analyzing raw IMU data
├── analyze_training_curve.py   # Script for analyzing training curves
├── clean.py                    # Data cleaning utilities
├── dl_analyze.py               # Deep learning analysis utilities
├── dl_train.py                 # Deep learning training script
├── dl_validate.py              # Deep learning validation script
├── test_dataset.py             # Dataset testing utilities
├── test_eval_mono.py           # Evaluation script for monocular data
├── test_eval_tri.py            # Evaluation script for ternary classification
├── tl_fine_tune.py             # Transfer learning fine-tuning script
├── tl_pre_train_simclr.py      # SimCLR pre-training script
├── tl_pre_train_vae.py         # VAE pre-training script
├── tl_validate_finetune.py     # Transfer learning validation script
└── update_edited_date.py       # Utility for updating file dates

### Key Components

- **Data Processing**: Scripts for loading, cleaning, and preprocessing IMU data
- **Deep Learning**: Implementation of various deep learning models and training pipelines
- **Transfer Learning**: Pre-training and fine-tuning scripts using SimCLR and VAE approaches
- **Analysis**: Comprehensive analysis tools for data and model performance
- **Evaluation**: Performance metrics and analysis scripts for different classification tasks
- **SLURM Job Scripts**: Pre-configured job submission scripts for running experiments on high-performance computing clusters

## Installation and Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/yewentai/A-Generic-AI-based-Framework-For-IMU-based-Food-Intake-Gesture-Detection
   ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Training Options:

  ```sh
  python batch_train_comb.py
  python batch_train_iter.py
  ```

4. Evaluation:

   ```sh
   python validate.py
   ```

5. Analysis:

   ```sh
    python analyze_validation_cross.py
    python analyze_validation_multi.py
    python analyze_validation_solo.py
    ```
