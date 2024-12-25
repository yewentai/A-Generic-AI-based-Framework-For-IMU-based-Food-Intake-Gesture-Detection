# A Generic AI-Based Framework for IMU-Based Food Intake Gesture Detection

**Author**: Wentai Ye

**Daily Supervisor**: Chunzhuo Wang

**Promotor**: Bart Vanrumste

**Institution**: KU Leuven

## Project Overview

This repository contains the code and resources for the master's thesis project titled: _"A Generic AI-Based Framework for IMU-Based Food Intake Gesture Detection."_

### Research Background

Maintaining a balanced dietary intake is essential for a high-quality lifestyle and is instrumental in preventing numerous eating-related diseases affecting millions globally. Issues such as malnutrition, metabolic disorders, and obesity are common consequences of unbalanced diets, underlining the importance of tracking dietary intake.

Recent research has explored the use of novel sensors for food intake behavior monitoring. Various sensors, including acoustic, camera, Electromyography (EMG), and inertial sensors, have been tested for this purpose. Among these, inertial sensors, commonly embedded in smartwatches and other wearables, have gained popularity due to their accessibility. However, detecting eating gestures with IMU (Inertial Measurement Unit) sensors is challenging due to sensitivity to device orientation, often resulting in models that perform poorly on data from different orientations. Besides, manual pre-processing of IMU data to mirror the left and right hands is cumbersome.

### Thesis Objective

The objective of this thesis is to design a generic AI-based framework to process IMU data for food intake gesture detection, minimizing the impact of sensor orientation differences and removing manual mirroring left hand data. This will involve:

1. Conducting a thorough literature review on food intake behavior monitoring with a focus on IMU sensors.
2. Designing and executing experiments to collect relevant data.
3. Developing a machine learning model capable of handling orientation variations and hand variations (left and right) in IMU data to accurately detect eating gestures.
4. Evaluating the model's performance on the collected data and comparing it with existing approaches with segement wise methods.

## Repository Structure

- **figs/**: Contains results and figures generated during the project.
- **models/**: Contains the trained models and model weights.
- **dataset/**: Contains the datasets used in the project.
