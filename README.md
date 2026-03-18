
# Pedestrian Action Prediction with V-JEPA2

### Self-Supervised World Models for Autonomous Driving

🚗 Predicting pedestrian crossing behavior **before it happens**.

This project investigates how **Meta AI’s V-JEPA2 world model** can be adapted to **pedestrian behavior prediction for autonomous driving systems**.

Instead of relying on complex pipelines (pose estimation → trajectory models → rule engines), this work explores a **world-model-based approach** that leverages **self-supervised video representations trained on massive datasets** to anticipate human actions.

The system predicts whether a pedestrian will **cross the road within the next 1–3 seconds**, a key capability required for **safe autonomous vehicle planning and collision avoidance**.

## 🚀 Key Contributions

• Adapted Meta AI’s **V-JEPA2 world model** for pedestrian behavior prediction  
• Built a **bounding-box aware video prediction system** for autonomous driving  
• Analyzed **temporal evaluation metrics (TTE, confidence stability)** for early anticipation  
• Developed an **end-to-end ML pipeline** for large-scale driving datasets (JAAD, PIE)  
• Demonstrated **early prediction of pedestrian crossing up to 3 seconds ahead**
---------------------------------------------------------------------

# What This Project Demonstrates

This system mirrors a **real-world autonomy stack component**, where predicting human behavior is essential for safe decision-making.

• Applying **self-supervised world models** to safety-critical robotics tasks  
• Designing **pedestrian behavior prediction systems**  
• Building **end-to-end ML pipelines for large video datasets**  
• Implementing **temporal deep learning models for action anticipation**  
• Developing **evaluation frameworks for autonomous driving prediction tasks**

---------------------------------------------------------------------

# Demo

<p align="center">
<img src="assets/demo.gif" width="750">
</p>

Example prediction showing **early anticipation of pedestrian crossing behavior**.

---------------------------------------------------------------------
## Quick Start

```bash
pip install -r requirements.txt
python evals/main.py --config configs/eval/vitl/jaad.yaml
```
---------------------------------------------------------------------

# Problem Motivation

Autonomous vehicles must understand **how humans will behave before it happens**.

Pedestrian intent is often communicated through subtle cues such as:

• head orientation toward traffic  
• walking speed changes  
• proximity to crosswalks  
• traffic signal context  

Traditional approaches rely on **multi-stage pipelines** involving:

- pedestrian detection
- pose estimation
- trajectory forecasting
- rule-based decision logic

These pipelines require **heavy annotation and complex engineering**, limiting generalization.

This project explores whether **world models trained on massive video corpora can transfer their learned visual intuition** to solve pedestrian prediction tasks with **simpler architectures and stronger generalization**.

---------------------------------------------------------------------

# System Overview

<p align="center">
<img src="assets/model_architecture.png" width="800">
</p>

Pipeline:

1. Extract video clips containing pedestrians  
2. Encode frames with **frozen V-JEPA2 encoder**  
3. Predict **future latent scene tokens** using the predictor  
4. Combine context tokens and predicted future tokens  
5. Augment features using **pedestrian bounding box embeddings**  
6. Train a lightweight **attention probe network** for prediction  

---------------------------------------------------------------------

# Model Architecture

## Backbone: V-JEPA2 World Model

• Vision Transformer architecture (ViT-L/16)  
• Trained using **self-supervised predictive learning**  
• Learns structured representations of **scene dynamics and motion**

The backbone remains **frozen**, acting as a general-purpose video world model.

---------------------------------------------------------------------

## Multi-Task Prediction Head

The probe predicts multiple attributes simultaneously.

Q1 – Crossing prediction  
Q2 – Looking vs not looking  
Q3 – Walking vs standing  
Q4 – Designated crossing zone  
Q5 – Intersection context  
Q6 – Traffic signal state  

Multi-task supervision encourages richer representations of pedestrian behavior.


## Why This Approach Works

Traditional models learn from labeled data only.

V-JEPA2 learns from **massive unlabeled video**, capturing:

• motion dynamics  
• human behavior patterns  
• scene structure  

This enables:

✔ Better generalization across environments  
✔ Robustness to occlusion and lighting  
✔ Early anticipation of actions (before they occur)
---------------------------------------------------------------------

# Datasets

## JAAD

Urban driving dataset with pedestrian bounding boxes and crossing labels.

## PIE

Large-scale dataset with detailed pedestrian behavior annotations and scene context.

---------------------------------------------------------------------

# Data Pipeline

<p align="center">
<img src="assets/sampling_and_annotation.png" width="750">
</p>

Clip Sampling

8 frames per clip  
15 FPS  
~0.5 second observation window  
30% overlap  

Prediction Target

Will the pedestrian cross within the next 1 second?

---------------------------------------------------------------------

# Training Configuration

Backbone: V-JEPA2 ViT-L  
Frames per clip: 8  
Frames per second: 15
Resolution: 256×256  
Batch size: 32
Optimizer: AdamW  
Loss: Softmax Focal Loss  

---------------------------------------------------------------------

# Evaluation Framework

<p align="center">
<img src="assets/metrics_big.png" width="750">
</p>

Sample-Level Metrics

• Accuracy  
• Balanced Accuracy  
• Precision  
• F1 Score  
• AUROC  
• mAP  

Instance-Level Metrics

Soft metrics – average probability across samples.

Hard metrics – prediction correct only if all samples agree.

Confidence Delta

Δ_conf = mean(|p_t+1 − p_t|)

Measures temporal stability of predictions.

---------------------------------------------------------------------
# Results

### Core Performance

• Crossing Accuracy: **80%**  
• Crossing Recall: **70%**

### Key Observations

✔ Model successfully anticipates crossing **up to 3 seconds before event**  
✔ Strong performance despite **class imbalance and subtle behavioral cues**  
✔ Multi-task supervision improves representation quality  

### Why This Matters

Early prediction is critical for:

• collision avoidance  
• safe braking decisions  
• real-time planning systems  

These results show that **world-model-based representations transfer effectively to autonomous driving tasks**.

---------------------------------------------------------------------

# Visualization

## Attention Heatmaps

<p align="center">
<img src="assets/attention_heatmaps.png" width="750">
</p>

The probe attends to:

• pedestrian motion patterns  
• crosswalk regions  
• traffic signals  

---------------------------------------------------------------------

## Anticipation Performance

<p align="center">
<img src="assets/tte_curve.png" width="750">
</p>

Performance is evaluated against **Time-To-Event (TTE)**.

The model anticipates crossing **up to 3 seconds ahead**.

---------------------------------------------------------------------

# Failure Case Analysis (Autonomy Research Style)

Understanding failure scenarios is critical for safe autonomous systems.

Common failure modes observed:

• **Heavy occlusion** – pedestrians partially hidden behind vehicles  
• **Night-time lighting** – limited visibility conditions  
• **Group crossings** – complex multi-agent interactions  
• **Ambiguous body language** – hesitation near curb

Future improvements could involve:

• longer temporal context  
• trajectory reasoning modules  
• multimodal sensors (LiDAR / radar)

---------------------------------------------------------------------

# Scenario Visualization

Autonomous driving models must reason about complex scenes.

Example scenario analysis:

Scenario: Pedestrian approaching crosswalk.

Model observations:

Frame 1 – pedestrian walking parallel to road  
Frame 2 – head orientation toward traffic  
Frame 3 – slowdown near curb  

Prediction probability:

t = -3s → 0.12  
t = -2s → 0.35  
t = -1s → 0.71  

The model correctly anticipates the crossing **before the step onto the road**.

---------------------------------------------------------------------

# Interactive Prediction Plots

Prediction probabilities across time provide insight into model behavior.

Example temporal prediction:

Time to event vs probability

3s before crossing → 0.10  
2s before crossing → 0.32  
1s before crossing → 0.68  
0s (crossing) → 0.91

Such plots help evaluate:

• early prediction ability  
• prediction stability  
• reaction time for planning systems

---------------------------------------------------------------------

# Tech Stack

Machine Learning

PyTorch  
Vision Transformers  
Self-Supervised Learning  
Multi-Task Learning  
Attention Mechanisms

Data Engineering

Video preprocessing pipelines  
Frame sampling and annotation alignment  
Bounding box encoding

Systems

CUDA GPU training  
Python  
NumPy  
OpenCV  
Matplotlib

---------------------------------------------------------------------

# Key Skills Demonstrated

Machine Learning Engineering

• Self-supervised video representation learning  
• Temporal deep learning models  
• Multi-task behavior prediction  
• Autonomous driving perception and prediction tasks  

Software Engineering

• Modular PyTorch training pipelines  
• Dataset ingestion and preprocessing systems  
• GPU accelerated training workflows  
• Experiment reproducibility and evaluation tooling  

Research Engineering

• Implementing state-of-the-art research models  
• Designing reproducible ML experiments  
• Analyzing failure cases and model behavior  

---------------------------------------------------------------------

# Repository Structure

pedestrian-action-anticipation-vjepa/
├── assets/                        # Figures used in the README/paper-style visuals
├── configs/                       # Evaluation and inference configs
│   ├── eval/vitl/
│   └── inference/vitl/
├── evals/                         # Task-specific evaluation code
│   ├── action_anticipation_frozen/
│   └── hub/
├── src/                           # Core model, dataset, mask, and utility code
│   ├── datasets/
│   ├── hub/
│   ├── masks/
│   ├── models/
│   └── utils/
├── your_data/                     # JAAD / PIE CSV splits and bbox annotations
├── .gitignore
├── README.md
└── requirements.txt

---------------------------------------------------------------------

# Future Work

• Fine-tuning the V-JEPA2 backbone  
• Multi-modal fusion with LiDAR and map data  
• Pedestrian trajectory prediction  
• Real-time deployment for autonomous vehicles  

---------------------------------------------------------------------
## Acknowledgment

This project builds upon and adapts Meta AI’s V-JEPA2 world model, leveraging their official research and implementation:

https://github.com/facebookresearch/vjepa2


# Author

Aditya Sanjaykumar Patel

MS Computer Science  
San Jose State University

Machine Learning Engineer focused on:

• autonomous driving AI  
• video understanding  
• world models  
• computer vision

LinkedIn: https://www.linkedin.com/in/adityapatel149
Email: imadityapatel149@gmail.com
