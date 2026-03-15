# vjepa2-action-anticipation

# Pedestrian Action Prediction using V-JEPA2
## What This Project Demonstrates

• Applying large-scale self-supervised world models to autonomous driving tasks  
• Designing prediction systems for pedestrian behavior anticipation  
• Building end-to-end ML pipelines for large video datasets  
• Implementing evaluation frameworks for safety-critical prediction systems


### Self-Supervised World Models for Autonomous Driving Safety

🚗 Predicting whether a pedestrian will cross the road **before it happens**.

This project explores how **Meta AI’s V-JEPA2 world model** can be adapted for **pedestrian intention prediction** in autonomous driving.

Instead of relying on complex multi-stage pipelines (pose estimation, trajectory models, etc.), this work leverages **self-supervised video representations trained on massive datasets** to anticipate human behavior.

---

## Demo

<p align="center">
<img src="assets/demo.gif" width="700">
</p>

Example prediction showing early crossing anticipation.

---

# Project Highlights

• Built a **pedestrian behavior anticipation system** using a frozen **V-JEPA2 world model**  
• Developed a **multi-query probing architecture** for multi-task prediction  
• Implemented **bounding-box aware video representation learning**  
• Designed **evaluation protocols for early anticipation performance (TTE curves)**  
• Trained and evaluated on **JAAD and PIE autonomous driving datasets**

---

# Why This Matters

Autonomous vehicles must anticipate human behavior **before it happens**.

Pedestrian actions contain subtle cues:

• head orientation  
• gait speed  
• proximity to crosswalk  
• traffic signal context

Traditional pipelines require **heavy annotations and multiple subsystems**, which limits generalization.

This work demonstrates that **world models trained on massive video corpora can transfer visual intuition to safety-critical tasks.**

---

# Architecture Overview

<p align="center">
<img src="assets/model\_architecture.png" width="750">
</p>

### Pipeline

1️⃣ Extract video clips and pedestrian bounding boxes  
2️⃣ Encode frames using **frozen V-JEPA2 encoder**  
3️⃣ Predict latent **future scene tokens**  
4️⃣ Concatenate context + predicted future tokens  
5️⃣ Augment tokens with **pedestrian bounding box embeddings**  
6️⃣ Train lightweight **attention probe heads**

The backbone remains frozen, enabling **efficient adaptation to new domains.**

---

# Model Design

## Backbone

• V-JEPA2 ViT-L/16 world model  
• Pretrained on billions of video frames  
• Encoder + predictor **kept frozen**

## Probe Network

Attention-based multi-query probing head.

Each query predicts a task:

|Query|Task|
|-|-|
|Q1|Crossing prediction|
|Q2|Looking vs not looking|
|Q3|Walking vs standing|
|Q4|Designated crossing|
|Q5|Intersection context|
|Q6|Traffic signal state|

Training updates **only probe parameters**, leveraging the pretrained world model.

---

# Datasets

## JAAD

Joint Attention in Autonomous Driving

• Urban driving videos  
• Pedestrian bounding boxes  
• Crossing intention labels

## PIE

Pedestrian Intention Estimation dataset

• Longer clips  
• More pedestrians  
• Additional scene annotations

Both datasets are widely used for **pedestrian intention prediction research.**

---

# Data Pipeline

<p align="center">
<img src="assets/data\_pipeline.png" width="700">
</p>

Clips are sampled as:

```
8 frames per clip
15 fps
~0.5 seconds observation window
30% overlap
```

Prediction target:

```
Will the pedestrian cross within 1 second?
```

Bounding boxes are extracted and normalized for each frame.

---

# Training

|Setting|Value|
|-|-|
|Backbone|V-JEPA2 ViT-L|
|Frames per clip|8|
|Resolution|256×256|
|Batch size|8|
|Optimizer|AdamW|
|Loss|Softmax Focal Loss|
|Gamma|2|

Class imbalance is handled using **focal loss weighting.**

---

# Evaluation Metrics

### Sample-level (Base Metrics)

• Accuracy  
• Balanced Accuracy  
• Precision  
• F1 Score  
• AUROC  
• mAP

### Instance-level Metrics

Pedestrian instances contain multiple samples.

Two evaluation strategies:

**Soft metrics**

```
Average probability across samples
```

**Hard metrics**

```
Prediction correct only if all samples agree
```

### Confidence Delta

Measures temporal stability:

```
Δ\_conf = average change in prediction probabilities across frames
```

Lower values indicate **more stable predictions.**

---

# Preliminary Results

|Task|Accuracy|Recall|
|-|-|-|
|Crossing|62-66%|~63%|
|Looking|up to 83%|up to 84%|
|Walking|70-79%|59-72%|
|Intersection|91-95%|70-75%|

Results demonstrate that **V-JEPA2 world model features transfer effectively to pedestrian behavior prediction.**

---

# Visualizations

## Attention Heatmaps

<p align="center">
<img src="assets/attention\_heatmaps.png" width="700">
</p>

The probe focuses on:

• pedestrian body motion  
• crosswalk regions  
• traffic signals

---

## Anticipation Performance

<p align="center">
<img src="assets/tte\_curve.png" width="700">
</p>

Performance is evaluated against **Time-To-Event (TTE)**.

The model can anticipate crossing **up to 3 seconds ahead.**

---

# Tech Stack

### Machine Learning

• PyTorch  
• V-JEPA2  
• Vision Transformers  
• Multi-task learning  
• Focal loss  
• Attention mechanisms

### Data Engineering

• Video preprocessing pipelines  
• Frame sampling \& annotation alignment  
• Bounding box encoding  
• dataset loaders for large video datasets

### Tools

• CUDA / GPU training  
• Python  
• NumPy  
• Matplotlib

---

# Key Skills Demonstrated

### Machine Learning Engineering

✔ Large-scale video models  
✔ Self-supervised representation learning  
✔ Multi-task neural architectures  
✔ Model evaluation and benchmarking

### Software Engineering

✔ Modular PyTorch training pipelines  
✔ Dataset preprocessing pipelines  
✔ GPU training optimization  
✔ reproducible research code

### Research Skills

✔ Literature review  
✔ experiment design  
✔ ablation studies  
✔ scientific writing

---

# Repository Structure

```
pedestrian-action-prediction-vjepa2/

datasets/
    JAAD/
    PIE/

models/
    vjepa\_probe.py
    probe\_heads.py

training/
    train.py
    losses.py
    dataloader.py

evaluation/
    metrics.py
    tte\_analysis.py

visualization/
    attention\_maps.py

notebooks/
    exploratory\_analysis.ipynb
```

---

# Future Work

• Fine-tuning the V-JEPA2 backbone  
• Multi-modal integration (LiDAR + map data)  
• Trajectory prediction models  
• Real-time deployment for AV systems

---

# References

V-JEPA2:  
https://arxiv.org/abs/2506.09985

JAAD Dataset  
PIE Dataset

---

# Author

**Aditya Sanjaykumar Patel**

MS Computer Science  
San Jose State University

Machine Learning Engineer focused on **video understanding, world models, and autonomous driving AI**.

LinkedIn: (add link)  
Email: aditya.s.patel@sjsu.edu

