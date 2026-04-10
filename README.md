# Pedestrian Action Prediction Using VJEPA-2 Self-Supervised World Model

### Demo

Examples showing **early anticipation of pedestrian crossing behavior**.

<table>
  <tr>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/59a3509c-674a-44c9-9b75-2eaffd23007b" controls width="100%"></video>
      <br>
      <sub><b>Example 1:</b> Early crossing anticipation in urban traffic</sub>
    </td>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/21df32f4-e365-455f-817b-56ad942497b3" controls width="100%"></video>
      <br>
      <sub><b>Example 2:</b> Pedestrian intent prediction before motion onset</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/0a8a0edb-e944-496a-a7f7-c3d0f5ba7453" controls width="100%"></video>
      <br>
      <sub><b>Example 3:</b> Anticipating crossing under complex scene dynamics</sub>
    </td>
    <td align="center" width="50%">
      <video src="https://github.com/user-attachments/assets/1f2f1fec-f7ab-4b03-b326-2a19366afedd" controls width="100%"></video>
      <br>
      <sub><b>Example 4:</b> Robust prediction across varied pedestrian behavior</sub>
    </td>
  </tr>
</table>

> **Note:** GitHub README does not reliably support `autoplay` for embedded videos, so viewers will usually need to press play manually.

---

## Overview
**Objective**: Predicting pedestrian crossing behavior **before it happens**.

This project presents a pedestrian action anticipation system built on **V-JEPA2 world model**, a self supervised world model for video understanding. The goal is to anticipate whether a pedestrian is liekly to cross the road in the next 1 to 3 seconds into the future using partial video observations(0.5s of context), enabling safer and more proactive planning in real-world driving systems. 

### Quick Start
To run inference on a video without annotations:

```bash
python -m py_app.main `
    --config configs/inference/vitl/pie.yaml `
    --encoder-model path/to/your/engines/encoder.engine `
    --classifier-model path/to/your/engines/classifier_1.engine `
    --video path/to/your/test.mp4 `
    --detector-conf 0.3 `
    --max-boxes 10 `
    --detector engines/yolo26m.engine `
    --anticipation-time 1.0 `
    --use-depth --depth-model path/to/your/engines/da3-small.engine `
    --display --render-scale 0.5

```

To run using .pt files:

```bash
python -m py_app.main `
    --config configs/inference/vitl/pie.yaml `
    --classifier-model your_folder/evals/vitl/pie/action_anticipation_frozen/pie-vitl16/latest.pt `
    --sweep-idx 1 `
    --video path/to/your/test.mp4 `
    --detector-conf 0.3 `
    --max-boxes 10 `
    --detector engines/yolo26n.pt `
    --anticipation-time 1.0 `
    --display --render-scale 0.5
    --use-depth --depth-model path/to/your/engines/da3-small.engine `

```

To run test evaluation on PIE benchmark dataset:
```bash
pip install -r requirements.txt
python -m evals.main --fname configs/eval/vitl/pie.yaml --devices cuda:0

```

The configs allow to train multiple probes in parallel with various optimization parameters. Change filepaths as needed (e.g. folder, checkpoint, dataset_train, dataset_val) to match locations of data and the downloaded vitl.pt vjepa2 checkpoint on your local filesystem. Change # nodes and local batch size as needed to not exceed available GPU memory. Download the probe checkpoints, rename it to 'latest.pt', and create a folder with the checkpoint inside, with the format matching the variables in the config:

```bash
[folder]/[eval_name]/[tag]/latest.pt
```

Comment out the following code from evals/main.py if **NOT** running locally on a Windows machine with single GPU:

```bash

    import torch.distributed as dist
    import tempfile

    temp_dir = tempfile.gettempdir()
    init_file = os.path.join(temp_dir, "shared_init_file")

    dist.init_process_group(
        backend="gloo",  # use 'nccl' only if running on GPU with compatible setup
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1
    )

```


---

## Research Motivation

Pedestrian intent is expressed through subtle cues such as head orientation, motion patterns, proximity to crosswalks, and scene context.

Traditional approaches rely entirely on large-scale manual annotations or on complex pipelines (pose estimation → vehicle odometer data -> trajectory models → rule engines).
By leveraging **self-supervised video representation learning**, this work investigates how well self-supervised world models trained on massive datasets can capture pedestrian intent, motion cues, and scene context.

---

## System Overview

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

---

## Model Architecture

### Backbone: V-JEPA2 World Model

• Vision Transformer architecture (ViT-L/16)  
• Trained using **self-supervised predictive learning**  
• Learns structured representations of **scene dynamics and motion**

The backbone remains **frozen**, acting as a general-purpose video world model.

### Multi-Task Prediction Head

The probe predicts multiple attributes simultaneously.

Q1 – Crossing prediction  
Q2 – Walking vs standing  
Q3 – Intersection context  
Q4 – Traffic signal state  

Multi-task supervision encourages richer representations of pedestrian behavior.


### Why This Approach Works

Traditional models learn from labeled data only. V-JEPA2 learns from **massive unlabeled video**, capturing:
- motion dynamics  
- human behavior patterns  
- scene structure  

This enables:

- Better generalization across environments  
- Robustness to occlusion and lighting

---

## Datasets

### JAAD

Urban driving dataset with pedestrian bounding boxes and crossing labels.

### PIE

Large-scale dataset with detailed pedestrian behavior annotations and scene context.

---

## Data Pipeline

<p align="center">
<img src="assets/sampling_and_annotation.png" width="750">
</p>

Clip sampling:

-   8 frames per clip
-   15 FPS
-   Approximately 0.5 second observation window
-   30 percent overlap

Prediction target:

-   Will the pedestrian cross within the next 1-3 seconds?


---

## Training Configuration

Backbone: V-JEPA2 ViT-L  
Frames per clip: 8  
Frames per second: 15
Resolution: 256×256  
Batch size: 32
Optimizer: AdamW  
Loss: Softmax Focal Loss  

---

## Evaluation Framework

<p align="center">
<img src="assets/metrics_big.png" width="750">
</p>

Sample
Sample level metrics:

-   Accuracy
-   Balanced accuracy
-   Precision
-   F1 score
-   AUROC
-   mAP

Instance level metrics:

-   Soft aggregation
-   Hard agreement

Confidence stability:

Delta conf equals mean absolute difference between consecutive probabilities.

---

## Results

### Core Performance on PIE test split

| Model        | mAP   | bAcc  | AUC   | Acc  | Prec | F1   | soft_F1 / hard_F1 | conf∆ max/avg |
|-------------|-------|-------|-------|------|------|------|-------------------|--------------|
| SF-GRU      | 0.75  | 0.75  | 0.87  | 0.83 | 0.79 | 0.65 | 0.67/0.43         | 0.10/0.04    |
| PCPA        | 0.79  | 0.81  | 0.89  | 0.86 | 0.81 | 0.74 | 0.76/0.46         | 0.17/0.07    |
| BiPed       | 0.84  | 0.84  | 0.93  | 0.89 | 0.84 | **0.79** | 0.82/0.51     | 0.16/0.06    |
| PedFormer   | **0.88** | **0.85** | **0.94** | **0.89** | **0.85** | **0.79** | **0.82/0.65** | **0.12/0.04** |
| Model        | mAP   | bAcc  | AUC   | Acc  | Prec | F1   | soft_F1 / hard_F1 | conf∆ max/avg |
| Ours (Sweep 0) | 0.799 | **0.860** | **0.922** | **0.866** | **0.706** | **0.771** | **0.799 / 0.637** | 0.366 / 0.054 |
| Ours (Sweep 1) | 0.796 | 0.848 | 0.914 | 0.850 | 0.674 | 0.750 | 0.759 / **0.648** | **0.147 / 0.038** |
| Ours (Sweep 2) | **0.825** | 0.850 | 0.918 | 0.855 | 0.687 | 0.755 | 0.786 / 0.638 | 0.159 / **0.035** |

> Note: The sweeps correspond to parameter sweeps where the model was trained with different learning rates.


#### Strengths of Each Sweep

**Sweep 0 (Best Overall Performance)**

-   Highest AUC, accuracy, and F1 score
-   Strongest performance across soft metrics
-   Best general-purpose model for prediction quality

**Sweep 1 (Most Stable and Conservative)**

-   Best performance on all hard metrics
-   Lowest conf_delta_max indicating stable predictions
-   More conservative behavior under strict evaluation

**Sweep 2 (Best Ranking and Temporal Smoothness)**

-   Highest mAP (best ranking performance)
-   Lowest conf_delta_avg indicating smooth temporal predictions
-   Good balance between performance and stability


#### Which Sweep to Use

-   Use Sweep 0 for best overall performance
-   Use Sweep 1 for more stable and conservative predictions
-   Use Sweep 2 for smoother temporal predictions and better ranking

#### Inference Usage

You can select the sweep during inference using:
```bash
    --sweep-idx 0   # Best overall performance
    --sweep-idx 1   # More stable / conservative predictions
    --sweep-idx 2   # Smoother temporal predictions
```
Example:
```bash
    python video_inference.py \
      --config configs/inference/vitl/pie.yaml \
      --classifier-model path/to/checkpoint.pt \
      --video path/to/video.mp4 \
      --sweep-idx 0
```

### Key Observations

- Model successfully anticipates crossing **up to 3 seconds before event**  
- Strong performance despite **class imbalance and subtle behavioral cues**  
- Multi-task supervision improves representation quality  

### Why This Matters

Early prediction is critical for:

• collision avoidance  
• safe braking decisions  
• real-time planning systems  

These results show that **world-model-based representations transfer effectively to autonomous driving tasks**.

---

## Visualization

### Attention Heatmaps
The probe attends to:
- pedestrians 
- possible regions for more pedestrians
- crosswalk regions  
- traffic signals  
- possible regions for more signs/traffic signals. (Looking at the top of street pole)

The attention maps show that, across all 7 context frames, the probe consistently focuses on semantically meaningful spatial regions corresponding to pedestrians, crosswalk areas, traffic signals, and other relevant street-scene cues. The attended regions align with the locations of visible pedestrians and traffic infrastructure, while also highlighting nearby candidate areas where additional pedestrians or signals may appear, such as crosswalk entrances and the upper parts of street poles. This suggests that the model is not attending randomly, but is tracking task-relevant regions over time in order to form its future-frame prediction in latent space.

Attention heatmap across frames and for "Future frame in latent space (predictor-output)" overlayed on last context frame:
<p align="center">
<img src="assets/attn_grid.png" width="750">
</p>



The intersection map, overlaid on the last context frame, may highlight regions that appear empty or irrelevant at that final timestep. However, these regions correspond to locations that were attended to in earlier frames, where pedestrians, crosswalks, traffic signals, or other relevant cues were present. Thus, the intersection captures temporally consistent areas of importance across all 7 frames, even if those regions are no longer occupied in the final frame. This indicates that the model maintains attention on semantically meaningful locations over time, rather than relying solely on instantaneous visual content.

Intersection of attention regions overlayed on last context frame:
<p align="center">
<img src="assets/attn_intersection.png" width="350">
</p>



### Anticipation Performance

<p align="center">
<img src="assets/tte_curve.png" width="750">
</p>

Performance is evaluated against **Time-To-Event (TTE)**.

We observe a decrement across all metrics with increase in anticipation time / time-to-event.

---------------------------------------------------------------------

## Failure Case Analysis

Understanding failure scenarios is critical for safe autonomous systems.
Common failure modes observed:

- **Heavy occlusion** : Pedestrians partially occluded by vehicles or scene elements reduce visibility of key motion cues, leading to uncertain or delayed predictions.
- **Group interactions** : Multi-agent scenarios introduce complex social dynamics, where individual intent becomes harder to isolate from group behavior.
- **Ambiguous body language** : Subtle or hesitant movements near the curb make intent difficult to infer, especially when clear motion signals are absent.
- **Implicit vehicle-motion bias** : The model exhibits a correlation between ego-vehicle motion and predicted pedestrian intent. When the vehicle is moving, predictions are biased toward *not crossing*, whereas stationary scenes increase *crossing* predictions. This likely arises from dataset priors where vehicle motion correlates with pedestrian behavior.

In real-world driving data, vehicle motion and pedestrian actions are tightly coupled. Vehicles slow down or stop because pedestrians are about to cross, and pedestrians often delay crossing because vehicles are moving. As a result, both signals co-occur and are highly predictive of each other. During training, the model learns these correlations as predictive features, but they do not encode a clear direction of causality. This leads to a form of causal ambiguity, where the model may rely on vehicle motion as a proxy for pedestrian behavior, even though it is not the underlying cause.

This reflects the model’s ability to capture contextual priors present in real-world driving data, where pedestrian actions are influenced by traffic flow. However, this can blur the distinction between intent (what the pedestrian plans to do) and action (what is immediately likely given the current scene constraints), and produce incorrect predictions.
  
---

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

---

# Key Skills Demonstrated

**Machine Learning Engineering**
- Temporal deep learning models  
- Multi-task behavior prediction  
- Autonomous driving perception and prediction tasks  

**Software Engineering**
- Modular PyTorch training pipelines  
- Dataset ingestion and preprocessing systems  
- GPU accelerated training workflows  
- Experiment reproducibility and evaluation tooling  

**Research Engineering**

- Implementing state-of-the-art research models  
- Designing reproducible ML experiments  
- Analyzing failure cases and model behavior  

---

# Repository Structure
```bash
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
├── video_inference.py            # Real-time inference demo script
├── .gitignore
├── README.md
└── requirements.txt
```
---

# Future Work

- Optimizing inference performance
- Fine-tuning the V-JEPA2 backbone
- Longer context 
- Multi-modal fusion with LiDAR and map data  
- Pedestrian trajectory prediction
- Real-time deployment for autonomous vehicles  

---
## Acknowledgment

This project builds upon and adapts Meta AI’s V-JEPA2 world model, leveraging their official research and implementation:

https://github.com/facebookresearch/vjepa2

This project builds upon the evaluation framework and benchmark introduced in [*Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment*](https://arxiv.org/abs/2407.00446), which provides a comprehensive analysis of pedestrian behavior understanding across multiple tasks, including intention estimation, action prediction, and event risk assessment.

---
# Author

Aditya Sanjaykumar Patel

MS Computer Science  
San Jose State University

Machine Learning Engineer focused on:

- autonomous driving AI  
- video understanding  
- world models  
- computer vision

LinkedIn: https://www.linkedin.com/in/adityapatel149

Portfolio: https://adityapatel149.github.io

Email: imadityapatel149@gmail.com
