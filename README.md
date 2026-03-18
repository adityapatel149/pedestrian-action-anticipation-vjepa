# Pedestrian Action Prediction with V-JEPA2
### Self-Supervised World Models for Autonomous Driving

🚗 **Predicting pedestrian crossing behavior *before it happens***

This project investigates how **Meta AI’s V-JEPA2 world model** can be adapted for **pedestrian behavior prediction in autonomous driving systems**.

Instead of relying on complex pipelines *(pose → trajectory → rules)*, this work leverages **self-supervised video representations trained on massive datasets** to anticipate human actions.

The system predicts whether a pedestrian will **cross within 1–3 seconds**, a key capability for **safe planning and collision avoidance**.

---

## 🚀 Key Contributions

- Adapted **V-JEPA2 world model** for pedestrian behavior prediction  
- Built a **bounding-box aware video prediction system**  
- Designed **temporal evaluation metrics** *(TTE, confidence stability)*  
- Developed an **end-to-end ML pipeline** for JAAD & PIE  
- Demonstrated **early prediction up to 3 seconds ahead**

---

## 🧠 What This Project Demonstrates

This system mirrors a **real-world autonomy stack component**:

- Self-supervised learning for **robotics perception**
- **Behavior prediction systems**
- Large-scale **video ML pipelines**
- **Temporal modeling** for action anticipation
- Evaluation frameworks for **autonomous driving**

---

## 🎥 Demo

<p align="center">
<img src="assets/demo.gif" width="750">
</p>

Early prediction of pedestrian crossing behavior.

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt
python evals/main.py --config configs/eval/vitl/jaad.yaml
```

---

## 🚧 Problem Motivation

Autonomous vehicles must understand **human intent before actions occur**.

Key cues:

- head orientation  
- walking speed  
- crosswalk proximity  
- traffic signals  

Traditional pipelines are:

- complex  
- annotation-heavy  
- hard to generalize  

This work explores **world models** as a simpler, more scalable alternative.

---

## 🏗️ System Overview

<p align="center">
<img src="assets/model_architecture.png" width="800">
</p>

**Pipeline**

1. Extract pedestrian video clips  
2. Encode with **frozen V-JEPA2 encoder**  
3. Predict **future latent tokens**  
4. Combine context + future  
5. Add **bounding box features**  
6. Train **attention-based probe**

---

## 🧩 Model Architecture

### Backbone: V-JEPA2

- Vision Transformer (**ViT-L/16**)  
- Self-supervised predictive learning  
- Encodes **motion, structure, interactions**

The backbone is **frozen** → acts as a **general world model**.

---

### Multi-Task Prediction Head

The probe predicts:

- **Crossing (primary)**
- Looking  
- Walking  
- Crossing type  
- Intersection  
- Signalized context  

---

### Why This Approach Works

V-JEPA2 learns from **massive unlabeled video**, capturing:

- motion dynamics  
- human behavior  
- scene structure  

This enables:

✔ Better **generalization**  
✔ Robustness to **occlusion & lighting**  
✔ **Early anticipation** of actions  

---

## 📊 Data & Pipeline

<p align="center">
<img src="assets/sampling_and_annotation.png" width="750">
</p>

**Sampling**

- 8 frames / clip  
- 15 FPS  
- ~0.5s window  
- 30% overlap  

**Target:**  
→ Will the pedestrian cross in **1 second?**

---

## ⚙️ Training Configuration

- Backbone: **V-JEPA2 ViT-L**  
- Frames: 8 @ 15 FPS  
- Resolution: 256×256  
- Batch size: 32  
- Optimizer: AdamW  
- Loss: **Softmax Focal Loss**

---

## 📈 Evaluation Framework

<p align="center">
<img src="assets/metrics_big.png" width="750">
</p>

### Sample Metrics
Accuracy • bAcc • Precision • F1 • AUROC • mAP  

### Instance Metrics
- **Soft:** average probability  
- **Hard:** strict agreement  

### Confidence Delta
Δ_conf = mean(|p_t+1 − p_t|)

Measures **temporal stability**

---

## 📊 Results

### Performance
- **Accuracy:** 80%  
- **Recall:** 70%

### Key Insights

✔ Predicts crossing **up to 3 seconds early**  
✔ Handles **subtle behavioral cues**  
✔ Multi-task learning improves representation  

### Why It Matters

Critical for:

- collision avoidance  
- braking decisions  
- real-time planning  

---

## 🔍 Visualization

### Attention Heatmaps
<p align="center">
<img src="assets/attention_heatmaps.png" width="750">
</p>

Focus areas:
- pedestrian motion  
- crosswalks  
- signals  

---

### Anticipation vs Time
<p align="center">
<img src="assets/tte_curve.png" width="750">
</p>

Shows prediction quality across **time-to-event**

---

## ⚠️ Failure Cases

- Occlusion  
- Night conditions  
- Group interactions  
- Ambiguous behavior  

---

## 🧠 Scenario Example

Pedestrian approaching road:

- t = -3s → 0.12  
- t = -2s → 0.35  
- t = -1s → 0.71  

✔ Model anticipates crossing **before movement begins**

---

## 🛠️ Tech Stack

**ML:** PyTorch, ViT, Self-supervised learning  
**CV:** OpenCV, bounding boxes  
**Systems:** CUDA, NumPy, Matplotlib  

---

## 💡 Skills Demonstrated

### Machine Learning
- Video representation learning  
- Temporal modeling  
- Multi-task learning  
- Autonomous driving prediction  

### Software Engineering
- Modular pipelines  
- Data processing systems  
- GPU training  

### Research
- Paper implementation  
- Experiment design  
- Failure analysis  

---

## 📁 Repository Structure

pedestrian-action-anticipation-vjepa/
├── assets/
├── configs/
│   ├── eval/vitl/
│   └── inference/vitl/
├── evals/
│   ├── action_anticipation_frozen/
│   └── hub/
├── src/
│   ├── datasets/
│   ├── hub/
│   ├── masks/
│   ├── models/
│   └── utils/
├── your_data/
├── README.md
└── requirements.txt

---

## 🔮 Future Work

- Fine-tune backbone  
- Multi-modal (LiDAR, maps)  
- Trajectory prediction  
- Real-time deployment  

---

## 🙏 Acknowledgment

Built upon Meta AI’s V-JEPA2:

https://github.com/facebookresearch/vjepa2

---

## 👤 Author

Aditya Sanjaykumar Patel  
MS Computer Science, SJSU  

Machine Learning Engineer focused on:

- Autonomous driving  
- Video understanding  
- World models  
- Computer vision  

LinkedIn: https://www.linkedin.com/in/adityapatel149  
Email: imadityapatel149@gmail.com
