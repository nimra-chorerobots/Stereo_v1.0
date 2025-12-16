# Stereo Vision–Based Perception Stack for Autonomous Robotics

This repository implements a **complete perception stack** for autonomous robotic systems using **stereo cameras**, classical computer vision, and lightweight deep learning for semantic understanding.

The pipeline demonstrates how raw stereo imagery can be transformed into:
- **Metric depth**
- **Semantic object understanding**
- **Free-space estimation**
- **Planning-oriented obstacle representations**
- **3D world reconstruction**

The implementation is intentionally modular, interpretable, and suitable for real-time robotic deployment.

---

## 🔍 Perception Stack Overview

<img width="384" height="460" alt="image" src="https://github.com/user-attachments/assets/5081287e-9414-4768-ab4a-15cfecd33c48" />

---

## 📂 Dataset

**KITTI Scene Flow / Stereo 2015 Dataset**

- Rectified stereo image pairs
- Urban driving scenarios
- Suitable for depth, disparity, and object perception

<img width="310" height="140" alt="image" src="https://github.com/user-attachments/assets/932e44a9-f4d7-47a6-81a8-5e8ba252b4cf" />


> **Note:** The training split is used for visualization and perception development.  
> The test split is reserved for benchmarking and evaluation, as per KITTI protocol.

---

## ⚙️ Core Components

### 1️⃣ Stereo Depth Estimation

- Algorithm: **StereoSGBM**
- Outputs:
  - Dense disparity map
  - Metric depth (meters)

Depth conversion:
\[
\text{Depth} = \frac{f \cdot B}{\text{disparity}}
\]

Where:
- `f` = focal length  
- `B` = stereo baseline  

This stage provides **geometry without learning**, ideal for reliability and explainability.

---

### 2️⃣ Semantic Perception (YOLO-Tiny)

- Model: **YOLOv3-Tiny**
- Framework: OpenCV DNN
- Detects:
  - Cars
  - Pedestrians
  - Urban objects (COCO classes)

Each detection is **fused with depth** to compute:
- Object class
- Bounding box
- **Estimated distance in meters**

Example output:
Detected Objects:
car 6.1m
car 11.0m
car 33.3m


---

### 3️⃣ Planning-Oriented Representation

From metric depth:
- **Near-obstacle regions** (< 10 m) are highlighted
- **Free space** (> 10 m) is extracted
- Outputs are suitable for:
  - Local planners
  - Cost maps
  - Collision avoidance logic

This bridges perception → planning.

---

### 4️⃣ 3D World Model (Point Cloud)

- Dense depth is converted into a **colored 3D point cloud**
- Exported as `.ply`
- Can be visualized in:
  - MeshLab
  - Open3D
  - RViz (ROS2)

This represents the robot’s **geometric understanding of the environment**.

---

## 🖼️ Results
<img width="1539" height="133" alt="Figure_3" src="https://github.com/user-attachments/assets/b65bfbd7-26a0-48d5-8c9d-b52b5e0b5828" />

### Multi-Stage Perception Outputs

The pipeline produces the following visual layers:

1. **Stereo Input**  
   Raw left camera image

2. **Disparity Map**  
   Pixel-level stereo correspondence

3. **Metric Depth Map**  
   Color-coded depth in meters

4. **Semantic Layer**  
   YOLO-Tiny detections fused with depth

5. **Planning View**  
   Near obstacles highlighted for navigation

6. **Free Space Map**  
   Binary traversable regions

These outputs clearly demonstrate the transition from **raw sensing → actionable spatial understanding**.

---

## 🚀 How to Run

### Requirements
```bash
pip install opencv-python numpy matplotlib
