                            JAIL CAMERA SURVEILLANCE

 # 🛡️ Real-Time Violence Detection using YOLOv8 Pose & MediaPipe Hands

This project is focused on **real-time violence or fight detection** in surveillance videos using a hybrid approach combining **YOLOv8n-pose** for body landmark detection and **MediaPipe Hands** for hand tracking. The system is capable of identifying aggressive behavior using motion, hand proximity, and pose dynamics.

## 📌 Project Objective

The goal of this project is to detect physical altercations or abnormal human behavior in surveillance footage with high accuracy and minimal delay. This can be used in:

- Jail monitoring systems
- Public surveillance
- Automated alert systems

---

## 🧠 Key Features

- ⚡ **Real-time detection** using webcam or video feed
- 🧍‍♂️ **YOLOv8n-pose** for multi-person pose estimation
- 🤲 **MediaPipe Hands** for detecting hand movement and proximity
- 📐 **Custom logic** for detecting:
  - Fast wrist movement (velocity-based)
  - Close hand proximity (potential aggression)
  - Fist detection and arm extension
  - Bounding box overlap and torso distance
- 📸 Frame saving on detection
- 🔔 Alert triggering mechanism

---

## 🧰 Technologies Used

| Component           | Library/Tool           |
|-------------------- |------------------------|
| Pose Detection      | YOLOv8n-pose (Ultralytics) |
| Hand Detection      | MediaPipe Hands        |
| Computer Vision     | OpenCV                 |
| Image Processing    | NumPy, PIL             |
| Data Encoding       | base64                 |      
| Database            | MySQL                  |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/adeebaimam/Violence_Detection_Camera_Surveillance.git
cd Violence_Detection_Camera_Surveillance

### 2. Create a virtual environment and install dependencies

python -m venv venv
source venv/bin/activate  # Linux
# OR
venv\Scripts\activate  # Windows
pip install -r requirements.txt


### 3. Run the detection script

python main.py

