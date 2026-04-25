# 🖥️ Hands-Free Computer Control System

A real-time AI-based dual-mode assistive system that enables computer control using **hand gestures and head movements**, designed to improve accessibility for users with limited motor abilities.

---

## 🚀 Overview
Traditional input devices like keyboards and mice require fine motor skills.  
This project introduces a **touchless interaction system** using computer vision and machine learning for real-time control.

---

## ✨ Features
- Dual-mode control:
  - Hand Gesture Control  
  - Head Movement Control  
- Real-time cursor movement  
- Gesture-based click, scroll, and actions  
- Virtual keyboard activation  
- Mode switching between hand and head  

---

## 🧠 Methodology
- Extracted hand (21 points) and face (468 points) landmarks using MediaPipe  
- Converted landmarks into numerical features  
- Trained **SVM models** for gesture classification  
- Performed real-time prediction using webcam input  
- Mapped predictions to cursor and system actions  

---

## 📊 Results
- Hand Gesture Accuracy: ~91.5%  
- Head Movement Accuracy: ~92.3%  
- Smooth real-time control achieved  

---

## 🏗️ Project Structure
Hands-Free-Computer-Control-System/
│
├── main_switch.py
├── requirements.txt
├── README.md
│
├── hand_module/
│ ├── control.py
│ ├── train_model.py
│ ├── gesture_dataset.csv
│
├── head_module/
│ ├── head_cursor.py
│ ├── train_svm.py
│ ├── final_gesture_dataset.csv
│
├── models/
│ ├── gesture_model.pkl
│ ├── gesture_svm_model.pkl
│ ├── scaler.pkl
│
└── screenshots/
├── gesture.png
├── keyboard.png
├── click.png



---

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-learn (SVM)
- NumPy
- PyAutoGUI
- Keyboard

## ▶️ How to Run
pip install -r requirements.txt  
python main_switch.py

## 🎮 Controls
| Input | Action |
|------|--------|
| Hand movement | Cursor control |
| Gesture | Click / Right-click |
| Open palm | Screenshot |
| Head movement | Cursor navigation |
| Mode selection | Switch control mode |

## 📸 Output
![Gesture](screenshots/move.png)  
![Keyboard](screenshots/keyboard.png)  
![Right Click](screenshots/right_click.png)
![Screen Shot](screenshots/screenshot.png)

## 🌍 Use Case
- Assistive technology for users with motor impairments
- Touchless computer interaction
- Smart human-computer interfaces

## 🔮 Future Improvements
- Voice command integration
- Deep learning-based gesture recognition
- Personalized user adaptation
- Mobile and IoT integration

## 📌 Conclusion
This project demonstrates how computer vision and machine learning can be combined to build an efficient, low-cost, and accessible human-computer interaction system.
