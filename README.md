# Deepfake IoT Detection

Lightweight CNN-based deepfake detector for IoT security.


# 🧠 Deepfake Facial Detection for IoT Biometric Security

This repository contains the full pipeline for a lightweight Convolutional Neural Network (CNN) designed to detect deepfake facial content, optimized for deployment in resource-constrained IoT environments such as Raspberry Pi and Jetson Nano. The model is trained using public datasets and converted to TensorFlow Lite for real-time edge inference.

## 🚀 Project Objectives
- Develop an efficient CNN for deepfake detection under 2MB.
- Simulate real-world IoT camera conditions via video degradations.
- Deploy the model to edge-class devices and benchmark performance.
- Provide a reproducible demo pipeline from training to deployment.

---

## 📁 Directory Structure

deepfake-iot-detection/
├── data/ # Input/output frame folders
├── scripts/ # Frame extraction, augmentation
├── models/ # CNN model, training and TFLite conversion
├── demo/ # On-device inference (Pi, Jetson)
├── results/ # Output metrics, predictions
├── notebooks/ # Optional notebooks for experiments
├── README.md
├── requirements.txt
└── requirements_pi.txt


---

## 🧪 Datasets Used

- **FaceForensics++** (Rössler et al., 2019)
- **Celeb-DF v2** (Li et al., 2020)
- **DeeperForensics-1.0** (Jiang et al., 2020)

All datasets were preprocessed to simulate IoT conditions such as compression, blur, and motion artifacts.

---

## 🛠️ Installation (Windows)

1. Clone the repository:
   ```bash
   git clone https://github.com/adeyosemanputra-pu/deepfake-iot-detection.git
   cd deepfake-iot-detection


Create a virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

🧠 Model Training

Train the lightweight CNN using the prepared dataset:

python models/train_model.py


Convert the trained model to TensorFlow Lite:

python models/convert_to_tflite.py

📷 Real-Time Inference on Edge Devices

Deploy to Raspberry Pi or Jetson Nano:

python demo/tflite_inference.py --model models/model.tflite --input data/frames/frame_001.jpg


Supports webcam capture and frame-by-frame inference.

📊 Evaluation Metrics

Accuracy, Precision, Recall, F1-score

Inference latency (ms/frame)

Memory usage (MB)

🔁 Reproducibility and Source Code Access

All code, scripts, and documentation are open-source under MIT License. The entire pipeline is reproducible on a clean install of Raspberry Pi OS or JetPack.

📂 Repository: github.com/adeyosemanputra-pu/deepfake-iot-detection

📬 Contact

For feedback, collaboration, or questions, reach out via GitHub Issues or open a pull request.
