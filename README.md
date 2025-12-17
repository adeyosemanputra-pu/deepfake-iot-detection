# Deepfake IoT Detection — Lightweight CNN for Biometric Security on Edge Devices

**Repository:** [https://github.com/adeyosemanputra-pu/deepfake-iot-detection](https://github.com/adeyosemanputra-pu/deepfake-iot-detection)  


---

## 🔍 Overview

This repository implements a fully reproducible pipeline for detecting deepfake facial manipulations on embedded IoT platforms using a lightweight Convolutional Neural Network (CNN). The goal is to enable secure and efficient biometric authentication on edge devices like Raspberry Pi and NVIDIA Jetson Nano.

### ✅ Key Features
- Dataset frame extraction & augmentation (IoT sensor noise simulation)
- MobileNetV2-inspired lightweight CNN under 2 MB
- TensorFlow + TFLite training and deployment scripts
- Real-time on-device inference with webcam
- Performance evaluation: accuracy, latency, memory footprint

---

<h2>📁 Directory Structure</h2>

<pre><code>deepfake-iot-detection/
├── data/              # Input/output frame folders
├── scripts/           # Frame extraction, augmentation
├── models/            # CNN model, training and TFLite conversion
├── demo/              # On-device inference (Pi, Jetson)
├── results/           # Output metrics, predictions
├── notebooks/         # Optional notebooks for experiments
├── README.md
├── requirements.txt
└── requirements_pi.txt
</code></pre>



---

## 📦 Requirements

### 💻 For training (GPU workstation)
```bash
pip install -r requirements.txt

🧠 For Raspberry Pi / Jetson Nano
pip3 install -r requirements_pi.txt


On Jetson Nano, install JetPack 4.6+ and ensure OpenCV + TensorRT are enabled for TFLite acceleration.

🧪 Dataset Preparation

We use:

FaceForensics++

Celeb-DF v2

DeeperForensics-1.0

❗ Datasets must be downloaded from their official repositories.

🎥 Extract frames
python scripts/extract_frames.py --video path/to/video.mp4 --outdir data/frames/

🧪 Augment with IoT-like noise
python scripts/augment.py --input data/frames --output data/augmented --blur 2 --jpeg 70

🏗️ Model Training
⚙️ Configuration
# models/train_config.yaml
input_size: 128
batch_size: 32
epochs: 50
learning_rate: 0.001
architecture: mobilenet_light

🚀 Train the model
python models/train.py --config models/train_config.yaml


Output: model.h5 and evaluation logs.

🧠 TFLite Conversion & Quantization
python models/convert_to_tflite.py --input model.h5 --output model.tflite
python models/quantize_tflite.py --input model.h5 --output model_quant.tflite

📸 Live Demo on Raspberry Pi / Jetson Nano
🎛️ Setup
pip3 install -r requirements_pi.txt

🎥 Run webcam detection
python demo/pi_live_demo.py --model models/model_quant.tflite


The live overlay displays prediction labels and inference speed (ms/frame).

📊 Evaluation
📈 Metrics:

Accuracy, Precision, Recall, F1-score

Inference latency (ms), memory usage (MB)

🧪 Run test set
python demo/evaluate_on_device.py --model models/model_quant.tflite --testdir data/test


### Install in PC @ windows

1. System Requirements

OS: Windows 10 or 11 (64-bit)

Python: 3.8 or 3.9 (recommended)

RAM: 8 GB or more

GPU (optional): NVIDIA GPU with CUDA support (for faster training)

🔧 2. Install Python

If not already installed:

Download and install Python from: https://www.python.org/downloads/

During installation, check "Add Python to PATH".

🧱 3. Create Virtual Environment (Recommended)

Open Command Prompt (CMD) or PowerShell:

python -m venv venv
venv\Scripts\activate

📦 4. Install Dependencies

Make sure requirements.txt is in your current folder. Then run:

pip install --upgrade pip
pip install -r requirements.txt

🗂 5. Run the Project
Example: Frame Extraction from Video
python scripts/extract_frames.py --video_path data/sample.mp4 --output_folder data/frames/

Example: Model Training
python models/train_model.py

Example: On-device TFLite Inference (simulate on PC)
python demo/tflite_inference.py --model models/model.tflite --input data/frames/frame_001.jpg

💡 Notes

Use Jupyter Notebooks in the notebooks/ folder if you prefer interactive experiments.

Real-time webcam testing may require installing additional packages:

pip install opencv-python-headless
🔁 Reproducibility & Citation

All components are fully documented and executable under the same configuration. If you use this project in your work, please cite:

@article{adeyosemanputra2025deepfakeiot,
  author={Ade Yoseman Putra and Coauthors},
  year={2025},
  url={https://github.com/adeyosemanputra-pu/deepfake-iot-detection}
}

📬 Contact

For issues or suggestions, open an issue or contact the repository maintainer via GitHub.

📄 License
This project is released under the MIT License. Dataset redistribution is not allowed; refer to the original dataset licenses.


