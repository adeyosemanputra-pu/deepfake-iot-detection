# Deepfake IoT Detection — Lightweight CNN for Edge Biometric Security

**Repository:** `https://github.com/yourprojectrepo/deepfake-iot-detection`  
**Paper:** *Deepfake Facial Detection Using a Lightweight CNN for Enhancing IoT Biometric Security* (submit to IEEE/Scopus)

This repository provides a complete, reproducible pipeline for training, converting, deploying and evaluating a lightweight CNN deepfake detector on edge IoT hardware (Raspberry Pi 4B, Jetson Nano). The repo includes code, small example datasets, inference demo scripts and instructions to reproduce the results reported in the paper.

---

## Table of contents
1. [Quickstart — Demo on Raspberry Pi (5 min)](#quickstart)
2. [Repository structure](#structure)
3. [Environment & Requirements](#requirements)
4. [Datasets and Licensing](#datasets)
5. [Preprocessing & Augmentation](#preprocessing)
6. [Training (full instructions)](#training)
7. [Export to TFLite & Quantization](#tflite)
8. [Deploy & Run Inference on IoT devices](#deploy)
9. [Evaluation & Logging](#evaluation)
10. [Reproducibility notes and expected outputs](#repro)
11. [Citation & License](#citation)

---

## Quickstart — Demo on Raspberry Pi (recommended)
> Minimal: run a prebuilt TFLite model on Pi and see live overlay predictions.

1. Clone repository:
```bash
git clone https://github.com/adeyosemanputra-pu/deepfake-iot-detection.git
cd deepfake-iot-detection
## Quickstart — Demo on Raspberry Pi (recommended)
> Minimal: run a prebuilt TFLite model on Pi and see live overlay predictions.

1. Clone repository:
```bash
git clone https://github.com/adeyosemanputra-pu/deepfake-iot-detection.git
cd deepfake-iot-detection
Transfer models/model_quant.tflite to the Pi (or build per steps below).

On Raspberry Pi (Raspbian) install runtime:

bash
Salin kode
sudo apt update
sudo apt install -y python3-pip libatlas-base-dev
pip3 install -r requirements_pi.txt
Plug in a webcam and run:

bash
Salin kode
python3 demo/pi_live_demo.py --model models/model_quant.tflite --camera 0
You should see a window with real-time overlay labels REAL / FAKE and latency printed.

Repository structure (recommended)
graphql
Salin kode
deepfake-iot-detection/
├─ README.md
├─ LICENSE
├─ CITATION.bib
├─ requirements.txt          # training env
├─ requirements_pi.txt       # tiny runtime for Raspberry Pi
├─ data/                     # pointers and small examples (not full datasets)
├─ scripts/
│  ├─ download_datasets.sh
│  ├─ extract_frames.py
│  ├─ augment.py
│  ├─ sample_frames.py
├─ models/
│  ├─ light_cnn.py           # architecture
│  ├─ train.py
│  ├─ train_config.yaml
│  ├─ model.h5
│  ├─ model.tflite
│  └─ model_quant.tflite
├─ notebooks/
│  ├─ training_notebook.ipynb
│  └─ eval_notebook.ipynb
├─ demo/
│  ├─ pi_live_demo.py
│  ├─ jetson_demo.py
│  └─ evaluate_on_device.py
└─ results/
   ├─ metrics.csv
   └─ sample_output_images/
Environment & Requirements <a name="requirements"></a>
Use a machine with GPU for training (NVIDIA CUDA) and the Pi/Jetson for edge testing.

For training (Ubuntu / Colab / server):

text
Salin kode
Python 3.9+
TensorFlow 2.10+
numpy, pandas, opencv-python, pillow
scikit-learn, tqdm, yaml
Install via:

bash
Salin kode
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
For Raspberry Pi runtime:

text
Salin kode
python3, tflite-runtime (or tensorflow-lite), opencv-python
Install via:

bash
Salin kode
pip3 install -r requirements_pi.txt
Example requirements.txt and requirements_pi.txt are included in the repo.

Datasets & Licensing <a name="datasets"></a>
We use three public deepfake benchmarks:

FaceForensics++ (Rössler et al., 2019) — face swapping, reenactment. Access: official site / GitHub.

Celeb-DF v2 (Li et al., 2020).

DeeperForensics-1.0 (Jiang et al., 2020).

Important: Datasets are not redistributed here. Use scripts/download_datasets.sh (requires proper credentials/consent for each dataset if applicable). The script shows official dataset URLs and commands; users must follow dataset licenses.

Preprocessing & Data Augmentation <a name="preprocessing"></a>
1) Extract frames from videos
Command:

bash
Salin kode
python3 scripts/extract_frames.py --video /path/to/video.mp4 --outdir data/frames/video001 --fps 2
extract_frames.py uses OpenCV and saves numbered JPEG frames.

2) Augment frames (simulate IoT noise)
Example operations: JPEG compression, Gaussian blur, downsampling, motion blur.

Command:

bash
Salin kode
python3 scripts/augment.py --input data/frames --output data/augmented --jpeg-quality 70 --blur-radius 2 --downsample 0.5
3) Class balancing & sampling
We sample N frames per video to avoid redundancy and balance classes:

bash
Salin kode
python3 scripts/sample_frames.py --input data/augmented --out data/sampled --frames-per-video 10
Model design & Training <a name="training"></a>
The lightweight CNN is implemented in models/light_cnn.py—a compact architecture using depthwise separable convs and residual bottlenecks (inspired by MobileNetV2/EfficientNet-lite). The model aims for <2M parameters.

Train (example)
bash
Salin kode
python3 models/train.py --config models/train_config.yaml
train_config.yaml includes:

yaml
Salin kode
input_shape: [128,128,3]
batch_size: 32
epochs: 50
learning_rate: 0.001
train_split: 0.8
val_split: 0.1
test_split: 0.1
optimizer: adam
loss: binary_crossentropy
Checkpointing & logs: TensorBoard logs and best model .h5 are saved in results/.

Convert to TFLite & Quantization <a name="tflite"></a>
Export best .h5 to TFLite (float model):

bash
Salin kode
python3 models/convert_to_tflite.py --input results/best_model.h5 --output models/model.tflite
Post-training integer quantization (recommended for Pi / Jetson Nano):

bash
Salin kode
python3 models/quantize_tflite.py --input results/best_model.h5 --output models/model_quant.tflite --representative data/rep_samples
Representative dataset: sample 1000 images in data/rep_samples for accurate quantization.

Deploy & Run Inference on IoT devices <a name="deploy"></a>
Raspberry Pi (example)
Copy models/model_quant.tflite to Pi.

Install runtime:

bash
Salin kode
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install -r requirements_pi.txt
Run demo:

bash
Salin kode
python3 demo/pi_live_demo.py --model /home/pi/model_quant.tflite --camera 0 --size 128
demo/pi_live_demo.py:

Captures frames, resizes to model input, runs inference with tflite_runtime.

Displays overlay label and per-frame latency.

Logs metrics to results/pi_inference_log.csv.

Jetson Nano
Use the included demo/jetson_demo.py that uses TensorRT (if available) or tflite_runtime.

Evaluation & Logging <a name="evaluation"></a>
We provide scripts to evaluate model classification metrics and on-device latency:

demo/evaluate_offline.py — compute Accuracy, Precision, Recall, F1 on held-out test set.

demo/evaluate_on_device.py — runs inference on device and logs per-frame latency, CPU usage, memory.

Example offline evaluation:

bash
Salin kode
python3 demo/evaluate_offline.py --model models/model.tflite --testdir data/test --out results/metrics.csv
We use the following evaluation protocol:

Test set size: 1,000 real + 1,000 fake frames (balanced).

Report: accuracy, precision, recall, F1-score, model size (MB), number of parameters, and median/mean inference latency (ms/frame).

Reproducibility notes and expected outputs <a name="repro"></a>
All random seeds are fixed in train.py (numpy, tf, random) for reproducible training.

Hardware used for published results: NVIDIA GTX 1080Ti for training; Raspberry Pi 4B (4GB), Jetson Nano for edge experiments.

Expected model size after quantization: ~0.8–2 MB (depends on architecture).

Typical on-device latency (quantized model): Raspberry Pi 4B ≈ 30–120 ms/frame (depending on input size), Jetson Nano ≈ 10–50 ms/frame.

Example result images with overlay (one sample provided in results/sample_output_images/).

CITATION & Paper reference <a name="citation"></a>
If you use this code or pre-trained models in your research, please cite the paper and this repository:

Bibtex (example)

bibtex
Salin kode
@article{yourname2025deepfakeiot,
  title={Deepfake Facial Detection Using a Lightweight CNN for Enhancing IoT Biometric Security},
  author={Your Name and Coauthors},
  journal={IEEE Transactions on X},
  year={2025},
  note={Code: https://github.com/adeyosemanputra-pu/deepfake-iot-detection}
}
Add the following in CITATION.bib and README.md as provided.
