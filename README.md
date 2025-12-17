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
git clone https://github.com/yourprojectrepo/deepfake-iot-detection.git
cd deepfake-iot-detection
