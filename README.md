# Jetson-VLA-benchmarks

🚀 Benchmarking **Vision-Language-Action (VLA) models** on NVIDIA Jetson and other edge AI devices.
Most existing works (e.g., NVIDIA Isaac GR00T, OpenVLA, GR00T, NanoLLM, MLC-LLM) only report performance on **desktop GPUs**.
This repo fills the gap by providing **reproducible benchmarks on edge hardware**, including performance, power, and accuracy.

---

## 📌 Goals

The goal is to provide benchmarks not only for performance (latency, throughput) but also **energy efficiency and accuracy** — areas often missing from existing reports which focus only on desktop GPUs.

At the current stage, we focus on **Jetson AGX Orin** as the primary platform.

---

## 🖥️ Target Models

- [OpenVLA](https://github.com/yanqiangmiffy/OpenVLA)
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [~~NanoLLM~~](https://github.com/dusty-nv/nano-llm)
- ~~[MLC-LLM](https://mlc.ai/mlc-llm) (quantized deployment on Jetson)~~

---

## 📊 Metrics Collected

- **Latency / FPS** – single-batch inference time.
- **Memory usage** – GPU/CPU RAM during inference.
- **Power consumption** – GPU power via `tegrastats`.
- **Accuracy** – dataset-based evaluation (task success rate, reward, top-1 acc, etc).

---

## 🔧 Setup

### Requirements

- NVIDIA Jetson device (AGX Orin recommended, JetPack ≥ 6.2)
- Docker with [jetson-containers](https://github.com/dusty-nv/jetson-containers)

### Quickstart

Clone repo:

```bash
git clone https://github.com/wzqvip/Jetson-VLA-benchmarks.git
cd Jetson-VLA-benchmarks
```


## Models

### ![ ](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white) GR00T

https://github.com/NVIDIA/Isaac-GR00T

NVIDIA-Isaac-GR00T-N1.5-3B
