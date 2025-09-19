# Jetson-VLA-benchmarks

🚀 Benchmarking **Vision-Language-Action (VLA) models** on NVIDIA Jetson and other edge AI devices.
Most existing works (e.g., NVIDIA Isaac GR00T, OpenVLA, GR00T, NanoLLM, MLC-LLM) only report performance on **desktop GPUs**.
This repo fills the gap by providing **reproducible benchmarks on edge hardware**, including performance, power, and accuracy.

---

## 📌 Goals

- Measure **inference speed**, **latency**, and **throughput** of VLA models on Jetson devices (AGX Orin, Orin NX, Nano, …).
- Record **energy consumption (tegrastats)** and memory footprint.
- Evaluate **accuracy/correctness** on public datasets (e.g., RLDS, Bridge, LeRobot, EuroSAT, custom robotics tasks).
- Provide **deployment recipes** (Docker + jetson-containers + TensorRT/MLC optimizations).
- Build an **open benchmark suite** for VLA models on edge.

---

## 🖥️ Target Models

- [OpenVLA](https://github.com/yanqiangmiffy/OpenVLA)
- [NanoLLM](https://github.com/dusty-nv/nano-llm)
- [MLC-LLM](https://mlc.ai/mlc-llm) (quantized deployment on Jetson)
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)

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
- Python ≥ 3.10
- Git LFS (for large models & datasets)

### Quickstart

Clone repo:

```bash
git clone https://github.com/wzqvip/Jetson-VLA-benchmarks.git
cd Jetson-VLA-benchmarks
```
