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

If https://pypi.jetson-ai-lab.dev timeout, try https://pypi.jetson-ai-lab.io

```
PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu128
```

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

*This uses the **official** docker container. Auto restart added.*

#### Container

```
sudo docker run --runtime nvidia --network host --shm-size=8g \
  --restart unless-stopped \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/data/hf -e HUGGINGFACE_HUB_CACHE=/data/hf -e TRANSFORMERS_CACHE=/data/hf \
  -v $PWD:/workspace -v $PWD/data:/data -w /workspace \
  --name gr00t -d isaac-gr00t:orin bash -lc 'tail -f /dev/null'

sudo docker exec -it gr00t bash

```

See the python files for details.


### OpenVLA

*This use the jetson-containers for basic environments.*

#### Container

```
jetson-containers run -v /path/on/host:/path/in/container $(autotag openvla nano_llm)
```

#### Run

```
export PIP_INDEX_URL=https://pypi.org/simple
pip install -U "transformers==4.40.1" "tokenizers==0.19.1"

export HF_HOME=/data/hf
export HUGGINGFACE_HUB_CACHE=/data/hf
export TRANSFORMERS_CACHE=/data/hf
export HF_TOKEN=hf_xxx  # add your hugging face token here

```



###  LLAMA + MLC

This use the jetson-containers for basic environments.


#### Benchmark

> To quantize and benchmark a model, run the [`benchmark.sh`](https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/mlc/benchmark.sh) script from the host (outside container)

```
HUGGINGFACE_TOKEN=your_token_here jetson-containers/packages/llm/mlc/benchmark.sh meta-llama/Llama-2-7b-hf
```

#### ~~Container~~

```

```
