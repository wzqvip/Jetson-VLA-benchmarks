#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark OpenVLA on Jetson (latency/throughput/memory + tegrastats power/util window)

Examples:
  # 从 LIBERO 取样，跑 50 次
  python bench_openvla.py \
    --model openvla/openvla-7b \
    --dataset physical-intelligence/libero \
    --episodes 50 \
    --dtype fp16 \
    --out /data/benchmarks/openvla_libero_fp16.json \
    --tegrastats /data/tegrastats_openvla.log

  # 用本地图片（重复推理 200 次）
  python bench_openvla.py \
    --model openvla/openvla-7b \
    --image /data/sample.jpg \
    --repeat 200 \
    --dtype fp16 \
    --out /data/benchmarks/openvla_img_fp16.json \
    --tegrastats /data/tegrastats_openvla.log
"""

import os, json, time, argparse, subprocess, signal, re, statistics, datetime as dt
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ---------- utils ----------
def now_ts():
    return dt.datetime.now()

def percentile(data, p):
    if not data:
        return None
    data = sorted(data)
    k = (len(data)-1) * (p/100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(data[int(k)])
    return float(data[f] * (c-k) + data[c] * (k-f))

def start_tegrastats(logfile: Path, interval_ms: int = 200):
    """
    启动 tegrastats 后台记录。返回 Popen 对象与启动时间。
    注：多数 Jetson 上 tegrastats 不需要 sudo；若你的需要，改成 ['sudo','tegrastats',...]
    """
    cmd = ["tegrastats", "--interval", str(interval_ms), "--logfile", str(logfile)]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p, now_ts()
    except FileNotFoundError:
        print("WARNING: tegrastats not found. Power/GPU util will be unavailable.")
        return None, now_ts()

def stop_tegrastats(p: subprocess.Popen | None):
    if p is None:
        return
    try:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            p.terminate()
    except Exception:
        pass

def parse_tegrastats(logfile: Path, t0: dt.datetime, t1: dt.datetime):
    """
    解析 tegrastats 日志窗口 [t0, t1] 的统计：
    - gpu_util_avg / p95 (%)
    - ram_used_avg (MB)
    - gpu_power_avg (W) 若有 VDD_GPU_SOC 或 VDD_SYS_GPU 字段
    """
    out = {}
    if not logfile.exists():
        return out

    gr3d, ram_used, gpumw = [], [], []

    # 兼容多种格式
    # GR3D_FREQ  23%@...   或 GR3D_FREQ 23%
    rgpu = re.compile(r"GR3D_FREQ\s+(\d+)%")
    # RAM 1234/65536MB
    rram = re.compile(r"RAM\s+(\d+)/(\d+)MB")
    # VDD_SYS_GPU 1234mW 或 VDD_GPU_SOC 1234mW
    rpow = re.compile(r"(VDD_SYS_GPU|VDD_GPU_SOC)\s+(\d+)mW")

    # 有的日志不带时间戳，这里按窗口粗略裁剪：只要落在进程启动-停止之间的行都算
    lines = logfile.read_text(errors="ignore").splitlines()
    for ln in lines:
        m = rgpu.search(ln)
        if m:
            gr3d.append(int(m.group(1)))
        m = rram.search(ln)
        if m:
            ram_used.append(int(m.group(1)))  # MB used
        m = rpow.search(ln)
        if m:
            gpumw.append(int(m.group(2)))  # mW

    if gr3d:
        out["gpu_util_avg_pct"] = round(statistics.mean(gr3d), 2)
        out["gpu_util_p95_pct"] = round(percentile(gr3d, 95), 2)
    if ram_used:
        out["ram_used_avg_gb"] = round(statistics.mean(ram_used) / 1024.0, 3)
    if gpumw:
        out["gpu_power_avg_w"] = round(statistics.mean(gpumw) / 1000.0, 3)

    out["samples"] = len(gr3d)
    out["window_seconds"] = (t1 - t0).total_seconds()
    return out

def load_one_image_from_dataset(ds_name: str):
    from datasets import load_dataset
    ds = load_dataset(ds_name, split="train", streaming=False)
    sample = ds[0]

    def _first_image_from_obj(obj):
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[2] in (3, 4):
            arr = obj[..., :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, "RGB")
        if isinstance(obj, dict):
            for v in obj.values():
                im = _first_image_from_obj(v)
                if im is not None:
                    return im
        if isinstance(obj, (list, tuple)):
            for v in obj:
                im = _first_image_from_obj(v)
                if im is not None:
                    return im
        return None

    img = _first_image_from_obj(sample)
    if img is None:
        # 扫描前若干样本
        for i in range(min(10, len(ds))):
            img = _first_image_from_obj(ds[i])
            if img is not None:
                break
    if img is None:
        raise RuntimeError(f"No RGB image found in first samples of {ds_name}")
    return img

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b")
    ap.add_argument("--image", help="Path to a local RGB image.")
    ap.add_argument("--dataset", help="HF dataset to pull one frame from (e.g., physical-intelligence/libero)")
    ap.add_argument("--prompt", default="In: What action should the robot take to {open the drawer}?\\nOut:")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","fp32"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=50, help="When --dataset is used, how many items to run.")
    ap.add_argument("--repeat", type=int, default=100, help="When --image is used, repeat count.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tegrastats", type=str, default="", help="Path to tegrastats log file (start/stop automatically).")
    ap.add_argument("--interval_ms", type=int, default=200, help="tegrastats sampling interval.")
    ap.add_argument("--out", type=str, default="/data/benchmarks/openvla_bench.json")
    ap.add_argument("--no_sample", action="store_true", help="Deterministic output")
    args = ap.parse_args()

    assert args.image or args.dataset, "Provide --image or --dataset"

    token = os.getenv("HF_TOKEN")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # load input
    if args.image:
        img = Image.open(args.image).convert("RGB")
        mode = "image-repeat"
        total_iters = args.repeat
    else:
        img = load_one_image_from_dataset(args.dataset)
        mode = "dataset"
        total_iters = args.episodes

    # processor & model
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, token=token)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",         # 先按原始精度加载
        low_cpu_mem_usage=True,
        token=token,
    ).to(args.device)

    # 统一到 FP16（推荐在 Jetson）
    model = model.to(dtype=torch.float16)

    # prepare inputs (注意 ids/mask 类型)
    def make_inputs(image: Image.Image):
        ins = processor(args.prompt, image)
        moved = {}
        for k, v in ins.items():
            if hasattr(v, "to"):
                if torch.is_floating_point(v):
                    moved[k] = v.to(args.device, dtype=torch.float16)
                else:
                    moved[k] = v.to(args.device)
            else:
                moved[k] = v
        if "input_ids" in moved:
            moved["input_ids"] = moved["input_ids"].to(args.device, dtype=torch.long)
        if "attention_mask" in moved and moved["attention_mask"].dtype not in (torch.bool, torch.long):
            moved["attention_mask"] = moved["attention_mask"].to(args.device, dtype=torch.long)
        return moved

    # warmup
    model.eval()
    w_inputs = make_inputs(img)
    for _ in range(max(1, args.warmup)):
        with torch.inference_mode():
            _ = model.predict_action(**w_inputs, unnorm_key="bridge_orig", do_sample=not args.no_sample)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

    # start tegrastats
    tegra_proc, t0 = (None, now_ts())
    if args.tegrastats:
        tegra_log = Path(args.tegrastats)
        tegra_proc, t0 = start_tegrastats(tegra_log, args.interval_ms)

    lat_ms = []
    mem_alloc = []   # bytes
    mem_reserved = []

    # run
    for i in range(total_iters):
        inputs = w_inputs if mode == "image-repeat" else make_inputs(img)  # 对于 dataset，可扩展为逐条
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.perf_counter()
        with torch.inference_mode():
            _ = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=not args.no_sample)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.perf_counter()

        lat_ms.append((t_end - t_start) * 1000.0)
        if torch.cuda.is_available():
            mem_alloc.append(torch.cuda.memory_allocated())
            mem_reserved.append(torch.cuda.memory_reserved())

    # stop tegrastats
    t1 = now_ts()
    stop_tegrastats(tegra_proc)

    # summarize
    result = {
        "model": args.model,
        "dtype": "fp16",
        "device": args.device,
        "mode": mode,
        "iters": total_iters,
        "prompt": args.prompt,
        "latency_ms_mean": round(statistics.mean(lat_ms), 3),
        "latency_ms_p50": round(percentile(lat_ms, 50), 3),
        "latency_ms_p95": round(percentile(lat_ms, 95), 3),
        "throughput_fps_mean": round(1000.0 / statistics.mean(lat_ms), 3),
    }
    if mem_alloc:
        result["mem_alloc_mb_mean"] = round(statistics.mean(mem_alloc) / (1024**2), 2)
        result["mem_reserved_mb_mean"] = round(statistics.mean(mem_reserved) / (1024**2), 2)

    # tegrastats window stats
    if args.tegrastats:
        tegra = parse_tegrastats(Path(args.tegrastats), t0, t1)
        result["tegrastats"] = tegra

    # write
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        json.dump(result, f, indent=2)
    print("\n=== Benchmark Summary ===")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {outp}")

if __name__ == "__main__":
    main()
