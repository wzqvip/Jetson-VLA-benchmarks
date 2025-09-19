#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T 推理基准（容器内执行）

stdbuf -oL sudo tegrastats --interval 500 | ts '%s' >> ~/Isaac-GR00T/data/tegrastats_ts.log &
echo $! > ~/Isaac-GR00T/data/tegrastats.pid


# 结束采样（测完再执行）
sudo kill "$(cat ~/Isaac-GR00T/data/tegrastats.pid)" && rm ~/Isaac-GR00T/data/tegrastats.pid

运行示例（容器内）：
  python getting_started/gr00t_benchmark.py \
    --model-path nvidia/GR00T-N1.5-3B \
    --embodiment gr1 \
    --data-config fourier_gr1_arms_only \
    --dataset-path /workspace/Isaac-GR00T/demo_data/robot_sim.PickNPlace \
    --warmup 10 --iters 100 \
    --markers /workspace/Isaac-GR00T/data/markers.json \
    --tegrastats /workspace/Isaac-GR00T/data/tegrastats_ts.log \
    --out-json /workspace/Isaac-GR00T/data/bench_gr00t.json

如果不想解析 tegrastats，去掉 --tegrastats 即可。
"""
import argparse, os, time, json, statistics, re, csv, pathlib, sys
from typing import Dict, Any, List, Optional

import numpy as np
import torch

# ---- GR00T imports（要求你已在容器内执行过：pip install -e .）
import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def setup_torch():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_policy_and_data(
    model_path: str,
    embodiment_tag: str,
    data_config_key: str,
    dataset_path: str,
    device: str,
):
    if data_config_key not in DATA_CONFIG_MAP:
        raise ValueError(f"Unknown data-config '{data_config_key}'. "
                         f"Available: {list(DATA_CONFIG_MAP.keys())[:10]} ...")
    data_cfg = DATA_CONFIG_MAP[data_config_key]
    modality_config = data_cfg.modality_config()
    modality_transform = data_cfg.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,              # 由 policy 侧处理
        embodiment_tag=embodiment_tag,
    )
    return policy, dataset


@torch.inference_mode()
def run_benchmark(policy, dataset, warmup: int, iters: int) -> Dict[str, Any]:
    # 简单的“索引循环”，避免超范围
    idx_seq = [i % len(dataset) for i in range(iters)]

    # 预热
    for k in range(warmup):
        _ = policy.get_action(dataset[idx_seq[0]])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式计时
    lat: List[float] = []
    start = time.time()
    for i in range(iters):
        t0 = time.perf_counter()
        _ = policy.get_action(dataset[idx_seq[i]])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    end = time.time()

    # 统计
    avg = statistics.mean(lat)
    p50 = statistics.median(lat)
    p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)
    thr = len(lat) / sum(lat)

    return {
        "count": len(lat),
        "avg_s": avg,
        "p50_s": p50,
        "p95_s": p95,
        "throughput_rps": thr,
        "window_start": start,
        "window_end": end,
        "latencies_s": lat,  # 如需精简可不保存
    }


def parse_tegrastats_window(
    tegra_log: pathlib.Path,
    t0: float,
    t1: float,
) -> Dict[str, Any]:
    """
    解析带时间戳(ts '%s' 前缀)的 tegrastats 行，仅统计 [t0, t1] 区间。
    兼容常见 JP 6.x 行格式（字段名可能略有出入，可按需调整正则）
    """
    # 例子（前缀为 epoch 秒）：1695091234 RAM 1234/32168MB ... GR3D_FREQ 45% ... VDD_SYS_GPU 1200mW
    pat = re.compile(
        r"^(\d+)\s+.*GR3D(?:_FREQ)?\s+(\d+)%.*RAM\s+(\d+)/(\d+)MB.*?VDD_SYS_GPU\s+(\d+)mW"
    )

    g, ram_used, gpu_mw = [], [], []
    for line in tegra_log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        ts = int(m.group(1))
        if ts < t0 or ts > t1:
            continue
        g.append(int(m.group(2)))        # GPU load %
        ram_used.append(int(m.group(3))) # MB
        gpu_mw.append(int(m.group(5)))   # mW

    out: Dict[str, Any] = {"samples": len(g)}
    if g:
        out.update({
            "gpu_load_avg_pct": statistics.mean(g),
            "gpu_load_p95_pct": statistics.quantiles(g, n=20)[18] if len(g)>=20 else max(g),
        })
    if gpu_mw:
        out.update({"gpu_power_avg_w": statistics.mean(gpu_mw) / 1000.0})
    if ram_used:
        out.update({"ram_used_avg_gb": statistics.mean(ram_used) / 1024.0})
    return out

def parse_tegrastats_window_jetpack64(log_path: str, t0: float, t1: float):
    log = pathlib.Path(log_path)
    if not log.exists():
        print("tegrastats log not found:", log)
        return {}

    gpu, ram_used_mb, pwr_mw = [], [], []

    # 更稳的做法：取每行第一个 token 作为 epoch 秒，其余用宽松的正则提取
    with log.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            ts = int(parts[0])
            if ts < t0 or ts > t1:
                continue

            # GPU 负载：兼容 GR3D 和 GR3D_FREQ，允许后缀 @[...]
            m_gpu = re.search(r"GR3D(?:_FREQ)?\s+(\d+)%", line)
            if m_gpu:
                gpu.append(int(m_gpu.group(1)))

            # 内存：RAM 12345/67890MB
            m_ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if m_ram:
                ram_used_mb.append(int(m_ram.group(1)))

            # 功耗：兼容 VDD_SYS_GPU 或 VDD_GPU_SOC；数值可能是 2393mW/3623mW，取第一个
            m_pwr = re.search(r"(VDD_SYS_GPU|VDD_GPU_SOC)\s+(\d+)mW", line)
            if m_pwr:
                pwr_mw.append(int(m_pwr.group(2)))

    out = {"samples": len(gpu) or len(ram_used_mb) or len(pwr_mw)}
    if gpu:
        out["gpu_load_avg_pct"] = statistics.mean(gpu)
        out["gpu_load_p95_pct"] = statistics.quantiles(gpu, n=20)[18] if len(gpu) >= 20 else max(gpu)
    if pwr_mw:
        out["gpu_power_avg_w"] = statistics.mean(pwr_mw) / 1000.0
    if ram_used_mb:
        out["ram_used_avg_gb"] = statistics.mean(ram_used_mb) / 1024.0
    return out

def save_json(obj: Dict[str, Any], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def save_csv(rows: List[Dict[str, Any]], path: pathlib.Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="GR00T Inference Benchmark (container)")
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.5-3B")
    ap.add_argument("--embodiment", default="gr1")
    ap.add_argument("--data-config", default="fourier_gr1_arms_only")
    ap.add_argument("--dataset-path", default="/workspace/demo_data/robot_sim.PickNPlace")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # 可选：只在推理窗口解析 tegrastats
    ap.add_argument("--tegrastats", type=str, default=None,
                    help="Path to host-side tegrastats_ts.log mounted at /data/....")
    ap.add_argument("--markers", type=str, default="/data/markers.json")

    # 输出
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=None)

    args = ap.parse_args()

    setup_torch()

    # 让 HF 缓存走持久卷（可选，避免重复下载）
    os.environ.setdefault("HF_HOME", "/data/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf")

    print(f"Device: {args.device}")
    print(f"Model : {args.model_path}")
    print(f"Config: {args.data_config}")
    print(f"Data  : {args.dataset_path}")
    sys.stdout.flush()

    policy, dataset = load_policy_and_data(
        model_path=args.model_path,
        embodiment_tag=args.embodiment,
        data_config_key=args.data_config,
        dataset_path=args.dataset_path,
        device=args.device,
    )

    # 基准
    bench = run_benchmark(policy, dataset, args.warmup, args.iters)
    print(f"\n== Inference ==")
    print(f"count={bench['count']}")
    print(f"avg={bench['avg_s']:.4f}s | p50={bench['p50_s']:.4f}s | p95={bench['p95_s']:.4f}s "
          f"| throughput={bench['throughput_rps']:.2f} req/s (≈ FPS)")

    # 记录时间窗口（供外部对齐）
    markers_path = pathlib.Path(args.markers)
    save_json({"start": bench["window_start"], "end": bench["window_end"]}, markers_path)
    print(f"saved markers -> {markers_path}")

    summary: Dict[str, Any] = {
        "model_path": args.model_path,
        "embodiment": args.embodiment,
        "data_config": args.data_config,
        "dataset_path": args.dataset_path,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        **{k: v for k, v in bench.items() if k != "latencies_s"},
    }

    # 解析 tegrastats（可选）
    # 解析 tegrastats（可选）
    if args.tegrastats:
        tegra_path = pathlib.Path(args.tegrastats)
        if tegra_path.exists():
            # 使用新的解析器，兼容 GR3D/GR3D_FREQ、VDD_SYS_GPU/VDD_GPU_SOC 等格式
            tegra = parse_tegrastats_window_jetpack64(
                str(tegra_path),
                bench["window_start"],
                bench["window_end"],
            )
            print(f"\n== tegrastats (window {bench['window_end'] - bench['window_start']:.2f}s) ==")
            if not tegra or not tegra.get("samples", 0):
                print("samples: 0  (检查时间窗口是否与日志重叠，或确认日志字段匹配)")
            else:
                for k, v in tegra.items():
                    print(f"{k}: {v}")
            summary.update({"tegrastats": tegra})
        else:
            print(f"\ntegrastats log not found: {tegra_path} (skip)")


    # 输出结果
    if args.out_json:
        save_json(summary, pathlib.Path(args.out_json))
        print(f"\nsaved summary json -> {args.out_json}")

    if args.out_csv:
        # 扁平化一行 CSV（仅基础字段 + tegra均值）
        row = {k: v for k, v in summary.items() if k not in ("latencies_s", "tegrastats")}
        if "tegrastats" in summary and isinstance(summary["tegrastats"], dict):
            for k, v in summary["tegrastats"].items():
                row[f"tegra.{k}"] = v
        save_csv([row], pathlib.Path(args.out_csv))
        print(f"saved summary csv  -> {args.out_csv}")


if __name__ == "__main__":
    main()
