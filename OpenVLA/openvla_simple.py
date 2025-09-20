#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal OpenVLA inference demo (image OR dataset frame) — Jetson friendly.

Examples:
  # A) local image
  python openvla_min_infer.py --image /data/sample.jpg \
    --prompt "In: What action should the robot take to {open the drawer}?\\nOut:" \
    --dtype fp16 --no_sample

  # B) grab one frame from a HuggingFace dataset (e.g., LIBERO / LeRobot)
  python openvla_min_infer.py --dataset physical-intelligence/libero --dtype fp16 --no_sample

Recommended env (inside container):
  export HF_HOME=/data/hf
  export HUGGINGFACE_HUB_CACHE=/data/hf
  export TRANSFORMERS_CACHE=/data/hf
  export HF_TOKEN=hf_xxx            # if the model is gated
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla/openvla-7b", help="HF repo id or local path")
    ap.add_argument("--image", help="Path to an RGB image")
    ap.add_argument("--dataset", help="HF dataset name (e.g., physical-intelligence/libero, lerobot/pusht)")
    ap.add_argument("--prompt", default="In: What action should the robot take to {open the drawer}?\\nOut:")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="Model compute dtype")
    ap.add_argument("--unnorm-key", default="bridge_orig", help="BridgeData V2 unnorm key")
    ap.add_argument("--no_sample", action="store_true", help="Deterministic output")
    return ap.parse_args()


def _first_image_from_obj(obj):
    """Recursively find first (H,W,3/4) image-like array and return as PIL.Image."""
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


def load_one_image_from_dataset(ds_name: str) -> Image.Image:
    from datasets import load_dataset  # lazy import
    ds = load_dataset(ds_name, split="train", streaming=False)
    img = _first_image_from_obj(ds[0])
    if img is None:
        for i in range(min(10, len(ds))):
            img = _first_image_from_obj(ds[i])
            if img is not None:
                break
    if img is None:
        raise RuntimeError(
            f"Could not find an RGB image in first samples of '{ds_name}'. "
            f"Inspect structure with: from datasets import load_dataset; ds=load_dataset('{ds_name}'); print(ds[0])"
        )
    return img


def main():
    args = parse_args()
    if not args.image and not args.dataset:
        raise SystemExit("Please provide --image or --dataset")

    # dtype & token
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    token = os.getenv("HF_TOKEN", None)

    # load image
    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"Loaded image: {args.image} size={img.size}")
    else:
        print(f"Loading one frame from dataset: {args.dataset}")
        img = load_one_image_from_dataset(args.dataset)
        print(f"Dataset frame size={img.size}")

    # processor & model
    print(f"Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=token,
        # use_fast=True  # uncomment if the model supports fast processor
    )

    print(f"Loading model: {args.model} (dtype={dtype}, device={args.device})")
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,       # affects model weight compute dtype
        low_cpu_mem_usage=True,
        token=token,
        # attn_implementation="flash_attention_2",  # enable only if flash-attn is installed and compatible
    ).to(args.device)

    vla = vla.to(dtype=torch.float16)

    prompt = args.prompt
    print("Prompt:", prompt)

    # preprocess
    inputs = processor(prompt, img)

    # move inputs to device with correct dtypes:
    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if torch.is_floating_point(v):
                moved[k] = v.to(args.device, dtype=dtype)   # float tensors -> fp16/fp32
            else:
                moved[k] = v.to(args.device)                # keep ids/masks as int/bool
        else:
            moved[k] = v

    # ensure ids/masks have valid types
    if "input_ids" in moved:
        moved["input_ids"] = moved["input_ids"].to(args.device, dtype=torch.long)
    if "attention_mask" in moved and moved["attention_mask"].dtype not in (torch.bool, torch.long):
        moved["attention_mask"] = moved["attention_mask"].to(args.device, dtype=torch.long)

    # inference
    vla.eval()
    with torch.inference_mode():
        action = vla.predict_action(
            **moved,
            unnorm_key=args.unnorm_key,
            do_sample=not args.no_sample
        )

    # pretty print
    if isinstance(action, dict):
        print("=== Predicted Action (per joint) ===")
        for k, v in action.items():
            try:
                print(f"{k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype',type(v))}")
            except Exception:
                print(f"{k}: type={type(v)}")
    else:
        try:
            print("Predicted action shape:", tuple(action.shape))
        except Exception:
            print("Predicted action:", action)


if __name__ == "__main__":
    main()
