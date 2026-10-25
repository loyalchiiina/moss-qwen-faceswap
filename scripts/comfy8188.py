# -*- coding: utf-8 -*-
"""Submit an upscale / face-swap workflow to ComfyUI 8188 and report output.

Subcommands:
  upscale <input_name> [scale]     4x-UltraSharp, then optional rescale
  swap    <body> <face> [lora]     Qwen-Image-2.1 + BFS head LoRA

`<input_name>` must already live in ComfyUI's input dir (or pass an absolute
path and it will be copied there).
"""
import json
import os
import random
import shutil
import sys
import time
import urllib.request

from PIL import Image

API = "http://127.0.0.1:8188"
ROOT = r"D:\ComfyUI_windows_portable\ComfyUI"
INDIR = os.path.join(ROOT, "input")
OUTDIR = os.path.join(ROOT, "output")

SWAP_PROMPT = (
    "将图1中人物的头部和面部替换为图2中人物的头部和面部，"
    "保持图1的姿势、服装、构图、背景和光线完全不变，"
    "新头部与图1的身体自然衔接，颈部过渡自然无缝隙，"
    "五官、脸型、肤色与图2完全一致，五官清晰精致，眼睛明亮有神，"
    "彩色写实摄影，超高清晰度，细节丰富，皮肤纹理真实，无文字无水印"
)
SWAP_NEG = (
    "lowres, watermark, text, blurry, deformed, bad anatomy, bad hands, extra hands, "
    "extra arms, mutated hands, extra fingers, missing fingers, ghosting, double face, seam, "
    "blurry face, soft face, "
    "monochrome, black and white, grayscale, non-asian, caucasian, western features"
)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def submit(wf):
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(API + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["prompt_id"]


def wait_for(pid, timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(API + "/history/" + pid, timeout=20) as r:
            h = json.loads(r.read().decode())
        if pid in h:
            return h[pid]
        time.sleep(3)
    return None


def ensure_input(path_or_name):
    """Return the bare filename ComfyUI can load, copying it in if needed."""
    if os.path.isabs(path_or_name):
        if not os.path.exists(path_or_name):
            raise SystemExit("missing: " + path_or_name)
        name = os.path.basename(path_or_name)
        dst = os.path.join(INDIR, name)
        if os.path.abspath(path_or_name) != os.path.abspath(dst):
            shutil.copy2(path_or_name, dst)
        return name
    if os.path.exists(os.path.join(INDIR, path_or_name)):
        return path_or_name
    raise SystemExit("not in input dir: " + path_or_name)


def build_upscale(img, prefix, scale=4):
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": img}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}},
        "3": {"class_type": "ImageUpscaleWithModel",
              "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
    }
    factor = scale / 4.0
    if abs(factor - 1.0) > 1e-6:
        wf["4"] = {"class_type": "ImageScaleBy",
                   "inputs": {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": factor}}
        src = ["4", 0]
    else:
        src = ["3", 0]
    wf["5"] = {"class_type": "SaveImage", "inputs": {"images": src, "filename_prefix": prefix}}
    return wf


def build_swap(body, face, prefix, seed, w, h, lora_strength=1.0, steps=30, cfg=2.5,
               lora="bfs_head_v1.1_qwen_2.1.safetensors"):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": body}},
        "2": {"class_type": "LoadImage", "inputs": {"image": face}},
        "3": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image",
                         "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "6": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["3", 0], "lora_name": lora, "strength_model": lora_strength}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": SWAP_PROMPT,
                         "image1": ["1", 0], "image2": ["2", 0]}},
        "8": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": SWAP_NEG,
                         "image1": ["1", 0], "image2": ["2", 0]}},
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
                          "latent_image": ["9", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


def report(res):
    st = res.get("status", {})
    if st.get("status_str") == "error":
        for m in st.get("messages", [])[:5]:
            log("  ERR", str(m)[:400])
        return None
    files = []
    for v in res.get("outputs", {}).values():
        for o in v.get("images", []):
            fp = os.path.join(OUTDIR, o.get("subfolder") or "", o["filename"])
            if os.path.exists(fp):
                im = Image.open(fp)
                log("OK %s  %dx%d  %.0fKB" % (fp, im.width, im.height, os.path.getsize(fp) / 1024))
            files.append(fp)
    return files[0] if files else None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd = sys.argv[1]

    if cmd == "upscale":
        img = ensure_input(sys.argv[2])
        scale = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        im0 = Image.open(os.path.join(INDIR, img))
        log("input %s  %dx%d" % (img, im0.width, im0.height))
        prefix = "UP_%s_%dx" % (os.path.splitext(img)[0][:20], scale)
        res = wait_for(submit(build_upscale(img, prefix, scale)))
        return 0 if (res and report(res)) else 1

    if cmd == "swap":
        body = ensure_input(sys.argv[2])
        face = ensure_input(sys.argv[3])
        lora_s = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        im = Image.open(os.path.join(INDIR, body))
        w, h = im.width - im.width % 16, im.height - im.height % 16
        log("body %s %dx%d -> %dx%d | face %s | lora %.2f" % (body, im.width, im.height, w, h, face, lora_s))
        seed = random.randint(1, 999999999)
        prefix = "BFS_" + os.path.splitext(body)[0][:16]
        log("seed", seed)
        res = wait_for(submit(build_swap(body, face, prefix, seed, w, h, lora_s)))
        return 0 if (res and report(res)) else 1

    print("unknown subcommand:", cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main())
