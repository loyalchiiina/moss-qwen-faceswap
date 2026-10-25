# -*- coding: utf-8 -*-
"""手部 / 四肢局部重绘：hand_yolov8s 检测手 → SAM 抠形 → FaceDetailer 重绘。

用法:
  python limb_fix.py <input_in_comfy_input_dir> [denoise] [steps]

依赖 Impact Pack 的 UltralyticsDetectorProvider / SAMLoader / FaceDetailer。
检测模型: bbox/hand_yolov8s.pt, sam_vit_b_01ec64.pth
重绘用 Qwen-Image-2.1（本机唯一能保证画质的底模）。
"""
import json
import os
import sys
import time
import urllib.request

API = "http://127.0.0.1:8188"
ROOT = r"D:\ComfyUI_windows_portable\ComfyUI"
INDIR = os.path.join(ROOT, "input")
OUTDIR = os.path.join(ROOT, "output")

POS = ("perfectly drawn hands, five fingers on each hand, natural hand pose, "
       "correct anatomy, detailed skin texture, realistic knuckles, "
       "color photograph, ultra sharp, no distortion")
NEG = ("bad hands, extra hands, three hands, extra arms, extra fingers, missing fingers, "
       "fused fingers, mutated hands, deformed limbs, blurry, lowres, watermark, text, "
       "monochrome, grayscale")


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


def build(img, prefix, denoise=0.45, steps=20, cfg=2.5, seed=42, target="hand"):
    """target: hand | face | person  -> 决定用哪个检测模型"""
    bbox_model = {
        "hand": "bbox/hand_yolov8s.pt",
        "face": "bbox/face_yolov8m.pt",
    }[target]
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": img}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image",
                         "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": POS}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}},
        "7": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": bbox_model}},
        "8": {"class_type": "SAMLoader",
              "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"}},
        "9": {"class_type": "FaceDetailer",
              "inputs": {
                  "image": ["1", 0],
                  "model": ["2", 0],
                  "clip": ["3", 0],
                  "vae": ["4", 0],
                  "positive": ["5", 0],
                  "negative": ["6", 0],
                  "bbox_detector": ["7", 0],
                  "sam_model_opt": ["8", 0],
                  "guide_size": 512,
                  "guide_size_for": True,
                  "max_size": 1024,
                  "seed": seed,
                  "steps": steps,
                  "cfg": cfg,
                  "sampler_name": "euler",
                  "scheduler": "simple",
                  "denoise": denoise,
                  "feather": 8,
                  "noise_mask": True,
                  "force_inpaint": True,
                  "bbox_threshold": 0.5,
                  "bbox_dilation": 10,
                  "bbox_crop_factor": 3.0,
                  "sam_detection_hint": "center-1",
                  "sam_dilation": 0,
                  "sam_threshold": 0.93,
                  "sam_bbox_expansion": 0,
                  "sam_mask_hint_threshold": 0.7,
                  "sam_mask_hint_use_negative": "False",
                  "drop_size": 10,
                  "wildcard": "",
                  "cycle": 1,
                  "inpaint_model": False,
                  "noise_mask_feather": 20,
              }},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    img = sys.argv[1]
    denoise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    target = sys.argv[4] if len(sys.argv) > 4 else "hand"

    if not os.path.exists(os.path.join(INDIR, img)):
        raise SystemExit("not in input dir: " + img)

    prefix = "LIMB_" + os.path.splitext(img)[0][:18]
    log("limb fix | %s | target=%s denoise=%.2f steps=%d" % (img, target, denoise, steps))
    res = wait_for(submit(build(img, prefix, denoise, steps, target=target)))
    if not res:
        log("TIMEOUT")
        return 1
    st = res.get("status", {})
    if st.get("status_str") == "error":
        for m in st.get("messages", [])[:5]:
            log("  ERR", str(m)[:500])
        return 1
    for v in res.get("outputs", {}).values():
        for o in v.get("images", []):
            fp = os.path.join(OUTDIR, o.get("subfolder") or "", o["filename"])
            log("OK", fp, os.path.getsize(fp) // 1024, "KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
