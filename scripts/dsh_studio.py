# -*- coding: utf-8 -*-
"""MOSS + Qwen 出图控制台（Gradio，图像优先排布）。

界面原则：**图放主位，参数收起来**。每个页签都是
「源图1 | 源图2 | 效果预览」三图并排，提示词就在下面直接填。

启动：python dsh_studio.py  →  http://127.0.0.1:7870
"""
import json
import os
import random
import shutil
import time
import urllib.request
import uuid

# httpx（gradio 依赖）解析 NO_PROXY 时会被 "[::1]" 这类条目噎住，
# 报 Invalid port: ':1]'。本机只连 127.0.0.1，直接把 no_proxy 收窄。
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "localhost,127.0.0.1"
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

import gradio as gr

MOSS = "http://127.0.0.1:8189"
QWEN = "http://127.0.0.1:8188"

MOSS_ROOT = r"D:\MOSS-Image\ComfyUI-MOSS"
QWEN_ROOT = r"D:\ComfyUI_windows_portable\ComfyUI"

MOSS_T2I = os.path.join(MOSS_ROOT, "workflows_moss_t2i_fix.api.json")
MOSS_EDIT = os.path.join(MOSS_ROOT, "wf_r2_white_gown.api.json")

OUT_DIR = r"D:\MOSS-Image\outputs\studio"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- HTTP
def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with _opener().open(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url, timeout=60):
    with _opener().open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _submit(base, wf):
    resp = _post(base + "/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    if resp.get("node_errors"):
        raise RuntimeError("节点报错: " + json.dumps(resp["node_errors"], ensure_ascii=False)[:900])
    return resp["prompt_id"]


def _wait(base, pid, timeout=2400, on_tick=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = _get(base + "/history/" + pid)
        except Exception:
            h = {}
        if pid in h:
            return h[pid]
        if on_tick:
            on_tick(int(time.time() - t0))
        time.sleep(2)
    raise TimeoutError("超时 %ds" % timeout)


def _collect(entry, root):
    files = []
    for out in (entry.get("outputs") or {}).values():
        for img in (out.get("images") or []):
            p = os.path.join(root, img.get("type") or "output",
                             img.get("subfolder") or "", img["filename"])
            if os.path.exists(p):
                files.append(p)
    return files


def _stage(src, root, name):
    if not src:
        raise RuntimeError("没有收到图片")
    dst = os.path.join(root, "input", name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    return name


def _load(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _status(entry):
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        for m in (st.get("messages") or [])[:4]:
            if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
                return "执行错误: " + json.dumps(m[1], ensure_ascii=False)[:700]
        return "执行错误（无详情）"
    return ""


def _keep(src, tag):
    dst = os.path.join(OUT_DIR, "%s_%d.png" % (tag, int(time.time())))
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------- ① 文生图
def run_t2i(prompt, negative, width, height, steps, cfg, seed, progress=gr.Progress()):
    try:
        wf = _load(MOSS_T2I)
        wf["4"]["inputs"]["text"] = prompt
        wf["5"]["inputs"]["text"] = negative
        wf["6"]["inputs"]["width"] = int(width)
        wf["6"]["inputs"]["height"] = int(height)
        wf["16"]["inputs"]["steps"] = int(steps)
        wf["16"]["inputs"]["cfg"] = float(cfg)
        wf["16"]["inputs"]["seed"] = int(seed) if int(seed) >= 0 else random.randint(1, 2**31 - 1)
        wf["18"]["inputs"]["filename_prefix"] = "studio/t2i"
        progress(0.05, desc="提交到 MOSS…")
        pid = _submit(MOSS, wf)
        entry = _wait(MOSS, pid, on_tick=lambda s: progress(min(0.9, 0.1 + s / 400),
                                                           desc="生成中 %ds" % s))
        err = _status(entry)
        if err:
            return None, err
        files = _collect(entry, MOSS_ROOT)
        if not files:
            return None, "没有产出图片"
        out = _keep(files[0], "t2i")
        return out, "完成 → " + out
    except Exception as e:
        return None, "失败: %s" % e


# ---------------------------------------------------------------- ② 换装/编辑
def run_edit(image, prompt, negative, ref_edge, strength, steps, cfg, seed,
             progress=gr.Progress()):
    try:
        if not image:
            return None, "请先在左边上传一张底图"
        name = _stage(image, MOSS_ROOT, "studio_edit_src.png")
        wf = _load(MOSS_EDIT)
        wf["6"]["inputs"]["image"] = name
        wf["9"]["inputs"]["text"] = prompt
        wf["10"]["inputs"]["text"] = negative
        wf["5"]["inputs"]["reference_max_edge"] = int(ref_edge)
        wf["5"]["inputs"]["strength"] = float(strength)
        wf["14"]["inputs"]["steps"] = int(steps)
        wf["14"]["inputs"]["cfg"] = float(cfg)
        wf["14"]["inputs"]["seed"] = int(seed) if int(seed) >= 0 else random.randint(1, 2**31 - 1)
        wf["16"]["inputs"]["filename_prefix"] = "studio/edit"
        progress(0.05, desc="提交到 MOSS…")
        pid = _submit(MOSS, wf)
        entry = _wait(MOSS, pid, on_tick=lambda s: progress(min(0.9, 0.1 + s / 300),
                                                           desc="编辑中 %ds" % s))
        err = _status(entry)
        if err:
            return None, err
        files = _collect(entry, MOSS_ROOT)
        if not files:
            return None, "没有产出图片"
        out = _keep(files[0], "edit")
        return out, "完成 → " + out
    except Exception as e:
        return None, "失败: %s" % e


# ---------------------------------------------------------------- ③ 前置：用提示词直接出身体图
def gen_body_for_swap(prompt, negative, width, height, steps, cfg, seed,
                      progress=gr.Progress()):
    """MOSS 文生图，结果直接填进「身体源图」那一栏。"""
    try:
        if not prompt.strip():
            return None, "提示词是空的"
        wf = _load(MOSS_T2I)
        wf["4"]["inputs"]["text"] = prompt
        wf["5"]["inputs"]["text"] = negative
        wf["6"]["inputs"]["width"] = int(width)
        wf["6"]["inputs"]["height"] = int(height)
        wf["16"]["inputs"]["steps"] = int(steps)
        wf["16"]["inputs"]["cfg"] = float(cfg)
        wf["16"]["inputs"]["seed"] = int(seed) if int(seed) >= 0 else random.randint(1, 2**31 - 1)
        wf["18"]["inputs"]["filename_prefix"] = "studio/body"
        progress(0.05, desc="MOSS 出身体图…")
        entry = _wait(MOSS, _submit(MOSS, wf),
                      on_tick=lambda s: progress(min(0.9, 0.1 + s / 400),
                                                 desc="出图中 %ds" % s))
        err = _status(entry)
        if err:
            return None, err
        files = _collect(entry, MOSS_ROOT)
        if not files:
            return None, "没有产出图片"
        out = _keep(files[0], "body")
        return out, "身体图已生成并填入左栏 → " + out
    except Exception as e:
        return None, "失败: %s" % e


def oneclick_swap(body_prompt, body_neg, bw, bh, b_steps, b_cfg, b_seed,
                  face_img, lora, lora_s, s_steps, s_cfg, s_seed, upscale_first,
                  progress=gr.Progress()):
    """提示词 → MOSS 出身体图 → 立刻 BFS 换脸，一步到位。"""
    body, msg = gen_body_for_swap(body_prompt, body_neg, bw, bh, b_steps, b_cfg, b_seed,
                                  progress=progress)
    if not body:
        return None, None, "身体图生成失败：" + msg
    progress(0.45, desc="身体图好了，开始换脸…")
    out, msg2 = run_swap(body, face_img, BFS_POS, BFS_NEG, lora, lora_s,
                         s_steps, s_cfg, s_seed, upscale_first, progress=progress)
    return body, out, msg2


# ---------------------------------------------------------------- ③ BFS 换脸
BFS_POS = ("将图1中人物的头部和面部替换为图2中人物的头部和面部，"
           "保持图1的姿势、服装、构图、背景和光线完全不变，"
           "新头部与图1的身体自然衔接，颈部过渡自然无缝隙，"
           "五官、脸型、肤色与图2完全一致，五官清晰精致，眼睛明亮有神，"
           "彩色写实摄影，超高清晰度，细节丰富，皮肤纹理真实，无文字无水印")
BFS_NEG = ("lowres, watermark, text, blurry, deformed, bad anatomy, bad hands, extra hands, "
           "extra arms, mutated hands, extra fingers, missing fingers, ghosting, double face, seam, "
           "blurry face, soft face, "
           "monochrome, black and white, grayscale, non-asian, caucasian, western features")


def _bfs_wf(body, face, seed, w, h, lora_s, steps, cfg, lora, prompt, negative, prefix):
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
              "inputs": {"model": ["3", 0], "lora_name": lora, "strength_model": float(lora_s)}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": prompt,
                         "image1": ["1", 0], "image2": ["2", 0]}},
        "8": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": negative,
                         "image1": ["1", 0], "image2": ["2", 0]}},
        "9": {"class_type": "EmptyLatentImage",
              "inputs": {"width": int(w), "height": int(h), "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["6", 0], "positive": ["7", 0], "negative": ["8", 0],
                          "latent_image": ["9", 0], "seed": int(seed), "steps": int(steps),
                          "cfg": float(cfg), "sampler_name": "euler", "scheduler": "simple",
                          "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


def run_swap(body_img, face_img, prompt, negative, lora, lora_s, steps, cfg, seed,
             upscale_first, progress=gr.Progress()):
    try:
        if not body_img:
            return None, "左边缺「身体源图」"
        if not face_img:
            return None, "中间缺「脸源图」"
        body = _stage(body_img, QWEN_ROOT, "studio_body.png")
        face = _stage(face_img, QWEN_ROOT, "studio_face.png")

        from PIL import Image
        im = Image.open(os.path.join(QWEN_ROOT, "input", body))
        w, h = im.width, im.height

        if upscale_first and max(w, h) < 1500:
            progress(0.1, desc="底图偏小，先放大 2x（脸会更清晰）…")
            up_wf = {
                "1": {"class_type": "LoadImage", "inputs": {"image": body}},
                "2": {"class_type": "UpscaleModelLoader",
                      "inputs": {"model_name": "4x-UltraSharp.pth"}},
                "3": {"class_type": "ImageUpscaleWithModel",
                      "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
                "4": {"class_type": "ImageScaleBy",
                      "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                 "scale_by": 0.5}},
                "5": {"class_type": "SaveImage",
                      "inputs": {"images": ["4", 0], "filename_prefix": "studio/up"}},
            }
            entry = _wait(QWEN, _submit(QWEN, up_wf), timeout=900)
            ups = _collect(entry, QWEN_ROOT)
            if ups:
                body = _stage(ups[0], QWEN_ROOT, "studio_body_up.png")
                im = Image.open(os.path.join(QWEN_ROOT, "input", body))
                w, h = im.width, im.height

        w -= w % 16
        h -= h % 16
        progress(0.3, desc="换脸中 %dx%d…" % (w, h))
        wf = _bfs_wf(body, face, seed, w, h, lora_s, steps, cfg, lora, prompt, negative,
                     "studio/swap")
        entry = _wait(QWEN, _submit(QWEN, wf),
                      on_tick=lambda s: progress(min(0.92, 0.35 + s / 300),
                                                 desc="换脸中 %ds" % s))
        err = _status(entry)
        if err:
            return None, err
        files = _collect(entry, QWEN_ROOT)
        if not files:
            return None, "没有产出图片"
        out = _keep(files[0], "swap")
        return out, "完成 %dx%d → %s" % (w, h, out)
    except Exception as e:
        return None, "失败: %s" % e


# ---------------------------------------------------------------- ④ 手部修复 + 超分
LIMB_POS = ("perfectly drawn hands, five fingers on each hand, natural hand pose, "
            "correct anatomy, detailed skin texture, realistic knuckles, "
            "color photograph, ultra sharp, no distortion")
LIMB_NEG = ("bad hands, extra hands, three hands, extra arms, extra fingers, missing fingers, "
            "fused fingers, mutated hands, deformed limbs, blurry, lowres, watermark, text, "
            "monochrome, grayscale")


def _limb_wf(img, denoise, steps, cfg, seed, target, prompt, negative, prefix):
    bbox = {"hand": "bbox/hand_yolov8s.pt", "face": "bbox/face_yolov8m.pt"}[target]
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": img}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "qwen_image_2.1_bf16.safetensors", "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image",
                         "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}},
        "7": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": bbox}},
        "8": {"class_type": "SAMLoader",
              "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"}},
        "9": {"class_type": "FaceDetailer",
              "inputs": {"image": ["1", 0], "model": ["2", 0], "clip": ["3", 0], "vae": ["4", 0],
                         "positive": ["5", 0], "negative": ["6", 0],
                         "bbox_detector": ["7", 0], "sam_model_opt": ["8", 0],
                         "guide_size": 512, "guide_size_for": True, "max_size": 1024,
                         "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": float(denoise), "feather": 8,
                         "noise_mask": True, "force_inpaint": True,
                         "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                         "sam_detection_hint": "center-1", "sam_dilation": 0,
                         "sam_threshold": 0.93, "sam_bbox_expansion": 0,
                         "sam_mask_hint_threshold": 0.7,
                         "sam_mask_hint_use_negative": "False",
                         "drop_size": 10, "wildcard": "", "cycle": 1,
                         "inpaint_model": False, "noise_mask_feather": 20}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def run_limb(image, target, denoise, steps, cfg, seed, then_upscale, scale,
             progress=gr.Progress()):
    try:
        if not image:
            return None, "请先上传一张图"
        name = _stage(image, QWEN_ROOT, "studio_limb_src.png")
        progress(0.1, desc="检测 + 局部重绘…")
        wf = _limb_wf(name, denoise, steps, cfg, seed, target, LIMB_POS, LIMB_NEG, "studio/limb")
        entry = _wait(QWEN, _submit(QWEN, wf),
                      on_tick=lambda s: progress(min(0.8, 0.15 + s / 300),
                                                 desc="重绘中 %ds" % s))
        err = _status(entry)
        if err:
            return None, err
        files = _collect(entry, QWEN_ROOT)
        if not files:
            return None, "没有产出图片"
        cur = files[0]

        if then_upscale:
            progress(0.85, desc="超分 %sx…" % scale)
            up_wf = {
                "1": {"class_type": "LoadImage",
                      "inputs": {"image": _stage(cur, QWEN_ROOT, "studio_up_src.png")}},
                "2": {"class_type": "UpscaleModelLoader",
                      "inputs": {"model_name": "4x-UltraSharp.pth"}},
                "3": {"class_type": "ImageUpscaleWithModel",
                      "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            }
            factor = float(scale) / 4.0
            src = ["3", 0]
            if abs(factor - 1.0) > 1e-6:
                up_wf["4"] = {"class_type": "ImageScaleBy",
                              "inputs": {"image": ["3", 0], "upscale_method": "lanczos",
                                         "scale_by": factor}}
                src = ["4", 0]
            up_wf["5"] = {"class_type": "SaveImage",
                          "inputs": {"images": src, "filename_prefix": "studio/up2"}}
            entry = _wait(QWEN, _submit(QWEN, up_wf), timeout=1200)
            ups = _collect(entry, QWEN_ROOT)
            if ups:
                cur = ups[0]

        out = _keep(cur, "limb")
        from PIL import Image
        im = Image.open(out)
        return out, "完成 %dx%d → %s" % (im.width, im.height, out)
    except Exception as e:
        return None, "失败: %s" % e


# ---------------------------------------------------------------- 界面
HINT = "### 结果会同时存到 `%s`" % OUT_DIR

with gr.Blocks(title="MOSS / Qwen 出图控制台") as demo:
    gr.Markdown(
        "# MOSS / Qwen 出图控制台\n"
        "每个页签都是 **源图 → 效果预览** 的直观看法，提示词就在图下面直接填。\n"
        "结果自动落盘，右侧预览可直接右键另存。")

    # ---------------- ① 文生图
    with gr.Tab("① MOSS 文生图"):
        with gr.Row():
            with gr.Column(scale=1, min_width=380):
                gr.Markdown("**① 提示词**")
                t2i_prompt = gr.Textbox(
                    label="正向提示词（想画什么写这里）", lines=8, value=(
                        "A full body studio portrait photograph of a young woman standing upright "
                        "facing the camera, wearing an elegant white satin evening gown with a long "
                        "flowing train, a pearl necklace and pearl earrings. Black seamless studio "
                        "background, soft key light from the left, gentle rim light, photorealistic, "
                        "sharp focus, 85mm lens, full body in frame, hands relaxed naturally at her "
                        "sides, five clearly drawn fingers on each hand."))
                t2i_neg = gr.Textbox(label="负向提示词（不想出现的写这里）", lines=3, value=(
                    "worst quality, low quality, blurry, deformed, extra limbs, extra fingers, "
                    "bad hands, extra hands, three hands, missing fingers, fused fingers, "
                    "watermark, text, cropped"))
            with gr.Column(scale=1, min_width=420):
                gr.Markdown("**② 效果预览**")
                t2i_out = gr.Image(label="生成的图", type="filepath", height=620)
                t2i_msg = gr.Textbox(label="状态", lines=2, interactive=False)
        with gr.Accordion("参数（一般不用动）", open=False):
            with gr.Row():
                t2i_w = gr.Slider(512, 1536, 768, step=16, label="宽")
                t2i_h = gr.Slider(512, 2048, 1024, step=16, label="高")
                t2i_steps = gr.Slider(10, 80, 50, step=1, label="步数")
                t2i_cfg = gr.Slider(1.0, 10.0, 3.0, step=0.1, label="CFG")
                t2i_seed = gr.Number(value=-1, label="种子（-1 随机）", precision=0)
        t2i_btn = gr.Button("开始生成", variant="primary", size="lg")
        t2i_btn.click(run_t2i,
                      [t2i_prompt, t2i_neg, t2i_w, t2i_h, t2i_steps, t2i_cfg, t2i_seed],
                      [t2i_out, t2i_msg])

    # ---------------- ② 换装/编辑
    with gr.Tab("② MOSS 换装 / 编辑"):
        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("**① 源图（要改的图）**")
                ed_img = gr.Image(label="上传底图", type="filepath", height=520)
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("**② 效果预览**")
                ed_out = gr.Image(label="改完的图", type="filepath", height=520)
                ed_msg = gr.Textbox(label="状态", lines=2, interactive=False)
        gr.Markdown("**③ 编辑指令（直接写要改什么）**")
        ed_prompt = gr.Textbox(label="编辑指令", lines=4, value=(
            "Change only the outfit into an elegant white silk evening gown with a long flowing "
            "train. Keep the exact same face, hair, pose, background, lighting and composition. "
            "Photorealistic, ultra sharp."))
        ed_neg = gr.Textbox(label="负向提示词", lines=2, value=(
            "worst quality, low quality, blurry, deformed, extra limbs, extra fingers, "
            "bad hands, watermark, text, cropped"))
        with gr.Accordion("参数（一般不用动）", open=False):
            with gr.Row():
                ed_ref = gr.Slider(384, 1536, 1024, step=64,
                                   label="参考图长边（越大越锁原图姿态）")
                ed_str = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="强度")
                ed_steps = gr.Slider(10, 60, 30, step=1, label="步数")
                ed_cfg = gr.Slider(1.0, 10.0, 4.0, step=0.1, label="CFG")
                ed_seed = gr.Number(value=-1, label="种子（-1 随机）", precision=0)
        ed_btn = gr.Button("开始换装", variant="primary", size="lg")
        ed_btn.click(run_edit,
                     [ed_img, ed_prompt, ed_neg, ed_ref, ed_str, ed_steps, ed_cfg, ed_seed],
                     [ed_out, ed_msg])

    # ---------------- ③ BFS 换脸（三图并排）
    with gr.Tab("③ BFS 换脸"):
        gr.Markdown(
            "**两种用法**：\n"
            "- **一键**：在最上面的提示词框里写要什么身体，点「一键出身体图 + 换脸」，全程不用切页面。\n"
            "- **分步**：自己上传身体图 → 上传脸源 → 点「开始换脸」。")

        with gr.Accordion("① 用提示词直接出身体图（一键链路）", open=True):
            bs_prompt = gr.Textbox(
                label="身体图提示词（想画什么样的身体/姿势/衣服）", lines=5, value=(
                    "A full body studio portrait photograph of a young woman standing upright "
                    "facing the camera, wearing an elegant white satin evening gown with a long "
                    "flowing train, a pearl necklace and pearl earrings. Black seamless studio "
                    "background, soft key light from the left, gentle rim light, photorealistic, "
                    "sharp focus, 85mm lens, full body in frame, hands relaxed naturally at her "
                    "sides, five clearly drawn fingers on each hand."))
            bs_neg = gr.Textbox(label="负向提示词", lines=2, value=(
                "worst quality, low quality, blurry, deformed, extra limbs, extra fingers, "
                "bad hands, extra hands, three hands, missing fingers, fused fingers, "
                "watermark, text, cropped"))
            with gr.Row():
                bs_w = gr.Slider(512, 1536, 768, step=16, label="宽")
                bs_h = gr.Slider(512, 2048, 1024, step=16, label="高")
                bs_steps = gr.Slider(10, 80, 50, step=1, label="步数")
                bs_cfg = gr.Slider(1.0, 10.0, 3.0, step=0.1, label="CFG")
                bs_seed = gr.Number(value=-1, label="种子（-1 随机）", precision=0)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("**② 身体源图**\n（上传，或用上面的提示词生成后自动填入）")
                sw_body = gr.Image(label="身体源图", type="filepath", height=520)
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("**③ 脸源图**\n（正面清晰照）")
                sw_face = gr.Image(label="脸源图", type="filepath", height=520)
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("**④ 效果预览**")
                sw_out = gr.Image(label="换脸结果", type="filepath", height=520)
                sw_msg = gr.Textbox(label="状态", lines=3, interactive=False)

        with gr.Row():
            gen_body_btn = gr.Button("只出身体图（填到左栏）", size="lg")
            oneclick_btn = gr.Button("一键出身体图 + 换脸", variant="primary", size="lg")

        gr.Markdown("**⑤ 换脸提示词（可改）**")
        sw_prompt = gr.Textbox(label="换脸提示词", lines=4, value=BFS_POS)
        sw_neg = gr.Textbox(label="负向提示词", lines=3, value=BFS_NEG)
        with gr.Accordion("参数（一般不用动）", open=False):
            with gr.Row():
                sw_lora = gr.Dropdown(
                    label="换脸 LoRA",
                    choices=["bfs_head_v1.1_qwen_2.1.safetensors",
                             "bfs_head_v1.1_alternative_qwen_2.1.safetensors",
                             "bfs_head_v1_qwen_2.1.safetensors"],
                    value="bfs_head_v1.1_qwen_2.1.safetensors")
                sw_lora_s = gr.Slider(0.5, 1.5, 1.0, step=0.05, label="LoRA 权重")
                sw_steps = gr.Slider(10, 60, 30, step=1, label="步数")
                sw_cfg = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="CFG")
                sw_seed = gr.Number(value=42, label="种子", precision=0)
                sw_up = gr.Checkbox(value=True, label="底图偏小时先放大 2x（推荐）")

        sw_btn = gr.Button("开始换脸", variant="primary", size="lg")

        gen_body_btn.click(gen_body_for_swap,
                           [bs_prompt, bs_neg, bs_w, bs_h, bs_steps, bs_cfg, bs_seed],
                           [sw_body, sw_msg])
        oneclick_btn.click(oneclick_swap,
                           [bs_prompt, bs_neg, bs_w, bs_h, bs_steps, bs_cfg, bs_seed,
                            sw_face, sw_lora, sw_lora_s, sw_steps, sw_cfg, sw_seed, sw_up],
                           [sw_body, sw_out, sw_msg])
        sw_btn.click(run_swap,
                     [sw_body, sw_face, sw_prompt, sw_neg, sw_lora, sw_lora_s,
                      sw_steps, sw_cfg, sw_seed, sw_up],
                     [sw_out, sw_msg])

    # ---------------- ④ 手部修复 + 超分
    with gr.Tab("④ 手部修复 + 超分"):
        with gr.Row():
            with gr.Column(scale=1, min_width=380):
                gr.Markdown("**① 源图**")
                lb_img = gr.Image(label="上传要修的图", type="filepath", height=520)
            with gr.Column(scale=1, min_width=380):
                gr.Markdown("**② 效果预览**")
                lb_out = gr.Image(label="修完的图", type="filepath", height=520)
                lb_msg = gr.Textbox(label="状态", lines=3, interactive=False)
        with gr.Accordion("参数（一般不用动）", open=False):
            with gr.Row():
                lb_target = gr.Dropdown(label="修复目标", choices=["hand", "face"], value="hand")
                lb_dn = gr.Slider(0.15, 0.85, 0.45, step=0.05, label="重绘幅度")
                lb_steps = gr.Slider(8, 40, 20, step=1, label="步数")
                lb_cfg = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="CFG")
                lb_seed = gr.Number(value=42, label="种子", precision=0)
                lb_up = gr.Checkbox(value=True, label="修复后超分")
                lb_scale = gr.Slider(2, 4, 4, step=1, label="放大倍数")
        lb_btn = gr.Button("开始修复", variant="primary", size="lg")
        lb_btn.click(run_limb,
                     [lb_img, lb_target, lb_dn, lb_steps, lb_cfg, lb_seed, lb_up, lb_scale],
                     [lb_out, lb_msg])

    gr.Markdown(HINT)


if __name__ == "__main__":
    # Gradio 默认只允许返回「当前工作目录」或系统临时目录里的文件，
    # 我们的产物在 D:\MOSS-Image\outputs\studio，必须显式放行，
    # 否则界面会报 InvalidPathError（图其实已经生成成功了）。
    demo.queue(max_size=8).launch(
        server_name="127.0.0.1",
        server_port=7870,
        show_error=True,
        inbrowser=False,
        allowed_paths=[OUT_DIR, MOSS_ROOT, QWEN_ROOT],
    )
