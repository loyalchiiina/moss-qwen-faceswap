# -*- coding: utf-8 -*-
"""给真人脸打马赛克，用于公开仓库的示意图。

自动检测图中所有人脸，按检测框外扩一点做像素化（mosaic），
不是模糊 —— 马赛克更彻底，避免逆向。

用法:
  python mosaic_faces.py <输入图> <输出图> [--blocks 12] [--pad 0.12]
"""
import argparse
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

INSIGHT_ROOT = r"D:\ComfyUI_windows_portable\ComfyUI\models\insightface"


def rd(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def wr(p, img):
    ext = os.path.splitext(p)[1].lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imencode(ext, img, params)[1].tofile(p)


def mosaic(img, x1, y1, x2, y2, blocks):
    """Replace the rect with a pixelated version."""
    h, w = y2 - y1, x2 - x1
    if h < 2 or w < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, blocks), max(1, int(blocks * h / max(1, w)))),
                       interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, default=12, help="马赛克块数（越小越糊）")
    ap.add_argument("--pad", type=float, default=0.12, help="检测框外扩比例")
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", root=INSIGHT_ROOT,
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = rd(args.src)
    if img is None:
        print("读图失败:", args.src)
        return 1

    faces = app.get(img)
    if not faces:
        print("未检测到人脸，原样输出")
        wr(args.dst, img)
        return 0

    H, W = img.shape[:2]
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        pw = int((x2 - x1) * args.pad)
        ph = int((y2 - y1) * args.pad)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(W, x2 + pw)
        y2 = min(H, y2 + ph)
        mosaic(img, x1, y1, x2, y2, args.blocks)
        print("  打码 bbox=(%d,%d)-(%d,%d)" % (x1, y1, x2, y2))

    wr(args.dst, img)
    print("saved:", args.dst, "faces=%d" % len(faces))
    return 0


if __name__ == "__main__":
    sys.exit(main())
