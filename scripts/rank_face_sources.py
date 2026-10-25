# -*- coding: utf-8 -*-
"""从照片库里挑最适合做换脸脸源的照片。

评分三维（技能包 face-swap-route1 §二 的标准）：
  1. 脸像素尺寸   —— 越大越好（决定换脸后能保留多少细节）
  2. 正面度       —— 双眼/鼻/嘴角 5 点对称性，越正越好
  3. 清晰度       —— 脸部区域 Laplacian 方差，越锐越好

用法:
  python rank_face_sources.py <目录> [--top 20] [--prefilter 400]

为了速度：先按文件体积降序取前 prefilter 张（大文件通常分辨率高），
再逐张跑 insightface。
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

INSIGHT_ROOT = r"D:\ComfyUI_windows_portable\ComfyUI\models\insightface"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

_APP = None


def get_app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", root=INSIGHT_ROOT,
                            providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(640, 640))
    return _APP


def rd(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def score_one(path):
    try:
        img = rd(path)
        if img is None or min(img.shape[:2]) < 256:
            return None
        app = get_app()
        faces = app.get(img)
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        if f.det_score < 0.75:
            return None
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        fw, fh = x2 - x1, y2 - y1
        if min(fw, fh) < 120:
            return None

        # 清晰度：脸部区域的 Laplacian 方差
        pad = int(min(fw, fh) * 0.1)
        X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
        X2, Y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 正面度：5 关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）的对称性
        kps = f.kps
        le, re_, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
        eye_mid = (le + re_) / 2.0
        mouth_mid = (lm + rm) / 2.0
        # 鼻尖应落在 眼中心—嘴中心 的竖直连线上
        axis = mouth_mid - eye_mid
        axis_len = np.linalg.norm(axis) + 1e-6
        dev = abs(np.cross(axis / axis_len, nose - eye_mid))
        frontal = 1.0 - min(1.0, dev / (fh * 0.18))
        # 两眼水平度
        tilt = abs(le[1] - re_[1]) / (abs(le[0] - re_[0]) + 1e-6)
        frontal *= 1.0 - min(0.6, tilt)

        return {
            "path": path,
            "fw": fw, "fh": fh,
            "det": float(f.det_score),
            "sharp": sharp,
            "frontal": float(frontal),
            "img_w": img.shape[1], "img_h": img.shape[0],
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--prefilter", type=int, default=400)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print("skip missing:", d)
            continue
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(EXTS):
                    p = os.path.join(root, n)
                    try:
                        files.append((os.path.getsize(p), p))
                    except OSError:
                        pass
    files.sort(reverse=True)
    cand = [p for _, p in files[:args.prefilter]]
    print("库里 %d 张，取体积最大的 %d 张做检测" % (len(files), len(cand)), flush=True)

    t0 = time.time()
    with Pool(args.procs) as pool:
        results = [r for r in pool.imap_unordered(score_one, cand, chunksize=4) if r]
    print("有效脸 %d 张，耗时 %.0fs" % (len(results), time.time() - t0), flush=True)

    # 综合分：脸像素（主）+ 清晰度 + 正面度
    for r in results:
        r["score"] = (min(r["fw"], r["fh"]) / 100.0) ** 1.2 \
                     * (r["sharp"] / 100.0) ** 0.5 \
                     * max(0.05, r["frontal"])
    results.sort(key=lambda r: -r["score"])

    print()
    print("%-3s %-9s %-8s %-7s %-7s %s" % ("#", "脸px", "清晰度", "正面度", "det", "文件"))
    for i, r in enumerate(results[:args.top], 1):
        print("%-3d %-9s %-8.1f %-7.3f %-7.3f %s" % (
            i, "%dx%d" % (r["fw"], r["fh"]), r["sharp"], r["frontal"], r["det"],
            os.path.basename(r["path"])))
        print("      %s" % r["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
