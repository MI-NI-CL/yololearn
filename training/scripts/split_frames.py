# 拆帧脚本：把 training\raw_videos\ 里所有视频拆成一帧帧图片，用于制作训练素材
# 运行命令（在 yololearn 目录执行）：
#   venv\Scripts\python training\scripts\split_frames.py
# 用法：
#   1. 把拍好的视频放到 training\raw_videos\ 目录（可多个，自动全部处理）
#   2. 运行上面的命令
# 说明：
#   - EVERY_N 越大，取出的帧越少越稀疏（避免连续帧太像，浪费标注时间）
#   - 默认每 5 帧取 1 帧；想让训练素材更丰富就改成 3，想少标一点改成 10
#   - 文件名带视频来源前缀，方便追溯是哪段视频的帧

import os
import glob
import cv2
import numpy as np

# 脚本所在目录（training\scripts\），所有路径基于它计算，与运行位置无关
BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 想改就改这里 =====
VIDEO_DIR = os.path.join(BASE, "..", "raw_videos")  # 输入视频目录 training\raw_videos\
OUT_DIR = os.path.join(BASE, "..", "frames")        # 输出文件夹 training\frames\
EVERY_N = 3                  # 每 3 帧取 1 帧（间隔，避免帧太像）
# =====================

os.makedirs(OUT_DIR, exist_ok=True)

# 保存图片（兼容中文路径）：
# OpenCV 的 cv2.imwrite 在 Windows 中文路径下会失败，改用 imencode 转字节 + 文件写入
def save_image(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    buf.tofile(path)   # 用 numpy 的 tofile 写入，支持中文路径
    return True

# 找目录下所有视频文件（mp4/avi/mov/mkv）
videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
                glob.glob(os.path.join(VIDEO_DIR, "*.avi")) +
                glob.glob(os.path.join(VIDEO_DIR, "*.mov")) +
                glob.glob(os.path.join(VIDEO_DIR, "*.mkv")))

if not videos:
    print(f"没有找到视频文件！请放到 {VIDEO_DIR} 目录（支持 mp4/avi/mov/mkv）")
    exit(1)

print(f"找到 {len(videos)} 个视频，每 {EVERY_N} 帧取 1 帧")

saved_total = 0
for v in videos:
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        print(f"打不开: {os.path.basename(v)}，跳过")
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 用视频名前 8 个字符做前缀，区分不同视频的帧
    prefix = os.path.splitext(os.path.basename(v))[0][:8]
    print(f"处理 {os.path.basename(v)[:20]}... 总帧数 {total}")

    i = 0        # 原始帧计数
    saved = 0    # 本视频已保存帧数
    while True:
        ret, frame = cap.read()      # 读一帧，ret=True 表示成功
        if not ret:                  # 读到头了（视频结束）
            break
        if i % EVERY_N == 0:         # 每隔 EVERY_N 帧才保存一张
            name = os.path.join(OUT_DIR, f"{prefix}_{saved:04d}.jpg")
            if save_image(name, frame):
                saved += 1
        i += 1

    cap.release()
    saved_total += saved
    print(f"  [OK] 完成: 保存 {saved} 张")

print(f"全部完成! 共保存 {saved_total} 张到 {OUT_DIR}/ 文件夹")
