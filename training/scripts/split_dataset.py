# 数据集划分脚本：把 images/labels 里的数据按比例分成 train/val 两个子文件夹
# 运行命令（在 yololearn 目录执行）：
#   venv\Scripts\python training\scripts\split_dataset.py
# 前置：
#   1. 已用标注工具（X-AnyLabeling）标注图片，并导出 YOLO .txt 格式
#   2. 把导出的所有图片放进 training\datasets\dataset1\images\
#      所有 .txt 标注放进 training\datasets\dataset1\labels\（文件名和图片一致）
# 说明：
#   - 按 VAL_RATIO 比例随机划分，图片和对应标注同步移动（不会错配）
#   - 图名必须和 txt 名一致（如 a.jpg 配 a.txt），脚本只处理有标注的图
#   - 已划分过（train/val 里有内容）会清空重分，可反复运行

import os
import glob
import random
import shutil

# 脚本所在目录（training\scripts\），所有路径基于它计算，与运行位置无关
BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 想改就改这里 =====
DS_DIR = os.path.join(BASE, "..", "datasets", "dataset1")  # 数据集目录
VAL_RATIO = 0.2              # 验证集比例（20%），其余为训练集
SEED = 42                    # 随机种子（固定值=每次划分结果一样，方便复现）
# =====================

IMG_DIR = os.path.join(DS_DIR, "images")
LBL_DIR = os.path.join(DS_DIR, "labels")

# 1. 先清空已有的 train/val 子文件夹（重分）
for split in ("train", "val"):
    for base_dir in (IMG_DIR, LBL_DIR):
        sub = os.path.join(base_dir, split)
        if os.path.isdir(sub):
            shutil.rmtree(sub)
        os.makedirs(sub, exist_ok=True)

# 2. 收集所有有标注的图片
imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
              glob.glob(os.path.join(IMG_DIR, "*.png")) +
              glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
valid = []
for f in imgs:
    base = os.path.splitext(os.path.basename(f))[0]
    if os.path.exists(os.path.join(LBL_DIR, base + ".txt")):
        valid.append(os.path.basename(f))

if not valid:
    print(f"没有找到有标注的图片！请把图片放进 {IMG_DIR}，标注放进 {LBL_DIR}")
    exit(1)

# 3. 随机打乱并按比例划分
random.seed(SEED)
random.shuffle(valid)
n_val = max(1, round(len(valid) * VAL_RATIO))
n_train = len(valid) - n_val
print(f"共 {len(valid)} 张有标注图片 → train {n_train} 张 / val {n_val} 张")

# 4. 移动（图片 + 标注同步）
for split, files in [("train", valid[:n_train]), ("val", valid[n_train:])]:
    for f in files:
        base = os.path.splitext(f)[0]
        shutil.move(os.path.join(IMG_DIR, f), os.path.join(IMG_DIR, split, f))
        shutil.move(os.path.join(LBL_DIR, base + ".txt"), os.path.join(LBL_DIR, split, base + ".txt"))

# 5. 验证
print("\n划分结果：")
ok = True
for split in ("train", "val"):
    ni = len(glob.glob(os.path.join(IMG_DIR, split, "*.jpg")) +
              glob.glob(os.path.join(IMG_DIR, split, "*.png")))
    nl = len(glob.glob(os.path.join(LBL_DIR, split, "*.txt")))
    match = "OK" if ni == nl else "不匹配!"
    if ni != nl:
        ok = False
    print(f"  {split}: 图片 {ni} = 标注 {nl}  {match}")

print("\n完成! 图片与标注已按 train/val 分好（图片数必须等于标注数才算成功）")
