# JSON 标注转 YOLO txt 脚本
# 用法：把 X-AnyLabeling 导出的 JSON 标注转成 YOLO 训练的 .txt 格式
# 运行命令（在 yololearn 目录执行）：
#   venv\Scripts\python training\scripts\json_to_yolo.py
# 说明：
#   - 读取 SOURCE_DIR 里的 .json 标注文件，生成同名 .txt（YOLO 格式）
#   - 输出到 OUT_DIR（默认 SOURCE_DIR 同目录，方便图+txt 配对）
#   - 只处理 rectangle（矩形框）标注，其他形状跳过
#   - 类别按 CLASSES 列表顺序编号（0=第一个）

import os
import glob
import json

# ===== 想改就改这里 =====
SOURCE_DIR = r"C:\Users\21404\Desktop\自动标注100张"   # JSON 标注所在目录
OUT_DIR = SOURCE_DIR                                     # txt 输出目录（默认同目录）
CLASSES = ["Gotchard"]                                   # 类别列表（顺序=编号，0 开始）
# =====================

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.json")))
print(f"找到 {len(files)} 个 JSON 标注")

converted = 0
empty = 0
for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    d = json.load(open(f, encoding="utf-8"))
    shapes = d.get("shapes", [])
    lines = []
    for s in shapes:
        if s.get("shape_type") != "rectangle":
            continue  # 只处理矩形框
        label = s.get("label")
        if label not in CLASSES:
            print(f"  [!] 未知标签 '{label}' 在 {base}, 跳过")
            continue
        cls_id = CLASSES.index(label)
        pts = s.get("points", [])
        if len(pts) != 4:
            continue
        # 矩形 4 点 → (x1,y1) (x2,y2)（X-AnyLabeling 存的是 4 个角点）
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w = float(d.get("imageWidth", 1))
        h = float(d.get("imageHeight", 1))
        # 归一化到 0-1（YOLO 格式）
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        # 越界保护
        cx = max(0.0, min(cx, 1.0))
        cy = max(0.0, min(cy, 1.0))
        bw = max(0.0, min(bw, 1.0))
        bh = max(0.0, min(bh, 1.0))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        out = os.path.join(OUT_DIR, base + ".txt")
        with open(out, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
        converted += 1
    else:
        empty += 1

print(f"完成! 转换 {converted} 个有标注的, {empty} 个空(无框)")
print(f"txt 输出到: {OUT_DIR}")
