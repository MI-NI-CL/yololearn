# YOLO 通用检测工具 —— 图片 / 视频 / 摄像头 全支持
# 用法示例：
#   python detect.py 图片路径                → 检测图片，结果存 runs/detect/
#   python detect.py 视频.mp4                → 检测视频
#   python detect.py 0                       → 打开摄像头实时检测（0=第一个摄像头）
#   python detect.py 图片 --model yolo26s.pt → 换模型
#   python detect.py 图片 --conf 0.5         → 调置信度阈值
#   python detect.py 图片 --show             → 弹窗实时显示结果

import argparse
from ultralytics import YOLO
import torch

def main():
    p = argparse.ArgumentParser(description="YOLO 通用检测工具")
    p.add_argument("source", help="图片路径 / 视频路径 / 摄像头编号(0)")
    p.add_argument("--model", default="yolo26n.pt", help="模型文件")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    p.add_argument("--show", action="store_true", help="弹窗实时显示")
    p.add_argument("--save", action="store_true", default=True, help="保存结果")
    args = p.parse_args()

    # 设备选择：有 GPU 用 GPU，没有用 CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"设备: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 加载模型
    model = YOLO(args.model)

    # 推理
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        show=args.show,
        save=args.save,
        save_txt=False,
        verbose=False,
    )

    # 图片模式打印结果
    for r in results:
        if hasattr(r, "boxes") and r.boxes is not None:
            print(f"\n检测到 {len(r.boxes)} 个目标:")
            for box in r.boxes:
                name = r.names[int(box.cls)]
                conf = float(box.conf)
                print(f"  - {name}: {conf:.2f}")

    print(f"\n结果已保存到 runs/detect/ （若开启了保存）")

if __name__ == "__main__":
    main()
