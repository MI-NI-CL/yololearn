# YOLO 环境验证脚本
# 运行：venv\Scripts\python test_yolo.py
from ultralytics import YOLO
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

# 加载 YOLO26 最小的模型 n 版本（首次运行会自动下载权重 ~6MB）
model = YOLO("yolo26n.pt")

# 用官方示例图做目标检测，设备自动选 GPU
results = model.predict("https://ultralytics.com/images/bus.jpg", device=0 if torch.cuda.is_available() else "cpu")

# 显示检测结果
r = results[0]
print(f"\n检测到 {len(r.boxes)} 个目标:")
for box in r.boxes:
    name = r.names[int(box.cls)]
    conf = float(box.conf)
    print(f"  - {name}: 置信度 {conf:.2f}")

# 保存标注后的图片
r.save("output.jpg")
print("\n✅ 推理成功！标注图片已保存为 output.jpg")
