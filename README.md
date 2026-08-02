# YOLO 学习指南（新手版）

## 零、日常使用流程（先看这个！）

**检测图片**（最常用）：
```bash
cd "C:\Users\21404\Desktop\打工必备\各种作业跟项目\yololearn"
venv\Scripts\python detect.py 图片路径.jpg
```

**检测视频**：
```bash
venv\Scripts\python detect.py 视频.mp4
```

**摄像头实时检测**（0 = 第一个摄像头）：
```bash
venv\Scripts\python detect.py 0
```

**换模型**（越大越准越慢）：
```bash
venv\Scripts\python detect.py 图片.jpg --model yolo26s.pt   # 或 m/l/x
```

**调置信度**（过滤低质量检测）：
```bash
venv\Scripts\python detect.py 图片.jpg --conf 0.5
```

**实时弹窗显示**（不存文件）：
```bash
venv\Scripts\python detect.py 图片.jpg --show
```

结果自动保存到 `runs/detect/` 文件夹。所有参数汇总：

| 参数 | 作用 | 示例 |
|------|------|------|
| `--model` | 换模型 | `--model yolo26m.pt` |
| `--conf` | 置信度阈值 | `--conf 0.5` |
| `--imgsz` | 分辨率 | `--imgsz 1024` |
| `--show` | 弹窗实时显示 | `--show` |

---

## 一、环境清单（已全部配好 ✅）

| 工具 | 版本 | 说明 |
|------|------|------|
| Git | 2.54.0 | 代码版本管理 |
| Python | 3.14.2 | 系统稳定版（venv 虚拟环境隔离） |
| PyTorch | 2.11.0 + cu128 | GPU 版，支持 RTX 5060 (Blackwell) |
| OpenCV | 5.0.0 | 图像处理 |
| Ultralytics | 8.4.114 | 官方 YOLO 库，含 YOLO26 |
| CMake | 4.4.1 | C++ 编译工具 |
| GPU | RTX 5060 8GB | 推理单张图仅 ~7ms |

## 二、怎么进入学习环境

每次要用 YOLO，先激活虚拟环境：

```bash
# 在 yololearn 目录下
source venv/Scripts/activate
```

或直接用完整路径（不用激活）：
```bash
venv/Scripts/python 你的脚本.py
```

## 三、什么是 YOLO（3 分钟理解）

YOLO = **You Only Look Once**（只看一次）。
- 输入一张图片 → 一次前向推理 → 同时输出"在哪"（框）+ "是什么"（类别）
- 特点是**快**，适合实时检测（视频、摄像头）
- 你装的是官方最新 **YOLO26**，支持 6 种任务：
  1. 目标检测（框出物体）← 入门先学这个
  2. 实例分割（像素级抠出物体）
  3. 语义分割（分类每个像素）
  4. 姿态估计（人体关键点）
  5. 图像分类
  6. 旋转框检测（OBB）

## 四、学习路线图（由浅入深）

### 第 1 阶段：会用（1~2 天）
运行 `test_yolo.py`，理解 YOLO 能干什么。
```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")        # n=最小最快，适合入门
results = model.predict("bus.jpg") # 检测
results[0].save("output.jpg")      # 保存标注图
```

### 第 2 阶段：会换模型（1 天）
不同模型不同权衡：
```python
YOLO("yolo26n.pt")   # 最快，精度一般
YOLO("yolo26s.pt")   # 均衡
YOLO("yolo26m.pt")   # 精度更高，稍慢
YOLO("yolo26l.pt")   # 高精度
YOLO("yolo26x.pt")   # 最高精度，最慢
```
视频检测：`model.predict("video.mp4", save=True)`
摄像头实时：`model.predict(source=0)`（0 表示摄像头）

### 第 3 阶段：会调参数（2~3 天）
- 置信度阈值：`predict(conf=0.5)`
- 图像大小：`predict(imgsz=1024)`（越大越准越慢）
- 训练时指定轮数、批次：`model.train(data="coco8.yaml", epochs=10)`

### 第 4 阶段：会自己训练（重点，1~2 周）
1. **准备数据**：收集图片 → 用 [LabelImg](https://github.com/HumanSignal/labelImg) 或 [Roboflow](https://roboflow.com) 标注
2. **数据格式**：YOLO 格式 = 每张图配一个 `.txt`，每行 `类别 cx cy w h`（归一化坐标）
3. **组织目录**：
   ```
   dataset/
     images/train/  images/val/
     labels/train/  labels/val/
   ```
4. **训练**：
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo26s.pt")
   model.train(data="dataset.yaml", epochs=100, batch=8, imgsz=640)
   ```
5. **验证**：`model.val()` 看 mAP 指标
6. **推理**：`model.predict(source=图片, save=True)`

### 第 5 阶段：会部署（进阶）
- 导出模型：`model.export(format="onnx")` / `format="tensorrt"`
- 部署到服务器、手机、边缘设备

## 五、必须掌握的 YOLO 概念

| 概念 | 一句话解释 |
|------|-----------|
| 权重 (.pt) | 训练好的模型参数文件 |
| 预测框 bbox | `(x1, y1, x2, y2)` 四个坐标 |
| IoU | 预测框和真实框的重合度 |
| NMS 去重 | 同一物体多个框，只留最好的 |
| mAP | 平均精度，衡量模型好坏，越高越好 |
| 置信度 conf | 模型对这个框有多自信，0~1 |
| Epoch | 训练把全部数据过一遍 = 1 轮 |
| Batch size | 一次同时喂多少张图 |

## 六、常见坑

1. **CUDA 报错**：确保装的是 cu128 版 torch（你已是），别用 CPU 版
2. **显存不足**：换小模型（n/s）、减小 batch、减小 imgsz
3. **Python 版本**：别用 3.15 alpha，torch 不支持（你已用 3.14 ✅）
4. **权重下载慢**：第一次跑会自动下载，后面走缓存

## 七、权威资料

- Ultralytics 官方文档：https://docs.ultralytics.com
- YOLO26 发布说明：https://github.com/ultralytics/ultralytics/issues/24844
- YOLO 官方仓库：https://github.com/ultralytics/ultralytics
- 数据集平台 Roboflow：https://roboflow.com
