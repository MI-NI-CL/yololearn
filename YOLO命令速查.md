# ⚡ YOLO 命令速查

> 纯命令速查，环境搭建见 `环境安装存档.md`。默认在 VS Code PowerShell，先 cd 到 yololearn。

```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
```

## 检测 detect
```powershell
venv\Scripts\yolo predict model=yolo26n.pt source=bus.jpg
venv\Scripts\yolo predict model=yolo26n.pt source="视频.mp4"
venv\Scripts\yolo predict model=yolo26n.pt source=0 show=True
venv\Scripts\yolo predict model=yolo26n.pt source=bus.jpg classes=0
```
## 分割 segment（描轮廓）
```powershell
venv\Scripts\yolo segment predict model=yolo11n-seg.pt source=bus.jpg
```
## 跟踪 track（目标带稳定 ID）
```powershell
venv\Scripts\yolo track predict model=yolo26n.pt source="视频.mp4"
```
## 分类 classify / 姿态 pose
```powershell
venv\Scripts\yolo classify predict model=yolo11n-cls.pt source=bus.jpg
venv\Scripts\yolo pose predict model=yolo11n-pose.pt source=bus.jpg
```
## 换模型（越大越准越慢）
```powershell
# n=5.5MB 最快 / s=20MB / m=44MB / l=53MB / x=最大最慢，自动下载
venv\Scripts\yolo predict model=yolo26m.pt source=bus.jpg
```

## C++ CPU（cpp_yolo.exe，图片/AVI视频/摄像头）
```powershell
cpp_yolo\build\cpp_yolo.exe bus.jpg yolo26n.onnx
cpp_yolo\build\cpp_yolo.exe 视频.avi yolo26n.onnx
cpp_yolo\build\cpp_yolo.exe 0 yolo26n.onnx
```

## 参数
```
model=xxx.pt  用哪个模型      source=xxx  检测什么(图/视频/0=摄像头)
classes=0,5   只要这些类别     conf=0.5    置信度阈值
show=True     摄像头弹窗      imgsz=640   输入分辨率
```
⚠️ 参数是 `名字=值`，**没有 `--`**；带空格的文件名加引号。

## 类别编号（COCO）
```
0=person  2=car  5=bus  7=truck  15=cat  16=dog  54=remote  56=chair
```
查全部 80 类：`... source=bus.jpg classes=--1`

## 常见报错
| 报错 | 解决 |
|------|------|
| `'yolo' 不是内部或外部命令` | 用 `venv\Scripts\yolo` 全路径 |
| 找不到 `venv\Scripts\python` | 先 cd 到 yololearn |
| 模型下载卡住 | 开梯子，或手动下载 .pt 放进目录 |
| 摄像头不弹窗 | 加 `show=True` |
