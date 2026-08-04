# 📖 YOLO 学习记录

> 每学一步记一步，随时回来复习。
> 进度：👇 顶部表格一眼看清学到哪了。

---

## 🚦 进度速览

| 步骤 | 学的内容 | 状态 |
|------|---------|------|
| 第 1 步 | 进入项目目录 | ✅ 完成 |
| 第 2 步 | 官方命令检测图片 | ✅ 完成 |
| 第 3 步 | 读懂官方输出 | ✅ 完成 |
| 第 4 步 | 只检测某一类（classes） | ✅ 完成 |
| 第 5 步 | 检测视频 | ✅ 完成 |
| 第 6 步 | 摄像头实时检测 | ✅ 完成 |
| 第 7 步 | 查看类别编号（classes=--1） | ✅ 完成 |
| 第 8 步 | 换更大的模型（n/s/m/l）+ 模型自动下载 | ✅ 完成 |
| 第 9 步 | 读懂 C++ 推理代码骨架（6 功能块流水线） | ✅ 完成 |
| 第 10 步 | C++ 代码画检测框到图片（手写像素循环） | ✅ 完成 |
| 第 11 步 | C++ 检测视频并输出带框视频（引入 OpenCV） | ✅ 完成 |
| 第 12 步 | C++ 摄像头实时检测（补编译 OpenCV highgui 模块） | ✅ 完成 |
| 第 13 步 | **C++ GPU 加速（TensorRT + MSVC，7.63ms）** | ✅ 完成 |
| 第 14 步 | **C++ GPU 跟踪（track 参数，跨帧 ID 稳定）** | ✅ 完成 |
| 第 15 步 | **图像分割（segment，描出物体轮廓）** | ✅ 完成 |
| 第 16 步 | **图像分类（classify，整图归为某类）** | ✅ 完成 |
| 第 17 步 | **姿态估计（pose，人体关键点骨架）** | ✅ 完成 |
| 第 18 步 | **定向检测（obb，旋转框，航拍场景）** | 📖 了解即可 |
| 第 19 步 | **训练入门（train，coco8 迷你数据集跑通流程）** | ✅ 完成 |
| 第 20 步 | **迭代训练 + 自动标定（自己数据的完整流程）** | 📖 已学流程 |

---

## ✅ 第 1 步：进入项目目录

**做什么**：让终端"站"在 yololearn 目录里，命令才找得到文件。

**怎么做**：VS Code 打开 `yololearn` 文件夹即可，终端自动进入。

**确认方法**：终端提示符结尾是 `yololearn>`，前面有 `(venv)`。

---

## ✅ 第 2 步：官方命令检测图片

> ⭐ **这是最核心的一条命令，务必记住**

```bash
yolo predict model=yolo26n.pt source=bus.jpg
```

**命令拆解**（每个词是什么）：

| 部分 | 含义 |
|------|------|
| `yolo` | 官方命令（激活 venv 后可直接用） |
| `predict` | 动作 = 推理 / 检测 |
| `model=yolo26n.pt` | 用哪个模型 |
| `source=bus.jpg` | 检测哪张图 |

**关键输出**（看这里）：
```
image 1/1 ...bus.jpg: 640x480 4 persons, 1 bus, 6.5ms
                          └──── 检测到 4人+1公交 ────┘
```
**结果存哪**：`runs\detect\predict-数字\`（每次自动新编号）

### 🔑 万能格式（记住这个框架）
```
yolo <任务> <动作> model=... source=... 参数=值
```
> ⚠️ 官方参数是 `名字=值` 写法，**没有 `--`**

---

## ✅ 第 3 步：读懂官方输出

| 输出 | 含义 |
|------|------|
| `CUDA:0 (RTX 5060)` | 用的是你的**显卡 GPU**，不是 CPU |
| `122 layers, 2.4M parameters` | 模型档案（新手不用深究） |
| `4 persons, 1 bus, 6.5ms` | **检测结果 + 推理耗时**（重点） |
| `Speed: ... 6.5ms inference ...` | 三阶段耗时，**看 inference** |

---

## ✅ 第 4 步：只检测某一类

**做什么**：让模型只框出**指定类别**的东西（比如只框人，不框公交车）。

**怎么做**：
```bash
yolo predict model=yolo26n.pt source=bus.jpg classes=0
```

**命令拆解**：
| 部分 | 含义 |
|------|------|
| `classes=0` | 只检测编号 0 的类别（0 = person 人） |

**关键输出**：原本的 `4 persons, 1 bus` 变成 `4 persons`（只框人，bus 被过滤掉了）

### 🔑 核心知识点：类别编号
- 模型认识 80 类（COCO），每类一个编号
- `classes=0` = 只检测 person；想多类用逗号：`classes=0,2,5`
- **怎么查某类的编号**：`yolo predict model=yolo26n.pt source=bus.jpg classes=--1`（无效编号会列出全部类别及其编号）或用 `python -c "from ultralytics import YOLO; print(YOLO('yolo26n.pt').names)"`

---

## ✅ 第 5 步：检测视频

**做什么**：让 YOLO 自动检测视频里每一帧的物体，生成带框的视频。

**怎么做**（视频文件名带空格时必须加**双引号**）：
```bash
yolo predict model=yolo26n.pt source="你的视频文件名.mp4"
```

**命令拆解**：
| 部分 | 含义 |
|------|------|
| `source="..."` | 检测这个视频（引号包住带空格的名字） |

### 🔑 关键坑：带空格的文件名要加引号
- `屏幕录制 2026-07-02.mp4` 有空格，不加引号系统会把名字拆开报错
- **引号 = "这是一个整体，别拆开"**

### 🔑 关键知识点：置信度阈值 conf
- 模型对每个框给置信度（0~1），默认只显示 ≥ **0.25** 的
- 调低 `conf=0.1` → 框更多但**误检变多**
- 调高 `conf=0.5` → 框更少但更准
- **这是精度与误检的权衡**，没有"最好"的值，看场景调

---

## ✅ 第 6 步：摄像头实时检测

**做什么**：让 YOLO 盯着摄像头实时画面检测，像"会看东西的眼睛"。

**怎么做**（`show=True` 是关键！）：
```bash
yolo predict model=yolo26n.pt source=0 show=True
```

**命令拆解**：
| 部分 | 含义 |
|------|------|
| `source=0` | 用摄像头 0（第一个摄像头） |
| `show=True` | **实时弹出画面窗口**（关键！不加这个不弹窗） |

**退出方式**：在弹窗上按 `q` 键，或 `Ctrl + C`

### 🔑 关键坑：show 默认是 False！
- ultralytics 的 `show` 参数默认值是 **False**（不弹窗）
- 只写 `source=0` 时摄像头会工作、会输出结果，但**画面不弹出来**
- 想要实时画面必须显式加 `show=True`
- 多摄像头：第一个没画面就试 `source=1`

### 📌 读懂摄像头输出
```
0: 480x640 1 person, 8.7ms
```
- 摄像头 0 号、画面 640x480、检测到 1 人、推理 8.7ms
- `WARNING Waiting for stream 0` = 推理太快在等摄像头下一帧（**不是错误**，是好事）

---

## ✅ 第 7 步：查看类别编号

**做什么**：让 YOLO 打印出它认识的**全部 80 个类别**和对应**编号**，以后想"只检测猫/狗/车"时查表填 `classes=几`。

**怎么做**（给 `classes` 一个无效编号 `--1`，YOLO 就会列出全部类别）：
```bash
yolo predict model=yolo26n.pt source=bus.jpg classes=--1
```

**命令拆解**：
| 部分 | 含义 |
|------|------|
| `classes=--1` | **无效编号**（-1 不存在），触发 YOLO 打印全部类别 |

**关键输出**：
```
The argument 'classes' must be a list of integers, not:
0: person, 1: bicycle, 2: car, 3: motorcycle, ...
15: cat, 16: dog, ...
78: hair drier, 79: toothbrush
```

### 🔑 核心知识点：80 个类别编号（COCO）
- 模型认识 **80 类**，每类一个**固定编号**（COCO 数据集定义）
- 常用几个记住即可：
  - `0` person（人）、`1` bicycle、`2` car、`5` bus
  - `15` cat（猫）、`16` dog（狗）、`7` truck
- 想只检测某几类：`classes=0,15`（检测人和猫）

---

## ✅ 第 8 步：换更大的模型（n/s/m/l）

**做什么**：用更大的模型检测同一张图，体验"越大越准"。

**怎么做**：
```bash
yolo predict model=yolo26s.pt source=bus.jpg   # 换成 s/m/l 对比
```

**命令拆解**：
| 部分 | 含义 |
|------|------|
| `model=yolo26s.pt` | 换模型文件（n/s/m/l/x） |

### 🔑 核心知识点 1：模型大小与精度的权衡
| 型号 | 文件大小 | 速度 | 精度 | 适用 |
|------|---------|------|------|------|
| `n` nano | 5.3MB | 最快 | 一般 | 手机/边缘设备 |
| `s` small | 20MB | 快 | 更好 | **日常推荐** |
| `m` medium | 43MB | 中 | 更好 | 精度优先 |
| `l` large | 51MB | 慢 | 最好 | 追求极致 |
| `x` xlarge | 更大 | 最慢 | 最好 | 最强精度 |

> 越大越准但越慢，看场景选。用户实测：n→s→m→l 依次更准 ✅

### 🔑 核心知识点 2：模型文件是自动下载的
- 本地没有的模型，YOLO 自动从 GitHub 下载（显示进度条）
- **下载一次永久使用**：.pt 文件存到当前目录，下次直接读本地
- 换个目录跑会重新下载到新目录

---

## ✅ 第 9 步：读懂 C++ 推理代码骨架（`cpp_yolo\yolo_infer.cpp`）

**做什么**：看懂之前写好的 C++ 推理程序（181 行），掌握部署骨架。

### 🔑 6 功能块流水线（所有 C++ 部署程序都是这个骨架）
```
读图片 → 预处理 → 建引擎 → 推理 → 解析结果 → 打印
  [1]      [2]      [3]     [4]      [5]      [6]
```

| 块 | 行号 | 函数/关键 | 在干嘛 |
|----|------|----------|--------|
| 1 读图片 | 29-40 | `load_image` + stb_image | 把 jpg/png 读成像素数组 |
| 2 预处理 | 49-75 | `preprocess` | 缩放到 640x640（黑色填充），记录 pad/scale |
| 3 建引擎 | 99-126 | `Ort::Session` | 加载 .onnx 模型，问输入输出名，选 GPU/CPU |
| 4 推理 | 131-148 | `session.Run()` | 喂数据给模型，跑出结果 |
| 5 解析 | 150-170 | 循环读 300×6 | 输出翻译成框：丢弃低置信度 + 坐标换算回原图 |
| 6 打印 | 172-178 | 输出 | 打印 class/conf/box |

**关键点**：
- 模型输入固定 640x640，任意图都要先缩放填充
- `pad_x/pad_y/scale` 记录缩放信息，解析结果时用来还原坐标
- 输出格式 `[1,300,6]`：300 个候选框 × 每框 6 个数
  `[x1,y1,x2,y2,置信度,类别编号]`
- 改功能就改第 6 块（打印→画框/存视频/发网络）

---

## ✅ 第 10 步：C++ 代码把检测框画到图片上

**做什么**：改 `yolo_infer.cpp`，在"打印结果"后加一段代码，把检测框画到原图并保存。

**怎么做**：
1. 下载 `stb_image_write.h`（保存 JPG 用）到 `cpp_yolo\` 目录
2. 文件顶部加：`#define STB_IMAGE_WRITE_IMPLEMENTATION` + `#include "stb_image_write.h"`
3. 在 `return 0;` 前加画框代码：遍历每个检测框的 4 条边像素，染成红色 (255,0,0)
4. 用 `stbi_write_jpg("output_cpp.jpg", ...)` 保存
5. 重新编译 + 运行：`cpp_yolo/build/cpp_yolo.exe bus.jpg yolo26n.onnx cpu`

**关键输出**：`带框图片已保存: output_cpp.jpg`（检测目标被红框标出，用户确认 OK ✅）

### 🔑 核心知识点 1：手写画框 vs OpenCV
- 手写循环：遍历框边像素改 RGB，微秒级，性能足够
- OpenCV 的 `cv::line`：内部同样是遍历像素，性能几乎一样
- **画框不是性能瓶颈**（推理才是）；OpenCV 的价值在画文字、处理视频流等功能
- 决策：现在零依赖手写够用；以后做视频/文字再引 OpenCV

### 🔑 核心知识点 2：C++ 程序怎么"跑起来"（重要架构）
一个 C++ 检测程序**不是"一个文件"**，而是由 4 部分构成：
```
cpp_yolo.exe          ← 编译产物（你的代码被编译成的可执行文件）
onnxruntime.dll       ← onnxruntime 库（负责真正跑神经网络的"引擎"）
yolo26n.onnx          ← 模型（YOLO 的"大脑"，9.5MB）
图片输入              ← 你要检测的图
```
- **编译时**：`.cpp 代码 + onnxruntime 头文件` → 编译链接 → `exe`
- **运行时**：`exe` 启动后去旁边找 `onnxruntime.dll` 加载，再读 `yolo26n.onnx` 模型，然后才干活
- 三个文件缺一不可，运行时必须在同目录能找到 DLL

---

## ✅ 第 11 步：C++ 检测视频并输出带框视频

**做什么**：让 C++ 程序逐帧读取视频、检测、画框，输出带框视频。

**为什么**：真实应用（监控、自动驾驶）处理的是视频流，不是单张图。

### 🔑 核心：视频 = 逐帧图片循环
```
读一帧 → 检测画框 → 写入输出视频 → 下一帧（循环）
```
一个 `while(cap.read(frame))` 循环搞定，逻辑和第 10 步画框一样。

### 引入 OpenCV C++（视频读写必须靠它）
- **为什么必须用 OpenCV**：视频编码/解码（H.264 等）极其复杂，不能手写，必须用库
- **为什么画框可以手写**：画框只是改像素，简单；视频编解码是"硬骨头"
- 之前 pip 装的 OpenCV 只有 Python 接口，**C++ 版需要单独编译**

### 从源码编译 OpenCV 5.0.0（MinGW）
- **中文路径的坑**：源码放 `C:\Users\...\打工必备\...` 会因 NASM 乱码失败 → **必须放纯英文路径**（`C:\opencv_build\src`）
- **只编核心模块**：`BUILD_LIST=core,imgproc,imgcodecs,videoio,calib3d` → 700 目标，约 3-4 分钟
- **静态库**：`BUILD_SHARED_LIBS=OFF`，输出 `.a` 文件
- 安装位置：`C:\opencv_build\install`（include + staticlib）

### OpenCV 5.0 特有的坑（版本相关）
- 相机标定代码在 **`libopencv_geometry500.a`**（不是 calib3d！OpenCV 5.0 重组了模块）
- 静态库链接要用 `-Wl,--start-group ... --end-group` 循环解析依赖
- 要链 `liblibclapack.a`（LAPACK 线性代数）

### 🔑 视频格式限制（MinGW 编译的 OpenCV）
- 关了 FFmpeg → **读不了 mp4（H.264 编码）**
- **能用 AVI + MJPG 编码**的视频
- 这是 MinGW 版 OpenCV 的已知限制；想要 mp4 需重新编译加 FFmpeg

### 最终成果
```bash
cpp_yolo/build/cpp_yolo.exe test_video.avi yolo26n.onnx cpu
```
→ 生成 `output_video.avi`（50 帧逐帧检测 + 红框，用户确认 OK ✅）

---

## ✅ 第 12 步：C++ 摄像头实时检测

**做什么**：用 C++ 程序打开摄像头，实时检测画面并弹窗显示。

**怎么做**：
```bash
cpp_yolo\build\cpp_yolo.exe 0 yolo26n.onnx cpu
```
（`0` = 摄像头编号，`cpu` = CPU 推理）

**关键输出**：弹出窗口显示摄像头画面 + 实时红框检测（用户确认成功 ✅）
**退出**：按 `q` 或 `Esc`

### 🔑 前置：补编译 OpenCV 的 highgui（窗口显示）模块
- 第 11 步编译 OpenCV 时**漏了 highgui**（只管视频读写，没管显示）
- 摄像头实时需要 `cv::imshow`（显示窗口）→ 必须补 highgui
- 重新配置加：`-DBUILD_LIST=... ,highgui`
- 配置输出确认：`highgui: using builtin backend: WIN32UI`（Windows 原生窗口）
- 安装后：`libopencv_highgui500.a` + `include/opencv2/highgui.hpp`

### 🔑 代码要点：视频/摄像头共用一套逐帧循环
- 输入 `0` 或 `1`（纯数字）= 摄像头；`.mp4/.avi` = 视频文件
- `is_camera()`：判断输入是不是纯数字（摄像头编号）
- 摄像头模式：`cv::imshow` 显示 + `cv::waitKey(1)` 刷新窗口，**不写文件**
- 视频模式：`cv::VideoWriter` 写文件，**不弹窗**
- 退出：`cv::waitKey` 返回 `q`(113) 或 Esc(27) 时 break

### ⚠️ 踩坑记录
- 摄像头模式**不该创建 writer**（视频输出文件），否则 fps=-1 报错 `无法创建输出视频`
- 修复：writer 只在 `!is_cam` 时创建
- 摄像头 fps 读取常为 -1，需兜底（`if (fps<1) fps=10`）

### 🔑 帧数低的原因与优化方向
- CPU 推理每帧 ~20ms + 显示开销 → 25-30 帧/秒
- 优化三招：GPU 推理（`gpu` 参数）/ 缩小 imgsz（320）/ 跳帧检测

---

*记录方式：每教一步 → 你亲手敲 → 成功 → 记入本档 ✅*

---

## ✅ 第 13 步：C++ GPU 加速（TensorRT + MSVC）

**目标**：让 C++ 程序用 GPU 推理（从 CPU 18ms 提速）。

**最终结果**：**C++ TensorRT GPU 推理 7.63ms**（成功！）

### 🔑 为什么之前的方案都不行（重要经验）
| 方案 | 结果 | 原因 |
|------|------|------|
| onnxruntime CUDA EP | ❌ 不支持 | **不支持 RTX 5060 (Blackwell sm_120)**，内核不含新架构 |
| MinGW 编译 TensorRT | ❌ ABI 崩 | MinGW 与 MSVC 编译的 TensorRT 库堆不兼容 |
| PyTorch C++ (LibTorch) | ❌ 头文件冲突 | PyTorch C++ 头文件 MinGW 下 dllimport 冲突 |
| **MSVC + TensorRT** | ✅ **7.63ms** | 官方支持路径 |

### 🔑 关键结论
- **onnxruntime 的 CUDA 加速器还不支持 RTX 5060**（新显卡），PyTorch 和 TensorRT 支持
- **C++ GPU 的正确路径**：**TensorRT + MSVC 编译**（不能 MinGW）
- TensorRT 需装完整 SDK（含 NvInfer.h + nvinfer_11.lib），要 NVIDIA 账号

### 🔑 环境配置
- CUDA Toolkit 13.3（官方安装，`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`）
- TensorRT 11.2.1 SDK（解压到 `C:\cpp_msvc_build\tensorrt`）
- 用 MSVC 编译（vcvars64 + cl + link），链接 `nvinfer_11.lib` + `cudart.lib`

### 🔑 性能对比
| 方案 | 耗时 |
|------|------|
| **C++ TensorRT (MSVC)** | **7.63ms** |
| Python TensorRT | 10ms |
| Python PyTorch GPU | 10.85ms |
| C++ CPU | 18ms |

### 🔑 程序位置
- 源码：`C:\cpp_msvc_build\tensorrt_yolo.cpp`
- MSVC 版 exe：`C:\cpp_msvc_build\tensorrt_yolo_msvc.exe`
- 运行：`tensorrt_yolo_msvc.exe yolo26n_sdk.engine 图片.jpg`

---

## ✅ 第 14 步：C++ GPU 跟踪（track 参数，跨帧 ID 稳定）

**目标**：让 C++ GPU 程序在视频/摄像头里**持续跟踪每个目标**（同一人跨帧保持同一编号，而不是每帧重新认人）。

**最终结果**：`tensorrt_yolo_msvc.exe` 第 5 个参数写 `track` 即开启跟踪 ✅

### 🔑 为什么需要跟踪
普通检测（detect）**每帧独立认人**：人一闪、被挡住，下一帧就被当成新的人。
跟踪（track）给每个目标分配**稳定 ID**，跨帧记住"谁是谁"，适合数人流、车辆追踪。

### 🔑 怎么做（命令）
```powershell
cd C:\cpp_msvc_build
# 先加 PATH（DLL 路径）后：
tensorrt_yolo_msvc.exe yolo26n_sdk.engine test.mp4 0.25 0 track
```
参数：`<engine> <输入> [conf] [classes] [track]`
- 第 5 个参数 `track` = 开启跟踪（仅视频/摄像头，图片不支持）
- `0` = 只跟踪 person（画面里两个人 → `person 1` / `person 2`）

**应该看到什么**：框上显示 `person 编号`，不同的人颜色不同；人在画面里走动编号不变。

### 🔑 实现原理（轻量 IoU 跟踪器，约 120 行已编进 exe）
- 每帧检测完，用**框重叠度（IoU）**把本帧的框和上一帧的跟踪目标做匹配
- 匹配上的 → 沿用旧 ID；没匹配上 → 分配新 ID；丢失超 10 帧 → 移除
- 优点：零依赖、已编进 exe、不用重训；局限：重遮挡时可能短暂"换号"

### 🔑 实测验证
- 检测模式回归：`bus.jpg` 正常检出 4 个 person
- 跟踪模式：60 帧测试视频里 4 个人全程保持 ID 1-4 不变 ✅
- **用户亲手实践确认**：在自己的视频/摄像头上跑通跟踪，ID 跨帧稳定 ✅
- ⭐ **实践中学到的点**：
  - 用 `0` 参数（只跟踪 person）可以过滤掉不关心的物体（如画面里被误认成 `remote` 遥控器的手柄/杂物）
  - 看到不认识的名字（如 `remote`）→ 查类别编号表即可（remote = 编号 54 遥控器）
  - 调高 conf（如 `0.25`→`0.5`）能减少误检，只留把握大的目标

### 🔑 更强的方案（Python 官方）
要更稳的跟踪（ByteTrack/BoTSORT），用 ultralytics Python API（结果存 `runs/track/`）：
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\python -c "from ultralytics import YOLO; YOLO('yolo26m.pt').track(source='test.mp4', classes=[0], conf=0.25, persist=True, tracker='bytetrack.yaml', save=True)"
```

### 🔑 换量级（C++ GPU 侧也支持）
4 个量级的 engine 已全部构建好（2026-08-03）：
| engine 文件 | 大小 | 实测 bus.jpg 结果 |
|-------------|------|------------------|
| `yolo26n_sdk.engine` | 11MB | 5 个目标（bus conf 0.92） |
| `yolo26s_sdk.engine` | 39MB | 5 个目标（更准） |
| `yolo26m_sdk.engine` | 85MB | 5 个目标（bus conf 0.96） |
| `yolo26l_sdk.engine` | 103MB | **6 个目标**（识别最准） |

> 换量级 = 换命令里的 `<engine>` 文件名，其余参数不变：
> `tensorrt_yolo_msvc.exe yolo26l_sdk.engine 视频.mp4 0.25 0 track`
> 实测：**模型越大越准**（l 检出 n 漏掉的第 6 个目标），代价是更慢。

---

## ✅ 第 15 步：图像分割（segment，描出物体轮廓）

**目标**：检测物体时把**轮廓也描出来**——不只是方框，而是物体边缘的精确形状（人形轮廓线）。

**最终结果**：`segment` 任务跑通，图片上人和公交车被彩色轮廓描出来 ✅

### 🔑 为什么值得学
- 检测（detect）= 给"框住物体的矩形"
- 分割（segment）= 给"物体精确的形状"（每个像素属于哪个物体）
- 用途：抠图、测物体面积/大小、精细区域分析

### 🔑 怎么做（命令）
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\yolo segment predict model=yolo11n-seg.pt source=bus.jpg
```
**命令拆解**：
| 部分 | 含义 |
|------|------|
| `segment` | 任务 = 分割（不是 detect） |
| `model=yolo11n-seg.pt` | 分割专用模型（**自动下载**，约 5MB） |

### 🔑 核心知识点
1. **`-seg` 结尾 = 分割专用模型**：`yolo11n-seg.pt`，和检测模型不同，自动下载一次永久使用
2. **万能格式通用**：`yolo <任务> <动作> model=... source=...` 框架不变，detect → segment 只换任务名和模型名
3. **输出不同**：每行多了轮廓坐标（mask 数据），保存图上是彩色轮廓而非简单红框
4. 结果存 `runs\segment\predict-数字\`（和 detect 分开）

**应该看到什么**：人和公交车被**彩色轮廓描边**，不再是简单矩形框。

---

## ✅ 第 16 步：图像分类（classify，整图归为某类）

**目标**：判断**整张图属于什么类别**——不是"图里有啥"，而是"这张图整体是什么"。

**最终结果**：`classify` 任务跑通，能输出类别置信度排名 ✅（用户实践确认）

### 🔑 怎么做（命令）
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\yolo classify predict model=yolo11n-cls.pt source=bus.jpg
```
**命令拆解**：`classify` 任务名在 `predict` 前；`yolo11n-cls.pt` 分类专用模型（自动下载）。

### 🔑 核心知识点
1. **`cls` = classification（分类）的缩写**：官方统一后缀 `-cls`（分割是 `-seg`）。是惯例缩写（取主体辅音），不是拼写错
2. **三个任务对比**：
   | 任务 | 模型后缀 | 回答的问题 |
   |------|---------|-----------|
   | detect | 无（yolo26n） | 图里有**哪些**东西、在哪 |
   | segment | `-seg` | 东西的**精确轮廓** |
   | classify | `-cls` | 整张图**是什么** |
3. **⚠️ 不能去掉 `-cls`**：检测模型和分类模型**输出格式完全不同**，用错模型会报错或结果乱。`-cls` 是分类模型的标志，必须保留
4. **分类模型识别的类别 ≠ 检测模型**：是 ImageNet 1000 类（不是 COCO 80 类）

### 🔑 实践中学到的点（用户反馈）
- ⭐ **场景图分类效果差是正常的**：`bus.jpg` 是街道场景图（多物体），而分类模型擅长**单一主体**图（如一只猫）。场景图"是什么"很模糊，置信度自然低
- ⭐ **想让分类准**：① 换更大的模型 `yolo11s-cls.pt`（>n）② 换单一主体图片（官方示例 `source=https://ultralytics.com/images/cat.jpg`）
- 用户确认：效果差但能跑 ✅（原因已讲清，非故障）

---

## ✅ 第 17 步：姿态估计（pose，人体关键点骨架）

**目标**：检测人体**关键点**（头、肩、肘、腕、膝、踝等 17 个点），连线成人形骨架。

**最终结果**：`pose` 任务跑通，图片上行人身上显示骨架线条 ✅（用户实践确认）

### 🔑 怎么做（命令）
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\yolo pose predict model=yolo11n-pose.pt source=bus.jpg
```
**命令拆解**：`pose` 任务名在 `predict` 前；`yolo11n-pose.pt` 姿态专用模型（自动下载）。

### 🔑 核心知识点
1. **pose 输出**：每个人 17 个关键点（坐标），连线成骨架，能反映人摆了什么姿势
2. 用途：人机交互、健身动作分析、动作识别
3. **四大任务框架齐了**：
   | 任务 | 模型后缀 | 回答的问题 |
   |------|---------|-----------|
   | detect | 无（yolo26n） | 有啥、在哪 |
   | segment | `-seg` | 精确轮廓 |
   | classify | `-cls` | 整图是什么 |
   | pose | `-pose` | 人的关节姿态 |

### 🔑 实践中学到的点（track 简写规律）
- ⭐ **track 是唯一能简写的任务**：`yolo track model=...`（官方简写，隐含 predict）等价于 `yolo track predict model=...`
- **原因**：track 既是"任务"也是"动作"（隐含预测+跟踪），其他任务必须写 `任务名 predict`
- **各任务写法**：
  ```
  yolo predict model=...            ← detect 可省（默认任务）
  yolo segment predict model=...    ← 必须写 segment
  yolo classify predict model=...   ← 必须写 classify
  yolo pose predict model=...       ← 必须写 pose
  yolo track model=...              ← track 可简写
  ```

---

## 📖 第 18 步：定向检测（obb，旋转框，航拍场景）

> 状态：**了解即可**（用户判断：日常用不上，知道概念就好）

**做什么**：检测物体用**带角度的旋转框**——贴合物体朝向的斜框，而非横平竖直的矩形。

**命令**：
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\yolo obb predict model=yolo11n-obb.pt source=bus.jpg
```

### 🔑 核心知识点
1. **obb = Oriented Bounding Box**（定向检测框）
2. **模型认 15 类，全是航拍/卫星视角**（DOTA 数据集训练）：
   plane飞机 / ship轮船 / 储油罐 / 棒球场 / 网球场 / 篮球场 / 田径场 / 港口 / 桥 / 大车 / 小车 / 直升机 / 环岛 / 足球场 / 游泳池
3. **为什么测 bus.jpg 只有飞机**：模型是"卫星图专家"，普通地面视角照片识别不出航拍目标，属正常（场景不匹配，非故障）
4. **想测它的效果**：用地图软件截图俯视图（停车场/球场/公路），旋转框效果最直观
5. **用不上很正常**：日常检测用 detect 就够；obb 是遥感/航拍专用场景

### 🔑 五大任务类型全集（齐了）
| 任务 | 模型后缀 | 回答的问题 | 用途 |
|------|---------|-----------|------|
| detect | 无（yolo26n） | 有啥、在哪 | 日常检测 |
| segment | `-seg` | 精确轮廓 | 抠图/测面积 |
| classify | `-cls` | 整图是什么 | 图片归类 |
| pose | `-pose` | 人的关节姿态 | 动作分析 |
| obb | `-obb` | 带角度的旋转框 | 航拍/遥感 |

---

## ✅ 第 19 步：训练入门（train，coco8 迷你数据集跑通流程）

**目标**：用官方迷你数据集验证"训练"这套流程能走通（还没到用自己的素材，先学会"怎么训练"）。

**最终结果**：coco8 训练 3 轮完成，生成 `runs\detect\train\weights\best.pt` ✅（用户实践确认）

### 🔑 怎么做（命令）
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\yolo train model=yolo11n.pt data=coco8.yaml epochs=3 imgsz=640
```

### 🔑 命令拆解
| 部分 | 含义 |
|------|------|
| `train` | 动作 = 训练（不是 predict） |
| `model=yolo11n.pt` | 从预训练模型开始（**迁移学习**，省时间） |
| `data=coco8.yaml` | 训练数据（官方迷你集，8 张图，自动下载） |
| `epochs=3` | 训练轮数（先验证流程，只跑 3 轮） |
| `imgsz=640` | 图片分辨率 |

### 🔑 核心知识点
1. **训练的本质**：给 YOLO 看"图片 + 标注"（这张图哪里有个物体），反复学习后掌握"物体长这样"。像教小孩认物
2. **迁移学习**：从别人训练好的 `yolo11n.pt` 开始继续训练，比自己从零学快得多、省数据
3. **coco8** = 官方迷你数据集（8 张图），专门用来**快速验证训练流程**，不是正式训练
4. **训练成果**：最好的模型在 `runs\detect\train\weights\best.pt`（每次训练都放这）
5. **训练产物**：`datasets\coco8`（数据集）、`runs\detect\train\`（结果：best.pt/last.pt/results.png/args.yaml）

### 🔑 怎么读训练输出（终端那堆内容里看 2 个）
| 输出 | 看什么 |
|------|--------|
| 每轮进度表的 `box_loss` 等 | **loss 是否在下降**（在降 = 模型在学） |
| 最后的 `mAP50` | 模型准不准（0~1，越高越好） |

> 其余（WARNING、wandb、版本信息等）全是无关提醒，忽略即可。

### 🔑 实测数据（用户训练 3 轮）
| 轮次 | mAP50 |
|------|-------|
| 第 1 轮 | 0.80 |
| 第 2 轮 | 0.80 |
| 第 3 轮 | 0.84 |
> mAP50 从 0.80 升到 0.84，说明模型在学到东西。3 轮太少（验证用），正式训练通常 50-300 轮。

### 📌 下一步预告
- 以后有了**自己的照片素材**：准备"图片 + 标注" → 写数据集配置 yaml → `yolo train model=... data=my.yaml` → 得到自己的 best.pt
- 训练好的模型可继续走「导出部署」链路（见环境安装存档.md 的"转格式部署"章节）

---

## 📖 第 20 步：迭代训练 + 自动标定（自己数据的完整流程）

> 状态：**已实操进行中**。核心思路：先手动标一小批 → 训个"种子模型" → 用它自动标剩下 → 每轮人工修正 → 再训更好的（业界标准做法）。

### 🔑 训练工作区（yololearn\training\）
已建好的目录结构（大文件已 gitignore，只提交脚本）：
```
training\
├── scripts\      → 拆帧等脚本（split_frames.py 已移入）
├── raw_videos\   → 拍好的视频放这
├── frames\       → 拆帧出的图片
├── annotate\     → 标定集（手动标注的图）
├── datasets\     → 训练集（dataset1\images{train,val} + labels{train,val} 已建好）
├── runs\         → 每次训练的输出（train1/train2…自动生成）
└── models\       → 训练好的 best.pt 备份（对应数据集命名）
```

### 🔑 完整流程（8 步）

**① 拍视频**：对目标物体变换角度/灯光/遮挡程度拍摄，覆盖各种情况 → 放 `training\raw_videos\`

**② 拆帧**（用脚本，自动间隔取帧避免帧太像）：
```powershell
cd "C:\Users\21404\Desktop\gowork\projects\yololearn"
venv\Scripts\python training\scripts\split_frames.py
```
> 视频放 `training\raw_videos\素材.mp4`，脚本自动把每 5 帧取 1 张存到 `training\frames\`（脚本里 `EVERY_N=5` 可调）

**③ 手动标 30 张**（种子数据）：用 **X-AnyLabeling**（免费，带 AI 辅助）或 LabelImg，框出物体 + 命名类别 → 存 `training\annotate\`

**④ 整理成数据集**：把 30 张图和标注按 8:2 分成 train/val，分别放 `training\datasets\dataset1\{images,labels}\{train,val}`

**⑤ 写数据集配置** `training\datasets\dataset1\data.yaml`：
```yaml
path: ../datasets/dataset1
train: images/train
val: images/val
names:
  0: 你的类别名
```

**⑥ 训练种子模型**：
```powershell
venv\Scripts\yolo train model=yolo11n.pt data=training/datasets/dataset1/data.yaml epochs=100 imgsz=640
```

**⑦ 用 best.pt 自动标剩余**（关键一步）——已封装成脚本：
```powershell
venv\Scripts\python training\scripts\auto_annotate.py
```
> - 脚本调用 ultralytics **官方 `auto_annotate`**（官方没有 yolo annotate 命令行，只有 Python API，故封装成脚本）
> - 对 `training\frames\` 所有图自动打框，结果存 `training\annotate\auto\`
> - 想调参数：改脚本顶部 SOURCE_DIR / DET_MODEL / OUT_DIR / CONF

**⑧ 抽查修正 → 再训练（循环）**：自动标完的图**必须人工抽查修正**标错的框（避免错误滚雪球），加入训练集再训练一轮 → 模型越滚越准

### 🔑 本次实操记录（2026-08-05）
- ✅ **项目已迁移到纯英文路径** `C:\Users\21404\Desktop\gowork\projects\yololearn`（解决中文路径导致 YOLO 训练报错的问题；旧中文路径的坑详见第 9 步记录）
- ✅ **手动标了 26 张** Gotchard（比计划的 30 张少，够种子模型用）
- ✅ **划分数据集**：21 train + 5 val，图片/标注 1:1 配对
- ✅ **训练种子模型**：`yolo26m.pt` + dataset1，50 轮，GPU 训练（CUDA:0 RTX 5060）
  - 最终 **mAP50 = 0.502**（数据仅 21 张，效果一般属正常；正是要继续自动标定扩充数据的原因）
  - 显存峰值显示 9G 是"含共享显存的读数"，8G 物理显存训练正常跑完（想更稳可加 `batch=8`）
- ✅ **换用 X-AnyLabeling v4.0.0（CVHub520 分支）** 做标注：
  - 原版 AnyLabeling v0.4.36 读不了 YOLO26 的 `[1,300,6]` onnx；X 版 v4.0.0 原生支持 YOLO26
  - **关键坑**：yaml 的 `type` 必须写 `yolo26`（不是 yolov5），字段用 `conf_threshold`/`max_det`
  - 关掉 `auto_switch_to_edit_mode`（设置里或 `.xanylabelingrc` 改 false）→ 画完框工具保持，不用每张重按 R
- ✅ **用 X 版 + best.onnx 自动标注 + 手动修正**，标出 **100 张**（97 张有框 + 3 张空）
- ✅ **建了 `json_to_yolo.py` 脚本**（把 X 版 JSON 标注转成 YOLO txt，绕开软件导出弹窗问题）
- ✅ **数据集扩充到 123 张**（原 26 + 新 97），重新划分：**train 98 + val 25**
- ⏳ 正在二次训练（123 张，epochs=100）

### 🔑 关键要点
- **适用前提**：你要识别的物体**不是** COCO 80 类（人/车/猫/狗…）时，才需要走"种子模型"路线；如果是 80 类，直接用官方模型自动标更快
- ⭐ **错误会滚雪球**：每轮自动标注后必须人工抽查修正，不能直接全信
- **训练用 GPU**：Python YOLO 侧一直是 GPU（PyTorch cu128），训练自动用 GPU（device 空 = 自动选）
- **8GB 显存**：m 模型 + imgsz640 + batch16 接近上限，遇 OOM 优先 `batch=8`
- **`yolo annotate` 命令行不存在**：自动标注官方只有 Python API（auto_annotate），已封装成脚本

---

*记录方式：每教一步 → 你亲手敲 → 成功 → 记入本档 ✅*
