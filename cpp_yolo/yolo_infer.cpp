// YOLO26 ONNX 推理（C++ / onnxruntime）
// 用法: cpp_yolo.exe <图片路径> [模型路径] [gpu|cpu]
//
// 模型: yolo26n.onnx
//   输入: images [1,3,640,640]   float32  范围[0,1]
//   输出: output0 [1,300,6]      float32  每行 = [x1,y1,x2,y2,conf,class_id]（已含 NMS）

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// stb_image: 单头文件图片解码库（JPEG/PNG/BMP/GIF 等），零依赖
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// OpenCV: 视频读写（VideoCapture/VideoWriter）
#include <opencv2/opencv.hpp>


// ---------------- 图片结构 ----------------
struct MyImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;  // HxWx3, RGB 顺序
};

// 用 stb_image 加载图片（自动支持 jpg/png/bmp）
bool load_image(const std::string& path, MyImage& img) {
    int n = 0;
    unsigned char* data = stbi_load(path.c_str(), &img.width, &img.height, &n, 3);
    if (!data) {
        std::cerr << "stbi_load 失败: " << stbi_failure_reason() << std::endl;
        return false;
    }
    img.rgb.assign(data, data + (size_t)img.width * img.height * 3);
    stbi_image_free(data);
    return true;
}

// 等比缩放 + 填充到 640x640，返回缩放前后坐标映射
struct PreprocessResult {
    std::vector<float> input_tensor;  // 1x3x640x640, RGB, [0,1]
    float scale;                      // 原始图 -> 640 的缩放系数
    int pad_x, pad_y;                 // 填充偏移
};

PreprocessResult preprocess(const MyImage& img, int target = 640) {
    PreprocessResult res;
    res.scale = std::min((float)target / img.width, (float)target / img.height);
    int new_w = (int)std::round(img.width * res.scale);
    int new_h = (int)std::round(img.height * res.scale);
    res.pad_x = (target - new_w) / 2;
    res.pad_y = (target - new_h) / 2;

    res.input_tensor.assign(3 * (size_t)target * target, 0.f);  // 填充区为0（黑色）

    for (int y = 0; y < new_h; ++y) {
        int sy = (int)(y / res.scale);
        sy = std::clamp(sy, 0, img.height - 1);
        for (int x = 0; x < new_w; ++x) {
            int sx = (int)(x / res.scale);
            sx = std::clamp(sx, 0, img.width - 1);
            const uint8_t* p = &img.rgb[((size_t)sy * img.width + sx) * 3];
            int dy = y + res.pad_y;
            int dx = x + res.pad_x;
            size_t base = (size_t)dy * target + dx;
            res.input_tensor[0 * (size_t)target * target + base] = p[0] / 255.f;
            res.input_tensor[1 * (size_t)target * target + base] = p[1] / 255.f;
            res.input_tensor[2 * (size_t)target * target + base] = p[2] / 255.f;
        }
    }
    return res;
}

// ---------------- 推理 ----------------
struct Detection {
    float x1, y1, x2, y2, conf;
    int class_id;
};

// 在 cv::Mat 上画检测框（红色矩形），frame 会被直接修改
void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(frame,
                      cv::Point((int)d.x1, (int)d.y1),
                      cv::Point((int)d.x2, (int)d.y2),
                      cv::Scalar(0, 0, 255),   // BGR 顺序，红色
                      2);                       // 线宽 2 像素
    }
}

int main(int argc, char* argv[]) {
    std::string img_path = argc > 1 ? argv[1] : "bus.jpg";
    std::string model_path = argc > 2 ? argv[2] : "yolo26n.onnx";
    std::string device = argc > 3 ? argv[3] : "cpu";  // 可选: "gpu" 或 "cpu"（MinGW 下 GPU 不可用，默认 CPU）
    float conf_thresh = 0.25f;

    // 判断输入是不是视频文件（按扩展名）
    auto is_video = [&]() {
        std::string p = img_path;
        std::string lower;
        for (char c : p) lower += (char)std::tolower(c);
        return lower.find(".mp4") != std::string::npos ||
               lower.find(".avi") != std::string::npos ||
               lower.find(".mov") != std::string::npos ||
               lower.find(".mkv") != std::string::npos;
    };
    // 判断输入是不是摄像头编号（source=0, 1, 2...）
    auto is_camera = [&]() {
        if (img_path.empty()) return false;
        for (char c : img_path)
            if (!std::isdigit(c)) return false;
        return true;
    };

    // 创建 onnxruntime 环境 + session（图片和视频共用）
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool use_gpu = (device == "gpu");
    if (use_gpu) {
        std::cout << "正在启用 CUDA provider..." << std::endl;
        try {
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const std::exception& e) {
            std::cout << "CUDA provider 配置失败: " << e.what() << "，回退 CPU" << std::endl;
            use_gpu = false;
        }
    }

    std::wstring wmodel(model_path.begin(), model_path.end());
    Ort::Session session(env, wmodel.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // ================= 视频/摄像头模式 =================
    if (is_video() || is_camera()) {
        bool is_cam = is_camera();
        std::cout << (is_cam ? "检测到摄像头输入，进入实时模式..." : "检测到视频输入，进入视频模式...") << std::endl;
        cv::VideoCapture cap;
        if (is_cam) {
            int cam_id = std::stoi(img_path);
            cap.open(cam_id);  // 打开摄像头
        } else {
            cap.open(img_path);  // 打开视频文件
        }
        if (!cap.isOpened()) {
            std::cerr << "无法打开: " << img_path << std::endl;
            return 1;
        }
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "画面: " << width << "x" << height << std::endl;

        // 只有视频文件模式才创建输出文件；摄像头模式只实时显示不保存
        cv::VideoWriter writer;
        if (!is_cam) {
            if (fps < 1) fps = 10;  // 兜底：部分视频 fps 读不到时用默认
            writer.open("output_video.avi",
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                std::cerr << "无法创建输出视频 output_video.avi" << std::endl;
                return 1;
            }
        }

        cv::Mat frame;
        int frame_count = 0;
        // 跳帧优化：每 DETECT_EVERY_N 帧检测一次，中间帧用上次结果
        const int DETECT_EVERY_N = (is_cam ? 3 : 1);  // 摄像头跳帧，视频不跳
        std::vector<Detection> last_dets;             // 保存上次检测结果

        while (cap.read(frame)) {
            // 跳帧逻辑：只有该检测时才跑模型
            bool do_detect = (frame_count % DETECT_EVERY_N == 0);
            if (do_detect) {
                // 将 OpenCV 帧(BGR) 转为 MyImage(RGB) 供预处理
                cv::Mat rgb;
                cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
                MyImage img;
                img.width = rgb.cols;
                img.height = rgb.rows;
                img.rgb.assign(rgb.data, rgb.data + (size_t)rgb.cols * rgb.rows * 3);

                auto pre = preprocess(img, 640);  // 模型固定 640 输入

                // 获取输入输出名
                Ort::AllocatedStringPtr inp = session.GetInputNameAllocated(0, allocator);
                Ort::AllocatedStringPtr out = session.GetOutputNameAllocated(0, allocator);
                std::string iname = inp.get(), oname = out.get();

                std::vector<int64_t> input_shape = {1, 3, 640, 640};
                Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                std::vector<Ort::Value> input_tensors;
                input_tensors.push_back(Ort::Value::CreateTensor<float>(
                    mem, pre.input_tensor.data(), pre.input_tensor.size(),
                    input_shape.data(), input_shape.size()));

                const char* inames[] = {iname.c_str()};
                const char* onames[] = {oname.c_str()};
                auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                  inames, input_tensors.data(), 1,
                                                  onames, 1);

                // 解析结果
                float* out_data = output_tensors[0].GetTensorMutableData<float>();
                auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                int num_dets = (int)out_shape[1];
                int stride = (int)out_shape[2];

                last_dets.clear();
                for (int i = 0; i < num_dets; ++i) {
                    const float* row = out_data + (size_t)i * stride;
                    float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
                    float conf = row[4];
                    int cls = (int)row[5];
                    if (conf < conf_thresh) continue;
                    x1 = (x1 - pre.pad_x) / pre.scale;
                    y1 = (y1 - pre.pad_y) / pre.scale;
                    x2 = (x2 - pre.pad_x) / pre.scale;
                    y2 = (y2 - pre.pad_y) / pre.scale;
                    last_dets.push_back({x1, y1, x2, y2, conf, cls});
                }
            }

            // 在帧上画框（用当前帧的检测结果，跳帧时用上次结果）
            draw_detections(frame, last_dets);

            // 摄像头模式：实时显示窗口
            if (is_cam) {
                cv::imshow("YOLO 实时检测", frame);
                int key = cv::waitKey(1);   // 刷新窗口（等 1ms）
                if (key == 'q' || key == 27)  // q 或 Esc 退出
                    break;
            }
            // 视频模式：写入输出文件
            if (!is_cam) {
                writer.write(frame);
            }

            ++frame_count;
            if (frame_count % 30 == 0)
                std::cout << "已处理 " << frame_count << " 帧" << std::endl;
        }
        cap.release();
        if (!is_cam) {
            writer.release();
            std::cout << "视频处理完成! 共 " << frame_count << " 帧，输出: output_video.avi" << std::endl;
        } else {
            cv::destroyAllWindows();
            std::cout << "摄像头实时检测结束。共处理 " << frame_count << " 帧" << std::endl;
        }
        return 0;
    }

    // ================= 图片模式 =================
    // 加载图片
    MyImage img;
    if (!load_image(img_path, img)) {
        return 1;
    }
    std::cout << "图片: " << img.width << "x" << img.height << std::endl;

    // 预处理
    auto pre = preprocess(img);

    // 获取输入输出名（env/session 已在前面创建）
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();
    std::string output_name = output_name_ptr.get();

    std::cout << "推理设备: " << (use_gpu ? "CUDA (GPU)" : "CPU") << std::endl;

    // 构造输入 tensor
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        mem_info, pre.input_tensor.data(), pre.input_tensor.size(),
        input_shape.data(), input_shape.size()));

    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    // 推理计时
    auto t0 = std::chrono::high_resolution_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, input_tensors.data(), 1,
                                      output_names, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "推理耗时: " << ms << " ms" << std::endl;

    // 解析输出 [1,300,6]
    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_dets = (int)out_shape[1];  // 300
    int stride = (int)out_shape[2];    // 6

    std::vector<Detection> dets;
    for (int i = 0; i < num_dets; ++i) {
        const float* row = out_data + (size_t)i * stride;
        float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        float conf = row[4];
        int cls = (int)row[5];
        if (conf < conf_thresh) continue;

        // 坐标是 640 空间的，映射回原始图
        x1 = (x1 - pre.pad_x) / pre.scale;
        y1 = (y1 - pre.pad_y) / pre.scale;
        x2 = (x2 - pre.pad_x) / pre.scale;
        y2 = (y2 - pre.pad_y) / pre.scale;
        dets.push_back({x1, y1, x2, y2, conf, cls});
    }

    std::cout << "检测到 " << dets.size() << " 个目标:" << std::endl;
    for (const auto& d : dets) {
        std::cout << "  class=" << d.class_id
                  << " conf=" << d.conf
                  << " box=[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]"
                  << std::endl;
    }
    // ===== 把检测框画到图片上并保存 =====
    // 在原始像素上画框（矩形边框），然后保存为带框图片
    for (const auto& d : dets) {
        // 把框坐标转成整数像素坐标
        int bx1 = (int)d.x1, by1 = (int)d.y1;
        int bx2 = (int)d.x2, by2 = (int)d.y2;
        // 越界保护
        bx1 = std::max(0, std::min(bx1, img.width - 1));
        by1 = std::max(0, std::min(by1, img.height - 1));
        bx2 = std::max(0, std::min(bx2, img.width - 1));
        by2 = std::max(0, std::min(by2, img.height - 1));
        // 画框：把边框线（上下左右各一条线）染成红色 (255,0,0)
        for (int x = bx1; x <= bx2; ++x) {
            img.rgb[((size_t)by1 * img.width + x) * 3 + 0] = 255;  // R
            img.rgb[((size_t)by1 * img.width + x) * 3 + 1] = 0;    // G
            img.rgb[((size_t)by1 * img.width + x) * 3 + 2] = 0;    // B
            img.rgb[((size_t)by2 * img.width + x) * 3 + 0] = 255;
            img.rgb[((size_t)by2 * img.width + x) * 3 + 1] = 0;
            img.rgb[((size_t)by2 * img.width + x) * 3 + 2] = 0;
        }
        for (int y = by1; y <= by2; ++y) {
            img.rgb[((size_t)y * img.width + bx1) * 3 + 0] = 255;
            img.rgb[((size_t)y * img.width + bx1) * 3 + 1] = 0;
            img.rgb[((size_t)y * img.width + bx1) * 3 + 2] = 0;
            img.rgb[((size_t)y * img.width + bx2) * 3 + 0] = 255;
            img.rgb[((size_t)y * img.width + bx2) * 3 + 1] = 0;
            img.rgb[((size_t)y * img.width + bx2) * 3 + 2] = 0;
        }
    }
    // 保存带框图片（stb_image 的写入函数）
    stbi_write_jpg("output_cpp.jpg", img.width, img.height, 3, img.rgb.data(), 95);
    std::cout << "带框图片已保存: output_cpp.jpg" << std::endl;


    return 0;
}
