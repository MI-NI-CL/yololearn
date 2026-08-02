cmake_minimum_required(VERSION 3.20)
project(cpp_yolo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- onnxruntime 路径 ----------
# onnxruntime 解压在 C:\cpp_msvc_build\ort127\，用完整绝对路径避免相对路径歧义
set(ORT_ROOT "C:/cpp_msvc_build/ort127/onnxruntime-win-x64-gpu_cuda13-1.27.0")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")

# ---------- OpenCV 路径（编译自源码安装到 C:\opencv_build\install）----------
set(CV_ROOT "C:/opencv_build/install")
set(CV_INCLUDE_DIR "${CV_ROOT}/include")
set(CV_LIB_DIR "${CV_ROOT}/x64/mingw/staticlib")

include_directories(${ORT_INCLUDE_DIR} ${CV_INCLUDE_DIR})

# ---------- 可执行文件 ----------
add_executable(cpp_yolo yolo_infer.cpp)

# 链接 onnxruntime.dll（MinGW 直接链接 .dll）
# 链接 OpenCV 静态库（用 --start-group 循环解析依赖，避免链接顺序问题）
target_link_libraries(cpp_yolo
    "${ORT_LIB_DIR}/onnxruntime.dll"
    -Wl,--start-group
    "${CV_LIB_DIR}/libopencv_videoio500.a"
    "${CV_LIB_DIR}/libopencv_imgproc500.a"
    "${CV_LIB_DIR}/libopencv_imgcodecs500.a"
    "${CV_LIB_DIR}/libopencv_highgui500.a"
    "${CV_LIB_DIR}/libopencv_geometry500.a"
    "${CV_LIB_DIR}/libopencv_flann500.a"
    "${CV_LIB_DIR}/libopencv_core500.a"
    # OpenCV 依赖的第三方库
    "${CV_LIB_DIR}/liblibclapack.a"
    "${CV_LIB_DIR}/liblibjpeg-turbo.a"
    "${CV_LIB_DIR}/liblibpng.a"
    "${CV_LIB_DIR}/liblibtiff.a"
    "${CV_LIB_DIR}/liblibopenjp2.a"
    "${CV_LIB_DIR}/liblibwebp.a"
    "${CV_LIB_DIR}/libzlib.a"
    -Wl,--end-group
)

# 把 onnxruntime DLL 拷贝到可执行文件旁，运行时才能找到
foreach(ort_dll
    onnxruntime.dll
    onnxruntime_providers_shared.dll
    onnxruntime_providers_cuda.dll
    onnxruntime_providers_tensorrt.dll
)
    add_custom_command(TARGET cpp_yolo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_LIB_DIR}/${ort_dll}" "$<TARGET_FILE_DIR:cpp_yolo>/${ort_dll}"
    )
endforeach()
