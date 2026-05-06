# C++ MTCNN 推理

该目录提供了一个基于 `LibTorch + OpenCV` 的 C++ 版 MTCNN 推理实现。

## 目录说明

- `include/mtcnn_detector.h`：MTCNN 检测类声明
- `src/mtcnn_detector.cpp`：MTCNN 检测类实现
- `examples/image_demo.cpp`：图片检测示例
- `examples/camera_demo.cpp`：摄像头实时检测示例
- `CMakeLists.txt`：CMake 构建文件

## 依赖

- OpenCV
- LibTorch（C++ 版 PyTorch）

OpenCV 下载地址：

 - [Windows OpenCV 4.12.0](https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-windows.exe)
 - [Source OpenCV 4.12.0](https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip)

LibTorch 下载地址，如果下载的是GPU版本，要对应自己系统上的CUDA版本：

 - [Windows libtorch 2.11.0（CUDA 13.0）](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.11.0%2Bcu130.zip)
 - [Windows libtorch 2.11.0（CPU）](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.11.0%2Bcpu.zip)
 - [Linux libtorch 2.11.0（CUDA 13.0）](https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip)
 - [Linux libtorch 2.11.0（CPU）](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip)
 - [更多版本下载地址](https://blog.csdn.net/liang_baikai/article/details/127849577)

## 注意事项

> 注意：这里直接加载项目根目录 `infer_models` 下的 `PNet.pth`、`RNet.pth`、`ONet.pth`，因此需要保证这些模型是可被 `torch::jit::load` 加载的 TorchScript 模型。

## 构建

以下示例以 Windows 为例，假设：

- OpenCV 已正确安装在 `D:/libs/opencv/build`
- LibTorch 解压路径为 `D:/libs/libtorch`
- 你当前位于项目根目录 `Pytorch-MTCNN/`

1. 进入到cpp目录
```powershell
cd Pytorch-MTCNN/cpp
```

2. 配置CMake并构建
```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH=D:/libs/libtorch -DOpenCV_DIR=D:/libs/opencv/build
cmake --build build --config Release
```

## 图片检测示例

```powershell
./build/Release/mtcnn_image_demo.exe `
  --model_dir=../infer_models `
  --image_path=../dataset/test.jpg `
  --save_path=result.jpg `
  --device=auto
```

参数说明：

- `--model_dir`：模型目录，必须包含 `PNet.pth`、`RNet.pth`、`ONet.pth`
- `--image_path`：待检测图片路径
- `--save_path`：可选，保存可视化结果
- `--device`：推理设备，可选 `auto`、`cpu`、`cuda`
- `--show`：是否显示窗口，默认 `1`

## 摄像头检测示例

```powershell
./build/Release/mtcnn_camera_demo.exe `
  --model_dir=../infer_models `
  --camera_id=0 `
  --device=auto
```

运行后：

- 按 `q` 退出
- 按 `ESC` 退出
