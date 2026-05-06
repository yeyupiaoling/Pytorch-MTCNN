#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "mtcnn_detector.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void InitConsoleEncoding() {
#ifdef _WIN32
    // 主判断：Windows 控制台默认可能不是 UTF-8，这里主动切换，避免中文输出乱码。
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

torch::Device ResolveDevice(const std::string& device_name) {
    if (device_name == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (device_name == "cuda") {
        // 主判断：用户显式要求 CUDA 时，如果不可用就直接报错。
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("当前环境不可用 CUDA，请改用 --device=cpu 或 --device=auto");
        }
        return torch::Device(torch::kCUDA);
    }
    // 主判断：auto 模式优先选择 CUDA，否则回退到 CPU。
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

void PrintUsage() {
    std::cout
        << "用法:\n"
        << "  mtcnn_image_demo --model_dir=<模型目录> --image_path=<图片路径> "
           "[--device=auto|cpu|cuda] [--save_path=<输出路径>] [--show=0|1]\n";
}

}  // namespace

int main(int argc, char** argv) {
    InitConsoleEncoding();

    const cv::CommandLineParser parser(
        argc,
        argv,
        "{help h||显示帮助}"
        "{model_dir||PNet/RNet/ONet TorchScript 模型目录}"
        "{image_path||待检测图片路径}"
        "{device|auto|推理设备，可选 auto/cpu/cuda}"
        "{save_path||可选，保存结果图片路径}"
        "{show|1|是否显示结果窗口，1 表示显示}");

    // 主判断：参数不完整时先打印帮助，避免直接运行失败。
    if (parser.has("help") || parser.get<std::string>("model_dir").empty() ||
        parser.get<std::string>("image_path").empty()) {
        PrintUsage();
        parser.printMessage();
        return 0;
    }

    try {
        const std::string model_dir = parser.get<std::string>("model_dir");
        const std::string image_path = parser.get<std::string>("image_path");
        const std::string save_path = parser.get<std::string>("save_path");
        const bool show = parser.get<int>("show") != 0;

        cv::Mat image = cv::imread(image_path);
        // 主判断：图片读取失败时直接退出，避免后续空图推理。
        if (image.empty()) {
            std::cerr << "读取图片失败: " << image_path << std::endl;
            return 1;
        }

        MtcnnDetector::Options options;
        options.device = ResolveDevice(parser.get<std::string>("device"));

        MtcnnDetector detector(model_dir, options);
        const std::vector<MtcnnDetector::FaceInfo> faces = detector.Detect(image);

        cv::Mat visualized = image.clone();
        MtcnnDetector::DrawDetections(visualized, faces);

        std::cout << "检测到人脸数量: " << faces.size() << std::endl;
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            std::cout << "face[" << i << "] bbox=("
                      << face.bbox.x << ", "
                      << face.bbox.y << ", "
                      << face.bbox.x + face.bbox.width << ", "
                      << face.bbox.y + face.bbox.height << "), score="
                      << face.score << std::endl;
        }

        // 主判断：只有传入保存路径时才写文件。
        if (!save_path.empty()) {
            cv::imwrite(save_path, visualized);
            std::cout << "结果已保存到: " << save_path << std::endl;
        }

        // 主判断：命令行显式关闭显示时不弹窗。
        if (show) {
            cv::imshow("mtcnn_image_demo", visualized);
            cv::waitKey(0);
        }
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << "Torch 推理失败: " << error.what() << std::endl;
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "运行失败: " << error.what() << std::endl;
        return 1;
    }
}
