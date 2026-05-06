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
        // 主判断：用户显式要求 CUDA 时，不可用就直接提示。
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("当前环境不可用 CUDA，请改用 --device=cpu 或 --device=auto");
        }
        return torch::Device(torch::kCUDA);
    }
    // 主判断：auto 模式优先使用 CUDA，否则退回 CPU。
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

void PrintUsage() {
    std::cout
        << "用法:\n"
        << "  mtcnn_camera_demo --model_dir=<模型目录> "
           "[--camera_id=0] [--device=auto|cpu|cuda]\n";
}

}  // namespace

int main(int argc, char** argv) {
    InitConsoleEncoding();

    const cv::CommandLineParser parser(
        argc,
        argv,
        "{help h||显示帮助}"
        "{model_dir||PNet/RNet/ONet TorchScript 模型目录}"
        "{camera_id|0|摄像头编号}"
        "{device|auto|推理设备，可选 auto/cpu/cuda}");

    // 主判断：缺少模型目录时先输出帮助。
    if (parser.has("help") || parser.get<std::string>("model_dir").empty()) {
        PrintUsage();
        parser.printMessage();
        return 0;
    }

    try {
        MtcnnDetector::Options options;
        options.device = ResolveDevice(parser.get<std::string>("device"));

        MtcnnDetector detector(parser.get<std::string>("model_dir"), options);
        cv::VideoCapture capture(parser.get<int>("camera_id"));

        // 主判断：摄像头打开失败时直接退出，避免空循环。
        if (!capture.isOpened()) {
            std::cerr << "打开摄像头失败" << std::endl;
            return 1;
        }

        std::cout << "按 q 或 ESC 退出" << std::endl;
        while (true) {
            cv::Mat frame;
            capture >> frame;

            // 主判断：读取到空帧时跳过本次循环。
            if (frame.empty()) {
                continue;
            }

            const std::vector<MtcnnDetector::FaceInfo> faces = detector.Detect(frame);
            MtcnnDetector::DrawDetections(frame, faces);
            cv::putText(
                frame,
                cv::format("faces: %d", static_cast<int>(faces.size())),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2);

            cv::imshow("mtcnn_camera_demo", frame);
            const int key = cv::waitKey(1);

            // 主判断：按 q 或 ESC 时退出实时检测。
            if (key == 'q' || key == 27) {
                break;
            }
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
