#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <ostream>
#include <string>

#include "DirectoryFrameProvider.hpp"
#include "LaplacianSharpnessPreprocessorOpenCV.hpp"
#include "AkazeFeatureExtractor.hpp"
#include "HammingRationMatcher.hpp"
#include "SfMPipeline.hpp"

#include "VideoFrameProvider.hpp"

cv::Mat createDefaultCameraMatrix(double width, double height) {
    double focalLength = 1.2 * std::max(width, height);

    double f = (24.0 * std::max(width, height)) / 36.0;

    return (cv::Mat_<double>(3, 3) <<
        f, 0.0,         width / 2.0,
        0.0,         f, height / 2.0,
        0.0,         0.0,         1.0);
}

int main(int argc, char** argv) {
    // Шлях до тестового фото
    std::string directoryPath = (argc > 1) 
        ? argv[1] 
        : "/Users/yuriishymkiv/studying/coursework/3D-reconstruction/CMakeProject1/data/video/test.mp4";

    cv::Mat K = createDefaultCameraMatrix(2268.0, 4032.0);

    // 1. Створення обробників
    //auto provider = std::make_unique<DirectoryFrameProvider>(directoryPath);
    auto provider = std::make_unique<VideoFrameProvider>(directoryPath);
    auto preprocessor = std::make_unique<LaplacianFramePreprocessorOpenCV>(30.0);
    auto extractor = std::make_unique<AkazeFeatureExtractor>();
    auto matcher = std::make_unique<HammingRatioMatcher>(0.75f);
    
    // SfMPipeline pipeline(
    //     std::move(provider),
    //     std::move(preprocessor),
    //     std::move(extractor),
    //     std::move(matcher),
    //     K
    // );

    // if (!pipeline.run("reconstructed_scene.ply")) {
    //     std::cerr << "Виконання пайплайну завершилося з помилкою!" << std::endl;
    //     return -1;
    // }

    cv::Mat frame;
    size_t counter = 0;
    while(provider->getNextFrame(frame))
    {
        std::cout << "Кадр: " << counter++ << std::endl;

        cv::imshow("Кадр", frame);

        char key = static_cast<char>(cv::waitKey(0));
        if (key == 27) {
            break;
        }
    }

    return 0;
}