#include <iostream>
#include <memory>
#include <opencv2/core.hpp>

#include "DirectoryFrameProvider.hpp"
#include "LaplacianSharpnessPreprocessorOpenCV.hpp"
#include "AkazeFeatureExtractor.hpp"
#include "HammingRationMatcher.hpp"
#include "SfMPipeline.hpp"

cv::Mat createDefaultCameraMatrix(double width, double height) {
    double focalLength = 1.2 * std::max(width, height);
    return (cv::Mat_<double>(3, 3) <<
        focalLength, 0.0,         width / 2.0,
        0.0,         focalLength, height / 2.0,
        0.0,         0.0,         1.0);
}

int main(int argc, char** argv) {
    // Шлях до тестового фото
    std::string directoryPath = (argc > 1) 
        ? argv[1] 
        : "/Users/yuriishymkiv/studying/coursework/3D-reconstruction/CMakeProject1/data/set3";


    std::cout << "Ініціалізація 3D Reconstruction Pipeline..." << std::endl;
    std::cout << "Шлях до датасету: " << directoryPath << std::endl;

    // 1. Створення обробників
    auto provider = std::make_unique<DirectoryFrameProvider>(directoryPath);

    // Матриця K має відповідати реальному розміру кадрів (а не захардкодженим 1920x1080),
    // інакше головна точка (cx, cy) зміщена і вся геометрія (E, PnP, ректифікація) спотворюється.
    cv::Mat firstFrame;
    if (!provider->getNextFrame(firstFrame)) {
        std::cerr << "Не знайдено жодного зображення у: " << directoryPath << std::endl;
        return -1;
    }
    provider->reset();
    cv::Mat K = createDefaultCameraMatrix(firstFrame.cols, firstFrame.rows);
    std::cout << "Розмір кадру: " << firstFrame.cols << "x" << firstFrame.rows << std::endl;
    auto preprocessor = std::make_unique<LaplacianFramePreprocessorOpenCV>(30.0);
    auto extractor = std::make_unique<AkazeFeatureExtractor>();
    auto matcher = std::make_unique<HammingRatioMatcher>(0.75f);
    
    SfMPipeline pipeline(
        std::move(provider),
        std::move(preprocessor),
        std::move(extractor),
        std::move(matcher),
        K
    );

    if (!pipeline.run("reconstructed_scene.ply")) {
        std::cerr << "Виконання пайплайну завершилося з помилкою!" << std::endl;
        return -1;
    }

    return 0;
}