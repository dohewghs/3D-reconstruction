#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <iostream>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

#include "DirectoryFrameProvider.hpp"
#include "LaplacianSharpnessPreprocessorOpenCV.hpp"
#include "SiftFeatureExtractor.hpp"
#include "RatioMatcher.hpp"
#include "SfMPipeline.hpp"

/**
 * Фокусна відстань у пікселях = focalFactor * більша сторона кадру.
 * 0.7 відповідає ~24-26 мм еквівалента — типова основна камера телефона.
 * (Попереднє 1.2 відповідало ~43 мм і завищувало f у ~1.7 раза: геометрія E/PnP спотворювалась.)
 */
cv::Mat createDefaultCameraMatrix(double width, double height, double focalFactor) {
    double focalLength = focalFactor * std::max(width, height);
    return (cv::Mat_<double>(3, 3) <<
        focalLength, 0.0,         width / 2.0,
        0.0,         focalLength, height / 2.0,
        0.0,         0.0,         1.0);
}

static void printUsage(const char* exe) {
    std::cout << "Використання: " << exe << " <папка з фото> [опції]\n"
              << "  --focal-factor F   f = F * більша сторона кадру (типово 0.7)\n"
              << "  --max-dim N        робоча роздільність SfM: більша сторона, px (типово 2000)\n"
              << "  --dense-scale S    роздільність щільної стадії відносно SfM (типово 0.7)\n"
              << "  --match-window N   зіставляти кадри на відстані <= N (типово 10, 0 = всі пари)\n"
              << "  --mesh-step N      крок сітки поверхні, px (типово 2; 1 = максимальна деталізація)\n"
              << "  --output PREFIX    префікс вихідних файлів (типово reconstruction)\n";
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // інакше українські повідомлення у консолі Windows нечитабельні
#endif

    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage(argv[0]);
        return argc < 2 ? -1 : 0;
    }

    std::string directoryPath = argv[1];
    double focalFactor = 0.7;
    int maxDimension = 2000;
    std::string outputPrefix = "reconstruction";
    SfMPipelineOptions options;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        bool hasValue = (i + 1 < argc);
        if (arg == "--focal-factor" && hasValue) focalFactor = std::stod(argv[++i]);
        else if (arg == "--max-dim" && hasValue) maxDimension = std::stoi(argv[++i]);
        else if (arg == "--dense-scale" && hasValue) options.denseScale = std::stod(argv[++i]);
        else if (arg == "--match-window" && hasValue) options.matchWindow = std::stoi(argv[++i]);
        else if (arg == "--mesh-step" && hasValue) options.meshStep = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--output" && hasValue) outputPrefix = argv[++i];
        else {
            std::cerr << "Невідомий аргумент: " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    std::cout << "Ініціалізація 3D Reconstruction Pipeline..." << std::endl;
    std::cout << "Шлях до датасету: " << directoryPath << std::endl;

    // 1. Створення обробників
    auto provider = std::make_unique<DirectoryFrameProvider>(directoryPath, maxDimension);

    // Матриця K має відповідати реальному розміру кадрів (після зменшення до робочої роздільності),
    // інакше головна точка (cx, cy) зміщена і вся геометрія (E, PnP, ректифікація) спотворюється.
    cv::Mat firstFrame;
    if (!provider->getNextFrame(firstFrame)) {
        std::cerr << "Не знайдено жодного зображення у: " << directoryPath << std::endl;
        return -1;
    }
    provider->reset();
    cv::Mat K = createDefaultCameraMatrix(firstFrame.cols, firstFrame.rows, focalFactor);
    std::cout << "Робочий розмір кадру: " << firstFrame.cols << "x" << firstFrame.rows
              << ", f = " << K.at<double>(0, 0) << " px" << std::endl;

    auto preprocessor = std::make_unique<LaplacianFramePreprocessorOpenCV>(30.0);
    auto extractor = std::make_unique<SiftFeatureExtractor>(8000);
    auto matcher = std::make_unique<RatioMatcher>(extractor->getDefaultNorm(), 0.8f);

    SfMPipeline pipeline(
        std::move(provider),
        std::move(preprocessor),
        std::move(extractor),
        std::move(matcher),
        K,
        options
    );

    if (!pipeline.run(outputPrefix)) {
        std::cerr << "Виконання пайплайну завершилося з помилкою!" << std::endl;
        return -1;
    }

    return 0;
}
