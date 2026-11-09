#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "AkazeFeatureExtractor.hpp"
#include "DirectoryFrameProvider.hpp"
#include "LaplacianSharpnessPreprocessorOpenCV.hpp"
#include "HammingRationMatcher.hpp"
#include "TwoViewGeometryEstimator.hpp"
#include "PlyExporter.hpp"

cv::Mat createDefaultCameraMatrix(double width, double height) {
    double focalLength = 1.2 * std::max(width, height);
    return (cv::Mat_<double>(3, 3) <<
        focalLength, 0.0,         width / 2.0,
        0.0,         focalLength, height / 2.0,
        0.0,         0.0,         1.0);
}

bool processSingleFrame(IFrameProvider* provider,
                        IFramePreprocessor* preprocessor,
                        IFeatureExtractor* extractor,
                        cv::Mat& outRawColor,
                        Frame& outFrame,
                        int frameId) 
{
    cv::Mat raw;
    while (provider->getNextFrame(raw)) {
        cv::Mat gray;
        // Відкидаємо змазані кадри
        if (!preprocessor->process(raw, gray)) {
            std::cout << "[Кадр #" << frameId << "] Пропущено через розмиття." << std::endl;
            continue;
        }

        outRawColor = raw;
        outFrame.id = frameId;
        outFrame.imageGray = gray;
        extractor->extract(outFrame);
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    // Шлях до тестового фото
    std::string directoryPath = (argc > 1) 
        ? argv[1] 
        : "/Users/yuriishymkiv/studying/coursework/3D-reconstruction/CMakeProject1/data";

    // 1. Створення обробників
    IFrameProvider* frameProvider = new DirectoryFrameProvider(directoryPath);
    IFramePreprocessor* framePreprocessor = new LaplacianFramePreprocessorOpenCV(30);
    IFeatureExtractor* featureExtractor = new AkazeFeatureExtractor();
    IFeatureMatcher* featureMatcher = new HammingRatioMatcher();
    
    cv::Mat previousRawColor, currentRawColor;
    Frame previousFrame, currentFrame;
    int frameCounter = 0;

    // 2. Обробка першого кадру
    if (!processSingleFrame(frameProvider, framePreprocessor, featureExtractor, previousRawColor, previousFrame, frameCounter)) 
    {
        std::cerr << "Не вдалося обробити жодного кадру." << std::endl;
        return -1;
    }

    // Визначення параметрів камери
    double width = static_cast<double>(previousFrame.imageGray.cols);
    double height = static_cast<double>(previousFrame.imageGray.rows);

    cv::Mat cameraMatrix = createDefaultCameraMatrix(width, height);

    ITwoViewGeometryEstimator* geometryEstimator = new TwoViewGeometryEstimator(cameraMatrix);

    while (processSingleFrame(frameProvider, framePreprocessor, featureExtractor, currentRawColor, currentFrame, ++frameCounter)) 
    {
        std::cout << "\n--- Зіставлення кадрів #" << previousFrame.id << " та #" << currentFrame.id << " ---" << std::endl;

        // Зіставлення дескрипторів між попереднім та поточним кадром
        std::vector<cv::DMatch> matches = featureMatcher->match(previousFrame.descriptors, currentFrame.descriptors);

        std::cout << "Знайдено кандидатів: " << matches.size() << std::endl;

        TwoViewReconstructionResult geomResult = geometryEstimator->reconstruct(previousFrame, currentFrame, matches);
        
        if (!geomResult.success) {
            std::cout << "Не вдалося оцінити геометрію між кадрами." << std::endl;
            continue;
        }

        std::cout << "Інлаєрів RANSAC: " << geomResult.inliersPrev.size() << std::endl;
        std::cout << "Тріангульовано 3D-точок: " << geomResult.points3D.size() << std::endl;

        std::string filename = "cloud_frame_" + std::to_string(frameCounter) + ".ply";
        PlyExporter::save(filename, geomResult.points3D);


        // 9. Візуалізація збігів
        cv::Mat matchesImage;
        cv::drawMatches(
            previousRawColor, previousFrame.keypoints,
            currentRawColor, currentFrame.keypoints,
            matches,
            matchesImage,
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            geomResult.inlierMask, // Відображатимуться тільки інлаєри RANSAC
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        );
        cv::imshow("Відповідності між кадрами", matchesImage);


        char key = static_cast<char>(cv::waitKey(0));
        if (key == 27) { // Escape
            std::cout << "Зупинено користувачем." << std::endl;
            break;
        }

        
        previousRawColor = currentRawColor.clone();
        previousFrame = currentFrame;
    }

    cv::destroyAllWindows();

    return 0;
}