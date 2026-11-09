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

void exportToPLY(const std::string& filename, const std::vector<cv::Point3d>& points) {
    std::ofstream plyFile(filename);
    if (!plyFile.is_open()) {
        std::cerr << "Не вдалося відкрити файл для запису: " << filename << std::endl;
        return;
    }

    // Запис заголовка PLY
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << points.size() << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "end_header\n";

    // Запис точок
    for (const auto& point : points) {
        plyFile << point.x << " " << point.y << " " << point.z << "\n";
    }

    plyFile.close();
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
    std::string directoryPath = (argc > 1) ? argv[1] : "/Users/yuriishymkiv/studying/coursework/3D-reconstruction/CMakeProject1/data";

    IFrameProvider* frameProvider = new DirectoryFrameProvider(directoryPath);
    IFramePreprocessor* framePreprocessor = new LaplacianFramePreprocessorOpenCV(30);
    IFeatureExtractor* featureExtractor = new AkazeFeatureExtractor();
    IFeatureMatcher* featureMatcher = new HammingRatioMatcher();

    
    cv::Mat previousRawColor, currentRawColor;
    Frame previousFrame, currentFrame;
    int frameCounter = 0;

    if (!processSingleFrame(frameProvider, framePreprocessor, featureExtractor, previousRawColor, previousFrame, frameCounter)) 
    {
        std::cerr << "Не вдалося обробити жодного кадру." << std::endl;
        return -1;
    }

    // Визначення параметрів камери
    double width = static_cast<double>(previousFrame.imageGray.cols);
    double height = static_cast<double>(previousFrame.imageGray.rows);
    double focalLength = 1.2 * std::max(width, height);

    cv::Point2d principalPoint(width / 2.0, height / 2.0);

    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<    focalLength,   0,              principalPoint.x,
                                                        0,              focalLength,    principalPoint.y,
                                                        0,              0,              1);

    while (processSingleFrame(frameProvider, framePreprocessor, featureExtractor, currentRawColor, currentFrame, ++frameCounter)) 
    {
        std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now();

        // Зіставлення дескрипторів між попереднім та поточним кадром
        std::vector<cv::DMatch> matches = featureMatcher->match(previousFrame.descriptors, currentFrame.descriptors);

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - st;

        std::cout << "[Кадр #" << frameCounter << "] Знайдено " << matches.size() << " відповідностей. Час обробки: " << duration.count() << " мс." << std::endl;


        if (matches.size() < 15) {
            std::cout << "Недостатньо відповідностей для обчислення Essential Matrix." << std::endl;
            previousRawColor = currentRawColor.clone();
            previousFrame = currentFrame;
            continue;
        }

        // 3. Формуємо вирівняні вектори 2D-точок (Point2f) за індексами збігів
        std::vector<cv::Point2f> pointsPrevious, pointsCurrent;
        pointsPrevious.reserve(matches.size());
        pointsCurrent.reserve(matches.size());

        for (const auto& match : matches) {
            pointsPrevious.push_back(previousFrame.keypoints[match.queryIdx].pt);
            pointsCurrent.push_back(currentFrame.keypoints[match.trainIdx].pt);
        }

        // 4. Оцінка Есенціальної матриці (Essential Matrix) через RANSAC
        cv::Mat inlierMask;
        cv::Mat essentialMatrix = cv::findEssentialMat(
            pointsPrevious, 
            pointsCurrent,
            cameraMatrix, 
            cv::RANSAC, 
            0.999, 
            1.0, 
            inlierMask
        );

        if (essentialMatrix.empty()) {
            std::cout << "Не вдалося обчислити Essential Matrix." << std::endl;
            previousRawColor = currentRawColor.clone();
            previousFrame = currentFrame;
            continue;
        }

        // 5. Відновлення відносної позиції камери (R, t) з Essential Matrix
        cv::Mat rotateMatrix, translateVector;
        cv::recoverPose(essentialMatrix, 
            pointsPrevious, 
            pointsCurrent, 
            cameraMatrix, 
            rotateMatrix, 
            translateVector, 
            inlierMask);

        // 6. Формування проєкційних матриць P0 та P1 (розмір 3x4)

        cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64F);
        cameraMatrix.copyTo(P0(cv::Rect(0, 0, 3, 3))); // Вставляємо K в P0

        cv::Mat Rt = cv::Mat::zeros(3, 4, CV_64F);
        rotateMatrix.copyTo(Rt(cv::Rect(0, 0, 3, 3))); // Вставляємо R в Rt
        translateVector.copyTo(Rt.col(3)); // Вставляємо t в Rt

        cv::Mat P1 = cameraMatrix * Rt; // P1 = K * [R|t]

        std::vector<cv::Point2f> inlierPointsPrev, inlierPointsCurr;

        for (size_t i = 0; i < matches.size(); ++i) {
            if (inlierMask.at<uchar>(i)) {
                inlierPointsPrev.push_back(pointsPrevious[i]);
                inlierPointsCurr.push_back(pointsCurrent[i]);
            }
        }

        std::cout << "Інлаєрів після RANSAC: " << inlierPointsPrev.size() << std::endl;

        // 7. Тріангуляція точок
        cv::Mat points4D;
        try {
            cv::triangulatePoints(P0, P1, inlierPointsPrev, inlierPointsCurr, points4D);
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Error у triangulatePoints: " << e.what() << std::endl;
            std::cerr << "P0 size: " << P0.size() << " type: " << P0.type() << std::endl;
            std::cerr << "P1 size: " << P1.size() << " type: " << P1.type() << std::endl;
            std::cerr << "PointsPrev count: " << inlierPointsPrev.size() << std::endl;
            std::cerr << "PointsCurr count: " << inlierPointsCurr.size() << std::endl;
            previousRawColor = currentRawColor.clone();
            previousFrame = currentFrame;
            continue;
        }

        // 8. Перетворення гомогенних координат у 3D
        std::vector<cv::Point3d> reconstructedPoints3D;
        reconstructedPoints3D.reserve(points4D.cols);

        for (int i = 0; i < points4D.cols; ++i) {
            double w = points4D.at<float>(3, i);

            if (std::abs(w) <= 1e-6) { // Перевірка на нульовий w
                continue; // Пропускаємо цю точку
            }

            double x = points4D.at<float>(0, i) / w;
            double y = points4D.at<float>(1, i) / w;
            double z = points4D.at<float>(2, i) / w;

            if (z < 0) { // Перевірка на негативну глибину
                continue; // Пропускаємо цю точку
            }

            reconstructedPoints3D.emplace_back(x, y, z);
        }
        
        std::cout << "Успішно тріангульовано 3D-точок: " << reconstructedPoints3D.size() << std::endl;

        exportToPLY("reconstructed_points.ply", reconstructedPoints3D);


        // 9. Візуалізація збігів
        cv::Mat matchesImage;
        cv::drawMatches(
            previousRawColor, previousFrame.keypoints,
            currentRawColor, currentFrame.keypoints,
            matches,
            matchesImage,
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            inlierMask, // Відображатимуться тільки інлаєри RANSAC
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