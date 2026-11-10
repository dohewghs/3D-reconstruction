#pragma once

#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <string>
#include <opencv2/core.hpp>

#include "Types.hpp"
#include "IFrameProvider.hpp"
#include "IFramePreprocessor.hpp"
#include "IFeatureExtractor.hpp"
#include "IFeatureMatcher.hpp"
#include "TwoViewGeometryEstimator.hpp"
#include "GlobalMap.hpp"
#include "PnpTracker.hpp"
#include "IncrementalTriangulator.hpp"
#include "BundleAdjuster.hpp"
#include "PlyExporter.hpp"

class SfMPipeline{
private:
    std::unique_ptr<IFrameProvider> m_provider;
    std::unique_ptr<IFramePreprocessor> m_preprocessor;
    std::unique_ptr<IFeatureExtractor> m_extractor;
    std::unique_ptr<IFeatureMatcher> m_matcher;

    cv::Mat m_K;
    GlobalMap m_map;
    std::unordered_map<int, Frame> m_processedFrames;

    bool getNextValidFrame(cv::Mat& outRawColor, Frame& outFrame, int frameId) {
        cv::Mat raw;
        while(m_provider->getNextFrame(raw))
        {
            cv::Mat gray;
            if (!m_preprocessor->process(raw, gray)) {
                std::cout << "[Пайплайн] Кадр #" << frameId << " пропущено (розмиття)." << std::endl;
                continue;
            }

            outRawColor = raw;
            outFrame.id = frameId;
            outFrame.imageGray = gray;
            m_extractor->extract(outFrame);
            return true;
        }

        return false;
    }

public:
    SfMPipeline(
        std::unique_ptr<IFrameProvider> provider,
        std::unique_ptr<IFramePreprocessor> preprocessor,
        std::unique_ptr<IFeatureExtractor> extractor,
        std::unique_ptr<IFeatureMatcher> matcher,
        const cv::Mat& cameraMatrix)
        : m_provider(std::move(provider)),
          m_preprocessor(std::move(preprocessor)),
          m_extractor(std::move(extractor)),
          m_matcher(std::move(matcher))
    {
        cameraMatrix.convertTo(m_K, CV_64F);
    }

    bool run(const std::string& outputPlyPath = "final_sparse_cloud.ply") {
        int frameCounter = 0;

        // Етап 1

        cv::Mat frame0Color;
        Frame frame0;
        if (!getNextValidFrame(frame0Color, frame0, frameCounter++)) {
            std::cerr << "[Помилка] Не вдалося завантажити перший кадр." << std::endl;
            return false;
        }

        m_processedFrames[frame0.id] = frame0;
        std::cout << "[Пайплайн] Базовий Кадр #" << frame0.id << " успішно зчитано ("
                  << frame0.keypoints.size() << " точок AKAZE)." << std::endl;

        // Етап 2

        cv::Mat frame1Color;
        Frame frame1;
        if (!getNextValidFrame(frame1Color, frame1, frameCounter++)) {
            std::cerr << "[Помилка] Не вдалося завантажити другий кадр для ініціалізації." << std::endl;
            return false;
        }
        m_processedFrames[frame1.id] = frame1;

        std::cout << "[Пайплайн] Зіставлення початкової пари (#0 <-> #1)..." << std::endl;
        auto initialMatches = m_matcher->match(frame0.descriptors, frame1.descriptors);

        TwoViewGeometryEstimator initializer(m_K);
        auto initResult = initializer.reconstruct(frame0, frame1, initialMatches);

        if (!initResult.success || initResult.points3D.size() < 30) { 
            std::cerr << "[Помилка] Ініціалізація не вдалася (замало інлаєрів: " 
                      << initResult.points3D.size() << ")." << std::endl;
            return false;
        }

        cv::Mat R0 = cv::Mat::eye(3,3,CV_64F);
        cv::Mat t0 = cv::Mat::zeros(3,1,CV_64F);
        m_map.addCameraPose(frame0.id, CameraPose(R0, t0));
        m_map.addCameraPose(frame1.id, CameraPose(initResult.R, initResult.t));

        // Заповнюємо карту початковими тріангульованими 3D-точками
        for (size_t i = 0; i < initResult.points3D.size(); ++i) {
            int matchIdx = initResult.inlierMatchIndices[i];
            int kpIdx0 = initialMatches[matchIdx].queryIdx;
            int kpIdx1 = initialMatches[matchIdx].trainIdx;

            cv::Mat desc = frame0.descriptors.row(kpIdx0).clone();
            m_map.addPoint3D(initResult.points3D[i], desc, frame0.id, kpIdx0, frame1.id, kpIdx1);
        }

        std::cout << "[Пайплайн] Ініціалізація завершена! В карті: " << m_map.getMapPointCount() << " точок." << std::endl;

        PnpTracker pnpTracker(m_K);
        IncrementalTriangulator triangulator(m_K);

        Frame prevFrame = frame1;

        // Етап 3

        cv::Mat currColor;
        Frame currFrame;

        while (getNextValidFrame(currColor, currFrame, frameCounter++)){
            std::cout << "\n==========================================" << std::endl;
            std::cout << "[Пайплайн] Обробка Кадру #" << currFrame.id << "..." << std::endl;
            m_processedFrames[currFrame.id] = currFrame;

            auto matches = m_matcher->match(prevFrame.descriptors, currFrame.descriptors);
            std::cout << "[Пайплайн] Збігів із Кадром #" << prevFrame.id << ": " << matches.size() << std::endl;

            bool tracked = pnpTracker.trackAndRegisterFrame(prevFrame.id, currFrame, matches, m_map);
            if (!tracked){
                std::cerr << "[Увага] Пропуск кадру #" << currFrame.id << " через зрив трекінгу PnP." << std::endl;
                continue;
            }

            int newPoints = triangulator.triangulateNewPoints(prevFrame, currFrame, matches, m_map);

            std::cout << "[Пайплайн] Кадр #" << currFrame.id << " успішно інтегровано. "
                      << "Додано точок: " << newPoints << " | Всього в карті: " << m_map.getMapPointCount() << std::endl;

            prevFrame = currFrame;
        }

        // Етам 4
        std::cout << "\n==========================================" << std::endl;
        std::cout << "[Пайплайн] Запуск глобального Bundle Adjustment..." << std::endl;
        std::cout << "Оптимізація для " << m_map.getCameraPoseCount() << " камер та " 
                  << m_map.getMapPointCount() << " точок..." << std::endl;

        BundleAdjuster ba(m_K, 40, true);
        ba.optimize(m_map, m_processedFrames, true);

        std::cout << "[Пайплайн] Експорт оптимізованої хмари точок у: " << outputPlyPath << std::endl;
        PlyExporter::save(outputPlyPath, m_map.getAllMapPoints());

        std::cout << "[Пайплайн] Роботу успішно завершено!" << std::endl;
        return true;
    } 

    const GlobalMap& getMap() const { return m_map; }
};