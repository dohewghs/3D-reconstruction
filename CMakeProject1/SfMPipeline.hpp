#pragma once

#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
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
#include "DenseReconstructor.hpp"
#include "PointCloudFilter.hpp"

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
            outFrame.imageColor = raw;
            m_extractor->extract(outFrame);
            return true;
        }

        return false;
    }

    struct DepthRange {
        double minDepth;
        double medianDepth;
        double maxDepth;
    };

    /**
     * Діапазон глибин сцени для кожного кадру за розрідженою хмарою (після BA).
     * Масштаб монокулярної реконструкції довільний (базис першої пари = 1), тому
     * захардкоджені "2..35 метрів" не мають сенсу — беремо перцентилі реальних глибин.
     */
    std::unordered_map<int, DepthRange> computeFrameDepthRanges(size_t minPoints = 20) const {
        std::unordered_map<int, std::vector<double>> depthsPerFrame;

        for (const auto& [pointId, mapPt] : m_map.getMapPoints()) {
            for (const auto& [frameId, kpIdx] : mapPt.observations) {
                if (!m_map.hasCameraPose(frameId)) continue;

                const CameraPose& pose = m_map.getCameraPose(frameId);
                cv::Matx33d R = cv::Mat(pose.rotation);
                cv::Vec3d t = cv::Mat(pose.translation);
                cv::Vec3d Xc = R * cv::Vec3d(mapPt.position.x, mapPt.position.y, mapPt.position.z) + t;

                if (std::isfinite(Xc[2]) && Xc[2] > 0.0) {
                    depthsPerFrame[frameId].push_back(Xc[2]);
                }
            }
        }

        std::unordered_map<int, DepthRange> ranges;
        for (auto& [frameId, depths] : depthsPerFrame) {
            if (depths.size() < minPoints) continue;

            std::sort(depths.begin(), depths.end());
            auto percentile = [&](double q) {
                return depths[static_cast<size_t>(q * (depths.size() - 1))];
            };

            // Запас: фасад може виходити за межі розріджених точок
            ranges[frameId] = DepthRange{
                percentile(0.05) * 0.7,
                percentile(0.50),
                percentile(0.95) * 1.5
            };
        }
        return ranges;
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

        // Етап 5: Dense Stereo Reconstruction
        std::cout << "\n==========================================" << std::endl;
        std::cout << "[Пайплайн] Генерація щільної хмари точок (Dense MVS)..." << std::endl;

        std::vector<int> registeredIds;
        for (const auto& [frameId, frame] : m_processedFrames) {
            if (m_map.hasCameraPose(frameId)) registeredIds.push_back(frameId);
        }
        std::sort(registeredIds.begin(), registeredIds.end());

        auto depthRanges = computeFrameDepthRanges();

        DenseReconstructor denseEngine(m_K);
        std::vector<DensePoint3D> totalDenseCloud;
        std::vector<double> medianDepths;
        const size_t maxLookahead = 30;
        int pairCount = 0;

        for (size_t a = 0; a < registeredIds.size(); ++a) {
            int refId = registeredIds[a];
            auto rangeIt = depthRanges.find(refId);
            if (rangeIt == depthRanges.end()) continue;
            const DepthRange& range = rangeIt->second;

            // Кандидати — наступні зареєстровані кадри (не лише сусідній: у нього замалий базис)
            std::vector<int> candidates(
                registeredIds.begin() + a + 1,
                registeredIds.begin() + std::min(registeredIds.size(), a + 1 + maxLookahead));

            int partnerId = denseEngine.selectStereoPartner(
                refId, candidates, m_map, range.medianDepth);
            if (partnerId < 0) continue;

            auto pairPoints = denseEngine.reconstructDensePair(
                m_processedFrames.at(refId), m_processedFrames.at(partnerId),
                m_map.getCameraPose(refId), m_map.getCameraPose(partnerId),
                range.minDepth, range.maxDepth,
                pairCount
            );
            if (pairPoints.empty()) continue;

            pairCount++;
            medianDepths.push_back(range.medianDepth);
            totalDenseCloud.insert(totalDenseCloud.end(), pairPoints.begin(), pairPoints.end());
        }

        if (totalDenseCloud.empty()) {
            std::cerr << "[Dense MVS] Не вдалося побудувати жодної стерео-пари." << std::endl;
        } else {
            PlyExporter::saveColored("final_dense_cloud_raw.ply", totalDenseCloud);

            // Розмір вокселя ~ 2 пікселі на типовій глибині сцени
            std::nth_element(medianDepths.begin(), medianDepths.begin() + medianDepths.size() / 2, medianDepths.end());
            double sceneDepth = medianDepths[medianDepths.size() / 2];
            double voxelSize = 2.0 * sceneDepth / m_K.at<double>(0, 0);
            int minSources = (pairCount >= 2) ? 2 : 1;

            auto fusedCloud = PointCloudFilter::voxelFuse(totalDenseCloud, voxelSize, minSources);
            PlyExporter::saveColored("final_dense_cloud.ply", fusedCloud);
        }

        std::cout << "[Пайплайн] Роботу успішно завершено!" << std::endl;
        return true;
    } 

    const GlobalMap& getMap() const { return m_map; }
};