#pragma once

#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <set>
#include <tuple>
#include <fstream>
#include <cstdint>
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
#include "DirectoryFrameProvider.hpp"
#include "TwoViewGeometryEstimator.hpp"
#include "GlobalMap.hpp"
#include "FeatureTracks.hpp"
#include "PnpTracker.hpp"
#include "IncrementalTriangulator.hpp"
#include "BundleAdjuster.hpp"
#include "PlyExporter.hpp"
#include "DenseReconstructor.hpp"
#include "PointCloudFilter.hpp"
#include "MeshReconstructor.hpp"
#include "GeometryUtils.hpp"

struct SfMPipelineOptions {
    int matchWindow = 10;                // Зіставляти кадри, що відстоять не більше ніж на N позицій (0 = всі пари)
    int maxInitPairGap = 3;              // Початкова пара шукається серед кадрів з різницею індексів <= N
    int minInitMatches = 80;
    int minInitPoints = 50;
    double minInitMedianAngleDeg = 2.0;  // Менший кут — базис замалий, глибини початкової пари ненадійні
    double maxReprojectionErrorPx = 4.0;
    double minTriangulationAngleDeg = 1.5;
    double denseScale = 0.7;             // Роздільність щільної стадії відносно роздільності SfM
    double fusionTolerance = 0.015;      // Допустима відносна розбіжність глибин між видами
    int fusionMinConsistent = 1;         // Скільки ІНШИХ карт глибин мають підтвердити точку
    int meshStep = 2;                    // Крок сітки поверхні в пікселях карти глибин (1 = максимальна деталізація)
};

class SfMPipeline{
private:
    std::unique_ptr<IFrameProvider> m_provider;
    std::unique_ptr<IFramePreprocessor> m_preprocessor;
    std::unique_ptr<IFeatureExtractor> m_extractor;
    std::unique_ptr<IFeatureMatcher> m_matcher;

    cv::Mat m_K;
    SfMPipelineOptions m_options;

    std::unordered_map<int, Frame> m_processedFrames;
    std::unordered_map<int, std::string> m_frameNames;
    std::vector<int> m_frameIds;

    /**
     * Окрема реконструкція. Якщо фото містять кілька не пов'язаних між собою сцен
     * (наприклад, два різні будинки), кожна з них стає окремою моделлю зі своїм масштабом.
     */
    struct Reconstruction {
        GlobalMap map;
        int anchorFrameId = -1;   // Кадр з позою R = I, t = 0
        int scaleFrameId = -1;    // Другий кадр початкової пари: задає масштаб (|t| = 1)
    };
    std::vector<Reconstruction> m_models;

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

    std::string frameLabel(int frameId) const {
        auto it = m_frameNames.find(frameId);
        std::string label = "#" + std::to_string(frameId);
        if (it != m_frameNames.end() && !it->second.empty()) label += " (" + it->second + ")";
        return label;
    }

    void loadFrames() {
        auto* directoryProvider = dynamic_cast<DirectoryFrameProvider*>(m_provider.get());
        int frameCounter = 0;
        cv::Mat color;
        Frame frame;
        while (getNextValidFrame(color, frame, frameCounter)) {
            if (directoryProvider) m_frameNames[frame.id] = directoryProvider->currentFrameName();
            m_processedFrames[frame.id] = frame;
            m_frameIds.push_back(frame.id);
            std::cout << "[Пайплайн] Кадр " << frameLabel(frame.id) << ": " << frame.keypoints.size()
                      << " ключових точок" << std::endl;
            frameCounter++;
            frame = Frame();
        }
    }

    double medianTriangulationAngle(const GlobalMap& map, int a, int b) const {
        cv::Vec3d Ca = geom::cameraCenter(map.getCameraPose(a));
        cv::Vec3d Cb = geom::cameraCenter(map.getCameraPose(b));
        std::vector<double> angles;
        for (const auto& [pointId, pt] : map.getMapPoints()) {
            angles.push_back(geom::rayAngleDeg(geom::toVec(pt.position), Ca, Cb));
        }
        if (angles.empty()) return 0.0;
        std::nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
        return angles[angles.size() / 2];
    }

    void bundleAdjust(Reconstruction& model, int iterations) {
        BundleAdjuster ba(m_K, iterations, true);
        ba.optimize(model.map, m_processedFrames, model.anchorFrameId, model.scaleFrameId);

        double rms = 0.0;
        int removed = ba.removeOutliers(model.map, m_processedFrames,
                                        m_options.maxReprojectionErrorPx, m_options.minTriangulationAngleDeg, &rms);
        std::cout << "[BA] Камер: " << model.map.getCameraPoseCount() << ", точок: " << model.map.getMapPointCount()
                  << ", RMS репроєкції: " << rms << " px, видалено спостережень-викидів: " << removed << std::endl;
    }

    bool tryInitialize(int a, int b, const FeatureTracks& tracks, Reconstruction& outModel) {
        auto matches = tracks.matchesBetween(a, b);
        if (static_cast<int>(matches.size()) < m_options.minInitMatches) return false;

        TwoViewGeometryEstimator initializer(m_K);
        auto initResult = initializer.reconstruct(m_processedFrames.at(a), m_processedFrames.at(b), matches);
        if (initResult.R.empty() || initResult.t.empty()) return false;

        Reconstruction model;
        model.anchorFrameId = a;
        model.scaleFrameId = b;
        model.map.addCameraPose(a, CameraPose(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)));
        model.map.addCameraPose(b, CameraPose(initResult.R.clone(), initResult.t.clone()));

        IncrementalTriangulator triangulator(m_K, m_options.minTriangulationAngleDeg, m_options.maxReprojectionErrorPx);
        int points = triangulator.triangulateTracks(tracks, m_processedFrames, model.map);
        double angle = medianTriangulationAngle(model.map, a, b);

        std::cout << "[Ініціалізація] Пара " << frameLabel(a) << " <-> " << frameLabel(b) << ": "
                  << matches.size() << " збігів, " << points << " точок, медіанний кут " << angle << "°" << std::endl;

        if (points < m_options.minInitPoints || angle < m_options.minInitMedianAngleDeg) return false;

        outModel = std::move(model);
        return true;
    }

    // Нарощує модель кадрами, доки хоч один з unregistered вдається зареєструвати
    void growModel(Reconstruction& model, const FeatureTracks& tracks, const std::set<int>& unregistered) {
        PnpTracker pnpTracker(m_K, cv::Mat(), 15, 0.3, m_options.maxReprojectionErrorPx);
        IncrementalTriangulator triangulator(m_K, m_options.minTriangulationAngleDeg, m_options.maxReprojectionErrorPx);

        while (true) {
            // Кандидати впорядковані за кількістю 2D-3D відповідностей (найнадійніші — першими)
            std::vector<std::pair<size_t, int>> candidates;
            for (int frameId : unregistered) {
                if (model.map.hasCameraPose(frameId)) continue;
                size_t n = pnpTracker.countCorrespondences(m_processedFrames.at(frameId), tracks, model.map);
                if (n >= 5) candidates.emplace_back(n, frameId);
            }
            std::sort(candidates.rbegin(), candidates.rend());

            bool registered = false;
            for (const auto& [count, frameId] : candidates) {
                std::cout << "\n==========================================" << std::endl;
                std::cout << "[Пайплайн] Реєстрація кадру " << frameLabel(frameId) << "..." << std::endl;
                if (!pnpTracker.registerFrame(m_processedFrames.at(frameId), m_processedFrames, tracks, model.map)) continue;

                int newPoints = triangulator.triangulateTracks(tracks, m_processedFrames, model.map, frameId);
                std::cout << "[Пайплайн] Кадр " << frameLabel(frameId) << " інтегровано. Додано точок: " << newPoints << std::endl;
                bundleAdjust(model, 25);
                registered = true;
                break;
            }
            if (!registered) break;
        }
    }

    void exportModel(const Reconstruction& model, const std::string& prefix) {
        PlyExporter::save(prefix + "_sparse.ply", model.map.getAllMapPoints(), 1e6);
        std::cout << "[Пайплайн] Розріджена хмара: " << prefix << "_sparse.ply ("
                  << model.map.getMapPointCount() << " точок)" << std::endl;

        // Пози камер (World-to-Camera: X_c = R * X_w + t) — для перевірки та подальшої обробки
        std::ofstream cams(prefix + "_cameras.txt");
        cams << "# frame_id file R(9, по рядках) t(3) center(3)\n";
        for (int frameId : model.map.getRegisteredFrameIds()) {
            const CameraPose& pose = model.map.getCameraPose(frameId);
            cv::Matx33d R = geom::rotation(pose);
            cv::Vec3d t = geom::translation(pose), C = geom::cameraCenter(pose);
            auto nameIt = m_frameNames.find(frameId);
            cams << frameId << " " << (nameIt != m_frameNames.end() ? nameIt->second : "-");
            for (int i = 0; i < 9; ++i) cams << " " << R.val[i];
            cams << " " << t[0] << " " << t[1] << " " << t[2] << " " << C[0] << " " << C[1] << " " << C[2] << "\n";
        }
    }

    /**
     * Щільна хмара однієї моделі: карти глибин зі стерео-пар -> злиття з перевіркою узгодженості.
     * @return кількість точок у фінальній хмарі
     */
    size_t densify(const Reconstruction& model, const std::string& prefix) {
        std::cout << "\n==========================================" << std::endl;
        std::cout << "[Пайплайн] Генерація щільної хмари точок (Dense MVS) для " << prefix << "..." << std::endl;

        DenseReconstructor denseEngine(m_K, m_options.denseScale);
        std::vector<DepthMap> depthMaps;
        std::set<std::pair<int, int>> processedPairs;

        for (int refId : model.map.getRegisteredFrameIds()) {
            const Frame& refFrame = m_processedFrames.at(refId);
            cv::Size denseSize = denseEngine.colorImage(refFrame).size();

            for (const auto& partner : denseEngine.selectStereoPartners(refId, model.map, denseSize)) {
                std::pair<int, int> key(std::min(refId, partner.frameId), std::max(refId, partner.frameId));
                if (!processedPairs.insert(key).second) continue;

                std::cout << "[Dense MVS] Пара " << frameLabel(refId) << " - " << frameLabel(partner.frameId)
                          << ": кут " << partner.angleDeg << "°, спільних точок " << partner.sharedPoints << std::endl;
                for (auto& dm : denseEngine.computeDepthMaps(refFrame, m_processedFrames.at(partner.frameId), model.map)) {
                    if (cv::countNonZero(dm.depth) > 1000) depthMaps.push_back(std::move(dm));
                }
            }
        }

        if (depthMaps.size() < 2) {
            std::cerr << "[Dense MVS] Замало карт глибин (" << depthMaps.size()
                      << ") для перевірки узгодженості — щільну хмару не побудовано." << std::endl;
            return 0;
        }

        std::unordered_map<int, cv::Mat> colorImages;
        for (int frameId : model.map.getRegisteredFrameIds()) {
            colorImages[frameId] = denseEngine.colorImage(m_processedFrames.at(frameId));
        }

        std::vector<DensePoint3D> rawCloud;
        std::vector<cv::Mat> fusedDepths;
        auto fused = PointCloudFilter::fuseDepthMaps(depthMaps, model.map.getCameraPoses(), denseEngine.getCameraMatrix(),
                                                     colorImages, m_options.fusionMinConsistent,
                                                     m_options.fusionTolerance, &rawCloud, &fusedDepths);
        PlyExporter::saveColored(prefix + "_dense_raw.ply", rawCloud);

        if (fused.empty()) return 0;

        // Суцільна поверхня з узгоджених карт глибин
        std::cout << "\n[Пайплайн] Побудова поверхні (mesh) для " << prefix << "..." << std::endl;
        MeshReconstructor meshBuilder(m_options.meshStep);
        ColoredMesh mesh = meshBuilder.buildFromDepthMaps(fusedDepths, depthMaps, model.map.getCameraPoses(),
                                                          denseEngine.getCameraMatrix(), colorImages);
        if (!mesh.faces.empty()) MeshReconstructor::savePlyMesh(prefix + "_mesh.ply", mesh);

        // Проріджування: воксель ~ 1 піксель на медіанній глибині сцени (прибирає дублікати з різних карт)
        std::vector<double> depths;
        for (const auto& dm : depthMaps) {
            for (int v = 0; v < dm.depth.rows; v += 8)
                for (int u = 0; u < dm.depth.cols; u += 8)
                    if (dm.depth.at<float>(v, u) > 0) depths.push_back(dm.depth.at<float>(v, u));
        }
        if (depths.empty()) depths.push_back(1.0);
        std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
        double voxelSize = depths[depths.size() / 2] / denseEngine.getCameraMatrix().at<double>(0, 0);

        auto thinned = PointCloudFilter::voxelFuse(fused, voxelSize, 1);
        PlyExporter::saveColored(prefix + "_dense.ply", thinned);
        return thinned.size();
    }

public:
    SfMPipeline(
        std::unique_ptr<IFrameProvider> provider,
        std::unique_ptr<IFramePreprocessor> preprocessor,
        std::unique_ptr<IFeatureExtractor> extractor,
        std::unique_ptr<IFeatureMatcher> matcher,
        const cv::Mat& cameraMatrix,
        const SfMPipelineOptions& options = SfMPipelineOptions())
        : m_provider(std::move(provider)),
          m_preprocessor(std::move(preprocessor)),
          m_extractor(std::move(extractor)),
          m_matcher(std::move(matcher)),
          m_options(options)
    {
        cameraMatrix.convertTo(m_K, CV_64F);
    }

    /**
     * @param outputPrefix файли моделей: <prefix>_model<i>_sparse.ply / _dense.ply / _dense_raw.ply / _mesh.ply.
     *                     Найбільша модель додатково копіюється у final_sparse_cloud.ply / final_dense_cloud.ply / final_mesh.ply.
     */
    bool run(const std::string& outputPrefix = "reconstruction") {
        // Етап 1: кадри та ознаки
        loadFrames();
        if (m_frameIds.size() < 2) {
            std::cerr << "[Помилка] Потрібно щонайменше 2 придатні кадри (знайдено " << m_frameIds.size() << ")." << std::endl;
            return false;
        }

        // Етап 2: збіги між усіма парами + треки
        std::cout << "\n==========================================" << std::endl;
        TrackBuilder trackBuilder(m_options.matchWindow, 30, 2.0);
        FeatureTracks tracks = trackBuilder.build(m_frameIds, m_processedFrames, *m_matcher);

        // Етап 3: інкрементальний SfM. Кадри, які не вдалось приєднати, починають нову модель
        std::set<int> unregistered(m_frameIds.begin(), m_frameIds.end());
        while (unregistered.size() >= 2) {
            std::vector<std::tuple<int, size_t, int, int>> initCandidates; // (позиція a, -збіги, a, b)
            for (size_t i = 0; i < m_frameIds.size(); ++i) {
                for (size_t j = i + 1; j < m_frameIds.size() && static_cast<int>(j - i) <= m_options.maxInitPairGap; ++j) {
                    int a = m_frameIds[i], b = m_frameIds[j];
                    if (!unregistered.count(a) || !unregistered.count(b)) continue;
                    size_t n = tracks.matchCount(a, b);
                    if (static_cast<int>(n) >= m_options.minInitMatches) initCandidates.emplace_back(static_cast<int>(i), SIZE_MAX - n, a, b);
                }
            }
            std::sort(initCandidates.begin(), initCandidates.end());

            Reconstruction model;
            bool initialized = false;
            std::cout << "\n==========================================" << std::endl;
            for (const auto& [pos, negMatches, a, b] : initCandidates) {
                if (tryInitialize(a, b, tracks, model)) { initialized = true; break; }
            }
            if (!initialized) break;

            bundleAdjust(model, 25);
            growModel(model, tracks, unregistered);

            std::cout << "\n==========================================" << std::endl;
            std::cout << "[Пайплайн] Фінальний Bundle Adjustment моделі #" << m_models.size() << "..." << std::endl;
            bundleAdjust(model, 100);

            for (int frameId : model.map.getRegisteredFrameIds()) unregistered.erase(frameId);
            m_models.push_back(std::move(model));
        }

        if (m_models.empty()) {
            std::cerr << "[Помилка] Не знайдено жодної пари кадрів, придатної для ініціалізації." << std::endl;
            return false;
        }

        std::cout << "\n==========================================" << std::endl;
        std::cout << "[Пайплайн] Побудовано моделей: " << m_models.size() << std::endl;
        for (size_t i = 0; i < m_models.size(); ++i) {
            std::cout << "   Модель " << i << ": кадри";
            for (int id : m_models[i].map.getRegisteredFrameIds()) std::cout << " " << frameLabel(id);
            std::cout << " | точок: " << m_models[i].map.getMapPointCount() << std::endl;
        }
        if (!unregistered.empty()) {
            std::cout << "   Не вдалося приєднати кадри:";
            for (int id : unregistered) std::cout << " " << frameLabel(id);
            std::cout << std::endl;
        }

        // Етап 4: щільна хмара для кожної моделі
        size_t largestModel = 0;
        for (size_t i = 0; i < m_models.size(); ++i) {
            if (m_models[i].map.getCameraPoseCount() > m_models[largestModel].map.getCameraPoseCount()) largestModel = i;
        }

        for (size_t i = 0; i < m_models.size(); ++i) {
            std::string prefix = outputPrefix + "_model" + std::to_string(i);
            exportModel(m_models[i], prefix);
            densify(m_models[i], prefix);
        }

        // Найбільша модель — під звичними іменами
        PlyExporter::save("final_sparse_cloud.ply", m_models[largestModel].map.getAllMapPoints(), 1e6);
        std::string largestPrefix = outputPrefix + "_model" + std::to_string(largestModel);
        std::error_code ec;
        fs::copy_file(largestPrefix + "_dense.ply", "final_dense_cloud.ply", fs::copy_options::overwrite_existing, ec);
        fs::copy_file(largestPrefix + "_dense_raw.ply", "final_dense_cloud_raw.ply", fs::copy_options::overwrite_existing, ec);
        fs::copy_file(largestPrefix + "_mesh.ply", "final_mesh.ply", fs::copy_options::overwrite_existing, ec);

        std::cout << "[Пайплайн] Роботу успішно завершено!" << std::endl;
        return true;
    }

    const std::vector<Reconstruction>& getModels() const { return m_models; }
};
