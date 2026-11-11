#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "GlobalMap.hpp"
#include "Types.hpp"
#include "FeatureTracks.hpp"
#include "GeometryUtils.hpp"
#include "TwoViewGeometryEstimator.hpp"

class PnpTracker {
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    int minInliersForValidPose;             // Мінімальна кількість інлаєрів для вважання оцінки PnP валідною
    double minInlierRatio;                  // Мінімальна частка інлаєрів серед 2D-3D кандидатів
    double reprojectionErrorThreshold;      // Поріг для вважання точки інлаєром
    double confidence;                      // Довіра для RANSAC
    int maxIterations;                      // Максимальна кількість ітерацій для RANSAC

    struct Correspondences {
        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;
        std::vector<int> pointIds;
        std::vector<int> keypointIndices;
    };

    // 2D-3D відповідності через треки: ключова точка кадру -> трек -> 3D-точка карти
    Correspondences collectCorrespondences(const Frame& frame, const FeatureTracks& tracks, const GlobalMap& map) const {
        Correspondences c;
        for (int kpIdx = 0; kpIdx < static_cast<int>(frame.keypoints.size()); ++kpIdx) {
            int trackId = tracks.trackOf(frame.id, kpIdx);
            if (trackId < 0) continue;
            int pointId = map.getPointIdForTrack(trackId);
            if (pointId < 0) continue;

            const cv::Point3d& X = map.getMapPoint(pointId).position;
            c.points3D.emplace_back(static_cast<float>(X.x), static_cast<float>(X.y), static_cast<float>(X.z));
            c.points2D.push_back(frame.keypoints[kpIdx].pt);
            c.pointIds.push_back(pointId);
            c.keypointIndices.push_back(kpIdx);
        }
        return c;
    }

    void addObservations(GlobalMap& map, int frameId, const Correspondences& c, const std::vector<int>& inliers) const {
        for (int idx : inliers) {
            if (map.getPointIdForFrameKeypoint(frameId, c.keypointIndices[idx]) != -1) continue;
            map.addObservation(c.pointIds[idx], frameId, c.keypointIndices[idx]);
        }
    }

    /**
     * Резервна реєстрація для кадрів з малою кількістю 2D-3D відповідностей (< 15).
     *
     * Поворот і НАПРЯМОК руху беремо з есенціальної матриці відносно найкраще зв'язаного
     * зареєстрованого кадру (там сотні 2D-2D збігів), невідомим лишається лише масштаб —
     * 1 параметр, який надійно оцінюється навіть з 5-10 точок карти (RANSAC по одній точці).
     */
    bool registerByRelativePose(const Frame& frame, const std::unordered_map<int, Frame>& frames,
                                const FeatureTracks& tracks, GlobalMap& map, const Correspondences& c) const
    {
        if (c.points3D.size() < 5) return false;

        int refId = -1;
        size_t bestMatches = 0;
        for (const auto& [frameId, pose] : map.getCameraPoses()) {
            size_t n = tracks.matchCount(frameId, frame.id);
            if (n > bestMatches) { bestMatches = n; refId = frameId; }
        }
        if (refId < 0 || bestMatches < 80) return false;

        TwoViewGeometryEstimator estimator(cameraMatrix);
        auto rel = estimator.reconstruct(frames.at(refId), frame, tracks.matchesBetween(refId, frame.id));
        if (rel.R.empty() || rel.t.empty()) return false;

        const CameraPose& refPose = map.getCameraPose(refId);
        cv::Matx33d Rrel(rel.R);
        cv::Vec3d dir(rel.t.reshape(1, 3));
        cv::Matx33d Rc = Rrel * geom::rotation(refPose);
        cv::Vec3d t0 = Rrel * geom::translation(refPose);   // t_c = t0 + s * dir

        cv::Matx33d K(cameraMatrix);
        std::vector<double> scales;
        for (size_t i = 0; i < c.points3D.size(); ++i) {
            cv::Vec3d a = Rc * cv::Vec3d(c.points3D[i].x, c.points3D[i].y, c.points3D[i].z) + t0;
            double xn = (c.points2D[i].x - K(0, 2)) / K(0, 0);
            double yn = (c.points2D[i].y - K(1, 2)) / K(1, 1);
            // (a + s*dir) має проєктуватися в (xn, yn): лінійно по s, розв'язок МНК
            cv::Vec2d A(a[0] - xn * a[2], a[1] - yn * a[2]);
            cv::Vec2d D(dir[0] - xn * dir[2], dir[1] - yn * dir[2]);
            double dd = D.dot(D);
            if (dd < 1e-12) continue;
            double s = -A.dot(D) / dd;
            if (s > 0) scales.push_back(s);
        }

        double bestScale = 0.0;
        std::vector<int> bestInliers;
        for (double s : scales) {
            CameraPose candidate(cv::Mat(Rc).clone(), cv::Mat(t0 + s * dir).clone());
            std::vector<int> inliers;
            for (size_t i = 0; i < c.points3D.size(); ++i) {
                cv::Point2d uv;
                cv::Vec3d X(c.points3D[i].x, c.points3D[i].y, c.points3D[i].z);
                if (geom::project(K, candidate, X, uv) && cv::norm(uv - cv::Point2d(c.points2D[i])) < reprojectionErrorThreshold) {
                    inliers.push_back(static_cast<int>(i));
                }
            }
            if (inliers.size() > bestInliers.size()) { bestInliers = std::move(inliers); bestScale = s; }
        }

        if (bestInliers.size() < 5 || bestInliers.size() < 0.5 * c.points3D.size()) {
            std::cerr << "[PnP] Відносна реєстрація кадру #" << frame.id << " не вдалася (інлаєрів: "
                      << bestInliers.size() << " / " << c.points3D.size() << ")" << std::endl;
            return false;
        }

        map.addCameraPose(frame.id, CameraPose(cv::Mat(Rc).clone(), cv::Mat(t0 + bestScale * dir).clone()));
        addObservations(map, frame.id, c, bestInliers);

        std::cout << "[PnP] Кадр #" << frame.id << " зареєстровано через відносну позу до кадру #" << refId
                  << " (масштаб підтвердили " << bestInliers.size() << " / " << c.points3D.size() << " точок)" << std::endl;
        return true;
    }

public:
    explicit PnpTracker(const cv::Mat& cameraMatrix,
                        const cv::Mat& distCoeffs = cv::Mat(),
                        int minInliersForValidPose = 15,
                        double minInlierRatio = 0.3,
                        double reprojectionErrorThreshold = 4.0,
                        double confidence = 0.999,
                        int maxIterations = 2000) :
        minInliersForValidPose(minInliersForValidPose),
        minInlierRatio(minInlierRatio),
        reprojectionErrorThreshold(reprojectionErrorThreshold),
        confidence(confidence),
        maxIterations(maxIterations)
    {
        cameraMatrix.convertTo(this->cameraMatrix, CV_64F);
        if (!distCoeffs.empty()) {
            distCoeffs.convertTo(this->distCoeffs, CV_64F);
        } else {
            this->distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // Assuming 4 distortion coefficients
        }
    };

    /**
     * Оцінює глобальну позу нового кадру через 2D-3D відповідності з картою
     */
    PnpTrackingResult estimatePose(const std::vector<cv::Point3f>& points3D,
                            const std::vector<cv::Point2f>& points2D) const
    {
        PnpTrackingResult result;

        if (static_cast<int>(points3D.size()) < this->minInliersForValidPose) {
            result.success = false;
            return result;
        }

        cv::Mat rvec, tvec;
        std::vector<int> inliers;

        // solvePnPRansac знаходить позу [R|t], яка проектує 3D точки світу на 2D пікселі камери:
        // s * [u, v, 1]^T = K * (R * [X, Y, Z]^T + t)
        bool pnpSuccess = cv::solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs,
                                             rvec, tvec, false, maxIterations,
                                             static_cast<float>(reprojectionErrorThreshold), confidence, inliers, cv::SOLVEPNP_SQPNP);

        result.inliersCount = static_cast<int>(inliers.size());
        if (!pnpSuccess || result.inliersCount < this->minInliersForValidPose ||
            result.inliersCount < minInlierRatio * points3D.size()) {
            result.success = false;
            return result;
        }

        // Уточнення по всіх інлаєрах (Левенберг-Марквардт)
        std::vector<cv::Point3f> in3D;
        std::vector<cv::Point2f> in2D;
        for (int idx : inliers) { in3D.push_back(points3D[idx]); in2D.push_back(points2D[idx]); }
        cv::solvePnPRefineLM(in3D, in2D, cameraMatrix, distCoeffs, rvec, tvec);

        cv::Rodrigues(rvec, result.rotation); // Перетворюємо вектор обертання в матрицю обертання
        result.rotation.convertTo(result.rotation, CV_64F);
        tvec.convertTo(result.translation, CV_64F);

        result.success = true;
        result.inlierPoints3D = std::move(inliers);

        return result;
    };

    /**
     * Реєструє кадр у карті: PnP за треками, а якщо відповідностей замало — через відносну позу.
     */
    bool registerFrame(const Frame& frame,
                       const std::unordered_map<int, Frame>& frames,
                       const FeatureTracks& tracks,
                       GlobalMap& map) const
    {
        Correspondences c = collectCorrespondences(frame, tracks, map);

        std::cout << "[PnP] 2D-3D кандидатів для кадру #" << frame.id << ": " << c.points3D.size() << std::endl;

        if (static_cast<int>(c.points3D.size()) < minInliersForValidPose) {
            return registerByRelativePose(frame, frames, tracks, map, c);
        }

        PnpTrackingResult pnpResult = estimatePose(c.points3D, c.points2D);
        if (!pnpResult.success) {
            std::cerr << "[PnP] Не вдалося визначити позу для кадру #" << frame.id
                      << " (інлаєрів: " << pnpResult.inliersCount << " / " << c.points3D.size() << ")" << std::endl;
            return false;
        }

        std::cout << "[PnP] Позу для кадру #" << frame.id
                  << " успішно знайдено! Інлаєрів: " << pnpResult.inliersCount
                  << " / " << c.points3D.size() << std::endl;

        map.addCameraPose(frame.id, CameraPose(pnpResult.rotation, pnpResult.translation));
        addObservations(map, frame.id, c, pnpResult.inlierPoints3D);
        return true;
    }

    // Кількість 2D-3D відповідностей — для вибору наступного кадру
    size_t countCorrespondences(const Frame& frame, const FeatureTracks& tracks, const GlobalMap& map) const {
        size_t n = 0;
        for (int kpIdx = 0; kpIdx < static_cast<int>(frame.keypoints.size()); ++kpIdx) {
            int trackId = tracks.trackOf(frame.id, kpIdx);
            if (trackId >= 0 && map.getPointIdForTrack(trackId) >= 0) n++;
        }
        return n;
    }
};
