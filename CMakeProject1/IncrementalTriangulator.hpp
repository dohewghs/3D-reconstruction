#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "GlobalMap.hpp"
#include "Types.hpp"
#include "FeatureTracks.hpp"
#include "GeometryUtils.hpp"

/**
 * Тріангуляція треків, які бачать щонайменше два зареєстровані кадри.
 *
 * Фільтри не залежать від масштабу сцени (у монокулярному SfM він довільний),
 * тому замість "глибина 0.1..50" використовуються кут тріангуляції та похибка репроєкції.
 */
class IncrementalTriangulator {
private:
    cv::Mat cameraMatrix;
    cv::Matx33d K;
    double minTriangulationAngleDeg;
    double maxReprojectionErrorPx;

    cv::Mat projectionMatrix(const CameraPose& pose) const {
        cv::Mat Rt = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat R, t;
        pose.rotation.convertTo(R, CV_64F);
        pose.translation.convertTo(t, CV_64F);
        R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
        t.reshape(1, 3).copyTo(Rt(cv::Rect(3, 0, 1, 3)));
        return cameraMatrix * Rt;
    }

    static bool triangulateTwoView(const cv::Mat& P1, const cv::Mat& P2,
                                   const cv::Point2f& p1, const cv::Point2f& p2, cv::Vec3d& outX)
    {
        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, std::vector<cv::Point2f>{p1}, std::vector<cv::Point2f>{p2}, points4D);
        points4D.convertTo(points4D, CV_64F);
        double w = points4D.at<double>(3, 0);
        if (std::abs(w) < 1e-12) return false;
        outX = cv::Vec3d(points4D.at<double>(0, 0) / w, points4D.at<double>(1, 0) / w, points4D.at<double>(2, 0) / w);
        return std::isfinite(outX[0]) && std::isfinite(outX[1]) && std::isfinite(outX[2]);
    }

public:
    explicit IncrementalTriangulator(
        const cv::Mat& cameraMatrix,
        double minTriangulationAngleDeg = 1.5,
        double maxReprojectionErrorPx = 4.0)
        : minTriangulationAngleDeg(minTriangulationAngleDeg),
          maxReprojectionErrorPx(maxReprojectionErrorPx)
    {
        cameraMatrix.convertTo(this->cameraMatrix, CV_64F);
        K = cv::Matx33d(this->cameraMatrix);
    }

    double getMinAngleDeg() const { return minTriangulationAngleDeg; }
    double getMaxReprojectionErrorPx() const { return maxReprojectionErrorPx; }

    /**
     * Тріангулює всі треки, що ще не мають 3D-точки і видимі у >= 2 зареєстрованих кадрах.
     * @param requiredFrameId якщо >= 0 — лише треки, які бачить цей (щойно доданий) кадр
     * @return кількість доданих точок
     */
    int triangulateTracks(const FeatureTracks& tracks,
                          const std::unordered_map<int, Frame>& frames,
                          GlobalMap& map,
                          int requiredFrameId = -1) const
    {
        std::unordered_map<int, cv::Mat> projections;
        std::unordered_map<int, cv::Vec3d> centers;
        for (const auto& [frameId, pose] : map.getCameraPoses()) {
            projections[frameId] = projectionMatrix(pose);
            centers[frameId] = geom::cameraCenter(pose);
        }

        int addedCount = 0;
        for (int trackId = 0; trackId < static_cast<int>(tracks.tracks.size()); ++trackId) {
            if (map.getPointIdForTrack(trackId) != -1) continue;

            const Track& track = tracks.tracks[trackId];
            if (requiredFrameId >= 0 && !track.observations.count(requiredFrameId)) continue;

            std::vector<int> registered;
            for (const auto& [frameId, kpIdx] : track.observations) {
                if (map.hasCameraPose(frameId)) registered.push_back(frameId);
            }
            if (registered.size() < 2) continue;

            // Пара кадрів з найбільшим кутом тріангуляції дає найстабільнішу точку
            double bestAngle = -1.0;
            cv::Vec3d bestX;
            for (size_t i = 0; i < registered.size(); ++i) {
                for (size_t j = i + 1; j < registered.size(); ++j) {
                    int a = registered[i], b = registered[j];
                    cv::Vec3d X;
                    if (!triangulateTwoView(projections[a], projections[b],
                                            frames.at(a).keypoints[track.observations.at(a)].pt,
                                            frames.at(b).keypoints[track.observations.at(b)].pt, X)) continue;
                    double angle = geom::rayAngleDeg(X, centers[a], centers[b]);
                    if (angle > bestAngle) { bestAngle = angle; bestX = X; }
                }
            }
            if (bestAngle < minTriangulationAngleDeg) continue;

            // Залишаємо лише ті спостереження, де точка перед камерою і добре репроєктується
            std::unordered_map<int, int> goodObservations;
            for (int frameId : registered) {
                cv::Point2d uv;
                if (!geom::project(K, map.getCameraPose(frameId), bestX, uv)) continue;
                int kpIdx = track.observations.at(frameId);
                cv::Point2f observed = frames.at(frameId).keypoints[kpIdx].pt;
                if (cv::norm(cv::Point2d(observed) - uv) < maxReprojectionErrorPx) {
                    goodObservations[frameId] = kpIdx;
                }
            }
            if (goodObservations.size() < 2) continue;

            int firstFrame = goodObservations.begin()->first;
            cv::Mat descriptor = frames.at(firstFrame).descriptors.row(goodObservations.begin()->second);
            map.addPoint3D(geom::toPoint(bestX), descriptor, goodObservations, trackId);
            addedCount++;
        }

        return addedCount;
    }
};
