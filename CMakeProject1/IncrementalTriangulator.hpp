#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "GlobalMap.hpp"
#include "Types.hpp"

class IncrementalTriangulator {
private:
    cv::Mat cameraMatrix;
    double minTriangulationAngleRad;
    double minDepth;
    double maxDepth;

    double calculateRayAngle(const cv::Point3d& point3D,
                             const cv::Mat& C1,
                             const cv::Mat& C2) const
    {
        cv::Point3d ray1(point3D.x - C1.at<double>(0),
                         point3D.y - C1.at<double>(1),
                         point3D.z - C1.at<double>(2));

        cv::Point3d ray2(point3D.x - C2.at<double>(0),
                         point3D.y - C2.at<double>(1),
                         point3D.z - C2.at<double>(2));

        double norm1 = std::sqrt(ray1.x * ray1.x + ray1.y * ray1.y + ray1.z * ray1.z);
        double norm2 = std::sqrt(ray2.x * ray2.x + ray2.y * ray2.y + ray2.z * ray2.z);

        if (norm1 < 1e-6 || norm2 < 1e-6) return 0.0;

        double dot = (ray1.x * ray2.x + ray1.y * ray2.y + ray1.z * ray2.z) / (norm1 * norm2);
        dot = std::max(-1.0, std::min(1.0, dot));

        return std::acos(dot);
    }

public:
    explicit IncrementalTriangulator(
        const cv::Mat& cameraMatrix,
        double minTriangulationAngleDeg = 0.5,
        double minDepth = 0.1,
        double maxDepth = 50.0)
        : minDepth(minDepth),
          maxDepth(maxDepth),
          minTriangulationAngleRad(minTriangulationAngleDeg * CV_PI / 180.0)
    {
        cameraMatrix.convertTo(this->cameraMatrix, CV_64F);
    }

    int triangulateNewPoints(
        const Frame& previousFrame,
        const Frame& currentFrame,
        const std::vector<cv::DMatch>& matches,
        GlobalMap& map) const
    {
        if (!map.hasCameraPose(previousFrame.id) || !map.hasCameraPose(currentFrame.id)) {
            std::cerr << "[Triangulator] Помилка: пози для кадру " << previousFrame.id 
                      << " або " << currentFrame.id << " відсутні в карті!" << std::endl;
            return 0;
        }
        
        const auto& posePrevious = map.getCameraPose(previousFrame.id);
        const auto& poseCurrent = map.getCameraPose(currentFrame.id);

        // Примусово конвертуємо R і t обох камер до CV_64F (double) для уникнення винятків
        cv::Mat R_prev, t_prev, R_curr, t_curr;
        posePrevious.rotation.convertTo(R_prev, CV_64F);
        posePrevious.translation.convertTo(t_prev, CV_64F);
        poseCurrent.rotation.convertTo(R_curr, CV_64F);
        poseCurrent.translation.convertTo(t_curr, CV_64F);

        // 1. Побудова матриць проєкції: P = K * [R | t]
        cv::Mat RtPrevious = cv::Mat::zeros(3, 4, CV_64F);
        R_prev.copyTo(RtPrevious(cv::Rect(0, 0, 3, 3)));
        t_prev.copyTo(RtPrevious(cv::Rect(3, 0, 1, 3)));
        cv::Mat P_previous = this->cameraMatrix * RtPrevious;

        cv::Mat RtCurrent = cv::Mat::zeros(3, 4, CV_64F);
        R_curr.copyTo(RtCurrent(cv::Rect(0, 0, 3, 3))); // Виправлено: беремо R_curr, а не posePrevious
        t_curr.copyTo(RtCurrent(cv::Rect(3, 0, 1, 3))); // Виправлено: беремо t_curr, а не posePrevious
        cv::Mat P_current = this->cameraMatrix * RtCurrent;

        // Оптичні центри камер у світових координатах: C = -R^T * t
        cv::Mat C_previous = -R_prev.t() * t_prev;
        cv::Mat C_current = -R_curr.t() * t_curr;

        // 2. Фільтрація: беремо тільки ті збіги, які ще НЕ існують у карті
        std::vector<cv::Point2f> pointsPrevious, pointsCurrent;
        std::vector<int> keypointIdxPrevious, keypointsIdxCurrent;

        pointsPrevious.reserve(matches.size());
        pointsCurrent.reserve(matches.size());
        keypointIdxPrevious.reserve(matches.size());
        keypointsIdxCurrent.reserve(matches.size());

        for (const auto& m : matches) {
            int kpIdx1 = m.queryIdx;
            int kpIdx2 = m.trainIdx;

            if (map.getPointIdForFrameKeypoint(previousFrame.id, kpIdx1) != -1 ||
                map.getPointIdForFrameKeypoint(currentFrame.id, kpIdx2) != -1)
            {
                continue;
            }
            
            pointsPrevious.push_back(previousFrame.keypoints[kpIdx1].pt);
            pointsCurrent.push_back(currentFrame.keypoints[kpIdx2].pt);
            keypointIdxPrevious.push_back(kpIdx1);
            keypointsIdxCurrent.push_back(kpIdx2);
        }

        if (pointsPrevious.empty()) {
            return 0;
        }

        // 3. Лінійна тріангуляція точок у 4D однорідних координатах
        cv::Mat points4D;
        cv::triangulatePoints(P_previous, P_current, pointsPrevious, pointsCurrent, points4D);

        int addedCount = 0;
        const bool isDouble = (points4D.type() == CV_64F || points4D.depth() == CV_64F);

        // 4. Валідація та додавання до GlobalMap
        for (int i = 0; i < points4D.cols; ++i) {
            double w = isDouble ? points4D.at<double>(3, i) : points4D.at<float>(3, i);
            if (std::abs(w) < 1e-6) {
                continue;
            }

            cv::Point3d ptWorld(
                (isDouble ? points4D.at<double>(0, i) : points4D.at<float>(0, i)) / w,
                (isDouble ? points4D.at<double>(1, i) : points4D.at<float>(1, i)) / w,
                (isDouble ? points4D.at<double>(2, i) : points4D.at<float>(2, i)) / w
            );

            // Перевіряємо глибину відносно обох камер: Z_cam = R_row2 * X_world + t_z
            double z1 = R_prev.row(2).dot(cv::Mat(ptWorld).t()) + t_prev.at<double>(2);
            double z2 = R_curr.row(2).dot(cv::Mat(ptWorld).t()) + t_curr.at<double>(2);

            if (z1 <= minDepth || z1 >= maxDepth || z2 <= minDepth || z2 >= maxDepth) {
                continue;
            }

            double angle = calculateRayAngle(ptWorld, C_previous, C_current);
            if (angle < this->minTriangulationAngleRad) {
                continue;
            }
            
            int kpIdx1 = keypointIdxPrevious[i];
            int kpIdx2 = keypointsIdxCurrent[i];

            cv::Mat descriptor = previousFrame.descriptors.row(kpIdx1).clone();

            map.addPoint3D(ptWorld, descriptor, previousFrame.id, kpIdx1, currentFrame.id, kpIdx2);
            addedCount++;
        }

        std::cout << "[Triangulator] Дотріангульовано нових точок: " << addedCount 
                  << " (кандидатів було: " << pointsPrevious.size() << ")" << std::endl;

        return addedCount;
    }
};