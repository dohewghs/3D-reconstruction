#pragma once

#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

#include "Types.hpp"

/**
 * Дрібні геометричні функції, спільні для SfM та Dense етапів.
 * Пози у форматі World-to-Camera: X_c = R * X_w + t.
 */
namespace geom {

    inline cv::Matx33d rotation(const CameraPose& pose) {
        cv::Mat R;
        pose.rotation.convertTo(R, CV_64F);
        return cv::Matx33d(R);
    }

    inline cv::Vec3d translation(const CameraPose& pose) {
        cv::Mat t;
        pose.translation.convertTo(t, CV_64F);
        return cv::Vec3d(t.reshape(1, 3));
    }

    // Оптичний центр камери у світових координатах: C = -R^T * t
    inline cv::Vec3d cameraCenter(const CameraPose& pose) {
        return -(rotation(pose).t() * translation(pose));
    }

    inline cv::Vec3d toCamera(const CameraPose& pose, const cv::Vec3d& Xw) {
        return rotation(pose) * Xw + translation(pose);
    }

    // Проєкція світової точки у пікселі. Повертає false, якщо точка позаду камери.
    inline bool project(const cv::Matx33d& K, const CameraPose& pose, const cv::Vec3d& Xw, cv::Point2d& outUv) {
        cv::Vec3d Xc = toCamera(pose, Xw);
        if (!(Xc[2] > 1e-9)) return false;
        outUv.x = K(0, 0) * Xc[0] / Xc[2] + K(0, 2);
        outUv.y = K(1, 1) * Xc[1] / Xc[2] + K(1, 2);
        return true;
    }

    // Кут (у градусах) між променями від двох оптичних центрів до точки
    inline double rayAngleDeg(const cv::Vec3d& X, const cv::Vec3d& C1, const cv::Vec3d& C2) {
        cv::Vec3d a = X - C1, b = X - C2;
        double na = cv::norm(a), nb = cv::norm(b);
        if (na < 1e-12 || nb < 1e-12) return 0.0;
        double c = std::clamp(a.dot(b) / (na * nb), -1.0, 1.0);
        return std::acos(c) * 180.0 / CV_PI;
    }

    inline cv::Vec3d toVec(const cv::Point3d& p) { return cv::Vec3d(p.x, p.y, p.z); }
    inline cv::Point3d toPoint(const cv::Vec3d& v) { return cv::Point3d(v[0], v[1], v[2]); }
}
