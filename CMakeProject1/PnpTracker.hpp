#pragma once

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "GlobalMap.hpp"
#include "Types.hpp"

class PnpTracker {
private:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    int minInliersForValidPose;             // Мінімальна кількість інлаєрів для вважання оцінки PnP валідною
    double reprojectionErrorThreshold;      // Поріг для вважання точки інлаєром
    double confidence;                      // Довіра для RANSAC
    int maxIterations;                      // Максимальна кількість ітерацій для RANSAC

public:
    explicit PnpTracker(const cv::Mat& cameraMatrix, 
                        const cv::Mat& distCoeffs = cv::Mat(),
                        int minInliersForValidPose = 6,
                        double reprojectionErrorThreshold = 7.0,
                        double confidence = 0.99,
                        int maxIterations = 200) :
        minInliersForValidPose(minInliersForValidPose),
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

        if (points3D.size() < this->minInliersForValidPose || points2D.size() < this->minInliersForValidPose) {
            result.success = false;
            return result;
        }

        cv::Mat rvec, tvec;
        std::vector<int> inliers;

        // solvePnPRansac знаходить позу [R|t], яка проектує 3D точки світу на 2D пікселі камери:
        // s * [u, v, 1]^T = K * (R * [X, Y, Z]^T + t)

        bool pnpSuccess = cv::solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs,
                                             rvec, tvec, false, maxIterations,
                                             reprojectionErrorThreshold, confidence, inliers, cv::SOLVEPNP_SQPNP);
        
        if (!pnpSuccess || inliers.size() < this->minInliersForValidPose) {
            result.success = false;
            return result;
        }

        cv::Rodrigues(rvec, result.rotation); // Перетворюємо вектор обертання в матрицю обертання
        result.rotation.convertTo(result.rotation, CV_64F);

        tvec.convertTo(result.translation, CV_64F);

        result.success = true;
        result.inliersCount = static_cast<int>(inliers.size());
        result.inlierPoints3D = std::move(inliers);

        return result;
    };

    /**
     * Комплексний метод: оцінює позу, оновлює спостереження в карті та зберігає нову камеру
     */
    bool trackAndRegisterFrame(
        int refFrameId,
        const Frame& currentFrame,
        const std::vector<cv::DMatch>& matches,
        GlobalMap& map) const
    {

        // 1. Отримуємо асоційовані пари 2D-3D за збігами з попереднім кадром
        std::vector<cv::Point3f> objectPoints3D;
        std::vector<cv::Point2f> imagePoints2D;
        std::vector<int> mapPointsIds;
        std::vector<int> currentKpIndices;

        map.find2D3DCorrespondences(
            refFrameId,
            currentFrame,
            matches,
            objectPoints3D,
            imagePoints2D,
            mapPointsIds,
            currentKpIndices
        );

        std::cout << "[PnP] Знайдено 2D-3D кандидатів між кадром #" << refFrameId 
                  << " та #" << currentFrame.id << ": " << objectPoints3D.size() << std::endl;

        // 2. Рахуємо глобальну позу

        PnpTrackingResult pnpResult = this->estimatePose(objectPoints3D, imagePoints2D);

        if (!pnpResult.success)
        {
            std::cerr << "[PnP] Не вдалося визначити позу для кадру #" << currentFrame.id 
                      << " (інлаєрів: " << pnpResult.inliersCount << ")" << std::endl;
            return false;
        }

        std::cout << "[PnP] Позу для кадру #" << currentFrame.id 
                  << " успішно знайдено! Інлаєрів: " << pnpResult.inliersCount 
                  << " / " << objectPoints3D.size() << std::endl;
        
        map.addCameraPose(currentFrame.id, CameraPose(pnpResult.rotation, pnpResult.translation));

        for (int idx : pnpResult.inlierPoints3D)
        {
            int pointId = mapPointsIds[idx];
            int kpIdx = currentKpIndices[idx]; // Точний індекс одразу за O(1)
            map.addObservation(pointId, currentFrame.id, kpIdx);
        }

        return true;
    }
};
