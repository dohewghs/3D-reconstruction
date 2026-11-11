#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/manifold.h>

#include "GlobalMap.hpp"
#include "Types.hpp"
#include "ReprojectionErrorCost.hpp"
#include "GeometryUtils.hpp"

class BundleAdjuster
{
private:
    double m_fx, m_fy, m_cx, m_cy;
    int maxIterations;
    bool verbose;
public:
    explicit BundleAdjuster(const cv::Mat& cameraMatrix, int maxIterations = 50, bool verbose = true)
        : maxIterations(maxIterations), verbose(verbose)
    {
        cv::Mat K;
        cameraMatrix.convertTo(K, CV_64F);

        m_fx = K.at<double>(0, 0);
        m_fy = K.at<double>(1, 1);
        m_cx = K.at<double>(0, 2);
        m_cy = K.at<double>(1, 2);
    }

    /**
     * Запускає глобальний Bundle Adjustment для всіх камер і точок у карті
     * @param map              посилання на глобальну карту
     * @param allFramesMap     доступ до 2D ключових точок кадру [frameId -> Frame]
     * @param anchorFrameId    кадр, що фіксується повністю (R = I, t = 0)
     * @param scaleFrameId     другий кадр початкової пари: його |t| фіксується
     *
     * Монокулярна реконструкція визначена з точністю до подібності (7 ступенів свободи).
     * Фіксація лише першої камери прибирає 6 з них, а масштаб лишається вільним — система
     * вироджена, і Ceres падав з "Unable to perform dense Cholesky factorization".
     * Тому на translation другої камери накладаємо SphereManifold (|t| = const).
     */
    bool optimize(GlobalMap& map,
                  const std::unordered_map<int, Frame>& allFramesMap,
                  int anchorFrameId,
                  int scaleFrameId = -1) {
        ceres::Problem problem;
        ceres::LossFunction* lossFunction = new ceres::HuberLoss(1.0); // Зрізає вплив аутлаєрів > 1 px

        // Камери: frameId -> double[6] (перші 3: rvec, наступні 3: tvec)
        std::unordered_map<int, std::vector<double>> cameraParams;
        // 3D-точки: pointId -> double[3] (X, Y, Z)
        std::unordered_map<int , std::vector<double>> pointParams;

        // отримуємо камери з карти
        for (const auto& [frameId, pose] : map.getCameraPoses())
        {
            cv::Mat rvec;
            cv::Rodrigues(pose.rotation, rvec);
            rvec.convertTo(rvec, CV_64F);
            cv::Vec3d t = geom::translation(pose);

            cameraParams[frameId] = {
                rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                t[0], t[1], t[2]
            };
        }

        // отримуємо точки з карти
        for(const auto& [pointId, mapPt] : map.getMapPoints())
        {
            pointParams[pointId] = { mapPt.position.x, mapPt.position.y, mapPt.position.z};
        }

        // додаємо спостереження
        int residualCount = 0;
        for (const auto& [pointId, mapPt] : map.getMapPoints()) {
            for (const auto& [frameId, kpIdx] : mapPt.observations) {
                if (cameraParams.find(frameId) == cameraParams.end())
                    continue;
                if (allFramesMap.find(frameId) == allFramesMap.end())
                    continue;

                const auto& frame = allFramesMap.at(frameId);
                const cv::Point2f& observedPixel = frame.keypoints[kpIdx].pt;

                ceres::CostFunction* costFunc = ReprojectionErrorCost::Create(
                    observedPixel.x, observedPixel.y,
                    m_fx, m_fy, m_cx, m_cy
                );

                problem.AddResidualBlock(
                    costFunc,
                    lossFunction,
                    cameraParams[frameId].data(),
                    cameraParams[frameId].data() + 3,
                    pointParams[pointId].data()
                );
                residualCount++;
            }
        }

        if (residualCount < 10) {
            std::cerr << "[BA] Недостатньо спостережень для запуску оптимізації!" << std::endl;
            return false;
        }

        // Усунення невизначеності калібрувальної системи (Gauge Freedom)
        auto anchorIt = cameraParams.find(anchorFrameId);
        if (anchorIt != cameraParams.end() && problem.HasParameterBlock(anchorIt->second.data())) {
            problem.SetParameterBlockConstant(anchorIt->second.data());
            problem.SetParameterBlockConstant(anchorIt->second.data() + 3);
        }
        auto scaleIt = cameraParams.find(scaleFrameId);
        if (scaleIt != cameraParams.end() && problem.HasParameterBlock(scaleIt->second.data() + 3)) {
            problem.SetManifold(scaleIt->second.data() + 3, new ceres::SphereManifold<3>());
        }

        // Налаштування розв'язувача
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = std::max(1u, std::thread::hardware_concurrency());

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if(verbose)
        {
            std::cout << "[BA] " << summary.BriefReport() << std::endl;
        }

        if (!summary.IsSolutionUsable()) return false;

        // Запис оптимізованих результатів назад у GlobalMap
        for (auto& [frameId, camData] : cameraParams) {
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << camData[0], camData[1], camData[2]);
            cv::Mat tvec = (cv::Mat_<double>(3, 1) << camData[3], camData[4], camData[5]);
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            map.addCameraPose(frameId, CameraPose(R, tvec));
        }

        for (auto& [pointId, ptData] : pointParams) {
            map.setPointPosition(pointId, cv::Point3d(ptData[0], ptData[1], ptData[2]));
        }

        return true;
    }

    /**
     * Видаляє спостереження з похибкою репроєкції > maxErrorPx, а потім точки,
     * у яких лишилось < 2 спостережень або замалий кут тріангуляції.
     * @return кількість видалених спостережень
     */
    int removeOutliers(GlobalMap& map,
                       const std::unordered_map<int, Frame>& allFramesMap,
                       double maxErrorPx,
                       double minAngleDeg,
                       double* outRmsError = nullptr) const
    {
        cv::Matx33d K(m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);

        std::vector<std::pair<int, int>> badObservations;
        std::vector<int> badPoints;
        double sqSum = 0.0;
        size_t count = 0;

        for (const auto& [pointId, mapPt] : map.getMapPoints()) {
            cv::Vec3d X = geom::toVec(mapPt.position);
            for (const auto& [frameId, kpIdx] : mapPt.observations) {
                cv::Point2d uv;
                double err = 1e9;
                if (geom::project(K, map.getCameraPose(frameId), X, uv)) {
                    err = cv::norm(uv - cv::Point2d(allFramesMap.at(frameId).keypoints[kpIdx].pt));
                    sqSum += err * err;
                    count++;
                }
                if (err > maxErrorPx) badObservations.emplace_back(pointId, frameId);
            }
        }

        for (const auto& [pointId, frameId] : badObservations) {
            map.removeObservation(pointId, frameId);
        }

        for (const auto& [pointId, mapPt] : map.getMapPoints()) {
            if (mapPt.observations.size() < 2) { badPoints.push_back(pointId); continue; }

            std::vector<cv::Vec3d> centers;
            for (const auto& [frameId, kpIdx] : mapPt.observations) {
                centers.push_back(geom::cameraCenter(map.getCameraPose(frameId)));
            }
            double maxAngle = 0.0;
            cv::Vec3d X = geom::toVec(mapPt.position);
            for (size_t i = 0; i < centers.size(); ++i)
                for (size_t j = i + 1; j < centers.size(); ++j)
                    maxAngle = std::max(maxAngle, geom::rayAngleDeg(X, centers[i], centers[j]));
            if (maxAngle < minAngleDeg) badPoints.push_back(pointId);
        }

        for (int pointId : badPoints) map.removePoint(pointId);

        if (outRmsError) *outRmsError = count ? std::sqrt(sqSum / count) : 0.0;
        return static_cast<int>(badObservations.size());
    }
};
