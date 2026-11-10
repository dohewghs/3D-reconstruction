#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>

#include "GlobalMap.hpp"
#include "Types.hpp"
#include "ReprojectionErrorCost.hpp"

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
     * @param fixFirstCamera   чи фіксувати Кадр 0 у початку координат (усуває Gauge Ambiguity)
     */
    bool optimize(GlobalMap& map,
                  const std::unordered_map<int, Frame>& allFramesMap,
                  bool fixFirstCamera = true) {
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

            std::vector<double> camData = {
                rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                pose.translation.at<double>(0),
                pose.translation.at<double>(1),
                pose.translation.at<double>(2),
            };

            cameraParams[frameId] = camData;
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

        // 3. Усунення невизначеності калібрувальної системи (Gauge Freedom):
        if (fixFirstCamera && !cameraParams.empty()) {
            // unordered_map не має порядку, тому begin() — випадкова камера. Фіксуємо саме найперший кадр (R=I, t=0).
            int anchorFrameId = cameraParams.begin()->first;
            for (const auto& entry : cameraParams) {
                anchorFrameId = std::min(anchorFrameId, entry.first);
            }
            problem.SetParameterBlockConstant(cameraParams[anchorFrameId].data());
            problem.SetParameterBlockConstant(cameraParams[anchorFrameId].data() + 3);
        }

        // 4. Налаштування розв'язувача
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = maxIterations;
        options.minimizer_progress_to_stdout = verbose;
        options.num_threads = 6;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if(verbose) 
        {
            std::cout << summary.BriefReport() << std::endl;
        }

        // 5. Запис оптимізованих результатів назад у GlobalMap
        for (auto& [frameId, camData] : cameraParams) {
            cv::Mat rvec = (cv::Mat_<double>(3, 1) << camData[0], camData[1], camData[2]);
            cv::Mat tvec = (cv::Mat_<double>(3, 1) << camData[3], camData[4], camData[5]);
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            map.addCameraPose(frameId, CameraPose(R, tvec));
        }

        for (auto& [pointId, ptData] : pointParams) {
            cv::Point3d optPos(ptData[0], ptData[1], ptData[2]);

            const_cast<std::unordered_map<int, MapPoint3D>&>(map.getMapPoints())[pointId].position = optPos;
        }

        return summary.IsSolutionUsable();
    }

};