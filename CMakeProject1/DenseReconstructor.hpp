#pragma once

#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "Types.hpp"
#include "GlobalMap.hpp"

class DenseReconstructor {
private:
    cv::Mat m_K;
    cv::Mat m_distCoeffs;
    int m_minDisparity;
    int m_numDisparities;
    int m_blockSize;

public:
    explicit DenseReconstructor(
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs = cv::Mat(),
        int numDisparities = 16 * 8, // Повинен бути кратним 16 (128 пікселів пошуку)
        int blockSize = 5)
        : m_minDisparity(0),
          m_numDisparities(numDisparities),
          m_blockSize(blockSize)
    {
        cameraMatrix.convertTo(m_K, CV_64F);
        if (!distCoeffs.empty()) {
            distCoeffs.convertTo(m_distCoeffs, CV_64F);
        } else {
            m_distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
        }
    }

    /**
     * Будує щільну хмару точок між двома зареєстрованими кадрами
     */
    std::vector<DensePoint3D> reconstructDensePair(
        const Frame& frame1,
        const Frame& frame2,
        const cv::Mat& colorImage1, // Оригінальне кольорове зображення кадру 1
        const CameraPose& pose1,
        const CameraPose& pose2,
        double maxDepth = 60.0,
        double minDepth = 1.0) const
    {
        std::vector<DensePoint3D> densePoints;

        // 1. Обчислюємо відносну трансформацію: Кадр 1 -> Кадр 2
        // Оскільки pose.rotation і pose.translation - це World-to-Camera (X_c = R * X_w + t):
        cv::Mat R1 = pose1.rotation, t1 = pose1.translation;
        cv::Mat R2 = pose2.rotation, t2 = pose2.translation;

        R1.convertTo(R1, CV_64F);
        t1.convertTo(t1, CV_64F);
        R2.convertTo(R2, CV_64F);
        t2.convertTo(t2, CV_64F);

        cv::Mat R_rel = R2 * R1.t();
        cv::Mat t_rel = t2 - R_rel * t1;

        cv::Size imgSize = frame1.imageGray.size();

        // 2. Стерео-ректифікація
        cv::Mat R_rect1, R_rect2, P1, P2, Q;
        cv::stereoRectify(
            m_K, m_distCoeffs,
            m_K, m_distCoeffs,
            imgSize, R_rel, t_rel,
            R_rect1, R_rect2, P1, P2, Q,
            cv::CALIB_ZERO_DISPARITY, 0, imgSize
        );

        // Карти трансформації для warp (remap)
        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(m_K, m_distCoeffs, R_rect1, P1, imgSize, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(m_K, m_distCoeffs, R_rect2, P2, imgSize, CV_32FC1, map2x, map2y);

        cv::Mat img1Rect, img2Rect, color1Rect;
        cv::remap(frame1.imageGray, img1Rect, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(frame2.imageGray, img2Rect, map2x, map2y, cv::INTER_LINEAR);
        cv::remap(colorImage1, color1Rect, map1x, map1y, cv::INTER_LINEAR);

        // 3. Стерео-метчер Semi-Global Matching (SGBM)
        int cn = 1; // для grayscale
        auto sgbm = cv::StereoSGBM::create(
            m_minDisparity,
            m_numDisparities,
            m_blockSize,
            8 * cn * m_blockSize * m_blockSize,     // P1 (гладкість)
            32 * cn * m_blockSize * m_blockSize,    // P2 (штраф за розриви глибини)
            1,                                      // disp12MaxDiff
            31,                                     // preFilterCap
            15,                                     // uniquenessRatio (фільтр невпевнених зон)
            150,                                    // speckleWindowSize
            2,                                      // speckleRange
            cv::StereoSGBM::MODE_SGBM_3WAY
        );

        cv::Mat disp16S;
        sgbm->compute(img1Rect, img2Rect, disp16S);

        // Диспаратність SGBM повертає у форматі 16-біт з фіксованою комою (множник 16)
        cv::Mat disp32F;
        disp16S.convertTo(disp32F, CV_32F, 1.0 / 16.0);

        // 4. Репроєкція карти диспарантності в локальні 3D координати
        cv::Mat localPoints3D;
        cv::reprojectImageTo3D(disp32F, localPoints3D, Q, true);

        // 5. Переведення з локальної ректифікованої системи назад у глобальні координати світу:
        // X_cam1 = R_rect1^T * X_rect
        // X_world = R1^T * (X_cam1 - t1) = R1^T * (R_rect1^T * X_rect) - R1^T * t1
        cv::Mat R_rect_to_world = (R_rect1 * R1).t(); // Матриця 3x3
        cv::Mat C1_world = -R1.t() * t1;              // Оптичний центр камери 1

        densePoints.reserve(imgSize.width * imgSize.height / 4);

        for (int y = 0; y < disp32F.rows; ++y) {
            const float* dispRow = disp32F.ptr<float>(y);
            const cv::Vec3f* ptRow = localPoints3D.ptr<cv::Vec3f>(y);
            const cv::Vec3b* colorRow = color1Rect.ptr<cv::Vec3b>(y);

            for (int x = 0; x < disp32F.cols; ++x) {
                float d = dispRow[x];

                // 1. Відсікаємо нерозраховані зони та малу диспарантність (небо, наддалекі точки)
                if (d < 2.0f) continue; // шукаємо тільки надійні зміщення від 2 пікселів

                cv::Vec3f pt = ptRow[x];
                double z = pt[2];

                // 2. Жорстко обмежуємо глибину Z для фасаду
                // Якщо будинок на відстані приблизно 10-35 метрів:
                if (!std::isfinite(z) || z <= 2.0 || z >= 35.0) continue;

                // 3. Відсікаємо екстремальні X та Y (щоб промені не летіли в космос)
                if (std::abs(pt[0]) > 40.0 || std::abs(pt[1]) > 40.0) continue;

                cv::Mat ptMat = (cv::Mat_<double>(3, 1) << pt[0], pt[1], pt[2]);
                cv::Mat worldPtMat = R_rect_to_world * ptMat + C1_world;

                DensePoint3D densePt;
                densePt.position = cv::Point3d(
                    worldPtMat.at<double>(0),
                    worldPtMat.at<double>(1),
                    worldPtMat.at<double>(2)
                );
                densePt.color = colorRow[x];

                densePoints.push_back(densePt);
            }
        }

        std::cout << "[Dense MVS] Кадри #" << frame1.id << " та #" << frame2.id 
                  << ": згенеровано " << densePoints.size() << " щільних 3D точок." << std::endl;

        return densePoints;
    }
};