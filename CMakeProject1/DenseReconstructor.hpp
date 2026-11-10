#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
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
    int m_maxNumDisparities;    // Верхня межа діапазону пошуку SGBM (кратна 16)
    int m_blockSize;
    double m_minDisparityPx;    // Мінімальна диспаратність: при d пікселів відносна похибка глибини ~ 1/d
    double m_maxForwardRatio;   // Максимальна частка руху "вперед" (вздовж осі Z) у базисі пари

    static int roundUpTo16(int value) {
        return ((value + 15) / 16) * 16;
    }

    static cv::Mat cameraCenter(const CameraPose& pose) {
        cv::Mat R, t;
        pose.rotation.convertTo(R, CV_64F);
        pose.translation.convertTo(t, CV_64F);
        return -R.t() * t;
    }

    std::vector<DensePoint3D> reconstructOriented(
        const Frame& frame1,
        const Frame& frame2,
        const CameraPose& pose1,
        const CameraPose& pose2,
        double minDepth,
        double maxDepth,
        int sourceId,
        bool allowSwap) const
    {
        std::vector<DensePoint3D> densePoints;

        // 1. Відносна трансформація Кадр 1 -> Кадр 2 (пози World-to-Camera: X_c = R * X_w + t)
        cv::Mat R1, t1, R2, t2;
        pose1.rotation.convertTo(R1, CV_64F);
        pose1.translation.convertTo(t1, CV_64F);
        pose2.rotation.convertTo(R2, CV_64F);
        pose2.translation.convertTo(t2, CV_64F);

        cv::Mat R_rel = R2 * R1.t();
        cv::Mat t_rel = t2 - R_rel * t1;

        // Центр камери 2 у системі координат камери 1
        cv::Mat c2InCam1 = -R_rel.t() * t_rel;
        double baseline = cv::norm(c2InCam1);
        if (baseline < 1e-9) return densePoints;

        // Якщо камера рухалась переважно вперед/назад, епіполь лежить усередині кадру
        // і ректифікація вироджується — така пара дає лише шум.
        if (std::abs(c2InCam1.at<double>(2)) / baseline > m_maxForwardRatio) {
            std::cout << "[Dense MVS] Пара #" << frame1.id << " - #" << frame2.id
                      << " пропущена: рух вздовж осі камери (ректифікація неможлива)." << std::endl;
            return densePoints;
        }

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

        // Вертикальне стерео (камера рухалась вгору/вниз): SGBM шукає лише вздовж рядків,
        // тому така ректифікація дає сміття.
        if (std::abs(P2.at<double>(1, 3)) > std::abs(P2.at<double>(0, 3))) {
            std::cout << "[Dense MVS] Пара #" << frame1.id << " - #" << frame2.id
                      << " пропущена: вертикальна ректифікація." << std::endl;
            return densePoints;
        }

        // P2(0,3) = -f * B. Якщо він додатний, камера 2 знаходиться ЛІВОРУЧ від камери 1,
        // диспаратність від'ємна і SGBM з minDisparity >= 0 нічого не знайде — міняємо кадри місцями.
        if (P2.at<double>(0, 3) > 0.0) {
            if (!allowSwap) return densePoints;
            return reconstructOriented(frame2, frame1, pose2, pose1, minDepth, maxDepth, sourceId, false);
        }

        // 3. Діапазон диспаратності з очікуваного діапазону глибин: d = f * B / Z
        double fRect = P1.at<double>(0, 0);
        double fB = -P2.at<double>(0, 3);
        double expectedMinDisp = fB / maxDepth;
        double expectedMaxDisp = fB / minDepth;

        int minDisparity = std::max(0, static_cast<int>(std::floor(expectedMinDisp)));
        int numDisparities = roundUpTo16(static_cast<int>(std::ceil(expectedMaxDisp)) - minDisparity + 1);
        numDisparities = std::clamp(numDisparities, 16, m_maxNumDisparities);

        // Карти трансформації для warp (remap)
        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(m_K, m_distCoeffs, R_rect1, P1, imgSize, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(m_K, m_distCoeffs, R_rect2, P2, imgSize, CV_32FC1, map2x, map2y);

        cv::Mat color1 = frame1.imageColor;
        if (color1.empty()) {
            cv::cvtColor(frame1.imageGray, color1, cv::COLOR_GRAY2BGR);
        }

        cv::Mat img1Rect, img2Rect, color1Rect;
        cv::remap(frame1.imageGray, img1Rect, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(frame2.imageGray, img2Rect, map2x, map2y, cv::INTER_LINEAR);
        cv::remap(color1, color1Rect, map1x, map1y, cv::INTER_LINEAR);

        // 4. Стерео-метчер Semi-Global Matching (SGBM)
        int cn = 1; // для grayscale
        auto sgbm = cv::StereoSGBM::create(
            minDisparity,
            numDisparities,
            m_blockSize,
            8 * cn * m_blockSize * m_blockSize,     // P1 (гладкість)
            32 * cn * m_blockSize * m_blockSize,    // P2 (штраф за розриви глибини)
            1,                                      // disp12MaxDiff (перевірка ліво-право)
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

        // 5. Репроєкція карти диспаратності в локальні 3D координати ректифікованої камери 1
        cv::Mat localPoints3D;
        cv::reprojectImageTo3D(disp32F, localPoints3D, Q, true);

        // 6. Переведення з ректифікованої системи у світові координати:
        // X_world = R1^T * (R_rect1^T * X_rect) - R1^T * t1
        // Matx замість cv::Mat у циклі: інакше на кожен піксель (~2 млн) виділяється купа пам'яті
        cv::Matx33d R_rect_to_world = cv::Mat(R1.t() * R_rect1.t());
        cv::Vec3d C1_world = cv::Mat(-R1.t() * t1);

        // Невалідні пікселі SGBM мають значення minDisparity - 1, тому поріг відсікає і їх
        float minAcceptedDisp = static_cast<float>(std::max<double>(minDisparity, m_minDisparityPx));

        densePoints.reserve(imgSize.width * imgSize.height / 4);

        for (int y = 0; y < disp32F.rows; ++y) {
            const float* dispRow = disp32F.ptr<float>(y);
            const cv::Vec3f* ptRow = localPoints3D.ptr<cv::Vec3f>(y);
            const cv::Vec3b* colorRow = color1Rect.ptr<cv::Vec3b>(y);

            for (int x = 0; x < disp32F.cols; ++x) {
                float d = dispRow[x];
                if (d < minAcceptedDisp) continue;

                const cv::Vec3f& pt = ptRow[x];
                double z = pt[2];

                // Глибина у тих же (довільних) одиницях, що й розріджена карта
                if (!std::isfinite(z) || z < minDepth || z > maxDepth) continue;

                cv::Vec3d world = R_rect_to_world * cv::Vec3d(pt[0], pt[1], pt[2]) + C1_world;

                DensePoint3D densePt;
                densePt.position = cv::Point3d(world[0], world[1], world[2]);
                densePt.color = colorRow[x];
                densePt.sourceId = sourceId;

                densePoints.push_back(densePt);
            }
        }

        std::cout << "[Dense MVS] Кадри #" << frame1.id << " та #" << frame2.id
                  << " (f*B=" << fB << ", fRect=" << fRect
                  << ", диспаратність " << minDisparity << ".." << minDisparity + numDisparities
                  << "): згенеровано " << densePoints.size() << " щільних 3D точок." << std::endl;

        return densePoints;
    }

public:
    explicit DenseReconstructor(
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs = cv::Mat(),
        int maxNumDisparities = 16 * 16, // Повинен бути кратним 16
        int blockSize = 5,
        double minDisparityPx = 8.0,
        double maxForwardRatio = 0.6)
        : m_maxNumDisparities(roundUpTo16(maxNumDisparities)),
          m_blockSize(blockSize),
          m_minDisparityPx(minDisparityPx),
          m_maxForwardRatio(maxForwardRatio)
    {
        cameraMatrix.convertTo(m_K, CV_64F);
        if (!distCoeffs.empty()) {
            distCoeffs.convertTo(m_distCoeffs, CV_64F);
        } else {
            m_distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
        }
    }

    /**
     * Обирає для кадру refId найкращого стерео-партнера серед candidateIds.
     *
     * Сусідні кадри відео мають крихітний базис: диспаратність фасаду ~2-5 px, тобто
     * похибка глибини 20-50% і "товстий" шумний фасад. Тому беремо партнера з НАЙБІЛЬШИМ
     * базисом, для якого діапазон диспаратностей ще вміщується у пошук SGBM.
     * @return id партнера або -1, якщо жодна пара не підходить
     */
    int selectStereoPartner(
        int refId,
        const std::vector<int>& candidateIds,
        const GlobalMap& map,
        double medianDepth) const
    {
        const CameraPose& refPose = map.getCameraPose(refId);
        cv::Mat refR;
        refPose.rotation.convertTo(refR, CV_64F);
        cv::Mat refCenter = cameraCenter(refPose);
        double fx = m_K.at<double>(0, 0);

        int bestId = -1;
        double bestBaseline = 0.0;

        for (int candId : candidateIds) {
            if (candId == refId || !map.hasCameraPose(candId)) continue;

            cv::Mat dirInRef = refR * (cameraCenter(map.getCameraPose(candId)) - refCenter);
            double baseline = cv::norm(dirInRef);
            if (baseline < 1e-9) continue;
            if (std::abs(dirInRef.at<double>(2)) / baseline > m_maxForwardRatio) continue;

            // Диспаратність "типової" точки сцени. Діапазон пошуку SGBM будується від дальньої межі,
            // тож якщо медіана займає до половини бюджету, фасад гарантовано вміщується у пошук.
            double medianDisp = fx * baseline / medianDepth;

            if (medianDisp < 2.0 * m_minDisparityPx) continue;       // замалий базис -> шумна глибина
            if (medianDisp > 0.5 * m_maxNumDisparities) continue;    // завеликий базис -> не вміститься у пошук

            if (baseline > bestBaseline) {
                bestBaseline = baseline;
                bestId = candId;
            }
        }

        return bestId;
    }

    /**
     * Будує щільну хмару точок між двома зареєстрованими кадрами.
     * minDepth / maxDepth задаються в одиницях карти (масштаб монокулярного SfM довільний,
     * тому їх треба брати з розрідженої хмари, а не з "метрів").
     */
    std::vector<DensePoint3D> reconstructDensePair(
        const Frame& frame1,
        const Frame& frame2,
        const CameraPose& pose1,
        const CameraPose& pose2,
        double minDepth,
        double maxDepth,
        int sourceId = -1) const
    {
        return reconstructOriented(frame1, frame2, pose1, pose2, minDepth, maxDepth, sourceId, true);
    }
};
