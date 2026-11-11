#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "Types.hpp"
#include "GlobalMap.hpp"
#include "GeometryUtils.hpp"

/**
 * Щільна реконструкція: карти глибин зі стерео-пар (ректифікація + SGBM).
 *
 * Фото фасаду, зняті з руки, мають великий базис і повороти 10-30° між кадрами, тому:
 *  - діапазон диспаратності рахується з реальних розріджених точок, спільних для пари
 *    (він може бути 100..600 px — фіксований бюджет у 128-256 px відкидав УСІ пари);
 *  - праве зображення зсувається на мінімальну диспаратність, щоб SGBM шукав лише у [0, span];
 *  - для кожної пари будуються ДВІ карти (для обох кадрів) — більше незалежних підтверджень при злитті.
 */
class DenseReconstructor {
private:
    cv::Mat m_K;                 // K на роздільності щільної стадії
    double m_scale;              // Щільна роздільність / роздільність SfM
    int m_blockSize;
    int m_uniquenessRatio;
    double m_minTextureStd;      // Мінімальне локальне СКВ яскравості (небо, рівні стіни -> шум)
    double m_minAngleDeg, m_maxAngleDeg;
    int m_minSharedPoints;
    int m_maxNumDisparities;
    bool m_useClahe = true;

    struct ScaledImages { cv::Mat gray, color; };
    mutable std::unordered_map<int, ScaledImages> m_cache;

    const ScaledImages& images(const Frame& frame) const {
        auto it = m_cache.find(frame.id);
        if (it != m_cache.end()) return it->second;

        ScaledImages s;
        cv::Mat color = frame.imageColor;
        if (color.empty()) cv::cvtColor(frame.imageGray, color, cv::COLOR_GRAY2BGR);
        if (std::abs(m_scale - 1.0) > 1e-6) {
            cv::resize(color, s.color, cv::Size(), m_scale, m_scale, cv::INTER_AREA);
        } else {
            s.color = color;
        }
        cv::cvtColor(s.color, s.gray, cv::COLOR_BGR2GRAY);
        // Нічні фото: стіни майже однотонні й темні. Локальне вирівнювання контрасту (CLAHE)
        // проявляє фактуру штукатурки/панелей, і SGBM знаходить відповідності там, де раніше були дірки
        if (m_useClahe) {
            auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(s.gray, s.gray);
        }
        return m_cache[frame.id] = s;
    }

    cv::Mat textureMask(const cv::Mat& gray) const {
        cv::Mat g, mean, sqMean;
        gray.convertTo(g, CV_32F);
        cv::blur(g, mean, cv::Size(7, 7));
        cv::blur(g.mul(g), sqMean, cv::Size(7, 7));
        cv::Mat variance = sqMean - mean.mul(mean);
        return variance > m_minTextureStd * m_minTextureStd;
    }

    static std::vector<int> sharedPoints(const GlobalMap& map, int a, int b) {
        std::vector<int> ids;
        for (const auto& [pointId, pt] : map.getMapPoints()) {
            if (pt.isObservedIn(a) && pt.isObservedIn(b)) ids.push_back(pointId);
        }
        return ids;
    }

    // Проєктує світові точки в кадр view і будує z-буфер глибин
    DepthMap renderDepth(int viewId, int srcId, const CameraPose& pose,
                         const std::vector<cv::Vec3d>& world, cv::Size size) const
    {
        DepthMap dm;
        dm.refId = viewId;
        dm.srcId = srcId;
        dm.depth = cv::Mat::zeros(size, CV_32F);

        cv::Matx33d K(m_K);
        cv::Matx33d R = geom::rotation(pose);
        cv::Vec3d t = geom::translation(pose);

        for (const auto& Xw : world) {
            cv::Vec3d Xc = R * Xw + t;
            if (!(Xc[2] > 0)) continue;
            int u = cvRound(K(0, 0) * Xc[0] / Xc[2] + K(0, 2));
            int v = cvRound(K(1, 1) * Xc[1] / Xc[2] + K(1, 2));
            if (u < 0 || v < 0 || u >= size.width || v >= size.height) continue;
            float& d = dm.depth.at<float>(v, u);
            float z = static_cast<float>(Xc[2]);
            if (d == 0.f || z < d) d = z;
        }
        return dm;
    }

public:
    explicit DenseReconstructor(
        const cv::Mat& cameraMatrix,         // K роздільності SfM
        double scale = 0.7,
        int blockSize = 7,
        int uniquenessRatio = 8,
        double minTextureStd = 1.5,
        double minAngleDeg = 4.0,
        double maxAngleDeg = 30.0,
        int minSharedPoints = 30,
        int maxNumDisparities = 512)
        : m_scale(scale),
          m_blockSize(blockSize),
          m_uniquenessRatio(uniquenessRatio),
          m_minTextureStd(minTextureStd),
          m_minAngleDeg(minAngleDeg),
          m_maxAngleDeg(maxAngleDeg),
          m_minSharedPoints(minSharedPoints),
          m_maxNumDisparities(maxNumDisparities)
    {
        cameraMatrix.convertTo(m_K, CV_64F);
        m_K.row(0) *= scale;
        m_K.row(1) *= scale;
    }

    const cv::Mat& getCameraMatrix() const { return m_K; }
    void setUseClahe(bool enabled) { m_useClahe = enabled; m_cache.clear(); }

    const cv::Mat& colorImage(const Frame& frame) const { return images(frame).color; }

    struct StereoPartner {
        int frameId;
        double angleDeg;
        size_t sharedPoints;
        double score;
    };

    /**
     * Обирає для кадру refId до maxPartners стерео-партнерів.
     * Критерії: достатньо спільних точок, медіанний кут тріангуляції у [minAngle, maxAngle]
     * (замалий — шумна глибина, завеликий — SGBM не зіставить сильно спотворені вікна)
     * і епіполь поза кадром (інакше ректифікація вироджується).
     */
    std::vector<StereoPartner> selectStereoPartners(int refId, const GlobalMap& map, cv::Size imageSize,
                                                    size_t maxPartners = 2) const
    {
        std::vector<StereoPartner> partners;
        const CameraPose& refPose = map.getCameraPose(refId);
        cv::Vec3d refCenter = geom::cameraCenter(refPose);
        cv::Matx33d K(m_K);

        for (int candId : map.getRegisteredFrameIds()) {
            if (candId == refId) continue;

            auto shared = sharedPoints(map, refId, candId);
            if (static_cast<int>(shared.size()) < m_minSharedPoints) continue;

            cv::Vec3d candCenter = geom::cameraCenter(map.getCameraPose(candId));
            std::vector<double> angles;
            for (int pointId : shared) {
                angles.push_back(geom::rayAngleDeg(geom::toVec(map.getMapPoint(pointId).position), refCenter, candCenter));
            }
            std::nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
            double angle = angles[angles.size() / 2];
            if (angle < m_minAngleDeg || angle > m_maxAngleDeg) continue;

            // Епіполь: проєкція центру іншої камери в опорний кадр
            cv::Vec3d e = K * geom::toCamera(refPose, candCenter);
            if (std::abs(e[2]) > 1e-12) {
                double ex = e[0] / e[2], ey = e[1] / e[2];
                if (ex > -0.1 * imageSize.width && ex < 1.1 * imageSize.width &&
                    ey > -0.1 * imageSize.height && ey < 1.1 * imageSize.height) continue;
            }

            double score = shared.size() * std::min(angle, 15.0) / 15.0;
            partners.push_back({candId, angle, shared.size(), score});
        }

        std::sort(partners.begin(), partners.end(),
                  [](const StereoPartner& a, const StereoPartner& b) { return a.score > b.score; });
        if (partners.size() > maxPartners) partners.resize(maxPartners);
        return partners;
    }

    /**
     * Будує карти глибин для обох кадрів пари. Порожній результат — пару неможливо обробити.
     */
    std::vector<DepthMap> computeDepthMaps(const Frame& frameA, const Frame& frameB,
                                           const GlobalMap& map, bool allowSwap = true) const
    {
        const ScaledImages& imgA = images(frameA);
        const ScaledImages& imgB = images(frameB);
        cv::Size size = imgA.gray.size();

        const CameraPose& poseA = map.getCameraPose(frameA.id);
        const CameraPose& poseB = map.getCameraPose(frameB.id);

        // 1. Відносна трансформація A -> B
        cv::Matx33d RA = geom::rotation(poseA), RB = geom::rotation(poseB);
        cv::Vec3d tA = geom::translation(poseA), tB = geom::translation(poseB);
        cv::Matx33d Rrel = RB * RA.t();
        cv::Vec3d trel = tB - Rrel * tA;

        // 2. Стерео-ректифікація. Беремо з stereoRectify лише ПОВОРОТИ R1/R2: її власні P1/P2 при великих
        // поворотах між кадрами "наближають" спільну зону (фокусна 1000 -> 15000 px), і щільність падає.
        cv::Mat R1, R2, P1cv, P2cv, Qcv;
        cv::Mat zeroDist = cv::Mat::zeros(5, 1, CV_64F);
        cv::stereoRectify(m_K, zeroDist, m_K, zeroDist, size, cv::Mat(Rrel), cv::Mat(trel),
                          R1, R2, P1cv, P2cv, Qcv, cv::CALIB_ZERO_DISPARITY, -1, size);

        // Вертикальне стерео: SGBM шукає лише вздовж рядків
        if (std::abs(P2cv.at<double>(1, 3)) > std::abs(P2cv.at<double>(0, 3))) {
            std::cout << "[Dense MVS] Пара #" << frameA.id << " - #" << frameB.id
                      << " пропущена: вертикальна ректифікація." << std::endl;
            return {};
        }
        // P2(0,3) = -f * B > 0 означає, що камера B ліворуч від A — міняємо кадри місцями
        if (P2cv.at<double>(0, 3) > 0.0) {
            if (!allowSwap) return {};
            return computeDepthMaps(frameB, frameA, map, false);
        }

        // Власні проєкційні матриці: фокусна як у вихідних кадрів, полотно охоплює ректифікований кадр A
        const double f = m_K.at<double>(0, 0);
        const double baselineX = P2cv.at<double>(0, 3) / P2cv.at<double>(0, 0); // < 0
        cv::Matx33d Kinv = cv::Matx33d(m_K).inv();
        cv::Matx33d Rr1(R1), Rr2(R2);

        auto rectBounds = [&](const cv::Matx33d& Rr, cv::Rect2d& box) {
            double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
            for (cv::Point2d c : {cv::Point2d(0, 0), cv::Point2d(size.width, 0),
                                  cv::Point2d(0, size.height), cv::Point2d(size.width, size.height)}) {
                cv::Vec3d ray = Rr * (Kinv * cv::Vec3d(c.x, c.y, 1.0));
                if (ray[2] < 0.1 * cv::norm(ray)) return false; // поворот ректифікації майже 90°
                double x = f * ray[0] / ray[2], y = f * ray[1] / ray[2];
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
            box = cv::Rect2d(x0, y0, x1 - x0, y1 - y0);
            return true;
        };

        cv::Rect2d boxA, boxB;
        if (!rectBounds(Rr1, boxA) || !rectBounds(Rr2, boxB)) {
            std::cout << "[Dense MVS] Пара #" << frameA.id << " - #" << frameB.id
                      << " пропущена: надто великий поворот ректифікації." << std::endl;
            return {};
        }
        // Рядки, які є в обох кадрах; ширина — як у кадру A (B вирівнюється зсувом нижче)
        double top = std::max(boxA.y, boxB.y), bottom = std::min(boxA.y + boxA.height, boxB.y + boxB.height);
        double left = boxA.x, right = boxA.x + boxA.width;
        if (bottom - top < 50) return {};
        // Обмеження полотна (сильні перспективні спотворення роздувають рамку)
        auto clampRange = [](double& lo, double& hi, double maxLen) {
            if (hi - lo <= maxLen) return;
            double mid = 0.5 * (lo + hi);
            lo = mid - 0.5 * maxLen; hi = mid + 0.5 * maxLen;
        };
        clampRange(left, right, 2.0 * size.width);
        clampRange(top, bottom, 2.0 * size.height);

        const double cx = -left, cy = -top;
        cv::Size rectSize(static_cast<int>(std::ceil(right - left)), static_cast<int>(std::ceil(bottom - top)));
        cv::Mat P1 = (cv::Mat_<double>(3, 4) << f, 0, cx, 0,  0, f, cy, 0,  0, 0, 1, 0);
        cv::Mat P2 = (cv::Mat_<double>(3, 4) << f, 0, cx, f * baselineX,  0, f, cy, 0,  0, 0, 1, 0);
        cv::Mat Q = (cv::Mat_<double>(4, 4) << 1, 0, 0, -cx,  0, 1, 0, -cy,  0, 0, 0, f,  0, 0, -1.0 / baselineX, 0);

        // 3. Діапазон диспаратності зі спільних розріджених точок
        auto shared = sharedPoints(map, frameA.id, frameB.id);
        if (shared.size() < 10) return {};

        std::vector<double> disparities;
        for (int pointId : shared) {
            cv::Vec3d Xr = Rr1 * (RA * geom::toVec(map.getMapPoint(pointId).position) + tA);
            if (!(Xr[2] > 0)) continue;
            disparities.push_back(-f * baselineX / Xr[2]); // x1 - x2 = f * |B| / Z
        }
        if (disparities.size() < 10) return {};
        std::sort(disparities.begin(), disparities.end());
        double lo = disparities[static_cast<size_t>(0.01 * (disparities.size() - 1))];
        double hi = disparities[static_cast<size_t>(0.99 * (disparities.size() - 1))];
        double span = hi - lo;
        lo -= 0.3 * span + 8;
        hi += 0.3 * span + 8;

        int shift = static_cast<int>(std::floor(lo));
        int numDisparities = static_cast<int>(std::ceil((hi - shift) / 16.0)) * 16;
        numDisparities = std::clamp(numDisparities, 16, m_maxNumDisparities);

        // Зсув головної точки правого кадру: справжня диспаратність = диспаратність SGBM + shift
        cv::Mat P2shifted = P2.clone();
        P2shifted.at<double>(0, 2) += shift;

        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(m_K, zeroDist, R1, P1, rectSize, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(m_K, zeroDist, R2, P2shifted, rectSize, CV_32FC1, map2x, map2y);

        cv::Mat g1, g2, valid1, valid2;
        cv::Mat ones(size, CV_8U, cv::Scalar(255));
        cv::remap(imgA.gray, g1, map1x, map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        cv::remap(imgB.gray, g2, map2x, map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        cv::remap(ones, valid1, map1x, map1y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
        cv::remap(ones, valid2, map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

        // 4. SGBM. Відступ ліворуч на numDisparities: інакше ліва смуга такої ширини не має пошуку
        auto sgbm = cv::StereoSGBM::create(
            0, numDisparities, m_blockSize,
            8 * m_blockSize * m_blockSize,      // P1 (гладкість)
            32 * m_blockSize * m_blockSize,     // P2 (штраф за розриви глибини)
            1,                                  // disp12MaxDiff (перевірка ліво-право)
            31,                                 // preFilterCap
            m_uniquenessRatio,                  // uniquenessRatio (фільтр невпевнених зон)
            100,                                // speckleWindowSize
            2,                                  // speckleRange
            cv::StereoSGBM::MODE_SGBM_3WAY);

        auto computePadded = [&](const cv::Mat& left, const cv::Mat& right) {
            cv::Mat lp, rp, disp;
            cv::copyMakeBorder(left, lp, 0, 0, numDisparities, 0, cv::BORDER_CONSTANT, 0);
            cv::copyMakeBorder(right, rp, 0, 0, numDisparities, 0, cv::BORDER_CONSTANT, 0);
            sgbm->compute(lp, rp, disp);
            cv::Mat out;
            disp(cv::Rect(numDisparities, 0, rectSize.width, rectSize.height)).convertTo(out, CV_32F, 1.0 / 16.0);
            return out;
        };

        // Ліва карта (опорний кадр A)
        cv::Mat dispL = computePadded(g1, g2);
        // Права карта (опорний кадр B): дзеркалимо обидва зображення і міняємо їх ролями
        cv::Mat g1f, g2f, dispR;
        cv::flip(g2, g2f, 1);
        cv::flip(g1, g1f, 1);
        cv::flip(computePadded(g2f, g1f), dispR, 1);

        cv::Mat texL = textureMask(g1), texR = textureMask(g2);

        // 5. Репроєкція у світові координати: X_world = RA^T * (R1^T * X_rect - tA)
        cv::Matx44d Qm(Q);
        cv::Matx33d rectToWorld = RA.t() * Rr1.t();
        cv::Vec3d worldOffset = -(RA.t() * tA);

        auto toWorld = [&](double xLeft, double y, double d, cv::Vec3d& out) {
            cv::Vec4d h = Qm * cv::Vec4d(xLeft, y, d, 1.0);
            if (std::abs(h[3]) < 1e-12) return false;
            cv::Vec3d Xr(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
            if (!(Xr[2] > 0) || !std::isfinite(Xr[0]) || !std::isfinite(Xr[1])) return false;
            out = rectToWorld * Xr + worldOffset;
            return true;
        };

        std::vector<cv::Vec3d> worldL, worldR;
        worldL.reserve(rectSize.area() / 2);
        worldR.reserve(rectSize.area() / 2);
        for (int y = 0; y < rectSize.height; ++y) {
            const float* dl = dispL.ptr<float>(y);
            const float* dr = dispR.ptr<float>(y);
            const uchar* vl = valid1.ptr<uchar>(y);
            const uchar* vr = valid2.ptr<uchar>(y);
            const uchar* tl = texL.ptr<uchar>(y);
            const uchar* tr = texR.ptr<uchar>(y);
            for (int x = 0; x < rectSize.width; ++x) {
                cv::Vec3d X;
                // Невалідні пікселі SGBM мають від'ємну диспаратність
                if (dl[x] >= 0 && vl[x] && tl[x] && toWorld(x, y, dl[x] + shift, X)) worldL.push_back(X);
                // Для правої карти: у зсунутих координатах x_left = x_right + d
                if (dr[x] >= 0 && vr[x] && tr[x] && toWorld(x + dr[x], y, dr[x] + shift, X)) worldR.push_back(X);
            }
        }

        std::cout << "[Dense MVS] Пара #" << frameA.id << " - #" << frameB.id
                  << ": полотно " << rectSize.width << "x" << rectSize.height
                  << ", диспаратність " << shift << ".." << shift + numDisparities
                  << ", точок " << worldL.size() << " + " << worldR.size() << std::endl;

        return {
            renderDepth(frameA.id, frameB.id, poseA, worldL, size),
            renderDepth(frameB.id, frameA.id, poseB, worldR, size)
        };
    }
};
