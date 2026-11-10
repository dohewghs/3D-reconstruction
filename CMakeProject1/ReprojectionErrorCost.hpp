#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/rotation.h>

struct ReprojectionErrorCost {
    ReprojectionErrorCost(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
        : observed_x(observed_x), observed_y(observed_y),
          fx(fx), fy(fy), cx(cx), cy(cy) 
    {};
    
    template <typename T>
    bool operator()(const T* const camera_rvec,  // 3 params: Rodrigues vector
                    const T* const camera_tvec,  // 3 params: Translation vector
                    const T* const point3d,      // 3 params: X, Y, Z
                    T* residuals) const
    {
        T p[3];
        ceres::AngleAxisRotatePoint(camera_rvec, point3d, p);

        p[0] += camera_tvec[0];
        p[1] += camera_tvec[1];
        p[2] += camera_tvec[2];


        // 3. Перспективна проєкція на площину зображення

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // 4. Переведення в пікселі за внутрішніми параметрами камери (fx, fy, cx, cy)
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        // 5. Невязка

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y,
                                        double fx, double fy, double cx, double cy)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorCost, 2, 3, 3, 3>( new ReprojectionErrorCost(observed_x, observed_y, fx, fy, cx, cy));
    }


    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;
};
