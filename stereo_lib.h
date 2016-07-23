// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description : Stereo matching util functions.

#ifndef STEREO_LIB_H_
#define STEREO_LIB_H_

#include <vector>
#include <algorithm>
#include "image.h"

namespace cvlab {

// define disparity value status
enum DispStatus {
    DISP_STATUS_ERROR = 0,
    DISP_STATUS_RELIABLE,
    DISP_STATUS_UNRELIABLE,
    DISP_STATUS_INTERPOLATED,
};

void speckle_filter(Image<int>& input_image,
                    int max_speckle_size,
                    int max_diff);

// This function compute the best disparity with interpolation
// disparity values are scaled by <disp_factor>
template<typename CostType>
inline void disp_best_cost_with_interpolation(Image<int>& disp_image,
                                              const Image<CostType>& cost_vol,
                                              int disp_factor,
                                              bool zero_disp_valid = false) {
    int width = cost_vol.width();
    int height = cost_vol.height();
    int disp_range = cost_vol.channels();
    int factor = disp_factor;

    const CostType* ptr_cost = NULL;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            ptr_cost = cost_vol.ptr_pixel(x, y);
            int disp = 0;
            CostType mincost = ptr_cost[0];
            if (!zero_disp_valid) {
                disp = 1;
                mincost = ptr_cost[1];
            }
            for (int d = 1; d < disp_range; ++d) {
                if (ptr_cost[d] < mincost) {
                    mincost = ptr_cost[d];
                    disp = d;
                }
            }
            // interpolate
            if (disp > 0 && disp < disp_range-1) {
                double center_cost = static_cast<double>(ptr_cost[disp]);
                double left_cost = static_cast<double>(ptr_cost[disp-1]);
                double right_cost = static_cast<double>(ptr_cost[disp+1]);
                if (right_cost < left_cost) {
                    disp = static_cast<int>(
                        0.5 + disp*factor +
                        (right_cost-left_cost)/
                        (center_cost-left_cost)/2.0*factor);
                } else {
                    disp = static_cast<int>(
                        0.5 + disp*factor +
                        (right_cost-left_cost)/
                        (center_cost-right_cost)/2.0*factor);
                }
            } else {
                disp = static_cast<int>(disp * factor);
            }
            disp_image.pixel(x, y) = disp;
        }
    }
}

// note: here we use int disparity values
inline void left_right_consistency_check(ImageU8& left_err_map,
                                         ImageU8& right_err_map,
                                         Image<int>& left_disp_image,
                                         Image<int>& right_disp_image,
                                         int disp_factor,
                                         int consistency_threshold = 1) {
    // make sure err_map is init to DISP_STATUS_RELIABLE
    left_err_map.dataset(DISP_STATUS_RELIABLE);
    right_err_map.dataset(DISP_STATUS_RELIABLE);
    int width = left_disp_image.width();
    int height = left_disp_image.height();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (left_disp_image.pixel(x, y) == 0) {
                left_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                continue;
            }
            int left_disp = static_cast<int>(
                static_cast<double>(left_disp_image.pixel(x, y))/disp_factor+0.5);
            if (x - left_disp < 0) {
                left_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                left_disp_image.pixel(x, y) = 0;
                continue;
            }
            int right_disp = static_cast<int>(
                static_cast<double>(right_disp_image.pixel(x-left_disp, y))/disp_factor+0.5);
            if (right_disp == 0 || abs(left_disp-right_disp) > consistency_threshold) {
                left_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                left_disp_image.pixel(x, y) = 0;
            }
        }
    }
    // check right disparity image if exist
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (right_disp_image.pixel(x, y) == 0) {
                right_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                continue;
            }
            int right_disp = static_cast<int>(
                static_cast<double>(right_disp_image.pixel(x, y))/disp_factor+0.5);
            if (x + right_disp >= width) {
                right_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                right_disp_image.pixel(x, y) = 0;
                continue;
            }
            int left_disp = static_cast<int>(
                static_cast<double>(left_disp_image.pixel(x+right_disp, y))/disp_factor+0.5);
            if (left_disp == 0 || abs(right_disp-left_disp) > consistency_threshold) {
                right_err_map.pixel(x, y) = DISP_STATUS_ERROR;
                right_disp_image.pixel(x, y) = 0;
            }
        }
    }
}

}  // namespace cvlab

#endif  // STEREO_LIB_H_
