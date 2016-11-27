// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// post-processing works for this method

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>               // c++11 timer functions
#include <png++/png.hpp>
#include "image.h"
#include "ctmf.h"
#include "match_cost.h"
#include "horiz_tree_cost_propagation.h"
#include "stereo_lib.h"

using namespace cvlab;

typedef float CostType;

int main(int argc, char *argv[]) {
    if (argc != 9) {
        std::cerr << "usage: " << argv[0] << " left(png) right(png) output(png)"
            " disp_range disp_factor with_post_processing sigma_r p_smooth"
                  << std::endl;
        return 1;
    }
    std::string left_name = argv[1];
    std::string right_name = argv[2];
    png::image<png::rgb_pixel> left_image(left_name);
	png::image<png::rgb_pixel> right_image(right_name);

    int width = left_image.get_width();
    int height = left_image.get_height();
    ImageU8 left(width, height, 3);
    ImageU8 right(width, height, 3);
    unsigned char* ptr_left = left.ptr_pixel(0, 0);
    unsigned char* ptr_right = right.ptr_pixel(0, 0);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            ptr_left = left.ptr_pixel(x, y);
            ptr_right = right.ptr_pixel(x, y);
            png::rgb_pixel left_rgb_pixel = left_image.get_pixel(x, y);
            png::rgb_pixel right_rgb_pixel = right_image.get_pixel(x, y);
            ptr_left[0] = left_rgb_pixel.red;
            ptr_left[1] = left_rgb_pixel.green;
            ptr_left[2] = left_rgb_pixel.blue;
            ptr_right[0] = right_rgb_pixel.red;
            ptr_right[1] = right_rgb_pixel.green;
            ptr_right[2] = right_rgb_pixel.blue;
        }
    }
    std::string disp_name = argv[3];
    // Set parameters
    int disp_range = atoi(argv[4]);
    int disp_factor = atoi(argv[5]);
    bool with_post_processing = atoi(argv[6]);
    double sigma_r = atof(argv[7]);
    double p_smooth = atof(argv[8]);

    int channels = 3;
    Image<int> left_disp_image(width, height, 1);
    Image<CostType> left_cost_vol(width, height, disp_range);
    Image<CostType> right_cost_vol(width, height, disp_range);
    MatchCostGradientAndCensus mcgc;
    mcgc.compute(left_cost_vol, right_cost_vol, left, right);

    disp_range = 178;
    left_cost_vol.resize_channels(disp_range);
    right_cost_vol.resize_channels(disp_range);

    int row_step = width * channels;
    int radius = 2;
    ImageU8 smoothed_left(left);
    // smooth with median filter
    ctmf(left.data(), smoothed_left.data(), width, height, row_step, row_step,
         radius, channels, width*height*channels);
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    HTreeCostPropagation<CostType> htcp_l(smoothed_left, left_cost_vol, sigma_r, p_smooth);
    // aggregation
    htcp_l.cost_propagate_with_smooth_prior();
    auto t_d = high_resolution_clock::now() - t0;
    auto msec = duration_cast<milliseconds>(t_d).count();
    // compute millions of disparity estimation per second
    std::cout << "MDE/s, " << width*height*disp_range/msec/1000.0 << std::endl;
    std::cout << "runtime, " <<  msec << "(msecs)" << std::endl;

    disp_best_cost_with_interpolation(left_disp_image, left_cost_vol, disp_factor, true);
    if(with_post_processing) {
        speckle_filter(left_disp_image, 100, static_cast<int>(2*disp_factor));
        // stereo post-processing
        ImageU8 smoothed_right(right);
        ctmf(right.data(), smoothed_right.data(), width, height, row_step, row_step,
             radius, channels, width*height*channels);

        HTreeCostPropagation<CostType> htcp_r(smoothed_right, right_cost_vol,
                                              sigma_r, p_smooth);
        htcp_r.cost_propagate_with_smooth_prior();
        Image<int> right_disp_image(width, height, 1);
        // disp_best_cost(right_disp_image, right_cost_vol);
        disp_best_cost_with_interpolation(right_disp_image, right_cost_vol,
                                          disp_factor, true);
        speckle_filter(right_disp_image, 100, static_cast<int>(2*disp_factor));

        // cross check
        ImageU8 left_err_map(width, height, 1, true);
        ImageU8 right_err_map(width, height, 1, true);
        // cross_check(left_err_map, left_disp_image, right_disp_image, 2);
        left_right_consistency_check(left_err_map, right_err_map,
                                     left_disp_image, right_disp_image,
                                     disp_factor, 1);
    }

    Image<float> out_disp_image(left_disp_image);
    out_disp_image.scale(1.0/disp_factor);
    png::image< png::gray_pixel_16 > pngimage(width, height);
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float disp_value = out_disp_image.pixel(x, y);
            if(disp_value<= 0.0 || disp_value > 255.0) {
                pngimage.set_pixel(x, y, 0);
            } else {
                pngimage.set_pixel(x, y, static_cast<int16_t>(disp_value*disp_factor+0.5));
            }
        }
    }
    pngimage.write(disp_name);
    return 0;
}
