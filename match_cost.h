// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description : Matching cost functions for stereo.

// These codes are adapt from the code for "Efficient Joint Segmentation,
// Occlusion Labeling" by Koichiro Yamaguchi et al..
// http://ttic.uchicago.edu/~dmcallester/SPS/index.html

#ifndef MATCH_COST_H_
#define MATCH_COST_H_

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "image.h"
#include "image_proc-inl.h"

namespace cvlab{

class MatchCostGradientAndCensus {
  public:
    MatchCostGradientAndCensus();
    ~MatchCostGradientAndCensus();

    template <typename CostType>
    void compute(Image<CostType>& left_cost_vol,
                 Image<CostType>& right_cost_vol,
                 const ImageU8& left_image,
                 const ImageU8& right_image);

  public:
    void set_aggregate_window_radius(int value) {
        aggregate_window_radius_ = value;
    }

  private:
    void init(const ImageU8& left_image, const ImageU8& right_image,
              int disp_range);

    void free_buffer();

    void compute_left_cost_image(unsigned char* left_grayscale_image,
                                 unsigned char* right_grayscale_image);

    void compute_right_cost_image();

    void compute_capped_sobel_image(unsigned char* sobel_image,
                                    const unsigned char* image,
                                    bool horizontal_flip);

    void compute_census_image(int* census_image, const unsigned char* image);


    void compute_top_row_cost(unsigned short* cost_image_row,
                              unsigned char*& left_sobel_row,
                              int*& left_census_row,
                              unsigned char*& right_sobel_row,
                              int*& right_census_row);

    void compute_row_cost(unsigned short* cost_image_row,
                          unsigned char*& left_sobel_row,
                          int*& left_census_row,
                          unsigned char*& right_sobel_row,
                          int*& right_census_row);

    void compute_pixelwise_sad(const unsigned char* left_sobel_row,
                               const unsigned char* right_sobel_row);

    void compute_half_pixel_right(const unsigned char* right_sobel_row);

    void add_pixelwise_hamming(const int* left_census_row,
                               const int* right_census_row);

  private:
    int width_;
    int height_;
    int disp_range_;
    int width_step_;

    int pixelwise_cost_row_buff_size_;
    int row_aggregate_cost_buff_size_;
    int half_pixel_right_buff_size_;

    // prameters
    int aggregate_window_radius_;
    int census_window_radius_;
    unsigned char sobel_cap_value_;
    double census_weight_factor_;

    // buffers
    uint16_t* left_cost_image_;
    uint16_t* right_cost_image_;
    unsigned char* pixelwise_cost_row_;
    uint16_t* row_aggregated_cost_;
    unsigned char* half_pixel_right_min_;
    unsigned char* half_pixel_right_max_;
};

template <typename CostType>
void MatchCostGradientAndCensus::
compute(Image<CostType>& left_cost_vol,
        Image<CostType>& right_cost_vol,
        const ImageU8& left_image,
        const ImageU8& right_image) {

    // init image sizes and buffer sizes
    init(left_image, right_image, left_cost_vol.channels());

    // convert to grayscale image
    ImageU8 left_gray(width_, height_, 1);
    ImageU8 right_gray(width_, height_, 1);
    rgb2gray(left_gray, left_image);
    rgb2gray(right_gray, right_image);
    unsigned char* left_grayscale_image = left_gray.data();
    unsigned char* right_grayscale_image = right_gray.data();

    // preset buffers
    memset(left_cost_image_, 0, width_*height_*disp_range_*sizeof(uint16_t));
    // compute left cost image
    compute_left_cost_image(left_grayscale_image, right_grayscale_image);
    compute_right_cost_image();
    // put into container
    CostType* ptr_left_cost_vol;
    CostType* ptr_right_cost_vol;
    uint16_t* ptr_left_cost_image;
    uint16_t* ptr_right_cost_image;
    for(int y = 0; y < height_; y++) {
        ptr_left_cost_vol = left_cost_vol.ptr_pixel(0, y);
        ptr_right_cost_vol = right_cost_vol.ptr_pixel(0, y);
        ptr_left_cost_image = left_cost_image_+y*width_*disp_range_;
        ptr_right_cost_image = right_cost_image_+y*width_*disp_range_;
        for(int i = 0; i < width_*disp_range_; i++) {
            *ptr_left_cost_vol++ =
                    static_cast<CostType>(*ptr_left_cost_image++);
            *ptr_right_cost_vol++ =
                    static_cast<CostType>(*ptr_right_cost_image++);
        }
    }
    // release data buffers
    free_buffer();
}

}
#endif  // MATCH_COST_H_
