// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description : Cost propagation on a simple tree for stereo matching.

#ifndef HORIZ_TREE_COST_AGGREGATION_H_
#define HORIZ_TREE_COST_AGGREGATION_H_

#include <cstdint>
#include <cmath>
#include "image.h"
#include "image_proc-inl.h"

namespace cvlab {

template<typename CostType>
class HTreeCostPropagation {
  public:
    HTreeCostPropagation();
    HTreeCostPropagation(ImageU8& image,
                         Image<CostType>& cost_vol,
                         CostType sigma_r = 0.1,
                         CostType p_smooth = 5.0);

    ~HTreeCostPropagation();

    void cost_propagate();
    void cost_propagate_with_smooth_prior();

    void compute_normalization_factor(Image<CostType>& norm_factor);

    void compute_norm_factor_h(Image<CostType>& norm_factor);
    void compute_norm_factor_v(Image<CostType>& norm_factor);

    void update_lut(CostType sigma_r);  // update LUT
    void set_p_smooth(CostType p_smooth) { p_smooth_ = p_smooth; }

  protected:
    void propagate(Image<CostType>& cost_vol,
                   bool with_normalization = false);
    void propagate_with_smooth_prior(Image<CostType>& cost_vol,
                                     bool with_normalization = false);
    void compute_propagation_coeff(ImageU8& image);

  protected:
    ImageU8* image_;
    Image<CostType>* cost_vol_;
    Image<int>* propagation_coeff_x_;
    Image<int>* propagation_coeff_y_;

    int img_width_;
    int img_height_;

    CostType sigma_r_;
    CostType p_smooth_;

    CostType lut_[256];  // LUT for propagation weight
};

template<typename CostType>
HTreeCostPropagation<CostType>::HTreeCostPropagation() :
        image_(NULL),
        cost_vol_(NULL),
        propagation_coeff_x_(NULL),
        propagation_coeff_y_(NULL),
        img_width_(0),
        img_height_(0),
        sigma_r_(0),
        p_smooth_(0) { }

template<typename CostType>
HTreeCostPropagation<CostType>::~HTreeCostPropagation() {
    delete propagation_coeff_x_;
    delete propagation_coeff_y_;
}

template<typename CostType>
HTreeCostPropagation<CostType>::
HTreeCostPropagation(ImageU8& image,
                     Image<CostType>& cost_vol,
                     CostType sigma_r,
                     CostType p_smooth) :
        image_(&image),
        cost_vol_(&cost_vol),
        sigma_r_(sigma_r),
        p_smooth_(p_smooth) {

    img_width_ = image.width();
    img_height_ = image.height();

    propagation_coeff_x_ = new Image<int>(img_width_, img_height_, 1);
    propagation_coeff_y_ = new Image<int>(img_width_, img_height_, 1);

    compute_propagation_coeff(image);
    update_lut(sigma_r);
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
cost_propagate() {
    propagate(*cost_vol_);
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
cost_propagate_with_smooth_prior() {
    propagate_with_smooth_prior(*cost_vol_);
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
compute_normalization_factor(Image<CostType>& norm_factor) {
    // propagate(norm_factor, false);
    compute_norm_factor_h(norm_factor);
    compute_norm_factor_v(norm_factor);
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
update_lut(CostType sigma_r) {
    sigma_r = std::max(sigma_r, static_cast<CostType>(0.01));
    // TODO: replace 256 with a configurable value
    for(int i = 0; i < 256; i++) {
        lut_[i] = exp(-static_cast<CostType>(i) / (255*sigma_r));
    }
}

// protected functions
template<typename CostType>
void HTreeCostPropagation<CostType>::
propagate(Image<CostType>& cost_vol, bool with_normalization) {
    int width = img_width_, height = img_height_;
    int disp_range = cost_vol.channels();

    Image<CostType> cost_vol_temp(cost_vol);
    // horizontal phase
    // forward-backward algorithm

    // forward pass
    CostType* ptr_pre_node = cost_vol_temp.ptr_pixel(0, 0);
    CostType* ptr_cur_node = cost_vol_temp.ptr_pixel(1, 0);
    CostType* ptr_end_node =
            cost_vol_temp.ptr_pixel(width-1, height-1) + (disp_range-1);
    int* ptr_coeff = propagation_coeff_x_->ptr_pixel(1, 0);
    CostType weight = 0.0;
    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        for(int d = 0; d < disp_range; d++) {  // only for loop counting
            *ptr_cur_node++ += weight * (*ptr_pre_node++);
        }
    }

    // backward pass
    ptr_pre_node = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    ptr_cur_node = cost_vol.ptr_pixel(width-2, height-1) + (disp_range-1);
    ptr_end_node = cost_vol.ptr_pixel(0, 0);
    ptr_coeff = propagation_coeff_x_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        for(int d = 0; d < disp_range; d++) {  // only for loop counting
            *ptr_cur_node-- += weight * (*ptr_pre_node--);
        }
    }

    // compute temp result of horizontal phase
    CostType* ptr_cost = cost_vol.data();
    CostType* end_cost = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    CostType* ptr_cost_temp = cost_vol_temp.data();
    while(ptr_cost <= end_cost) {
        for(int d = 0; d < disp_range; d++) {
            *ptr_cost++ += *ptr_cost_temp++;
        }
    }  // to save memory, we did not minus the cost in place, this will not
    // change the result much

    // normalize cost value
    if (with_normalization) {
        Image<CostType> norm_factor(width, height, 1);
        norm_factor.dataset(1.0);
        compute_norm_factor_h(norm_factor);

        ptr_cost = cost_vol.data();
        end_cost = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
        CostType* ptr_norm_factor = norm_factor.data();
        while(ptr_cost <= end_cost) {
            CostType norm = *ptr_norm_factor++;
            for(int d = 0; d < disp_range; d++) {
                *ptr_cost++ /= norm;
            }
        }  // to save memory, we did not minus the cost in place, this will not
        // change the result much
    }

    memcpy(cost_vol_temp.data(), cost_vol.data(),
           sizeof(CostType)*width*height*disp_range);
    // update_lut(sigma_r_ / 1.5);  // optional, I did not find much improvement

    // vertical phase
    // forward pass
    // in fact, ptr_pre_node does not point to the "node" but cost values
    ptr_pre_node = cost_vol_temp.ptr_pixel(0, 0);
    ptr_cur_node = cost_vol_temp.ptr_pixel(0, 1);  // the next row
    ptr_end_node = cost_vol_temp.ptr_pixel(width-1, height-1) + (disp_range-1);
    ptr_coeff = propagation_coeff_y_->ptr_pixel(0, 1);

    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        for(int d = 0; d < disp_range; d++) {
            *ptr_cur_node++ += weight * (*ptr_pre_node++);
        }
    }
    // backward pass
    ptr_pre_node = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    ptr_cur_node = cost_vol.ptr_pixel(width-1, height-2) + (disp_range-1);
    ptr_end_node = cost_vol.data();
    ptr_coeff = propagation_coeff_y_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        for(int d = 0; d < disp_range; d++) {  // only for loop counting
            *ptr_cur_node-- += weight * (*ptr_pre_node--);
        }
    }
    // get the final result
    ptr_cost = cost_vol.data();
    end_cost = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    ptr_cost_temp = cost_vol_temp.data();
    while(ptr_cost <= end_cost) {
        *ptr_cost++ += *ptr_cost_temp++;
    }  // again to save memory, we did not minus the cost in place, this will not
    // change the result much
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
propagate_with_smooth_prior(Image<CostType>& cost_vol,
                            bool with_normalization) {
    using std::min;
    int width = img_width_, height = img_height_;
    int disp_range = cost_vol.channels();
    CostType p_smooth = p_smooth_;

    Image<CostType> cost_vol_temp(cost_vol);
    // horizontal phase
    // forward-backward algorithm

    // forward pass
    CostType* ptr_pre_node = cost_vol_temp.ptr_pixel(0, 0);
    CostType* ptr_cur_node = cost_vol_temp.ptr_pixel(1, 0);
    CostType* ptr_end_node = cost_vol_temp.ptr_pixel(width-1, height-1);
    int* ptr_coeff = propagation_coeff_x_->ptr_pixel(1, 0);
    CostType weight = 0.0;
    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        int d = 0;
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d+1] + p_smooth);

        for(d = 1; d < disp_range-1; d++) {
            ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                            min(ptr_pre_node[d-1],
                                                ptr_pre_node[d+1]) + p_smooth);
        }
        // d = disp_range-1
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d-1] + p_smooth);
        ptr_cur_node += disp_range;
        ptr_pre_node += disp_range;
    }

    // backward pass
    ptr_pre_node = cost_vol.ptr_pixel(width-1, height-1);
    ptr_cur_node = cost_vol.ptr_pixel(width-2, height-1);
    ptr_end_node = cost_vol.ptr_pixel(0, 0);
    ptr_coeff = propagation_coeff_x_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        int d = 0;
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d+1] + p_smooth);

        for(d = 1; d < disp_range-1; d++) {
            ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                            min(ptr_pre_node[d-1],
                                                ptr_pre_node[d+1]) + p_smooth);
        }
        // d = disp_range-1
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d-1] + p_smooth);
        ptr_cur_node -= disp_range;
        ptr_pre_node -= disp_range;
    }

    CostType* ptr_cost = cost_vol.data();
    CostType* end_cost = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    CostType* ptr_cost_temp = cost_vol_temp.data();
    while(ptr_cost <= end_cost) {
        *ptr_cost++ += *ptr_cost_temp++;
    }
    // to save memory, we did not minus the cost in place, this will not
    // change the result much

    if (with_normalization) {
        // compute temp result of horizontal phase and normalize
        Image<CostType> norm_factor(width, height, 1);
        compute_norm_factor_h(norm_factor);

        ptr_cost = cost_vol.data();
        end_cost = cost_vol.ptr_pixel(width-1, height-1);
        CostType* ptr_norm_factor = norm_factor.data();
        while (ptr_cost <= end_cost) {
            CostType norm = *ptr_norm_factor++;
            for (int d = 0; d < disp_range; ++d) {
                ptr_cost[d] /= norm;
            }
            ptr_cost += disp_range;
        }
    }
    memcpy(cost_vol_temp.data(), cost_vol.data(),
           sizeof(CostType)*width*height*disp_range);
    // update_lut(sigma_r_ / 2);  // optional, I did not file much improvement
    // p_smooth_ /= 2;

    // vertical phase
    // forward pass
    // infact, ptr_pre_node does not point to the "node" but cost values
    ptr_pre_node = cost_vol_temp.ptr_pixel(0, 0);
    ptr_cur_node = cost_vol_temp.ptr_pixel(0, 1);  // the next row
    ptr_end_node = cost_vol_temp.ptr_pixel(width-1, height-1);
    ptr_coeff = propagation_coeff_y_->ptr_pixel(0, 1);

    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        int d = 0;
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d+1] + p_smooth);

        for(d = 1; d < disp_range-1; d++) {
            ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                            min(ptr_pre_node[d-1],
                                                ptr_pre_node[d+1]) + p_smooth);
        }
        // d = disp_range-1
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d-1] + p_smooth);
        ptr_cur_node += disp_range;
        ptr_pre_node += disp_range;
    }
    // backward pass
    ptr_pre_node = cost_vol.ptr_pixel(width-1, height-1);
    ptr_cur_node = cost_vol.ptr_pixel(width-1, height-2);
    ptr_end_node = cost_vol.data();
    ptr_coeff = propagation_coeff_y_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        int d = 0;
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d+1] + p_smooth);

        for(d = 1; d < disp_range-1; d++) {
            ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                            min(ptr_pre_node[d-1],
                                                ptr_pre_node[d+1]) + p_smooth);
        }
        // d = disp_range-1
        ptr_cur_node[d] += weight * min(ptr_pre_node[d],
                                        ptr_pre_node[d-1] + p_smooth);
        ptr_cur_node -= disp_range;
        ptr_pre_node -= disp_range;
    }
    // get the final result
    ptr_cost = cost_vol.data();
    end_cost = cost_vol.ptr_pixel(width-1, height-1) + (disp_range-1);
    ptr_cost_temp = cost_vol_temp.data();
    while(ptr_cost <= end_cost) {
        *ptr_cost++ += *ptr_cost_temp++;
    }  // again to save memory, we did not minus the cost in place, this will not
    // change the result much
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
compute_propagation_coeff(ImageU8& image) {
    ImageU8 smoothed(image);
    int width = img_width_, height = img_height_;
    int channels = image.channels();

    int* ptr_coeff_x = NULL;
    int* ptr_coeff_y = NULL;

    unsigned char* ptr_img_idx0 =  NULL;
    unsigned char* ptr_img_idx1 =  NULL;
    // left -> right
    for(int y = 0; y < height; y++) {
        ptr_img_idx0 = smoothed.ptr_pixel(0, y);
        ptr_img_idx1 = smoothed.ptr_pixel(1, y);
        ptr_coeff_x = propagation_coeff_x_->ptr_pixel(0, y);
        // deal with x = 0;
        ptr_coeff_x[0] =
                pixel_abs_diff_max_in_channel<int, unsigned char> (
                    ptr_img_idx1, ptr_img_idx0, channels);
        ptr_coeff_x[1] = ptr_coeff_x[0];

        ptr_img_idx0 += channels;
        ptr_img_idx1 += channels;
        for(int x = 2; x < width; x++) {
            ptr_coeff_x[x] = pixel_abs_diff_max_in_channel<int, unsigned char> (
                ptr_img_idx1, ptr_img_idx0, channels);
            ptr_img_idx0 += channels;
            ptr_img_idx1 += channels;
        }
    }
    // up -> down
    for(int y = 1; y < height; y++) {
        ptr_img_idx0 = smoothed.ptr_pixel(0, y-1);
        ptr_img_idx1 = smoothed.ptr_pixel(0, y);
        ptr_coeff_y = propagation_coeff_y_->ptr_pixel(0, y);
        for(int x = 0; x < width; x++) {
            *ptr_coeff_y++ =
                    pixel_abs_diff_max_in_channel<int, unsigned char> (
                        ptr_img_idx1, ptr_img_idx0, channels);
            ptr_img_idx0 += channels;
            ptr_img_idx1 += channels;
        }
    }
    // copy to y = 0
    memcpy(propagation_coeff_y_->ptr_pixel(0, 0),
           propagation_coeff_y_->ptr_pixel(0, 1), sizeof(int)*width);
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
compute_norm_factor_h(Image<CostType>& norm_factor) {
    int width = img_width_, height = img_height_;
    norm_factor.dataset(1.0);
    Image<CostType> norm_factor_temp(norm_factor);
    // horizontal phase
    // forward-backward algorithm

    // forward pass
    CostType* ptr_pre_node = norm_factor_temp.ptr_pixel(0, 0);
    CostType* ptr_cur_node = norm_factor_temp.ptr_pixel(1, 0);
    CostType* ptr_end_node = norm_factor_temp.ptr_pixel(width-1, height-1);
    int* ptr_coeff = propagation_coeff_x_->ptr_pixel(1, 0);
    CostType weight = 0.0;
    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        *ptr_cur_node++ += weight * (*ptr_pre_node++);
    }

    // backward pass
    ptr_pre_node = norm_factor.ptr_pixel(width-1, height-1);
    ptr_cur_node = norm_factor.ptr_pixel(width-2, height-1);
    ptr_end_node = norm_factor.ptr_pixel(0, 0);
    ptr_coeff = propagation_coeff_x_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        *ptr_cur_node-- += weight * (*ptr_pre_node--);
    }

    CostType* ptr_norm = norm_factor.data();
    CostType* end_norm = norm_factor.ptr_pixel(width-1, height-1);
    CostType* ptr_norm_temp = norm_factor_temp.data();
    while(ptr_norm <= end_norm) {
        *ptr_norm++ += (*ptr_norm_temp++ - 1.0);
    }
}

template<typename CostType>
void HTreeCostPropagation<CostType>::
compute_norm_factor_v(Image<CostType>& norm_factor) {

    int width = img_width_, height = img_height_;
    // norm_factor.dataset(1.0);
    Image<CostType> norm_factor_temp(norm_factor);
    Image<CostType> norm_factor_back(norm_factor);
    // vertical phase
    // forward-backward algorithm

    // forward pass
    CostType* ptr_pre_node = norm_factor_temp.ptr_pixel(0, 0);
    CostType* ptr_cur_node = norm_factor_temp.ptr_pixel(0, 1);  // the next row
    CostType* ptr_end_node = norm_factor_temp.ptr_pixel(width-1, height-1);
    int* ptr_coeff = propagation_coeff_y_->ptr_pixel(0, 1);
    CostType weight = 0.0;
    while(ptr_cur_node <= ptr_end_node) {
        weight = lut_[*ptr_coeff++];
        *ptr_cur_node++ += weight * (*ptr_pre_node++);
    }

    // backward pass
    ptr_pre_node = norm_factor.ptr_pixel(width-1, height-1);
    ptr_cur_node = norm_factor.ptr_pixel(width-1, height-2);
    ptr_end_node = norm_factor.data();
    ptr_coeff = propagation_coeff_y_->ptr_pixel(width-1, height-1);
    while(ptr_cur_node >= ptr_end_node) {
        weight = lut_[*ptr_coeff--];
        *ptr_cur_node-- += weight * (*ptr_pre_node--);
    }

    CostType* ptr_norm = norm_factor.data();
    CostType* end_norm = norm_factor.ptr_pixel(width-1, height-1);
    CostType* ptr_norm_temp = norm_factor_temp.data();
    CostType* ptr_norm_back = norm_factor_back.data();
    while(ptr_norm <= end_norm) {
        *ptr_norm++ += (*ptr_norm_temp++ - *ptr_norm_back++);
    }
}

}  // namespace cvlab

#endif  // HORIZ_TREE_COST_AGGREGATION_H_
