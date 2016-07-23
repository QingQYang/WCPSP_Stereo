// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// These codes are adapt from the code for "Efficient Joint Segmentation,
// Occlusion Labeling" by Koichiro Yamaguchi et al..
// http://ttic.uchicago.edu/~dmcallester/SPS/index.html

#include "match_cost.h"
#include <nmmintrin.h>

namespace cvlab {

MatchCostGradientAndCensus::MatchCostGradientAndCensus() :
        aggregate_window_radius_(2),
        census_window_radius_(2),
        sobel_cap_value_(15),
        census_weight_factor_(1.0/6) {
}

MatchCostGradientAndCensus::~MatchCostGradientAndCensus() {}

void MatchCostGradientAndCensus::
init(const ImageU8& left_image, const ImageU8& right_image,
     int disp_range) {

    // set image size
    width_ = left_image.width();
    height_ = left_image.height();
    if (right_image.width() != width_ || right_image.height() != height_) {
		throw std::invalid_argument("[MatchCostGradientAndCensus::init] sizes of left and right images are different.");
    }
    width_step_ = width_ + 15 - (width_-1)%16;
    disp_range_ = disp_range;

    // calculate buffer size
    pixelwise_cost_row_buff_size_ = width_*disp_range_;
    row_aggregate_cost_buff_size_ = width_*disp_range_*(aggregate_window_radius_*2+2);
    half_pixel_right_buff_size_ = width_step_;

    // allocate buffers
    left_cost_image_ = reinterpret_cast<uint16_t*>(
        _mm_malloc(width_*height_*disp_range_*sizeof(uint16_t), 16));
    right_cost_image_ = reinterpret_cast<uint16_t*>(
        _mm_malloc(width_*height_*disp_range_* sizeof(uint16_t), 16));

    pixelwise_cost_row_ = reinterpret_cast<unsigned char*>(
        _mm_malloc(pixelwise_cost_row_buff_size_*sizeof(unsigned char), 16));
    row_aggregated_cost_ = reinterpret_cast<uint16_t*>(
        _mm_malloc(row_aggregate_cost_buff_size_*sizeof(uint16_t), 16));
    half_pixel_right_min_ = reinterpret_cast<unsigned char*>(
        _mm_malloc(half_pixel_right_buff_size_*sizeof(unsigned char), 16));
    half_pixel_right_max_ = reinterpret_cast<unsigned char*>(
        _mm_malloc(half_pixel_right_buff_size_*sizeof(unsigned char), 16));
}

void MatchCostGradientAndCensus::free_buffer() {
    _mm_free(left_cost_image_);
    _mm_free(right_cost_image_);
    _mm_free(pixelwise_cost_row_);
    _mm_free(row_aggregated_cost_);
    _mm_free(half_pixel_right_min_);
    _mm_free(half_pixel_right_max_);
}

void MatchCostGradientAndCensus::
compute_left_cost_image(unsigned char* left_grayscale_image,
                        unsigned char* right_grayscale_image) {
    // compute sobel image
    int width = width_;
    int height = height_;
    int width_step = width_step_;
    int disp_range = disp_range_;
    unsigned char* left_sobel_image = reinterpret_cast<unsigned char*>(
        _mm_malloc(width_step*height*sizeof(unsigned char), 16));
    unsigned char* right_sobel_image = reinterpret_cast<unsigned char*>(
        _mm_malloc(width_step*height*sizeof(unsigned char), 16));

    compute_capped_sobel_image(left_sobel_image, left_grayscale_image, false);
    compute_capped_sobel_image(right_sobel_image, right_grayscale_image, true);

    int* left_census_image = new int[width*height];
    int* right_census_image = new int[width*height];
    compute_census_image(left_census_image, left_grayscale_image);
    compute_census_image(right_census_image, right_grayscale_image);

    unsigned char* left_sobel_row = left_sobel_image;
    unsigned char* right_sobel_row = right_sobel_image;
    int* left_census_row = left_census_image;
    int* right_census_row = right_census_image;
    uint16_t* cost_image_row = left_cost_image_;
    compute_top_row_cost(cost_image_row,
                         left_sobel_row, left_census_row,
                         right_sobel_row, right_census_row);

    cost_image_row += width*disp_range;
    compute_row_cost(cost_image_row,
                     left_sobel_row, left_census_row,
                     right_sobel_row, right_census_row);

    // release buffer
    _mm_free(left_sobel_image);
    _mm_free(right_sobel_image);
    delete []left_census_image;
    delete []right_census_image;
}

void MatchCostGradientAndCensus::
compute_right_cost_image() {
    const int width_step_cost = width_*disp_range_;
	for(int y = 0; y < height_; y++) {
		uint16_t* left_cost_row = left_cost_image_ + width_step_cost*y;
		uint16_t* right_cost_row = right_cost_image_ + width_step_cost*y;

		for(int x = 0; x < disp_range_; x++) {
			uint16_t* ptr_left_cost = left_cost_row + disp_range_*x;
			uint16_t* ptr_right_cost = right_cost_row + disp_range_*x;
			for(int d = 0; d <= x; d++) {
				*(ptr_right_cost) = *(ptr_left_cost);
				ptr_right_cost -= disp_range_ - 1;
				++ptr_left_cost;
			}
		}

		for(int x = disp_range_; x < width_; x++) {
			uint16_t* ptr_left_cost = left_cost_row + disp_range_*x;
			uint16_t* ptr_right_cost = right_cost_row + disp_range_*x;
			for(int d = 0; d < disp_range_; d++) {
				*(ptr_right_cost) = *(ptr_left_cost);
				ptr_right_cost -= disp_range_ - 1;
				++ptr_left_cost;
			}
		}

		for(int x = width_-disp_range_+1; x < width_; x++) {
			int max_disp_index = width_ - x;
			uint16_t last_value = *(right_cost_row + disp_range_*x + max_disp_index - 1);

			uint16_t* ptr_right_cost = right_cost_row + disp_range_*x + max_disp_index;
			for(int d = max_disp_index; d < disp_range_; d++) {
				*(ptr_right_cost) = last_value;
				++ptr_right_cost;
			}
		}
	}
}

void MatchCostGradientAndCensus::
compute_capped_sobel_image(unsigned char* sobel_image,
                           const unsigned char* image,
                           bool horizontal_flip) {

    int width = width_;
    int height = height_;
    int width_step = width_step_;

    memset(sobel_image, sobel_cap_value_, width_step*height);

    if(horizontal_flip) {
        for(int y = 1; y < height-1; y++) {
            for(int x = 1; x < width-1; x++) {
                int sobel_value =
                        (image[width*(y-1)+x+1] + 2*image[width*y+x+1] + image[width*(y+1)+x+1]) -
                        (image[width*(y-1)+x-1] + 2*image[width*y+x-1] + image[width*(y+1)+x-1]);
                if(sobel_value > sobel_cap_value_)
                    sobel_value = 2*sobel_cap_value_;
                else if(sobel_value < -sobel_cap_value_)
                    sobel_value = 0;
                else
                    sobel_value += sobel_cap_value_;
                sobel_image[width_step*y + width-x-1] = sobel_value;
            }
        }
    } else {
        for(int y = 1; y < height-1; y++) {
            for(int x = 1; x < width-1; x++) {
                int sobel_value =
                        (image[width*(y-1)+x+1] + 2*image[width*y+x+1] + image[width*(y+1)+x+1]) -
                        (image[width*(y-1)+x-1] + 2*image[width*y+x-1] + image[width*(y+1)+x-1]);
                if(sobel_value > sobel_cap_value_)
                    sobel_value = 2*sobel_cap_value_;
                else if(sobel_value < -sobel_cap_value_)
                    sobel_value = 0;
                else
                    sobel_value += sobel_cap_value_;
                sobel_image[width_step*y + x] = sobel_value;
            }
        }
    }
}

void MatchCostGradientAndCensus::
compute_census_image(int* census_image,
                     const unsigned char* image) {
    int width = width_;
    int height = height_;
    int window_radius = census_window_radius_;
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            unsigned char center_value = image[width*y+x];

            int census_code = 0;
            for(int offsetY = -window_radius; offsetY <= window_radius; offsetY++) {
                for(int offsetX = -window_radius; offsetX <= window_radius; offsetX++) {
                    census_code = census_code << 1;
                    if(y+offsetY >= 0 && y+offsetY < height &&
                       x+offsetX >= 0 && x+offsetX < width &&
                       image[width*(y+offsetY)+x+offsetX] >= center_value) {
                        census_code += 1;
                    }
                }
            }
            census_image[width*y + x] = census_code;
        }
    }
}

void MatchCostGradientAndCensus::
compute_top_row_cost(uint16_t* cost_image_row,
                     unsigned char*& left_sobel_row, int*& left_census_row,
                     unsigned char*& right_sobel_row, int*& right_census_row) {

    int width = width_;
    int height = height_;
    int disp_range = disp_range_;
    int window_radius = aggregate_window_radius_;

    for(int row_index = 0; row_index <= window_radius; row_index++) {
        // row aggregated cost index
        int row_aggr_cost_index =
                std::min(row_index, height-1) % (window_radius*2+2);
        // row aggregated cost current
        uint16_t* row_aggr_cost_cur =
                row_aggregated_cost_ + row_aggr_cost_index*width*disp_range;

        compute_pixelwise_sad(left_sobel_row, right_sobel_row);
        add_pixelwise_hamming(left_census_row, right_census_row);

        memset(row_aggr_cost_cur, 0, disp_range*sizeof(uint16_t));
        // x = 0
        for(int x = 0; x <= window_radius; x++) {
            int scale = (x == 0 ? window_radius+1 : 1);
            for(int d = 0; d < disp_range; d++) {
                row_aggr_cost_cur[d] +=
                        static_cast<uint16_t>(pixelwise_cost_row_[disp_range*x+d]*scale);
            }
        }
        // x = 1...width-1
        for(int x = 1; x < width; x++) {
            const unsigned char* add_pixelwise_cost =
                    pixelwise_cost_row_ + std::min((x+window_radius), (width-1))*disp_range;
            const unsigned char* sub_pixelwise_cost =
                    pixelwise_cost_row_ + std::max((x-window_radius-1)*disp_range, 0);

            for(int d = 0; d < disp_range; d++) {
                row_aggr_cost_cur[disp_range*x+d]
                        = static_cast<uint16_t>(row_aggr_cost_cur[disp_range*(x-1)+d]
                                                + add_pixelwise_cost[d] - sub_pixelwise_cost[d]);
            }
        }

        // add to cost
        int scale = (row_index == 0 ? window_radius+1 : 1);
        for(int i = 0; i < width*disp_range; i++) {
            cost_image_row[i] += row_aggr_cost_cur[i]*scale;
        }

        left_sobel_row += width_step_;
        right_sobel_row += width_step_;
        left_census_row += width;
        right_census_row += width;
    }
}

void MatchCostGradientAndCensus::
compute_row_cost(unsigned short* cost_image_row,
                 unsigned char*& left_sobel_row, int*& left_census_row,
                 unsigned char*& right_sobel_row, int*& right_census_row) {

	const int width_step_cost = width_*disp_range_;
	const __m128i register_zero = _mm_setzero_si128();

	for(int y = 1; y < height_; y++) {
		int add_row_index = y + aggregate_window_radius_;
		int add_row_aggr_cost_index =
                std::min(add_row_index, height_-1)%(aggregate_window_radius_*2+2);
		unsigned short* add_row_aggregated_cost =
                row_aggregated_cost_ + width_*disp_range_*add_row_aggr_cost_index;

		if (add_row_index < height_) {
			compute_pixelwise_sad(left_sobel_row, right_sobel_row);
			add_pixelwise_hamming(left_census_row, right_census_row);

			memset(add_row_aggregated_cost, 0, disp_range_*sizeof(uint16_t));
			// x = 0
			for(int x = 0; x <= aggregate_window_radius_; x++) {
				int scale = (x == 0 ? aggregate_window_radius_ + 1 : 1);
				for(int d = 0; d < disp_range_; d++) {
					add_row_aggregated_cost[d] +=
                            static_cast<uint16_t>(pixelwise_cost_row_[disp_range_*x+d]*scale);
				}
			}
			// x = 1...width-1
			int sub_row_aggr_cost_index =
                    std::max(y-aggregate_window_radius_-1, 0)%(aggregate_window_radius_*2+2);
			const uint16_t* sub_row_aggregated_cost =
                    row_aggregated_cost_ + width_*disp_range_*sub_row_aggr_cost_index;
			const uint16_t* previous_cost_row = cost_image_row - width_step_cost;
			for(int x = 1; x < width_; x++) {
				const unsigned char* add_pixelwise_cost =
                        pixelwise_cost_row_ + std::min((x+aggregate_window_radius_),
                                                       (width_-1))*disp_range_;
				const unsigned char* sub_pixelwise_cost =
                        pixelwise_cost_row_ + std::max((x-aggregate_window_radius_-1)*disp_range_, 0);

				for(int d = 0; d < disp_range_; d += 16) {
					__m128i register_add_pixelwise_low =
                            _mm_load_si128(reinterpret_cast<const __m128i*>(add_pixelwise_cost + d));
					__m128i register_add_pixelwise_high =
                            _mm_unpackhi_epi8(register_add_pixelwise_low, register_zero);
					register_add_pixelwise_low =
                            _mm_unpacklo_epi8(register_add_pixelwise_low, register_zero);
					__m128i register_sub_pixelwise_low =
                            _mm_load_si128(reinterpret_cast<const __m128i*>(sub_pixelwise_cost + d));
					__m128i register_sub_pixelwise_high =
                            _mm_unpackhi_epi8(register_sub_pixelwise_low, register_zero);
					register_sub_pixelwise_low =
                            _mm_unpacklo_epi8(register_sub_pixelwise_low, register_zero);

					// Low
					__m128i register_add_aggregated =
                            _mm_load_si128(reinterpret_cast<const __m128i*>(
                                add_row_aggregated_cost + disp_range_*(x-1)+d));
					register_add_aggregated = _mm_adds_epi16(
                        _mm_subs_epi16(register_add_aggregated, register_sub_pixelwise_low),
                        register_add_pixelwise_low);
					__m128i register_cost =
                            _mm_load_si128(reinterpret_cast<const __m128i*>(
                                previous_cost_row + disp_range_*x+d));
					register_cost = _mm_adds_epi16(
                        _mm_subs_epi16(register_cost,
                                       _mm_load_si128(reinterpret_cast<const __m128i*>(
                                           sub_row_aggregated_cost + disp_range_*x+d))),
                        register_add_aggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(
                        add_row_aggregated_cost + disp_range_*x+d), register_add_aggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(
                        cost_image_row + disp_range_*x+d), register_cost);

					// High
					register_add_aggregated =
                            _mm_load_si128(reinterpret_cast<const __m128i*>(
                                add_row_aggregated_cost + disp_range_*(x-1)+d+8));
					register_add_aggregated = _mm_adds_epi16(
                        _mm_subs_epi16(register_add_aggregated, register_sub_pixelwise_high),
                        register_add_pixelwise_high);
					register_cost = _mm_load_si128(reinterpret_cast<const __m128i*>(
                        previous_cost_row + disp_range_*x+d+8));
					register_cost = _mm_adds_epi16(
                        _mm_subs_epi16(register_cost,
                                       _mm_load_si128(reinterpret_cast<const __m128i*>(
                                           sub_row_aggregated_cost + disp_range_*x + d + 8))),
                        register_add_aggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(
                        add_row_aggregated_cost + disp_range_*x+d+8), register_add_aggregated);
					_mm_store_si128(reinterpret_cast<__m128i*>(
                        cost_image_row + disp_range_*x+d+8), register_cost);
				}
			}
		}

		left_sobel_row += width_step_;
		right_sobel_row += width_step_;
		left_census_row += width_;
		right_census_row += width_;
		cost_image_row += width_step_cost;
	}
}

void MatchCostGradientAndCensus::
compute_pixelwise_sad(const unsigned char* left_sobel_row,
                      const unsigned char* right_sobel_row) {

	compute_half_pixel_right(right_sobel_row);

	for(int x = 0; x < 16; x++) {
		int left_center_value = left_sobel_row[x];
		int left_half_left_value = x > 0 ? (left_center_value + left_sobel_row[x-1])/2 :
                                   left_center_value;
		int left_half_right_value = x < width_-1 ? (left_center_value + left_sobel_row[x+1])/2 :
                                    left_center_value;
		int left_min_value = std::min(left_half_left_value, left_half_right_value);
		left_min_value = std::min(left_min_value, left_center_value);
		int left_max_value = std::max(left_half_left_value, left_half_right_value);
		left_max_value = std::max(left_max_value, left_center_value);

		for(int d = 0; d <= x; d++) {
			int right_center_value = right_sobel_row[width_-1-x+d];
			int right_min_value = half_pixel_right_min_[width_-1-x+d];
			int right_max_value = half_pixel_right_max_[width_-1-x+d];

			int cost_l2r = std::max(0, left_center_value - right_max_value);
			cost_l2r = std::max(cost_l2r, right_min_value - left_center_value);
			int cost_r2l = std::max(0, right_center_value - left_max_value);
			cost_r2l = std::max(cost_r2l, left_min_value - right_center_value);
			int cost_value = std::min(cost_l2r, cost_r2l);

			pixelwise_cost_row_[disp_range_*x+d] = cost_value;
		}
		for(int d = x+1; d < disp_range_; d++) {
			pixelwise_cost_row_[disp_range_*x+d] =
                    pixelwise_cost_row_[disp_range_*x+d-1];
		}
	}
	for(int x = 16; x < disp_range_; x++) {
		int left_center_value = left_sobel_row[x];
		int left_half_left_value =
                x > 0 ? (left_center_value + left_sobel_row[x-1])/2 : left_center_value;
		int left_half_right_value =
                x < width_ - 1 ? (left_center_value + left_sobel_row[x+1])/2 : left_center_value;
		int left_min_value = std::min(left_half_left_value, left_half_right_value);
		left_min_value = std::min(left_min_value, left_center_value);
		int left_max_value = std::max(left_half_left_value, left_half_right_value);
		left_max_value = std::max(left_max_value, left_center_value);

		__m128i register_left_center_value =
                    _mm_set1_epi8(static_cast<char>(left_center_value));
		__m128i register_left_min_value =
                    _mm_set1_epi8(static_cast<char>(left_min_value));
		__m128i register_left_max_value =
                    _mm_set1_epi8(static_cast<char>(left_max_value));

		for (int d = 0; d < x/16; d += 16) {
			__m128i register_right_center_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        right_sobel_row+width_-1-x+d));
			__m128i register_right_min_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        half_pixel_right_min_+width_-1-x+d));
			__m128i register_right_max_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        half_pixel_right_max_+width_-1-x+d));

			__m128i register_cost_l2r = _mm_max_epu8(
                _mm_subs_epu8(register_left_center_value, register_right_max_value),
                _mm_subs_epu8(register_right_min_value, register_left_center_value));
			__m128i register_cost_r2l = _mm_max_epu8(
                _mm_subs_epu8(register_right_center_value, register_left_max_value),
                _mm_subs_epu8(register_left_min_value, register_right_center_value));
			__m128i register_cost = _mm_min_epu8(register_cost_l2r, register_cost_r2l);

			_mm_store_si128(reinterpret_cast<__m128i*>(
                pixelwise_cost_row_ + disp_range_*x + d), register_cost);
		}
		for (int d = x/16; d <= x; d++) {
			int right_center_value = right_sobel_row[width_-1-x+d];
			int right_min_value = half_pixel_right_min_[width_-1-x+d];
			int right_max_value = half_pixel_right_max_[width_-1-x+d];

			int cost_l2r = std::max(0, left_center_value - right_max_value);
			cost_l2r = std::max(cost_l2r, right_min_value - left_center_value);
			int cost_r2l = std::max(0, right_center_value - left_max_value);
			cost_r2l = std::max(cost_r2l, left_min_value - right_center_value);
			int cost_value = std::min(cost_l2r, cost_r2l);

			pixelwise_cost_row_[disp_range_*x + d] = cost_value;
		}
		for (int d = x + 1; d < disp_range_; d++) {
			pixelwise_cost_row_[disp_range_*x+d] =
                    pixelwise_cost_row_[disp_range_*x+d-1];
		}
	}
	for (int x = disp_range_; x < width_; x++) {
		int left_center_value = left_sobel_row[x];
		int left_half_left_value =
                x > 0 ? (left_center_value + left_sobel_row[x-1])/2 :
                left_center_value;
		int left_half_right_value =
                x < width_ - 1 ? (left_center_value + left_sobel_row[x+1])/2 :
                left_center_value;
		int left_min_value = std::min(left_half_left_value, left_half_right_value);
		left_min_value = std::min(left_min_value, left_center_value);
		int left_max_value = std::max(left_half_left_value, left_half_right_value);
		left_max_value = std::max(left_max_value, left_center_value);

		__m128i register_left_center_value =
                    _mm_set1_epi8(static_cast<char>(left_center_value));
		__m128i register_left_min_value =
                    _mm_set1_epi8(static_cast<char>(left_min_value));
		__m128i register_left_max_value =
                    _mm_set1_epi8(static_cast<char>(left_max_value));

		for (int d = 0; d < disp_range_; d += 16) {
			__m128i register_right_center_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        right_sobel_row + width_-1-x+d));
			__m128i register_right_min_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        half_pixel_right_min_ + width_-1-x+d));
			__m128i register_right_max_value =
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                        half_pixel_right_max_ + width_-1-x+d));

			__m128i register_cost_l2r = _mm_max_epu8(
                _mm_subs_epu8(register_left_center_value, register_right_max_value),
                _mm_subs_epu8(register_right_min_value, register_left_center_value));
			__m128i register_cost_r2l = _mm_max_epu8(
                _mm_subs_epu8(register_right_center_value, register_left_max_value),
                _mm_subs_epu8(register_left_min_value, register_right_center_value));
			__m128i register_cost = _mm_min_epu8(register_cost_l2r, register_cost_r2l);

			_mm_store_si128(reinterpret_cast<__m128i*>(
                pixelwise_cost_row_ + disp_range_*x+d), register_cost);
		}
	}
}

void MatchCostGradientAndCensus::
compute_half_pixel_right(const unsigned char* right_sobel_row) {
	for(int x = 0; x < width_; x++) {
		int center_value = right_sobel_row[x];
		int left_half_value =
                x > 0 ? (center_value + right_sobel_row[x-1])/2 : center_value;
		int right_half_value =
                x < width_-1 ? (center_value + right_sobel_row[x+1])/2 : center_value;
		int min_value = std::min(left_half_value, right_half_value);
		min_value = std::min(min_value, center_value);
		int max_value = std::max(left_half_value, right_half_value);
		max_value = std::max(max_value, center_value);

		half_pixel_right_min_[x] = min_value;
		half_pixel_right_max_[x] = max_value;
	}
}

void MatchCostGradientAndCensus::
add_pixelwise_hamming(const int* left_census_row, const int* right_census_row) {
	for(int x = 0; x < disp_range_; x++) {
		int left_census_code = left_census_row[x];
		int hamming_distance = 0;
		for(int d = 0; d <= x; d++) {
			int right_census_code = right_census_row[x-d];
			hamming_distance =
                    static_cast<int>(_mm_popcnt_u32(
                        static_cast<unsigned int>(left_census_code^right_census_code)));
			pixelwise_cost_row_[disp_range_*x+d] +=
                    static_cast<unsigned char>(hamming_distance*census_weight_factor_);
		}
		hamming_distance =
                static_cast<unsigned char>(hamming_distance*census_weight_factor_);
		for(int d = x + 1; d < disp_range_; d++) {
			pixelwise_cost_row_[disp_range_*x+d] += hamming_distance;
		}
	}
	for(int x = disp_range_; x < width_; x++) {
		int left_census_code = left_census_row[x];
		for(int d = 0; d < disp_range_; d++) {
			int right_census_code = right_census_row[x-d];
			int hamming_distance =
                    static_cast<int>(_mm_popcnt_u32(
                        static_cast<uint32_t>(left_census_code^right_census_code)));
			pixelwise_cost_row_[disp_range_*x+d] +=
                    static_cast<unsigned char>(hamming_distance*census_weight_factor_);
		}
	}
}

}  // namespace cvlab
