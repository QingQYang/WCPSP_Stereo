// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description : Image processing functions.

#ifndef IMAGE_PROC_H_
#define IMAGE_PROC_H_

#include <algorithm>
#include "image.h"

namespace cvlab {

typedef unsigned char uchar;

using std::max;
using std::min;
// Comparison
template <typename T>
T max_in_3(T* a) {
    return (max(max(a[0], a[1]), a[2]));
}

template <typename T>
T max_in_3(T a, T b, T c) {
    return (max(max(a, b), c));
}

// image convert
// rgb2gray using luma coding
inline double rgb2gray(const uchar *rgb) {
    return (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]);
}

template<typename T>
inline void rgb2gray(Image<T>& gray_image,
                     const Image<uchar>& rgb_image) {
    T* ptr_gray = gray_image.ptr_pixel(0, 0);
    T* end_gray = gray_image.ptr_pixel(gray_image.width()-1,
                                       gray_image.height()-1);
    uchar* ptr_rgb = rgb_image.ptr_pixel(0, 0);
    while(ptr_gray != end_gray) {
        *ptr_gray++ = static_cast<T>(rgb2gray(ptr_rgb));
        ptr_rgb += 3;
    }
}

template<typename T>
inline void compute_gradient(Image<T>& gradient_image,
                             Image<uchar>& image) {
    T gray, gray_minus, gray_plus;
    int width = image.width();
    int height = image.height();
    int channels = image.channels();
    Image<T> gray_image(width, height, 1);
    if(channels == 1) {
        gray_image.copy_data_from(image);
    } else if(channels == 3) {
        rgb2gray(gray_image, image);
    } else {
        // TODO: other cases or error handling.
    }

    for(int y = 0; y < height; y++) {
        gray_minus = gray_image.pixel(0, y);
        gray = gray_plus = gray_image.pixel(1, y);
        gradient_image.pixel(0, y) = gray_plus - gray_minus + 127.5;
        for(int x = 1; x < width-1; x++) {
            gray_plus = gray_image.pixel(x+1, y);
            gradient_image.pixel(x, y) = 0.5*(gray_plus - gray_minus) + 127.5;
            gray_minus = gray;
            gray = gray_plus;
        }
        gradient_image.pixel(width-1, y) = gray_plus - gray_minus + 127.5;
    }
}

// copy a pixel depending on the channels a pixel have
template<typename T>
inline void copy_pixel(T* dst, const T* src, int channels) {
    for(int i = 0; i < channels; i++) {
        *dst++ = *src++;
    }
}

// compuate max diff in channels of two pixels
template<typename T, typename U>
inline T pixel_abs_diff_max_in_channel(U* a, U* b, int channels) {
    T diff = 0;
    if(channels == 1) {
        diff = abs(*a - *b);
    } else if(channels == 3) {
        T diff_1 = abs(a[0] - b[0]);
        T diff_2 = abs(a[1] - b[1]);
        T diff_3 = abs(a[2] - b[2]);
        diff = max_in_3(diff_1, diff_2, diff_3);
    } else {
        T temp_diff = 0;
        for(int i = 0; i < channels; i++) {
            temp_diff = abs(a[i] - b[i]);
            diff = std::max(diff, temp_diff);
        }
    }
    return diff;
}

}      // namespace cvlab

#endif  // IMAGE_PROC_H_
