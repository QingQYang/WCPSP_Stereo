// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description: Image class for most image processing methods.

#ifndef IMAGE_H_
#define IMAGE_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

typedef unsigned char uchar;

namespace cvlab {

// Padding (border) mode for neighborhood operations
enum BorderMode {
    BORDER_ZERO = 0,      // zero padding
    BORDER_REPLICATE = 1, // replicate border values
    BORDER_REFLECT = 2,   // reflect border pixels
    BORDER_CYCLIC = 3,    // wrap pixel values
};

struct ImageProperty {
    int origin[2];  // x and y coordinate origin
    BorderMode border_mode;
};

// Image class
template <class T>
class Image {
  public:
    Image();
    Image(int width, int height, int channels, bool init = true);
    Image(const Image<T>& image);  // copy constructer
    template <class U> Image(const Image<U>& image);

    ~Image();

    void realloc();
    void realloc(int width, int height, int channels, bool init = true);
    // void resize(int width, int height, int channels);
    void resize_channels(int channels);

    void scale(double fact);
    void dataset(T value);
    template <class U> void copy_data_from(const Image<U>& src);

    // member related functions

    int width() const {return width_;}
    int height() const {return height_;}
    int channels() const {return channels_;}
    BorderMode border_mode() const {return border_mode_;}

    T* ptr_pixel(int x, int y) const {return &(access_[y][x*channels_]);}
    T& pixel(int x, int y) const {return access_[y][x*channels_];}
    T& pixel(int x, int y, int z) const {return access_[y][x*channels_+z];}
    T* ptr_first_pixel() const {return data_;}
    T* ptr_last_pixel() const {return &(access_[height_-1][(width_-1)*channels_]);}
    T* data() const {return data_;}
    T* ptr_last_data() const {return data_ + width_*height_*channels_-1;}

    // some operators
    Image<T>& operator=(const Image<T>& rhs) {
        if(this == &rhs) {
            return *this;
        }
        realloc(rhs.width(), rhs.height(), rhs.channels());
        memcpy(data_, rhs.data(), sizeof(T)*width_*height_*channels_);
        return *this;
    }

    template <class U>
    Image<T>& operator=(const Image<U>& rhs) {
        if(this == &rhs) {
            return *this;
        }
        realloc(rhs.width(), rhs.height(), rhs.channels());
        this->copy_data_from(rhs);
        return *this;
    }

    Image<T>& operator+=(const Image<T>& rhs);
    Image<T>& operator-=(const Image<T>& rhs);

    Image<T>& operator+=(T rhs) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        while(ptr <= end) {
            *ptr++ += rhs;
        }
        return *this;
    }

    Image<T>& operator-=(T rhs) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        while(ptr <= end) {
            *ptr++ -= rhs;
        }
        return *this;
    }

    const Image<T> operator+(const Image<T>& other) {
        Image<T> result = *this;
        result += other;
        return result;
    }

    const Image<T> operator-(const Image<T>& other) {
        Image<T> result = *this;
        result -= other;
        return result;
    }

    const Image<T> operator+(T other) {
        Image<T> result = *this;
        result += other;
        return result;
    }

    const Image<T> operator-(T other) {
        Image<T> result = *this;
        result -= other;
        return result;
    }

    // element wise devide and multiply
    Image<T>& elem_wise_devide(const Image<T>& divisor);
    Image<T>& elem_wise_multiply(const Image<T>& multiplier);

    // != returns true if either the size or the data point are not the same
    bool operator!=(const Image<T>& other) const {
        return (this != &other ||
                this->data_ != other.data() ||
                this->width_ != other.width() ||
                this->height_ != other.height() ||
                this->channels_ != other.channels());
    }

    bool operator==(const Image<T>& other) const {
        return !(*this != other);
    }

    bool same_shape(const Image<T>& other) const {
        return (other.width() == width_ &&
                other.height() ==  height_ &&
                other.channels() == channels_);
    }

    template <class U>
    bool same_shape(const Image<U>& other) const {
        return (other.width() == width_ &&
                other.height() ==  height_ &&
                other.channels() == channels_);
    }

  protected:
    void set_shape(int width, int height, int channels);

  private:
    T* data_;  // image data
    T** access_;  // row pointers

    // image property
    int width_;
    int height_;
    int channels_;

    int origin_[2];  // x and y coordinate origin
    BorderMode border_mode_;
};

template <class T>
Image<T>::Image() {
    width_ = 0; height_ = 0; channels_ = 0;
    data_ = NULL; access_ = NULL;
    origin_[0] = 0; origin_[1] = 0;
    border_mode_ = BORDER_ZERO;
}

template <class T>
Image<T>::Image(int width, int height, int channels, bool init) {
    width_ = width;
    height_ = height;
    channels_ = channels;

    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for(int h = 0; h < height_; h++) {
        access_[h] = data_ + (h*width_*channels_);
    }
    origin_[0] = 0;
    origin_[1] = 0;
    border_mode_ = BORDER_ZERO;
    // init to 0
    if(init) {
        memset(data_, 0, width_* height_*channels_* sizeof(T));
    }

}

template <class T>
Image<T>::Image(const Image<T>& image) {  // copy constructer
    width_ = image.width();
    height_ = image.height();
    channels_ = image.channels();
    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for(int h = 0; h < height_; h++) {
        access_[h] = data_ + (h*width_*channels_);
    }
    origin_[0] = image.origin_[0];
    origin_[1] = image.origin_[1];
    border_mode_ = image.border_mode_;
    memcpy(data_, image.data(), width_* height_*channels_* sizeof(T));
}


template <class T> template <class U>
Image<T>::Image(const Image<U>& image) {
    width_ = image.width();
    height_ = image.height();
    channels_ = image.channels();
    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for(int h = 0; h < height_; h++) {
        access_[h] = data_ + (h*width_*channels_);
    }
    this->copy_data_from(image);
}

template <class T>
Image<T>::~Image() {
    delete []data_;
    delete []access_;
}

template <class T>
void Image<T>::realloc() {
    delete []data_;
    delete []access_;
    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for(int h = 0; h < height_; h++) {
        access_[h] = data_ + (h*width_*channels_);
    }
}

template <class T>
void Image<T>::realloc(int width, int height, int channels, bool init) {
    set_shape(width, height, channels);
    delete []data_;
    delete []access_;
    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for(int h = 0; h < height_; h++) {
        access_[h] = data_ + (h*width_*channels_);
    }
    if (init) {
        memset(data_, 0, width_* height_*channels_* sizeof(T));
    }
}

template <class T>
void Image<T>::resize_channels(int channels) {
    Image<T> backup(*this);
    set_shape(width_, height_, channels);
    delete []data_;
    delete []access_;
    data_ = new T[width_*height_*channels_];
    access_ = new T*[height_];
    for (int y = 0; y < height_; ++y) {
        access_[y] = data_ + (y*width_*channels_);
    }
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            memcpy(ptr_pixel(x, y), backup.ptr_pixel(x, y),
                   std::min(channels_, backup.channels())*sizeof(T));
        }
    }
}

template <class T>
void Image<T>::scale(double fact) {
    T* ptr = ptr_pixel(0, 0);
    T* end = ptr_last_data();
    while(ptr <= end) {
        *ptr = static_cast<T>(*ptr * fact);
        ptr++;
    }
}

template <class T>
void Image<T>::dataset(T value) {
    T* ptr = data();
    T* end = ptr_last_data();
    while(ptr <= end) {
        *ptr++ = value;
    }
}

template <class T> template <class U>
void Image<T>::copy_data_from(const Image<U>& src) {
    if(this->same_shape(src)) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        const U* ptr_src = src.ptr_pixel(0, 0);
        while(ptr <= end) {
            *ptr++ = static_cast<T>(*ptr_src++);
        }
    } else {
        // TODO: error handling
        fprintf(stderr, "Image::copydatafrom() Error: "
                "Images are not the same size.\n");
    }
}

template <class T>
Image<T>& Image<T>::operator+=(const Image<T>& rhs) {
    if(this->same_shape(rhs)) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        const T* ptr_rhs = rhs.ptr_pixel(0, 0);
        while(ptr <= end) {
            *ptr++ += *ptr_rhs++;
        }
    } else {
        // TODO: error handling
        fprintf(stderr, "Image::elem_wise_devide() Error: "
                "Images are not the same shape.\n");
    }
    return *this;
}

template <class T>
Image<T>& Image<T>::operator-=(const Image<T>& rhs) {
    if(this->same_shape(rhs)) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        const T* ptr_rhs = rhs.ptr_pixel(0, 0);
        while(ptr <= end) {
            *ptr++ -= *ptr_rhs++;
        }
    } else {
        // TODO: error handling
        fprintf(stderr, "Image::elem_wise_devide() Error: "
                "Images are not the same shape.\n");
    }
    return *this;
}

template <class T>
Image<T>& Image<T>::elem_wise_devide(const Image<T>& devisor) {
    if(this->same_shape(devisor)) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        const T* ptr_devisor = devisor.ptr_pixel(0, 0);
        while(ptr <= end) {
            *ptr++ /= *ptr_devisor++;
        }
    } else {
        // TODO: error handling
        fprintf(stderr, "Image::elem_wise_devide() Error: Images are not the same shape.\n");
    }
    return *this;
}

template <class T>
Image<T>& Image<T>::elem_wise_multiply(const Image<T>& multiplier) {
    if(this->same_shape(multiplier)) {
        T* ptr = ptr_pixel(0, 0);
        T* end = ptr_pixel(width_-1, height_-1)+(channels_-1);
        const T* ptr_multiplier = multiplier.ptr_pixel(0, 0);
        while(ptr <= end) {
            *ptr++ *= *ptr_multiplier++;
        }
    } else {
        // TODO: error handling
        fprintf(stderr, "Image::elem_wise_devide() Error: Images are not the same shape.\n");
    }
    return *this;
}

template <class T>
void Image<T>::set_shape(int width, int height, int channels) {
    width_ = width; height_ = height; channels_ = channels;
}


// other functions
template <class T>
const Image<T> elem_wise_multiply(const Image<T>& lhs, const Image<T>& rhs) {
    Image<T> result = lhs;
    result.elem_wise_multiply(rhs);
    return result;
}

template <class T>
const Image<T> elem_wise_devide(const Image<T>& lhs, const Image<T>& rhs) {
    Image<T> result = lhs;
    result.elem_wise_devide(rhs);
    return result;
}

typedef Image<unsigned char> ImageU8;
typedef Image<uint16_t> ImageU16;
typedef Image<float> ImageF32;
typedef Image<double> ImageF64;

}  // namespace cvlab

#endif  // IMAGE_IMAGE_H_
