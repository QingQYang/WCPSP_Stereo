// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

#include "timer.h"
#include <stdlib.h>

Timer::Timer() {
#ifdef WIN32
    QueryPerformanceFrequency(&frequency_);
    start_count_.QuadPart = 0;
    end_count_.QuadPart = 0;
#else
    start_count_.tv_sec = start_count_.tv_usec = 0;
    end_count_.tv_sec = end_count_.tv_usec = 0;
#endif

    stopped_ = 0;
    start_time_in_micro_sec_ = 0;
    end_time_in_micro_sec_ = 0;
}

Timer::~Timer() {
}

void Timer::tic() {
    stopped_ = 0; // reset stop flag
#ifdef WIN32
    QueryPerformanceCounter(&start_count_);
#else
    gettimeofday(&start_count_, NULL);
#endif
}

void Timer::toc() {
    stopped_ = 1; // set timer stopped_ flag

#ifdef WIN32
    QueryPerformanceCounter(&end_count_);
#else
    gettimeofday(&end_count_, NULL);
#endif
}

double Timer::get_elapsed_time_in_micro_sec() {
#ifdef WIN32
    if(!stopped_)
        QueryPerformanceCounter(&end_count_);

    start_time_in_micro_sec_ = start_count_.QuadPart * (1000000.0 / frequency_.QuadPart);
    end_time_in_micro_sec_ = end_count_.QuadPart * (1000000.0 / frequency_.QuadPart);
#else
    if(!stopped_)
        gettimeofday(&end_count_, NULL);

    start_time_in_micro_sec_ = (start_count_.tv_sec * 1000000.0) + start_count_.tv_usec;
    end_time_in_micro_sec_ = (end_count_.tv_sec * 1000000.0) + end_count_.tv_usec;
#endif

    return end_time_in_micro_sec_ - start_time_in_micro_sec_;
}

double Timer::get_elapsed_time_in_milli_sec() {
    return this->get_elapsed_time_in_micro_sec() * 0.001;
}

double Timer::get_elapsed_time_in_sec() {
    return this->get_elapsed_time_in_micro_sec() * 0.000001;
}

double Timer::get_elapsed_time() {
    return this->get_elapsed_time_in_sec();
}
