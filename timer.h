// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c) Qingqing Yang

// Description : Util functions for compute elapsed time
// Others : This file is based on the one originally written by Song Ho Ahn.

#ifndef TIMER_H_
#define TIMER_H_

#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif


class Timer {
public:
    Timer();  // default constructor
    ~Timer(); // default destructor

    void   tic();                         // start timer
    void   toc();                          // stop the timer
    double get_elapsed_time();              // get elapsed time in second
    double get_elapsed_time_in_sec();       // get elapsed time in second,
    // (same as get_elapsed_time)
    double get_elapsed_time_in_milli_sec();  // get elapsed time in milli-second
    double get_elapsed_time_in_micro_sec();  // get elapsed time in micro-second

 protected:

 private:
    double start_time_in_micro_sec_;  // starting time in micro-second
    double end_time_in_micro_sec_;    // ending time in micro-second
    int    stopped_;                  // stop flag
#ifdef WIN32
    LARGE_INTEGER frequency_;   // ticks per second
    LARGE_INTEGER start_count_;
    LARGE_INTEGER end_count_;
#else
    timeval start_count_;
    timeval end_count_;
#endif
};

#endif  // TIMER_H_
