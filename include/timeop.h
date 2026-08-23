#pragma once

#include <cstdlib>
#include <ctime>
#include <string_view>

#include <fmt/printf.h>
#include <fmt/color.h>

inline double operator - (const timespec &t1, const timespec &t2) {
    auto s  = int64_t{t1.tv_sec  - t2.tv_sec};
    auto ns = int64_t{t1.tv_nsec - t2.tv_nsec};
    if (0 <= s) {
        if (0 > ns) {
            --s;
            ns += 1'000'000'000;
        }
    } else {
        if (0 < ns) {
            ++s;
            ns -= 1'000'000'000;
        }
    }
    return s + ns * 1e-9;
}

struct Timer {
    auto start() {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }
    auto stop(std::string_view msg) {
        clock_gettime(CLOCK_MONOTONIC, &finish_time);
        auto elapsed_time = finish_time - start_time;
        fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
                   "{}: {:.6g} s\n", msg, elapsed_time);
    }
    timespec start_time{};
    timespec finish_time{};
};
