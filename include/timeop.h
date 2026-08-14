#pragma once

#include <cstdlib>
#include <ctime>

#if defined(_WIN32)
#include <Windows.h>
inline double operator - (const LARGE_INTEGER &t1, const LARGE_INTEGER &t2) {
    auto freq = LARGE_INTEGER{};
    QueryPerformanceFrequency(&freq);
    return double(t1.QuadPart - t2.QuadPart) / freq.QuadPart;
}
#else
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
#endif