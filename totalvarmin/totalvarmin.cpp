#include <cstdint>
#include <ctime>
#include <memory>
#include <tuple>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/math/constants/constants.hpp>

#include <fmt/printf.h>
#include <fmt/color.h>

#include <opencv2/opencv.hpp>

#include <io/fileinit.h>
#include <io/biniarchive.h>
#include <tiffio.h>

#include <cuda-context.h>
#include <field.h>
#include <timeop.h>

#include "find_intensity_range.h"
#include "convert_to_fp.h"
#include "calc_hist.h"
#include "chambolle.h"

namespace bc = boost::math::double_constants;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);

template <typename ElementType, typename Extents>
void write_png(std::mdspan<ElementType, Extents, layout_cylinder> img, uint32_t col, float 𝜆) {
    auto m = img.mapping();
    auto slice = cv::Mat(
        m.row_end(col) - m.row_begin(col)
      , m.n_images()
      , CV_32FC1
      , &img[0, m.row_begin(col), col]
      , m.nstride() * 4
      );
    auto gamma_corrected = cv::Mat{};
    cv::pow(slice, 1.0 / 2.2, gamma_corrected);
    auto vis = cv::Mat{};
    gamma_corrected.convertTo(vis, CV_8U, 255.0);
    cv::imwrite(fmt::format("slice-{} (𝜆 = {:.3f}).png", col, 𝜆), vis);
}

int main(int argc, char *argv[]) try {
    auto start_time   = timespec{};
    auto finish_time  = timespec{};
    auto elapsed_time = 0.0;

    auto in_file = boost::iostreams::mapped_file_source{"sample1_recon_Export.bin"};
    auto it      = in_file.begin();
    auto end     = it + in_file.size();
    auto ia      = io::BinIArchive{it, end};

    auto original_img = Field<uint16_t>{};
    clock_gettime(CLOCK_MONOTONIC, &start_time);
        ia >> original_img;
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Read time: {:.6g} s\n", elapsed_time);

    auto cuda_context = make_context();

    clock_gettime(CLOCK_MONOTONIC, &start_time);
        auto d_original_img = upload(*cuda_context, original_img.view());
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Upload to GPU time: {:.6g} s\n", elapsed_time);

    clock_gettime(CLOCK_MONOTONIC, &start_time);
        auto [lo, hi] = find_intensity_range(*cuda_context, *d_original_img, 0.001);
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Find intensity range time: {:.6g} s\n", elapsed_time);
    fmt::print("used lo = {}\n", lo); // 5399
    fmt::print("used hi = {}\n", hi); // 12524

    clock_gettime(CLOCK_MONOTONIC, &start_time);
        auto d_img = convert_to_fp(*cuda_context, *d_original_img, lo, hi);
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Convert to FP32 time: {:.6g} s\n", elapsed_time);
    d_original_img.reset();

    auto img = Field<float>{original_img.view().mapping()};
    clock_gettime(CLOCK_MONOTONIC, &start_time);
        download(*cuda_context, *d_img, img.view());
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Download from GPU time: {:.6g} s\n", elapsed_time);

    static auto constexpr col = 481;
    static auto constexpr 𝜆   = 0.05f;

    write_png(img.view(), col, 0.0f);

    const auto nbins = 500;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
        auto hist = calc_hist(*cuda_context, *d_img, 1.0 / (hi - lo), nbins);
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Histogram time: {:.6g} s\n", elapsed_time);
    auto f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen("original-hist.txt", "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) / nbins, hist[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &start_time);
        chambolle(*cuda_context, *d_img, 𝜆, 1.0f / 6.0f * 0.99f, 150);
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Chambolle time: {:.6g} s\n", elapsed_time);
    download(*cuda_context, *d_img, img.view());

    hist = calc_hist(*cuda_context, *d_img, 1.0 / (hi - lo), nbins);
    f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen(fmt::format("hist (𝜆 = {:.3f}).txt", 𝜆).c_str(), "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) / nbins, hist[i]);
    }

    write_png(img.view(), col, 𝜆);
}
catch (const std::exception& e) {
    fmt::print(stderr, "{}\n", e.what());
}
