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

int main(int argc, char *argv[]) {
    auto in_file = boost::iostreams::mapped_file_source{"sample1_recon_Export.bin"};
    auto it      = in_file.begin();
    auto end     = it + in_file.size();
    auto ia      = io::BinIArchive{it, end};

    auto original_img = Field<uint16_t>{};
    ia >> original_img;

    auto [lo, hi] = find_intensity_range(original_img.view(), 0.001);
    fmt::print("used lo = {}\n", lo);
    fmt::print("used hi = {}\n", hi);

    static auto constexpr col = 481;
    static auto constexpr 𝜆   = 0.04f;

    auto img = convert_to_fp(original_img.view(), lo, hi);
    write_png(img.view(), col, 0.0f);

    Field<uint16_t>{}.swap(original_img); // free memory

    const auto nbins = 500;
    auto hist = calc_hist(img.view(), 1.0 / (hi - lo), nbins);
    auto f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen("original-hist.txt", "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) / nbins, hist[i]);
    }

    chambolle(img.view(), 𝜆, 1.0f / 6.0f * 0.99f, 150);

    hist = calc_hist(img.view(), 1.0 / (hi - lo), nbins);
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
