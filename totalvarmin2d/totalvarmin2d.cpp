#include <array>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

#include <fmt/printf.h>

#include <opencv2/opencv.hpp>

#include <range/v3/algorithm/max_element.hpp>

#include <tiffio.h>

namespace rs = ranges;

void write_hist(cv::Mat const &g, cv::Mat const &mask, const char *file_name) {
    auto constexpr nbins = int{256};
    auto hist   = cv::Mat{};
    auto imgf32 = cv::Mat{};
    g.convertTo(imgf32, CV_32FC1);

    auto const  histSize = std::array{nbins};
    auto const  range    = std::array{0.0f, 1.0f + std::numeric_limits<float>::epsilon()};
    auto const *ranges   = range.data();

    cv::calcHist(&imgf32, 1, 0, mask, hist, 1, histSize.data(), &ranges);
    auto f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen(file_name, "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = 0; nbins > i; ++i) {
        fmt::print(
            f.get(), "{} {}\n"
          , (i + 0.5) / nbins
          , hist.at<float>(i)
          );
    }
}

void write_png(cv::Mat const &g, const char *file_name) {
    auto gamma_corrected = cv::Mat{};
    cv::pow(g, 1.0 / 2.2, gamma_corrected);
    auto vis = cv::Mat{};
    gamma_corrected.convertTo(vis, CV_8U, 255.0);
    cv::imwrite(file_name, vis);
}

auto chambolle(cv::Mat const &g, double lambda, double tau, int iters) -> cv::Mat {
    auto const rows = g.rows, cols = g.cols;
    auto p1   = cv::Mat{cv::Mat::zeros(rows, cols, CV_64FC1)};
    auto p2   = cv::Mat{cv::Mat::zeros(rows, cols, CV_64FC1)};
    auto divp = cv::Mat{cv::Mat::zeros(rows, cols, CV_64FC1)};

    for (auto n = int{}; iters > n; ++n) {
        // div p (backward differences with Chambolle's boundary rules)
        for (auto i = int{}; rows > i; ++i) {
            auto const *r1 = p1.ptr<double>(i);
            auto const *r1m = i > 0 ? p1.ptr<double>(i - 1) : nullptr;
            auto const *r2 = p2.ptr<double>(i);
            auto *d = divp.ptr<double>(i);
            for (auto j = int{}; cols > j; ++j) {
                auto v = 0.0;
                if (rows - 1 > i) { v += r1[j]; }
                if (0 < i)        { v -= r1m[j]; }
                if (cols - 1 > j) { v += r2[j]; }
                if (0 < j)        { v -= r2[j - 1]; }
                d[j] = v;
            }
        }
        // grad(div p - g / lambda), then semi-implicit update
        for (auto i = int{}; rows > i; ++i) {
            auto const *w  = divp.ptr<double>(i);
            auto const *wn = rows - 1 > i ? divp.ptr<double>(i + 1) : nullptr;
            auto const *gg  = g.ptr<double>(i);
            auto const *ggn = rows - 1 > i ? g.ptr<double>(i + 1) : nullptr;
            auto *q1 = p1.ptr<double>(i);
            auto *q2 = p2.ptr<double>(i);
            for (auto j = int{}; cols > j; ++j) {
                auto gx = rows - 1 > i
                    ? (wn[j] - w[j]) - (ggn[j] - gg[j]) / lambda : 0.0;
                auto gy = cols - 1 > j
                    ? (w[j + 1] - w[j]) - (gg[j + 1] - gg[j]) / lambda : 0.0;
                auto denom = 1.0 + tau * std::sqrt(gx * gx + gy * gy);
                q1[j] = (q1[j] + tau * gx) / denom;
                q2[j] = (q2[j] + tau * gy) / denom;
            }
        }
    }

    // final divergence for u = g - lambda * div p
    for (auto i = int{}; rows > i; ++i) {
        auto const* r1 = p1.ptr<double>(i);
        auto const* r1m = i > 0 ? p1.ptr<double>(i - 1) : nullptr;
        auto const* r2 = p2.ptr<double>(i);
        auto* d = divp.ptr<double>(i);
        for (auto j = int{}; cols > j; ++j) {
            auto v = 0.0;
            if (rows - 1 > i) { v += r1[j]; }
            if (0 < i)        { v -= r1m[j]; }
            if (cols - 1 > j) { v += r2[j]; }
            if (0 < j)        { v -= r2[j - 1]; }
            d[j] = v;
        }
    }
    return g - lambda * divp;
}

int main(int argc, char *argv[]) {
    auto tif = std::unique_ptr<TIFF, decltype(&TIFFClose)>{
        TIFFOpen("sample1_recon_Export.tiff", "r")
      , &TIFFClose
      };
    TIFFSetDirectory(tif.get(), 497);

    auto w = uint32_t{}, h = uint32_t{};
    auto bps = uint16_t{}, spp = uint16_t{};

    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);

    fmt::print("width  = {}\n", w);
    fmt::print("height = {}\n", h);
    fmt::print("bps = {}\n", bps);
    fmt::print("spp = {}\n", spp);

    auto stride = ((w + 3) / 4) * 4;
    fmt::print("stride = {}\n", stride);

    auto data = std::vector<uint16_t>(stride * h);
    for (auto row = uint32_t{}; h > row; ++row) {
        TIFFReadScanline(tif.get(), data.data() + row * stride, row);
    }

    auto img  = cv::Mat(h, w, CV_16UC1, data.data(), stride * 2);
    auto imgf = cv::Mat{};
    auto mask = cv::Mat{};

    // detect circular mask and convert to double
    {
        auto img8 = cv::Mat{};
        cv::normalize(img, img8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        auto bin = cv::Mat{};
        cv::threshold(img8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        auto contours = std::vector<std::vector<cv::Point>>{};
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        auto largest = rs::max_element(
            contours
          , [](auto &&a, auto &&b) {
                return cv::contourArea(a) < cv::contourArea(b);
          });

        auto center = cv::Point2f{};
        auto radius = float{};
        cv::minEnclosingCircle(*largest, center, radius);
        fmt::print("mask radius = {}\n", radius);
        fmt::print("mask location = {}, {}\n", center.x, center.y);

        mask = cv::Mat{cv::Mat::zeros(img.size(), CV_8UC1)};
        cv::circle(mask, center, static_cast<int>(radius * 0.98f), cv::Scalar{255}, cv::FILLED);

        auto raw_hist = cv::Mat{};
        auto constexpr raw_bins = int{65536};
        auto const  rhist_size = std::array{raw_bins};
        auto const  rrange     = std::array{0.0f, 65536.0f};
        auto const *rranges    = rrange.data();

        cv::calcHist(&img, 1, 0, mask, raw_hist, 1, rhist_size.data(), &rranges);

        auto const total  = cv::sum(raw_hist)[0];
        auto const pct_lo = 0.001 * total;
        auto const pct_hi = 0.999 * total;

        auto lo = int{}, hi = int{raw_bins - 1};
        auto cum = 0.0;
        for (auto i = int{}; raw_bins > i; ++i) {
            cum += raw_hist.at<float>(i);
            if (pct_lo >= cum) { lo = i; }
            if (pct_hi <= cum) { hi = i; break; }
        }

        auto true_lo = double{}, true_hi = double{};
        cv::minMaxLoc(img, &true_lo, &true_hi, nullptr, nullptr, mask);
        fmt::print("image lo = {}\n", true_lo);
        fmt::print("image hi = {}\n", true_hi);
        fmt::print("used  lo = {}\n", lo);
        fmt::print("used  hi = {}\n", hi);

        imgf = cv::Mat{};
        img.convertTo(imgf, CV_64FC1, 1.0 / (hi - lo), -static_cast<double>(lo) / (hi - lo));
        cv::min(imgf, 1.0, imgf);
        cv::max(imgf, 0.0, imgf);

        write_png(imgf, "normalized.png");
        write_hist(imgf, mask, "hist.txt");
    }

    auto denoised = chambolle(imgf, 0.05, 0.12, 100);
    write_png(denoised, "normalized (𝜆 = 0.05).png");
    write_hist(denoised, mask, "hist (𝜆 = 0.05).txt");

    denoised = chambolle(imgf, 0.1, 0.12, 100);
    write_png(denoised, "normalized (𝜆 = 0.1).png");
    write_hist(denoised, mask, "hist (𝜆 = 0.1).txt");

    denoised = chambolle(imgf, 0.2, 0.12, 100);
    write_png(denoised, "normalized (𝜆 = 0.2).png");
    write_hist(denoised, mask, "hist (𝜆 = 0.2).txt");

    denoised = chambolle(imgf, 0.4, 0.12, 100);
    write_png(denoised, "normalized (𝜆 = 0.4).png");
    write_hist(denoised, mask, "hist (𝜆 = 0.4).txt");
}
