#include <cstdint>
#include <ctime>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>

#include <fmt/printf.h>

#include <opencv2/opencv.hpp>

#include <io/biniarchive.h>
#include <io/binoarchive.h>

#include <cuda-context.h>
#include <calc_hist.h>
#include <field.h>
#include <timeop.h>

#include "chambolle.h"

namespace bc = boost::math::double_constants;
namespace fs = std::filesystem;
namespace po = boost::program_options;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<uint16_t>::save(io::BinOArchive<uint8_t*>&, unsigned) const;
extern template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<float>::save(io::BinOArchive<uint8_t*>&, unsigned) const;

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
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name       = std::string{};
    auto denoised_file_name = std::string{};

    auto 𝜆 = 0.06f;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
        ("output,o", po::value<std::string>(&denoised_file_name), "denoised output file")
        ("lambda", po::value(&𝜆), "Chambolle's algorithm parameter")
      ;
    positional.add("input", 1);

    try {
        po::store(po::command_line_parser(argc, argv).
                  positional(positional).
                  options(cmd_line_options).run(), vm);
        po::notify(vm);
    } catch (po::error &x) {
        fmt::print(stderr, "Command line error: {}\n", x.what());
        return 1;
    }

    auto out_file      = std::unique_ptr<FILE, decltype(&fclose)>{nullptr, &fclose};
    auto denoised_file = std::unique_ptr<FILE, decltype(&fclose)>{nullptr, &fclose};
    try {
        if (!fs::exists(in_file_name)) {
            fmt::print(stderr, "Input file {} doesn't exist\n", in_file_name);
            return 1;
        }

        if (0 != vm.count("output")) {
            denoised_file.reset(fopen(denoised_file_name.c_str(), "w"));
            if (!denoised_file) {
                throw std::system_error(errno, std::system_category());
            }
        }
    } catch (std::exception &x) {
        fmt::print(stderr, "{}\n", x.what());
        return 1;
    }

    auto in_file = boost::iostreams::mapped_file_source{in_file_name};
    auto it      = in_file.begin();
    auto end     = it + in_file.size();
    auto ia      = io::BinIArchive{it, end};

    auto t = Timer{};

    auto img = Field<float>{};
    t.start();
        ia >> img;
    t.stop("Read time");

    auto lo = img.lo();
    auto hi = img.hi();

    auto cuda_context = make_context();

    t.start();
        auto d_img = upload(*cuda_context, img.view());
    t.stop("Upload to GPU time");

    static auto constexpr col = 481;

    write_png(img.view(), col, 0.0f);

    const auto nbins = 1024;
    auto 𝛿 = 1.0 / (hi - lo);
    𝛿 = sqr(𝛿) < 0.5 / nbins ? 𝛿 : 0.0;
    t.start();
        auto [hist, min, max] = calc_hist(*cuda_context, *d_img, nbins, 𝛿);
    t.stop("Histogram time");
    const auto bin_width = (double(hi) - double(lo)) / nbins;
    auto f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen("original-hist.txt", "wt")
      , &fclose
      };
    fmt::print(f.get(), "brightness ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) * bin_width + lo, hist[i] * (max - min) / (hi - lo));
    }

    t.start();
        chambolle(*cuda_context, *d_img, 𝜆, 1.0f / 6.0f * 0.99f, 150);
    t.stop("Chambolle time");
    download(*cuda_context, *d_img, img.view());
    if (denoised_file) {
        auto m      = img.view().mapping();
        auto buffer = std::vector<uint8_t>(size_t{4} * m.required_span_size() * 2);
        auto outit  = buffer.data();
        auto oa     = io::BinOArchive{outit};

        t.start();
            oa << img;
        t.stop("Writing denoised file time");

        fwrite(buffer.data(), oa.size(), 1, denoised_file.get());
    }

    hist = calc_hist(*cuda_context, *d_img, min, max, nbins, 𝛿);
    f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen(fmt::format("hist (𝜆 = {:.3f}).txt", 𝜆).c_str(), "wt")
      , &fclose
      };
    fmt::print(f.get(), "brightness ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) * bin_width + lo, hist[i] * (max - min) / (hi - lo));
    }

    write_png(img.view(), col, 𝜆);
}
catch (const std::exception& e) {
    fmt::print(stderr, "{}\n", e.what());
}
