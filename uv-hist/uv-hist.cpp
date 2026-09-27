#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>

#include <fmt/printf.h>

#include <opencv2/opencv.hpp>

#include <io/biniarchive.h>

#include <cuda-context.h>
#include <field.h>
#include <grad_magn.h>
#include <timeop.h>
#include "uv-hist.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);

int main(int argc, char *argv[]) {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name  = std::string{};
    auto out_file_name = std::string{};

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
        ("output,o", po::value<std::string>(&out_file_name), "output file")
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

    auto out_file = std::unique_ptr<FILE, decltype(&fclose)>{nullptr, &fclose};
    try {
        if (!fs::exists(in_file_name)) {
            fmt::print(stderr, "Input file {} doesn't exist\n", in_file_name);
            return 1;
        }

        if (0 == vm.count("output")) {
            out_file = decltype(out_file)(stdout, [](FILE *)->int { return 0; });
        } else {
            out_file.reset(fopen(out_file_name.c_str(), "w"));
            if (!out_file) {
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

    auto cuda_context = make_context();
    t.start();
        auto d_img = upload(*cuda_context, img.view());
    t.stop("Upload to GPU time");

    t.start();
        auto d_grad_magn_img = grad_magn(*cuda_context, *d_img);
    t.stop("Gradient magnitude time");

    t.start();
        auto hist_res = uv_hist(*cuda_context, *d_img, *d_grad_magn_img, 512);
    t.stop("Histogram time");

    fmt::print("u ∈ [{}, {}], v ∈ [{}, {}]\n"
      , hist_res.u_start, hist_res.u_end
      , hist_res.v_start, hist_res.v_end
      );

    auto hist = std::mdspan{&hist_res.hist[0], hist_res.mapping};
    const auto rows = int(hist.extent(0));
    const auto cols = int(hist.extent(1));

    auto counts = cv::Mat(rows, cols, CV_64FC1);
    for (auto row = 0; row < rows; ++row) {
        auto dst = counts.ptr<double>(rows - 1 - row);
        for (auto col = 0; col < cols; ++col) {
            dst[col] = double(std::log(hist[row, col] + 1));
        }
    }

    auto max_count = 0.0;
    cv::minMaxLoc(counts, nullptr, &max_count);

    auto indexed = cv::Mat{};
    counts.convertTo(
        indexed, CV_8UC1,
        max_count > 0.0 ? 255.0 / max_count : 0.0
    );

    auto coloured = cv::Mat{};
    cv::applyColorMap(indexed, coloured, cv::COLORMAP_VIRIDIS);

    cv::imwrite(out_file_name.c_str(), coloured);
}
