#include <array>
#include <exception>
#include <filesystem>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>

#include <fmt/printf.h>

#include <opencv2/opencv.hpp>

#include <io/biniarchive.h>

#include <cuda-context.h>
#include <field.h>
#include <timeop.h>

#include "potts-min-partition.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);

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
    cv::imwrite(fmt::format("slice-{} (𝜆 = {:07.3f}).png", col, 𝜆), vis);
}

int main(int argc, char *argv[]) try {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name       = std::string{};

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
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

    write_png(img.view(), 480, 0.0f);

    auto cuda_context = make_context();

    t.start();
        auto d_img = upload(*cuda_context, img.view());
    t.stop("Upload to GPU time");

    const auto 𝜆 = 30.0f;

    auto 𝜏  = 0.99f * std::sqrt(1.0f / 12.0f);
    auto ci = std::array<float, 3>{0.155431f, 0.22989f, 0.451221f};
    t.start();
        potts_min_partition(
            *cuda_context
          , *d_img
          , ci
          , 𝜆, 𝜏, 𝜏
          , 100);
    t.stop("Find intensity range time");

    t.start();
        download(*cuda_context, *d_img, img.view());
    t.stop("Download from GPU time");

    write_png(img.view(), 480, 𝜆);
}
catch (std::exception &x) {
    fmt::print(stderr, "{}\n", x.what());
    return 1;
}
