#include <filesystem>
#include <string>

#include <boost/program_options.hpp>

#include <fmt/printf.h>
#include <fmt/color.h>

#include <opencv2/opencv.hpp>

#include <tiffio.h>

#include <field.h>
#include <timeop.h>

#include <io/fileoutit.h>
#include <io/binoarchive.h>

#include "find_circular_mask.h"
#include "max_inscribed_circle.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name  = std::string{};
    auto out_file_name = std::string{};

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
        ("output,o", po::value<std::string>(&out_file_name), "output file");
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
        auto in_file = std::unique_ptr<FILE, decltype(&fclose)>{
            fopen(in_file_name.c_str(), "r")
          , &fclose
          };
        if (!in_file) {
            throw std::system_error(errno, std::system_category());
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


    auto tif = std::unique_ptr<TIFF, decltype(&TIFFClose)>{
        TIFFOpen(in_file_name.c_str(), "r")
      , &TIFFClose
      };

    if (!tif) {
        fmt::print("Can't open \"{}\"\n", in_file_name);
        return 1;
    }

    const auto [width, height] = [&]() {
        TIFFSetDirectory(tif.get(), 0);
        auto w = uint32_t{};
        auto h = uint32_t{};
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);

        return std::tuple{w, h};
    }();

    const auto num_images = TIFFNumberOfDirectories(tif.get());

    for (auto n_dir = tdir_t{}; num_images > n_dir; ++n_dir) {
        TIFFSetDirectory(tif.get(), n_dir);

        auto w = uint32_t{};
        auto h = uint32_t{};
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);
        if (width != w || height != h) {
            fmt::print(stderr, "Unexpected image size: {}x{}, {}x{} was expected\n", w, h, width, height);
            return 1;
        }
    }

    auto circles = std::vector<std::tuple<cv::Point2d, double>>{};
    auto slice_storage = std::vector<uint16_t>(width * height);
    auto slice_view = std::mdspan{slice_storage.data(), height, width};
    for (auto n_dir = tdir_t{}; num_images > n_dir; ++n_dir) {
        TIFFSetDirectory(tif.get(), n_dir);

        for (auto row = uint32_t{}; height > row; ++row) {
            TIFFReadScanline(tif.get(), &slice_view[row, 0], row);
        }
        auto g = cv::Mat(height, width, CV_16UC1, &slice_view[0, 0]);
        circles.push_back(find_circular_mask(g));
    }
    auto [center, radius] = max_inscribed_circle(circles);
    fmt::print("mask radius = {}\n", radius);
    fmt::print("mask location = {}, {}\n", center.x, center.y);

    auto original_img = Field<uint16_t>(num_images, width, height, center.x, center.y, radius);

    auto constexpr B = uint32_t{128};
    auto batch_storage = std::vector<uint16_t>(size_t{B} * height * width);
    auto batch = std::mdspan{batch_storage.data(), B, height, width};

    auto start_time  = timespec{};
    auto finish_time = timespec{};
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (auto n0 = tdir_t{}; num_images > n0; n0 += B) {
        auto nb = std::min<uint32_t>(B, num_images - n0);
        for (auto k = uint32_t{}; nb > k; ++k) {
            TIFFSetDirectory(tif.get(), n0 + k);
            for (auto row = uint32_t{}; height > row; ++row) {
                TIFFReadScanline(tif.get(), &batch[k, row, 0], row);
            }
        }
        #pragma omp parallel for schedule(static, 1)
        for (auto col = original_img.col_begin(); original_img.col_end() > col; ++col) {
            for (auto row = original_img.row_begin(col); original_img.row_end(col) > row; ++row) {
                for (auto k = uint32_t{}; nb > k; ++k) {
                    original_img[n0 + k, row, col] = batch[k, row, col];
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish_time);
    auto elapsed_time = finish_time - start_time;
    fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
               "Copy time: {:.6g} s\n", elapsed_time);

    auto buffer = std::vector<uint8_t>(2 * num_images * width * height);
    auto outit  = buffer.data();
    auto oa     = io::BinOArchive{outit};

    oa << original_img;

    fwrite(buffer.data(), oa.size(), 1, out_file.get());
}
