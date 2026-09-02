#include <filesystem>
#include <string>

#include <boost/hana/type.hpp>
#include <boost/program_options.hpp>
#include <boost/scope_exit.hpp>

#include <fmt/printf.h>
#include <fmt/color.h>

#include <mdspan/mdspan.hpp>

#include <opencv2/opencv.hpp>

#include <tiffio.h>

#include <field.h>
#include <timeop.h>

#include <io/binoarchive.h>

#include <find_circular_mask.h>
#include <max_inscribed_circle.h>

namespace fs   = std::filesystem;
namespace hana = boost::hana;
namespace po   = boost::program_options;

int main(int argc, char *argv[]) try {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name  = std::string{};
    auto out_file_name = std::string{};

    auto reduced_height = 0;
    auto reduced_radius = 0.0;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
        ("output,o", po::value<std::string>(&out_file_name), "output file")
        ("height", po::value(&reduced_height), "reduced height")
        ("radius", po::value(&reduced_radius), "reduced radius")
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

    TIFFSetErrorHandler([](const char *module, const char *format, va_list ap) {
        va_list ap2;
        va_copy(ap2, ap);
        BOOST_SCOPE_EXIT_ALL(&) { va_end(ap2); };
        auto buf_size = vsnprintf(nullptr, 0, format, ap);
        if (0 > buf_size) {
            fmt::print(
                stderr
              , fmt::fg(fmt::color::crimson) | fmt::emphasis::bold
              , "Error decoding \"{}\"\n", format);
            return;
        }
        auto buf = std::vector<char>(buf_size + 1);
        vsnprintf(buf.data(), std::size(buf), format, ap2);
        fmt::print(
            stderr
          , fmt::fg(fmt::color::crimson) | fmt::emphasis::bold
          , "{}: {}\n", module, buf.data());
    });

    auto tif = std::unique_ptr<TIFF, decltype(&TIFFClose)>{
        TIFFOpen(in_file_name.c_str(), "r")
      , &TIFFClose
      };

    if (!tif) {
        fmt::print("Can't open \"{}\"\n", in_file_name);
        return 1;
    }

    const auto [width, height, sample_format, bits_per_sample, samples_per_pixel] = [&]() {
        TIFFSetDirectory(tif.get(), 0);
        auto w = uint32_t{};
        auto h = uint32_t{};
        TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);

        auto sample_format = uint16_t{};
        TIFFGetField(tif.get(), TIFFTAG_SAMPLEFORMAT, &sample_format);

        auto bits_per_sample = uint16_t{};
        TIFFGetField(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits_per_sample);

        uint16_t samples_per_pixel;
        TIFFGetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);

        return std::tuple{w, h, sample_format, bits_per_sample, samples_per_pixel};
    }();

    if ((32 != bits_per_sample || SAMPLEFORMAT_IEEEFP != sample_format || 1 != samples_per_pixel) &&
        (16 != bits_per_sample || SAMPLEFORMAT_UINT   != sample_format || 1 != samples_per_pixel))
    {
        fmt::print(stderr,
            "The input is expected to be either monochrome 32-bit floating point or "
            "16-bit unsigned integer\n");
        return 1;
    }

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

    auto img_start  = tdir_t{};
    auto img_finish = num_images;
    if (0 != vm.count("height") && 0 < reduced_height && reduced_height < num_images) {
        img_start  = (num_images - reduced_height) / 2;
        img_finish = img_start + reduced_height;
    }
    const auto outimgs = img_finish - img_start;

    auto find_mask = [&](auto cpp_type, int type) -> std::tuple<cv::Point2d, double> {
        using sample_type = decltype(+cpp_type)::type;

        auto circles       = std::vector<std::tuple<cv::Point2d, double>>{};
        auto slice_storage = std::vector<sample_type>(width * height);
        auto slice_view    = std::mdspan{slice_storage.data(), height, width};
        for (auto n_dir = img_start; img_finish > n_dir; ++n_dir) {
            TIFFSetDirectory(tif.get(), n_dir);

            for (auto row = uint32_t{}; height > row; ++row) {
                if (-1 == TIFFReadScanline(tif.get(), &slice_view[row, 0], row)) {
                    throw std::runtime_error("Unexpected end of the TIFF file");
                }
            }
            auto g = cv::Mat(height, width, type, &slice_view[0, 0]);
            circles.push_back(find_circular_mask(g));
        }

        return max_inscribed_circle(circles);
    };
    auto radius = 0.0;
    auto center = cv::Point2d{};

    if (SAMPLEFORMAT_UINT == sample_format) {
        std::tie(center, radius) = find_mask(hana::type_c<uint16_t>, CV_16UC1);
    } else if (SAMPLEFORMAT_IEEEFP == sample_format) {
        std::tie(center, radius) = find_mask(hana::type_c<float>, CV_32FC1);
    }
    fmt::print("mask radius = {:.2f}\n", radius);
    fmt::print("mask location = {:.2f}, {:.2f}\n", center.x, center.y);


    if (0 != vm.count("radius") && 1.0 < reduced_radius && reduced_radius < radius) {
        radius = reduced_radius;
    }

    auto transpose_and_save = [&](auto cpp_type) {
        using sample_type = decltype(+cpp_type)::type;

        auto original_img = Field<sample_type>(outimgs, width, height, center.x, center.y, radius);

        auto constexpr B = uint32_t{128};
        auto batch_storage = std::vector<sample_type>(size_t{B} * height * width);
        auto batch = std::mdspan{batch_storage.data(), B, height, width};

        auto t = Timer{};
        t.start();
        for (auto n0 = img_start; img_finish > n0; n0 += B) {
            auto nb = std::min<uint32_t>(B, img_finish - n0);
            for (auto k = uint32_t{}; nb > k; ++k) {
                TIFFSetDirectory(tif.get(), n0 + k);
                for (auto row = uint32_t{}; height > row; ++row) {
                    if (-1 == TIFFReadScanline(tif.get(), &batch[k, row, 0], row)) {
                        throw std::runtime_error("Unexpected end of the TIFF file");
                    }
                }
            }
            auto v = original_img.view();
            auto m = v.mapping();
            #pragma omp parallel for schedule(static, 1)
            for (auto col = m.col_begin(); m.col_end() > col; ++col) {
                for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                    for (auto k = uint32_t{}; nb > k; ++k) {
                        v[n0 + k - img_start, row, col] = batch[k, row, col];
                    }
                }
            }
        }
        t.stop("Transpose time");

        static auto constexpr K = 3;

        auto m             = original_img.view().mapping();
        auto required_size = double(size_t{4} * (5 * K - 4) * m.required_span_size()) / (uint64_t(1) << 30);
        fmt::print(fmt::fg(fmt::color::light_coral) | fmt::emphasis::bold,
                   "Memory requirement for the CCP algorithm is {:.3f} GiB ({} labels)\n"
                   , required_size, K);

        auto buffer = std::vector<uint8_t>(sizeof(sample_type) * outimgs * width * height);
        auto outit  = buffer.data();
        auto oa     = io::BinOArchive{outit};

        oa << original_img;

        fwrite(buffer.data(), oa.size(), 1, out_file.get());
    };

    if (SAMPLEFORMAT_UINT == sample_format) {
        transpose_and_save(hana::type_c<uint16_t>);
    } else if (SAMPLEFORMAT_IEEEFP == sample_format) {
        transpose_and_save(hana::type_c<float>);
    }
}
catch (std::exception &x) {
    fmt::print(stderr, "{}\n", x.what());
    return 1;
}
