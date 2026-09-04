#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>

#include <boost/hana.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/parser/parser.hpp>
#include <boost/program_options.hpp>

#include <fmt/color.h>
#include <fmt/printf.h>

#include <opencv2/opencv.hpp>

#include <io/biniarchive.h>

#include <cuda-context.h>
#include <field.h>
#include <timeop.h>

#include "potts-min-partition.h"

namespace fs   = std::filesystem;
namespace hana = boost::hana;
namespace po   = boost::program_options;

using namespace hana::literals;

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

struct LabelsArray {
    std::vector<float> data;
};

void validate(boost::any &v,
              const std::vector<std::string> &values,
              LabelsArray *targetType, int)
{
    using namespace boost::program_options;
    namespace bp = boost::parser;

    validators::check_first_occurrence(v);
    const auto& s = validators::get_single_string(values);
    auto data = std::vector<float>{};
    auto it  = s.begin();
    auto end = s.end();
    auto result = bp::prefix_parse(
        it, end
      , '{' >> (bp::float_ % ',') >> '}'
      , bp::ws
      , data
      );
    if (!result || data.size() < 2 || data.size() > max_potts_labels) {
        throw validation_error(validation_error::invalid_option_value);
    }
    v = boost::any(LabelsArray{.data = data});
}


int main(int argc, char *argv[]) try {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name     = std::string{};
    auto rawout_file_name = std::string{};

    auto labels = LabelsArray{};
    auto 𝜆      = 30.0f;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "input file")
        ("rawout", po::value<std::string>(&rawout_file_name), "denoised output file")
        ("labels", po::value(&labels)->required(), "labels")
        ("lambda", po::value(&𝜆)->required(), "Segmentation algorithm parameter")
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

    auto rawout_file = std::unique_ptr<FILE, decltype(&fclose)>{nullptr, &fclose};
    try {
        if (!fs::exists(in_file_name)) {
            fmt::print(stderr, "Input file {} doesn't exist\n", in_file_name);
            return 1;
        }
        if (0 != vm.count("rawout")) {
            rawout_file.reset(fopen(rawout_file_name.c_str(), "w"));
            if (!rawout_file) {
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

    write_png(img.view(), 480, 0.0f);

    auto cuda_context = make_context();

    auto 𝜏  = 0.99f * std::sqrt(1.0f / 12.0f);
    hana::for_each(hana::make_range(2_c, hana::llong_c<max_potts_labels + 1>), [&](auto i) {
        if (labels.data.size() == hana::value(i)) {
            auto ci = hana::unpack(hana::make_range(0_c, i), [&](auto ...j) {
                return std::array<float, i>{{labels.data[j]...}};
            });
            auto lo = img.lo();
            auto hi = img.hi();
            fmt::print("labels: ");
            for (auto j = 0; ci.size() > j; ++j) {
                if (0 != j) {
                    fmt::print(", ");
                }
                auto c = (ci[j] - lo) / (hi - lo);
                fmt::print("{}", std::round(255 * c));
            }
            fmt::print("\n");
            auto [available, total] = device_memory(*cuda_context);
            auto required_mem_size = size_t{4} * (5 * i - 4) * img.view().mapping().required_span_size();
            if (available < required_mem_size) {
                fmt::print(
                    fmt::fg(fmt::color::light_coral) | fmt::emphasis::bold
                  , "Memory requirement for the segmentation algorithm is {:.3f} GiB ({} labels)\n"
                    "Available memory: {:.3f} GiB out of {:.3f} GiB\n"
                  , double(required_mem_size) / (uint64_t(1) << 30)
                  , hana::value(i)
                  , double(available) / (uint64_t(1) << 30)
                  , double(total) / (uint64_t(1) << 30)
                  );
            }

            t.start();
                auto d_img = upload(*cuda_context, img.view());
            t.stop("Upload to GPU time");

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
            d_img.reset();
        }
    });

    write_png(img.view(), 480, 𝜆);

    if (rawout_file) {
        auto v = img.view();
        auto m = v.mapping();
        auto row_begin = std::numeric_limits<std::decay_t<decltype(m.row_begin(0))>>::max();
        auto row_end   = std::numeric_limits<std::decay_t<decltype(m.row_end(0))>>::min();
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            row_begin = std::min(row_begin, m.row_begin(col));
            row_end   = std::max(row_end, m.row_end(col));
        }
        auto row_size = row_end - row_begin;
        auto col_size = m.col_end() - m.col_begin();
        fmt::print(
            "Dimensions of the raw file: slices = {}, rows = {}, columns = {}\n"
          , m.n_images(), row_size, col_size
          );
        fmt::print("Slices (TIFF images, the plane is perpendicular to the cylinder axis) is"
                   " the most dense dimension. The next is rows. Finally, columns is the most sparse.\n");
        auto buffer = std::vector<uint8_t>();
        for (auto c = 0; col_size > c; ++c) {
            for (auto r = 0; row_size > r; ++r) {
                for (auto i = 0; m.n_images() > i; ++i) {
                    auto col = c + m.col_begin();
                    auto row = r + row_begin;
                    if ((m.col_begin() <= col && col < m.col_end()) &&
                        (m.row_begin(col) <= row && row < m.row_end(col)))
                    {
                        buffer.push_back(
                            uint8_t(
                                std::min(
                                    255
                                  , std::max(
                                        0
                                      , int(std::round(255 * v[i, row, col]))
                                      )
                                  )
                              )
                          );
                    } else {
                        buffer.push_back(255);
                    }
                }
            }
        }
        fmt::print(rawout_file.get(),
            "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"ImageData\" byte_order=\"LittleEndian\">\n"
            "  <ImageData WholeExtent=\"0 {} 0 {} 0 {}\">\n"
            "    <Piece Extent=\"0 {} 0 {} 0 {}\">\n"
            "      <PointData Scalars=\"labels\">\n"
            "        <DataArray type=\"UInt8\" Name=\"labels\" format=\"appended\" offset=\"0\"/>\n"
            "      </PointData>\n"
            "    </Piece>\n"
            "  </ImageData>\n"
            "  <AppendedData encoding=\"raw\">\n_",
            m.n_images() - 1, row_size - 1, col_size - 1, m.n_images() - 1, row_size - 1, col_size - 1);
        auto nbytes = uint32_t(buffer.size() * sizeof(buffer[0]));
        fwrite(&nbytes, 4, 1, rawout_file.get());
        fwrite(buffer.data(), sizeof(buffer[0]), buffer.size(), rawout_file.get());
        fmt::print(rawout_file.get(), "\n  </AppendedData>\n</VTKFile>\n");
        rawout_file.reset();
    }
    cuda_context.reset();
}
catch (std::exception &x) {
    fmt::print(stderr, "{}\n", x.what());
    return 1;
}
