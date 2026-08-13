#include <array>
#include <concepts>
#include <cstdint>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <boost/align/aligned_allocator.hpp>
#include <boost/math/constants/constants.hpp>

#include <clarabel.hpp>

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

#include <Eigen/Dense>

#include <fmt/printf.h>
#include <fmt/color.h>

#include <opencv2/opencv.hpp>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/max_element.hpp>
#include <range/v3/algorithm/min_element.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/transform.hpp>

#include <tiffio.h>

namespace rs = ranges;
namespace rv = rs::views;
namespace bc = boost::math::double_constants;

using namespace Eigen;

inline
auto sqr(auto x) { return x * x; }

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

auto find_circular_mask(const cv::Mat &g) {
    auto img8 = cv::Mat{};
    cv::normalize(g, img8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

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

    return std::tuple{cv::Point2d{center.x, center.y}, double(radius)};
}

auto max_inscribed_circle(const std::vector<std::tuple<cv::Point2d, double>> &circles) {
    using namespace Eigen;

    auto P     = SparseMatrix<double>(3, 3);
    auto q     = Vector3d{0, 0, -1};
    auto A     = SparseMatrix<double>(3 * circles.size(), 3);
    auto b     = VectorXd(3 * circles.size());
    auto trip  = std::vector<Triplet<double>>();
    auto cones = std::vector<clarabel::SupportedConeT<>>();

    trip.reserve(3 * circles.size());
    cones.reserve(3 * circles.size());

    for (auto i = size_t{}; circles.size() > i; ++i) {
        auto &&[p, r] = circles[i];

        auto row = 3 * i;
        trip.emplace_back(row,     2, 1.0); // (A z)_0 = r
        trip.emplace_back(row + 1, 0, 1.0); // (A z)_1 = x_c
        trip.emplace_back(row + 2, 1, 1.0); // (A z)_2 = y_c

        b[row]     =   r;
        b[row + 1] = p.x;
        b[row + 2] = p.y;

        cones.emplace_back(clarabel::SecondOrderConeT<>(3));
    }
    A.setFromTriplets(trip.begin(), trip.end());
    A.makeCompressed();

    auto settings = clarabel::DefaultSettings<>::default_settings();
    settings.verbose = false;

    auto solver = clarabel::DefaultSolver<>(P, q, A, b, cones, settings);

    solver.solve();
    auto solution = solver.solution();

    if (clarabel::SolverStatus::Solved != solution.status) {
        throw std::runtime_error("Clarabel did not converge.");
    }

    return std::tuple{cv::Point2d(solution.x[0], solution.x[1]), solution.x[2]};
}

template <std::integral ElementType, typename Extents, typename LayoutPolicy>
auto find_intensity_range(
    std::mdspan<ElementType, Extents, LayoutPolicy> a
  , double xc
  , double yc
  , double r
  , double tail_thr)
{
    using mdspan     = std::mdspan<ElementType, Extents, LayoutPolicy>;
    using value_type = mdspan::value_type;

    auto constexpr minv      = std::numeric_limits<value_type>::min();
    auto constexpr maxv      = std::numeric_limits<value_type>::max();
    auto constexpr hist_size = maxv - minv + 1;

    const auto num_images = a.extent(0);
    const auto height     = a.extent(1);
    const auto width      = a.extent(2);
    const auto r2         = sqr(r);

    using alloc = boost::alignment::aligned_allocator<uint64_t, std::hardware_destructive_interference_size>;
    auto hist = std::vector<uint64_t, alloc>(hist_size, 0);
    auto hist_p = &hist[0];
    #pragma omp parallel for reduction(+:hist_p[:hist_size])
    for (auto n = tdir_t{}; num_images > n; ++n) {
        auto start_row = uint32_t(std::max(0.0, std::ceil(yc - r)));
        auto end_row   = uint32_t(std::min(double(height - 1), std::floor(yc + r)));
        for (auto row = start_row; end_row >= row; ++row) {
            auto sq = std::sqrt(r2 - sqr(row - yc));
            auto start_col = uint32_t(std::max(0.0, std::ceil(xc - sq)));
            auto end_col   = uint32_t(std::min(double(width - 1), std::floor(xc + sq)));
            for (auto col = start_col; end_col >= col; ++col) {
                auto v = a[n, row, col];
                ++hist_p[v - minv];
            }
        }
    }

    auto total = rs::accumulate(hist, uint64_t{});

    auto const pct_lo = tail_thr * total;
    auto const pct_hi = (1.0 - tail_thr) * total;
    auto lo = size_t{}, hi = hist.size();
    auto cum = uint64_t{};
    for (auto i = size_t{}; hist.size() > i; ++i) {
        cum += hist[i];
        if (pct_lo >= cum) { lo = i; }
        if (pct_hi <= cum) { hi = i; break; }
    }

    return std::tuple{value_type(lo + minv), value_type(hi + minv)};
}

template <std::integral ElementType, typename Extents, typename LayoutPolicy>
auto convert_to_fp(
    std::mdspan<ElementType, Extents, LayoutPolicy> a
  , double xc
  , double yc
  , double r
  , std::remove_cv_t<ElementType> lo
  , std::remove_cv_t<ElementType> hi)
{
    const auto num_images = a.extent(0);
    const auto height     = a.extent(1);
    const auto width      = a.extent(2);
    const auto r2         = sqr(r);

    static auto constexpr fp_align = uint32_t(std::hardware_destructive_interference_size / sizeof(float));
    const auto fp_stride     = uint32_t(((width + (fp_align - 1)) / fp_align) * fp_align);
    const auto fp_slice_size = uint32_t(fp_stride * height);

    using alloc = boost::alignment::aligned_allocator<float, 64>;
    auto img_storage = std::vector<float, alloc>(num_images * fp_slice_size, 0.0f);
    auto img = std::mdspan{
        img_storage.data()
      , std::layout_stride::mapping{
            std::extents{num_images, height, width}
          , std::array<uint32_t, 3>{fp_slice_size, fp_stride, 1}
          }
      };
    #pragma omp parallel for
    for (auto n = tdir_t{}; num_images > n; ++n) {
        auto start_row = uint32_t(std::max(0.0, std::ceil(yc - r)));
        auto end_row   = uint32_t(std::min(double(height - 1), std::floor(yc + r)));
        for (auto row = start_row; end_row >= row; ++row) {
            auto sq = std::sqrt(r2 - sqr(row - yc));
            auto start_col = uint32_t(std::max(0.0, std::ceil(xc - sq)));
            auto end_col   = uint32_t(std::min(double(width - 1), std::floor(xc + sq)));
            for (auto col = start_col; end_col >= col; ++col) {
                auto v = a[n, row, col];
                if (v <= lo) {
                    img[n, row, col] = 0.0f;
                    continue;
                }
                if (v >= hi) {
                    img[n, row, col] = 1.0f;
                    continue;
                }
                img[n, row, col] = float(v - lo) / (hi - lo);
            }
        }
    }

    return std::tuple{std::move(img_storage), img};
}

template <std::floating_point ElementType, typename Extents, typename LayoutPolicy>
auto calc_hist(
    std::mdspan<ElementType, Extents, LayoutPolicy> a
  , double xc
  , double yc
  , double r
  , double 𝛿
  , uint16_t nbins)
{
    using mdspan     = std::mdspan<ElementType, Extents, LayoutPolicy>;
    using value_type = mdspan::value_type;

    const auto num_images = a.extent(0);
    const auto height     = a.extent(1);
    const auto width      = a.extent(2);
    const auto r2         = sqr(r);

    auto rd   = std::random_device{};

    using alloc = boost::alignment::aligned_allocator<uint64_t, std::hardware_destructive_interference_size>;
    auto hist = std::vector<uint64_t, alloc>(nbins, 0);
    auto hist_p = &hist[0];
    #pragma omp parallel
    {
        auto gen  = std::mt19937(rd());
        auto dist = std::uniform_real_distribution{-𝛿 / 2, 𝛿 / 2};
        #pragma omp for reduction(+:hist_p[:nbins])
        for (auto n = tdir_t{}; num_images > n; ++n) {
            auto start_row = uint32_t(std::max(0.0, std::ceil(yc - r)));
            auto end_row   = uint32_t(std::min(double(height - 1), std::floor(yc + r)));
            for (auto row = start_row; end_row >= row; ++row) {
                auto sq = std::sqrt(r2 - sqr(row - yc));
                auto start_col = uint32_t(std::max(0.0, std::ceil(xc - sq)));
                auto end_col   = uint32_t(std::min(double(width - 1), std::floor(xc + sq)));
                for (auto col = start_col; end_col >= col; ++col) {
                    if (sqr(col - xc) + sqr(row - yc) < r2) {
                        auto v = a[n, row, col];
                        v += dist(gen);
                        auto bin = int32_t(std::round(nbins * v - value_type(0.5)));
                        if (0 > bin) bin = 0;
                        if (nbins <= bin) bin = nbins - 1;
                        ++hist_p[bin];
                    }
                }
            }
        }
    }

    auto total = rs::accumulate(hist, uint64_t{}) / double(nbins);
    auto fp_hist = std::vector<double>(nbins, 0);
    rs::copy(rv::transform(hist, [&](auto v) { return v / total; }), fp_hist.begin());

    return fp_hist;
}

template <std::floating_point ElementType, typename Extents, typename LayoutPolicy>
auto chambolle(
    std::mdspan<ElementType, Extents, LayoutPolicy> g
  , double xc
  , double yc
  , double r
  , typename std::mdspan<ElementType, Extents, LayoutPolicy>::value_type 𝜆
  , typename std::mdspan<ElementType, Extents, LayoutPolicy>::value_type 𝜏
  , int iters)
{
    using mdspan     = std::mdspan<ElementType, Extents, LayoutPolicy>;
    using value_type = mdspan::value_type;

    const auto num_images = g.extent(0);
    const auto height     = g.extent(1);
    const auto width      = g.extent(2);
    const auto r2         = sqr(r);

    static auto constexpr fp_align = uint32_t(std::hardware_destructive_interference_size / sizeof(value_type));
    const auto fp_stride     = uint32_t(((width + (fp_align - 1)) / fp_align) * fp_align);
    const auto fp_slice_size = uint32_t(fp_stride * height);

    using alloc = boost::alignment::aligned_allocator<value_type, std::hardware_destructive_interference_size>;
    auto create_storage = [&]() {
        return std::vector<value_type, alloc>(num_images * fp_slice_size, 0);
    };
    auto make_span = [&](auto &&storage) {
        return std::mdspan{
            storage.data()
          , std::layout_stride::mapping{
                std::extents{num_images, height, width}
              , std::array<uint32_t, 3>{fp_slice_size, fp_stride, 1}
              }
          };
    };
    auto p1_storage = create_storage();
    auto p1 = make_span(p1_storage);
    auto p2_storage = create_storage();
    auto p2 = make_span(p2_storage);
    auto p3_storage = create_storage();
    auto p3 = make_span(p3_storage);
    auto divp_storage = create_storage();
    auto divp = make_span(divp_storage);

    auto col_span = std::vector<std::tuple<uint32_t, uint32_t>>(height);
    auto row_span = std::vector<std::tuple<uint32_t, uint32_t>>(width);
    for (auto row = size_t{}; height > row; ++row) {
        auto 𝛿2 = r2 - sqr(row - yc);
        if (0 < 𝛿2) {
            auto 𝛿 = std::sqrt(𝛿2);
            auto c = std::ceil(xc - 𝛿);
            if (xc - 𝛿 == c) c += 1.0;
            auto start_col = uint32_t(std::max(0.0, c));
            auto f = std::floor(xc + 𝛿);
            if (xc + 𝛿 == f) f -= 1.0;
            auto end_col   = uint32_t(std::min(double(width - 1), f));
            col_span[row] = {start_col, end_col};
        } else {
            col_span[row] = {};
        }
    }
    for (auto col = size_t{}; width > col; ++col) {
        auto 𝛿2 = r2 - sqr(col - xc);
        if (0 < 𝛿2) {
            auto 𝛿 = std::sqrt(𝛿2);
            auto c = std::ceil(yc - 𝛿);
            if (yc - 𝛿 == c) c += 1.0;
            auto start_row = uint32_t(std::max(0.0, c));
            auto f = std::floor(yc + 𝛿);
            if (yc + 𝛿 == f) f -= 1.0;
            auto end_row   = uint32_t(std::min(double(height - 1), f));
            row_span[col] = {start_row, end_row};
        } else {
            row_span[col] = {};
        }
    }

    auto c = std::ceil(yc - r);
    if (yc - r == c) c += 1.0;
    auto start_row = uint32_t(std::max(0.0, c));
    auto f = std::floor(yc + r);
    if (yc + r == f) f -= 1.0;
    auto end_row   = uint32_t(std::min(double(height - 1), f));

    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        #pragma omp parallel
        for (auto n = tdir_t{}; num_images > n; ++n) {
            for (auto row = start_row; end_row >= row; ++row) {
                auto [start_col, end_col] = col_span[row];
                for (auto col = start_col; end_col >= col; ++col) {
                    auto d = value_type{};
                    if (num_images - 1 > n) d += p1[n, row, col];
                    if (0 < n)              d -= p1[n - 1, row, col];
                    if (end_col > col)      d += p3[n, row, col];
                    if (start_col < col)    d -= p3[n, row, col - 1];

                    auto [local_start_row, local_end_row] = row_span[col];
                    if (local_end_row > row)   d += p2[n, row, col];
                    if (local_start_row < row) d -= p2[n, row - 1, col];

                    divp[n, row, col] = d;
                }
            }
        }
        // auto start_time  = timespec{};
        // auto finish_time = timespec{};
        // clock_gettime(CLOCK_MONOTONIC, &start_time);
        auto maxΔq = 0.0;
        #pragma omp parallel for reduction(max:maxΔq)
        for (auto n = tdir_t{}; num_images > n; ++n) {
            for (auto row = start_row; end_row >= row; ++row) {
                auto [start_col, end_col] = col_span[row];
                for (auto col = start_col; end_col >= col; ++col) {
                    auto g1 = value_type{};
                    if (num_images - 1 > n) {
                        g1 = (divp[n + 1, row, col] - divp[n, row, col]) -
                             (g[n + 1, row, col] - g[n, row, col]) / 𝜆;
                    }
                    auto g2 = value_type{};
                    auto [local_start_row, local_end_row] = row_span[col];
                    if (local_end_row > row) {
                        g2 = (divp[n, row + 1, col] - divp[n, row, col]) -
                             (g[n, row + 1, col] - g[n, row, col]) / 𝜆;
                    }
                    auto g3 = value_type{};
                    if (end_col > col) {
                        g3 = (divp[n, row, col + 1] - divp[n, row, col]) -
                             (g[n, row, col + 1] - g[n, row, col]) / 𝜆;
                    }
                    auto denom = 1.0 + 𝜏 * std::sqrt(sqr(g1) + sqr(g2) + sqr(g3));
                    auto &q1 = p1[n, row, col];
                    auto &q2 = p2[n, row, col];
                    auto &q3 = p3[n, row, col];

                    auto next_q1 = (q1 + 𝜏 * g1) / denom;
                    auto next_q2 = (q2 + 𝜏 * g2) / denom;
                    auto next_q3 = (q3 + 𝜏 * g3) / denom;

                    auto Δq = sqr(q1 - next_q1) + sqr(q2 - next_q2) + sqr(q3 - next_q3);
                    maxΔq = std::max(maxΔq, Δq);

                    q1 = next_q1;
                    q2 = next_q2;
                    q3 = next_q3;
                }
            }
        }
        // clock_gettime(CLOCK_MONOTONIC, &finish_time);
        // auto elapsed_time = finish_time - start_time;
        // fmt::print(fmt::fg(fmt::color::medium_purple) | fmt::emphasis::bold,
        //            "Elapsed time: {:.6g} s\n", elapsed_time);
        fmt::print("{} {}\n", n_iter, maxΔq);
    }

    auto u_storage = create_storage();
    auto u = make_span(u_storage);
    #pragma omp parallel for
    for (auto n = tdir_t{}; num_images > n; ++n) {
        for (auto row = start_row; end_row >= row; ++row) {
            auto [start_col, end_col] = col_span[row];
            for (auto col = start_col; end_col >= col; ++col) {
                auto d = value_type{};

                if (num_images - 1 > n) d += p1[n, row, col];
                if (0 < n)              d -= p1[n - 1, row, col];
                if (end_col > col)      d += p3[n, row, col];
                if (start_col < col)    d -= p3[n, row, col - 1];

                auto [local_start_row, local_end_row] = row_span[col];
                if (local_end_row > row)   d += p2[n, row, col];
                if (local_start_row < row) d -= p2[n, row - 1, col];

                u[n, row, col] = g[n, row, col] - 𝜆 * d;
            }
        }
    }

    return std::tuple{std::move(u_storage), u};
}

int main(int argc, char *argv[]) {
    auto file_name = "sample1_recon_Export.tiff";
    auto tif = std::unique_ptr<TIFF, decltype(&TIFFClose)>{
        TIFFOpen(file_name, "r")
      , &TIFFClose
      };

    if (!tif) {
        fmt::print("Can't open \"{}\"\n", file_name);
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

    // align to 64 bytes
    static auto constexpr uint16_align = uint32_t(std::hardware_destructive_interference_size / sizeof(uint16_t));
    const auto stride     = ((width + (uint16_align - 1)) / uint16_align) * uint16_align;
    const auto slice_size = stride * height;

    using alloc = boost::alignment::aligned_allocator<uint16_t, 64>;
    auto original_img_storage = std::vector<uint16_t, alloc>(num_images * slice_size);
    auto original_img = std::mdspan{
        original_img_storage.data()
      , std::layout_stride::mapping{
            std::extents{num_images, height, width}
          , std::array<uint32_t, 3>{slice_size, stride, 1}
          }
      };

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
    for (auto n_dir = tdir_t{}; num_images > n_dir; ++n_dir) {
        TIFFSetDirectory(tif.get(), n_dir);

        for (auto row = uint32_t{}; height > row; ++row) {
            TIFFReadScanline(tif.get(), &original_img[n_dir, row, 0], row);
        }
        auto g = cv::Mat(height, width, CV_16UC1, &original_img[n_dir, 0, 0], stride * 2);
        circles.push_back(find_circular_mask(g));
    }
    auto [center, radius] = max_inscribed_circle(circles);
    fmt::print("mask radius = {}\n", radius);
    fmt::print("mask location = {}, {}\n", center.x, center.y);

    auto [lo, hi] = find_intensity_range(original_img, center.x, center.y, radius, 0.001);
    fmt::print("used  lo = {}\n", lo);
    fmt::print("used  hi = {}\n", hi);

    auto [img_storage, img] = convert_to_fp(original_img, center.x, center.y, radius, lo, hi);

    std::vector<uint16_t, alloc>{}.swap(original_img_storage);

    const auto nbins = 500;
    auto hist = calc_hist(img, center.x, center.y, radius, 1.0 / (hi - lo), nbins);
    auto f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen("original-hist.txt", "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) / nbins, hist[i]);
    }

    auto [u_storage, u] = chambolle(img, center.x, center.y, radius, 0.05f, 1.0f / 6.0f * 0.99f, 500);

    hist = calc_hist(u, center.x, center.y, radius, 1.0 / (hi - lo), nbins);
    f = std::unique_ptr<FILE, decltype(&fclose)>{
        fopen("hist (𝜆 = 0.05).txt", "wt")
      , &fclose
      };
    fmt::print(f.get(), "intensity ch0\n");
    for (auto i = size_t{1}; hist.size() - 1 > i; ++i) {
        fmt::print(f.get(), "{} {}\n", (i + 0.5) / nbins, hist[i]);
    }

    auto slice = cv::Mat(height, width, CV_32FC1, &u[479, 0, 0], img.stride(1) * 4);
    auto gamma_corrected = cv::Mat{};
    cv::pow(slice, 1.0 / 2.2, gamma_corrected);
    auto vis = cv::Mat{};
    gamma_corrected.convertTo(vis, CV_8U, 255.0);
    cv::imwrite("slice-479 (𝜆 = 0.05).png", vis);

    // auto slice = cv::Mat(height, width, CV_32FC1, &img[479, 0, 0], img.stride(1) * 4);
    // auto gamma_corrected = cv::Mat{};
    // cv::pow(slice, 1.0 / 2.2, gamma_corrected);
    // auto vis = cv::Mat{};
    // gamma_corrected.convertTo(vis, CV_8U, 255.0);
    // cv::imwrite("slice-479.png", vis);
}
