#include <cstdio>
#include <exception>
#include <filesystem>
#include <random>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/program_options.hpp>

#include <fmt/printf.h>

#include <Eigen/Dense>

#include <range/v3/algorithm/sample.hpp>
#include <range/v3/numeric/accumulate.hpp>

#include <io/biniarchive.h>

#include <calc_hist.h>
#include <cuda-context.h>
#include <field.h>
#include <timeop.h>

#include <gmm.h>

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace rs = ranges;
namespace rv = rs::views;

using namespace Eigen;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);

int main(int argc, char *argv[]) try {
    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto denoised_file_name = std::string{};
    auto noised_file_name   = std::string{};
    auto init_peaks         = std::vector<double>{};
    auto tail_thr           = 0.003f;

    cmd_line_options.add_options()
        ("denoised", po::value<std::string>(&denoised_file_name)->required(), "Denoised input file")
        ("noised", po::value<std::string>(&noised_file_name)->required(), "Raw input file")
        ("peaks", po::value(&init_peaks)->multitoken(), "Initial peaks")
        ("tailthr", po::value(&tail_thr)->multitoken(), "Threshold to cut the tail of the histogram")
      ;

    try {
        po::store(po::command_line_parser(argc, argv).
                  positional(positional).
                  options(cmd_line_options).run(), vm);
        po::notify(vm);
    } catch (po::error &x) {
        fmt::print(stderr, "Command line error: {}\n", x.what());
        return 1;
    }

    try {
        if (!fs::exists(denoised_file_name)) {
            fmt::print(stderr, "Input file {} doesn't exist\n", denoised_file_name);
            return 1;
        }
        if (!fs::exists(noised_file_name)) {
            fmt::print(stderr, "Input file {} doesn't exist\n", noised_file_name);
            return 1;
        }
    } catch (std::exception &x) {
        fmt::print(stderr, "{}\n", x.what());
        return 1;
    }

    auto denoised_file = boost::iostreams::mapped_file_source{denoised_file_name};
    auto it            = denoised_file.begin();
    auto end           = it + denoised_file.size();
    auto ia            = io::BinIArchive{it, end};

    auto t = Timer{};
    auto denoised_img = Field<float>{};
    t.start();
        ia >> denoised_img;
    t.stop("Read time");

    auto lo = denoised_img.lo();
    auto hi = denoised_img.hi();

    auto hi_data_thr = 0.0f;
    {
        auto cuda_context = make_context();
        t.start();
            auto d_denoised_img = upload(*cuda_context, denoised_img.view());
        t.stop("Upload to GPU time");
        const auto nbins = 500;
        auto 𝛿 = 1.0 / (denoised_img.hi() - denoised_img.lo());
        𝛿 = sqr(𝛿) < 0.5 / nbins ? 𝛿 : 0.0;
        t.start();
            auto [hist, min, max] = calc_hist(*cuda_context, *d_denoised_img, nbins, 𝛿);
        t.stop("Histogram time");
        const auto total      = rs::accumulate(hist, float{});
        auto const pct_hi_thr = (1.0 - tail_thr) * total;
        auto hi_thr_idx = hist.size();
        auto hist_cum = float{};
        for (auto i = size_t{}; hist.size() > i; ++i) {
            hist_cum += hist[i];
            if (pct_hi_thr <= hist_cum) { hi_thr_idx = i; break; }
        }
        hi_data_thr = float(hi_thr_idx) / hist.size();
    }

    using samples_t  = Matrix<double,  1, Dynamic, RowMajor>;
    using centers_t  = Matrix<double, 1, Dynamic, RowMajor>;
    using clusters_t = Matrix<int,  Dynamic, 1, ColMajor>;

    auto centers = centers_t{Map<const centers_t>(init_peaks.data(), init_peaks.size())};

    auto v = denoised_img.view();
    auto m = v.mapping();
    auto linear_data = std::vector<float>(m.data_size());
    auto eligible    = std::vector<size_t>();
    eligible.reserve(m.data_size());
    t.start();
    auto count = size_t{};
    for (auto col = m.col_begin(); m.col_end() > col; ++col) {
        for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
            for (auto k = uint32_t{}; m.n_images() > k; ++k) {
                auto s = linear_data[count] = v[k, row, col];
                if (hi_data_thr >= s) {
                    eligible.push_back(count);
                }
                ++count;
            }
        }
    }
    t.stop("Linearize time");

    static auto constexpr N = 10'000'000;

    auto rd       = std::random_device{};
    auto gen      = std::mt19937{rd()};
    auto samples  = samples_t{1, N};
    auto indices  = std::vector<size_t>(N);

    if (eligible.size() > N) {
        rs::sample(eligible, indices.begin(), N, gen);
    } else {
        indices = eligible;
    }

    auto clusters = clusters_t{indices.size(), 1};
    auto dist     = VectorXf{init_peaks.size()};
    for (auto i = size_t{}; indices.size() > i; ++i) {
        samples[i] = lo + (hi - lo) * linear_data[indices[i]];
        for (auto c = size_t{}; init_peaks.size() > c; ++c) {
            dist[c] = std::abs(centers[c] - samples[i]);
        }
        auto cluster = Index{};
        dist.minCoeff(&cluster);
        clusters[i] = cluster;
    }
    auto [log_ℒ, 𝜎, 𝜇, new_clust] = gmm(
        samples
      , centers
      , clusters
      , 10
      , [&](int attempt, int iter, double log_ℒ, auto &&covs, auto &&centers, auto &&clusters) {
            fmt::print(
                "attempt {:3}, iteration {:4}: NLL = {:.9f}\n",
                attempt, iter, log_ℒ);
        }
      );
    fmt::print("NLL = {:.9f}\n", log_ℒ);
    using cov_t = Matrix<double, 1, 1>;
    for (auto k = size_t{}; init_peaks.size() > k; ++k) {
        auto es = SelfAdjointEigenSolver<cov_t>(𝜎[k]);
        const auto &evals = es.eigenvalues();
        fmt::print("Cluster {}:\n", k);
        fmt::print("    Deviations along axes: {:6g}\n", std::sqrt(evals(0)));
        fmt::print("    Center: {:6g}\n", 𝜇(0, k));
    }

    auto noised_file = boost::iostreams::mapped_file_source{noised_file_name};
    it               = noised_file.begin();
    end              = it + noised_file.size();
    auto noised_ia   = io::BinIArchive{it, end};

    auto noised_img = Field<float>{};
    t.start();
        noised_ia >> noised_img;
    t.stop("Noised image read time");

    auto nv = noised_img.view();
    auto nm = nv.mapping();
    if (nm.data_size() != m.data_size()) {
        fmt::print(stderr, "Size mismatch: raw {} vs denoised {}\n",
                   nm.data_size(), m.data_size());
        return 1;
    }

    t.start();
    count = 0;
    for (auto col = nm.col_begin(); nm.col_end() > col; ++col) {
        for (auto row = nm.row_begin(col); nm.row_end(col) > row; ++row) {
            for (auto k = uint32_t{}; nm.n_images() > k; ++k) {
                linear_data[count++] = nv[k, row, col];
            }
        }
    }
    t.stop("Linearize time");

    const auto K = init_peaks.size();
    auto ncount  = std::vector<size_t>(K, 0);
    auto n𝜇      = std::vector<double>(K, 0);
    auto n𝜎²     = std::vector<double>(K, 0);

    t.start();
    for (auto i = size_t{}; indices.size() > i; ++i) {
        const auto idx = indices[i];
        const auto lbl = new_clust[i];
        const auto val = lo + (hi - lo) * double(linear_data[idx]);

        auto 𝛿 = val - n𝜇[lbl];
        n𝜇[lbl] += 𝛿 / (ncount[lbl] + 1);
        if (0 == ncount[lbl]) {
            n𝜎²[lbl] = 0;
        } else {
            n𝜎²[lbl] += (𝛿 * 𝛿) / (ncount[lbl] + 1) - n𝜎²[lbl] / ncount[lbl];
        }
        ++ncount[lbl];
    }
    t.stop("Raw statistics time");

    for (auto k = size_t{}; K > k; ++k) {
        fmt::print("Phase {} (raw data): c = {:.6g}, sigma = {:.6g}, fraction = {:.4g}\n",
                   k, n𝜇[k], std::sqrt(n𝜎²[k]), double(ncount[k]) / indices.size());
    }
}
catch (std::exception &x) {
    fmt::print(stderr, "{}\n", x.what());
    return 1;
}

// sample1_recon_Export.tiff
// Cluster 0:
//     Deviations along axes: 0.0196109
//     Center: 0.166565
// Cluster 1:
//     Deviations along axes: 0.0179284
//     Center: 0.231481
// Cluster 2:
//     Deviations along axes: 0.0746632
//     Center: 0.423983
// Cluster 3:
//     Deviations along axes: 0.0237601
//     Center: 0.451109
// Cluster 4:
//     Deviations along axes: 0.0202666
//     Center: 0.530193
// Phase 0 (raw data): c = 0.155431, sigma = 0.0657028, fraction = 0.07526
// Phase 1 (raw data): c = 0.22989,  sigma = 0.0617947, fraction = 0.07489
// Phase 2 (raw data): c = 0.349481, sigma = 0.11344,   fraction = 0.02923
// Phase 3 (raw data): c = 0.451221, sigma = 0.071215,  fraction = 0.7909
// Phase 4 (raw data): c = 0.580403, sigma = 0.0593533, fraction = 0.02975
