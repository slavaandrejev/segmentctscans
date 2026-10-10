#include <concepts>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <random>
#include <system_error>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/parser/parser.hpp>
#include <boost/program_options.hpp>

#include <fmt/printf.h>

#include <gsl/gsl_multimin.h>

#include <range/v3/algorithm/copy.hpp>
#include <range/v3/algorithm/sample.hpp>
#include <range/v3/algorithm/shuffle.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/single.hpp>
#include <range/v3/view/transform.hpp>

#include <io/biniarchive.h>

#include <calc_hist.h>
#include <cuda-context.h>
#include <field.h>
#include <grad_magn.h>
#include <timeop.h>

#include "gap_set_project.h"
#include "layout_strict_upper.h"
#include "puv.h"

namespace bc = boost::container;
namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace rs = ranges;
namespace rv = rs::views;

extern template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
extern template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);

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
    if (!result || data.size() < 2 || data.size() > max_phases) {
        throw validation_error(validation_error::invalid_option_value);
    }
    v = boost::any(LabelsArray{.data = data});
}

static auto constexpr max_params = max_phases * (max_phases + 3) / 2 + 3;

template <typename  CostFunc, typename ProgressFunc>
auto gsl_minimize(
    CostFunc &&func
  , std::span<const double> x_init
  , std::span<const double> step_sizes
  , size_t max_iter
  , double stop_size
  , ProgressFunc &&pr
  )
{
    struct Adapt {
        const CostFunc &func;
        static double invoke(const gsl_vector *x, void *params) {
            auto self = static_cast<Adapt *>(params);

            auto xx = bc::static_vector<double, max_params>(x->size);
            for (auto i = 0; xx.size() > i; ++i) {
                xx[i] = gsl_vector_get(x, i);
            }

            return self->func(xx);
        }
    };

    auto const n = x_init.size();

    auto s = std::unique_ptr<
        gsl_multimin_fminimizer
      , decltype(&gsl_multimin_fminimizer_free)
      >
    {
        gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, n),
        &gsl_multimin_fminimizer_free
    };

    auto adapt      = Adapt{std::forward<CostFunc>(func)};
    auto minex_func = gsl_multimin_function{&Adapt::invoke, n, &adapt};

    auto gsl_x_init = std::unique_ptr<gsl_vector, decltype(&gsl_vector_free)>{
        gsl_vector_alloc(n)
      , &gsl_vector_free
      };
    auto gsl_step_size = std::unique_ptr<gsl_vector, decltype(&gsl_vector_free)>{
        gsl_vector_alloc(n)
      , &gsl_vector_free
      };

    for (auto i = 0; gsl_x_init->size > i; ++i) {
        gsl_vector_set(gsl_x_init.get(), i, x_init[i]);
    }
    for (auto i = 0; gsl_step_size->size > i; ++i) {
        gsl_vector_set(gsl_step_size.get(), i, step_sizes[i]);
    }

    auto status = gsl_multimin_fminimizer_set(s.get(), &minex_func, gsl_x_init.get(), gsl_step_size.get());

    auto x = bc::static_vector<double, max_params>(s->x->size);

    if (GSL_SUCCESS != status) {
        return std::tuple{status, x, 0.0};
    }

    auto cost = 0.0;
    for (auto iter = 0; max_iter > iter; ++iter) {
        status = gsl_multimin_fminimizer_iterate(s.get());
        if (0 != status) break;

        auto size = gsl_multimin_fminimizer_size(s.get());
        status = gsl_multimin_test_size(size, stop_size);

        cost = gsl_multimin_fminimizer_minimum(s.get());
        for (auto i = 0; x.size() > i; ++i) {
            x[i] = gsl_vector_get(s->x, i);
        }
        if (pr(iter, cost, size, x)) {
            for (auto i = 0; gsl_step_size->size > i; ++i) {
                auto step = gsl_vector_get(gsl_step_size.get(), i);
                step *= 0.95;
                gsl_vector_set(gsl_step_size.get(), i, step);
            }
            status = gsl_multimin_fminimizer_set(s.get(), &minex_func, s->x, gsl_step_size.get());
            if (GSL_SUCCESS != status) {
                break;
            }
        } else if (status != GSL_CONTINUE) {
            break;
        }

    }

    if (GSL_CONTINUE == status) {
        status = GSL_SUCCESS;
    }

    return std::tuple{status, x, cost};
}

std::pair<float, float> tail_cutoffs(
    const std::vector<double> &hist
  , double lo_tail_thr
  , double hi_tail_thr
  , double range_start
  , double range_end
  )
{
    const auto bin_width = (range_end - range_start) / hist.size();
    const auto total     = rs::accumulate(hist, 0.0);

    auto quantile = [&](double p) -> float {
        if (p <= 0) return float(range_start);
        if (p >= 1) return float(range_end);

        const auto target = p * total;
        auto cum = 0.0;

        for (size_t i = 0; i < hist.size(); ++i) {
            const auto next = cum + hist[i];
            if (hist[i] > 0 && target <= next) {
                const auto fraction = (target - cum) / hist[i];
                return float(range_start + (i + fraction) * bin_width);
            }
            cum = next;
        }
        return float(range_end);
    };

    return {quantile(lo_tail_thr), quantile(1.0 - hi_tail_thr)};
}


std::pair<float, float> tail_cutoffs(
    CudaContext &ctx
  , Field<float> &img
  , double lo_tail_thr
  , double hi_tail_thr
  , double range_start
  , double range_end
  , double jitter = 0.0
  )
{
    static constexpr auto nbins = 1024;

    auto d_img = upload(ctx, img.view());
    jitter = sqr(jitter) < 0.5 / nbins ? jitter : 0.0;
    auto hist = calc_hist(ctx, *d_img, range_start, range_end, nbins, jitter);

    return tail_cutoffs(hist, lo_tail_thr, hi_tail_thr, range_start, range_end);
}

std::pair<float, float> tail_cutoffs(
    CudaContext &ctx
  , Field<float> &img
  , double lo_tail_thr
  , double hi_tail_thr
  , double jitter = 0.0
  )
{
    static constexpr auto nbins = 1024;

    auto d_img = upload(ctx, img.view());
    jitter = sqr(jitter) < 0.5 / nbins ? jitter : 0.0;
    auto [hist, range_start, range_end] = calc_hist(ctx, *d_img, nbins, jitter);

    return tail_cutoffs(hist, lo_tail_thr, hi_tail_thr, range_start, range_end);
}

int main(int argc, char *argv[]) try {
    gsl_set_error_handler_off();

    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name  = std::string{};
    auto pdf_name      = std::string{};
    auto init_peaks    = LabelsArray{};
    auto both_tail_thr = 0.01;
    auto lo_tail_thr   = 0.01;
    auto hi_tail_thr   = 0.01;
    auto v_tail_thr    = 0.01;
    auto max_ds        = 1.5;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "Input file")
        ("peaks", po::value(&init_peaks)->required(), "Initial peaks")
        ("pdf-name", po::value(&pdf_name), "Base name to write resulting distributions")
        ("tailthr", po::value(&both_tail_thr), "Threshold to cut the tail of the histogram")
        ("lotailthr", po::value(&lo_tail_thr), "Lower threshold to cut the tail of the histogram")
        ("hitailthr", po::value(&hi_tail_thr), "Higher threshold to cut the tail of the histogram")
        ("vtailthr", po::value(&v_tail_thr), "Threshold to cut the gradient magnitude tail of the histogram")
        ("maxds", po::value(&max_ds), "Maximum interface thickness in 𝜎b units")
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

    auto cuda_context = make_context();
    t.start();
        auto d_img = upload(*cuda_context, img.view());
    t.stop("Upload to GPU time");

    t.start();
        auto [d_grad_magn_img, vmin, vmax] = grad_magn(*cuda_context, *d_img);
    t.stop("Gradient magnitude time");

    auto grad_magn_img = Field<float>(img.view().mapping());
    t.start();
        download(*cuda_context, *d_grad_magn_img, grad_magn_img.view());
    t.stop("Download from GPU time");
    d_img.reset();
    d_grad_magn_img.reset();

    auto lo = img.lo();
    auto hi = img.hi();

    if (0 != vm.count("tailthr")) {
        lo_tail_thr = hi_tail_thr = both_tail_thr;
    }

    auto [lo_u_thr, hi_u_thr] = tail_cutoffs(
        *cuda_context
      , img
      , lo_tail_thr
      , hi_tail_thr
      , 1.0 / (hi - lo)
      );
    fmt::print("Fitting [{}, {}] range of intensities\n", lo_u_thr * (hi - lo) + lo, hi_u_thr * (hi - lo) + lo);

    auto [lo_v_thr, hi_v_thr] = tail_cutoffs(
        *cuda_context
      , grad_magn_img
      , 0
      , v_tail_thr
      , 0
      , vmax
      );
    fmt::print("Fitting [{}, {}] range of gradients\n", lo_v_thr * (hi - lo), hi_v_thr * (hi - lo));

    auto u_view = img.view();
    auto v_view = grad_magn_img.view();
    auto m      = u_view.mapping();
    auto linear_u = std::vector<float>(m.data_size());
    auto linear_v = std::vector<float>(m.data_size());
    auto eligible = std::vector<size_t>();
    eligible.reserve(m.data_size());
    t.start();
    auto count = size_t{};
    for (auto col = m.col_begin(); m.col_end() > col; ++col) {
        for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
            for (auto k = uint32_t{}; m.n_images() > k; ++k) {
                linear_u[count] = u_view[k, row, col];
                linear_v[count] = v_view[k, row, col];
                if (
                    0 < v_view[k, row, col] && v_view[k, row, col] <= hi_v_thr &&
                    lo_u_thr <= u_view[k, row, col] && u_view[k, row, col] <= hi_u_thr)
                {
                    eligible.push_back(count);
                }
                ++count;
            }
        }
    }
    t.stop("Linearize time");

    static auto constexpr N              = 10'000;
    static auto constexpr N_batches      = 100;
    static auto constexpr restart_period = 100;

    auto rd        = std::random_device{};
    auto seed      = rd();
    auto gen       = std::mt19937{seed};
    auto indices   = std::vector<size_t>(N * N_batches);
    auto samples_u = std::vector<double>(N * N_batches);
    auto samples_v = std::vector<double>(N * N_batches);

    if (eligible.size() < indices.size()) {
        fmt::print(stderr, "Not enough eligible voxels\n");
        return 1;
    }

    auto reset_samples = [&]() {
        rs::sample(eligible, indices.begin(), indices.size(), gen);
        rs::shuffle(indices, gen);
        for (auto i = size_t{}; indices.size() > i; ++i) {
            samples_u[i] = linear_u[indices[i]];
            samples_v[i] = linear_v[indices[i]];
        }
    };

    const auto n_labels = init_peaks.data.size();
    const auto n_params = int(n_labels * (n_labels + 3) / 2 + 3);

    const auto n_weights   = n_labels * (n_labels + 1) / 2 - 1;
    const auto I_start_idx = n_weights;
    const auto 𝜎n_idx      = I_start_idx + n_labels;
    const auto 𝜎b_idx      = 𝜎n_idx + 1;
    const auto 𝜌_idx       = 𝜎b_idx + 1;
    const auto ds_idx      = 𝜌_idx  + 1;

    const auto w_min  = -4.6,   w_max  = 4.6; // somewhat logarithm of the real range,
                                              // will be transformed by softmax
    const auto I_min  =  0.0,   I_max  = 1.0;
    const auto 𝜎n_min = std::log(1e-3), 𝜎n_max = std::log(2.0);   // logarithm of the real range
    const auto 𝜎b_min = std::log(0.1),  𝜎b_max = std::log(10.0);  // logarithm of the real range
    const auto 𝜌_min  = 0.0,            𝜌_max  = 0.999;
    const auto ds_min = std::log(0.5), ds_max = std::log(max_ds); // logarithm of the real range

    // fix the first weight to zero, then use softmax to produce probability weights
    auto get_weights = [](std::span<const double> a) {
        auto all_w = rv::concat(rv::single(0.0), a);
        auto exp_vew = all_w | rv::transform([](auto x) {
            return std::exp(x);
        });
        auto sum = rs::accumulate(exp_vew, 0.0);

        auto w = bc::static_vector<double, max_params>(a.size() + 1);
        auto w_view = all_w | rv::transform([&](auto x) {
            return std::exp(x) / sum;
        });
        rs::copy(w_view, w.begin());

        return w;
    };

    auto get_labels = [](std::span<const double> a) {
#if 0
        return gap_set_project(a, 0.1);
#else
        return bc::static_vector<double, max_phases>(a.begin(), a.end());
#endif
    };

    auto sigm = [](double unr, double min, double max) {
        return (max - min) / (1.0 + std::exp(-unr)) + min;
    };
    // assume parameters are clamped between min and max with a sigmoid
    auto restrict_params = [&](std::span<double> x) {
        auto restr = bc::static_vector<double, max_params>(n_params);

        for (auto i = 0; n_weights > i; ++i) {
            restr[i] = sigm(x[i], w_min, w_max);
        }
        for (auto i = 0; n_labels > i; ++i) {
            restr[I_start_idx + i] = sigm(x[I_start_idx + i], I_min, I_max);
        }
        restr[𝜎n_idx] = sigm(x[𝜎n_idx], 𝜎n_min, 𝜎n_max);
        restr[𝜎b_idx] = sigm(x[𝜎b_idx], 𝜎b_min, 𝜎b_max);
        restr[𝜌_idx]  = sigm(x[𝜌_idx],  𝜌_min,  𝜌_max);
        restr[ds_idx] = sigm(x[ds_idx], ds_min, ds_max);

        return restr;
    };

    auto cost_func = [&](std::span<double> x, int batch_start, int batch_end) {
        auto res = 0.0;

        auto restr = restrict_params(x);

        auto w  = get_weights({&restr[0], n_weights});
        auto Is = get_labels({&restr[I_start_idx], n_labels});

        auto Z = 0.0;
        for (auto i = 0; n_labels > i; ++i) {
            Z += w[i] * M_i(
                lo_u_thr
              , hi_u_thr
              , hi_v_thr
              , Is[i]
              , std::exp(restr[𝜎n_idx])
              , restr[𝜌_idx]
              );
        }
        auto wij = strict_upper_span<const double>{&w[n_labels], n_labels, n_labels};
        for (auto i = 0; n_labels - 1 > i; ++i) {
            for (auto j = i + 1; n_labels > j; ++j) {
                Z += wij[i, j] * M_ij(
                    lo_u_thr
                  , hi_u_thr
                  , hi_v_thr
                  , Is[i]
                  , Is[j]
                  , std::exp(restr[𝜎n_idx])
                  , std::exp(restr[𝜎b_idx])
                  , std::exp(restr[ds_idx])
                  , restr[𝜌_idx]
                  );
            }
        }

        #pragma omp parallel for reduction(+:res) schedule(dynamic)
        for (auto i = batch_start; batch_end > i; ++i) {
            res += -std::log(full_p(
                samples_u[i]
              , samples_v[i]
              , w
              , Is
              , restr[𝜎n_idx]
              , restr[𝜎b_idx]
              , restr[ds_idx]
              , restr[𝜌_idx]
              ));
        }

        return res / (batch_end - batch_start) + std::log(Z);
    };

    auto print_log = [&](int iter, double cost, double size, std::span<double> x) {
        auto restr = restrict_params(x);

        auto w  = get_weights({&restr[0], n_weights});
        auto Is = get_labels({&restr[I_start_idx], n_labels});
        fmt::print("{:5} {:14.9f} {:9.6f} |", iter, cost, size);
        for (auto i = 0; w.size() > i; ++i) {
            fmt::print("{:9.6f}", w[i]);
        }
        fmt::print(" |");
        for (auto i = 0; Is.size() > i; ++i) {
            fmt::print("{:6.3f}", Is[i]);
        }
        fmt::print(" |");
        fmt::print("{:7.4f}", std::exp(restr[𝜎n_idx]));
        fmt::print("{:7.4f}", std::exp(restr[𝜎b_idx]));
        fmt::print("{:7.4f}", std::exp(restr[ds_idx]));
        fmt::print("{:7.4f}\n", restr[𝜌_idx]);
    };

    auto inv_sigm = [](double restr, double min, double max) {
        return -std::log((max - min) / (restr - min) - 1.0);
    };
    auto x0 = bc::static_vector<double, max_params>(n_params, 0);
    for (auto i = 0; n_labels > i; ++i) {
        x0[I_start_idx + i] = inv_sigm((init_peaks.data[i] - lo) / (hi - lo), I_min, I_max);
    }
    for (auto i = n_labels - 1; n_weights > i; ++i) {
        x0[i] = inv_sigm(w_min * 0.99, w_min, w_max); // minimize interface terms
    }

    auto step_sizes = bc::static_vector<double, max_params>(n_params, 1.0);
    reset_samples();
    auto batch_count = 0;
    auto [status, res, cost] = gsl_minimize(
        [&](std::span<double> x) { return cost_func(x, N * batch_count, N * batch_count + N); }
      , x0
      , step_sizes
      , restart_period * N_batches
      , 1e-5
      , [&](int iter, double cost, double size, std::span<double> x) {
            print_log(iter, cost, size, x);

            auto change_batch = 0 != iter && 0 == iter % restart_period;

            if (change_batch) {
                batch_count = (batch_count + 1) % N_batches;
            }

            return change_batch;
        }
      );

    // Run on the full dataset
    std::tie(status, res, cost) = gsl_minimize(
        [&](std::span<double> x) { return cost_func(x, 0, N * N_batches); }
      , res
      , step_sizes
      , 10'000
      , 1e-4
      , [&](int iter, double cost, double size, std::span<double> x) {
            print_log(iter, cost, size, x);
            return false;
        }
      );
    if (GSL_SUCCESS != status) {
        fmt::print(stderr,
            "GSL fminimizer failed: {} (status = {})\n"
          , gsl_strerror(status), status
          );
        return 1;
    }

    auto restr = restrict_params(res);

    fmt::print(
        "𝜎b = {}, 𝜎n = {}, ds = {}, 𝜌 = {}\n"
      , std::exp(restr[𝜎b_idx])
      , std::exp(restr[𝜎n_idx]) * (hi - lo)
      , std::exp(restr[ds_idx])
      , restr[𝜌_idx]);

    auto w  = get_weights({&restr[0], n_weights});
    auto Is = get_labels({&restr[I_start_idx], n_labels});
    for (auto i = 0; n_labels > i; ++i) {
        fmt::print("{}\n", (hi - lo) * Is[i] + lo);
    }

    for (auto i = 0; n_labels > i; ++i) {
        fmt::print("w[{}] = {}\n", i, w[i]);
    }
    auto wij = strict_upper_span<const double>{&w[n_labels], n_labels, n_labels};
    for (auto i = 0; n_labels - 1 > i; ++i) {
        for (auto j = i + 1; n_labels > j; ++j) {
            fmt::print("w[{}, {}] = {}\n", i, j, wij[i, j]);
        }
    }

    if (0 != vm.count("pdf-name")) {
        static auto constexpr steps = 512;

        auto full_pdf_file_name = pdf_name + "-full.txt";
        auto full_pdf_file = std::unique_ptr<FILE, decltype(&fclose)>{
            fopen(full_pdf_file_name.c_str(), "wt")
          , &fclose
          };
        if (!full_pdf_file) {
            throw std::system_error(
                errno
              , std::system_category()
              , fmt::format("Cannot open {}\n", full_pdf_file_name).c_str());
        }
        fmt::print(full_pdf_file.get(), "u pdf\n");

        const auto step = (hi_tail_thr - lo_tail_thr) / steps;
        for (auto i = 0; steps >= i; ++i) {
            auto u = (lo_tail_thr + i * step);
            auto pdf = marginal_p(
                u
              , w
              , Is
              , std::exp(restr[𝜎n_idx])
              , std::exp(restr[ds_idx])
              );
            fmt::print(full_pdf_file.get(), "{} {}\n", u * (hi - lo) + lo, pdf / (hi - lo));
        }
        for (auto c = 0; Is.size() > c; ++c) {
            auto pdf_file_name = fmt::format("{}-{:02}.txt", pdf_name, c);
            auto pdf_file = std::unique_ptr<FILE, decltype(&fclose)>{
                fopen(pdf_file_name.c_str(), "wt")
              , &fclose
              };
            if (!pdf_file) {
                throw std::system_error(
                    errno
                  , std::system_category()
                  , fmt::format("Cannot open {}\n", pdf_file_name).c_str());
            }
            fmt::print(pdf_file.get(), "u pdf\n");
            for (auto i = 0; steps >= i; ++i) {
                auto u = (lo_tail_thr + i * step);
                auto pdf = w[c] * marginal_p_i(
                    u
                  , Is[c]
                  , std::exp(restr[𝜎n_idx])
                  );
                fmt::print(pdf_file.get(), "{} {}\n", u * (hi - lo) + lo, pdf / (hi - lo));
            }
        }
        auto idx = Is.size();
        for (auto i = 0; Is.size() - 1 > i; ++i) {
            for (auto j = i + 1; Is.size() > j; ++j) {
                auto pdf_file_name = fmt::format("{}-{:02}-{:02}.txt", pdf_name, i, j);
                auto pdf_file = std::unique_ptr<FILE, decltype(&fclose)>{
                    fopen(pdf_file_name.c_str(), "wt")
                  , &fclose
                  };
                if (!pdf_file) {
                    throw std::system_error(
                        errno
                      , std::system_category()
                      , fmt::format("Cannot open {}\n", pdf_file_name).c_str());
                }
                fmt::print(pdf_file.get(), "u pdf\n");
                for (auto k = 0; steps >= k; ++k) {
                    auto u = (lo_tail_thr + k * step);
                    auto pdf = w[idx] * marginal_p_ij(
                        u
                      , Is[i]
                      , Is[j]
                      , std::exp(restr[𝜎n_idx])
                      , std::exp(restr[ds_idx])
                      );
                    fmt::print(pdf_file.get(), "{} {}\n", u * (hi - lo) + lo, pdf / (hi - lo));
                }
                ++idx;
            }
        }
    }
}
catch (std::exception &x) {
    fmt::print(stderr, "{}\n", x.what());
    return 1;
}
