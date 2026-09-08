#include <concepts>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <random>
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
#include <timeop.h>

#include "gap_set_project.h"
#include "grad_magn.h"
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
        return std::tuple{status, x};
    }

    for (auto iter = 0; max_iter > iter; ++iter) {
        status = gsl_multimin_fminimizer_iterate(s.get());
        if (0 != status) break;

        auto size = gsl_multimin_fminimizer_size(s.get());
        status = gsl_multimin_test_size(size, stop_size);

        auto cost = gsl_multimin_fminimizer_minimum(s.get());
        for (auto i = 0; x.size() > i; ++i) {
            x[i] = gsl_vector_get(s->x, i);
        }
        if (pr(iter, cost, size, x)) {
            for (auto i = 0; gsl_step_size->size > i; ++i) {
                auto step = gsl_vector_get(gsl_step_size.get(), i);
                step *= 0.97;
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

    return std::tuple{status, x};
}

int main(int argc, char *argv[]) try {
    gsl_set_error_handler_off();

    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name             = std::string{};
    auto pdf_name                 = std::string{};
    auto init_peaks               = LabelsArray{};
    auto tail_thr                 = 0.01;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "Input file")
        ("peaks", po::value(&init_peaks)->required(), "Initial peaks")
        ("pdf-name", po::value(&pdf_name), "Base name to write resulting distributions")
        ("tailthr", po::value(&tail_thr), "Threshold to cut the tail of the histogram")
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
        auto d_grad_magn_img = grad_magn(*cuda_context, *d_img);
    t.stop("Gradient magnitude time");

    auto grad_magn_img = Field<float>(img.view().mapping());
    t.start();
        download(*cuda_context, *d_grad_magn_img, grad_magn_img.view());
    t.stop("Download from GPU time");
    d_img.reset();
    d_grad_magn_img.reset();

    auto lo = img.lo();
    auto hi = img.hi();

    auto hi_data_thr = 0.0f;
    auto lo_data_thr = 0.0f;
    {
        t.start();
            auto d_img = upload(*cuda_context, img.view());
        t.stop("Upload to GPU time");
        const auto nbins = 1024;
        auto 𝛿 = 1.0 / (hi - lo);
        𝛿 = sqr(𝛿) < 0.5 / nbins ? 𝛿 : 0.0;
        t.start();
            auto [hist, min, max] = calc_hist(*cuda_context, *d_img, nbins, 𝛿);
        t.stop("Histogram time");

        const auto bin_width = (double(max) - double(min)) / nbins;

        auto const pct_lo_thr = tail_thr / bin_width;
        auto const pct_hi_thr = (1.0 - tail_thr) / bin_width;

        auto lo_thr_idx = size_t{}, hi_thr_idx = hist.size();

        auto cum = 0.0;
        for (auto i = size_t{}; hist.size() > i; ++i) {
            cum += hist[i];
            if (pct_lo_thr >= cum) { lo_thr_idx = i; }
            if (pct_hi_thr <= cum) { hi_thr_idx = i; break; }
        }

        lo_data_thr = float(lo_thr_idx * bin_width + min);
        hi_data_thr = float(hi_thr_idx * bin_width + min);
    }
    fmt::print("Fitting [{}, {}] range of intensities\n", lo_data_thr * (hi - lo) + lo, hi_data_thr * (hi - lo) + lo);

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
                if (0 < v_view[k, row, col] && lo_data_thr <= u_view[k, row, col] && u_view[k, row, col] <= hi_data_thr) {
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
    auto gen       = std::mt19937{rd()};
    auto indices   = std::vector<size_t>(N * N_batches);
    auto samples_u = std::vector<double>(N * N_batches);
    auto samples_v = std::vector<double>(N * N_batches);

    if (eligible.size() < indices.size()) {
        fmt::print(stderr, "Not enough eligible voxels\n");
        return 1;
    }

    rs::sample(eligible, indices.begin(), indices.size(), gen);
    rs::shuffle(indices, gen);
    for (auto i = size_t{}; indices.size() > i; ++i) {
        samples_u[i] = linear_u[indices[i]];
        samples_v[i] = linear_v[indices[i]];
    }

    const auto n_labels = init_peaks.data.size();
    const auto n_params = int(n_labels * (n_labels + 3) / 2 + 3);

    const auto n_weights   = n_labels * (n_labels + 1) / 2 - 1;
    const auto I_start_idx = n_weights;
    const auto 𝜎n_idx      = I_start_idx + n_labels;
    const auto 𝜎b_idx      = 𝜎n_idx + 1;
    const auto 𝜌_idx       = 𝜎b_idx + 1;
    const auto ds_idx      = 𝜌_idx  + 1;

    const auto w_min  = -4.6,   w_max  = 4.6;    // somewhat logarithm of the real range,
                                                 // will be transformed by softmax
    const auto I_min  =  0.0,   I_max  = 1.0;
    const auto 𝜎n_min = -7.0,   𝜎n_max = 0.69;   // logarithm of the real range
    const auto 𝜎b_min = -2.3,   𝜎b_max = 2.3;    // logarithm of the real range
    const auto 𝜌_min  =  0.0,   𝜌_max  = 0.999;
    const auto ds_min = -0.69,  ds_max = 1.0986; // logarithm of the real range

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

    auto batch_start = 0;
    auto batch_end   = N;
    auto cost_func = [&](std::span<double> x) {
        auto res = 0.0;

        auto restr = restrict_params(x);

        auto w  = get_weights({&restr[0], n_weights});
        auto Is = get_labels({&restr[I_start_idx], n_labels});

        auto Z = 0.0;
        for (auto i = 0; n_labels > i; ++i) {
            Z += w[i] * M_i(lo_data_thr, hi_data_thr, Is[i], std::exp(restr[𝜎n_idx]));
        }
        auto wij = strict_upper_span<const double>{&w[n_labels], n_labels, n_labels};
        for (auto i = 0; n_labels - 1 > i; ++i) {
            for (auto j = i + 1; n_labels > j; ++j) {
                Z += wij[i, j] * M_ij(
                    lo_data_thr
                  , hi_data_thr
                  , Is[i]
                  , Is[j]
                  , std::exp(restr[𝜎n_idx])
                  , std::exp(restr[ds_idx]));
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
    auto batch_count = 0;
    auto step_sizes = bc::static_vector<double, max_params>(n_params, 1.0);
    auto [status, res] = gsl_minimize(
        cost_func
      , x0
      , step_sizes
      , restart_period * N_batches
      , 1e-5
      , [&](int iter, double cost, double size, std::span<double> x) {
            print_log(iter, cost, size, x);

            auto change_batch = 0 != iter && 0 == iter % restart_period;

            if (change_batch) {
                batch_count = (batch_count + 1) % N_batches;
                batch_start = N * batch_count;
                batch_end   = batch_start + N;
            }

            return change_batch;
        }
      );

    // Run on the full dataset
    batch_start = 0;
    batch_end   = N * N_batches;
    std::tie(status, res) = gsl_minimize(
        cost_func
      , res
      , step_sizes
      , 10'000
      , 1e-3
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
        fmt::print(full_pdf_file.get(), "u pdf\n");

        const auto step = (hi_data_thr - lo_data_thr) / steps;
        for (auto i = 0; steps >= i; ++i) {
            auto u = (lo_data_thr + i * step);
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
            fmt::print(pdf_file.get(), "u pdf\n");
            for (auto i = 0; steps >= i; ++i) {
                auto u = (lo_data_thr + i * step);
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
                fmt::print(pdf_file.get(), "u pdf\n");
                for (auto k = 0; steps >= k; ++k) {
                    auto u = (lo_data_thr + k * step);
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
