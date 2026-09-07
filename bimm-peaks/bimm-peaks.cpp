#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/parser/parser.hpp>
#include <boost/program_options.hpp>

#include <cmaes.h>

#include <fmt/printf.h>

#include <range/v3/algorithm/sample.hpp>
#include <range/v3/algorithm/stable_sort.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/single.hpp>
#include <range/v3/view/transform.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_erf.h>

#include <io/biniarchive.h>

#include <cuda-context.h>
#include <field.h>
#include <timeop.h>

#include "grad_magn.h"

namespace fs = std::filesystem;
namespace mc = boost::math::double_constants;
namespace po = boost::program_options;
namespace rs = ranges;
namespace rv = rs::views;

using namespace libcmaes;

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

template <typename F>
auto gsl_integration(double a, double b, double epsabs, double epsrel, F func)
{
    struct Adapt {
        Adapt(const F& func) : func(func) {}
        static double invoke(double x, void *params) {
            return static_cast<Adapt*>(params)->func(x);
        }
        const F &func;
    };
    auto constexpr limit = 100'000u;
    using gsl_worspace = std::unique_ptr<
        gsl_integration_workspace
      , decltype(&gsl_integration_workspace_free)
      >;
    static thread_local auto ws = gsl_worspace{
        gsl_integration_workspace_alloc(limit)
      , &gsl_integration_workspace_free
      };
    auto adapt = Adapt{func};
    auto f = gsl_function{&Adapt::invoke, &adapt};
    auto result = 0.0, abserr = 0.0;
    const auto status = gsl_integration_qags(&f, a, b, epsabs, epsrel, limit, ws.get(), &result, &abserr);
    return std::tuple{status, result, abserr};
}

struct layout_strict_upper {
    template <class Extents>
    struct mapping {
        using layout_type  = layout_strict_upper;
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using size_type    = typename Extents::size_type;
        using rank_type    = typename Extents::rank_type;

        extents_type ext_{};

        mapping() = default;

        mapping(const layout_strict_upper::mapping<extents_type> &other)
          : ext_{other.ext_}
        {}

        mapping(const extents_type &e)
          : ext_(e)
        {}

        constexpr auto operator()(index_type i, index_type j) const -> index_type {
            return (2 * ext_.extent(0) - 1 - i) * i / 2 + j - i - 1;
        }

        constexpr index_type required_span_size() const {
            auto n = ext_.extent(0);
            return n * (n - 1) / 2;
        }

        constexpr const extents_type& extents() const { return ext_; }

        static constexpr bool is_always_unique()     { return true; }
        static constexpr bool is_always_exhaustive() { return true; }
        static constexpr bool is_always_strided()    { return false; }
        static constexpr bool is_unique()            { return true; }
        static constexpr bool is_exhaustive()        { return true; }
        static constexpr bool is_strided()           { return false; }

        friend constexpr bool
        operator==(const mapping &, const mapping &) = default;
    };
};
template <typename T>
using strict_upper_span = std::mdspan<T, std::dextents<uint32_t, 2>, layout_strict_upper>;


int main(int argc, char *argv[]) try {
    gsl_set_error_handler_off();

    auto positional       = po::positional_options_description{};
    auto cmd_line_options = po::options_description{};
    auto vm               = po::variables_map{};

    auto in_file_name             = std::string{};
    auto pdf_name                 = std::string{};
    auto cmaes_progress_file_name = std::string{};
    auto init_peaks               = LabelsArray{};
    auto u_start                  = 0.0;
    auto u_end                    = 0.0;

    cmd_line_options.add_options()
        ("input", po::value<std::string>(&in_file_name)->required(), "Input file")
        ("peaks", po::value(&init_peaks)->required(), "Initial peaks")
        ("pdf-name", po::value(&pdf_name), "Base name to write resulting distributions")
        ("cmaes-pr", po::value(&cmaes_progress_file_name), "Base name to write resulting distributions")
        ("u-start", po::value(&u_start), "Start of the PDF output range")
        ("u-end", po::value(&u_end), "End of the PDF output range")
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

    auto lo = img.lo();
    auto hi = img.hi();

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
                if (0 < v_view[k, row, col]) {
                    eligible.push_back(count);
                }
                ++count;
            }
        }
    }
    t.stop("Linearize time");

    static auto constexpr N         = 10000;
    static auto constexpr N_batches = 20;

    auto rd        = std::random_device{};
    auto gen       = std::mt19937{rd()};
    auto indices   = std::vector<size_t>(N * N_batches);
    auto samples_u = std::vector<double>(N * N_batches);
    auto samples_v = std::vector<double>(N * N_batches);

    rs::sample(eligible, indices.begin(), N * N_batches, gen);
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

    auto p_i = [](double u, double v, double I_i, double log𝜎_n, double 𝜌) {
        static auto constexpr two_root_two_div_pi = 0.9003163161571060695551992;
        auto 𝜎_n2 = std::exp(2 * log𝜎_n);
        auto v2 = sqr(v);
        return two_root_two_div_pi * v2 / (sqr(𝜎_n2) * std::pow(1 - 𝜌, 1.5)) *
               std::exp(-0.5 * sqr(u - I_i) / 𝜎_n2 - v2 / 𝜎_n2 / (1.0 - 𝜌));
    };

    auto p_ij = [](
        double u
      , double v
      , double I_i
      , double I_j
      , double log𝜎_n
      , double log𝜎_b
      , double logds
      , double 𝜌
      )
    {
        static auto constexpr one_two_root_two_pi = 0.1125395395196382586943999;
        static auto constexpr one_root_two_pi     = 0.3989422804014326779399461;

        auto 𝜎_n2 = std::exp(2 * log𝜎_n);
        auto 𝜎_n  = std::exp(log𝜎_n);
        auto 𝜎_b  = std::exp(log𝜎_b);
        auto ds   = std::exp(logds);
        auto a    = 𝜎_n2 * (1 - 𝜌);

        auto [status, integral, abserr] =
            gsl_integration(-ds, ds, 1e-10, 1e-6, [&](double t) {
                auto I = I_i + 0.5 * (I_j - I_i) *
                    (1 + gsl_sf_erf(t / mc::root_two));

                if (30 > std::abs(t) && 1e-9 < std::abs(I_j - I_i)) {
                    auto G = one_root_two_pi * std::abs(I_j - I_i) / 𝜎_b
                      * std::exp(-sqr(t) / 2);

                    return std::exp(
                        -0.5 * sqr(u - I) / 𝜎_n2 - sqr(v - G) / a
                      )
                      * (-std::expm1(-4 * v * G / a)) / G;
                } else {
                    return std::exp(
                        -0.5 * sqr(u - I) / 𝜎_n2 - sqr(v) / a
                      )
                      * 4 * v / a;
                }
            });

        if (GSL_SUCCESS != status) {
            fmt::print(stderr,
                "GSL QAGS failed: {} (status = {})\n"
                "integral = {}, abserr = {}\n"
                "u = {}, v = {}\n"
                "I_i ={}, I_j = {}\n"
                "𝜎_n = {}, 𝜎_b = {}\n"
                "ds = {}, 𝜌 = {}\n"
                "a = {}\n"
              , gsl_strerror(status), status
              , integral, abserr
              , u, v
              , I_i, I_j
              , 𝜎_n, 𝜎_b
              , ds, 𝜌
              , a);
        }

        return one_two_root_two_pi * v / (ds * 𝜎_n * std::sqrt(a)) * integral;
    };

    auto p = [&](
        double u
      , double v
      , std::span<const double> w
      , std::span<const double> I
      , double log𝜎_n
      , double log𝜎_b
      , double logds
      , double 𝜌
      )
    {
        auto res = 0.0;

        for (auto i = 0; I.size() > i; ++i) {
            res += w[i] * p_i(u, v, I[i], log𝜎_n, 𝜌);
        }
        auto idx = I.size();
        for (auto i = 0; I.size() - 1 > i; ++i) {
            for (auto j = i + 1; I.size() > j; ++j) {
                res += w[idx] * p_ij(u, v, I[i], I[j], log𝜎_n, log𝜎_b, logds, 𝜌);
                ++idx;
            }
        }

        return res;
    };

    using GP         = GenoPheno<pwqBoundStrategy, linScalingStrategy>;
    using Parameters = CMAParameters<GP>;

    auto x0 = std::vector<double>(n_params, 0);
    auto ub = std::vector<double>(n_params, 0);
    auto lb = std::vector<double>(n_params, 0);

    for (auto i = 0; n_weights > i; ++i) {
        lb[i] = std::log(1e-2);
        ub[i] = std::log(1e2);
    }
    for (auto i = 0; n_labels > i; ++i) {
        x0[I_start_idx + i] = (init_peaks.data[i] - lo) / (hi - lo);
        lb[I_start_idx + i] = 0.0;
        ub[I_start_idx + i] = 1.0;
    }
    lb[𝜎n_idx] = std::log(0.001); ub[𝜎n_idx] = std::log(2.0);
    lb[𝜎b_idx] = std::log(0.1);   ub[𝜎b_idx] = std::log(10.0);
    lb[𝜌_idx]  =   0;                ub[𝜌_idx]  = 0.9;
    lb[ds_idx] = std::log(0.5);   ub[ds_idx] = std::log(3);

    auto 𝜎 = 0.5;

    auto        gp = GP{&lb[0], &ub[0], n_params};
    auto cmaparams = Parameters(n_params, &x0[0], 𝜎, -1, 0, gp);

    if (0 != vm.count("cmaes-pr")) {
        cmaparams.set_fplot(cmaes_progress_file_name);
    }
    cmaparams.set_mt_feval(true);
    cmaparams.set_ftolerance(1e-18);
    cmaparams.set_max_iter(2000);
    cmaparams.set_uh(true);
    cmaparams.set_stopping_criteria(STAGNATION, false);
    // cmaparams.set_stopping_criteria(AUTOMAXITER, false);
    // cmaparams.set_algo(aBIPOP_CMAES);
    // cmaparams.set_restarts(9);

    auto get_weights = [](std::span<const double> a) {
        auto all_w = rv::concat(rv::single(0.0), a);
        auto exp_vew = all_w | rv::transform([](auto x) {
            return std::exp(x);
        });
        auto sum = rs::accumulate(exp_vew, 0.0);

        auto w = all_w | rv::transform([&](auto x) {
            return std::exp(x) / sum;
        }) | rs::to_vector;

        return w;
    };
    auto get_labels = [](std::span<const double> a) {
#if 1
        static auto constexpr 𝜀 = 0.1;

        const auto n = a.size();
        auto labels = std::vector<double>(n);
        if (n == 0) return labels;

        const auto upper = 1.0 - 𝜀 * double(n - 1);

        auto order = rv::iota(size_t{}, n) | rs::to_vector;
        rs::stable_sort(order, [&](auto i, auto j) { return a[i] < a[j]; });

        struct Block {
            double sum;
            std::size_t count;
        };

        auto blocks = std::vector<Block>{};
        blocks.reserve(n);

        for (auto i = size_t{}; n > i; ++i) {
            blocks.push_back({a[order[i]] - 𝜀 * double(i), 1});

            while (blocks.size() > 1) {
                auto &left = blocks[blocks.size() - 2];
                const auto &right = blocks.back();

                if (left.sum / double(left.count) <= right.sum / double(right.count)) {
                    break;
                }

                left.sum += right.sum;
                left.count += right.count;
                blocks.pop_back();
            }
        }

        auto i = std::size_t{};
        for (const auto &block : blocks) {
            const auto value = std::clamp(
                block.sum / double(block.count), 0.0, upper);

            for (std::size_t j = 0; j < block.count; ++j, ++i) {
                labels[order[i]] = value + 𝜀 * double(i);
            }
        }

        return labels;
#else
        return std::vector<double>(a.begin(), a.end());
#endif
    };

    auto batch_count = 0;
    FitFunc cost_fun = [&](const double *a, const int) {
        auto res = 0.0;

        auto w  = get_weights({a, n_weights});
        auto Is = get_labels({&a[I_start_idx], n_labels});

        for (auto i = 0; N > i; ++i) {
            res += -std::log(p(
                samples_u[i + N * batch_count]
              , samples_v[i + N * batch_count]
              , w
              , Is
              , a[𝜎n_idx]
              , a[𝜎b_idx]
              , a[ds_idx]
              , a[𝜌_idx]
              ));
        }

        return res / N;
    };

    ProgressFunc<Parameters, CMASolutions> pf = [&](const Parameters &, const CMASolutions &cmasols) {
        fmt::print("{:6} {:14.10f}\r", cmasols.niter(), cmasols.get_best_seen_candidate().get_fvalue());
        fflush(stdout);

        if (0 == cmasols.niter() % N_batches) {
            batch_count = 0;
        } else {
            ++batch_count;
        }

        return 0;
    };

    auto cmasols = cmaes<GP>(cost_fun, cmaparams, pf);
    fmt::print("\n");
    fmt::print("stop: {} ({})\n", cmasols.status_msg(), cmasols.run_status());

    // auto x = gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());
    auto x = gp.pheno(cmasols.xmean());

    fmt::print(
        "𝜎b = {}, 𝜎n = {}, ds = {}, 𝜌 = {}\n"
      , std::exp(x[𝜎b_idx])
      , std::exp(x[𝜎n_idx]) * (hi - lo)
      , std::exp(x[ds_idx])
      , x[𝜌_idx]);

    auto w  = get_weights({&x[0], n_weights});
    auto Is = get_labels({&x[I_start_idx], n_labels});
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

    if (0 != vm.count("pdf-name") && 0 != vm.count("u-start") && 0 != vm.count("u-end")) {
        static auto constexpr steps = 512;

        auto marginal_p_i = [](double u, double I_i, double 𝜎_n) {
            static auto constexpr one_root_two_pi = 0.3989422804014326779399461;

            return one_root_two_pi / 𝜎_n * std::exp(
                -sqr(u - I_i) / (2 * sqr(𝜎_n))
            );
        };

        auto marginal_p_ij = [](
            double u
          , double I_i
          , double I_j
          , double 𝜎_n
          , double ds
          )
        {
            static auto constexpr one_root_two_pi = 0.3989422804014326779399461;

            auto [status, integral, abserr] =
                gsl_integration(-ds, ds, 1e-10, 1e-6, [&](double t) {
                    auto I = I_i + 0.5 * (I_j - I_i) *
                        (1 + gsl_sf_erf(t / mc::root_two));

                    return std::exp(
                        -sqr(u - I) / (2 * sqr(𝜎_n))
                    );
                });

            if (GSL_SUCCESS != status) {
                fmt::print(stderr,
                    "GSL QAGS failed: {} (status = {})\n"
                    "integral = {}, abserr = {}\n"
                    "u = {}\n"
                    "I_i ={}, I_j = {}\n"
                    "𝜎_n = {}\n"
                    "ds = {}\n"
                  , gsl_strerror(status), status
                  , integral, abserr
                  , u
                  , I_i, I_j
                  , 𝜎_n
                  , ds);
            }

            return one_root_two_pi / (2 * ds * 𝜎_n) * integral;
        };

        auto marginal_p = [&](
            double u
          , std::span<const double> w
          , std::span<const double> I
          , double 𝜎_n
          , double ds
          )
        {
            auto res = 0.0;

            for (auto i = 0; I.size() > i; ++i) {
                res += w[i] * marginal_p_i(u, I[i], 𝜎_n);
            }
            auto idx = I.size();
            for (auto i = 0; I.size() - 1 > i; ++i) {
                for (auto j = i + 1; I.size() > j; ++j) {
                    res += w[idx] * marginal_p_ij(u, I[i], I[j], 𝜎_n, ds);
                    ++idx;
                }
            }

            return res;
        };

        auto full_pdf_file_name = pdf_name + "-full.txt";
        auto full_pdf_file = std::unique_ptr<FILE, decltype(&fclose)>{
            fopen(full_pdf_file_name.c_str(), "wt")
          , &fclose
          };
        fmt::print(full_pdf_file.get(), "u pdf\n");

        const auto step = (u_end - u_start) / steps;
        for (auto i = 0; steps >= i; ++i) {
            auto u = (u_start + i * step);
            auto normalized_u = (u - lo) / (hi - lo);
            auto pdf = marginal_p(
                normalized_u
              , w
              , Is
              , std::exp(x[𝜎n_idx])
              , std::exp(x[ds_idx])
              );
            fmt::print(full_pdf_file.get(), "{} {}\n", u, pdf / (hi - lo));
        }
        for (auto c = 0; Is.size() > c; ++c) {
            auto pdf_file_name = fmt::format("{}-{:02}.txt", pdf_name, c);
            auto pdf_file = std::unique_ptr<FILE, decltype(&fclose)>{
                fopen(pdf_file_name.c_str(), "wt")
              , &fclose
              };
            fmt::print(pdf_file.get(), "u pdf\n");
            for (auto i = 0; steps >= i; ++i) {
                auto u = (u_start + i * step);
                auto normalized_u = (u - lo) / (hi - lo);
                auto pdf = w[c] * marginal_p_i(
                    normalized_u
                  , Is[c]
                  , std::exp(x[𝜎n_idx])
                  );
                fmt::print(pdf_file.get(), "{} {}\n", u, pdf / (hi - lo));
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
                    auto u = (u_start + k * step);
                    auto normalized_u = (u - lo) / (hi - lo);
                    auto pdf = w[idx] * marginal_p_ij(
                        normalized_u
                      , Is[i]
                      , Is[j]
                      , std::exp(x[𝜎n_idx])
                      , std::exp(x[ds_idx])
                      );
                    fmt::print(pdf_file.get(), "{} {}\n", u, pdf / (hi - lo));
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
