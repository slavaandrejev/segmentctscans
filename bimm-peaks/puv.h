#pragma once

#include <cmath>
#include <memory>
#include <span>
#include <tuple>

#include <boost/math/constants/constants.hpp>

#include <fmt/printf.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_erf.h>

#include <sqr.h>

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

inline
auto full_p_i(double u, double v, double I_i, double log𝜎_n, double 𝜌) {
    static auto constexpr two_root_two_div_pi = 0.9003163161571060695551992;
    auto 𝜎_n2 = std::exp(2 * log𝜎_n);
    auto v2 = sqr(v);
    return two_root_two_div_pi * v2 / (sqr(𝜎_n2) * std::pow(1 - 𝜌, 1.5)) *
           std::exp(-0.5 * sqr(u - I_i) / 𝜎_n2 - v2 / 𝜎_n2 / (1.0 - 𝜌));
}

inline
auto full_p_ij(
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
    namespace mc = boost::math::double_constants;

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
}

inline
auto full_p(
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
        res += w[i] * full_p_i(u, v, I[i], log𝜎_n, 𝜌);
    }
    auto idx = I.size();
    for (auto i = 0; I.size() - 1 > i; ++i) {
        for (auto j = i + 1; I.size() > j; ++j) {
            res += w[idx] * full_p_ij(u, v, I[i], I[j], log𝜎_n, log𝜎_b, logds, 𝜌);
            ++idx;
        }
    }

    return res;
};

inline
auto marginal_p_i(double u, double I_i, double 𝜎_n) {
    static auto constexpr one_root_two_pi = 0.3989422804014326779399461;

    return one_root_two_pi / 𝜎_n * std::exp(
        -sqr(u - I_i) / (2 * sqr(𝜎_n))
    );
}

inline
auto marginal_p_ij(
    double u
  , double I_i
  , double I_j
  , double 𝜎_n
  , double ds
  )
{
    namespace mc = boost::math::double_constants;

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

inline
auto marginal_p(
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
}

inline
auto M_i(double lo, double hi, double I_i, double 𝜎_n) {
    namespace mc = boost::math::double_constants;
    return 0.5 * (
        gsl_sf_erf((hi - I_i) / (mc::root_two * 𝜎_n)) -
        gsl_sf_erf((lo - I_i) / (mc::root_two * 𝜎_n))
    );
}

inline
auto M_ij(double lo, double hi, double I_i, double I_j, double 𝜎_n, double ds) {
    namespace mc = boost::math::double_constants;

    auto [status, integral, abserr] =
        gsl_integration(-ds, ds, 1e-10, 1e-6, [&](double t) {
            auto I = I_i + 0.5 * (I_j - I_i) *
                (1 + gsl_sf_erf(t / mc::root_two));

            return
                gsl_sf_erf((hi - I) / (mc::root_two * 𝜎_n)) -
                gsl_sf_erf((lo - I) / (mc::root_two * 𝜎_n));
        });

    if (GSL_SUCCESS != status) {
        fmt::print(stderr,
            "GSL QAGS failed: {} (status = {})\n"
            "integral = {}, abserr = {}\n"
            "I_i ={}, I_j = {}\n"
            "𝜎_n = {}\n"
            "ds = {}\n"
          , gsl_strerror(status), status
          , integral, abserr
          , I_i, I_j
          , 𝜎_n
          , ds);
    }

    return integral / (4 * ds);
}
