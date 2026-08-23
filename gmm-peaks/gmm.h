/*
 * Copyright (c) 2024, Slava Andrejev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * --- Optional exception to the license ---
 *
 * As an exception, if, as a result of your compiling your source code, portions
 * of this Software are embedded into a machine-executable object form of such
 * source code, you may redistribute such embedded portions in such object form
 * without including the above copyright and permission notices.
 */
#pragma once

#include <array>
#include <concepts>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename Matrix>
concept EigenMatrix = std::derived_from<
    std::remove_cv_t<Matrix>
  , Eigen::MatrixBase<
        std::remove_reference_t<
            decltype(std::declval<std::remove_cv_t<Matrix>>().derived())
          >
      >
  >;

template <typename Matrix>
concept EigenVector = EigenMatrix<Matrix> && (1 == Matrix::ColsAtCompileTime);

template <EigenMatrix samples_t, EigenMatrix centers_t, EigenVector clusters_t>
auto gmm(
    const samples_t &x
  , const centers_t &init_centers
  , const clusters_t &init_clusters
  , int attempts
  , auto progress
  )
{
    using namespace Eigen;
    namespace mc = boost::math::constants;

    static auto constexpr dims = (Index)samples_t::RowsAtCompileTime;

    const auto num_clusters = init_centers.cols();
    const auto num_samples  = x.cols();

    using RealType   = centers_t::value_type;
    using SampleType = samples_t::value_type;
    using cov_t      = Matrix<RealType, dims, dims>;
    using mean_t     = Matrix<RealType, dims, 1>;

    static const auto pitod = std::pow(mc::one_div_two_pi<RealType>(), 0.5 * dims);

    auto 𝜋        = VectorX<RealType>{num_clusters};
    auto 𝜎        = std::vector<cov_t>(num_clusters);
    auto inv_𝜎    = std::vector<cov_t>(num_clusters);
    auto dets     = VectorX<RealType>{num_clusters};
    auto 𝜇        = centers_t{init_centers};
    auto clusters = clusters_t{init_clusters};

    auto rd          = std::random_device{};
    auto gen         = std::mt19937{rd()};
    auto mins        = x.rowwise().minCoeff().eval();
    auto maxs        = x.rowwise().maxCoeff().eval();
    auto bbox_span   = (maxs - mins).eval();
    auto coords_rand = std::array<std::uniform_real_distribution<RealType>, dims>{};
    for (auto i = 0; dims > i; ++i) {
        coords_rand[i] = std::uniform_real_distribution<RealType>{
            -static_cast<RealType>(bbox_span(i)) / 4
          , static_cast<RealType>(bbox_span(i)) / 4
          };
    }

    auto rescue_𝜎 = [&](Index k) {
        𝜎[k].setZero();
        // set deviations to the half of the bounding box
        𝜎[k].diagonal() = square(bbox_span.template cast<RealType>().array() / 2.0).matrix();
        dets(k) = std::sqrt(std::abs(𝜎[k].determinant()));
    };

    𝜋.setConstant(1.0 / num_clusters);
    for (auto &&m : 𝜎) { m.setZero(); }
    auto cl_counts = std::vector<int>(num_clusters);
    for (auto i = Index{}; num_samples > i; ++i) {
        auto c = init_clusters(i);
        𝜎[c].diagonal() += square((x.col(i).template cast<RealType>() - 𝜇.col(c)).array()).matrix();
        ++cl_counts[c];
    }
    for (auto k = Index{}; num_clusters > k; ++k) {
        if (0 != cl_counts[k]) {
            𝜎[k] /= cl_counts[k];
            dets(k) = std::sqrt(std::abs(𝜎[k].determinant()));
        } else {
            rescue_𝜎(k);
        }
        inv_𝜎[k] = 𝜎[k].colPivHouseholderQr().inverse();
    }
    progress(-1, -1, 0, 𝜎, 𝜇, clusters);

    // Fused E+M pass: one sweep over the samples per iteration, no
    // responsibility matrix.  Accumulates, per cluster, the raw weighted
    // moments
    //     w  = Σᵢ 𝛾ᵢ,   s1 = Σᵢ 𝛾ᵢ xᵢ,   s2 = Σᵢ 𝛾ᵢ xᵢ xᵢᵀ
    // and the NLL; the M-step is then
    //     𝜋 = w/N,   𝜇 = s1/w,   𝜎 = s2/w − 𝜇𝜇ᵀ.
    auto fused_pass = [&](VectorX<RealType> &w,
                          std::vector<mean_t> &s1,
                          std::vector<cov_t> &s2) -> RealType {
        w.setZero();
        for (auto &&m : s1) { m.setZero(); }
        for (auto &&m : s2) { m.setZero(); }
        auto nll = RealType{};
        #pragma omp parallel
        {
            auto w_loc   = VectorX<RealType>{VectorX<RealType>::Zero(num_clusters)};
            auto s1_loc  = std::vector<mean_t>(num_clusters, mean_t::Zero());
            auto s2_loc  = std::vector<cov_t>(num_clusters, cov_t::Zero());
            auto dens    = VectorX<RealType>{num_clusters};
            auto nll_loc = RealType{};
            #pragma omp for nowait
            for (auto i = Index{}; num_samples > i; ++i) {
                const auto xi = x.col(i).template cast<RealType>().eval();
                for (auto k = Index{}; num_clusters > k; ++k) {
                    const auto arg     = (xi - 𝜇.col(k)).eval();
                    const auto exp_arg = RealType((-0.5 * arg.transpose() * inv_𝜎[k] * arg)(0, 0));
                    dens(k) = 𝜋(k) * pitod * std::exp(exp_arg) / dets(k);
                }
                const auto sum = dens.sum();
                if (0 < sum) {
                    nll_loc -= std::log(sum);
                    dens    /= sum;
                } else {
                    // All densities underflowed; spread the sample uniformly
                    // and charge it the smallest representable likelihood.
                    nll_loc -= std::log(std::numeric_limits<RealType>::min());
                    dens.setConstant(1.0 / num_clusters);
                }
                for (auto k = Index{}; num_clusters > k; ++k) {
                    const auto 𝛾 = dens(k);
                    w_loc(k)  += 𝛾;
                    s1_loc[k] += 𝛾 * xi;
                    s2_loc[k] += 𝛾 * xi * xi.transpose();
                }
            }
            #pragma omp critical
            {
                nll += nll_loc;
                w   += w_loc;
                for (auto k = Index{}; num_clusters > k; ++k) {
                    s1[k] += s1_loc[k];
                    s2[k] += s2_loc[k];
                }
            }
        }
        return nll;
    };

    // Hard assignment, needed only once per attempt (after convergence).
    auto assign_clusters = [&]() {
        auto dens = VectorX<RealType>{num_clusters};
        #pragma omp parallel for private(dens)
        for (auto i = Index{}; num_samples > i; ++i) {
            dens.resize(num_clusters);
            for (auto k = Index{}; num_clusters > k; ++k) {
                const auto arg     = (x.col(i).template cast<RealType>() - 𝜇.col(k)).eval();
                const auto exp_arg = RealType((-0.5 * arg.transpose() * inv_𝜎[k] * arg)(0, 0));
                dens(k) = 𝜋(k) * pitod * std::exp(exp_arg) / dets(k);
            }
            auto cluster = Index{};
            dens.maxCoeff(&cluster);
            clusters(i) = cluster;
        }
    };

    auto const n_iters = 4000;
    auto min_log_ℒ     = std::numeric_limits<RealType>::lowest();
    auto min_𝜇         = centers_t{};
    auto min_𝜎         = std::vector<cov_t>(num_clusters);
    auto min_𝜋         = VectorX<RealType>{num_clusters};
    auto min_clusters  = clusters_t{};

    auto w  = VectorX<RealType>{num_clusters};
    auto s1 = std::vector<mean_t>(num_clusters);
    auto s2 = std::vector<cov_t>(num_clusters);

    for (auto attempt = 0; attempts > attempt; ++attempt) {
        auto log_ℒ      = 0.0;
        auto prev_log_ℒ = 0.0;
        for (auto iter = 0; n_iters > iter; ++iter) {
            log_ℒ = fused_pass(w, s1, s2);
            progress(attempt, iter, log_ℒ, 𝜎, 𝜇, clusters);

            // Maximization step from the raw moments
            for (auto k = Index{}; num_clusters > k; ++k) {
                if (𝜋.minCoeff() * 1e-9 < w(k) && 0 < w(k)) {
                    𝜇.col(k) = s1[k] / w(k);
                    𝜎[k]     = s2[k] / w(k) - 𝜇.col(k) * 𝜇.col(k).transpose();
                    dets(k)  = std::sqrt(std::abs(𝜎[k].determinant()));
                } else {
                    // Cluster lost all its samples; re-seed it wide.
                    rescue_𝜎(k);
                }
                𝜋(k) = w(k) / num_samples;
            }
            𝜋 /= 𝜋.sum();
            // Check for degeneration
            for (auto k = Index{}; num_clusters > k; ++k) {
                if (1e-12 > dets(k)) {
                    rescue_𝜎(k);
                    // redistribute weights
                    auto borrowed_weight = 1.0 - 1.0 / num_clusters;
                    𝜋 *= borrowed_weight;
                    𝜋(k) = 1.0 / num_clusters;
                    𝜋 /= 𝜋.sum();
                }
                inv_𝜎[k] = 𝜎[k].colPivHouseholderQr().inverse();
            }
            if (0 < iter && 1e-9 > std::abs(prev_log_ℒ - log_ℒ) / std::abs(log_ℒ)) {
                break;
            }
            prev_log_ℒ = log_ℒ;
        }
        if (0 == attempt || log_ℒ < min_log_ℒ) {
            assign_clusters();
            min_log_ℒ    = log_ℒ;
            min_𝜇        = 𝜇;
            min_𝜎        = 𝜎;
            min_𝜋        = 𝜋;
            min_clusters = clusters;
        }
        // Random perturbation of centers
        for (auto k = Index{}; num_clusters > k; ++k) {
            for (auto i = 0; dims > i; ++i) {
                auto 𝛿 = coords_rand[i](gen);
                auto new_𝜇 = 𝜇(i, k) + 𝛿;
                // Clip centers' coordinates values if sample type is integral.
                if constexpr (!std::is_same_v<RealType, SampleType> && std::is_integral_v<SampleType>) {
                    if (new_𝜇 < std::numeric_limits<SampleType>::min()) {
                        new_𝜇 = std::numeric_limits<SampleType>::min();
                    }
                    if (new_𝜇 > std::numeric_limits<SampleType>::max()) {
                        new_𝜇 = std::numeric_limits<SampleType>::max();
                    }
                }
                𝜇(i, k) = new_𝜇;
            }
            auto es = SelfAdjointEigenSolver<cov_t>(𝜎[k]);
            const auto &evals = es.eigenvalues();
            const auto max_𝜆 = evals.maxCoeff();
            𝜎[k].setZero();
            𝜎[k].diagonal().setConstant(max_𝜆);
            dets(k)  = std::sqrt(std::abs(𝜎[k].determinant()));
            inv_𝜎[k] = 𝜎[k].colPivHouseholderQr().inverse();
        }
    }
    return std::tuple{min_log_ℒ, min_𝜎, min_𝜇, min_clusters};
}
