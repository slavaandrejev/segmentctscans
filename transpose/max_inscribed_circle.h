#pragma once

#include <tuple>
#include <vector>

#include <clarabel.hpp>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

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
