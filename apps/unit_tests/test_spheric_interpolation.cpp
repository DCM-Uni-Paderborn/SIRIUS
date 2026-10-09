/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "function3d/spheric_function_interpolator.hpp"

using namespace sirius;

template <typename T>
int
test_spline_transpose(int n__)
{
    std::vector<double> x(n__);
    for (int i = 0; i < n__; i++) {
        x[i] = 0.02 + 3 * std::pow(double(i) / (n__ - 1), 1.7);
    }
    Radial_grid_ext<double> grid(n__, x.data());
    Spline<T> spline(grid);
    mdarray<T, 2> derivative({n__, 4});
    for (int i = 0; i < n__; i++) {
        spline(i) = std::sin(i + 0.3);
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            spline(i) += T(0, std::cos(i + 0.2));
        }
        for (int j = 0; j < 4; j++) {
            derivative(i, j) = std::cos(3 * i + 2 * j + 0.7);
        }
    }
    spline.interpolate();
    auto result = spline.interpolation_adjoint(derivative);
    T forward{0}, reverse{0};
    for (int i = 0; i < n__; i++) {
        reverse += result[i] * spline(i);
        for (int j = 0; j < 4; j++) {
            forward += derivative(i, j) * spline.coeffs()(i, j);
        }
    }
    double error = std::abs(forward - reverse) / std::max(1.0, std::abs(forward));
    std::cout << "spline transpose n=" << n__ << ": " << error << '\n';
    return !std::isfinite(error) || error > 2e-12;
}

int
test_sampling_geometry(Spheric_function_interpolator const& sampler, Spheric_function_interpolator const& dsampler,
                       std::vector<r3::vector<double>> const& points,
                       std::vector<std::array<double, 4>> const& covectors, Flm const& field_direction,
                       bool include_radial = false)
{
    std::vector<r3::vector<double>> directions;
    double lhs{0}, rhs{0}, error{0};
    for (size_t p = 0; p < points.size(); p++) {
        directions.emplace_back(0.1 * r3::vector<double>(points[p][1], -points[p][2], points[p][0]));
        if (include_radial) {
            directions.back() += 0.1 * points[p];
        }
        auto response = dsampler.position_response(points[p], directions.back());
        for (int k = 0; k < 4; k++) {
            lhs += covectors[p][k] * response[k];
        }
    }
    auto reverse = sampler.geometry_adjoint_response(points, covectors, directions);
    for (size_t i = 0; i < reverse.size(); i++) {
        rhs += reverse[i] * field_direction[i];
    }
    error             = std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs));
    bool dual_failure = !std::isfinite(error) || error > 2e-10;
    // Gradient sampling is C1 at the inner join; its transpose-direction FD is only first-order there.
    for (double h : {5e-6, 2.5e-6}) {
        auto plus = points, minus = points;
        for (size_t p = 0; p < points.size(); p++) {
            plus[p] += h * directions[p];
            minus[p] -= h * directions[p];
            auto dp = sampler.position_response(points[p], directions[p]);
            auto fp = sampler.value_gradient(plus[p]), fm = sampler.value_gradient(minus[p]);
            for (int k = 0; k < 4; k++) {
                double e = std::abs(dp[k] - (fp[k] - fm[k]) / (2 * h)) / std::max(1.0, std::abs(dp[k]));
                error    = std::max(error, e);
            }
        }
        auto rp = sampler.adjoint(plus, covectors), rm = sampler.adjoint(minus, covectors);
        double peak{0}, fd_error{0};
        for (size_t i = 0; i < reverse.size(); i++) {
            peak     = std::max(peak, std::abs(reverse[i]));
            fd_error = std::max(fd_error, std::abs(reverse[i] - (rp[i] - rm[i]) / (2 * h)));
        }
        error = std::max(error, fd_error / std::max(1.0, peak));
    }
    std::cout << "sampling geometry forward/transpose response: " << error << '\n';
    return dual_failure || !std::isfinite(error) || error > 2e-6;
}

int
test_spheric_interpolation()
{
    std::vector<double> x{0.02, 0.09, 0.2, 0.45, 0.75, 1.1, 1.8, 2.4, 3.0};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n = sf::lmmax(3);
    Flm field(n, grid), direction(n, grid);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            field(lm, ir)     = std::sin(0.7 * lm + grid[ir]) / (lm + 1);
            direction(lm, ir) = std::cos(0.2 * lm + 2 * grid[ir]) / (lm + 1);
        }
    }
    Spheric_function_interpolator sampler(field), dsampler(direction);
    std::vector<r3::vector<double>> points{{0.4, 0.3, 0.2},  {0, 0, 0.75},      {0, 0, -1.1}, {0.45, 0, 0},
                                           {0.3, -0.5, 0.7}, {-0.2, 0.4, -0.8}, {0.02, 0, 0}, {0, 3.0, 0}};
    int failures{0};
    Test_check check(failures);
    Flm polynomial(n, grid);
    polynomial.zero();
    double pnorm = std::sqrt(3 / fourpi);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        double r                      = grid[ir];
        polynomial(0, ir)             = (1 + 0.7 * r * r) / y00;
        polynomial(sf::lm(1, 1), ir)  = -0.2 * r / pnorm;
        polynomial(sf::lm(1, -1), ir) = 0.3 * r / pnorm;
        polynomial(sf::lm(1, 0), ir)  = 0.4 * r / pnorm;
        polynomial(sf::lm(2, 1), ir)  = -0.17 * r * r / std::sqrt(15 / fourpi);
    }
    Spheric_function_interpolator polynomial_sample(polynomial);
    double exact_error{0};
    for (auto const& p : points) {
        auto actual     = polynomial_sample.value_gradient(p);
        double expected = 1 + 0.7 * std::pow(p.length(), 2) + 0.2 * p[0] - 0.3 * p[1] + 0.4 * p[2] + 0.17 * p[0] * p[2];
        exact_error     = std::max(exact_error, std::abs(actual[0] - expected));
        exact_error     = std::max(exact_error, std::abs(polynomial_sample.value(p) - actual[0]));
        std::array<double, 4> covector{0.4, -0.2, 0.5, 0.7};
        auto position_derivative = polynomial_sample.position_adjoint(p, covector);
        for (int a = 0; a < 3; a++) {
            double d    = 1.4 * p[a] + std::array<double, 3>{0.2 + 0.17 * p[2], -0.3, 0.4 + 0.17 * p[0]}[a];
            exact_error = std::max(exact_error, std::abs(actual[a + 1] - d));
            double dp   = covector[0] * d + 1.4 * covector[a + 1] + (a == 1 ? 0 : 0.17 * covector[3 - a]);
            exact_error = std::max(exact_error, std::abs(position_derivative[a] - dp));
        }
    }
    check("Cartesian polynomial values/gradients/geometry adjoint including endpoints", exact_error, 2e-12);
    auto surface = polynomial_sample.value_gradient({0, grid.last(), 0});
    auto rounded = polynomial_sample.value_gradient(
            {0, std::nextafter(grid.last(), std::numeric_limits<double>::infinity()), 0});
    double surface_error{0};
    for (int k = 0; k < 4; k++) {
        surface_error = std::max(surface_error, std::abs(surface[k] - rounded[k]));
    }
    check("radial endpoint tolerates coordinate roundoff", surface_error, 2e-12);
    for (double step : {1e-4, 5e-5}) {
        double error{0};
        for (size_t i = 0; i + 2 < points.size(); i++) {
            auto result = sampler.value_gradient(points[i]);
            for (int a = 0; a < 3; a++) {
                auto plus = points[i], minus = points[i];
                plus[a] += step;
                minus[a] -= step;
                double fd = (sampler.value_gradient(plus)[0] - sampler.value_gradient(minus)[0]) / (2 * step);
                error     = std::max(error, std::abs(fd - result[a + 1]));
            }
        }
        check("Cartesian gradient FD including poles and radial knots", error, 2e-6);
    }
    std::vector<std::array<double, 4>> weights(points.size());
    double lhs{0}, rhs{0};
    for (size_t i = 0; i < points.size(); i++) {
        auto values = dsampler.value_gradient(points[i]);
        for (int a = 0; a < 4; a++) {
            weights[i][a] = std::sin(0.5 + 2 * i + a);
            lhs += weights[i][a] * values[a];
        }
    }
    auto back = sampler.adjoint(points, weights);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            rhs += direction(lm, ir) * back(lm, ir);
        }
    }
    check("combined value/gradient adjoint", std::abs(lhs - rhs), 2e-12);
    failures += test_sampling_geometry(sampler, dsampler, {points.begin(), points.end() - 2},
                                       {weights.begin(), weights.end() - 2}, direction);

    // A source atom and one of its periodic images both contribute to each target.
    std::vector<r3::vector<double>> displacements;
    for (int i = 0; i < 4; i++) {
        displacements.push_back(points[i]);
        displacements.push_back(points[i] - r3::vector<double>({0.7, 0.2, 0.1}));
    }
    auto evaluate = [&](Flm const& input, std::vector<std::array<double, 4>>* derivative) {
        Spheric_function_interpolator op(input);
        std::array<std::array<double, 4>, 4> values{};
        double sum{0}, energy{0};
        for (int i = 0; i < 4; i++) {
            auto first = op.value_gradient(displacements[2 * i]);
            auto image = op.value_gradient(displacements[2 * i + 1]);
            for (int a = 0; a < 4; a++) {
                values[i][a] = first[a] + image[a];
                energy += 0.5 * i * (a == 0 ? 1 : 0.2) * values[i][a] * values[i][a];
            }
            sum += (1 + 0.1 * i) * values[i][0];
        }
        energy += 0.13 * sum * sum;
        if (derivative) {
            derivative->resize(displacements.size());
            for (int i = 0; i < 4; i++) {
                for (int a = 0; a < 4; a++) {
                    double value = i * (a == 0 ? 1 : 0.2) * values[i][a];
                    if (a == 0) {
                        value += 0.26 * sum * (1 + 0.1 * i);
                    }
                    (*derivative)[2 * i][a] = (*derivative)[2 * i + 1][a] = value;
                }
            }
        }
        return energy;
    };
    std::vector<std::array<double, 4>> derivative;
    evaluate(field, &derivative);
    back = sampler.adjoint(displacements, derivative);
    double analytic{0};
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            analytic += direction(lm, ir) * back(lm, ir);
        }
    }
    for (double step : {1e-4, 5e-5}) {
        Flm plus(n, grid), minus(n, grid);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            for (int lm = 0; lm < n; lm++) {
                plus(lm, ir)  = field(lm, ir) + step * direction(lm, ir);
                minus(lm, ir) = field(lm, ir) - step * direction(lm, ir);
            }
        }
        double fd = (evaluate(plus, nullptr) - evaluate(minus, nullptr)) / (2 * step);
        check("nonlocal image-sum energy derivative", std::abs(analytic - fd), 2e-9);
    }
    auto empty = sampler.adjoint({}, {});
    check("empty sample set", std::abs(empty.checksum()), 0);
    for (auto p : {r3::vector<double>({0, 0, 0}), r3::vector<double>({0.01, 0, 0}), r3::vector<double>({4, 0, 0}),
                   r3::vector<double>({grid.last() + 1e-6, 0, 0}),
                   r3::vector<double>({std::numeric_limits<double>::quiet_NaN(), 0, 0})}) {
        check.reject("invalid sampling point", [&] { sampler.value_gradient(p); });
        check.reject("invalid value-only sampling point", [&] { sampler.value(p); });
    }
    return failures;
}

int
test_inner_extension(spheric_inner_extension_t mode)
{
    std::vector<double> x{0.02, 0.027, 0.04, 0.065, 0.1, 0.16, 0.3};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n  = sf::lmmax(3);
    auto l = sf::l_by_lm(3);
    Flm field(n, grid), direction(n, grid);
    auto power  = [&](int lm) { return mode == spheric_inner_extension_t::charge ? l[lm] : std::max(0, l[lm] - 1); };
    auto radial = [&](int lm, double r, int derivative) {
        int p = power(lm);
        double value{0};
        // Cubics are reproduced by the spline and also belong to the regular inner ansatz.
        for (int j = 0; j <= std::min(2, 3 - p); j++) {
            int exponent = p + j;
            if (exponent < derivative) {
                continue;
            }
            double c = (1 + 0.13 * j) / (lm + 1);
            for (int k = 0; k < derivative; k++) {
                c *= exponent - k;
            }
            value += c * std::pow(r, exponent - derivative);
        }
        return value;
    };
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            field(lm, ir)     = radial(lm, grid[ir], 0);
            direction(lm, ir) = std::cos(lm + 0.2 * ir) * std::pow(grid[ir], power(lm));
        }
    }
    Spheric_function_interpolator sampler(field, mode), dsampler(direction, mode);
    std::vector<r3::vector<double>> points;
    std::vector<std::array<double, 4>> covectors;
    double error{0}, lhs{0}, rhs{0};
    for (double fraction : {1e-6, 0.2, 0.5, 0.99999, 1.0, 1.00001, 1.2}) {
        double r                    = grid.first() * fraction;
        r3::vector<double> position = (r / std::sqrt(14.0)) * r3::vector<double>({1, -2, 3});
        points.push_back(position);
        auto angles = r3::spherical_coordinates(position);
        std::vector<double> y(n);
        mdarray<double, 2> gy({n, 3});
        sf::spherical_harmonics(3, angles[1], angles[2], y.data());
        sf::dRlm_dr(3, position, gy);
        std::array<double, 4> exact{};
        for (int lm = 0; lm < n; lm++) {
            exact[0] += radial(lm, r, 0) * y[lm];
            for (int a = 0; a < 3; a++) {
                exact[a + 1] += radial(lm, r, 1) * position[a] / r * y[lm] + radial(lm, r, 0) * gy(lm, a);
            }
        }
        auto actual = sampler.value_gradient(position);
        auto delta  = dsampler.value_gradient(position);
        error = std::max(error, std::abs(sampler.value(position) - actual[0]) / std::max(1.0, std::abs(actual[0])));
        covectors.push_back({0.7, 0.2, -0.1, 0.3});
        for (int a = 0; a < 4; a++) {
            error = std::max(error, std::abs(actual[a] - exact[a]) / std::max(1.0, std::abs(exact[a])));
            lhs += covectors.back()[a] * delta[a];
        }
    }
    auto back = sampler.adjoint(points, covectors);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            rhs += back(lm, ir) * direction(lm, ir);
        }
    }
    double adjoint = std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs));
    std::cout << "inner multipole value/gradient and adjoint: " << error << ' ' << adjoint << '\n';
    int failures = !std::isfinite(error) || error > 2e-10 || !std::isfinite(adjoint) || adjoint > 2e-11;
    failures += test_sampling_geometry(sampler, dsampler, points, covectors, direction);

    // Value gradients must also be the spatial derivatives when crossing the join at r_first.
    double gradient_error{0}, curvature_error{0}, hessian_error{0};
    for (auto const& position : points) {
        std::array<double, 4> covector{0.7, 0.2, -0.1, 0.3};
        auto analytic = sampler.position_adjoint(position, covector);
        for (double fraction : {1e-4, 5e-5}) {
            double step = fraction * position.length();
            for (int a = 0; a < 3; a++) {
                auto plus = position, minus = position;
                plus[a] += step;
                minus[a] -= step;
                auto vp = sampler.value_gradient(plus), vm = sampler.value_gradient(minus);
                double fd{0};
                for (int j = 0; j < 4; j++) {
                    fd += covector[j] * (vp[j] - vm[j]) / (2 * step);
                }
                hessian_error = std::max(hessian_error, std::abs(fd - analytic[a]) / std::max(1.0, analytic.length()));
            }
        }
    }
    std::cout << "inner geometry adjoint FD: " << hessian_error << '\n';
    failures += !std::isfinite(hessian_error) || hessian_error > 2e-6;
    for (double fraction : {0.5, 1.0}) {
        auto position = (grid.first() * fraction / std::sqrt(14.0)) * r3::vector<double>({1, -2, 3});
        auto analytic = sampler.value_gradient(position);
        for (int a = 0; a < 3; a++) {
            auto plus = position, minus = position;
            double step = 1e-7;
            plus[a] += step;
            minus[a] -= step;
            double fd      = (sampler.value_gradient(plus)[0] - sampler.value_gradient(minus)[0]) / (2 * step);
            gradient_error = std::max(gradient_error, std::abs(fd - analytic[a + 1]) / std::max(1.0, std::abs(fd)));
        }
    }
    // C2 continuity for arbitrary samples, not only the polynomial fixture.
    double h   = 1e-6 * grid.first();
    auto at    = dsampler.value_gradient({grid.first(), 0, 0});
    auto left  = dsampler.value_gradient({grid.first() - h, 0, 0});
    auto right = dsampler.value_gradient({grid.first() + h, 0, 0});
    double cl = (at[1] - left[1]) / h, cr = (right[1] - at[1]) / h;
    curvature_error = std::abs(cl - cr) / std::max({1.0, std::abs(cl), std::abs(cr)});
    std::cout << "inner spatial FD and C2 join: " << gradient_error << ' ' << curvature_error << '\n';
    failures += !std::isfinite(gradient_error) || gradient_error > 2e-7;
    failures += !std::isfinite(curvature_error) || curvature_error > 2e-4;
    return failures;
}

int
test_spheric_geometry(int lmax)
{
    std::vector<double> x{0.02, 0.09, 0.2, 0.45, 0.75, 1.1, 1.8, 2.4, 3.0};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    Flm field(sf::lmmax(lmax), grid);
    for (int lm = 0; lm < field.angular_domain_size(); lm++) {
        for (int ir = 0; ir < grid.num_points(); ir++) {
            field(lm, ir) = std::sin(0.7 * lm + grid[ir]) / (lm + 1);
        }
    }
    Spheric_function_interpolator sampler(field);
    std::vector<r3::vector<double>> points{{0.4, 0.3, 0.2},  {0, 0, 0.75},         {0, 0, -1.1},          {0.45, 0, 0},
                                           {0.3, -0.5, 0.7}, {1e-10, -2e-10, 1.1}, {-1e-10, 2e-10, -0.75}};
    double error{0}, symmetry_error{0}, gradient_error{0};
    for (auto const& p : points) {
        std::array<double, 4> covector{0.3, 0.2, -0.4, 0.5};
        auto actual = sampler.position_adjoint(p, covector);
        auto first  = sampler.position_adjoint(p, {1, 0, 0, 0});
        auto values = sampler.value_gradient(p);
        r3::matrix<double> hessian;
        for (int a = 0; a < 3; a++) {
            gradient_error = std::max(gradient_error, std::abs(first[a] - values[a + 1]));
            std::array<double, 4> unit{};
            unit[a + 1] = 1;
            auto row    = sampler.position_adjoint(p, unit);
            for (int b = 0; b < 3; b++) {
                hessian(a, b) = row[b];
            }
            for (double step : {1e-6, 5e-7}) {
                auto plus = p, minus = p;
                plus[a] += step;
                minus[a] -= step;
                auto vp = sampler.value_gradient(plus), vm = sampler.value_gradient(minus);
                double fd{0};
                for (int j = 0; j < 4; j++) {
                    fd += covector[j] * (vp[j] - vm[j]) / (2 * step);
                }
                error = std::max(error, std::abs(fd - actual[a]));
            }
        }
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                symmetry_error = std::max(symmetry_error, std::abs(hessian(a, b) - hessian(b, a)));
            }
        }
    }
    std::cout << "spherical geometry lmax=" << lmax << " FD/gradient/symmetry: " << error << ' ' << gradient_error
              << ' ' << symmetry_error << '\n';
    return !std::isfinite(error) || error > 2e-6 || !std::isfinite(gradient_error) || gradient_error > 2e-12 ||
           !std::isfinite(symmetry_error) || symmetry_error > 2e-11;
}

int
test_cusp_dipole()
{
    // For psi=exp(-a r)+x exp(-b r), the s/p term in positive tau is
    // (-a+a*b*r)*exp(-(a+b)*r)*x/r. Its l=1 coefficient is not O(r).
    double a = 1.3, b = 0.7;
    auto dipole     = [&](double r) { return (-a + a * b * r) * std::exp(-(a + b) * r); };
    double previous = 1, last_error{0};
    int failures{0};
    for (double r0 : {0.004, 0.002}) {
        std::vector<double> x(100);
        for (int ir = 0; ir < 100; ir++) {
            x[ir] = r0 * std::exp(0.03 * ir);
        }
        Radial_grid_ext<double> grid(100, x.data());
        Flm field(4, grid);
        field.zero();
        for (int ir = 0; ir < 100; ir++) {
            field(sf::lm(1, 0), ir) = dipole(grid[ir]) / std::sqrt(3 / fourpi);
        }
        Spheric_function_interpolator sampler(field, spheric_inner_extension_t::kinetic);
        double error{0};
        for (double fraction : {1e-6, 0.1, 0.5, 0.99}) {
            double r = r0 * fraction;
            error    = std::max(error, std::abs(sampler.value_gradient({0, 0, r})[0] - dipole(r)));
        }
        std::cout << "cusp kinetic dipole r_first=" << r0 << " error=" << error << '\n';
        failures += !std::isfinite(error) || error >= previous / 4;
        previous   = error;
        last_error = error;
        Spheric_function_interpolator wrong(field, spheric_inner_extension_t::charge);
        if (std::abs(wrong.value_gradient({0, 0, r0 * 1e-6})[0] - dipole(r0 * 1e-6)) < 1) {
            failures++;
        }
    }
    return failures + (last_error > 1e-7);
}

int
test_coulomb_kinetic_interpolation()
{
    std::vector<double> x{
            1e-6, 1.976562109375e-6, 4.9062484375e-6, 9.789058984375e-6, 2.5414052734375e-5, 1e-4, 1e-3, 1e-2, 1};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n = sf::lmmax(1), failures{0};
    std::vector<double> power;
    for (int l : sf::l_by_lm(1)) {
        power.push_back(radial_solver_local::scalar_relativistic_origin_power(l / 2, 3) +
                        radial_solver_local::scalar_relativistic_origin_power((l + 1) / 2, 3) - 4);
    }
    Flm field(n, grid), direction(n, grid);
    for (int lm = 0; lm < n; lm++) {
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r = grid[ir], c = lm ? 1e-8 / (lm + 1) : 1e-7 / y00;
            field(lm, ir)     = c * std::pow(r, power[lm]) * (1 + 0.4 * r + 0.3 * r * r);
            direction(lm, ir) = 0.2 * c * std::pow(r, power[lm]) * (1 + 0.1 * r + r * r);
        }
    }
    Spheric_function_interpolator sampler(field, spheric_inner_extension_t::kinetic, power);
    Spheric_function_interpolator dsampler(direction, spheric_inner_extension_t::kinetic, power);
    std::vector<r3::vector<double>> points{{2e-7, 0, 0},  {0, 0, 1e-6},           {3.393485299e-6, 0, 0},
                                           {0, -7e-6, 0}, {0.001, -0.002, 0.003}, {0, 0, 0.5}};
    std::vector<std::array<double, 4>> covectors;
    double error{0}, curvature_error{0}, lhs{0};
    for (auto p : points) {
        double r = p.length();
        std::vector<double> harmonics(n);
        mdarray<double, 2> gradient({n, 3});
        sf::spherical_harmonics(1, p, harmonics.data());
        sf::dRlm_dr(1, p, gradient);
        std::array<double, 4> expected{};
        r3::matrix<double> hessian;
        auto unit = (1 / r) * p;
        for (int lm = 0; lm < n; lm++) {
            double c          = lm ? 1e-8 / (lm + 1) : 1e-7 / y00;
            double value      = c * std::pow(r, power[lm]) * (1 + 0.4 * r + 0.3 * r * r);
            double derivative = power[lm] * value / r + c * std::pow(r, power[lm]) * (0.4 + 0.6 * r);
            double second     = power[lm] * (power[lm] - 1) * value / (r * r) +
                            2 * power[lm] * c * std::pow(r, power[lm] - 1) * (0.4 + 0.6 * r) +
                            0.6 * c * std::pow(r, power[lm]);
            expected[0] += value * harmonics[lm];
            for (int k = 0; k < 3; k++) {
                expected[k + 1] += derivative * harmonics[lm] * p[k] / r + value * gradient(lm, k);
            }
            // The l=1 field is (f(r)/r) times a Cartesian linear polynomial.
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    if (lm == 0) {
                        hessian(a, b) += harmonics[lm] * ((second - derivative / r) * unit[a] * unit[b] +
                                                          (a == b ? derivative / r : 0));
                    } else {
                        double first = derivative / r - value / (r * r);
                        double next  = second / r - 2 * derivative / (r * r) + 2 * value / (r * r * r);
                        double ca    = r * gradient(lm, a) + harmonics[lm] * unit[a];
                        double cb    = r * gradient(lm, b) + harmonics[lm] * unit[b];
                        hessian(a, b) +=
                                first * (unit[a] * cb + ca * unit[b]) +
                                r * harmonics[lm] * ((next - first / r) * unit[a] * unit[b] + (a == b ? first / r : 0));
                    }
                }
            }
        }
        auto actual = sampler.value_gradient(p);
        covectors.push_back({r * r, 0.2 * r * r * r, 0.1 * r * r * r, -0.3 * r * r * r});
        error        = std::max(error, std::abs(sampler.value(p) - actual[0]) / std::max(1.0, std::abs(actual[0])));
        auto tangent = dsampler.value_gradient(p);
        for (int k = 0; k < 4; k++) {
            error = std::max(error, std::abs(actual[k] - expected[k]) / std::max(1.0, std::abs(expected[k])));
            lhs += covectors.back()[k] * tangent[k];
        }
        for (auto v : {0.1 * r3::vector<double>(p[1], -p[2], p[0]), 0.1 * p}) {
            auto response   = sampler.position_response(p, v);
            double exact    = r3::dot(v, r3::vector<double>(expected[1], expected[2], expected[3]));
            curvature_error = std::max(curvature_error, std::abs(response[0] - exact) / std::max(1.0, std::abs(exact)));
            double peak{0}, difference{0};
            for (int a = 0; a < 3; a++) {
                exact = 0;
                for (int b = 0; b < 3; b++) {
                    exact += hessian(a, b) * v[b];
                }
                peak       = std::max(peak, std::abs(exact));
                difference = std::max(difference, std::abs(response[a + 1] - exact));
            }
            // Use a vector norm: symmetry-zero components coexist with a singular radial curvature.
            curvature_error = std::max(curvature_error, difference / std::max(1.0, peak));
        }
    }
    Test_check check(failures);
    check("Coulomb kinetic multipole values and gradients", error, 2e-11);
    check("Coulomb kinetic analytic radial and tangential Hessian", curvature_error, 2e-11);
    auto adjoint = sampler.adjoint(points, covectors);
    double rhs{0};
    for (size_t i = 0; i < adjoint.size(); i++) {
        rhs += adjoint[i] * direction[i];
    }
    check("factored radial-sample transpose", std::abs(lhs - rhs) / std::abs(lhs), 2e-11);
    Spheric_function_interpolator unscaled(field, spheric_inner_extension_t::kinetic);
    check("ordinary spline overshoot is exercised", unscaled.value_gradient(points[2])[0] < 0 ? 0 : 1, 0);
    check("factored interpolation retains positive kinetic density", sampler.value_gradient(points[2])[0] > 0 ? 0 : 1,
          0);
    // A radial component avoids subtracting almost equal singular monopole gradients.
    return failures + test_sampling_geometry(sampler, dsampler, points, covectors, direction, true);
}

int
test_positive_coulomb_interpolation(spheric_inner_extension_t inner)
{
    std::vector<double> x{
            1e-6, 1.8578115234375e-5, 7.13124609375e-5, 1.59203037109375e-4, 2.8224984375e-4, 1e-3, 1e-2, 0.1, 1.8};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n = sf::lmmax(1), failures{0};
    double power =
            2 * std::sqrt(1 - std::pow(10 / speed_of_light, 2)) - (inner == spheric_inner_extension_t::kinetic ? 4 : 2);
    std::vector<double> powers(n, power);
    Flm field(n, grid), direction(n, grid);
    field.zero();
    for (int ir = 0; ir < grid.num_points(); ir++) {
        // A positive core-like field with a sharply varying regular remainder.
        field(0, ir) = 1e-8 * std::pow(x[ir], power) * (ir == 0 ? 3 : 0.4) / y00;
        for (int lm = 0; lm < n; lm++) {
            direction(lm, ir) = 1e-8 * std::pow(x[ir], power) * std::cos(ir + 0.3 * lm) / (lm + 1);
        }
    }
    Spheric_function_interpolator cubic(field, inner, powers);
    Spheric_function_interpolator sampler(field, inner, powers, spheric_interpolation_t::positive_spline);
    Spheric_function_interpolator dsampler(direction, inner, powers, spheric_interpolation_t::positive_spline);
    std::vector<r3::vector<double>> points{{5e-7, 0, 0}, {1e-6, 0, 0},           {3.3835446e-5, 0, 0}, {0, 0, 1.3e-4},
                                           {x[3], 0, 0}, {0.003, -0.004, 0.007}, {0, 0, 0.6}};
    Test_check check(failures);
    check("factored cubic field overshoot is exercised", cubic.value(points[2]) < 0 ? 0 : 1, 0);
    std::vector<std::array<double, 4>> covectors, positive(points.size(), {1, 0, 0, 0});
    double lhs{0}, rhs{0};
    for (auto const& p : points) {
        check("positive Coulomb spline retains positive fields", sampler.value(p) > 0 ? 0 : 1, 0);
        double r = p.length();
        covectors.push_back({r * r, 0.2 * r * r * r, -0.1 * r * r * r, 0.3 * r * r * r});
        auto values = dsampler.value_gradient(p);
        for (int k = 0; k < 4; k++) {
            lhs += covectors.back()[k] * values[k];
        }
    }
    auto reverse = sampler.adjoint(points, covectors);
    for (size_t i = 0; i < reverse.size(); i++) {
        rhs += reverse[i] * direction[i];
    }
    check("positive factored value/gradient transpose", std::abs(lhs - rhs) / std::abs(lhs), 2e-11);
    auto positive_reverse = sampler.adjoint(points, positive);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        check("positive local kernel retains positive nodal weights", positive_reverse(0, ir) >= 0 ? 0 : 1, 0);
    }
    Flm linear(n, grid);
    linear.zero();
    for (int ir = 0; ir < grid.num_points(); ir++) {
        linear(0, ir) = std::pow(x[ir], power) * (1 + 0.2 * x[ir]) / y00;
    }
    Spheric_function_interpolator exact(linear, inner, powers, spheric_interpolation_t::positive_spline);
    double error{0};
    for (double r : {0.001, 0.3, 1.0, grid.last()}) {
        auto sampled      = exact.value_gradient({r, 0, 0});
        double value      = std::pow(r, power) * (1 + 0.2 * r);
        double derivative = power * value / r + 0.2 * std::pow(r, power);
        error             = std::max(error, std::abs(sampled[0] - value) / std::max(1.0, std::abs(value)));
        error             = std::max(error, std::abs(sampled[1] - derivative) / std::max(1.0, std::abs(derivative)));
    }
    check("positive spline reproduces the linear regular remainder", error, 2e-11);
    return failures + test_sampling_geometry(sampler, dsampler, points, covectors, direction, true);
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    for (int n : {4, 9, 31}) {
        result += call_test("real spline transpose", test_spline_transpose<double>, n);
        result += call_test("complex spline transpose", test_spline_transpose<std::complex<double>>, n);
    }
    result += call_test("spherical interpolation", test_spheric_interpolation);
    for (int lmax : {0, 3, 8}) {
        result += call_test("spherical geometry", test_spheric_geometry, lmax);
    }
    result += call_test("inner charge continuation", test_inner_extension, spheric_inner_extension_t::charge);
    result += call_test("inner kinetic continuation", test_inner_extension, spheric_inner_extension_t::kinetic);
    result += call_test("cusp kinetic dipole", test_cusp_dipole);
    result += call_test("Coulomb kinetic interpolation", test_coulomb_kinetic_interpolation);
    for (auto inner : {spheric_inner_extension_t::charge, spheric_inner_extension_t::kinetic}) {
        result += call_test("positive Coulomb interpolation", test_positive_coulomb_interpolation, inner);
    }
    sirius::finalize();
    return result;
}
