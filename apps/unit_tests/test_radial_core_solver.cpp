/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "radial/radial_core_solver.hpp"
#include "function3d/spheric_function_interpolator.hpp"
#include "unit_cell/atom_symmetry_class.hpp"
#include "lapw/matching_coefficients.hpp"
#include "lapw/dirac_basis.hpp"
#include "density/lapw_xc_atom.hpp"
#include <sirius.hpp>
#include <testing.hpp>
#include <limits>
#include <map>
#include <gsl/gsl_sf_bessel.h>

using namespace sirius;

template <typename Array>
static std::pair<double, double>
derivative_error(Array const& analytic__, Array const& plus__, Array const& minus__, double step__)
{
    double error{0}, peak{0};
    for (size_t i = 0; i < analytic__.size(); i++) {
        double value      = std::abs(analytic__[i]);
        double difference = std::abs(analytic__[i] - (plus__[i] - minus__[i]) / (2 * step__));
        if (!std::isfinite(value) || !std::isfinite(difference)) {
            return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        }
        error = std::max(error, difference);
        peak  = std::max(peak, value);
    }
    return {error, peak};
}

static std::vector<double>
radial_weights(Radial_grid<double> const& grid__)
{
    Spline<double> prototype(grid__);
    mdarray<double, 2> integral({grid__.num_points(), 4});
    integral.zero();
    for (int ir = 0; ir < grid__.num_points() - 1; ir++) {
        double h = grid__[ir + 1] - grid__[ir];
        for (int j = 0; j < 4; j++) {
            integral(ir, j) = std::pow(h, j + 1) / (j + 1);
        }
    }
    auto weights = prototype.interpolation_adjoint(integral);
    for (int ir = 0; ir < grid__.num_points(); ir++) {
        weights[ir] *= fourpi * grid__[ir] * grid__[ir];
    }
    return weights;
}

int
test_radial_core_solver()
{
    int failures{0};
    Test_check check(failures);
    for (int z : {1, 30}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 5000, 1e-8 / z, 120.0 / z, 3.0);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
        }
        for (int l = 0; l <= 3; l++) {
            double reference = -0.5 * z * z / std::pow(l + 1.0, 2);
            double coarse_error{0};
            for (int nb : {64, 128}) {
                Radial_core_solver solver(grid, l, z, potential, nb);
                auto states  = solver.solve(l == 0 ? 2 : 1, zero, zero);
                double error = std::abs(states[0].energy / reference - 1);
                std::cout << "Coulomb Z=" << z << " l=" << l << " basis=" << nb << '\n';
                check("Coulomb energy", error, nb == 64 ? 1e-6 : 2e-8);
                if (nb == 64) {
                    coarse_error = error;
                    continue;
                }
                check("basis refinement", error - coarse_error, 2e-9);
                for (size_t j = 0; j < states.size(); j++) {
                    auto const& state = states[j];
                    double expected   = -0.5 * z * z / std::pow(l + 1.0 + j, 2);
                    check("radial eigenvalue ordering", std::abs(state.energy / expected - 1), 2e-8);
                    check("normalization", std::abs(fourpi * Spline<double>(grid, state.rho).integrate(2) - 1), 2e-8);
                    check("Coulomb kinetic energy", std::abs(state.kinetic_energy / (-expected) - 1), 2e-7);
                    check("sampled tau integral",
                          std::abs(fourpi * Spline<double>(grid, state.tau).integrate(2) / state.kinetic_energy - 1),
                          2e-7);
                    if (l == 0 && j == 1) {
                        auto points  = grid.values();
                        auto inside  = std::lower_bound(points.begin(), points.end(), 1.0 / z) - points.begin();
                        auto outside = std::lower_bound(points.begin(), points.end(), 3.0 / z) - points.begin();
                        check("signed 2s core node", state.value[inside] * state.value[outside] < 0 ? 0 : 1, 0);
                    }
                }
                double maximum{0}, error_tau{0};
                double a             = double(z) / (l + 1);
                double normalization = std::pow(2 * a, 2 * l + 3) / std::tgamma(2 * l + 3);
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double r   = grid[ir];
                    double r2  = normalization * std::pow(r, 2 * l) * std::exp(-2 * a * r);
                    double tau = 0.5 * r2 * (std::pow(l / r - a, 2) + l * (l + 1.0) / (r * r)) / fourpi;
                    maximum    = std::max(maximum, tau);
                    error_tau  = std::max(error_tau, std::abs(states[0].tau[ir] - tau));
                }
                check("analytic positive tau", error_tau / maximum, 2e-5);
                if (z == 1 && l <= 1) {
                    // Include both 1s and the noded 2s core state in the s channel.
                    auto points = grid.values();
                    int nr      = std::upper_bound(points.begin(), points.end(), 12.0) - points.begin();
                    Radial_grid_ext<double> mt_grid(nr, points.data());
                    std::vector<double> mt_potential(potential.begin(), potential.begin() + nr);
                    Radial_solver local(z, mt_potential, mt_grid);
                    double energy     = -0.5 / std::pow(l + 4.0, 2);
                    auto radial       = local.solve(relativity_t::none, 0, l, energy);
                    auto boundary     = radial.uderiv;
                    double boundary_p = radial.p.back(), boundary_d = radial.rdudr.back();
                    orthogonalize_radial_to_core(mt_grid, states, radial);
                    Spline<double> product(mt_grid);
                    for (int ir = 0; ir < nr; ir++) {
                        product(ir) = radial.p[ir] * radial.p[ir];
                    }
                    double norm = std::sqrt(product.interpolate().integrate(0));
                    for (auto const& state : states) {
                        for (int ir = 0; ir < nr; ir++) {
                            product(ir) = state.value[ir] * radial.p[ir];
                        }
                        check("projected core-valence overlap", std::abs(product.interpolate().integrate(1)) / norm,
                              1e-11);
                    }
                    check("projection boundary value", std::abs(radial.p.back() - boundary_p), 0);
                    check("projection boundary slope", std::abs(radial.rdudr.back() - boundary_d), 0);
                    for (int j = 0; j < 3; j++) {
                        check("projection matching data", std::abs(radial.uderiv[j] - boundary[j]), 0);
                    }
                    auto twice = radial;
                    orthogonalize_radial_to_core(mt_grid, states, twice);
                    double idempotence{0};
                    for (int ir = 0; ir < nr; ir++) {
                        idempotence = std::max(idempotence, std::abs(radial.p[ir] - twice.p[ir]) / norm);
                    }
                    check("core projection idempotence", idempotence, 1e-11);
                    // Fixed boundary normalization makes the energy derivative unique.
                    Radial_core_solver auxiliary(mt_grid, l, z, mt_potential, 96, 0, true);
                    std::vector<double> no_xc(nr, 0);
                    auto derivative = auxiliary.solve_at_energy(1, energy, no_xc, no_xc);
                    orthogonalize_radial_to_core(mt_grid, states, derivative);
                    for (double h : {1e-5, 5e-6}) {
                        auto plus  = auxiliary.solve_at_energy(0, energy + h, no_xc, no_xc);
                        auto minus = auxiliary.solve_at_energy(0, energy - h, no_xc, no_xc);
                        orthogonalize_radial_to_core(mt_grid, states, plus);
                        orthogonalize_radial_to_core(mt_grid, states, minus);
                        double error{0}, peak{0};
                        for (int ir = 0; ir < nr; ir++) {
                            for (auto fields : {std::make_tuple(&derivative.p, &plus.p, &minus.p),
                                                std::make_tuple(&derivative.rdudr, &plus.rdudr, &minus.rdudr)}) {
                                double d = (*std::get<0>(fields))[ir];
                                peak     = std::max(peak, std::abs(d));
                                error    = std::max(
                                        error, std::abs(d - ((*std::get<1>(fields))[ir] - (*std::get<2>(fields))[ir]) /
                                                                    (2 * h)));
                            }
                        }
                        check("projected radial energy derivative", error / peak, 2e-6);
                    }
                }
            }
        }
    }

    for (int z : {1, 30}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 5000, 1e-8 / z, 120.0 / z, 3.0);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
        }
        for (int l : {0, 1}) {
            double guess = -0.5 * z * z / std::pow(l + 1.0, 2);
            Bound_state reference(relativity_t::zora, z, l + 1, l, 0, grid, potential, guess, 0.33, 4.5, 1e-24, 1e-12);
            Radial_core_solver solver(grid, l, z, potential, 256, 0, false, {}, relativity_t::zora);
            auto state = solver.solve(1, zero, zero)[0];
            std::cout << "ZORA Coulomb Z=" << z << " l=" << l << '\n';
            check("ZORA energy versus independent ODE", std::abs(state.energy / reference.enu() - 1), 2e-8);
            double rho_error{0}, tau_error{0}, rho_peak{0}, tau_peak{0};
            int worst_tau{0};
            for (int ir = 0; ir < grid.num_points() - 1; ir++) {
                rho_peak     = std::max(rho_peak, reference.rho()(ir));
                tau_peak     = std::max(tau_peak, reference.positive_kinetic_density()(ir));
                rho_error    = std::max(rho_error, std::abs(fourpi * state.rho[ir] - reference.rho()(ir)));
                double error = std::abs(fourpi * state.tau[ir] - reference.positive_kinetic_density()(ir));
                if (error > tau_error) {
                    tau_error = error;
                    worst_tau = ir;
                }
            }
            check("ZORA density versus independent ODE", rho_error / rho_peak, 2e-6);
            check("ZORA positive tau versus independent ODE", tau_error / tau_peak, 2e-6);
            std::cout << "largest ZORA tau difference at r=" << grid[worst_tau] << '\n';
            check("ZORA orbital normalization", std::abs(fourpi * Spline<double>(grid, state.rho).integrate(2) - 1),
                  2e-8);
            if (l == 0) {
                double gamma = std::sqrt(1 - std::pow(z / speed_of_light, 2)), beta = 2 * z / (1 + gamma);
                double normalization = std::pow(2 * beta, 2 * gamma + 1) / std::tgamma(2 * gamma + 1);
                double error{0}, peak{0};
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double r   = grid[ir];
                    double tau = 0.5 * normalization * std::pow(r, 2 * gamma - 2) * std::exp(-2 * beta * r) *
                                 std::pow((gamma - 1) / r - beta, 2);
                    peak       = std::max(peak, tau);
                    error      = std::max(error, std::abs(fourpi * state.tau[ir] - tau));
                }
                check("analytic ZORA 1s energy", std::abs(state.energy / (-0.5 * beta * beta) - 1), 2e-8);
                check("analytic ZORA 1s positive tau", error / peak, 2e-6);
            }
        }
    }

    // A constant v_tau rescales the bare kinetic operator. The effective Coulomb
    // charge of its eigenfunctions is Z/(1+v_tau), not Z.
    for (auto [l, beta] : {std::pair<int, double>{0, -0.2}, {0, 0.4}, {1, 0.4}, {2, 0.4}, {3, 0.4}}) {
        int z        = 3;
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 70.0, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), drho(grid.num_points()), dtau(grid.num_points());
        double shift = 0.13;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
            drho[ir]      = shift * weights[ir];
            dtau[ir]      = beta * weights[ir];
        }
        Radial_core_solver solver(grid, l, z, potential, 144);
        auto state      = solver.solve(1, drho, dtau)[0];
        double expected = -0.5 * z * z / (std::pow(l + 1.0, 2) * (1 + beta)) + shift;
        std::cout << "Constant tau potential l=" << l << " beta=" << beta << '\n';
        check("scaled Coulomb energy", std::abs(state.energy - expected), 2e-8);
        check("scaled Coulomb kinetic energy", std::abs(state.kinetic_energy - (shift - expected) / (1 + beta)), 2e-7);
        if (l == 0 && beta > 0) {
            // With K=r/(r+c), the manufactured XC potential makes exp(-a*r)
            // an exact ZORA eigenstate of the regular kinetic coefficient K+beta.
            double a = z / beta, c = sq_alpha_half * z;
            double energy = -0.5 * (1 + beta) * a * a;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double r = grid[ir], k = r / (r + c), dk = c / std::pow(r + c, 2);
                drho[ir] = weights[ir] * (energy + 0.5 * (k + beta) * a * a - 0.5 * a * dk - a * k / r);
            }
            Radial_core_solver zora(grid, l, z, potential, 144, 0, false, {}, relativity_t::zora, true);
            auto core = zora.solve(1, drho, dtau)[0];
            double error{0}, peak{0};
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double tau = 0.5 * std::pow(a, 5) / pi * std::exp(-2 * a * grid[ir]);
                peak       = std::max(peak, tau);
                error      = std::max(error, std::abs(core.tau[ir] - tau));
            }
            check("regular ZORA meta-GGA energy", std::abs(core.energy / energy - 1), 2e-8);
            check("regular ZORA meta-GGA positive tau", error / peak, 2e-6);
            std::vector<double> zero(grid.num_points(), 0);
            auto response = zora.solve_response(1, drho, dtau, zero, zero, weights)[0];
            check("regular ZORA tau Hellmann-Feynman response", std::abs(response.derivative.energy - 0.5 * a * a),
                  2e-7);
            for (double h : {1e-5, 5e-6}) {
                auto plus = dtau, minus = dtau;
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    plus[ir] += h * weights[ir];
                    minus[ir] -= h * weights[ir];
                }
                double fd = (zora.solve(1, drho, plus)[0].energy - zora.solve(1, drho, minus)[0].energy) / (2 * h);
                check("regular ZORA tau response", std::abs(fd - response.derivative.energy), 2e-6);
            }
        }
    }

    // Manufactured variable-mass problem. For v_tau=beta*r^2 and the potential
    // below, R=r^l exp(-a*r) is an exact nodeless eigenstate with E=-a^2/2.
    // This tests the divergence term from grad(v_tau), independently of the matrix assembly.
    for (int l : {0, 1, 2, 3}) {
        double a = 1.5, beta = 0.1 * a * a, z = a * (l + 1);
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 40.0 / a, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0), dtau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r      = grid[ir];
            potential[ir] = -z / r + 0.5 * a * a * beta * r * r - a * (l + 2) * beta * r + beta * l;
            dtau[ir]      = weights[ir] * beta * r * r;
        }
        Radial_core_solver solver(grid, l, z, potential, 128);
        auto state = solver.solve(1, zero, dtau)[0];
        std::cout << "Variable tau potential l=" << l << '\n';
        check("manufactured variable-tau energy", std::abs(state.energy + 0.5 * a * a), 2e-8);
        check("manufactured kinetic energy", std::abs(state.kinetic_energy - 0.5 * a * a), 2e-7);

        // Add arbitrary raw radial covectors, without quadrature weights, and
        // differentiate the optimized eigenvalue at two finite-difference steps.
        auto drho = zero;
        std::vector<double> direction_rho(grid.num_points()), direction_tau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r          = grid[ir];
            double envelope   = std::exp(-r - 1 / r);
            drho[ir]          = 1e-4 * std::sin(r) * envelope;
            direction_rho[ir] = 1e-4 * std::cos(2 * r) * envelope;
            direction_tau[ir] = 2e-4 * std::sin(3 * r) * envelope;
        }
        auto center = solver.solve(1, drho, dtau)[0];
        double derivative{0};
        for (int ir = 0; ir < grid.num_points(); ir++) {
            derivative += center.rho[ir] * direction_rho[ir] + center.tau[ir] * direction_tau[ir];
        }
        for (double h : {1e-3, 5e-4}) {
            auto plus_rho = drho, minus_rho = drho, plus_tau = dtau, minus_tau = dtau;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                plus_rho[ir] += h * direction_rho[ir];
                minus_rho[ir] -= h * direction_rho[ir];
                plus_tau[ir] += h * direction_tau[ir];
                minus_tau[ir] -= h * direction_tau[ir];
            }
            double fd =
                    (solver.solve(1, plus_rho, plus_tau)[0].energy - solver.solve(1, minus_rho, minus_tau)[0].energy) /
                    (2 * h);
            std::cout << "raw-field eigenvalue derivative h=" << h << '\n';
            check("Hellmann-Feynman derivative", std::abs(fd - derivative), 2e-7);
        }
        auto invalid = dtau;
        invalid[2]   = std::numeric_limits<double>::quiet_NaN();
        check.reject("nonfinite core covector", [&] { solver.solve(1, drho, invalid); });
        invalid.clear();
        check.reject("incompatible core covector size", [&] { solver.solve(1, invalid, dtau); });
    }
    // A localized mass variation checks resolution of a narrow core feature.
    // Construct V from R=r^l exp(-a*r), so both its energy and tau are known.
    for (int l : {0, 1}) {
        double z = 10, a = z / (l + 1);
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 100.0 / a, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0), dtau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r = grid[ir], x = a * r, width = 0.2;
            double envelope = 0.12 * std::exp(-std::pow((x - 1.3) / width, 2));
            double v        = x * x * envelope;
            double dv       = a * envelope * (2 * x - 2 * x * x * (x - 1.3) / (width * width));
            potential[ir]   = -z / r + 0.5 * a * a * v - z * v / r + 0.5 * dv * (l / r - a);
            dtau[ir]        = weights[ir] * v;
        }
        double previous_error{0};
        for (int nb : {128, 256, 384}) {
            Radial_core_solver solver(grid, l, z, potential, nb);
            auto state   = solver.solve(1, zero, dtau)[0];
            double error = std::abs(state.energy + 0.5 * a * a);
            std::cout << "Localized tau potential l=" << l << " basis=" << nb << '\n';
            if (nb != 128)
                check("localized basis refinement", error - previous_error, 1e-9);
            previous_error = error;
            if (nb != 384)
                continue;
            check("localized manufactured energy", error, 5e-8);
            double peak{0}, tau_error{0};
            double normalization = std::pow(2 * a, 2 * l + 3) / std::tgamma(2 * l + 3);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double r   = grid[ir];
                double rho = normalization * std::pow(r, 2 * l) * std::exp(-2 * a * r) / fourpi;
                double tau = 0.5 * rho * (std::pow(l / r - a, 2) + l * (l + 1.0) / (r * r));
                peak       = std::max(peak, tau);
                tau_error  = std::max(tau_error, std::abs(state.tau[ir] - tau));
            }
            check("localized manufactured tau", tau_error / peak, 1e-6);
        }
    }
    auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 2001, 1e-7, 2.0, 2.0);
    auto weights = radial_weights(grid);
    constexpr double q{0.7}, beta{0.2}, shift{0.3};
    std::vector<double> potential(grid.num_points(), shift), rho(grid.num_points(), 0), tau(weights);
    for (auto& w : tau) {
        w *= beta;
    }
    for (auto rel : {relativity_t::none, relativity_t::zora}) {
        double minv   = rel == relativity_t::none ? 1 : 1 / (1 - sq_alpha_half * shift);
        double energy = shift + 0.5 * (minv + beta) * q * q;
        for (int l : {0, 1, 2}) {
            Radial_core_solver solver(grid, l, 0, potential, 96, 0, true, {}, rel);
            auto radial = solver.solve_at_energy(0, energy, rho, tau);
            double error{0}, error_d{0};
            double norm = gsl_sf_bessel_jl(l, q * grid.last());
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double r = grid[ir], u = gsl_sf_bessel_jl(l, q * r) / norm;
                double du = l * u / r - q * gsl_sf_bessel_jl(l + 1, q * r) / norm;
                error     = std::max(error, std::abs(radial.p[ir] - r * u));
                error_d   = std::max(error_d, std::abs(radial.rdudr[ir] - r * du));
            }
            check("regular boundary solution", error, 2e-7);
            check("regular boundary derivative", error_d, 2e-6);
            auto derivative = solver.solve_at_energy(1, energy, rho, tau);
            auto plus       = solver.solve_at_energy(0, energy + 1e-5, rho, tau);
            auto minus      = solver.solve_at_energy(0, energy - 1e-5, rho, tau);
            error           = 0;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                error = std::max(error, std::abs(derivative.p[ir] - (plus.p[ir] - minus.p[ir]) / 2e-5));
            }
            check("radial energy derivative", error, 2e-7);
            auto points = solver.breakpoints();
            std::vector<double> extra;
            for (size_t i = 1; i < points.size(); i++) {
                extra.push_back(0.5 * (points[i - 1] + points[i]));
            }
            Radial_core_solver refined(grid, l, 0, potential, 96, 0, true, extra, rel);
            auto finer = refined.solve_at_energy(0, energy, rho, tau);
            error      = 0;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                error = std::max(error, std::abs(finer.p[ir] - radial.p[ir]));
            }
            check("nested radial basis refinement", error, 2e-7);
        }
        // The closed s-channel has an exact sine orbital, independent of a constant mass or XC tau field.
        Radial_core_solver closed(grid, 0, 0, potential, 96, 0, false, {}, rel);
        auto response = closed.solve_response(1, rho, tau, std::vector<double>(grid.num_points(), 0.7), rho, rho)[0];
        double t0     = 0.5 * std::pow(pi / grid.last(), 2);
        double dm     = rel == relativity_t::none ? 0 : 0.7 * sq_alpha_half * minv * minv;
        check("constant-mass box eigenvalue", std::abs(response.value.energy - shift - (minv + beta) * t0), 2e-8);
        check("constant-mass box kinetic energy", std::abs(response.value.kinetic_energy - minv * t0), 2e-8);
        check("unweighted box tau integral",
              std::abs(fourpi * Spline<double>(grid, response.value.tau).integrate(2) - t0), 2e-7);
        check("constant-mass eigenvalue derivative", std::abs(response.derivative.energy - 0.7 - dm * t0), 2e-10);
        check("explicit kinetic mass derivative", std::abs(response.derivative.kinetic_energy - dm * t0), 2e-10);
    }
    check.reject("unsupported relativistic core equation",
                 [&] { Radial_core_solver invalid(grid, 0, 0, potential, 96, 0, false, {}, relativity_t::iora); });
    for (double mass : {0.0, -1.0}) {
        check.reject("nonpositive ZORA core mass", [&] {
            double v = std::nextafter((1 - mass) / sq_alpha_half, std::numeric_limits<double>::infinity());
            Radial_core_solver invalid(grid, 0, 0, std::vector<double>(grid.num_points(), v), 96, 0, false, {},
                                       relativity_t::zora);
        });
    }
    return failures;
}

int
test_dirac_core_reference()
{
    int failures{0};
    Test_check check(failures);
    int z     = 30;
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 5000, 1e-8 / z, 120.0 / z, 3.0);
    std::vector<double> potential(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        potential[ir] = -z / grid[ir];
    }
    for (int kappa : {-1, 1, -2}) {
        int l = kappa > 0 ? kappa : -kappa - 1, k = std::abs(kappa);
        double gamma = std::sqrt(k * k - std::pow(z / speed_of_light, 2));
        double previous_error{0};
        for (int nb : {128, 256}) {
            Radial_dirac_core_solver solver(grid, kappa, z, potential, nb);
            auto states = solver.solve(kappa == -1 ? 2 : 1);
            std::cout << "Dirac Coulomb kappa=" << kappa << " basis=" << nb << '\n';
            double error{0};
            for (size_t j = 0; j < states.size(); j++) {
                int n    = l + 1 + j;
                double v = n - k + gamma, s = std::sqrt(1 + std::pow(z / (speed_of_light * v), 2));
                double expected   = -z * z / (v * v * s * (1 + s));
                auto const& state = states[j];
                error             = std::max(error, std::abs(state.energy / expected - 1));
                check("Dirac Coulomb eigenvalue and ordering", std::abs(state.energy / expected - 1), 2e-8);
                check("Dirac discrete spinor normalization", std::abs(solver.overlap(state, state) - 1), 2e-11);
                if (nb == 128) {
                    continue;
                }
                Bound_state reference(relativity_t::dirac, z, n, l, k, grid, potential, expected, 0.33, 4.5, 1e-30,
                                      1e-12);
                double rho_error{0}, tau_error{0}, rho_peak{0}, tau_peak{0};
                for (int ir = 0; ir < grid.num_points() - 1; ir++) {
                    rho_peak  = std::max(rho_peak, reference.rho()(ir));
                    tau_peak  = std::max(tau_peak, reference.positive_kinetic_density()(ir));
                    rho_error = std::max(rho_error, std::abs(fourpi * state.rho[ir] - reference.rho()(ir)));
                    tau_error = std::max(tau_error,
                                         std::abs(fourpi * state.tau[ir] - reference.positive_kinetic_density()(ir)));
                }
                check("Dirac density versus independent ODE", rho_error / rho_peak, 2e-6);
                check("Dirac positive tau versus independent ODE", tau_error / tau_peak, 2e-6);
                if (kappa < 0 && j == 0) {
                    double beta = double(z) / k, ratio = -z / (speed_of_light * (k + gamma));
                    double normalization =
                            std::pow(2 * beta, 2 * gamma + 1) / ((1 + ratio * ratio) * std::tgamma(2 * gamma + 1));
                    double exact_error{0}, ode_error{0}, peak{0};
                    int worst{0};
                    for (int ir = 0; ir < grid.num_points() - 1; ir++) {
                        double r = grid[ir], d = (gamma - 1) / r - beta;
                        double expected_tau = 0.5 * normalization * std::pow(r, 2 * gamma - 2) *
                                              std::exp(-2 * beta * r) *
                                              ((1 + ratio * ratio) * d * d +
                                               (l * (l + 1.0) + ratio * ratio * k * (k + 1.0)) / (r * r));
                        peak                = std::max(peak, expected_tau);
                        ode_error =
                                std::max(ode_error, std::abs(reference.positive_kinetic_density()(ir) - expected_tau));
                        double difference = std::abs(fourpi * state.tau[ir] - expected_tau);
                        if (difference > exact_error) {
                            exact_error = difference;
                            worst       = ir;
                        }
                    }
                    check("analytic Dirac Coulomb tau", exact_error / peak, 2e-6);
                    check("independent ODE Coulomb tau", ode_error / peak, 2e-6);
                    std::cout << "largest Dirac tau difference at r=" << grid[worst] << '\n';
                }
                check("Dirac sampled spinor normalization",
                      std::abs(fourpi * Spline<double>(grid, state.rho).integrate(2) - 1), 2e-8);
            }
            if (nb == 256) {
                check("Dirac radial refinement", error - previous_error, 2e-9);
                if (states.size() == 2) {
                    check("Dirac core orthogonality", std::abs(solver.overlap(states[0], states[1])), 2e-11);
                }
            }
            previous_error = error;
        }
    }
    auto regular = potential;
    std::vector<double> direction(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        double x = z * grid[ir];
        regular[ir] += 0.05 * std::exp(-x);
        direction[ir] = 0.6 * std::exp(-x) + 0.2 * std::sin(x) / (1 + x);
    }
    for (int kappa : {-1, 1, -2}) {
        Radial_dirac_core_solver solver(grid, kappa, z, regular, 128);
        auto response = solver.solve_response(kappa == -1 ? 2 : 1, direction);
        for (auto const& state : response) {
            auto const& value   = state.value;
            auto const& tangent = state.derivative;
            std::vector<double> product(grid.num_points());
            for (int ir = 0; ir < grid.num_points(); ir++) {
                product[ir] = value.rho[ir] * direction[ir];
            }
            check("Dirac response Hellmann-Feynman identity",
                  std::abs(fourpi * Spline<double>(grid, product).integrate(2) - tangent.energy), 2e-8);
            check("Dirac spinor normalization derivative",
                  std::abs(fourpi * Spline<double>(grid, tangent.rho).integrate(2)), 2e-8);
        }
        for (double step : {0.01, 0.005}) {
            auto plus = regular, minus = regular;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                plus[ir] += step * direction[ir];
                minus[ir] -= step * direction[ir];
            }
            auto p = Radial_dirac_core_solver(grid, kappa, z, plus, 128).solve(response.size());
            auto m = Radial_dirac_core_solver(grid, kappa, z, minus, 128).solve(response.size());
            std::cout << "Dirac potential response kappa=" << kappa << " h=" << step << '\n';
            for (size_t j = 0; j < response.size(); j++) {
                auto const& value   = response[j].value;
                auto const& tangent = response[j].derivative;
                check("Dirac eigenvalue response", std::abs((p[j].energy - m[j].energy) / (2 * step) - tangent.energy),
                      2e-7);
                check("Dirac kinetic-energy response",
                      std::abs((p[j].kinetic_energy - m[j].kinetic_energy) / (2 * step) - tangent.kinetic_energy),
                      2e-7);
                double ps{0}, ms{0};
                for (int i = 0; i < solver.size(); i++) {
                    ps += value.coefficients[i] * p[j].coefficients[i];
                    ms += value.coefficients[i] * m[j].coefficients[i];
                }
                ps = ps < 0 ? -1 : 1;
                ms = ms < 0 ? -1 : 1;
                for (auto member :
                     {&radial_core_state_t::value, &radial_core_state_t::derivative, &radial_core_state_t::small_value,
                      &radial_core_state_t::small_derivative, &radial_core_state_t::rho, &radial_core_state_t::tau}) {
                    double error{0}, peak{0};
                    int worst{0};
                    bool orbital = member != &radial_core_state_t::rho && member != &radial_core_state_t::tau;
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        double fd =
                                ((orbital ? ps : 1) * (p[j].*member)[ir] - (orbital ? ms : 1) * (m[j].*member)[ir]) /
                                (2 * step);
                        double difference = std::abs(fd - (tangent.*member)[ir]);
                        if (difference > error) {
                            error = difference;
                            worst = ir;
                        }
                        peak = std::max(peak, std::abs((value.*member)[ir]));
                    }
                    check("Dirac orbital/density/tau response", error / std::max(peak, 1e-30), 2e-7);
                    if (error > 2e-7 * peak) {
                        std::cout << "largest Dirac response difference at r=" << grid[worst] << '\n';
                    }
                }
            }
        }
        check.reject("invalid Dirac potential response", [&] { solver.solve_response(1, {0}); });
    }
    check.reject("zero Dirac angular channel", [&] { Radial_dirac_core_solver invalid(grid, 0, z, potential, 128); });
    check.reject("supercritical Dirac nucleus",
                 [&] { Radial_dirac_core_solver invalid(grid, -1, 2 * speed_of_light, potential, 128); });
    return failures;
}

int
test_dirac_xc_core(bool native__, int native_points__)
{
    int failures{0};
    Test_check check(failures);
    int z        = 30;
    auto grid    = Radial_grid_factory<double>(radial_grid_t::power, native__ ? native_points__ : 1001,
                                               (native__ ? 1e-4 : 1e-8) / z, 120.0 / z, 3.0);
    auto weights = radial_weights(grid);
    std::vector<double> potential(grid.num_points()), rho(grid.num_points()), tau(grid.num_points());
    std::vector<double> zero(grid.num_points(), 0);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        double x      = z * grid[ir];
        potential[ir] = -z / grid[ir] + 0.02 * std::exp(-x);
        rho[ir]       = 0.08 * weights[ir] * std::exp(-x);
        tau[ir]       = 0.005 * weights[ir] * std::exp(-x - 1 / x);
    }
    auto nuclear = potential;
    for (int ir = 0; ir < grid.num_points(); ir++) {
        nuclear[ir] = -z / grid[ir];
    }
    for (int kappa : native__ ? std::vector<int>{-1} : std::vector<int>{-1, 1, -2}) {
        auto solver = [&] {
            auto input_grid = grid.segment(grid.num_points());
            return Radial_dirac_xc_solver(input_grid, kappa, z, native__ ? 0 : 32);
        }();
        int count      = kappa == -1 ? 2 : 1;
        auto bare      = Radial_dirac_core_solver(grid, kappa, z, nuclear, native__ ? 0 : 32).solve(count);
        auto projected = solver.solve(count, nuclear, zero, zero);
        for (int j = 0; j < count; j++) {
            check("Dirac no-pair nuclear eigenvalue", std::abs(projected[j].energy - bare[j].energy), 2e-8);
            for (auto member : {&radial_core_state_t::rho, &radial_core_state_t::tau}) {
                double error{0}, peak{0};
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    error = std::max(error, std::abs((projected[j].*member)[ir] - (bare[j].*member)[ir]));
                    peak  = std::max(peak, std::abs((bare[j].*member)[ir]));
                }
                check("Dirac no-pair nuclear fields", error / peak, 2e-7);
            }
        }
        for (int part : {0, 1, 2}) {
            auto dv = zero, drho = zero, dtau = zero;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double x = z * grid[ir];
                if (part == 0)
                    dv[ir] = 0.3 * std::cos(x) * std::exp(-0.3 * x);
                if (part == 1)
                    drho[ir] = 0.2 * weights[ir] * std::cos(x) * std::exp(-x);
                if (part == 2)
                    dtau[ir] = 0.01 * weights[ir] * std::sin(x) * std::exp(-x - 1 / x);
            }
            auto response        = solver.solve_response(count, potential, rho, tau, dv, drho, dtau);
            double radial_energy = -0.3, radial_energy_direction = 0.17;
            int radial_order     = native__ ? 0 : 1;
            auto radial_response = Radial_solver(z, potential, grid, 1e-12, 1e-12)
                                           .solve_dirac_potential_response(radial_order, kappa, radial_energy, dv,
                                                                           radial_energy_direction);
            auto radial          = radial_response.value;
            radial_solver_result_t dradial{0,
                                           radial_response.p,
                                           radial_response.rdudr,
                                           radial_response.uderiv,
                                           radial_response.small_p,
                                           radial_response.small_rdudr,
                                           radial_response.small_uderiv,
                                           kappa};
            std::vector<radial_core_state_t> core, dcore;
            for (auto const& c : response) {
                core.push_back(c.value);
                dcore.push_back(c.derivative);
            }
            orthogonalize_radial_to_core(grid, core, radial, &dcore, &dradial);
            auto twice = radial;
            orthogonalize_radial_to_core(grid, core, twice);
            Spline<double> product(grid);
            for (auto member : {&radial_solver_result_t::p, &radial_solver_result_t::rdudr,
                                &radial_solver_result_t::small_p, &radial_solver_result_t::small_rdudr}) {
                double error{0};
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    error = std::max(error, std::abs((twice.*member)[ir] - (radial.*member)[ir]));
                }
                check("Dirac core projection idempotence", error, 2e-9);
                check("Dirac core projection boundary",
                      std::abs((radial.*member).back() - (radial_response.value.*member).back()), 0);
            }
            for (auto member : {&radial_solver_result_t::uderiv, &radial_solver_result_t::small_uderiv}) {
                for (int j = 0; j < 3; j++) {
                    check("Dirac core projection matching data",
                          std::abs((radial.*member)[j] - (radial_response.value.*member)[j]), 0);
                }
            }
            for (int j = 0; j < count; j++) {
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    product(ir) = core[j].value[ir] * radial.p[ir] + core[j].small_value[ir] * radial.small_p[ir];
                }
                check("Dirac projected core-valence overlap", std::abs(product.interpolate().integrate(1)), 2e-10);
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    product(ir) = dcore[j].value[ir] * radial.p[ir] + core[j].value[ir] * dradial.p[ir] +
                                  dcore[j].small_value[ir] * radial.small_p[ir] +
                                  core[j].small_value[ir] * dradial.small_p[ir];
                }
                check("Dirac core-valence overlap derivative", std::abs(product.interpolate().integrate(1)), 2e-9);
            }
            auto wrong_core     = core;
            wrong_core[0].kappa = -kappa;
            check.reject("Dirac projection cannot mix kappa channels",
                         [&] { orthogonalize_radial_to_core(grid, wrong_core, twice); });
            for (auto const& state : response) {
                check("Dirac XC spinor normalization",
                      std::abs(solver.overlap(state.value.coefficients, state.value.coefficients) - 1), 2e-11);
                check("Dirac XC normalization derivative",
                      std::abs(2 * solver.overlap(state.value.coefficients, state.derivative.coefficients)), 2e-11);
                if (part != 0) {
                    double hf{0};
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        hf += state.value.rho[ir] * drho[ir] + state.value.tau[ir] * dtau[ir];
                    }
                    check("Dirac XC raw-covector derivative", std::abs(hf - state.derivative.energy), 2e-9);
                }
            }
            if (count == 2) {
                check("Dirac XC occupied orthogonality",
                      std::abs(solver.overlap(response[0].value.coefficients, response[1].value.coefficients)), 2e-11);
            }
            for (double step : {0.001, 0.0005}) {
                auto vp = potential, vm = potential, rp = rho, rm = rho, tp = tau, tm = tau;
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    vp[ir] += step * dv[ir];
                    vm[ir] -= step * dv[ir];
                    rp[ir] += step * drho[ir];
                    rm[ir] -= step * drho[ir];
                    tp[ir] += step * dtau[ir];
                    tm[ir] -= step * dtau[ir];
                }
                auto plus = solver.solve(count, vp, rp, tp), minus = solver.solve(count, vm, rm, tm);
                auto radial_plus =
                        Radial_solver(z, vp, grid, 1e-12, 1e-12)
                                .solve_dirac(radial_order, kappa, radial_energy + step * radial_energy_direction);
                auto radial_minus =
                        Radial_solver(z, vm, grid, 1e-12, 1e-12)
                                .solve_dirac(radial_order, kappa, radial_energy - step * radial_energy_direction);
                orthogonalize_radial_to_core(grid, plus, radial_plus);
                orthogonalize_radial_to_core(grid, minus, radial_minus);
                for (auto member : {&radial_solver_result_t::p, &radial_solver_result_t::rdudr,
                                    &radial_solver_result_t::small_p, &radial_solver_result_t::small_rdudr}) {
                    double error{0}, peak{0};
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        double fd = ((radial_plus.*member)[ir] - (radial_minus.*member)[ir]) / (2 * step);
                        error     = std::max(error, std::abs(fd - (dradial.*member)[ir]));
                        peak      = std::max(peak, std::abs((dradial.*member)[ir]));
                    }
                    check("Dirac projected basis response", error / std::max(peak, 1e-12), 2e-6);
                }
                std::cout << "Dirac XC response kappa=" << kappa << " part=" << part << " h=" << step << '\n';
                for (int j = 0; j < count; j++) {
                    auto const& value   = response[j].value;
                    auto const& tangent = response[j].derivative;
                    check("Dirac XC eigenvalue response",
                          std::abs((plus[j].energy - minus[j].energy) / (2 * step) - tangent.energy) /
                                  std::max(1.0, std::abs(tangent.energy)),
                          2e-7);
                    check("Dirac XC kinetic-energy response",
                          std::abs((plus[j].kinetic_energy - minus[j].kinetic_energy) / (2 * step) -
                                   tangent.kinetic_energy) /
                                  std::max(1.0, std::abs(tangent.kinetic_energy)),
                          2e-7);
                    double ps = solver.overlap(value.coefficients, plus[j].coefficients) < 0 ? -1 : 1;
                    double ms = solver.overlap(value.coefficients, minus[j].coefficients) < 0 ? -1 : 1;
                    for (auto member : {&radial_core_state_t::value, &radial_core_state_t::small_value,
                                        &radial_core_state_t::derivative, &radial_core_state_t::small_derivative,
                                        &radial_core_state_t::rho, &radial_core_state_t::tau}) {
                        bool orbital = member != &radial_core_state_t::rho && member != &radial_core_state_t::tau;
                        double error{0}, peak{0};
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            double fd = ((orbital ? ps : 1) * (plus[j].*member)[ir] -
                                         (orbital ? ms : 1) * (minus[j].*member)[ir]) /
                                        (2 * step);
                            error     = std::max(error, std::abs(fd - (tangent.*member)[ir]));
                            peak      = std::max(peak, std::abs((value.*member)[ir]));
                        }
                        check("Dirac XC spinor/density/tau response", error / std::max(peak, 1e-30), 2e-6);
                    }
                }
            }
        }
        check.reject("invalid Dirac XC covector", [&] { solver.solve(1, potential, {0}, tau); });
        if (native__) {
            int nmtp = grid.num_points() - 4;
            std::array<std::vector<double>, 2> fields{std::vector<double>(rho.begin(), rho.begin() + nmtp),
                                                      std::vector<double>(tau.begin(), tau.begin() + nmtp)};
            std::shared_ptr<Radial_dirac_xc_solver const> reference;
            [&] {
                auto local_grid = grid.segment(grid.num_points());
                return solve_radial_dirac_core_states(local_grid, kappa, z, potential, count, fields, nmtp, nullptr,
                                                      nullptr, {}, &reference);
            }();
            auto projector = reference;
            auto changed   = potential;
            auto direction = zero;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                changed[ir] += 0.01 * std::exp(-z * grid[ir]);
                direction[ir] = 0.03 * std::cos(z * grid[ir]) * std::exp(-z * grid[ir]);
            }
            fields[0][nmtp / 2] *= 1.01;
            std::vector<radial_core_state_t> cached_response, fresh_response;
            auto cached = solve_radial_dirac_core_states(grid, kappa, z, changed, count, fields, nmtp, &direction,
                                                         &cached_response, {}, &reference);
            auto fresh  = solve_radial_dirac_core_states(grid, kappa, z, changed, count, fields, nmtp, &direction,
                                                         &fresh_response);
            check("unchanged Dirac nuclear projector", reference == projector ? 0 : 1, 0);
            for (int j = 0; j < count; j++) {
                check("exact cached Dirac core and response",
                      cached[j].energy == fresh[j].energy && cached[j].rho == fresh[j].rho &&
                                      cached[j].tau == fresh[j].tau &&
                                      cached_response[j].energy == fresh_response[j].energy &&
                                      cached_response[j].rho == fresh_response[j].rho &&
                                      cached_response[j].tau == fresh_response[j].tau
                              ? 0
                              : 1,
                      0);
            }
            cached[0].energy = std::numeric_limits<double>::quiet_NaN();
            cached[0].rho[0] = std::numeric_limits<double>::quiet_NaN();
            auto repeated    = solve_radial_dirac_core_states(grid, kappa, z, changed, count, fields, nmtp, &direction,
                                                              &cached_response, {}, &reference);
            for (int j = 0; j < count; j++) {
                check("repeated Dirac spectrum and response are independent of returned states",
                      repeated[j].energy == fresh[j].energy && repeated[j].coefficients == fresh[j].coefficients &&
                                      repeated[j].rho == fresh[j].rho && repeated[j].tau == fresh[j].tau &&
                                      cached_response[j].energy == fresh_response[j].energy &&
                                      cached_response[j].rho == fresh_response[j].rho &&
                                      cached_response[j].tau == fresh_response[j].tau
                              ? 0
                              : 1,
                      0);
            }
            auto points = grid.values();
            points[10]  = 0.5 * (points[9] + points[10]);
            Radial_grid_ext<double> other(grid.num_points(), points.data());
            check("Dirac nuclear projector channel/grid identity",
                  reference->matches(grid, kappa, z) && !reference->matches(grid, 1, z) &&
                                  !reference->matches(grid, kappa, z + 1) && !reference->matches(other, kappa, z)
                          ? 0
                          : 1,
                  0);
            check.reject("invalid Dirac core response direction", [&] {
                solve_radial_dirac_core_states(grid, kappa, z, changed, count, fields, nmtp, &zero, nullptr, {},
                                               &reference);
            });
            Simulation_parameters parameters;
            parameters.lmax_apw(0);
            parameters.lmax_rho(0);
            parameters.lmax_pot(0);
            parameters.cfg().parameters().auto_rmt(0);
            parameters.cfg().parameters().use_symmetry(false);
            Unit_cell cell(parameters, mpi::Communicator::world());
            cell.set_lattice_vectors({{40, 0, 0}, {0, 40, 0}, {0, 0, 40}});
            auto& type = cell.add_atom_type("Zn");
            type.zn(z);
            type.set_radial_grid(nmtp, grid.values().data());
            type.set_configuration(1, 0, 1, 2.0, true);
            type.add_aw_descriptor(1, 0, -0.3, 0, 0);
            type.add_aw_descriptor(1, 0, -0.3, 1, 0);
            cell.add_atom("Zn", {0, 0, 0});
            cell.initialize();
            lapw_radial_basis_t atom_basis(0, type);
            std::vector<radial_core_state_t> atom_response;
            auto atom_states = atom_basis.solve_core_states(grid, kappa, changed, count, fields, relativity_t::dirac,
                                                            &direction, &atom_response);
            for (int j = 0; j < count; j++) {
                check("shared Dirac atom core solver",
                      atom_states[j].energy == fresh[j].energy && atom_states[j].rho == fresh[j].rho &&
                                      atom_states[j].tau == fresh[j].tau &&
                                      atom_response[j].energy == cached_response[j].energy &&
                                      atom_response[j].rho == cached_response[j].rho &&
                                      atom_response[j].tau == cached_response[j].tau
                              ? 0
                              : 1,
                      0);
            }
            cell.atom(0).set_position({0.01, 0, 0});
            cell.update();
            lapw_radial_basis_t moved_basis(0, type);
            moved_basis.inherit_dirac_core_reference(atom_basis);
            auto moved_states = moved_basis.solve_core_states(grid, kappa, changed, count, fields, relativity_t::dirac,
                                                              &direction, &atom_response);
            for (int j = 0; j < count; j++) {
                check("Dirac core projector survives symmetry-class rebuild",
                      moved_states[j].energy == fresh[j].energy && moved_states[j].rho == fresh[j].rho &&
                                      moved_states[j].tau == fresh[j].tau &&
                                      atom_response[j].energy == cached_response[j].energy &&
                                      atom_response[j].rho == cached_response[j].rho &&
                                      atom_response[j].tau == cached_response[j].tau
                              ? 0
                              : 1,
                      0);
            }
        }
    }
    return failures;
}

int
test_radial_core_response(relativity_t rel__ = relativity_t::none, int num_basis__ = 128, int num_grid_points__ = 4001)
{
    int failures{0};
    Test_check check(failures);
    int z        = 8;
    auto grid    = Radial_grid_factory<double>(radial_grid_t::power, num_grid_points__, 1e-7, 12.0, 3.0);
    auto weights = radial_weights(grid);
    auto points  = grid.values();
    int nr       = std::upper_bound(points.begin(), points.end(), 1.8) - points.begin();
    Radial_grid_ext<double> mt_grid(nr, points.data());
    std::vector<double> potential(grid.num_points()), rho(grid.num_points()), tau(grid.num_points());
    std::vector<double> zero(grid.num_points(), 0);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        double r      = grid[ir];
        potential[ir] = -z / r + 0.05 * std::exp(-r);
        rho[ir]       = ir < nr ? 0.03 * weights[ir] * std::exp(-r) : 0;
        tau[ir]       = ir < nr ? 0.02 * weights[ir] * std::exp(-r) : 0;
    }
    for (int l : {0, 1}) {
        Radial_core_solver solver(grid, l, z, potential, num_basis__, mt_grid.last(), false, {}, rel__);
        for (int part : {0, 1, 2}) {
            std::cout << "Core response l=" << l << " field=" << part << " ZORA=" << (rel__ == relativity_t::zora)
                      << '\n';
            auto dv = zero, drho = zero, dtau = zero;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double r = grid[ir], envelope = std::exp(-r - 1 / r);
                if (part == 0)
                    dv[ir] = 2 * std::exp(-0.3 * r) + 0.4 * std::sin(r);
                if (part == 1 && ir < nr)
                    drho[ir] = 0.4 * weights[ir] * std::cos(2 * r) * envelope;
                if (part == 2 && ir < nr)
                    dtau[ir] = 0.1 * weights[ir] * std::sin(3 * r) * envelope;
            }
            auto response = solver.solve_response(2, rho, tau, dv, drho, dtau);
            std::vector<radial_core_state_t> core, dcore;
            for (auto const& state : response) {
                core.push_back(state.value);
                dcore.push_back(state.derivative);
                check("discrete core normalization",
                      std::abs(solver.overlap(state.value.coefficients, state.value.coefficients) - 1), 2e-11);
                check("core normalization derivative",
                      std::abs(2 * solver.overlap(state.value.coefficients, state.derivative.coefficients)), 2e-11);
                if (part != 0) {
                    double hf{0};
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        hf += state.value.rho[ir] * drho[ir] + state.value.tau[ir] * dtau[ir];
                    }
                    check("core response Hellmann-Feynman identity", std::abs(hf - state.derivative.energy), 1e-11);
                }
            }
            std::vector<double> mt_potential(potential.begin(), potential.begin() + nr);
            std::vector<double> mt_direction(dv.begin(), dv.begin() + nr);
            Radial_solver local(z, mt_potential, mt_grid, 1e-11, 1e-11);
            auto valence   = local.solve_potential_response(0, l, 0.1, mt_direction);
            auto projected = valence.value;
            radial_solver_result_t projected_response{0, valence.p, valence.rdudr, valence.uderiv};
            orthogonalize_radial_to_core(mt_grid, core, projected, &dcore, &projected_response);
            Spline<double> product(mt_grid);
            for (size_t j = 0; j < core.size(); j++) {
                for (int ir = 0; ir < nr; ir++) {
                    product(ir) = dcore[j].value[ir] * projected.p[ir] + core[j].value[ir] * projected_response.p[ir];
                }
                check("core-valence orthogonality derivative", std::abs(product.interpolate().integrate(1)), 2e-9);
            }
            for (int order = 0; order < 3; order++) {
                check("core projection preserves boundary response",
                      std::abs(projected_response.uderiv[order] - valence.uderiv[order]), 0);
            }
            std::map<double, std::pair<std::vector<radial_core_state_t>, radial_solver_result_t>> samples;
            auto sample = [&](double step) -> auto const& {
                auto found = samples.find(step);
                if (found != samples.end()) {
                    return found->second;
                }
                auto v = potential, r = rho, t = tau;
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    v[ir] += step * dv[ir];
                    r[ir] += step * drho[ir];
                    t[ir] += step * dtau[ir];
                }
                Radial_core_solver shifted(grid, l, z, v, num_basis__, mt_grid.last(), false, {}, rel__);
                auto states = shifted.solve(2, r, t);
                v.resize(nr);
                Radial_solver radial(z, v, mt_grid, 1e-11, 1e-11);
                auto orbital = radial.solve(relativity_t::none, 0, l, 0.1);
                orthogonalize_radial_to_core(mt_grid, states, orbital);
                return samples.emplace(step, std::make_pair(std::move(states), std::move(orbital))).first->second;
            };
            // The native s-channel needs larger fourth-order steps to avoid near-origin gradient roundoff.
            double step = !num_basis__ && l == 0 ? (part ? 0.16 : 0.04) : 1e-2;
            for (double h : {step, step / 2}) {
                auto const& plus   = sample(h);
                auto const& minus  = sample(-h);
                auto const& plus2  = sample(2 * h);
                auto const& minus2 = sample(-2 * h);
                auto difference    = [h](double p, double m, double p2, double m2) {
                    return (8 * (p - m) - (p2 - m2)) / (12 * h);
                };
                std::cout << "Core response l=" << l << " field=" << part << " h=" << h << '\n';
                for (size_t j = 0; j < response.size(); j++) {
                    auto const& d = response[j].derivative;
                    check("core eigenvalue response",
                          std::abs(d.energy - difference(plus.first[j].energy, minus.first[j].energy,
                                                         plus2.first[j].energy, minus2.first[j].energy)),
                          2e-7);
                    check("core kinetic-energy response",
                          std::abs(d.kinetic_energy -
                                   difference(plus.first[j].kinetic_energy, minus.first[j].kinetic_energy,
                                              plus2.first[j].kinetic_energy, minus2.first[j].kinetic_energy)),
                          2e-7);
                    auto charge = [&](radial_core_state_t const& state) {
                        return fourpi * Spline<double>(grid, state.rho).integrate(2);
                    };
                    check("sampled electron-integral response",
                          std::abs(charge(d) - difference(charge(plus.first[j]), charge(minus.first[j]),
                                                          charge(plus2.first[j]), charge(minus2.first[j]))),
                          2e-8);
                    auto phase = [&](radial_core_state_t const& state) {
                        return solver.overlap(core[j].coefficients, state.coefficients) < 0 ? -1.0 : 1.0;
                    };
                    for (auto member : {&radial_core_state_t::value, &radial_core_state_t::derivative,
                                        &radial_core_state_t::rho, &radial_core_state_t::tau}) {
                        bool signed_field =
                                member == &radial_core_state_t::value || member == &radial_core_state_t::derivative;
                        double sign_plus   = signed_field ? phase(plus.first[j]) : 1;
                        double sign_minus  = signed_field ? phase(minus.first[j]) : 1;
                        double sign_plus2  = signed_field ? phase(plus2.first[j]) : 1;
                        double sign_minus2 = signed_field ? phase(minus2.first[j]) : 1;
                        double error{0}, peak{0};
                        int worst{0};
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            peak = std::max(peak, std::abs((d.*member)[ir]));
                            double delta =
                                    std::abs((d.*member)[ir] - difference(sign_plus * (plus.first[j].*member)[ir],
                                                                          sign_minus * (minus.first[j].*member)[ir],
                                                                          sign_plus2 * (plus2.first[j].*member)[ir],
                                                                          sign_minus2 * (minus2.first[j].*member)[ir]));
                            if (delta > error) {
                                error = delta;
                                worst = ir;
                            }
                        }
                        if (error > 2e-5 * peak) {
                            std::cout << "state=" << j << " r=" << grid[worst] << " peak=" << peak << " error=" << error
                                      << '\n';
                        }
                        check("nonzero core field response", peak > 1e-6 ? 0 : 1, 0);
                        check("core orbital/density/tau response", error / peak, 2e-5);
                    }
                }
                for (auto member : {&radial_solver_result_t::p, &radial_solver_result_t::rdudr}) {
                    double error{0}, peak{0};
                    for (int ir = 0; ir < nr; ir++) {
                        peak  = std::max(peak, std::abs((projected_response.*member)[ir]));
                        error = std::max(error,
                                         std::abs((projected_response.*member)[ir] -
                                                  difference((plus.second.*member)[ir], (minus.second.*member)[ir],
                                                             (plus2.second.*member)[ir], (minus2.second.*member)[ir])));
                    }
                    check("core-projected valence response", error / peak, 2e-5);
                }
            }
        }
        auto shift = solver.solve_response(2, rho, tau, std::vector<double>(grid.num_points(), 0.7), zero, zero);
        auto null  = solver.solve_response(2, rho, tau, zero, zero, zero);
        for (size_t j = 0; j < shift.size(); j++) {
            double peak{0}, error{0};
            for (int ir = 0; ir < grid.num_points(); ir++) {
                peak  = std::max(peak, std::abs(shift[j].value.value[ir]));
                error = std::max(error, std::abs(shift[j].derivative.value[ir]));
            }
            if (rel__ == relativity_t::none) {
                check("constant potential eigenvalue response", std::abs(shift[j].derivative.energy - 0.7), 2e-11);
                check("constant potential leaves core orbital unchanged", error / peak, 2e-10);
            } else {
                // A ZORA potential shift also changes the inverse mass, unlike the nonrelativistic operator.
                check("ZORA potential shift changes the core mass", shift[j].derivative.energy > 0.7 ? 0 : 1, 0);
                check("ZORA potential shift changes the core orbital", error / peak > 1e-9 ? 0 : 1, 0);
            }
            check("zero core eigenvalue response", std::abs(null[j].derivative.energy), 0);
            double null_error = std::abs(null[j].derivative.kinetic_energy);
            for (auto member :
                 {&radial_core_state_t::coefficients, &radial_core_state_t::value, &radial_core_state_t::derivative,
                  &radial_core_state_t::rho, &radial_core_state_t::tau}) {
                for (auto x : null[j].derivative.*member) {
                    null_error = std::max(null_error, std::abs(x));
                }
            }
            check("zero core orbital response", null_error, 0);
        }
        for (int part : {0, 1, 2}) {
            for (bool nonfinite : {false, true}) {
                std::array<std::vector<double>, 3> directions{zero, zero, zero};
                if (nonfinite) {
                    directions[part][0] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    directions[part].pop_back();
                }
                check.reject("invalid core response direction",
                             [&] { solver.solve_response(2, rho, tau, directions[0], directions[1], directions[2]); });
            }
        }
    }
    return failures;
}

int
test_dirac_radial_solution()
{
    int failures{0};
    Test_check check(failures);
    for (int z : {0, 30}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 1001, 1e-6, 1.8, 2.0);
        std::vector<double> potential(grid.num_points()), direction(grid.num_points()), zero(grid.num_points(), 0);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir] + 0.2 * std::exp(-grid[ir]);
            direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
        }
        double energy = z ? -0.3 : 0.5;
        Radial_solver solver(z, potential, grid, 1e-12, 1e-12);
        for (int kappa : {-1, 1, -2, 2, -3}) {
            for (int order : {0, 1, 2, 3}) {
                auto response  = solver.solve_dirac_potential_response(order, kappa, energy, direction, -0.17);
                auto reference = solver.solve_dirac(order, kappa, energy);
                auto null      = solver.solve_dirac_potential_response(order, kappa, energy, zero);
                Spline<double> product(grid);
                double value_error{0}, null_error{0};
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    product(ir) =
                            response.value.p[ir] * response.p[ir] + response.value.small_p[ir] * response.small_p[ir];
                    value_error = std::max({value_error, std::abs(reference.p[ir] - response.value.p[ir]),
                                            std::abs(reference.small_p[ir] - response.value.small_p[ir])});
                    null_error  = std::max({null_error, std::abs(null.p[ir]), std::abs(null.small_p[ir]),
                                            std::abs(null.rdudr[ir]), std::abs(null.small_rdudr[ir])});
                }
                std::cout << "Dirac LAPW radial Z=" << z << " kappa=" << kappa << " order=" << order << '\n';
                check("Dirac radial reference value", value_error, 2e-9);
                check("Dirac radial zero direction", null_error, 0);
                check("Dirac radial channel", reference.kappa == kappa ? 0 : 1, 0);
                if (!order) {
                    check("Dirac joint normalization derivative", std::abs(product.interpolate().integrate(0)), 2e-10);
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        product(ir) = reference.p[ir] * reference.p[ir] + reference.small_p[ir] * reference.small_p[ir];
                    }
                    check("Dirac joint radial normalization", std::abs(product.interpolate().integrate(0) - 1), 2e-11);
                }
                for (double h : {1e-3, 5e-4}) {
                    auto plus = potential, minus = potential;
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        plus[ir] += h * direction[ir];
                        minus[ir] -= h * direction[ir];
                    }
                    auto p = Radial_solver(z, plus, grid, 1e-12, 1e-12).solve_dirac(order, kappa, energy - 0.17 * h);
                    auto m = Radial_solver(z, minus, grid, 1e-12, 1e-12).solve_dirac(order, kappa, energy + 0.17 * h);
                    for (auto fields :
                         {std::tuple{&response.p, &p.p, &m.p}, std::tuple{&response.rdudr, &p.rdudr, &m.rdudr},
                          std::tuple{&response.small_p, &p.small_p, &m.small_p},
                          std::tuple{&response.small_rdudr, &p.small_rdudr, &m.small_rdudr}}) {
                        double error{0}, peak{0};
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            double d  = (*std::get<0>(fields))[ir];
                            double fd = ((*std::get<1>(fields))[ir] - (*std::get<2>(fields))[ir]) / (2 * h);
                            error     = std::max(error, std::abs(d - fd));
                            peak      = std::max(peak, std::abs(d));
                        }
                        check("Dirac radial mixed potential/energy response", error / std::max(peak, 1e-12), 2e-6);
                    }
                    for (auto fields : {std::tuple{response.uderiv, p.uderiv, m.uderiv},
                                        std::tuple{response.small_uderiv, p.small_uderiv, m.small_uderiv}}) {
                        for (int j = 0; j < 3; j++) {
                            double fd = (std::get<1>(fields)[j] - std::get<2>(fields)[j]) / (2 * h);
                            check("Dirac radial boundary response", std::abs(fd - std::get<0>(fields)[j]), 2e-7);
                        }
                    }
                    if (order) {
                        auto p = solver.solve_dirac(order - 1, kappa, energy + h);
                        auto m = solver.solve_dirac(order - 1, kappa, energy - h);
                        for (auto member : {&radial_solver_result_t::p, &radial_solver_result_t::rdudr,
                                            &radial_solver_result_t::small_p, &radial_solver_result_t::small_rdudr}) {
                            double error{0}, peak{0};
                            for (int ir = 0; ir < grid.num_points(); ir++) {
                                double fd = ((p.*member)[ir] - (m.*member)[ir]) / (2 * h);
                                error     = std::max(error, std::abs(fd - (reference.*member)[ir]));
                                peak      = std::max(peak, std::abs((reference.*member)[ir]));
                            }
                            check("Dirac normalized energy derivative", error / std::max(peak, 1e-12), 2e-6);
                        }
                    }
                }
            }
            // Free-particle Bessel spinors independently fix the P/Q phase and kappa branches.
            if (!z) {
                Spline<double> product(grid);
                int l           = kappa > 0 ? kappa : -kappa - 1;
                double momentum = std::sqrt(energy * (2 + energy / (speed_of_light * speed_of_light)));
                auto spinor     = Radial_solver(z, zero, grid, 1e-12, 1e-12).solve_dirac(0, kappa, energy);
                auto p = zero, q = zero;
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double x = momentum * grid[ir], jl = gsl_sf_bessel_jl(l, x);
                    p[ir]       = grid[ir] * jl;
                    q[ir]       = ((l + 1 + kappa) * jl - x * gsl_sf_bessel_jl(l + 1, x)) /
                                  (2 * speed_of_light + energy / speed_of_light);
                    product(ir) = p[ir] * p[ir] + q[ir] * q[ir];
                }
                double norm = std::sqrt(product.interpolate().integrate(0)), error{0};
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    error = std::max({error, std::abs(spinor.p[ir] - p[ir] / norm),
                                      std::abs(spinor.small_p[ir] - q[ir] / norm)});
                }
                check("Dirac free spinor versus spherical Bessel reference", error, 2e-9);
            }
        }
        check.reject("Dirac radial missing channel", [&] { solver.solve_dirac(0, 0, energy); });
        check.reject("Dirac radial missing potential direction",
                     [&] { solver.solve_dirac_potential_response(0, -1, energy, {}); });
    }
    // A short sphere avoids the exponentially growing branch of outward bound-state integration.
    int z     = 30;
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 1001, 1e-8 / z, 6.0 / z, 2.0);
    std::vector<double> potential(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        potential[ir] = -z / grid[ir];
    }
    for (int kappa : {-1, -2}) {
        double k = -kappa, gamma = std::sqrt(k * k - std::pow(z / speed_of_light, 2));
        double ratio = -z / (speed_of_light * (k + gamma)), beta = z / k;
        double s      = std::sqrt(1 + std::pow(z / (speed_of_light * gamma), 2));
        double energy = -z * z / (gamma * gamma * s * (1 + s));
        auto value    = Radial_solver(z, potential, grid, 1e-14, 1e-14).solve_dirac(0, kappa, energy);
        Spline<double> product(grid);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            product(ir) = (1 + ratio * ratio) * std::pow(grid[ir], 2 * gamma) * std::exp(-2 * beta * grid[ir]);
        }
        double norm = std::sqrt(product.interpolate().integrate(0)), error{0};
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double p = std::pow(grid[ir], gamma) * std::exp(-beta * grid[ir]) / norm;
            error    = std::max({error, std::abs(value.p[ir] - p), std::abs(value.small_p[ir] - ratio * p),
                                 std::abs(value.rdudr[ir] - ((gamma - 1) / grid[ir] - beta) * p),
                                 std::abs(value.small_rdudr[ir] - ratio * ((gamma - 1) / grid[ir] - beta) * p)});
        }
        check("Dirac Coulomb radial spinor and derivative", error, 2e-7);
    }
    return failures;
}

int
test_radial_potential_response()
{
    int failures{0};
    Test_check check(failures);
    // The long sphere also exercises rescaling of growing forward solutions.
    for (auto [rel, z, radius] : {std::tuple{relativity_t::none, 2, 1.8}, std::tuple{relativity_t::none, 2, 14.0},
                                  std::tuple{relativity_t::koelling_harmon, 30, 1.8},
                                  std::tuple{relativity_t::zora, 30, 1.8}, std::tuple{relativity_t::iora, 30, 1.8}}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 801, 1e-6, radius, 2.0);
        std::vector<double> potential(grid.num_points()), direction(grid.num_points()), zero(grid.num_points(), 0);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir] + 0.2 * std::exp(-grid[ir]);
            direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
        }
        double energy = radius > 2 ? -4.0 : -0.3;
        Radial_solver solver(z, potential, grid, 1e-11, 1e-11);
        for (int l : {0, 1, 2, 3}) {
            for (int order : {0, 1, 2, 3}) {
                for (auto [de, energy_only] :
                     {std::tuple{0.0, false}, std::tuple{0.23, true}, std::tuple{-0.17, false}}) {
                    if (de != 0 && (l != 2 || order != 3)) {
                        continue;
                    }
                    auto const& field = energy_only ? zero : direction;
                    auto response     = solver.solve_potential_response(order, l, energy, field, rel, de);
                    auto reference    = solver.solve(rel, order, l, energy);
                    auto null         = solver.solve_potential_response(order, l, energy, zero, rel);
                    double value_error{0}, null_error{0};
                    Spline<double> product(grid);
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        value_error = std::max(value_error, std::abs(response.value.p[ir] - reference.p[ir]));
                        null_error  = std::max({null_error, std::abs(null.p[ir]), std::abs(null.rdudr[ir])});
                        product(ir) = response.value.p[ir] * response.p[ir];
                    }
                    std::cout << "Potential response relativity=" << static_cast<int>(rel) << " Z=" << z
                              << " radius=" << radius << " l=" << l << " order=" << order << " dE=" << de << '\n';
                    check("unchanged radial solution", value_error, 2e-9);
                    check("zero potential direction", null_error, 0);
                    check("normalization response", std::abs(product.interpolate().integrate(0)), 2e-10);
                    double step_size = de == 0 ? 1e-4 : 5e-5;
                    for (double step : {step_size, step_size / 2}) {
                        auto shifted = [&](double sign) {
                            auto v = potential;
                            for (int ir = 0; ir < grid.num_points(); ir++) {
                                v[ir] += sign * step * field[ir];
                            }
                            return Radial_solver(z, v, grid, 1e-11, 1e-11)
                                    .solve(rel, order, l, energy + sign * step * de);
                        };
                        auto plus = shifted(1), minus = shifted(-1);
                        double error{0}, peak{0}, boundary_error{0};
                        for (auto fields : {std::make_tuple(&response.p, &plus.p, &minus.p),
                                            std::make_tuple(&response.rdudr, &plus.rdudr, &minus.rdudr)}) {
                            for (int ir = 0; ir < grid.num_points(); ir++) {
                                double analytic = (*std::get<0>(fields))[ir];
                                double fd = ((*std::get<1>(fields))[ir] - (*std::get<2>(fields))[ir]) / (2 * step);
                                error     = std::max(error, std::abs(analytic - fd));
                                peak      = std::max(peak, std::abs(analytic));
                            }
                        }
                        for (int j : {0, 1, 2}) {
                            double fd      = (plus.uderiv[j] - minus.uderiv[j]) / (2 * step);
                            boundary_error = std::max(boundary_error, std::abs(response.uderiv[j] - fd));
                        }
                        check("nonzero potential response", peak > 1e-5 ? 0 : 1, 0);
                        check("radial potential derivative", error / peak, 2e-6);
                        check("matching-data potential derivative", boundary_error, 2e-7);
                    }
                }
            }
        }
        for (auto invalid : {std::vector<double>{},
                             std::vector<double>(grid.num_points(), std::numeric_limits<double>::quiet_NaN())}) {
            check.reject("invalid potential response direction",
                         [&] { solver.solve_potential_response(0, 0, energy, invalid); });
        }
        check.reject("unsupported Dirac forward response",
                     [&] { solver.solve_potential_response(0, 0, energy, zero, relativity_t::dirac); });
        check.reject("invalid radial energy direction", [&] {
            solver.solve_potential_response(0, 0, energy, zero, rel, std::numeric_limits<double>::infinity());
        });
    }
    return failures;
}

int
test_linearization_energy_response()
{
    int failures{0};
    Test_check check(failures);
    for (auto [rel, z, radius] :
         {std::tuple{relativity_t::none, 2, 3.0}, std::tuple{relativity_t::koelling_harmon, 12, 0.8},
          std::tuple{relativity_t::zora, 12, 0.8}, std::tuple{relativity_t::iora, 12, 0.8}}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 1601, 1e-6, radius, 2.0);
        std::vector<double> v(grid.num_points()), direction(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            v[ir]         = -z / grid[ir] + 0.2 * std::exp(-grid[ir]);
            direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
        }
        for (int mode : {1, 2, 3}) {
            for (int l : {0, 1, 2}) {
                enu_search_t initial{-0.3, -0.3, -0.3, mode};
                auto base       = find_enu(rel, z, l + 1, l, grid, v, initial);
                double analytic = linearization_energy_response(z, l, grid, v, base, direction, rel);
                std::cout << "Linearisation energy response relativity=" << static_cast<int>(rel) << " mode=" << mode
                          << " l=" << l << '\n';
                check("nonzero linearisation energy response", std::abs(analytic) > 1e-3 ? 0 : 1, 0);
                if (rel == relativity_t::none || rel == relativity_t::koelling_harmon) {
                    check("constant potential shifts linearisation energy",
                          std::abs(linearization_energy_response(z, l, grid, v, base,
                                                                 std::vector<double>(grid.num_points(), 1), rel) -
                                   1),
                          2e-12);
                }
                check("zero linearisation energy response",
                      std::abs(linearization_energy_response(z, l, grid, v, base,
                                                             std::vector<double>(grid.num_points(), 0), rel)),
                      0);
                for (double step : {1e-3, 5e-4}) {
                    auto sample = [&](double h) {
                        auto shifted = v;
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            shifted[ir] += h * direction[ir];
                        }
                        return find_enu(rel, z, l + 1, l, grid, shifted, base).enu;
                    };
                    check("boundary-root energy derivative",
                          std::abs(analytic - (sample(step) - sample(-step)) / (2 * step)), 2e-6);
                }
                auto invalid = base;
                invalid.etop += 0.1;
                check.reject("unconverged energy root rejected",
                             [&] { linearization_energy_response(z, l, grid, v, invalid, direction, rel); });
            }
        }
    }
    return failures;
}

int
test_radial_basis_response(bool dirac_core_only)
{
    int failures{0};
    Test_check check(failures);
    struct setup_t
    {
        bool core;
        bool orthogonalize;
        int part; // Potential, density covector, tau covector, or all three.
        int auto_enu{0};
        std::string relativity{"none"};
        std::string core_relativity{"none"};
    };
    for (auto [core, orthogonalize, part, auto_enu, relativity, core_relativity] :
         {setup_t{false, false, 0}, setup_t{false, true, 0}, setup_t{true, false, 0}, setup_t{true, true, 0},
          setup_t{true, true, 1}, setup_t{true, true, 2}, setup_t{false, true, 0, 1}, setup_t{false, true, 0, 2},
          setup_t{false, true, 0, 3}, setup_t{true, true, 3, 1}, setup_t{false, true, 0, 2, "koelling_harmon"},
          setup_t{false, true, 0, 1, "zora"}, setup_t{false, true, 0, 3, "iora"},
          setup_t{true, true, 3, 0, "none", "zora"}, setup_t{true, true, 3, 0, "zora", "zora"},
          setup_t{true, true, 3, 0, "iora", "none"}, setup_t{true, true, 3, 0, "none", "dirac"},
          setup_t{true, true, 3, 0, "zora", "dirac"}}) {
        if (dirac_core_only && core_relativity != "dirac") {
            continue;
        }
        Simulation_parameters parameters;
        parameters.valence_relativity(relativity);
        parameters.core_relativity(core_relativity);
        parameters.lmax_apw(3);
        parameters.lmax_rho(6);
        parameters.lmax_pot(6);
        parameters.cfg().parameters().auto_rmt(0);
        parameters.cfg().parameters().use_symmetry(false);
        parameters.cfg().control().ortho_rf(orthogonalize);
        Unit_cell cell(parameters, mpi::Communicator::world());
        cell.set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
        std::string label = core ? "Ne" : "He";
        int z             = core ? 10 : 2;
        auto& type        = cell.add_atom_type(label);
        type.zn(z);
        type.set_radial_grid(radial_grid_t::power,
                             core_relativity == "dirac" ? 321 : (core && core_relativity == "none" ? 3201 : 801), 1e-6,
                             1.8, 2.0);
        if (core) {
            auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
            std::vector<double> density(free_grid.num_points());
            for (int ir = 0; ir < free_grid.num_points(); ir++) {
                density[ir] = 2 * std::pow(9.0, 3) / pi * std::exp(-18 * free_grid[ir]) +
                              8 * std::pow(1.6, 3) / pi * std::exp(-3.2 * free_grid[ir]);
            }
            type.free_atom_density(density);
            type.set_free_atom_radial_grid(std::move(free_grid));
            type.set_configuration(1, 0, 1, 2.0, true);
            if (core_relativity == "dirac") {
                type.set_configuration(2, 0, 1, 2.0, true);
                type.set_configuration(2, 1, 1, 2.0, true);
                type.set_configuration(2, 1, 2, 4.0, true);
            }
        }
        for (int l = 0; l <= 3; l++) {
            for (int order = 0; order < (l < 2 ? 3 : (l == 2 ? 2 : 1)); order++) {
                type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, order, auto_enu);
            }
        }
        // Two and three boundary conditions, with two LOs of the same l to test orthogonalization.
        for (int l : {0, 1}) {
            for (int lo : {0, 1}) {
                for (int order = 0; order < 2 + lo; order++) {
                    type.add_lo_descriptor(2 * l + lo, std::max(core ? 2 : 1, l + 1), l, lo ? 0.8 : -0.7, order,
                                           auto_enu);
                }
            }
        }
        cell.add_atom(label, {0.23, 0.31, 0.43});
        cell.initialize();
        auto& atom = cell.atom(0);
        fft::Gvec gkvec({0.07, -0.03, 0.02}, cell.reciprocal_lattice_vectors(), 2.0, cell.comm(), false);
        Matching_coefficients matching(cell, gkvec);
        auto const& grid = type.radial_grid();
        std::vector<double> potential(grid.num_points()), direction(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir] + 0.2 * std::exp(-grid[ir]);
            direction[ir] = (part == 0 || part == 3) ? 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]) : 0;
        }
        std::array<std::vector<double>, 2> fields, xc_direction;
        if (core) {
            auto weights = radial_weights(grid);
            fields       = {weights, weights};
            if (part == 1 || part == 3) {
                xc_direction[0] = weights;
            }
            if (part == 2 || part == 3) {
                xc_direction[1] = weights;
            }
            for (int ir = 0; ir < grid.num_points(); ir++) {
                fields[0][ir] *= 0.03 * std::exp(-grid[ir]);
                fields[1][ir] *= 0.02 * std::exp(-grid[ir]);
                if (!xc_direction[0].empty()) {
                    xc_direction[0][ir] *= 4 * std::sin(3 * grid[ir]) * std::exp(-grid[ir]);
                }
                if (!xc_direction[1].empty()) {
                    xc_direction[1][ir] *= 0.03 * std::sin(3 * grid[ir]) * std::exp(-grid[ir]);
                }
            }
        }
        LAPW_radial_basis radial_basis(cell, parameters.valence_relativity(), {potential}, {fields});
        auto& cls   = radial_basis.radial_basis_for_class(0);
        auto sample = [&](double step) {
            auto v  = potential;
            auto xc = fields;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                v[ir] += step * direction[ir];
                for (int j : {0, 1}) {
                    if (!xc_direction[j].empty()) {
                        xc[j][ir] += step * xc_direction[j][ir];
                    }
                }
            }
            cls.set_spherical_potential(v);
            cls.set_spherical_xc_derivatives(xc);
            // Exercise the production owner-rank generation and replicated response.
            radial_basis.generate_radial_functions();
            std::vector<double> values;
            for (int j : {0, 1}) {
                for (int i = 0; i < type.mt_radial_basis_size(); i++) {
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        values.push_back(j ? cls.radial_function_derivative(ir, i) : cls.radial_function(ir, i));
                    }
                }
            }
            for (int l = 0; l <= 3; l++) {
                for (int order = 0; order < type.aw_order(l); order++) {
                    for (int dm : {0, 1, 2}) {
                        values.push_back(cls.aw_surface_deriv(l, order, dm));
                    }
                }
            }
            mdarray<std::complex<double>, 2> alm({gkvec.count(), type.mt_aw_basis_size()});
            matching.generate<false>(atom, alm, cls);
            return std::make_pair(std::move(values), std::move(alm));
        };
        auto baseline = sample(0);
        auto energies = [&]() {
            std::vector<double> values;
            for (int l = 0; l < cls.num_aw_descriptors(); l++) {
                for (auto const& d : cls.aw_descriptor(l)) {
                    values.push_back(d.enu);
                }
            }
            for (int i = 0; i < cls.num_lo_descriptors(); i++) {
                for (auto const& d : cls.lo_descriptor(i).rsd_set) {
                    values.push_back(d.enu);
                }
            }
            return values;
        };
        auto baseline_energies = energies();
        auto response          = cls.radial_basis_potential_response(direction, xc_direction);
        if (core_relativity == "dirac") {
            auto [extended_grid, extended_potential] = type.core_radial_problem(potential);
            lapw_radial_basis_t::core_states_t dcores;
            auto extended_direction = direction;
            extended_direction.resize(extended_grid.num_points(), direction.back());
            auto cores = cls.solve_occupied_core_states(extended_grid, extended_potential, fields, relativity_t::dirac,
                                                        &extended_direction, &dcores, xc_direction);
            double overlap_error{0}, tangent_error{0};
            Spline<double> product(grid);
            for (auto const& rf : type.indexr()) {
                int i = int(rf.idxrf);
                for (auto const& [channel, states] : cores) {
                    if ((channel < 0 ? -channel - 1 : channel) != rf.am.l()) {
                        continue;
                    }
                    for (size_t j = 0; j < states.size(); j++) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            product(ir) = states[j].value[ir] * cls.radial_function(ir, i);
                        }
                        overlap_error = std::max(overlap_error, std::abs(product.interpolate().integrate(2)));
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            product(ir) = dcores.at(channel)[j].value[ir] * cls.radial_function(ir, i) +
                                          states[j].value[ir] * response.radial_functions(ir, i, 0);
                        }
                        tangent_error = std::max(tangent_error, std::abs(product.interpolate().integrate(2)));
                    }
                }
            }
            check("both Dirac j channels orthogonal to scalar valence", overlap_error, 2e-10);
            check("both Dirac j channels' overlap response", tangent_error, 2e-9);
        }
        auto zero = cls.radial_basis_potential_response(std::vector<double>(grid.num_points(), 0));
        std::vector<double> analytic(response.radial_functions.begin(), response.radial_functions.end());
        analytic.insert(analytic.end(), response.surface_derivatives.begin(), response.surface_derivatives.end());
        double null_error{0}, norm_error{0}, boundary_error{0}, unchanged{0};
        for (double x : zero.radial_functions) {
            null_error = std::max(null_error, std::abs(x));
        }
        for (double x : zero.surface_derivatives) {
            null_error = std::max(null_error, std::abs(x));
        }
        for (int i = 0; i < type.mt_radial_basis_size(); i++) {
            Spline<double> product(grid);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                product(ir) = cls.radial_function(ir, i) * response.radial_functions(ir, i, 0);
                auto j      = ir + grid.num_points() * i;
                unchanged   = std::max(unchanged, std::abs(cls.radial_function(ir, i) - baseline.first[j]));
            }
            norm_error = std::max(norm_error, std::abs(product.interpolate().integrate(2)));
            if (i >= type.mt_radial_basis_size() - type.num_lo_descriptors()) {
                boundary_error =
                        std::max(boundary_error, std::abs(response.radial_functions(grid.num_points() - 1, i, 0)));
            }
        }
        std::cout << "Radial basis response core=" << core << " orthogonalize=" << orthogonalize << " part=" << part
                  << " auto_enu=" << auto_enu << " relativity=" << relativity << " core_relativity=" << core_relativity
                  << '\n';
        check("unchanged stored radial basis", unchanged, 0);
        check("unchanged stored linearisation energies", baseline_energies == energies() ? 0 : 1, 0);
        check("unchanged stored radial fields",
              cls.spherical_potential() == potential && cls.spherical_xc_derivatives() == fields ? 0 : 1, 0);
        check("zero basis response", null_error, 0);
        check("zero core eigenvalue-sum response", std::abs(zero.core_eval_sum), 0);
        check("basis normalization response", norm_error, 2e-9);
        check("LO boundary response", boundary_error, 2e-10);
        mdarray<std::complex<double>, 2> dalm({gkvec.count(), type.mt_aw_basis_size()});
        mdarray<std::complex<double>, 2> conjugate({gkvec.count(), type.mt_aw_basis_size()});
        matching.generate_potential_response<false>(atom, response, dalm, cls);
        matching.generate_potential_response<true>(atom, response, conjugate, cls);
        double conjugation_error{0};
        for (size_t i = 0; i < dalm.size(); i++) {
            conjugation_error = std::max(conjugation_error, std::abs(dalm[i] - std::conj(conjugate[i])));
        }
        cell.comm().allreduce<double, mpi::op_t::max>(&conjugation_error, 1);
        check("conjugated matching response", conjugation_error, 0);
        double step = auto_enu ? 1e-2 : (core ? 1e-3 : 1e-4);
        for (double h : {step, step / 2}) {
            auto plus = sample(h), minus = sample(-h);
            auto [error, peak] = derivative_error(analytic, plus.first, minus.first, h);
            check("nonzero radial basis response", peak > 1e-5 ? 0 : 1, 0);
            check("AW/LO radial-field derivative", error / peak, 2e-6);
            std::tie(error, peak) = derivative_error(dalm, plus.second, minus.second, h);
            cell.comm().allreduce<double, mpi::op_t::max>(&error, 1);
            cell.comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            check("nonzero matching response", peak > 1e-5 ? 0 : 1, 0);
            check("APW/LAPW matching radial-field derivative", error / peak, 2e-6);
        }
        for (auto invalid : {std::vector<double>{},
                             std::vector<double>(grid.num_points(), std::numeric_limits<double>::quiet_NaN())}) {
            check.reject("invalid radial basis direction", [&] { cls.radial_basis_potential_response(invalid); });
        }
        for (int j : {0, 1}) {
            for (auto invalid : {std::vector<double>(1, 0),
                                 std::vector<double>(grid.num_points(), std::numeric_limits<double>::quiet_NaN())}) {
                std::array<std::vector<double>, 2> bad;
                bad[j] = invalid;
                check.reject("invalid radial basis XC direction",
                             [&] { cls.radial_basis_potential_response(direction, bad); });
            }
        }
        if (core && part) {
            cls.set_spherical_xc_derivatives({});
            check.reject("uninitialized core projection response",
                         [&] { cls.radial_basis_potential_response(direction, xc_direction); });
        }
        parameters.valence_relativity("dirac");
        check.reject("unsupported relativistic basis response",
                     [&] { cls.radial_basis_potential_response(direction); });
        if (core) {
            parameters.valence_relativity("none");
            parameters.core_relativity("iora");
            cls.set_spherical_xc_derivatives(fields);
            check.reject("unsupported relativistic core projection response",
                         [&] { cls.radial_basis_potential_response(direction); });
        }
    }
    return failures;
}

int
test_dirac_radial_basis()
{
    int failures{0};
    Test_check check(failures);
    for (r3::vector<double> q :
         {r3::vector<double>{0, 0, 0}, r3::vector<double>{0.3, -0.4, 0.8}, r3::vector<double>{70, -90, 130}}) {
        Dirac_plane_wave pw(q);
        auto u = pw.spinors();
        double norm_error{0}, equation_error{0};
        std::complex<double> imaginary(0, 1);
        for (int s : {0, 1}) {
            std::array<std::complex<double>, 4> action{
                    speed_of_light * (q[2] * u[s][2] + (q[0] - imaginary * q[1]) * u[s][3]),
                    speed_of_light * ((q[0] + imaginary * q[1]) * u[s][2] - q[2] * u[s][3]),
                    speed_of_light * (q[2] * u[s][0] + (q[0] - imaginary * q[1]) * u[s][1]) -
                            2 * speed_of_light * speed_of_light * u[s][2],
                    speed_of_light * ((q[0] + imaginary * q[1]) * u[s][0] - q[2] * u[s][1]) -
                            2 * speed_of_light * speed_of_light * u[s][3]};
            for (int t : {0, 1}) {
                std::complex<double> norm{0};
                for (int c = 0; c < 4; c++) {
                    norm += std::conj(u[s][c]) * u[t][c];
                    equation_error = std::max(equation_error, std::abs(action[c] - pw.energy() * u[s][c]));
                }
                norm_error = std::max(norm_error, std::abs(norm - (s == t ? 1.0 : 0.0)));
            }
        }
        check("positive-energy Dirac PW orthonormal spinors", norm_error, 3e-15);
        check("free four-component Dirac eigenproblem", equation_error / std::max(1.0, pw.energy()), 3e-14);
        r3::vector<double> dq{0.13, -0.27, 0.19};
        auto du = pw.response(dq);
        for (double h : {1e-4, 5e-5}) {
            auto plus = Dirac_plane_wave(q + dq * h).spinors(), minus = Dirac_plane_wave(q - dq * h).spinors();
            double de = (Dirac_plane_wave(q + dq * h).energy() - Dirac_plane_wave(q - dq * h).energy()) / (2 * h);
            check("Dirac PW energy momentum derivative",
                  std::abs(de - pw.energy_response(dq)) / std::max(1.0, std::abs(pw.energy_response(dq))), 2e-8);
            double error{0}, peak{0};
            for (int s : {0, 1}) {
                for (int c = 0; c < 4; c++) {
                    error = std::max(error, std::abs(du[s][c] - (plus[s][c] - minus[s][c]) / (2 * h)));
                    peak  = std::max(peak, std::abs(du[s][c]));
                }
            }
            check("nonzero Dirac PW momentum derivative", peak > 1e-5 ? 0 : 1, 0);
            check("Dirac PW momentum derivative", error / peak, 2e-8);
        }
    }
    double qsmall = 1e-4;
    check("stable Dirac PW nonrelativistic energy limit",
          std::abs(Dirac_plane_wave({qsmall, 0, 0}).energy() / (0.5 * qsmall * qsmall) - 1), 2e-13);
    check.reject("nonfinite Dirac PW momentum rejected",
                 [&] { Dirac_plane_wave({std::numeric_limits<double>::quiet_NaN(), 0, 0}); });
    check.reject("nonfinite Dirac PW momentum direction rejected",
                 [&] { Dirac_plane_wave({0.1, 0.2, 0.3}).response({0, std::numeric_limits<double>::quiet_NaN(), 0}); });
    for (auto [core, auto_enu, orthogonalize] :
         {std::tuple{false, 0, false}, std::tuple{false, 1, true}, std::tuple{true, 0, true}}) {
        Simulation_parameters parameters;
        parameters.valence_relativity("dirac");
        parameters.core_relativity("dirac");
        parameters.lmax_apw(2);
        parameters.lmax_rho(6);
        parameters.lmax_pot(6);
        parameters.cfg().parameters().auto_rmt(0);
        parameters.cfg().parameters().use_symmetry(false);
        parameters.cfg().control().ortho_rf(orthogonalize);
        Unit_cell cell(parameters, mpi::Communicator::world());
        cell.set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
        auto& type = cell.add_atom_type("Ne");
        type.zn(10);
        type.set_radial_grid(radial_grid_t::power, core ? 64 : 801, 1e-6, 1.8, 2.0);
        if (core) {
            type.set_configuration(1, 0, 1, 2.0, true);
        }
        for (int l = 0; l <= 2; l++) {
            for (int order = 0; order < (l == 0 ? 3 : (l == 2 ? 1 : 2)); order++) {
                type.add_aw_descriptor(std::max(2, l + 1), l, -0.3, order, auto_enu);
            }
        }
        for (int l : {0, 1}) {
            for (int lo : {0, 1}) {
                for (int order : {0, 1}) {
                    type.add_lo_descriptor(2 * l + lo, std::max(2, l + 1), l, lo ? 0.8 : -0.7, order,
                                           lo ? 0 : auto_enu);
                }
            }
        }
        cell.add_atom("Ne", {0.23, 0.31, 0.43});
        cell.initialize();
        fft::Gvec gkvec({0.07, -0.03, 0.02}, cell.reciprocal_lattice_vectors(), 2.0, cell.comm(), false);
        Matching_coefficients matching(cell, gkvec);
        auto const& grid = type.radial_grid();
        int nr = grid.num_points(), nrf = type.mt_radial_basis_size();
        check("full-j LAPW basis index", type.indexb().full_j() ? 0 : 1, 0);
        check("paired Dirac radial basis size", nrf == 17 && type.num_lo_descriptors() == 6 ? 0 : 1, 0);
        check("two-component AW completion", type.aw_order(2) == 2 ? 0 : 1, 0);
        for (int i = 0; i < type.num_lo_descriptors(); i++) {
            check("three Dirac LO boundary constraints", type.lo_descriptor(i).rsd_set.size() == 3 ? 0 : 1, 0);
        }
        std::vector<double> potential(nr), direction(nr);
        std::array<std::vector<double>, 2> fields, xc_direction;
        if (core) {
            fields = xc_direction = {radial_weights(grid), radial_weights(grid)};
        }
        for (int ir = 0; ir < nr; ir++) {
            double r      = grid[ir];
            potential[ir] = -10 / r + 0.2 * std::exp(-r);
            direction[ir] = 0.13 * std::exp(-0.3 * r) + 0.04 * std::sin(r);
            if (core) {
                fields[0][ir] *= 0.03 * std::exp(-r);
                fields[1][ir] *= 0.002 * std::exp(-r);
                xc_direction[0][ir] *= 0.02 * std::sin(3 * r) * std::exp(-r);
                xc_direction[1][ir] *= 0.001 * std::sin(3 * r) * std::exp(-r);
            }
        }
        LAPW_radial_basis radial_basis(cell, parameters.valence_relativity(), {potential}, {fields});
        auto& cls   = radial_basis.radial_basis_for_class(0);
        auto sample = [&](double h) {
            auto v  = potential;
            auto xc = fields;
            for (int ir = 0; ir < nr; ir++) {
                v[ir] += h * direction[ir];
                for (int c : {0, 1}) {
                    if (core) {
                        xc[c][ir] += h * xc_direction[c][ir];
                    }
                }
            }
            cls.set_spherical_potential(v);
            cls.set_spherical_xc_derivatives(xc);
            radial_basis.generate_radial_functions();
            std::vector<double> values;
            for (int c : {0, 1}) {
                for (int field : {0, 1}) {
                    for (int i = 0; i < nrf; i++) {
                        for (int ir = 0; ir < nr; ir++) {
                            values.push_back(field ? cls.radial_function_derivative(ir, i, c)
                                                   : cls.radial_function(ir, i, c));
                        }
                    }
                }
            }
            for (auto const& rf : type.indexr()) {
                if (int(rf.idxlo) < 0) {
                    for (int c : {0, 1}) {
                        for (int dm : {0, 1, 2}) {
                            values.push_back(cls.aw_surface_deriv(rf.am, rf.order, dm, c));
                        }
                    }
                }
            }
            mdarray<std::complex<double>, 3> alm({gkvec.count(), type.mt_aw_basis_size(), 4});
            matching.generate<false>(cell.atom(0), alm, cls);
            return std::pair{std::move(values), std::move(alm)};
        };
        auto baseline = sample(0);
        if (auto_enu) {
            auto splitting = std::abs(cls.aw_descriptor(angular_momentum(1, -1))[0].enu -
                                      cls.aw_descriptor(angular_momentum(1, 1))[0].enu);
            check("distinct j-resolved linearisation energies", splitting > 1e-6 ? 0 : 1, 0);
        }
        std::vector<radial_basis_response_t> responses;
        responses.push_back(cls.radial_basis_potential_response(direction, xc_direction));
        auto const& response = responses.front();
        Dirac_lapw_basis global_basis(cell, gkvec, radial_basis);
        auto metric       = global_basis.overlap();
        auto dmetric      = global_basis.overlap(&responses);
        auto hamiltonian  = global_basis.hamiltonian();
        auto dhamiltonian = global_basis.hamiltonian(&responses, {direction});
        double metric_hermitian_error{0}, metric_peak{0}, cross_spin{0}, lo_norm{0};
        int ng = gkvec.num_gvec();
        for (int j = 0; j < global_basis.size(); j++) {
            for (int i = 0; i < global_basis.size(); i++) {
                metric_hermitian_error =
                        std::max(metric_hermitian_error, std::abs(metric(i, j) - std::conj(metric(j, i))));
                metric_peak = std::max(metric_peak, std::abs(metric(i, j)));
                if (i < ng && j >= ng && j < 2 * ng) {
                    cross_spin = std::max(cross_spin, std::abs(metric(i, j)));
                }
            }
        }
        for (int lo = 0; lo < type.mt_lo_basis_size(); lo++) {
            lo_norm = std::max(lo_norm, std::abs(metric(2 * ng + lo, 2 * ng + lo) - 1.0));
        }
        auto positive_metric = global_basis.overlap();
        check("Dirac global physical metric Hermitian", metric_hermitian_error / metric_peak, 2e-13);
        check("Dirac global physical metric positive definite",
              la::wrap(la::lib_t::lapack)
                              .potrf(global_basis.size(), positive_metric.at(memory_t::host), positive_metric.ld()) !=
                      0,
              0);
        check("full-j local orbitals counted once", global_basis.size() == 2 * ng + type.mt_lo_basis_size() ? 0 : 1, 0);
        check("Dirac global LO norm", lo_norm, 2e-12);
        check("Dirac global metric retains inter-spin coupling", cross_spin > 1e-7 ? 0 : 1, 0);
        double h_error{0}, h_peak{0};
        for (int j = 0; j < global_basis.size(); j++) {
            for (int i = 0; i < global_basis.size(); i++) {
                h_error = std::max(h_error, std::abs(hamiltonian(i, j) - std::conj(hamiltonian(j, i))));
                h_peak  = std::max(h_peak, std::abs(hamiltonian(i, j)));
            }
        }
        check("global four-component Dirac Hamiltonian Hermitian", h_error / h_peak, 2e-13);
        std::vector<radial_basis_response_t> missing;
        check.reject("collective incomplete Dirac metric response rejected",
                     [&] { global_basis.overlap(cell.comm().rank() == 0 ? &missing : &responses); });
        check.reject("Dirac Hamiltonian direction requires a radial response",
                     [&] { global_basis.hamiltonian(nullptr, {direction}); });
        if (cell.comm().size() > 1) {
            check.reject("collective Dirac Hamiltonian direction mode", [&] {
                global_basis.hamiltonian(&responses, cell.comm().rank() == 0
                                                             ? std::vector<std::vector<double>>{}
                                                             : std::vector<std::vector<double>>{direction});
            });
        }
        fft::Gvec field_gvec(cell.reciprocal_lattice_vectors(), 4.1, cell.comm(), false);
        mdarray<std::complex<double>, 2> rho_pw({field_gvec.num_gvec(), 4}), tau_pw({field_gvec.num_gvec(), 4});
        mdarray<std::complex<double>, 2> field_matrix, dfield_matrix;
        std::array<std::vector<Flm>, 2> covectors;
        auto local_operator = [&](radial_basis_response_t const* tangent = nullptr) {
            LAPW_xc_atom op(cell.atom(0), cls, 6);
            std::vector<mdarray<std::complex<double>, 2>> result;
            result.push_back(mdarray<std::complex<double>, 2>({type.mt_basis_size(), type.mt_basis_size()}));
            result[0].zero();
            for (int s = 0; s < 4; s++) {
                auto matrix = op.matrix_elements(covectors[0][s], covectors[1][s], tangent, s);
                for (size_t i = 0; i < matrix.size(); i++) {
                    result[0][i] += matrix[i];
                }
            }
            return result;
        };
        if (!core && !auto_enu) {
            for (int part : {0, 1}) {
                for (int s = 0; s < 4; s++) {
                    covectors[part].emplace_back(sf::lmmax(6), grid);
                    auto& v = covectors[part].back();
                    v.zero();
                    for (int ir : {nr / 4, nr / 2, 3 * nr / 4}) {
                        for (int lm : {0, 2, 6}) {
                            v(lm, ir) = (part ? 0.009 : 0.07) * std::cos(s + 0.3 * lm + grid[ir]);
                        }
                    }
                }
            }
            std::array<r3::vector<double>, 2> targets{{{0.21, 0.4, 0.17}, {4.8, 5.1, 6.3}}};
            auto point_covector = [](int part, int p, int s) {
                return (part ? 0.033 : 0.19) * std::cos(0.7 * p + 0.9 * s + 0.3 * part);
            };
            for (int ig = 0; ig < field_gvec.num_gvec(); ig++) {
                auto g = field_gvec.gvec_cart(gvec_index_t::global(ig));
                for (int s = 0; s < 4; s++) {
                    rho_pw(ig, s) = tau_pw(ig, s) = 0;
                    for (int p = 0; p < 2; p++) {
                        auto phase = std::exp(std::complex<double>(0, -dot(g, targets[p]))) / cell.omega();
                        rho_pw(ig, s) += point_covector(0, p, s) * phase;
                        tau_pw(ig, s) += point_covector(1, p, s) * phase;
                    }
                }
            }
            auto local = local_operator(), dlocal = local_operator(&response);
            field_matrix  = global_basis.field_operator(field_gvec, rho_pw, tau_pw, local);
            dfield_matrix = global_basis.field_operator(field_gvec, rho_pw, tau_pw, local, &responses, &dlocal);
            check.reject("Dirac fields require matching local response",
                         [&] { global_basis.field_operator(field_gvec, rho_pw, tau_pw, local, &responses); });
            mdarray<std::complex<double>, 2> short_rho({ng, 4}), short_tau({ng, 4});
            short_rho.zero();
            short_tau.zero();
            check.reject("Dirac fields reject insufficient Fourier coverage",
                         [&] { global_basis.field_operator(gkvec, short_rho, short_tau, local); });
            rho_pw(0, 0) += std::complex<double>(0, 1);
            check.reject("Dirac fields reject complex scalar covectors",
                         [&] { global_basis.field_operator(field_gvec, rho_pw, tau_pw, local); });
            rho_pw(0, 0) -= std::complex<double>(0, 1);
            auto saved = local[0](0, 1);
            local[0](0, 1) += 1;
            check.reject("Dirac fields reject non-Hermitian atom matrices",
                         [&] { global_basis.field_operator(field_gvec, rho_pw, tau_pw, local); });
            local[0](0, 1) = saved;
            std::vector<std::complex<double>> orbital(global_basis.size());
            orbital[0]      = {0.4, -0.2};
            orbital[ng + 1] = {-0.3, 0.17};
            orbital[2 * ng] = {0.21, -0.13};
            mdarray<std::complex<double>, 3> dm({type.mt_basis_size(), type.mt_basis_size(), 1});
            std::vector<std::complex<double>> mt(type.mt_basis_size());
            auto const& a = global_basis.atom_coefficients(0);
            for (int xi = 0; xi < type.mt_basis_size(); xi++) {
                for (int i = 0; i < global_basis.size(); i++) {
                    mt[xi] += a(xi, i) * orbital[i];
                }
            }
            for (int j = 0; j < type.mt_basis_size(); j++) {
                for (int i = 0; i < type.mt_basis_size(); i++) {
                    dm(i, j, 0) = std::conj(mt[i]) * mt[j];
                }
            }
            auto fields     = LAPW_xc_atom(cell.atom(0), cls, 6).fields(dm);
            auto const& rho = fields[0][0];
            Spline<double> mt_charge(grid), sphere_pw_charge(grid), sphere_pw_kinetic(grid);
            SHT angular(device_t::CPU, 10);
            auto position = cell.get_cartesian_coordinates(cell.atom(0).position());
            for (int ir = 0; ir < nr; ir++) {
                mt_charge(ir)         = fourpi * y00 * rho(0, ir);
                sphere_pw_charge(ir)  = 0;
                sphere_pw_kinetic(ir) = 0;
                for (int tp = 0; tp < angular.num_points(); tp++) {
                    auto point = position + angular.coord(tp) * grid[ir];
                    std::array<std::complex<double>, 4> psi{};
                    std::array<std::complex<double>, 4> action{};
                    for (int i : {0, ng + 1}) {
                        auto q     = gkvec.gkvec_cart(gvec_index_t::global(i % ng));
                        auto u     = Dirac_plane_wave(q).spinors()[i / ng];
                        auto phase = std::exp(std::complex<double>(0, dot(q, point))) / std::sqrt(cell.omega());
                        for (int c = 0; c < 4; c++) {
                            psi[c] += orbital[i] * phase * u[c];
                        }
                        std::complex<double> imaginary(0, 1);
                        std::array<std::complex<double>, 4> hu{
                                speed_of_light * (q[2] * u[2] + (q[0] - imaginary * q[1]) * u[3]),
                                speed_of_light * ((q[0] + imaginary * q[1]) * u[2] - q[2] * u[3]),
                                speed_of_light * (q[2] * u[0] + (q[0] - imaginary * q[1]) * u[1]) -
                                        2 * speed_of_light * speed_of_light * u[2],
                                speed_of_light * ((q[0] + imaginary * q[1]) * u[0] - q[2] * u[1]) -
                                        2 * speed_of_light * speed_of_light * u[3]};
                        for (int c = 0; c < 4; c++) {
                            action[c] += orbital[i] * phase * hu[c];
                        }
                    }
                    for (auto value : psi) {
                        sphere_pw_charge(ir) += fourpi * angular.weight(tp) * std::norm(value);
                    }
                    for (int c = 0; c < 4; c++) {
                        sphere_pw_kinetic(ir) += fourpi * angular.weight(tp) * (std::conj(psi[c]) * action[c]).real();
                    }
                }
            }
            double reference = std::norm(orbital[0]) + std::norm(orbital[ng + 1]) -
                               sphere_pw_charge.interpolate().integrate(2) + mt_charge.interpolate().integrate(2);
            std::complex<double> norm{0};
            for (int j = 0; j < global_basis.size(); j++) {
                for (int i = 0; i < global_basis.size(); i++) {
                    norm += std::conj(orbital[i]) * metric(i, j) * orbital[j];
                }
            }
            check("global Dirac metric matches integrated four-component charge", std::abs(norm - reference), 2e-10);
            auto radial_matrices = cls.dirac_spherical_matrices();
            auto const& radial_h = radial_matrices[0];
            std::complex<double> mt_energy{0}, matrix_energy{0}, field_energy{0};
            for (int j = 0; j < type.mt_basis_size(); j++) {
                for (int i = 0; i < type.mt_basis_size(); i++) {
                    auto const& a = type.indexb(i);
                    auto const& b = type.indexb(j);
                    if (a.am == b.am && a.two_mj == b.two_mj) {
                        mt_energy += std::conj(mt[i]) * radial_h(a.idxrf, b.idxrf) * mt[j];
                    }
                }
            }
            for (int j = 0; j < global_basis.size(); j++) {
                for (int i = 0; i < global_basis.size(); i++) {
                    matrix_energy += std::conj(orbital[i]) * hamiltonian(i, j) * orbital[j];
                    field_energy += std::conj(orbital[i]) * field_matrix(i, j) * orbital[j];
                }
            }
            double energy_reference = mt_energy.real() - sphere_pw_kinetic.interpolate().integrate(2);
            for (int i : {0, ng + 1}) {
                energy_reference += std::norm(orbital[i]) *
                                    Dirac_plane_wave(gkvec.gkvec_cart(gvec_index_t::global(i % ng))).energy();
            }
            check("global Dirac energy matches Cartesian alpha.p integration",
                  std::abs(matrix_energy - energy_reference), 2e-10);
            double field_reference{0};
            for (int part : {0, 1}) {
                for (int s = 0; s < 4; s++) {
                    for (size_t i = 0; i < covectors[part][s].size(); i++) {
                        field_reference += covectors[part][s][i] * fields[part][s][i];
                    }
                }
            }
            for (int p = 0; p < 2; p++) {
                std::array<std::array<std::complex<double>, 4>, 4> values{};
                for (int i : {0, ng + 1}) {
                    auto q     = gkvec.gkvec_cart(gvec_index_t::global(i % ng));
                    auto u     = Dirac_plane_wave(q).spinors()[i / ng];
                    auto phase = std::exp(std::complex<double>(0, dot(q, targets[p]))) / std::sqrt(cell.omega());
                    for (int c = 0; c < 4; c++) {
                        auto value = orbital[i] * phase * u[c];
                        values[0][c] += value;
                        for (int x = 0; x < 3; x++) {
                            values[x + 1][c] += std::complex<double>(0, q[x]) * value;
                        }
                    }
                }
                for (int part : {0, 1}) {
                    for (int d = part ? 1 : 0; d < (part ? 4 : 1); d++) {
                        std::array<double, 4> observable{};
                        for (int block : {0, 1}) {
                            auto up = values[d][2 * block], down = values[d][2 * block + 1];
                            double sign = block ? -1.0 : 1.0;
                            observable[0] += std::norm(up) + std::norm(down);
                            observable[1] += sign * (std::norm(up) - std::norm(down));
                            observable[2] += sign * 2 * (std::conj(up) * down).real();
                            observable[3] += sign * 2 * (std::conj(up) * down).imag();
                        }
                        for (int s = 0; s < 4; s++) {
                            field_reference += (part ? 0.5 : 1.0) * point_covector(part, p, s) * observable[s];
                        }
                    }
                }
            }
            check("Dirac scalar/magnetic/tau operator matches raw point energy",
                  std::abs(field_energy - field_reference), 2e-12);
            mdarray<std::complex<double>, 2> constant_rho({field_gvec.num_gvec(), 1}),
                    constant_tau({field_gvec.num_gvec(), 1});
            constant_tau.zero();
            double shift = 0.37;
            for (int ig = 0; ig < field_gvec.num_gvec(); ig++) {
                auto g = field_gvec.gvec_cart(gvec_index_t::global(ig));
                constant_rho(ig, 0) =
                        shift * ((g.length() < 1e-12 ? 1.0 : 0.0) -
                                 fourpi / cell.omega() * unit_step_function_form_factors(type.mt_radius(), g.length()) *
                                         std::exp(std::complex<double>(0, -dot(g, position))));
            }
            auto radial_overlap = cls.dirac_spherical_matrices();
            for (int j = 0; j < type.mt_basis_size(); j++) {
                for (int i = 0; i < type.mt_basis_size(); i++) {
                    auto const& a = type.indexb(i);
                    auto const& b = type.indexb(j);
                    local[0](i, j) =
                            a.am == b.am && a.two_mj == b.two_mj ? shift * radial_overlap[1](a.idxrf, b.idxrf) : 0;
                }
            }
            auto constant_operator = global_basis.field_operator(field_gvec, constant_rho, constant_tau, local);
            double shift_error{0};
            for (size_t i = 0; i < metric.size(); i++) {
                shift_error = std::max(shift_error, std::abs(constant_operator[i] - shift * metric[i]));
            }
            check("constant Dirac scalar potential equals shift times physical metric", shift_error, 2e-13);
            int n = global_basis.size();
            la::dmatrix<std::complex<double>> eigen_h(n, n), eigen_o(n, n), vectors(n, n);
            std::vector<double> eigenvalues(n);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    eigen_h(i, j) = hamiltonian(i, j) + field_matrix(i, j);
                    eigen_o(i, j) = metric(i, j);
                }
            }
            int info = la::Eigensolver_lapack().solve(n, n, eigen_h, eigen_o, eigenvalues.data(), vectors);
            check("global Dirac generalized eigenproblem", info, 0);
            if (!info) {
                double residual{0}, norm_error{0};
                for (int b = 0; b < 8; b++) {
                    std::complex<double> norm{0};
                    for (int i = 0; i < n; i++) {
                        std::complex<double> h{0}, o{0};
                        for (int j = 0; j < n; j++) {
                            h += (hamiltonian(i, j) + field_matrix(i, j)) * vectors(j, b);
                            o += metric(i, j) * vectors(j, b);
                        }
                        norm += std::conj(vectors(i, b)) * o;
                        residual = std::max(residual, std::abs(h - eigenvalues[b] * o));
                    }
                    norm_error = std::max(norm_error, std::abs(norm - 1.0));
                }
                check("Dirac generalized eigenvectors have physical norm", norm_error, 2e-11);
                check("Dirac generalized eigenproblem residual", residual, 2e-10);
            }
        }
        auto zero       = cls.radial_basis_potential_response(std::vector<double>(nr, 0));
        auto spherical  = cls.dirac_spherical_matrices();
        auto dspherical = cls.dirac_spherical_matrices(&response, direction);
        auto zspherical = cls.dirac_spherical_matrices(&zero);
        double spherical_norm_error{0}, spherical_hermitian_error{0}, channel_error{0}, spherical_null_error{0};
        double equation_error{0};
        for (auto const& a : type.indexr()) {
            int i                = int(a.idxrf);
            spherical_norm_error = std::max(spherical_norm_error, std::abs(spherical[1](i, i) - 1));
            if (!core && int(a.idxlo) < 0 && a.order == 0) {
                equation_error =
                        std::max(equation_error, std::abs(spherical[0](i, i) - cls.aw_descriptor(a.am)[0].enu));
            }
            for (auto const& b : type.indexr()) {
                int j = int(b.idxrf);
                for (int part : {0, 1}) {
                    spherical_hermitian_error = std::max(spherical_hermitian_error,
                                                         std::abs(spherical[part](i, j) - spherical[part](j, i)));
                    spherical_null_error      = std::max(spherical_null_error, std::abs(zspherical[part](i, j)));
                    if (a.am != b.am) {
                        channel_error = std::max(
                                {channel_error, std::abs(spherical[part](i, j)), std::abs(dspherical[part](i, j))});
                    }
                }
            }
        }
        check("Dirac spherical physical overlap normalization", spherical_norm_error, 2e-12);
        check("Dirac spherical H/O Hermitian", spherical_hermitian_error, 0);
        check("Dirac spherical kappa channels remain separate", channel_error, 0);
        check("zero Dirac spherical H/O response", spherical_null_error, 0);
        if (!core) {
            check("Dirac spherical Hamiltonian satisfies the radial equation", equation_error, 2e-10);
        }
        std::vector<double> analytic(response.radial_functions.begin(), response.radial_functions.end());
        analytic.insert(analytic.end(), response.surface_derivatives.begin(), response.surface_derivatives.end());
        check("Dirac radial response layout", baseline.first.size() == analytic.size() ? 0 : 1, 0);
        mdarray<std::complex<double>, 3> dalm({gkvec.count(), type.mt_aw_basis_size(), 4});
        mdarray<std::complex<double>, 3> conjugated({gkvec.count(), type.mt_aw_basis_size(), 4});
        matching.generate_potential_response<false>(cell.atom(0), response, dalm, cls);
        double matching_conjugation_error{0};
        for (bool derivative : {false, true}) {
            if (derivative) {
                matching.generate_potential_response<true>(cell.atom(0), response, conjugated, cls);
            } else {
                matching.generate<true>(cell.atom(0), conjugated, cls);
            }
            auto const& original = derivative ? dalm : baseline.second;
            for (size_t i = 0; i < original.size(); i++) {
                matching_conjugation_error =
                        std::max(matching_conjugation_error, std::abs(conjugated[i] - std::conj(original[i])));
            }
        }
        double matching_boundary_error{0};
        for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
            auto const& basis = type.indexb(xi);
            if (basis.order != 0) {
                continue;
            }
            int n = type.aw_order(basis.am.l());
            for (int igk = 0; igk < gkvec.count(); igk++) {
                auto q         = gkvec.gkvec_cart(gvec_index_t::local(igk));
                auto spherical = r3::spherical_coordinates(q);
                std::vector<std::complex<double>> ylm(sf::lmmax(type.lmax_apw() + 1));
                sf::spherical_harmonics(type.lmax_apw() + 1, spherical[1], spherical[2], ylm.data());
                std::vector<double> bessel(type.lmax_apw() + 3);
                gsl_sf_bessel_jl_array(type.lmax_apw() + 2, spherical[0] * type.mt_radius(), bessel.data());
                auto phase = std::exp(
                        std::complex<double>(0, dot(q, cell.get_cartesian_coordinates(cell.atom(0).position()))));
                for (int row = 0; row < n; row++) {
                    int component = row == 1 ? 1 : 0, dm = row == 2 ? 1 : 0;
                    int l         = component ? basis.am.two_j() - basis.am.l() : basis.am.l();
                    double radial = dm ? l / type.mt_radius() * bessel[l] - spherical[0] * bessel[l + 1] : bessel[l];
                    for (int c = 0; c < 4; c++) {
                        std::complex<double> actual{0}, expected{0};
                        for (int order = 0; order < n; order++) {
                            int index = type.indexb().index_by_j_mj_order(basis.am, basis.two_mj, order);
                            actual += cls.aw_surface_deriv(basis.am, order, dm, component) *
                                      baseline.second(igk, index, c);
                        }
                        if (component) {
                            actual *= std::complex<double>(0, -1);
                        }
                        int m = (basis.two_mj + 2 * (c % 2) - 1) / 2;
                        if (c / 2 == component && std::abs(m) <= l) {
                            expected = fourpi / std::sqrt(cell.omega()) * phase *
                                       std::pow(std::complex<double>(0, 1), l) * radial * std::conj(ylm[sf::lm(l, m)]) *
                                       sht::ClebschGordan(l, basis.am.j(), basis.two_mj / 2.0, c % 2);
                        }
                        matching_boundary_error = std::max(matching_boundary_error, std::abs(actual - expected));
                    }
                }
            }
        }
        check("Dirac both-component plane-wave matching", matching_boundary_error, 2e-10);
        check("Dirac matching conjugation", matching_conjugation_error, 1e-14);
        matching.generate_potential_response<false>(cell.atom(0), zero, conjugated, cls);
        double matching_null_error{0};
        for (auto z : conjugated) {
            matching_null_error = std::max(matching_null_error, std::abs(z));
        }
        check("zero Dirac matching response", matching_null_error, 0);
        mdarray<std::complex<double>, 2> scalar_alm({gkvec.count(), type.mt_aw_basis_size()});
        check.reject("Dirac scalar matching rejected",
                     [&] { matching.generate<false>(cell.atom(0), scalar_alm, cls); });
        double norm_error{0}, tangent_error{0}, boundary_error{0}, null_error{0}, small_norm{0}, core_overlap{0};
        lapw_radial_basis_t::core_states_t cores;
        if (core) {
            auto [extended_grid, extended_potential] = type.core_radial_problem(potential);
            cores = cls.solve_occupied_core_states(extended_grid, extended_potential, fields, relativity_t::dirac);
        }
        Spline<double> product(grid);
        for (auto const& rf : type.indexr()) {
            int i = int(rf.idxrf);
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = std::pow(cls.radial_function(ir, i), 2) + std::pow(cls.radial_function(ir, i, 1), 2);
            }
            norm_error = std::max(norm_error, std::abs(product.interpolate().integrate(2) - 1));
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = cls.radial_function(ir, i) * response.radial_functions(ir, i, 0) +
                              cls.radial_function(ir, i, 1) * response.radial_functions(ir, i, 2);
            }
            tangent_error = std::max(tangent_error, std::abs(product.interpolate().integrate(2)));
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = std::pow(cls.radial_function(ir, i, 1), 2);
            }
            small_norm = std::max(small_norm, product.interpolate().integrate(2));
            if (int(rf.idxlo) >= 0) {
                for (int c : {0, 1}) {
                    boundary_error = std::max({boundary_error, std::abs(cls.radial_function(nr - 1, i, c)),
                                               std::abs(response.radial_functions(nr - 1, i, 2 * c))});
                }
            }
            if (cores.count(rf.am.kappa())) {
                auto const& state = cores.at(rf.am.kappa()).front();
                for (int ir = 0; ir < nr; ir++) {
                    product(ir) = state.value[ir] * cls.radial_function(ir, i) +
                                  state.small_value[ir] * cls.radial_function(ir, i, 1);
                }
                core_overlap = std::max(core_overlap, std::abs(product.interpolate().integrate(2)));
            }
        }
        for (double x : zero.radial_functions) {
            null_error = std::max(null_error, std::abs(x));
        }
        for (double x : zero.surface_derivatives) {
            null_error = std::max(null_error, std::abs(x));
        }
        std::cout << "Dirac LAPW radial basis core=" << core << " auto_enu=" << auto_enu
                  << " orthogonalize=" << orthogonalize << '\n';
        check("joint Dirac radial normalization", norm_error, 2e-12);
        check("joint Dirac normalization response", tangent_error, 2e-10);
        check("both Dirac LO components confined", boundary_error, 1e-10);
        check("retained Dirac lower component", small_norm > 1e-6 ? 0 : 1, 0);
        check("Dirac physical-metric core orthogonality", core_overlap, 2e-10);
        check("zero Dirac basis response", null_error, 0);
        for (double h : {1e-3, 5e-4}) {
            auto plus           = sample(h);
            auto plus_spherical = cls.dirac_spherical_matrices();
            auto plus_metric    = Dirac_lapw_basis(cell, gkvec, radial_basis).overlap();
            auto plus_h         = Dirac_lapw_basis(cell, gkvec, radial_basis).hamiltonian();
            auto plus_field = !core && !auto_enu ? Dirac_lapw_basis(cell, gkvec, radial_basis)
                                                           .field_operator(field_gvec, rho_pw, tau_pw, local_operator())
                                                 : mdarray<std::complex<double>, 2>();
            auto minus      = sample(-h);
            auto minus_spherical = cls.dirac_spherical_matrices();
            auto minus_metric    = Dirac_lapw_basis(cell, gkvec, radial_basis).overlap();
            auto minus_h         = Dirac_lapw_basis(cell, gkvec, radial_basis).hamiltonian();
            auto minus_field     = !core && !auto_enu
                                           ? Dirac_lapw_basis(cell, gkvec, radial_basis)
                                                     .field_operator(field_gvec, rho_pw, tau_pw, local_operator())
                                           : mdarray<std::complex<double>, 2>();
            auto [metric_error, derivative_peak] = derivative_error(dmetric, plus_metric, minus_metric, h);
            check("nonzero global Dirac metric potential derivative", derivative_peak > 1e-8 ? 0 : 1, 0);
            check("global Dirac metric potential derivative", metric_error / derivative_peak, 2e-6);
            for (int part : {0, 1}) {
                if (part && (core || auto_enu)) {
                    continue;
                }
                auto const& analytic     = part ? dfield_matrix : dhamiltonian;
                auto const& plus_matrix  = part ? plus_field : plus_h;
                auto const& minus_matrix = part ? minus_field : minus_h;
                auto [error, peak]       = derivative_error(analytic, plus_matrix, minus_matrix, h);
                check("nonzero global Dirac Hamiltonian potential derivative", peak > 1e-8 ? 0 : 1, 0);
                check(part ? "global Dirac field operator radial derivative"
                           : "global Dirac Hamiltonian potential derivative",
                      error / peak, 2e-6);
            }
            std::array<double, 2> error{}, peak{};
            size_t radial_size = response.radial_functions.size();
            for (size_t i = 0; i < analytic.size(); i++) {
                int c    = i < radial_size ? i / (2 * nr * nrf) : ((i - radial_size) % 6) / 3;
                error[c] = std::max(error[c], std::abs(analytic[i] - (plus.first[i] - minus.first[i]) / (2 * h)));
                peak[c]  = std::max(peak[c], std::abs(analytic[i]));
            }
            for (int c : {0, 1}) {
                check("nonzero Dirac component response", peak[c] > 1e-8 ? 0 : 1, 0);
                check(c ? "Dirac lower-component AW/LO potential derivative"
                        : "Dirac upper-component AW/LO potential derivative",
                      error[c] / peak[c], 2e-6);
            }
            error = peak = {};
            for (int c = 0; c < 4; c++) {
                for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                    for (int igk = 0; igk < gkvec.count(); igk++) {
                        error[c / 2] = std::max(
                                error[c / 2], std::abs(dalm(igk, xi, c) -
                                                       (plus.second(igk, xi, c) - minus.second(igk, xi, c)) / (2 * h)));
                        peak[c / 2] = std::max(peak[c / 2], std::abs(dalm(igk, xi, c)));
                    }
                }
            }
            for (int c : {0, 1}) {
                check("nonzero Dirac matching response", peak[c] > 1e-8 ? 0 : 1, 0);
                check(c ? "Dirac lower-component matching potential derivative"
                        : "Dirac upper-component matching potential derivative",
                      error[c] / peak[c], 2e-6);
            }
            for (int part : {0, 1}) {
                auto [matrix_error, matrix_peak] =
                        derivative_error(dspherical[part], plus_spherical[part], minus_spherical[part], h);
                check("nonzero Dirac spherical H/O response", matrix_peak > 1e-8 ? 0 : 1, 0);
                check(part ? "Dirac spherical overlap potential derivative"
                           : "Dirac spherical Hamiltonian potential derivative",
                      matrix_error / matrix_peak, 2e-6);
            }
        }
        if (!core && !auto_enu) {
            sample(0);
            auto lattice   = cell.lattice_vectors();
            auto position  = cell.atom(0).position();
            auto cartesian = cell.get_cartesian_coordinates(position);
            r3::vector<double> displacement{0.23, -0.37, 0.19};
            r3::matrix<double> strain{{0.07, -0.04, 0}, {0, -0.03, 0}, {0.05, 0, 0.02}};
            for (r3::vector<double> vk : {r3::vector<double>{0, 0, 0}, r3::vector<double>{0.07, -0.03, 0.02}}) {
                fft::Gvec vectors(vk, cell.reciprocal_lattice_vectors(), 2.0, cell.comm(), false);
                Matching_coefficients fixed(cell, vectors);
                mdarray<std::complex<double>, 3> dalm({vectors.count(), type.mt_aw_basis_size(), 4});
                mdarray<std::complex<double>, 3> conjugated({vectors.count(), type.mt_aw_basis_size(), 4});
                Dirac_lapw_basis basis(cell, vectors, radial_basis);
                mdarray<double, 2> atoms({3, cell.num_atoms()});
                for (int x = 0; x < 3; x++) {
                    atoms(x, 0) = displacement[x];
                }
                auto bare_geometry = basis.geometry_response(atoms, strain);
                std::array<mdarray<std::complex<double>, 2>, 3> matrix_geometry{
                        std::move(bare_geometry[0]), std::move(bare_geometry[1]),
                        basis.field_operator(field_gvec, rho_pw, tau_pw, local_operator(), nullptr, nullptr, &atoms,
                                             &strain)};
                std::array<std::array<double, 3>, 2> matrix_error{};
                std::array<double, 3> matrix_peak{}, richardson_error{};
                std::array<mdarray<std::complex<double>, 2>, 3> coarse_fd;
                for (int part = 0; part < 3; part++) {
                    coarse_fd[part] = mdarray<std::complex<double>, 2>({basis.size(), basis.size()});
                    double hermitian_error{0};
                    auto const& matrix = matrix_geometry[part];
                    for (int j = 0; j < basis.size(); j++) {
                        for (int i = 0; i < basis.size(); i++) {
                            matrix_peak[part] = std::max(matrix_peak[part], std::abs(matrix(i, j)));
                            hermitian_error =
                                    std::max(hermitian_error, std::abs(matrix(i, j) - std::conj(matrix(j, i))));
                        }
                    }
                    check("nonzero native Dirac matrix geometry derivative", matrix_peak[part] > 1e-7 ? 0 : 1, 0);
                    check("Hermitian native Dirac matrix geometry derivative", hermitian_error / matrix_peak[part],
                          2e-12);
                }
                mdarray<double, 2> zero_atoms({3, cell.num_atoms()});
                zero_atoms.zero();
                auto zero_matrices = basis.geometry_response(zero_atoms, r3::matrix<double>{});
                double zero_error{0};
                for (auto const& matrix : zero_matrices) {
                    for (auto value : matrix) {
                        zero_error = std::max(zero_error, std::abs(value));
                    }
                }
                check("zero native Dirac H/O geometry direction", zero_error, 0);
                mdarray<double, 2> short_atoms({2, cell.num_atoms()});
                check.reject("incomplete native Dirac atom geometry rejected",
                             [&] { basis.geometry_response(short_atoms, strain); });
                auto bad_strain = strain;
                if (cell.comm().rank() == 0) {
                    bad_strain(0, 0) = std::numeric_limits<double>::quiet_NaN();
                }
                check.reject("nonfinite native Dirac geometry rejected collectively",
                             [&] { basis.geometry_response(atoms, bad_strain); });
                if (cell.comm().size() > 1) {
                    bad_strain = strain;
                    if (cell.comm().rank() == 0) {
                        bad_strain(0, 0) += 0.1;
                    }
                    check.reject("different native Dirac geometry rejected collectively",
                                 [&] { basis.geometry_response(atoms, bad_strain); });
                }
                fixed.generate_geometry_response<false>(cell.atom(0), displacement, strain, dalm, cls);
                fixed.generate_geometry_response<true>(cell.atom(0), displacement, strain, conjugated, cls);
                mdarray<std::complex<double>, 3> positive_dalm({vectors.count(), type.mt_aw_basis_size(), 2});
                mdarray<std::complex<double>, 3> positive_conjugated({vectors.count(), type.mt_aw_basis_size(), 2});
                fixed.generate_geometry_response<false>(cell.atom(0), displacement, strain, positive_dalm, cls);
                fixed.generate_geometry_response<true>(cell.atom(0), displacement, strain, positive_conjugated, cls);
                std::array<double, 2> error{}, peak{};
                double conjugation_error{0}, positive_error{0}, positive_peak{0};
                for (size_t i = 0; i < dalm.size(); i++) {
                    int c             = i / (2 * vectors.count() * type.mt_aw_basis_size());
                    peak[c]           = std::max(peak[c], std::abs(dalm[i]));
                    conjugation_error = std::max(conjugation_error, std::abs(conjugated[i] - std::conj(dalm[i])));
                }
                for (size_t i = 0; i < positive_dalm.size(); i++) {
                    positive_peak = std::max(positive_peak, std::abs(positive_dalm[i]));
                    conjugation_error =
                            std::max(conjugation_error, std::abs(positive_conjugated[i] - std::conj(positive_dalm[i])));
                }
                struct geometry_sample_t
                {
                    mdarray<std::complex<double>, 3> physical, positive;
                    std::array<mdarray<std::complex<double>, 2>, 3> matrices;
                };
                auto evaluate = [&](double h) {
                    auto shifted = lattice + dot(strain, lattice) * h;
                    cell.set_lattice_vectors(shifted);
                    cell.atom(0).set_position(dot(inverse(shifted), cartesian + displacement * h));
                    vectors.lattice_vectors(cell.reciprocal_lattice_vectors());
                    Matching_coefficients changed(cell, vectors);
                    mdarray<std::complex<double>, 3> result({vectors.count(), type.mt_aw_basis_size(), 4});
                    mdarray<std::complex<double>, 3> positive_result({vectors.count(), type.mt_aw_basis_size(), 2});
                    changed.generate<false>(cell.atom(0), result, cls);
                    changed.generate<false>(cell.atom(0), positive_result, cls);
                    field_gvec.lattice_vectors(cell.reciprocal_lattice_vectors());
                    Dirac_lapw_basis shifted_basis(cell, vectors, radial_basis);
                    return geometry_sample_t{
                            std::move(result),
                            std::move(positive_result),
                            {shifted_basis.hamiltonian(), shifted_basis.overlap(),
                             shifted_basis.field_operator(field_gvec, rho_pw, tau_pw, local_operator())}};
                };
                for (double h : {1e-4, 5e-5}) {
                    auto plus = evaluate(h), minus = evaluate(-h);
                    for (size_t i = 0; i < dalm.size(); i++) {
                        int c    = i / (2 * vectors.count() * type.mt_aw_basis_size());
                        error[c] = std::max(error[c],
                                            std::abs((plus.physical[i] - minus.physical[i]) / (2 * h) - dalm[i]));
                    }
                    for (size_t i = 0; i < positive_dalm.size(); i++) {
                        positive_error =
                                std::max(positive_error,
                                         std::abs((plus.positive[i] - minus.positive[i]) / (2 * h) - positive_dalm[i]));
                    }
                }
                // Matrix contractions amplify matching roundoff; extrapolate two larger central stencils.
                int stencil{0};
                for (double h : {1e-3, 5e-4}) {
                    auto plus = evaluate(h), minus = evaluate(-h);
                    for (int part = 0; part < 3; part++) {
                        for (size_t i = 0; i < matrix_geometry[part].size(); i++) {
                            auto fd = (plus.matrices[part][i] - minus.matrices[part][i]) / (2 * h);
                            matrix_error[stencil][part] =
                                    std::max(matrix_error[stencil][part], std::abs(fd - matrix_geometry[part][i]));
                            if (!stencil) {
                                coarse_fd[part][i] = fd;
                            } else {
                                richardson_error[part] = std::max(
                                        richardson_error[part],
                                        std::abs((4.0 * fd - coarse_fd[part][i]) / 3.0 - matrix_geometry[part][i]));
                            }
                        }
                    }
                    stencil++;
                }
                cell.set_lattice_vectors(lattice);
                cell.atom(0).set_position(position);
                vectors.lattice_vectors(cell.reciprocal_lattice_vectors());
                field_gvec.lattice_vectors(cell.reciprocal_lattice_vectors());
                cell.comm().allreduce<double, mpi::op_t::max>(error.data(), 2);
                cell.comm().allreduce<double, mpi::op_t::max>(peak.data(), 2);
                cell.comm().allreduce<double, mpi::op_t::max>(&positive_error, 1);
                cell.comm().allreduce<double, mpi::op_t::max>(&positive_peak, 1);
                for (int c : {0, 1}) {
                    check("nonzero Dirac matching geometry derivative", peak[c] > 1e-7 ? 0 : 1, 0);
                    check(c ? "Dirac lower-component matching atom/cell derivative"
                            : "Dirac upper-component matching atom/cell derivative",
                          error[c] / peak[c], 3e-8);
                }
                check("Dirac matching geometry conjugation", conjugation_error, 1e-14);
                check("nonzero positive-energy Dirac matching geometry derivative", positive_peak > 1e-7 ? 0 : 1, 0);
                check("positive-energy Dirac PW matching atom/cell derivative", positive_error / positive_peak, 3e-8);
                for (int part = 0; part < 3; part++) {
                    check("native Dirac matrix stencil refinement",
                          std::max(0.0, matrix_error[1][part] - 0.5 * matrix_error[0][part]) / matrix_peak[part], 1e-9);
                    check(part == 0   ? "native Dirac Hamiltonian atom/cell derivative"
                          : part == 1 ? "native Dirac physical metric atom/cell derivative"
                                      : "native Dirac scalar/magnetic/tau operator atom/cell derivative",
                          richardson_error[part] / matrix_peak[part], 3e-8);
                }
            }
        }
        check.reject("Dirac scalar-Hamiltonian assembly rejected",
                     [&] { cls.generate_radial_integrals(relativity_t::dirac); });
        check.reject("Dirac spherical potential direction requires a response",
                     [&] { cls.dirac_spherical_matrices(nullptr, direction); });
        auto invalid = direction;
        invalid[0]   = std::numeric_limits<double>::quiet_NaN();
        check.reject("nonfinite Dirac spherical potential direction rejected",
                     [&] { cls.dirac_spherical_matrices(&response, invalid); });
    }
    return failures;
}

int
test_core_reference()
{
    int failures{0};
    Test_check check(failures);
    {
        bspline_basis<4> cubic({0, 0, 0, 0, 1, 1, 1, 1}, 4, true);
        double error{0};
        for (double x : {0.0, 0.25, 0.75, 1.0}) {
            std::array<double, 4> expected{6 * (1 - x), -12 + 18 * x, 6 - 18 * x, 6 * x};
            for (int i = 0; i < 4; i++) {
                error = std::max(error, std::abs(cubic.deriv2(i, x) - expected[i]));
            }
        }
        check("analytic B-spline second derivatives", error, 2e-13);
    }
    {
        // A weak, negative point-tau derivative must not create an origin-localized core state.
        auto inner_grid = Radial_grid_factory<double>(radial_grid_t::exponential, 401, 1e-6, 20.0, 1.0);
        std::vector<double> potential(inner_grid.num_points()), zero(inner_grid.num_points(), 0), dtau;
        Flm field(1, inner_grid);
        double power = 2 * radial_solver_local::scalar_relativistic_origin_power(0, 10) - 4;
        for (int ir = 0; ir < inner_grid.num_points(); ir++) {
            potential[ir] = -10 / inner_grid[ir];
            field(0, ir)  = std::pow(inner_grid[ir], power);
        }
        std::vector<r3::vector<double>> points;
        std::vector<std::array<double, 4>> covectors;
        constexpr double beta{-1e-4};
        for (double fraction : {0.2, 0.7}) {
            double r = fraction * inner_grid.first();
            points.push_back({r, 0, 0});
            covectors.push_back({beta * fourpi * r * r * r, 0, 0, 0});
            check("positive point kinetic coefficient", 1 / (1 + sq_alpha_half * 10 / r) + beta > 0 ? 0 : 1, 0);
        }
        Spheric_function_interpolator sampling(field, spheric_inner_extension_t::kinetic, {power});
        auto projected = sampling.adjoint(points, covectors);
        for (int ir = 0; ir < inner_grid.num_points(); ir++) {
            dtau.push_back(projected(0, ir) / y00);
        }
        Radial_core_solver solver(inner_grid, 0, 10, potential, 0, 0, false, {}, relativity_t::zora);
        auto baseline = solver.solve(1, zero, zero)[0];
        auto state    = solver.solve(1, zero, dtau)[0];
        check("weak inner tau perturbation", std::abs(state.energy - baseline.energy), 1e-6);
    }
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 121, 1e-6, 20.0, 2.0);
    std::vector<double> knots(6, 0);
    knots.insert(knots.end(), {1e-6, 0.01, 0.1, 0.1, 0.1, 0.5});
    knots.insert(knots.end(), 6, grid.last());
    for (bool boundary : {false, true}) {
        for (double power : {1.0, 2.5}) {
            bspline_basis<6> basis(knots, 8, boundary, power);
            bool valid = !basis.basis_pairs().empty();
            for (auto const& pair : basis.basis_pairs()) {
                for (size_t i = 0; i < pair.r.size(); i++) {
                    valid &= pair.Bi[i] == basis(pair.i, pair.r[i]) && pair.Bj[i] == basis(pair.j, pair.r[i]) &&
                             pair.dBi[i] == basis.deriv(pair.i, pair.r[i]) &&
                             pair.dBj[i] == basis.deriv(pair.j, pair.r[i]);
                }
            }
            check("cached B-spline quadrature values", valid ? 0 : 1, 0);
        }
    }
    auto points = grid.values();
    int nmtp    = 81;
    std::vector<double> potential(points.size()), direction(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        potential[i] = -2 / points[i];
        direction[i] = 0.01 * std::exp(-points[i]);
    }
    std::array<std::vector<double>, 2> fields;
    for (auto& field : fields) {
        field.resize(nmtp, 0);
    }
    Simulation_parameters parameters;
    parameters.lmax_apw(0);
    parameters.lmax_rho(0);
    parameters.lmax_pot(0);
    parameters.cfg().parameters().auto_rmt(0);
    parameters.cfg().parameters().use_symmetry(false);
    Unit_cell cell(parameters, mpi::Communicator::world());
    cell.set_lattice_vectors({{40, 0, 0}, {0, 40, 0}, {0, 0, 40}});
    auto& type = cell.add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(nmtp, points.data());
    type.add_aw_descriptor(1, 0, -0.3, 0, 0);
    type.add_aw_descriptor(1, 0, -0.3, 1, 0);
    cell.add_atom("He", {0, 0, 0});
    cell.initialize();
    lapw_radial_basis_t atom_class(0, type);
    radial_core_reference_t reference;
    auto equal = [](auto const& a, auto const& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].energy != b[i].energy || a[i].kinetic_energy != b[i].kinetic_energy ||
                a[i].coefficients != b[i].coefficients || a[i].value != b[i].value ||
                a[i].derivative != b[i].derivative || a[i].rho != b[i].rho || a[i].tau != b[i].tau) {
                return false;
            }
        }
        return true;
    };
    for (int variant = 0; variant < 10; variant++) {
        solve_radial_core_states(grid, 0, 2, potential, 1, fields, nmtp, nullptr, nullptr, {}, relativity_t::none,
                                 &reference);
        auto p = potential, r = points;
        auto f = fields;
        int l = variant == 8, z = variant == 9 ? 3 : 2, count = variant == 7 ? 2 : 1;
        int nr   = nmtp - (variant == 5 ? 2 : 0);
        auto rel = variant == 6 ? relativity_t::zora : relativity_t::none;
        if (variant == 1) {
            p[40] += 0.001;
        }
        if (variant == 2 || variant == 3) {
            f[variant - 2][40] = 1e-5;
        }
        if (variant == 4) {
            r[40] = 0.5 * (r[40] + r[41]);
        }
        if (variant == 9) {
            for (size_t i = 0; i < p.size(); i++) {
                p[i] = -z / r[i];
            }
        }
        for (auto& field : f) {
            field.resize(nr);
        }
        Radial_grid_ext<double> changed(static_cast<int>(r.size()), r.data());
        auto fresh  = solve_radial_core_states(changed, l, z, p, count, f, nr, nullptr, nullptr, {}, rel);
        auto cached = solve_radial_core_states(changed, l, z, p, count, f, nr, nullptr, nullptr, {}, rel, &reference);
        check("exact core reference and invalidation", equal(fresh, cached) ? 0 : 1, 0);
        check("cached core basis", (rel == relativity_t::none ? reference.basis > 6 : reference.basis == 0) ? 0 : 1, 0);
        std::vector<radial_core_state_t> dfresh, dcached;
        solve_radial_core_states(changed, l, z, p, count, f, nr, &direction, &dfresh, {}, rel);
        solve_radial_core_states(changed, l, z, p, count, f, nr, &direction, &dcached, {}, rel, &reference);
        check("exact cached core response", equal(dfresh, dcached) ? 0 : 1, 0);
        if (nr == nmtp && z == type.zn()) {
            auto shared = atom_class.solve_core_states(changed, l, p, count, f, rel);
            check("shared density/basis core reference", equal(fresh, shared) ? 0 : 1, 0);
            std::vector<radial_core_state_t> dshared;
            atom_class.solve_core_states(changed, l, p, count, f, rel, &direction, &dshared);
            check("shared density/basis core response", equal(dfresh, dshared) ? 0 : 1, 0);
        }
    }
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    if (argc == 2 && std::string(argv[1]) == "--scalar-dirac-basis") {
        int result = call_test("test_scalar_dirac_core_basis", test_radial_basis_response, true);
        sirius::finalize();
        return result;
    }
    if (argc == 2 && std::string(argv[1]) == "--dirac-radial") {
        int result = call_test("test_dirac_radial_solution", test_dirac_radial_solution);
        sirius::finalize();
        return result;
    }
    if (argc == 2 && std::string(argv[1]) == "--dirac-basis") {
        int result = call_test("test_dirac_radial_basis", test_dirac_radial_basis);
        sirius::finalize();
        return result;
    }
    if ((argc == 2 || argc == 3) && std::string(argv[1]) == "--dirac-xc") {
        int result{0};
        if (argc == 2) {
            result += call_test("test_dirac_xc_core", test_dirac_xc_core, false, 49);
        }
        int points = argc == 3 ? std::stoi(argv[2]) : 49;
        result += call_test("test_native_dirac_xc_core", test_dirac_xc_core, true, points);
        sirius::finalize();
        return result;
    }
    int result = call_test("test_radial_core_solver", test_radial_core_solver);
    result += call_test("test_dirac_core_reference", test_dirac_core_reference);
    result += call_test("test_dirac_xc_core", test_dirac_xc_core, false, 49);
    result += call_test("test_native_dirac_xc_core", test_dirac_xc_core, true, 49);
    result += call_test("test_core_reference", test_core_reference);
    result += call_test("test_radial_core_response", [] { return test_radial_core_response(); });
    result += call_test("test_zora_core_response", [] { return test_radial_core_response(relativity_t::zora); });
    result += call_test("test_native_zora_core_response",
                        [] { return test_radial_core_response(relativity_t::zora, 0, 801); });
    result += call_test("test_radial_potential_response", test_radial_potential_response);
    result += call_test("test_dirac_radial_solution", test_dirac_radial_solution);
    result += call_test("test_linearization_energy_response", test_linearization_energy_response);
    result += call_test("test_radial_basis_response", test_radial_basis_response, false);
    result += call_test("test_dirac_radial_basis", test_dirac_radial_basis);
    sirius::finalize();
    return result;
}
