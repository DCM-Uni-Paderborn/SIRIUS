/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file radial_solver.cpp
 *
 *  \brief Contains partial implementation of sirius::Radial_solver class.
 */

#include "radial_solver.hpp"
#include <memory>

namespace sirius {

radial_solver_response_t
Radial_solver::solve_dirac_impl(int dme__, int kappa__, double enu__, std::vector<double> const* direction__,
                                double energy_direction__) const
{
    double k = kappa__, zc = zn_ / speed_of_light;
    if (dme__ < 0 || dme__ > 3 || !kappa__ || kappa__ == std::numeric_limits<int>::min() || zn_ < 0 ||
        zc >= std::abs(k) || !std::isfinite(enu__) || !std::isfinite(energy_direction__) ||
        (direction__ &&
         (direction__->size() != static_cast<size_t>(num_points()) ||
          !std::all_of(direction__->begin(), direction__->end(), [](double v) { return std::isfinite(v); })))) {
        RTE_THROW("Dirac radial solution: invalid channel, energy, nucleus or potential direction");
    }
    int orders = dme__ + 1, block = 2 * orders, dimension = block * (direction__ ? 2 : 1);
    Spline<double> direction(radial_grid_);
    if (direction__) {
        direction = Spline<double>(radial_grid_, *direction__);
    }
    struct parameters_t
    {
        Spline<double> const& potential;
        Spline<double> const& direction;
        double energy, de, kappa, zn, x0{0};
        int orders, block, dimension, ir{0};
    } parameters{ve_, direction, enu__, energy_direction__, k, double(zn_), 0, orders, block, dimension};
    auto rhs = [](double r, double const* y, double* f, void* pointer) -> int {
        auto const& a = *static_cast<parameters_t*>(pointer);
        double v      = a.potential(a.ir, r - a.x0) - a.zn / r;
        double b      = (a.energy - v) / speed_of_light;
        double db     = (a.de - a.direction(a.ir, r - a.x0)) / speed_of_light;
        for (int offset = 0; offset < a.dimension; offset += a.block) {
            for (int j = 0; j < a.orders; j++) {
                int i    = offset + 2 * j;
                f[i]     = -a.kappa * y[i] / r + (2 * speed_of_light + b) * y[i + 1];
                f[i + 1] = a.kappa * y[i + 1] / r - b * y[i];
                if (j) {
                    f[i] += y[i - 1] / speed_of_light;
                    f[i + 1] -= y[i - 2] / speed_of_light;
                }
                if (offset) {
                    f[i] += db * y[i - offset + 1];
                    f[i + 1] -= db * y[i - offset];
                }
            }
        }
        return GSL_SUCCESS;
    };

    // Taylor coefficients in energy are propagated together. The Coulomb Frobenius
    // series fixes a common phase and includes the potential/energy dependence at the origin.
    double gamma = std::sqrt(k * k - zc * zc), r0 = radial_grid_.first();
    constexpr int origin_order{8};
    std::array<std::vector<double>, origin_order + 1> origin;
    for (auto& c : origin) {
        c.resize(dimension, 0);
    }
    if (zn_) {
        origin[0][0] = 1;
        origin[0][1] = k < 0 ? -zc / (gamma - k) : (gamma + k) / zc;
    } else {
        origin[0][k < 0 ? 0 : 1] = 1;
    }
    auto origin_coefficients = [r0](Spline<double> const& field) {
        double c0 = field(0), c1 = field.deriv(1, 0), c2 = field.deriv(2, 0) / 2, c3 = field.deriv(3, 0) / 6;
        return std::array<double, 4>{c0 - r0 * (c1 - r0 * (c2 - r0 * c3)), c1 - r0 * (2 * c2 - 3 * r0 * c3),
                                     c2 - 3 * r0 * c3, c3};
    };
    auto v = origin_coefficients(ve_), dv = origin_coefficients(direction);
    for (int n = 1; n <= origin_order; n++) {
        for (int offset = 0; offset < dimension; offset += block) {
            for (int j = 0; j < orders; j++) {
                int i = offset + 2 * j;
                double p{0}, q{0};
                for (int m = 0; m < std::min(n, 4); m++) {
                    auto const& c = origin[n - 1 - m];
                    double b      = ((m == 0 ? enu__ : 0) - v[m]) / speed_of_light;
                    p += (b + (m == 0 ? 2 * speed_of_light : 0)) * c[i + 1];
                    q -= b * c[i];
                    if (offset) {
                        double db = ((m == 0 ? energy_direction__ : 0) - dv[m]) / speed_of_light;
                        p += db * c[i - offset + 1];
                        q -= db * c[i - offset];
                    }
                }
                if (j) {
                    p += origin[n - 1][i - 1] / speed_of_light;
                    q -= origin[n - 1][i - 2] / speed_of_light;
                }
                double a = n + gamma, determinant = n * (n + 2 * gamma);
                origin[n][i]     = ((a - k) * p + zc * q) / determinant;
                origin[n][i + 1] = ((a + k) * q - zc * p) / determinant;
            }
        }
    }
    std::vector<double> y(dimension, 0);
    for (int n = origin_order; n >= 0; n--) {
        for (int i = 0; i < dimension; i++) {
            y[i] = y[i] * r0 + origin[n][i];
        }
    }
    // The omitted common r0^gamma factor is independent of E and V and cancels
    // in the joint normalization. Keeping the initial amplitude finite avoids
    // absolute ODE errors dominating small Coulomb-origin spinors.
    gsl_odeiv2_system system{+rhs, nullptr, static_cast<size_t>(dimension), &parameters};
    std::unique_ptr<gsl_odeiv2_step, decltype(&gsl_odeiv2_step_free)> step(
            gsl_odeiv2_step_alloc(gsl_odeiv2_step_rk8pd, dimension), gsl_odeiv2_step_free);
    std::unique_ptr<gsl_odeiv2_control, decltype(&gsl_odeiv2_control_free)> control(
            gsl_odeiv2_control_y_new(epsabs_, epsrel_), gsl_odeiv2_control_free);
    std::unique_ptr<gsl_odeiv2_evolve, decltype(&gsl_odeiv2_evolve_free)> evolve(gsl_odeiv2_evolve_alloc(dimension),
                                                                                 gsl_odeiv2_evolve_free);
    std::vector<std::vector<double>> values(num_points(), std::vector<double>(dimension));
    values[0] = y;
    for (int ir = 0; ir < num_points() - 1; ir++) {
        parameters.ir = ir;
        parameters.x0 = radial_grid_[ir];
        double x = parameters.x0, end = radial_grid_[ir + 1], h = end - x;
        while (x < end) {
            int status =
                    gsl_odeiv2_evolve_apply(evolve.get(), control.get(), step.get(), &system, &x, end, &h, y.data());
            if (status != GSL_SUCCESS) {
                RTE_THROW("Dirac radial integration failed: " + std::to_string(status));
            }
        }
        values[ir + 1] = y;
        double peak{0};
        for (auto c : y) {
            if (!std::isfinite(c)) {
                RTE_THROW("Dirac radial integration returned a nonfinite spinor");
            }
            peak = std::max(peak, std::abs(c));
        }
        if (peak > 1e6) {
            for (int j = 0; j <= ir + 1; j++) {
                for (auto& c : values[j]) {
                    c /= 1e6;
                }
            }
            for (auto& c : y) {
                c /= 1e6;
            }
            gsl_odeiv2_evolve_reset(evolve.get());
            gsl_odeiv2_step_reset(step.get());
        }
    }

    // Differentiate the joint P/Q normalization, rather than normalizing each energy derivative separately.
    std::vector<double> norm(orders, 0), dnorm(orders, 0), scale(orders, 0), dscale(orders, 0);
    Spline<double> product(radial_grid_);
    for (int j = 0; j < orders; j++) {
        for (int ir = 0; ir < num_points(); ir++) {
            auto const& c = values[ir];
            product(ir)   = 0;
            for (int n = 0; n <= j; n++) {
                for (int part = 0; part < 2; part++) {
                    product(ir) += c[2 * n + part] * c[2 * (j - n) + part];
                }
            }
        }
        norm[j] = product.interpolate().integrate(0);
        if (direction__) {
            for (int ir = 0; ir < num_points(); ir++) {
                auto const& c = values[ir];
                product(ir)   = 0;
                for (int n = 0; n <= j; n++) {
                    for (int part = 0; part < 2; part++) {
                        product(ir) += 2 * c[2 * n + part] * c[block + 2 * (j - n) + part];
                    }
                }
            }
            dnorm[j] = product.interpolate().integrate(0);
        }
    }
    if (!std::isfinite(norm[0]) || norm[0] <= 0) {
        RTE_THROW("Dirac radial solution has a nonpositive or nonfinite spinor norm");
    }
    scale[0]  = 1 / std::sqrt(norm[0]);
    dscale[0] = -0.5 * scale[0] * dnorm[0] / norm[0];
    for (int j = 1; j < orders; j++) {
        for (int n = 1; n <= j; n++) {
            scale[j] -= (j - 0.5 * n) * norm[n] * scale[j - n];
            dscale[j] -= (j - 0.5 * n) * (dnorm[n] * scale[j - n] + norm[n] * dscale[j - n]);
        }
        scale[j] /= j * norm[0];
        dscale[j] = dscale[j] / (j * norm[0]) - scale[j] * dnorm[0] / norm[0];
    }
    for (auto& c : values) {
        auto raw = c;
        std::fill(c.begin(), c.end(), 0);
        for (int j = 0; j < orders; j++) {
            for (int n = 0; n <= j; n++) {
                for (int part = 0; part < 2; part++) {
                    c[2 * j + part] += scale[n] * raw[2 * (j - n) + part];
                    if (direction__) {
                        c[block + 2 * j + part] +=
                                scale[n] * raw[block + 2 * (j - n) + part] + dscale[n] * raw[2 * (j - n) + part];
                    }
                }
            }
        }
    }
    radial_solver_response_t result{};
    result.value.kappa     = kappa__;
    result.value.num_nodes = 0;
    for (auto* field : {&result.value.p, &result.value.rdudr, &result.value.small_p, &result.value.small_rdudr}) {
        field->resize(num_points(), 0);
    }
    if (direction__) {
        for (auto* field : {&result.p, &result.rdudr, &result.small_p, &result.small_rdudr}) {
            field->resize(num_points(), 0);
        }
    }
    double factorial = std::tgamma(dme__ + 1);
    std::vector<double> first(dimension), second(dimension);
    for (int ir = 0; ir < num_points(); ir++) {
        double r      = radial_grid_[ir];
        parameters.ir = ir == num_points() - 1 ? ir - 1 : ir;
        parameters.x0 = radial_grid_[parameters.ir];
        double dx     = r - parameters.x0;
        auto const& c = values[ir];
        rhs(r, c.data(), first.data(), &parameters);
        for (int offset = 0; offset < dimension; offset += block) {
            int i    = offset + 2 * dme__;
            auto& p  = offset ? result.p : result.value.p;
            auto& dp = offset ? result.rdudr : result.value.rdudr;
            auto& q  = offset ? result.small_p : result.value.small_p;
            auto& dq = offset ? result.small_rdudr : result.value.small_rdudr;
            p[ir]    = factorial * c[i];
            q[ir]    = factorial * c[i + 1];
            dp[ir]   = factorial * (first[i] - c[i] / r);
            dq[ir]   = factorial * (first[i + 1] - c[i + 1] / r);
            if (!std::isfinite(p[ir]) || !std::isfinite(q[ir]) || !std::isfinite(dp[ir]) || !std::isfinite(dq[ir])) {
                RTE_THROW("Dirac radial solution returned nonfinite radial data");
            }
        }
        if (ir && result.value.p[ir] * result.value.p[ir - 1] < 0) {
            result.value.num_nodes++;
        }
        if (ir == num_points() - 1) {
            // Spatial derivative of the coupled ODE, including the direction of V'(R).
            rhs(r, first.data(), second.data(), &parameters);
            double vp  = ve_.deriv(1, parameters.ir, dx) + zn_ / (r * r);
            double dvp = direction.deriv(1, parameters.ir, dx);
            for (int offset = 0; offset < dimension; offset += block) {
                int i = offset + 2 * dme__;
                second[i] += k * c[i] / (r * r) - vp * c[i + 1] / speed_of_light;
                second[i + 1] += -k * c[i + 1] / (r * r) + vp * c[i] / speed_of_light;
                if (offset) {
                    second[i] -= dvp * c[i - offset + 1] / speed_of_light;
                    second[i + 1] += dvp * c[i - offset] / speed_of_light;
                }
                auto& surface_p = offset ? result.uderiv : result.value.uderiv;
                auto& surface_q = offset ? result.small_uderiv : result.value.small_uderiv;
                for (int part = 0; part < 2; part++) {
                    auto& surface = part ? surface_q : surface_p;
                    int a         = i + part;
                    surface       = {factorial * c[a] / r, factorial * (first[a] - c[a] / r) / r,
                                     factorial * (second[a] - 2 * first[a] / r + 2 * c[a] / (r * r)) / r};
                }
            }
        }
    }
    return result;
}

enu_search_t
find_enu(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
         std::vector<double> const& v__, enu_search_t const& enu__, int kappa__)
{
    if ((rel__ == relativity_t::dirac) != (kappa__ != 0) ||
        (kappa__ && l__ != (kappa__ < 0 ? -kappa__ - 1 : kappa__))) {
        RTE_THROW("linearisation energy search requires a consistent angular channel");
    }
    double tolerance = kappa__ ? 1e-13 : 1e-3;
    Radial_solver solver(zn__, v__, radial_grid__, tolerance, tolerance);

    enu_search_t enu_out = enu__;

    int np = solver.num_points();

    Spline<double> chi_p(radial_grid__);
    Spline<double> chi_q(radial_grid__);

    std::vector<double> p(np);
    std::vector<double> q(np);
    std::vector<double> dpdr(np);
    std::vector<double> dqdr(np);

    auto integrate_until = [&](double energy, std::function<bool(int, int, double&)> condition) {
        if (!kappa__) {
            return solver.integrate_forward_until(rel__, energy, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                  condition);
        }
        for (int i = 0; i < 1000; i++) {
            auto solution = solver.solve_dirac(0, kappa__, energy);
            p             = std::move(solution.p);
            q             = std::move(solution.small_p);
            dpdr.back()   = solution.uderiv[0] + radial_grid__.last() * solution.uderiv[1];
            if (condition(i, solution.num_nodes, energy)) {
                return energy;
            }
        }
        RTE_THROW("Dirac linearisation boundary root was not found in 1000 iterations");
        return 0.0;
    };

    double const enu_tol = kappa__ ? 1e-13 : 1e-11;

    std::stringstream sinfo;
    sinfo << "find_enu()" << std::endl
          << "  rel : " << get_relativity(rel__) << std::endl
          << "  zn  : " << zn__ << std::endl
          << "  n   : " << n__ << std::endl
          << "  l   : " << l__ << std::endl
          << "  auto_enu        : " << enu__.auto_enu << std::endl
          << "  etop            : " << enu__.etop << std::endl
          << "  etop_first_pass : " << enu__.etop_first_pass << std::endl
          << "  ebot            : " << enu__.ebot << std::endl;

    auto compute_etop = [&]() -> int {
        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        int s{1};
        int sp;
        double denu{1e-6};
        double e1;
        sinfo << "  step : estimate top of the energy" << std::endl;
        try {
            /* 1st pass: estimate upper and lower boundaries of the etop */
            e1 = integrate_until(enu__.etop, [&s, &sp, &denu, n__, l__](int iter, int nn, double& enu) {
                sp = s;
                s  = (nn > (n__ - l__ - 1)) ? -1 : 1;
                if (s != sp && iter > 0) {
                    return true;
                }
                denu = std::min(0.5, denu * 2);
                enu += s * denu;
                return false;
            });
        } catch (std::exception const& e) {
            sinfo << "  denu  : " << denu << std::endl;
            sinfo << "  error : " << e.what() << std::endl;
            return 1;
        }

        if (std::abs(e1 - enu__.etop_first_pass) > enu_tol) {
            sinfo << "  step : refine top of the energy" << std::endl;

            enu_out.etop_first_pass = e1;

            double e2 = e1 - sp * denu;

            /* e1 is bottom, e2 is top energy */
            if (e1 > e2) {
                std::swap(e1, e2);
            }

            sinfo << "  e1 : " << e1 << std::endl << "  e2 : " << e2 << std::endl;

            try {
                /* 2nd pass: refine by bisection */
                enu_out.etop =
                        integrate_until((e1 + e2) / 2, [&e1, &e2, enu_tol, n__, l__](int iter, int nn, double& enu) {
                            if (nn > (n__ - l__ - 1)) {
                                e2 = enu;
                            } else {
                                e1 = enu;
                            }
                            enu = (e1 + e2) / 2.0;
                            return std::abs(e1 - e2) < enu_tol;
                        });
            } catch (std::exception const& e) {
                sinfo << " last e1 : " << e1 << std::endl
                      << " last e2 : " << e2 << std::endl
                      << " error   : " << e.what() << std::endl;
                return 1;
            }
        } else {
            enu_out.etop = enu__.etop;
        }

        return 0;
    };

    /* start by computing top of linearization energy */
    if (compute_etop() != 0) {
        sinfo << "  error : top of the linearization energy interval is not found";
        RTE_THROW(sinfo);
    }

    /* check for quick exit */
    if (enu_out.etop == enu__.etop) {
        enu_out = enu__;
        return enu_out;
    }

    auto surface_deriv = [&dpdr, &p, &radial_grid__]() {
        if (true) {
            /* return  p'(R) */
            return dpdr.back();
        } else {
            /* return R*u'(R) */
            return dpdr.back() - p.back() / radial_grid__.last();
        }
    };

    /* current surface derivative */
    double sd = surface_deriv();
    /* try several steps first */
    double denu{1e-5};
    //for (auto de: {1e-4, 1e-3, 1e-2, 1e-1}) {
    //    int num_nodes;
    //    solver.integrate_forward_until(rel__, enu_out.etop - de, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
    //                     [&num_nodes](int iter, int nn, double& enu) {
    //                            num_nodes = nn;
    //                            return true;
    //                        });
    //    if (surface_deriv() * sd < 0 || num_nodes != (n__ - l__ - 1)) {
    //        break;
    //    }
    //    denu = de;
    //}

    /* Now we go down in energy and search for enu such that the wave-function derivative is zero
     * at the muffin-tin boundary. This will be the bottom of the band. Here we look at a sign change
     * of the derivative. */
    try {
        sinfo << "  step : estimate bottom of the energy" << std::endl;
        auto sd = surface_deriv();
        auto e1 = integrate_until(enu_out.etop, [&denu, sd, &surface_deriv, &p](int iter, int nn, double& enu) {
            if (surface_deriv() * sd < 0) {
                return true;
            }
            /* do not allow step in energy to grow too much */
            //denu = std::min(0.1, denu * 1.1);
            denu *= 1.1;
            enu -= denu;
            return false;
        });

        /* refine bottom energy */
        auto e2 = e1 + denu;
        /* simple estimation of the bottom energy */
        if (enu__.auto_enu == 3) {
            enu_out.ebot = (e1 + e2) * 0.5;
            enu_out.enu  = (enu_out.ebot + enu_out.etop) / 2.0;
        } else {
            sinfo << "  step : refine bottom of the energy";
            sinfo << "  e1 : " << e1 << std::endl << "  e2 : " << e2 << std::endl;
            enu_out.ebot = integrate_until((e1 + e2) / 2,
                                           [&e1, &e2, sd, &surface_deriv, enu_tol](int iter, int nn, double& enu) {
                                               if (std::abs(e1 - e2) < enu_tol) {
                                                   return true;
                                               }
                                               if (surface_deriv() * sd > 0) {
                                                   e2 = enu;
                                               } else {
                                                   e1 = enu;
                                               }
                                               enu = (e1 + e2) / 2.0;
                                               return false;
                                           });
            switch (enu__.auto_enu) {
                case 1: {
                    enu_out.enu = (enu_out.ebot + enu_out.etop) / 2.0;
                    break;
                }
                case 2: {
                    enu_out.enu = enu_out.ebot;
                    break;
                }
                default: {
                    RTE_THROW("wrong type of auto_enu");
                }
            }
        }
    } catch (std::exception const& e) {
        sinfo << "  last p(R)  : " << p.back() << std::endl
              << "  last p'(R) : " << dpdr.back() << std::endl
              << "  error      : " << e.what() << std::endl;
        RTE_THROW(sinfo);
    }

    return enu_out;
}

double
linearization_energy_response(int zn__, int l__, Radial_grid<double> const& grid__,
                              std::vector<double> const& potential__, enu_search_t const& enu__,
                              std::vector<double> const& direction__, relativity_t rel__, int kappa__)
{
    if ((rel__ == relativity_t::dirac) != (kappa__ != 0) ||
        (kappa__ && l__ != (kappa__ < 0 ? -kappa__ - 1 : kappa__))) {
        RTE_THROW("linearisation energy response requires a consistent angular channel");
    }
    if (enu__.auto_enu < 1 || enu__.auto_enu > 3 || potential__.size() != static_cast<size_t>(grid__.num_points()) ||
        !std::all_of(potential__.begin(), potential__.end(), [](double x) { return std::isfinite(x); })) {
        RTE_THROW("invalid automatic linearisation energy response");
    }
    double tolerance = kappa__ ? 1e-13 : 1e-3;
    Radial_solver solver(zn__, potential__, grid__, tolerance, tolerance);
    bool nonrelativistic = rel__ == relativity_t::none;
    std::vector<double> shift_direction(grid__.num_points(), nonrelativistic ? 1 : 0);
    auto boundary_response = [&](double energy, bool bottom) {
        auto solve = [&](auto const& direction, double de) {
            return kappa__ ? solver.solve_dirac_potential_response(0, kappa__, energy, direction, de)
                           : solver.solve_potential_response(0, l__, energy, direction, rel__, de);
        };
        auto response = solve(direction__, 0);
        auto shift    = solve(shift_direction, nonrelativistic ? 0 : 1);
        auto boundary = [&](std::array<double, 3> const& u) { return bottom ? u[0] + grid__.last() * u[1] : u[0]; };
        // F is u(R) at the band top and p'(R) at the bottom: dE = -D_V F[dV]/D_E F.
        // Only nonrelativistic states use the original equivalent D_V F[1] denominator.
        double denominator = (nonrelativistic ? 1 : -1) * boundary(shift.uderiv);
        double residual    = boundary(response.value.uderiv) / denominator;
        double derivative  = boundary(response.uderiv) / denominator;
        if (!std::isfinite(residual) || !std::isfinite(derivative) || std::abs(residual) > 1e-7) {
            RTE_THROW("linearisation energy response requires a converged boundary root");
        }
        return derivative;
    };
    double top = boundary_response(enu__.etop, false);
    if (enu__.auto_enu == 3) {
        // The coarse bottom bracket is a fixed offset from etop on each search branch.
        return top;
    }
    double bottom = boundary_response(enu__.ebot, true);
    return enu__.auto_enu == 1 ? (top + bottom) / 2 : bottom;
}

enu_search_t
find_enu_v2(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
            std::vector<double> const& v__, enu_search_t const& enu__)
{
    Radial_solver solver(zn__, v__, radial_grid__);

    enu_search_t enu_out = enu__;

    int np = solver.num_points();

    Spline<double> chi_p(radial_grid__);
    Spline<double> chi_q(radial_grid__);

    std::vector<double> p(np);
    std::vector<double> q(np);
    std::vector<double> dpdr(np);
    std::vector<double> dqdr(np);

    double constexpr enu_tol{1e-9};

    std::stringstream sinfo;

    auto compute_etop = [&]() -> int {
        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        int s{1};
        int sp;
        double denu{1e-5};
        double e1;
        sinfo << "find_enu(): find top enery" << std::endl
              << "  n         : " << n__ << ", l : " << l__ << std::endl
              << "  enu_start : " << enu__.etop << std::endl;
        try {
            /* 1st pass: estimate upper and lower boundaries of the etop */
            e1 = solver.integrate_forward_until(rel__, enu__.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                [&s, &sp, &denu, n__, l__](int iter, int nn, double& enu) {
                                                    sp = s;
                                                    s  = (nn > (n__ - l__ - 1)) ? -1 : 1;
                                                    if (s != sp && iter > 0) {
                                                        return true;
                                                    }
                                                    denu = std::min(0.5, denu * 2);
                                                    enu += s * denu;
                                                    return false;
                                                });
        } catch (std::exception const& e) {
            sinfo << e.what() << std::endl << "denu : " << denu;
            return 1;
        }

        /* only with this condition we need to refine top energy by bisection;
         * otherwise etop stays untouched */
        if (std::abs(e1 - enu__.etop_first_pass) > enu_tol) {
            enu_out.etop_first_pass = e1;

            double e2 = e1 - sp * denu;

            /* e1 is bottom, e2 is top energy */
            if (e1 > e2) {
                std::swap(e1, e2);
            }

            sinfo << "find_enu(): refine top energy" << std::endl
                  << "  e1 : " << e1 << ", e2 : " << e2 << std::endl
                  << "  enu_start : " << (e1 + e2) / 2 << std::endl;

            try {
                /* 2nd pass: refine by bisection */
                enu_out.etop = solver.integrate_forward_until(
                        rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                        [&e1, &e2, enu_tol, n__, l__](int iter, int nn, double& enu) {
                            if (nn > (n__ - l__ - 1)) {
                                e2 = enu;
                            } else {
                                e1 = enu;
                            }
                            enu = (e1 + e2) / 2.0;
                            return std::abs(e1 - e2) < enu_tol;
                        });
            } catch (std::exception const& e) {
                sinfo << e.what() << std::endl;
                return 1;
            }
        }
        return 0;
    };

    /* start by computing top of linearization energy */
    if (compute_etop() != 0) {
        sinfo << "find_enu(): top of the linearization energy interval is not found";
        RTE_THROW(sinfo);
    }

    auto surface_deriv = [&dpdr, &p, &radial_grid__]() {
        if (true) {
            /* return  p'(R) */
            return dpdr.back();
        } else {
            /* return R*u'(R) */
            return dpdr.back() - p.back() / radial_grid__.last();
        }
    };

    /* current surface derivative */
    double sd = surface_deriv();
    /* try several steps first */
    double denu{1e-5};
    for (auto de : {1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
        int num_nodes;
        solver.integrate_forward_until(rel__, enu_out.etop - de, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                       [&num_nodes](int iter, int nn, double& enu) {
                                           num_nodes = nn;
                                           return true;
                                       });
        if (surface_deriv() * sd < 0 || num_nodes != (n__ - l__ - 1)) {
            break;
        }
        denu = de;
    }

    /* Now we go down in energy and search for enu such that the wave-function derivative is zero
     * at the muffin-tin boundary. This will be the bottom of the band. Here we look at a sign change
     * of the derivative. */
    try {
        sinfo << "find_enu(): find bottom energy" << std::endl << "  enu_start : " << enu__.etop << std::endl;
        auto sd = surface_deriv();
        auto e1 = solver.integrate_forward_until(rel__, enu_out.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                 [&denu, sd, &surface_deriv, &p](int iter, int nn, double& enu) {
                                                     if (surface_deriv() * sd < 0) {
                                                         return true;
                                                     }
                                                     /* do not allow step in energy to grow too much */
                                                     //denu = std::min(0.1, denu * 1.2);
                                                     denu *= 1.1;
                                                     enu -= denu;
                                                     return false;
                                                 });

        /* refine bottom energy */
        auto e2 = e1 + denu;
        /* simple estimation of the bottom energy */
        if (enu__.auto_enu == 3) {
            enu_out.ebot = (e1 + e2) * 0.5;
            enu_out.enu  = (enu_out.ebot + enu_out.etop) / 2.0;
        } else {
            sinfo << "find_enu(): refine bottom energy" << std::endl << "  enu_start : " << (e1 + e2) / 2 << std::endl;
            enu_out.ebot =
                    solver.integrate_forward_until(rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                   [&e1, &e2, sd, &surface_deriv](int iter, int nn, double& enu) {
                                                       if (surface_deriv() * sd > 0) {
                                                           e2 = enu;
                                                       } else {
                                                           e1 = enu;
                                                       }
                                                       enu = (e1 + e2) / 2.0;
                                                       return std::abs(surface_deriv()) < enu_tol;
                                                   });
            switch (enu__.auto_enu) {
                case 1: {
                    enu_out.enu = (enu_out.ebot + enu_out.etop) / 2.0;
                    break;
                }
                case 2: {
                    enu_out.enu = enu_out.ebot;
                    break;
                }
                default: {
                    RTE_THROW("wrong type of auto_enu");
                }
            }
        }
    } catch (std::exception const& e) {
        sinfo << e.what() << std::endl << "denu : " << denu << std::endl;
    }

    return enu_out;
}

}; // namespace sirius
