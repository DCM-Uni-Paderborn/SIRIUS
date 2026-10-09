/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "density/paw_xc_correction.hpp"
#include "density/paw_xc_field_map.hpp"
#include "density/paw_xc_reconstruction.hpp"
#include "function3d/spheric_function_interpolator.hpp"

using namespace sirius;

int
test_paw_xc_correction(int ns__, bool compact__)
{
    auto conf                          = R"({"parameters": {"electronic_structure_method": "pseudopotential",
        "use_symmetry": false, "pw_cutoff": 6.0, "gk_cutoff": 1.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"]},
        "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    conf["parameters"]["num_mag_dims"] = ns__ - 1;
    Simulation_context ctx(conf);
    ctx.unit_cell().set_lattice_vectors({{1.7, 0, 0}, {0.3, 2, 0}, {0.2, -0.4, 2.2}});
    auto& type = ctx.unit_cell().add_atom_type("H");
    type.zn(1);
    type.is_paw(true);
    type.set_radial_grid(radial_grid_t::linear, 801, 0.001, 2.4, 1.0);
    int nr = type.num_mt_points();
    // Synthetic setup with independent nonzero core rho/tau and several Q multipoles.
    for (int l = 0; l <= 1; l++) {
        std::vector<double> ae(nr), ps(nr), beta(nr);
        for (int ir = 0; ir < nr; ir++) {
            double r = type.radial_grid(ir);
            ae[ir]   = std::pow(r, l + 1) * std::exp(-0.55 * r * r);
            ps[ir]   = std::pow(r, l + 1) * std::exp(-0.35 * r * r);
            beta[ir] = std::pow(r, l + 1) * std::exp(-2 * r * r);
        }
        type.add_beta_radial_function(angular_momentum(l), beta);
        if (compact__) {
            ae.resize(301);
            ps.resize(301);
        }
        type.add_ae_paw_wf(ae);
        type.add_ps_paw_wf(ps);
    }
    for (auto ijl : {std::array<int, 3>{0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 2}}) {
        std::vector<double> q(nr);
        for (int ir = 0; ir < nr; ir++) {
            double r = type.radial_grid(ir);
            q[ir]    = 0.17 * std::pow(r, ijl[2] + 2) * std::exp(-1.2 * r * r);
            if (compact__ && ir >= 481) {
                q[ir] = 0;
            }
        }
        type.add_q_radial_function(ijl[0], ijl[1], ijl[2], q);
    }
    std::vector<double> core_rho(nr), core_tau(nr), ps_core(nr);
    for (int ir = 0; ir < nr; ir++) {
        double r     = type.radial_grid(ir);
        core_rho[ir] = 0.2 * std::exp(-4 * r);
        core_tau[ir] = 0.3 * std::exp(-3 * r);
        ps_core[ir]  = 0.11 * std::exp(-r);
        if (compact__) {
            if (ir >= 441)
                core_rho[ir] = 0;
            if (ir >= 461)
                core_tau[ir] = 0;
        }
    }
    type.paw_ae_core_charge_density(core_rho);
    type.paw_ae_core_kinetic_density(core_tau);
    type.ps_core_charge_density(ps_core);
    type.paw_wf_occ({1, 0});
    type.local_potential(std::vector<double>(nr, 0));
    type.ps_total_charge_density(std::vector<double>(nr, 0));
    matrix<double> dion({2, 2});
    dion.zero();
    type.d_mtrx_ion(dion);
    ctx.unit_cell().add_atom("H", {0.1, 0.2, 0.3}, {0, 0, 1});
    ctx.unit_cell().add_atom("H", {0.7, 0.6, 0.8}, {0, 0, 1});
    ctx.initialize();
    Density density(ctx);
    PAW_xc_correction op(type);
    int n    = type.indexb().size();
    auto& dm = density.density_matrix(0);
    std::vector<std::complex<double>> c(n), dc(n), down(n), ddown(n);
    for (int i = 0; i < n; i++) {
        c[i]     = {0.2 * std::cos(0.3 + i), 0.15 * std::sin(0.4 + 2 * i)};
        dc[i]    = {0.1 * std::sin(0.2 + i), -0.13 * std::cos(0.6 + 3 * i)};
        down[i]  = {0.13 * std::sin(0.7 + 2 * i), 0.11 * std::cos(0.2 + i)};
        ddown[i] = {-0.09 * std::cos(0.3 + i), 0.07 * std::sin(0.9 + 3 * i)};
    }
    auto set_dm = [&](double step) {
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto& atom_dm = density.density_matrix(ia);
            for (int s = 0; s < static_cast<int>(atom_dm.size(2)); s++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        auto const& left  = ns__ == 4 && s == 1 ? down : c;
                        auto const& right = ns__ == 4 && s > 0 ? down : c;
                        auto const& dl    = ns__ == 4 && s == 1 ? ddown : dc;
                        auto const& dr    = ns__ == 4 && s > 0 ? ddown : dc;
                        atom_dm(i, j, s)  = (ns__ == 4 ? 0.8 : 0.8 / (s + 1)) * (left[i] + step * (ia + 1) * dl[i]) *
                                           std::conj(right[j] + step * (ia + 1) * dr[j]);
                    }
                }
            }
        }
    };
    auto component_dm = [&](int ia, int i, int j, int s) {
        auto const& d = density.density_matrix(ia);
        if (ns__ != 4) {
            return std::real(d(i, j, s));
        }
        if (s < 2) {
            return std::real(d(i, j, 0) + (1.0 - 2 * s) * d(i, j, 1));
        }
        auto ud = d(i, j, 2), du = std::conj(d(j, i, 2));
        return s == 2 ? std::real(ud + du) : -std::imag(ud - du);
    };
    auto projector_derivative = [&](mdarray<double, 3> const& matrix, int ia) {
        double value{0};
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                if (ns__ == 4) {
                    auto flip = std::complex<double>(matrix(i, j, 2), -matrix(i, j, 3));
                    value +=
                            1.6 * (ia + 1) *
                            std::real(std::conj(dc[i]) * ((matrix(i, j, 0) + matrix(i, j, 1)) * c[j] + flip * down[j]) +
                                      std::conj(ddown[i]) *
                                              (std::conj(flip) * c[j] + (matrix(i, j, 0) - matrix(i, j, 1)) * down[j]));
                } else {
                    for (int s = 0; s < ns__; s++) {
                        value += 1.6 / (s + 1) * (ia + 1) * std::real(std::conj(dc[i]) * matrix(i, j, s) * c[j]);
                    }
                }
            }
        }
        return value;
    };
    int failures{0};
    Test_check check(failures, ctx.comm(),
                     "PAW spins=" + std::to_string(ns__) + " compact support=" + std::to_string(compact__));
    check("source support excludes Q", std::abs(op.support_radius() - type.radial_grid(compact__ ? 461 : nr - 1)), 0);
    if (compact__) {
        auto extended_core    = core_rho;
        extended_core[nr - 2] = 1e-100;
        type.paw_ae_core_charge_density(extended_core);
        check("nonzero core charge tail is retained", std::abs(op.support_radius() - type.radial_grid().last()), 0);
        type.paw_ae_core_charge_density(core_rho);
        extended_core         = core_tau;
        extended_core[nr - 2] = 1e-100;
        type.paw_ae_core_kinetic_density(extended_core);
        check("nonzero core tau tail is retained", std::abs(op.support_radius() - type.radial_grid().last()), 0);
        type.paw_ae_core_kinetic_density(core_tau);
    }
    set_dm(0);
    auto fields = density.generate_paw_xc_correction(0);
    density.generate_paw_density();
    Gaunt_coefficients<double> gaunt(type.indexr().lmax(), 2 * type.indexr().lmax(), type.indexr().lmax(),
                                     SHT::gaunt_rrr);
    double error{0};
    for (auto it : ctx.unit_cell().spl_num_paw_atoms()) {
        int ia = ctx.unit_cell().paw_atom_index(it.i);
        for (int s = 0; s < ns__; s++) {
            for (int ir = 0; ir < nr; ir++) {
                for (int lm = 0; lm < fields[0][s].angular_domain_size(); lm++) {
                    int component = ns__ == 4 ? s : 0;
                    double ref    = density.paw_density().ae_density(component, ia)(lm, ir) -
                                 density.paw_density().ps_density(component, ia)(lm, ir);
                    if (ns__ == 2) {
                        ref = 0.5 * (ref + (1 - 2 * s) * (density.paw_density().ae_density(1, ia)(lm, ir) -
                                                          density.paw_density().ps_density(1, ia)(lm, ir)));
                    }
                    // The electrostatic pseudo density includes Q; the XC field must not.
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            for (auto const& g : gaunt.gaunt_vector(type.indexb(i).lm, type.indexb(j).lm)) {
                                if (g.lm3 == lm) {
                                    ref += component_dm(ia, i, j, s) * g.coef *
                                           type.q_radial_function(type.indexb(i).idxrf, type.indexb(j).idxrf,
                                                                  g.l3)(ir) /
                                           std::pow(type.radial_grid(ir), 2);
                                }
                            }
                        }
                    }
                    if (lm == 0 && (ns__ != 4 || s == 0)) {
                        ref += core_rho[ir] / ((ns__ == 4 ? 1 : ns__) * y00);
                    }
                    error = std::max(error, std::abs(fields[0][s](lm, ir) - ref));
                }
            }
        }
    }
    check("charge correction excludes electrostatic Q", error, 2e-13);
    auto valence_tau = density.generate_paw_valence_kinetic_density(0);
    error            = 0;
    for (int s = 0; s < ns__; s++) {
        for (int ir = 0; ir < nr; ir++) {
            for (int lm = 0; lm < fields[1][s].angular_domain_size(); lm++) {
                double ref = valence_tau[0][s](lm, ir) - valence_tau[1][s](lm, ir);
                if (lm == 0 && (ns__ != 4 || s == 0)) {
                    ref += core_tau[ir] / ((ns__ == 4 ? 1 : ns__) * y00);
                }
                error = std::max(error, std::abs(fields[1][s](lm, ir) - ref));
            }
        }
    }
    check("positive core tau and no Q in tau", error, 2e-13);
    if (ns__ == 4) {
        auto valence = op.fields(dm, false);
        error        = 0;
        for (int f = 0; f < 2; f++) {
            for (int s = 0; s < 4; s++) {
                for (int ir = 0; ir < nr; ir++) {
                    for (int lm = 0; lm < fields[f][s].angular_domain_size(); lm++) {
                        double core = s == 0 && lm == 0 ? (f == 0 ? core_rho[ir] : core_tau[ir]) / y00 : 0;
                        error       = std::max(error, std::abs(fields[f][s](lm, ir) - valence[f][s](lm, ir) - core));
                    }
                }
            }
        }
        check("frozen core has no spin-vector components", error, 2e-13);
        double angle = 0.71, a = std::cos(angle / 2), b = std::sin(angle / 2);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                auto uu = 0.8 * c[i] * std::conj(c[j]), dd = 0.8 * down[i] * std::conj(down[j]);
                auto ud = 0.8 * c[i] * std::conj(down[j]), du = 0.8 * down[i] * std::conj(c[j]);
                dm(i, j, 0) = a * a * uu + b * b * dd - a * b * (ud + du);
                dm(i, j, 1) = b * b * uu + a * a * dd + a * b * (ud + du);
                dm(i, j, 2) = a * b * (uu - dd) + a * a * ud - b * b * du;
            }
        }
        auto rotated = op.fields(dm);
        error        = 0;
        for (int f = 0; f < 2; f++) {
            for (int ir = 0; ir < nr; ir++) {
                for (int lm = 0; lm < fields[f][0].angular_domain_size(); lm++) {
                    std::array<double, 4> ref{
                            fields[f][0](lm, ir),
                            std::cos(angle) * fields[f][1](lm, ir) - std::sin(angle) * fields[f][2](lm, ir),
                            std::sin(angle) * fields[f][1](lm, ir) + std::cos(angle) * fields[f][2](lm, ir),
                            fields[f][3](lm, ir)};
                    for (int s = 0; s < 4; s++) {
                        error = std::max(error, std::abs(rotated[f][s](lm, ir) - ref[s]));
                    }
                }
            }
        }
        check("local rho and tau spin-rotation covariance", error, 2e-13);
        set_dm(0);
    }
    if (ns__ == 2) {
        mdarray<std::complex<double>, 3> total_dm({n, n, 1});
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                total_dm(i, j, 0) = dm(i, j, 0) + dm(i, j, 1);
            }
        }
        auto total = op.fields(total_dm);
        error      = 0;
        for (int f = 0; f < 2; f++) {
            for (int ir = 0; ir < nr; ir++) {
                for (int lm = 0; lm < total[f][0].angular_domain_size(); lm++) {
                    error = std::max(error,
                                     std::abs(total[f][0](lm, ir) - fields[f][0](lm, ir) - fields[f][1](lm, ir)));
                }
            }
        }
        check("unpolarized core spin conversion", error, 2e-13);
    }

    // Reconstruct first, then evaluate a joint nonlinear/nonlocal energy.
    std::vector<r3::vector<double>> points{{0.31, 0.2, 0.1}, {0, 0, 0.75}, {-0.2, -0.4, 0.3}, {0.3, 0.6, -0.2}};
    std::vector<r3::vector<double>> displacements;
    for (auto p : points) {
        displacements.push_back(p);
        displacements.push_back(p - r3::vector<double>({0.6, 0.1, 0.2}));
    }
    using derivatives_t = std::array<std::vector<std::vector<std::array<double, 4>>>, 2>;
    auto energy         = [&](double step, derivatives_t* derivatives) {
        set_dm(step);
        auto f = density.generate_paw_xc_correction(0);
        mdarray<double, 3> values({5, 4, ns__});
        double sum{0}, result{0};
        for (int s = 0; s < ns__; s++) {
            Spheric_function_interpolator rho(f[0][s]), tau(f[1][s]);
            for (int p = 0; p < 4; p++) {
                auto a = rho.value_gradient(displacements[2 * p]);
                auto b = rho.value_gradient(displacements[2 * p + 1]);
                for (int x = 0; x < 4; x++) {
                    // A smooth background remains fixed in this projector derivative.
                    values(x, p, s) = (x == 0 ? 0.8 : 0.04) + a[x] + b[x];
                    result += 0.5 * p * (x == 0 ? 1 : 0.1) * std::pow(values(x, p, s), 2);
                }
                values(4, p, s) = 0.4 + tau.value_gradient(displacements[2 * p])[0] +
                                  tau.value_gradient(displacements[2 * p + 1])[0];
                result += 0.15 * p * std::pow(values(4, p, s), 2) + 0.07 * values(0, p, s) * values(4, p, s);
                sum += values(0, p, s) + 0.2 * values(4, p, s);
            }
        }
        if (derivatives) {
            (*derivatives)[0].resize(ns__);
            (*derivatives)[1].resize(ns__);
            for (int s = 0; s < ns__; s++) {
                (*derivatives)[0][s].resize(displacements.size());
                (*derivatives)[1][s].resize(displacements.size());
                for (int p = 0; p < 4; p++) {
                    for (int image = 0; image < 2; image++) {
                        auto& drho = (*derivatives)[0][s][2 * p + image];
                        auto& dtau = (*derivatives)[1][s][2 * p + image];
                        drho[0]    = p * values(0, p, s) + 0.07 * values(4, p, s) + 0.22 * sum;
                        for (int x = 1; x < 4; x++) {
                            drho[x] = 0.1 * p * values(x, p, s);
                        }
                        dtau = {0.3 * p * values(4, p, s) + 0.07 * values(0, p, s) + 0.044 * sum, 0, 0, 0};
                    }
                }
            }
        }
        return result + 0.11 * sum * sum;
    };
    derivatives_t derivatives;
    energy(0, &derivatives);
    mdarray<double, 3> local_matrix({n, n, ns__});
    for (int s = 0; s < ns__; s++) {
        auto drho   = Spheric_function_interpolator(fields[0][s]).adjoint(displacements, derivatives[0][s]);
        auto dtau   = Spheric_function_interpolator(fields[1][s]).adjoint(displacements, derivatives[1][s]);
        auto matrix = op.matrix_elements(drho, dtau);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                local_matrix(i, j, s) = matrix(i, j);
            }
        }
    }
    double analytic = projector_derivative(local_matrix, 0);
    for (double step : {1e-4, 5e-5}) {
        double fd = (energy(step, nullptr) - energy(-step, nullptr)) / (2 * step);
        check("joint reconstructed energy/projector derivative", std::abs(analytic - fd), 2e-8);
    }

    std::vector<r3::vector<double>> global_points{
            {0.63, 0.42, 0.29}, {1.3, 0.23, 0.94}, {0.27, 1.44, 1.19}, {1.66, 1.07, 0.38}};
    global_points[0] = r3::dot(ctx.unit_cell().lattice_vectors(), ctx.unit_cell().atom(0).position()) +
                       r3::vector<double>({0.0003, -0.0002, 0.0004});
    global_points[1] = r3::dot(ctx.unit_cell().lattice_vectors(), ctx.unit_cell().atom(1).position()) +
                       r3::vector<double>({-0.0002, 0.0003, -0.0004});
    for (int np : {1, 4}) {
        std::vector<r3::vector<double>> local_points;
        std::vector<int> ids;
        for (int p = 0; p < np; p++) {
            if (p % ctx.comm().size() == ctx.comm().rank()) {
                local_points.push_back(global_points[p]);
                ids.push_back(p);
            }
        }
        int local_size = static_cast<int>(local_points.size());
        mdarray<double, 2> rho({local_size, ns__}), tau({local_size, ns__});
        mdarray<double, 3> grad({local_size, 3, ns__});
        auto initialize = [&]() {
            for (int s = 0; s < ns__; s++) {
                for (int p = 0; p < local_size; p++) {
                    rho(p, s) = 0.8;
                    tau(p, s) = 0.4;
                    for (int x = 0; x < 3; x++) {
                        grad(p, x, s) = 0.04;
                    }
                }
            }
        };
        set_dm(0);
        initialize();
        PAW_xc_reconstruction map(density, local_points);
        map.add_to(rho, grad, tau);
        // Independent, deliberately oversized brute-force image range.
        error = 0;
        for (int s = 0; s < ns__; s++) {
            for (int p = 0; p < local_size; p++) {
                std::array<double, 5> reference{0.8, 0.04, 0.04, 0.04, 0.4};
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto f = density.generate_paw_xc_correction(ia);
                    Spheric_function_interpolator r(f[0][s], spheric_inner_extension_t::charge),
                            t(f[1][s], spheric_inner_extension_t::kinetic);
                    for (int z = -4; z <= 4; z++) {
                        for (int y = -4; y <= 4; y++) {
                            for (int x = -4; x <= 4; x++) {
                                auto source = r3::dot(ctx.unit_cell().lattice_vectors(),
                                                      ctx.unit_cell().atom(ia).position() +
                                                              r3::vector<double>({double(x), double(y), double(z)}));
                                auto d      = local_points[p] - source;
                                if (d.length() <= op.support_radius()) {
                                    auto value = r.value_gradient(d);
                                    for (int a = 0; a < 4; a++) {
                                        reference[a] += value[a];
                                    }
                                    reference[4] += t.value_gradient(d)[0];
                                }
                            }
                        }
                    }
                }
                error = std::max({error, std::abs(rho(p, s) - reference[0]), std::abs(tau(p, s) - reference[4])});
                for (int x = 0; x < 3; x++) {
                    error = std::max(error, std::abs(grad(p, x, s) - reference[x + 1]));
                }
            }
        }
        check("all-source periodic image reconstruction", error, 2e-12);

        auto periodic_energy = [&](double step, std::vector<mdarray<double, 3>>* matrices,
                                   xc_geometry_adjoint_t* geometry = nullptr) {
            set_dm(step);
            initialize();
            PAW_xc_reconstruction reconstruction(density, local_points);
            reconstruction.add_to(rho, grad, tau);
            double sum{0}, e{0};
            for (int s = 0; s < ns__; s++) {
                for (int p = 0; p < local_size; p++) {
                    double w = 0.2 + 0.1 * ids[p];
                    sum += w * (rho(p, s) + 0.3 * tau(p, s));
                    e += w * (0.5 * rho(p, s) * rho(p, s) + 0.2 * tau(p, s) * tau(p, s) + 0.1 * rho(p, s) * tau(p, s));
                    for (int x = 0; x < 3; x++) {
                        e += 0.03 * w * grad(p, x, s) * grad(p, x, s);
                    }
                }
            }
            ctx.comm().allreduce(&sum, 1);
            ctx.comm().allreduce(&e, 1);
            if (matrices || geometry) {
                xc_point_fields_t derivatives(local_size, ns__);
                auto& drho  = derivatives.rho;
                auto& dtau  = derivatives.tau;
                auto& dgrad = derivatives.gradient;
                for (int s = 0; s < ns__; s++) {
                    for (int p = 0; p < local_size; p++) {
                        double w   = 0.2 + 0.1 * ids[p];
                        drho(p, s) = w * (rho(p, s) + 0.1 * tau(p, s) + 0.16 * sum);
                        dtau(p, s) = w * (0.4 * tau(p, s) + 0.1 * rho(p, s) + 0.048 * sum);
                        for (int x = 0; x < 3; x++) {
                            dgrad(p, x, s) = 0.06 * w * grad(p, x, s);
                        }
                    }
                }
                if (matrices) {
                    *matrices = reconstruction.projector_matrices(drho, dgrad, dtau);
                }
                if (geometry) {
                    *geometry = reconstruction.geometry_adjoint(derivatives);
                }
            }
            return e + 0.08 * sum * sum;
        };
        std::vector<mdarray<double, 3>> matrices;
        periodic_energy(0, &matrices);
        double ad{0};
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            ad += projector_derivative(matrices[ia], ia);
        }
        for (double step : {1e-4, 5e-5}) {
            double fd = (periodic_energy(step, nullptr) - periodic_energy(-step, nullptr)) / (2 * step);
            check("distributed periodic reconstruction/projector derivative", std::abs(ad - fd), 2e-8);
        }

        xc_geometry_adjoint_t geometry(local_size, ctx.unit_cell().num_atoms());
        periodic_energy(0, nullptr, &geometry);
        auto cell         = ctx.unit_cell().lattice_vectors();
        auto inverse      = ctx.unit_cell().inverse_lattice_vectors();
        auto saved_points = local_points;
        std::vector<r3::vector<double>> atoms;
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            atoms.push_back(r3::dot(cell, ctx.unit_cell().atom(ia).position()));
        }
        double point_error{0}, atom_error{0}, cell_error{0};
        auto fd_error = [](double fd, double ad) { return std::abs(fd - ad) / std::max(1.0, std::abs(ad)); };
        for (int x = 0; x < 3; x++) {
            double dp{0}, translation{0};
            for (int p = 0; p < local_size; p++) {
                dp += geometry.points[p][x] * (0.13 + 0.02 * ids[p]);
                translation += geometry.points[p][x];
            }
            ctx.comm().allreduce(&dp, 1);
            ctx.comm().allreduce(&translation, 1);
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                translation += geometry.atoms(x, ia);
            }
            check("local geometry translation sum", std::abs(translation), 2e-11);
            for (double h : {1e-7, 5e-8}) {
                auto point_energy = [&](double h) {
                    local_points = saved_points;
                    for (int p = 0; p < local_size; p++) {
                        local_points[p][x] += h * (0.13 + 0.02 * ids[p]);
                    }
                    return periodic_energy(0, nullptr);
                };
                double plus = point_energy(h), minus = point_energy(-h);
                point_error  = std::max(point_error, fd_error((plus - minus) / (2 * h), dp));
                local_points = saved_points;
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto atom_energy = [&](double h) {
                        auto p = atoms[ia];
                        p[x] += h;
                        ctx.unit_cell().atom(ia).set_position(r3::dot(inverse, p));
                        return periodic_energy(0, nullptr);
                    };
                    plus       = atom_energy(h);
                    minus      = atom_energy(-h);
                    atom_error = std::max(atom_error, fd_error((plus - minus) / (2 * h), geometry.atoms(x, ia)));
                    ctx.unit_cell().atom(ia).set_position(r3::dot(inverse, atoms[ia]));
                }
                for (int y = 0; y < 3; y++) {
                    auto cell_energy = [&](double h) {
                        auto trial = cell;
                        trial(x, y) += h;
                        ctx.unit_cell().set_lattice_vectors(trial);
                        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                            ctx.unit_cell().atom(ia).set_position(r3::dot(r3::inverse(trial), atoms[ia]));
                        }
                        return periodic_energy(0, nullptr);
                    };
                    plus       = cell_energy(h);
                    minus      = cell_energy(-h);
                    cell_error = std::max(cell_error, fd_error((plus - minus) / (2 * h), geometry.lattice(x, y)));
                    cell_energy(0);
                }
            }
        }
        check("local geometry point FD", point_error, 2e-6);
        check("local geometry atom FD including frozen core", atom_error, 2e-6);
        check("local geometry all nine cell components FD", cell_error, 2e-6);

        // Moving the target by several cell vectors cannot change its fields.
        set_dm(0);
        initialize();
        map.add_to(rho, grad, tau);
        mdarray<double, 2> shifted_rho({local_size, ns__}), shifted_tau({local_size, ns__});
        mdarray<double, 3> shifted_grad({local_size, 3, ns__});
        copy(rho, shifted_rho);
        copy(tau, shifted_tau);
        copy(grad, shifted_grad);
        auto translated = local_points;
        auto shift      = r3::dot(ctx.unit_cell().lattice_vectors(), r3::vector<double>({3, -2, 1}));
        for (auto& p : translated) {
            p = p + shift;
        }
        initialize();
        PAW_xc_reconstruction(density, translated).add_to(rho, grad, tau);
        error = 0;
        for (int s = 0; s < ns__; s++) {
            for (int p = 0; p < local_size; p++) {
                error = std::max(
                        {error, std::abs(rho(p, s) - shifted_rho(p, s)), std::abs(tau(p, s) - shifted_tau(p, s))});
                for (int x = 0; x < 3; x++) {
                    error = std::max(error, std::abs(grad(p, x, s) - shifted_grad(p, x, s)));
                }
            }
        }
        check("lattice-translation invariance", error, 2e-12);
    }
    // The reconstructed XC density and its derivative must both exclude Q.
    // Retaining the host's ordinary Q projector contraction is a negative control.
    for (int target_count : {1, 5}) {
        std::vector<r3::vector<double>> points;
        std::vector<int> ids;
        for (int p = ctx.comm().rank(); p < target_count; p += ctx.comm().size()) {
            points.push_back(r3::dot(ctx.unit_cell().lattice_vectors(),
                                     r3::vector<double>({0.37 + 0.08 * p, 0.14 + 0.11 * p, 0.26 + 0.09 * p})));
            if (p == 0) {
                points.back() = global_points[0];
            }
            ids.push_back(p);
        }
        int np = static_cast<int>(points.size());
        Potential host_potential(ctx);
        std::vector<Smooth_periodic_function<double>> smooth_rho, smooth_tau;
        for (int s = 0; s < ns__; s++) {
            smooth_rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
            smooth_tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        }
        double smooth_tau_derivative{0};
        auto joint_energy = [&](double step, std::vector<mdarray<double, 3>>* matrices, bool spurious_smooth_q,
                                xc_geometry_adjoint_t* geometry = nullptr) {
            set_dm(step);
            for (int s = 0; s < ns__; s++) {
                for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                    bool zero_g                  = ctx.gvec().offset() + ig == 0;
                    smooth_rho[s].f_pw_local(ig) = zero_g ? 0.8 : 0;
                    smooth_tau[s].f_pw_local(ig) = zero_g ? 0.4 + step * 0.06 * (s + 1) : 0;
                    auto g                       = ctx.gvec().gvec(gvec_index_t::local(ig));
                    if (std::abs(g[0]) == 1 && g[1] == 0 && g[2] == 0) {
                        smooth_rho[s].f_pw_local(ig) = std::complex<double>(0.02, 0.01 * g[0]) * double(s + 1);
                        smooth_tau[s].f_pw_local(ig) = std::complex<double>(0.03, -0.02 * g[0]) * double(s + 1);
                    }
                }
            }
            PAW_xc_field_map reconstruction(density, points);
            auto fields      = reconstruction.sample(smooth_rho, smooth_tau);
            auto const& rho  = fields.rho;
            auto const& grad = fields.gradient;
            auto const& tau  = fields.tau;
            double e{0}, sum{0};
            for (int s = 0; s < ns__; s++) {
                for (int p = 0; p < np; p++) {
                    double w = 0.3 + 0.07 * ids[p];
                    e += w * (0.5 * rho(p, s) * rho(p, s) + 0.2 * tau(p, s) * tau(p, s));
                    sum += w * (rho(p, s) + 0.3 * tau(p, s));
                    for (int x = 0; x < 3; x++) {
                        e += 0.03 * w * grad(p, x, s) * grad(p, x, s);
                    }
                }
            }
            ctx.comm().allreduce(&e, 1);
            ctx.comm().allreduce(&sum, 1);
            if (matrices || geometry) {
                xc_point_fields_t derivatives(np, ns__);
                for (int s = 0; s < ns__; s++) {
                    for (int p = 0; p < np; p++) {
                        double w              = 0.3 + 0.07 * ids[p];
                        derivatives.rho(p, s) = w * (rho(p, s) + 0.16 * sum);
                        derivatives.tau(p, s) = w * (0.4 * tau(p, s) + 0.048 * sum);
                        for (int x = 0; x < 3; x++) {
                            derivatives.gradient(p, x, s) = 0.06 * w * grad(p, x, s);
                        }
                    }
                }
                if (geometry) {
                    *geometry = reconstruction.geometry_adjoint(smooth_rho, smooth_tau, derivatives);
                }
                if (!matrices) {
                    return e + 0.08 * sum * sum;
                }
                auto adjoint          = reconstruction.adjoint(derivatives);
                *matrices             = std::move(adjoint.local_matrices);
                smooth_tau_derivative = 0;
                if (ctx.gvec().offset() == 0 && ctx.gvec().count() > 0) {
                    for (int s = 0; s < ns__; s++) {
                        smooth_tau_derivative +=
                                ctx.unit_cell().omega() * 0.06 * (s + 1) * adjoint.tau[s].f_pw_local(0).real();
                    }
                }
                ctx.comm().allreduce(&smooth_tau_derivative, 1);
                if (spurious_smooth_q) {
                    for (int j = 0; j < ns__; j++) {
                        host_potential.component(j).rg().zero();
                    }
                    for (int s = 0; s < ns__; s++) {
                        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                            if (ns__ == 4) {
                                host_potential.component(s).rg().f_pw_local(ig) = adjoint.rho[s].f_pw_local(ig);
                            } else {
                                auto v = adjoint.rho[s].f_pw_local(ig) / double(ns__);
                                host_potential.component(0).rg().f_pw_local(ig) += v;
                                if (ns__ == 2) {
                                    host_potential.component(1).rg().f_pw_local(ig) += (1.0 - 2 * s) * v;
                                }
                            }
                        }
                    }
                    host_potential.generate_d_mtrx();
                    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                        auto const& qmatrix = host_potential.d_mtrx(ia);
                        for (int s = 0; s < ns__; s++) {
                            for (int j = 0; j < n; j++) {
                                for (int i = 0; i < n; i++) {
                                    double q = qmatrix(i, j, ns__ == 4 ? s : 0);
                                    if (ns__ == 2) {
                                        q += (1 - 2 * s) * qmatrix(i, j, 1);
                                    }
                                    (*matrices)[ia](i, j, s) += q;
                                }
                            }
                        }
                    }
                }
            }
            return e + 0.08 * sum * sum;
        };
        auto derivative = [&](bool spurious_smooth_q, bool include_smooth_tau) {
            std::vector<mdarray<double, 3>> matrices;
            joint_energy(0, &matrices, spurious_smooth_q);
            double value{0};
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                value += projector_derivative(matrices[ia], ia);
            }
            return value + (include_smooth_tau ? smooth_tau_derivative : 0);
        };
        double analytic    = derivative(false, true);
        double spurious_q  = derivative(true, true);
        double missing_tau = derivative(false, false);
        for (double step : {1e-4, 5e-5}) {
            double fd = (joint_energy(step, nullptr, false) - joint_energy(-step, nullptr, false)) / (2 * step);
            check("joint PAW field map smooth/local rho/tau derivative", std::abs(analytic - fd), 2e-8);
            check("uncancelled smooth-Q adjoint is detected", std::abs(spurious_q - fd) > 1e-5 ? 0 : 1, 0);
            check("missing smooth-tau adjoint is detected", std::abs(missing_tau - fd) > 1e-5 ? 0 : 1, 0);
        }
        xc_geometry_adjoint_t geometry(np, ctx.unit_cell().num_atoms());
        joint_energy(0, nullptr, false, &geometry);
        auto original_points = points;
        double geometry_error{0};
        for (int x = 0; x < 3; x++) {
            double analytic{0};
            for (int p = 0; p < np; p++) {
                analytic += geometry.points[p][x] * (0.2 + 0.03 * ids[p]);
            }
            ctx.comm().allreduce(&analytic, 1);
            for (double h : {1e-7, 5e-8}) {
                auto displaced = [&](double h) {
                    points = original_points;
                    for (int p = 0; p < np; p++) {
                        points[p][x] += h * (0.2 + 0.03 * ids[p]);
                    }
                    return joint_energy(0, nullptr, false);
                };
                double plus = displaced(h), minus = displaced(-h);
                geometry_error = std::max(geometry_error, std::abs((plus - minus) / (2 * h) - analytic) /
                                                                  std::max(1.0, std::abs(analytic)));
            }
        }
        points = original_points;
        check("joint smooth/local geometry point FD", geometry_error, 2e-6);
        PAW_xc_field_map map(density, points);
        xc_point_fields_t bad(np, ns__);
        bad.rho.zero();
        bad.gradient.zero();
        bad.tau.zero();
        if (ctx.comm().rank() == 0) {
            bad.rho(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("field-map collective nonfinite-covector rejection", [&] { map.adjoint(bad); });
        check.reject("geometry collective nonfinite-covector rejection",
                     [&] { map.geometry_adjoint(smooth_rho, smooth_tau, bad); });
        check.reject("field-map collective spin-layout rejection", [&] {
            std::vector<Smooth_periodic_function<double>> empty;
            map.sample(ctx.comm().rank() == 0 ? empty : smooth_rho, smooth_tau);
        });

        PAW_xc_field_map empty_map(density, {});
        auto empty_fields   = empty_map.sample(smooth_rho, smooth_tau);
        auto empty_adjoint  = empty_map.adjoint(empty_fields);
        auto empty_geometry = empty_map.geometry_adjoint(smooth_rho, smooth_tau, empty_fields);
        double empty_error{0};
        for (int s = 0; s < ns__; s++) {
            for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                empty_error +=
                        std::abs(empty_adjoint.rho[s].f_pw_local(ig)) + std::abs(empty_adjoint.tau[s].f_pw_local(ig));
            }
        }
        for (auto const& matrix : empty_adjoint.local_matrices) {
            for (size_t i = 0; i < matrix.size(); i++) {
                empty_error += std::abs(matrix[i]);
            }
        }
        for (size_t i = 0; i < empty_geometry.atoms.size(); i++) {
            empty_error += std::abs(empty_geometry.atoms[i]);
        }
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                empty_error += std::abs(empty_geometry.lattice(x, y));
            }
        }
        check("globally empty field map has zero adjoint", empty_error, 0);
    }
    // A rank-local invalid point must fail collectively rather than strand other ranks.
    check.reject("collective exact-nucleus rejection", [&] {
        std::vector<r3::vector<double>> invalid;
        if (ctx.comm().rank() == 0) {
            invalid.push_back(r3::dot(ctx.unit_cell().lattice_vectors(), ctx.unit_cell().atom(0).position()));
        }
        PAW_xc_reconstruction map(density, invalid);
    });
    return failures;
}

int
test_hydrogen_support(std::string const& path)
{
    auto data       = read_json_from_file(path);
    auto& pp        = data.at("pseudo_potential");
    auto const core = pp.at("paw_data").at("ae_core_charge_density").get<std::vector<double>>();
    if (pp.at("header").at("element") != "H" || pp.at("header").at("z_valence") != 1 ||
        std::any_of(core.begin(), core.end(), [](double x) { return x != 0; })) {
        RTE_THROW("This support fixture requires the core-free hydrogen PAW setup");
    }
    pp["paw_data"]["ae_core_kinetic_density"] = std::vector<double>(core.size(), 0);
    auto conf                                 = R"({"parameters": {"electronic_structure_method": "pseudopotential",
        "use_symmetry": false, "pw_cutoff": 8.0, "gk_cutoff": 2.6,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"]},
        "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    Simulation_context ctx(conf);
    ctx.unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    ctx.electronic_structure_method("pseudopotential");
    auto& type = ctx.unit_cell().add_atom_type("H");
    type.read_input(data);
    ctx.unit_cell().add_atom("H", {0.2, 0.3, 0.4});
    ctx.initialize();
    Density density(ctx);
    PAW_xc_correction op(type);
    auto centre    = r3::dot(ctx.unit_cell().lattice_vectors(), ctx.unit_cell().atom(0).position());
    double support = op.support_radius();
    std::vector<r3::vector<double>> points;
    // The image point also probes continuation below the first radial sample.
    for (double r : {0.1 * type.radial_grid().first(), 0.5 * type.radial_grid().first(), type.radial_grid().first(),
                     support * (1 - 1e-7), support, support * (1 + 1e-7)}) {
        points.push_back(centre + (r / std::sqrt(14.0)) * r3::vector<double>({1, -2, 3}));
    }
    points.push_back(points[0] + r3::dot(ctx.unit_cell().lattice_vectors(), r3::vector<double>({1, 0, 0})));
    int np = static_cast<int>(points.size());
    if (ctx.comm().rank() != 0) {
        points.clear();
    }
    int local = static_cast<int>(points.size());
    mdarray<double, 2> rho({local, 1}), tau({local, 1});
    mdarray<double, 3> grad({local, 3, 1});
    auto& dm = density.density_matrix(0);
    int n    = type.indexb().size();
    double edge_error{0}, image_error{0};
    int failed{0};
    // Probe each projector and every real cross term without choosing an SCF density matrix.
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            dm.zero();
            dm(i, j, 0) = dm(j, i, 0) = 1;
            rho.zero();
            grad.zero();
            tau.zero();
            PAW_xc_reconstruction(density, points).add_to(rho, grad, tau);
            for (int p = 0; p < local; p++) {
                failed |= !std::isfinite(rho(p, 0)) || !std::isfinite(tau(p, 0));
                for (int a = 0; a < 3; a++) {
                    failed |= !std::isfinite(grad(p, a, 0));
                }
                if (p >= 3 && p <= 5) {
                    edge_error = std::max({edge_error, std::abs(rho(p, 0)), std::abs(tau(p, 0))});
                    for (int a = 0; a < 3; a++) {
                        edge_error = std::max(edge_error, std::abs(grad(p, a, 0)));
                    }
                }
            }
            if (local) {
                failed |= i == j && std::abs(rho(0, 0)) < 1e-6;
                image_error = std::max(
                        {image_error, std::abs(rho(0, 0) - rho(np - 1, 0)), std::abs(tau(0, 0) - tau(np - 1, 0))});
                for (int a = 0; a < 3; a++) {
                    image_error = std::max(image_error, std::abs(grad(0, a, 0) - grad(np - 1, a, 0)));
                }
            }
        }
    }
    ctx.comm().allreduce<double, mpi::op_t::max>(&edge_error, 1);
    ctx.comm().allreduce<double, mpi::op_t::max>(&image_error, 1);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (ctx.comm().rank() == 0) {
        std::cout << "H PAW radius=" << support << " boundary rho/gradient/tau=" << edge_error
                  << " inner image difference=" << image_error << '\n';
    }
    return failed || !std::isfinite(edge_error) || edge_error > 2e-10 || !std::isfinite(image_error) ||
           image_error > 2e-10;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    if (argc == 2) {
        result += call_test("H PAW support", test_hydrogen_support, std::string(argv[1]));
    } else {
        for (int ns : {1, 2, 4}) {
            result += call_test("PAW XC correction", test_paw_xc_correction, ns, false);
            result += call_test("compact PAW XC correction", test_paw_xc_correction, ns, true);
        }
    }
    sirius::finalize();
    return result;
}
