/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "dft/dft_ground_state.hpp"
#include "dft/energy.hpp"

using namespace sirius;

std::array<double, 3>
calculate(double displacement, double strain, bool derivatives)
{
    Simulation_context ctx(R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
        "pw_cutoff": 5.0, "gk_cutoff": 1.5, "num_bands": 4, "xc_functionals": [],
        "core_relativity": "none", "valence_relativity": "none", "auto_rmt": 0,
        "lmax_apw": 3, "lmax_rho": 6, "lmax_pot": 6, "smearing_width": 0.01
    }, "control": {"processing_unit": "CPU", "verbosity": 0, "use_second_variation": false},
    "iterative_solver": {"type": "exact", "min_tolerance": 1e-11},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false}})"_json);
    auto& cell = ctx.unit_cell();
    double lx  = 6 * (1 + strain);
    cell.set_lattice_vectors({{lx, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    auto& type = cell.add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::power, 801, 1e-6, 1.8, 2.0);
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
    std::vector<double> rho(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ++ir) {
        rho[ir] = 16 / pi * std::exp(-4 * grid[ir]);
    }
    type.free_atom_density(rho);
    type.set_free_atom_radial_grid(std::move(grid));
    for (int l = 0; l <= 3; ++l) {
        type.add_aw_descriptor(l + 1, l, -0.3, 0, 0);
        type.add_aw_descriptor(l + 1, l, -0.3, 1, 0);
    }
    type.add_lo_descriptor(0, 1, 0, -0.7, 0, 0);
    type.add_lo_descriptor(0, 1, 0, -0.7, 1, 0);
    double x_he = 0.23 * lx + displacement;
    cell.add_atom("He", {x_he / lx, 0.31, 0.43});
    auto points     = std::make_shared<xc_point_set_t>(ctx);
    constexpr int n = 10;
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                points->coordinates.push_back({lx * (x + 0.37) / n, 7 * (y + 0.29) / n, 8 * (z + 0.41) / n});
                points->energy_weights.push_back(lx * 56 / (n * n * n));
                points->descriptor_weights.push_back(lx * 56 / (n * n * n));
            }
        }
    }
    points->block_sizes          = {n * n * n};
    points->block_centres        = {{x_he, 2.17, 3.44}};
    ctx.cb().external_xc_points_ = points;
    ctx.cb().external_xc_ = [](Simulation_context const&, xc_callback_input_t const& in, xc_callback_output_t& out) {
        for (int i = 0; i < in.num_points; ++i) {
            double w    = in.points->energy_weights[i];
            out.vrho[i] = -0.1 * w * in.rho[i];
            out.vtau[i] = 0.03 * w * in.tau[i];
            out.energy += 0.5 * (in.rho[i] * out.vrho[i] + in.tau[i] * out.vtau[i]);
            if (out.geometry) {
                out.geometry->energy_weights[i] = 0.5 * (-0.1 * in.rho[i] * in.rho[i] + 0.03 * in.tau[i] * in.tau[i]);
            }
        }
    };
    ctx.cb().external_xc_geometry_          = ctx.cb().external_xc_;
    ctx.cb().external_xc_geometry_response_ = true;
    ctx.cb().external_xc_response_          = [](auto const&, auto const& in, auto const& delta, auto& out) {
        for (int i = 0; i < in.num_points; ++i) {
            double w = in.points->energy_weights[i], dw = delta.geometry ? delta.geometry->energy_weights[i] : 0;
            out.vrho[i] = -0.1 * (w * delta.rho[i] + dw * in.rho[i]);
            out.vtau[i] = 0.03 * (w * delta.tau[i] + dw * in.tau[i]);
            if (out.geometry) {
                out.geometry->energy_weights[i] = -0.1 * in.rho[i] * delta.rho[i] + 0.03 * in.tau[i] * delta.tau[i];
            }
        }
    };
    // Fractional sampling points follow the cell while the block centre follows the atom.
    ctx.cb().external_xc_quadrature_direction_ = [](auto const& c, auto const& q, auto const& in, auto& out) {
        auto const& inverse = c.unit_cell().inverse_lattice_vectors();
        r3::matrix<double> dlattice;
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                dlattice(i, j) = in.lattice[i + 3 * j];
            }
        }
        auto deformation = dot(dlattice, inverse);
        double trace     = deformation(0, 0) + deformation(1, 1) + deformation(2, 2);
        for (size_t p = 0; p < q.coordinates.size(); ++p) {
            auto const& r             = q.coordinates[p];
            auto dr                   = dot(deformation, r3::vector<double>{r[0], r[1], r[2]});
            out.coordinates[p]        = {dr[0], dr[1], dr[2]};
            out.energy_weights[p]     = trace * q.energy_weights[p];
            out.descriptor_weights[p] = trace * q.descriptor_weights[p];
        }
        out.block_centres[0] = in.atoms[0];
    };
    ctx.cb().external_xc_pullback_ = [](auto const& c, auto const& q, auto const& in, auto& out) {
        auto const& inverse = c.unit_cell().inverse_lattice_vectors();
        for (size_t p = 0; p < q.coordinates.size(); ++p) {
            auto const& r = q.coordinates[p];
            auto f        = dot(inverse, r3::vector<double>{r[0], r[1], r[2]});
            double measure =
                    q.energy_weights[p] * in.energy_weights[p] + q.descriptor_weights[p] * in.descriptor_weights[p];
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) {
                    out.lattice[i + 3 * j] += in.coordinates[p][i] * f[j] + measure * inverse(j, i);
                }
            }
        }
        out.atoms[0] = in.block_centres[0];
    };
    ctx.initialize();
    K_point_set ks(ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto const result     = gs.find(1e-11, 1e-12, 1e-11, 160, false);
    auto const& potential = gs.potential();
    auto c                = potential.lapw_xc_contractions(gs.density());
    auto energy           = potential.energy_exc(gs.density());
    if (!result.at("converged").get<bool>() || !std::isfinite(energy) || std::abs(c.tau) < 1e-8 ||
        std::abs(c.rho + c.tau - 2 * energy) > 1e-10) {
        RTE_THROW("LAPW point XC SCF or XC contraction check failed");
    }
    std::array<double, 3> values{free_energy(ctx, ks, gs.density(), potential), 0, 0};
    if (derivatives) {
        values[1] = -gs.forces().calc_forces_total()(0, 0);
        values[2] = cell.omega() * gs.stress().calc_stress_total()(0, 0);
    }
    return values;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    auto reference     = calculate(0, 0, true);
    constexpr double h = 1e-4;
    double force_fd    = (calculate(h, 0, false)[0] - calculate(-h, 0, false)[0]) / (2 * h);
    double stress_fd   = (calculate(0, h, false)[0] - calculate(0, -h, false)[0]) / (2 * h);
    double error       = std::max(std::abs(reference[1] - force_fd), std::abs(reference[2] - stress_fd));
    std::cout << "FP-LAPW XC force and stress finite-difference error: " << error << '\n';
    auto status = !std::isfinite(error) || error > 2e-6;
    sirius::finalize();
    return status;
}
