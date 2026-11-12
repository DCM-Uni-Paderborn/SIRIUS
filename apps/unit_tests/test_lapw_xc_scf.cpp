/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "dft/dft_ground_state.hpp"

using namespace sirius;

int
test_lapw_xc_scf()
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
    cell.set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
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
    cell.add_atom("He", {0.23, 0.31, 0.43});
    auto points     = std::make_shared<xc_point_set_t>(ctx);
    constexpr int n = 10;
    for (int z = 0; z < n; ++z) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                points->coordinates.push_back({6 * (x + 0.37) / n, 7 * (y + 0.29) / n, 8 * (z + 0.41) / n});
                points->energy_weights.push_back(336.0 / (n * n * n));
                points->descriptor_weights.push_back(336.0 / (n * n * n));
            }
        }
    }
    points->block_sizes          = {n * n * n};
    points->block_centres        = {{1.38, 2.17, 3.44}};
    ctx.cb().external_xc_points_ = points;
    ctx.cb().external_xc_ = [](Simulation_context const&, xc_callback_input_t const& in, xc_callback_output_t& out) {
        for (int i = 0; i < in.num_points; ++i) {
            double w    = in.points->energy_weights[i];
            out.vrho[i] = -0.1 * w * in.rho[i];
            out.vtau[i] = 0.03 * w * in.tau[i];
            out.energy += 0.5 * (in.rho[i] * out.vrho[i] + in.tau[i] * out.vtau[i]);
        }
    };
    ctx.initialize();
    K_point_set ks(ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto const result     = gs.find(1e-9, 1e-10, 1e-11, 120, false);
    auto const& potential = gs.potential();
    auto c                = potential.lapw_xc_contractions(gs.density());
    auto energy           = potential.energy_exc(gs.density());
    if (!result.at("converged").get<bool>() || !std::isfinite(energy) || std::abs(c.tau) < 1e-8 ||
        std::abs(c.rho + c.tau - 2 * energy) > 1e-10) {
        return 1;
    }
    return 0;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    auto status = test_lapw_xc_scf();
    sirius::finalize();
    return status;
}
