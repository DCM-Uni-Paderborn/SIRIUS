/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "radial/radial_core_solver.hpp"
#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_radial_core_xc()
{
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 4001, 1e-8, 24.0, 3.0);
    std::vector<double> v(grid.num_points()), zero(v.size(), 0), dt(v.size());
    for (int i = 0; i < grid.num_points(); i++) {
        double r = grid[i];
        v[i]     = -2 / r;
        dt[i]    = 4 * pi * r * r * grid.dx(std::min(i, grid.num_points() - 2)) * std::exp(-r - 1 / r);
    }
    Radial_core_solver solver(grid, 0, 2, v, 128);
    auto state         = solver.solve(1, zero, zero).front();
    constexpr double h = 1e-4;
    auto plus = dt, minus = dt;
    for (size_t i = 0; i < dt.size(); i++) {
        plus[i] *= h;
        minus[i] *= -h;
    }
    auto response = solver.solve_response(1, zero, zero, zero, zero, dt).front();
    auto ep       = solver.solve(1, zero, plus).front().energy;
    auto em       = solver.solve(1, zero, minus).front().energy;
    double error =
            std::max({std::abs(state.energy + 2), std::abs(fourpi * Spline<double>(grid, state.rho).integrate(2) - 1),
                      std::abs(response.derivative.energy - (ep - em) / (2 * h))});
    std::cout << "Radial core normalization, energy and kinetic-potential response error: " << error << '\n';
    return !std::isfinite(error) || error > 2e-8;
}

int
main()
{
    sirius::initialize(1);
    int error = call_test("radial core XC", test_radial_core_xc);
    sirius::finalize();
    return error;
}
