/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef SIRIUS_XC_DERIVATIVE_CHECKS_HPP
#define SIRIUS_XC_DERIVATIVE_CHECKS_HPP

#include "dft/dft_ground_state.hpp"

namespace sirius::testing {

/// Hamiltonian derivative and central differences at step and step/2 for two orbital rotations.
/** Geometry, occupations and core states are fixed. Orbitals and callbacks are restored on failure. */
std::array<std::array<double, 3>, 2>
check_external_xc_derivative(DFT_ground_state& gs__, double step__);

/// Fixed-basis PW/LO orbital check for ordinary or external LAPW XC.
std::array<std::array<double, 3>, 2>
check_lapw_xc_derivative(DFT_ground_state& gs__, double step__);

/// Finite-difference check of all seven independent point-input Hessian blocks.
std::array<double, 2>
check_external_xc_point_hessian(Potential const& potential__, Density const& density__, double step__);

/// Relative XC field, radial-potential and enabled point-input response errors at step and step/2.
/** Uses scratch fields without modifying the ground state. */
std::array<double, 2>
check_lapw_xc_response(DFT_ground_state& gs__, double step__);

} // namespace sirius::testing

#endif
