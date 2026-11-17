/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_CALLBACK_HPP__
#define __XC_CALLBACK_HPP__

#include <functional>
#include <memory>
#include <array>
#include <vector>

namespace sirius {

class Simulation_context;
class XC_point_symmetry;

/// A fixed distributed quadrature, grouped into complete local atom blocks.
/** Coordinates and block centres are Cartesian bohr. Blocks and their rows
 *  belong to exactly one context rank, independent of FFT replicas. Energy and
 *  descriptor weights are distinct and never multiply the physical fields.
 *  Construct after defining the cell/atoms, populate the arrays, then register
 *  as immutable callback state. Geometry changes require a new point set.
 */
struct xc_point_set_t
{
    std::vector<std::array<double, 3>> coordinates;
    std::vector<double> energy_weights;
    std::vector<double> descriptor_weights;
    std::vector<int> block_sizes;
    std::vector<std::array<double, 3>> block_centres;

    explicit xc_point_set_t(Simulation_context const& ctx__);
    xc_point_set_t(xc_point_set_t const& other);
    xc_point_set_t&
    operator=(xc_point_set_t const& other);
    void
    validate(Simulation_context const& ctx__) const;
    XC_point_symmetry const&
    symmetry(Simulation_context const& ctx__) const;

  private:
    std::array<double, 9> lattice_;
    std::vector<std::array<double, 3>> atoms_;
    mutable std::shared_ptr<XC_point_symmetry> symmetry_;
};

/// Primitive fields on the local FFT slab or explicit point rows, in atomic units.
/** Scalar arrays have Fortran shape (num_points, num_spins), gradients
 *  (num_points, 3, num_spins). One spin component denotes total density/tau,
 *  two components denote up/down. The neutralizing PP background is excluded.
 *  No density clipping or renormalization is applied. Explicit spinor mode supplies
 *  four total/z/x/y components for each field; num_spins then denotes this component count, not occupations.
 *  The context supplies the cell, atoms, FFT layout and communicators.
 */
struct xc_callback_input_t
{
    int num_points;
    int num_spins;
    double point_weight;
    double const* rho;
    double const* gradient;
    double const* tau;
    /// Non-null for arbitrary points. point_weight is then unused.
    xc_point_set_t const* points{nullptr};
};

/// Local raw derivatives with respect to the independent quadrature inputs.
struct xc_point_derivatives_t
{
    std::vector<std::array<double, 3>> coordinates, block_centres;
    std::vector<double> energy_weights, descriptor_weights;

    explicit xc_point_derivatives_t(xc_point_set_t const& points__)
        : coordinates(points__.coordinates.size())
        , block_centres(points__.block_centres.size())
        , energy_weights(points__.energy_weights.size())
        , descriptor_weights(points__.descriptor_weights.size())
    {
    }
};

/// Globally summed energy derivatives, not force or stress components.
/** Cell columns vary at fixed Cartesian atoms. No sign or volume factor applies. */
struct xc_geometry_derivatives_t
{
    std::vector<std::array<double, 3>> atoms;
    std::array<double, 9> lattice{};

    explicit xc_geometry_derivatives_t(int num_atoms__)
        : atoms(num_atoms__)
    {
    }
};

/// Replicated Cartesian atom and lattice-column directions, in bohr.
/** Atom and cell directions are independent; no affine atom displacement is implied. */
struct xc_geometry_direction_t
{
    std::vector<std::array<double, 3>> atoms;
    std::array<double, 9> lattice{};

    explicit xc_geometry_direction_t(int num_atoms__)
        : atoms(num_atoms__)
    {
    }
};

/// Total XC energy and primitive-field derivatives.
/** energy is the same globally reduced energy on every context rank.
 *  For the FFT callback, derivatives are divided by the grid quadrature weight.
 *  With input.points != nullptr, return raw discrete covectors, NOT derivatives
 *  divided by a quadrature weight. Zero energy-weight rows may have nonzero
 *  derivatives through the nonlocal descriptors.
 *  vrho/vtau and vgradient have the corresponding input shapes.
 *  The caller initializes all outputs to zero.
 *  A nonlocal evaluator must return derivatives of the entire discrete energy,
 *  not derivatives of independent local energy terms. SIRIUS applies the
 *  gradient adjoint and the tau operator; the callback must not apply them.
 */
struct xc_callback_output_t
{
    double energy{0};
    double* vrho;
    double* vgradient;
    double* vtau;
    /// Non-null only for the geometry callback. Fields and weights are independent inputs.
    xc_point_derivatives_t* geometry{nullptr};
};

/// Called collectively on the context communicator, including empty local work.
/** Register before constructing Density/Potential and keep it fixed during SCF. */
using xc_callback_t = std::function<void(Simulation_context const&, xc_callback_input_t const&, xc_callback_output_t&)>;

/// Independent input direction, with the shapes and spin convention of the base input.
struct xc_callback_direction_t
{
    double const* rho;
    double const* gradient;
    double const* tau;
    /// Optional quadrature direction; absent means fixed points, weights and block centres.
    xc_point_derivatives_t const* geometry{nullptr};
};

/// Directional derivatives of the XC covectors.
/** Return the Hessian of the entire discrete energy applied to the direction.
 *  Point callbacks return raw covectors, including nonlocal coupling between
 *  all points, spins and ranks. No gradient adjoint or tau operator is applied
 *  here. Outputs are initialized to zero by the caller.
 */
struct xc_callback_response_t
{
    double* vrho;
    double* vgradient;
    double* vtau;
    /// Optional raw quadrature-covector response, with no sampling or geometry pullback.
    xc_point_derivatives_t* geometry{nullptr};
};

using xc_response_callback_t = std::function<void(Simulation_context const&, xc_callback_input_t const&,
                                                  xc_callback_direction_t const&, xc_callback_response_t&)>;

/// Pull back local point covectors, including field sampling, to replicated atom/cell derivatives.
using xc_quadrature_pullback_t = std::function<void(Simulation_context const&, xc_point_set_t const&,
                                                    xc_point_derivatives_t const&, xc_geometry_derivatives_t&)>;

/// Differentiate the host's distributed quadrature at fixed row topology and radial/angular rules.
using xc_quadrature_direction_callback_t = std::function<void(Simulation_context const&, xc_point_set_t const&,
                                                              xc_geometry_direction_t const&, xc_point_derivatives_t&)>;

void
validate_external_xc_context(Simulation_context const& ctx__);

} // namespace sirius

#endif
