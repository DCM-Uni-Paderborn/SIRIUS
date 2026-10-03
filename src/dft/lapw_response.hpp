/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_RESPONSE_HPP__
#define __LAPW_RESPONSE_HPP__

#include "density/density.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

/// Linearized electronic SCF map, including Hartree/XC, orbitals, occupations and radial functions.
/** CPU fp64, nonrelativistic scalar/collinear LAPW with an external XC response.
 *  The reference potential, radial basis, bands and density must remain unchanged.
 *  All operations are context-collective and use scratch fields only. Geometry
 *  derivatives are explicit right-hand sides, not part of this fixed-geometry map.
 */
class LAPW_response
{
  private:
    K_point_set& ks_;
    Density const& density_;
    Potential& potential_;
    std::unique_ptr<Hamiltonian0<double>> h0_;
    std::unique_ptr<LAPW_density_response> spectrum_;
    std::vector<mdarray<std::complex<double>, 3>> matrices_;

    std::unique_ptr<Density>
    potential_response(Periodic_function<double> const& hartree__, lapw_xc_field_adjoint_t const& xc__,
                       xc_geometry_direction_t const* geometry__ = nullptr, double* entropy__ = nullptr) const;

  public:
    struct result_t
    {
        std::unique_ptr<Density> fields;
        int iterations;
        double residual;
        double entropy;
    };

    struct radial_geometry_response_t
    {
        std::vector<std::vector<double>> scalar;
        std::vector<std::array<std::vector<double>, 2>> core;
        std::unique_ptr<Density> fields;
        double entropy{0};
    };

    LAPW_response(K_point_set& ks__, Density const& density__, Potential& potential__);

    /// Induced fields J*x. Does not include an explicit perturbation right-hand side.
    /** Directions use synchronized total/magnetization rho/tau fields. */
    std::unique_ptr<Density>
    apply(Density const& direction__, double* entropy__ = nullptr) const;

    /// Explicit SCF-map geometry derivative at fixed input rho/tau coefficients.
    /** Includes Poisson/XC sampling, radial functions, PW/LO matrices, occupations
     *  and the output field map. Requires the host quadrature tangent and joint XC
     *  Hessian. MT radii and point-domain membership remain fixed. Use solve to
     *  include the induced SCF response.
     */
    std::unique_ptr<Density>
    geometry_rhs(xc_geometry_direction_t const& direction__, double* entropy__ = nullptr) const;

    /// Solve (I-J)*x=rhs by stable Anderson mixing with an independently checked residual.
    /** The residual is the maximum fieldwise infinity norm of rhs+J*x-x,
     *  divided by max(||rhs||,||x||,||rhs+J*x||,1e-12) for each rho/tau component.
     *  Failure to reach tolerance throws; it never returns an unconverged solution.
     */
    result_t
    solve(Density const& rhs__, double tolerance__, int max_iterations__, int history__ = 8, double beta__ = 0.5) const;

    /// Total radial-solver potential derivative, including explicit geometry and induced SCF fields.
    /** Also returns the complete physical rho/tau and entropy derivatives.
     *  Uses the same checked response solve as solve(); the reference state is unchanged. */
    radial_geometry_response_t
    radial_geometry_response(xc_geometry_direction_t const& direction__, double tolerance__,
                             int max_iterations__) const;

    /// Derivative of the discrete free energy using the complete SCF field response and explicit XC geometry covectors.
    double
    energy_geometry_derivative(xc_geometry_direction_t const& direction__, radial_geometry_response_t const& response__,
                               xc_geometry_derivatives_t const& explicit_xc__) const;
};

} // namespace sirius

#endif
