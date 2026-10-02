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

  public:
    struct result_t
    {
        std::unique_ptr<Density> fields;
        int iterations;
        double residual;
    };

    LAPW_response(K_point_set& ks__, Density const& density__, Potential& potential__);

    /// Induced fields J*x. Does not include an explicit perturbation right-hand side.
    /** Directions use synchronized total/magnetization rho/tau fields. */
    std::unique_ptr<Density>
    apply(Density const& direction__) const;

    /// Solve (I-J)*x=rhs by stable Anderson mixing with an independently checked residual.
    /** The residual is the maximum fieldwise infinity norm of rhs+J*x-x,
     *  divided by max(||rhs||,||x||,||rhs+J*x||,1e-12) for each rho/tau component.
     *  Failure to reach tolerance throws; it never returns an unconverged solution.
     */
    result_t
    solve(Density const& rhs__, double tolerance__, int max_iterations__, int history__ = 8, double beta__ = 0.5) const;
};

} // namespace sirius

#endif
