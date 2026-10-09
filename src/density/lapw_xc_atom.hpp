/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_XC_ATOM_HPP__
#define __LAPW_XC_ATOM_HPP__

#include <memory>
#include "function3d/kinetic_density.hpp"
#include "unit_cell/atom.hpp"

namespace sirius {

/// Muffin-tin valence fields and the adjoint in the complex orbital or full-j Dirac basis.
/** The radial functions are snapshotted. The atom's radial grid and basis index
 *  must outlive this object. There is no smooth-field subtraction or PAW Q term.
 *  Core charge/tau and their responses are separate from this valence operator.
 */
class LAPW_xc_atom
{
  private:
    Atom_type const& type_;
    int lmmax_;
    Spheric_kinetic_density_operator<std::complex<double>> kinetic_;
    std::unique_ptr<Spheric_kinetic_density_operator<std::complex<double>>> small_;

    static mdarray<double, 2>
    radial_functions(Atom const& atom__, bool derivative__ = false, int component__ = 0)
    {
        auto const& type = atom__.type();
        mdarray<double, 2> result({type.num_mt_points(), type.indexr().size()});
        for (auto const& rf : type.indexr()) {
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                result(ir, rf.idxrf) =
                        derivative__ ? atom__.symmetry_class().radial_function_derivative(ir, rf.idxrf, component__) /
                                               type.radial_grid(ir)
                                     : atom__.symmetry_class().radial_function(ir, rf.idxrf, component__);
                if (!std::isfinite(result(ir, rf.idxrf))) {
                    RTE_THROW("LAPW XC atom: nonfinite radial function");
                }
            }
        }
        return result;
    }

    std::unique_ptr<Spheric_kinetic_density_operator<std::complex<double>>>
    kinetic_response(radial_basis_response_t const* response__, int component__ = 0) const
    {
        if (!response__) {
            return nullptr;
        }
        auto const& rf = response__->radial_functions;
        if (rf.size(0) != type_.num_mt_points() || rf.size(1) != type_.indexr().size() ||
            rf.size(2) != (type_.indexb().full_j() ? 4 : 2)) {
            RTE_THROW("LAPW XC atom: incompatible radial response");
        }
        mdarray<double, 2> radial({rf.size(0), rf.size(1)}), derivative({rf.size(0), rf.size(1)});
        for (int i = 0; i < type_.indexr().size(); i++) {
            for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                radial(ir, i)     = rf(ir, i, 2 * component__);
                derivative(ir, i) = rf(ir, i, 2 * component__ + 1) / type_.radial_grid(ir);
            }
        }
        return std::make_unique<Spheric_kinetic_density_operator<std::complex<double>>>(
                type_.radial_grid(), type_.indexb(), radial, sf::lmax(lmmax_), radial_wave_function_t::R,
                std::vector<int>{}, derivative, false,
                component__ ? spinor_harmonic_t::minus_kappa : spinor_harmonic_t::kappa);
    }

  public:
    LAPW_xc_atom(Atom const& atom__, int lmax__)
        : type_(atom__.type())
        , lmmax_(sf::lmmax(lmax__))
        , kinetic_(type_.radial_grid(), type_.indexb(), radial_functions(atom__), lmax__, radial_wave_function_t::R, {},
                   radial_functions(atom__, true))
    {
        if (type_.indexb().full_j()) {
            small_ = std::make_unique<Spheric_kinetic_density_operator<std::complex<double>>>(
                    type_.radial_grid(), type_.indexb(), radial_functions(atom__, false, 1), lmax__,
                    radial_wave_function_t::R, std::vector<int>{}, radial_functions(atom__, true, 1), false,
                    spinor_harmonic_t::minus_kappa);
        }
    }

    /// Return {rho, tau} per spin, with D_ij = sum_n f_n conj(c_ni) c_nj.
    /** Three input blocks {uu, dd, du} give canonical total/z/x/y fields.
     *  Four Hermitian input blocks already use the total/z/x/y convention.
     *  A full-j basis uses one Hermitian matrix and returns total/z/x/y fields:
     *  charge and positive tau add both Dirac components, whereas magnetization
     *  and its tau counterpart use beta Sigma and subtract the lower contribution.
     *  A supplied radial response returns the field variation at fixed D. */
    std::array<std::vector<Flm>, 2>
    fields(mdarray<std::complex<double>, 3> const& dm__, radial_basis_response_t const* response__ = nullptr) const
    {
        int n       = type_.indexb().size();
        int blocks  = static_cast<int>(dm__.size(2));
        bool spinor = blocks == 3;
        int ns      = type_.indexb().full_j() || spinor ? 4 : blocks;
        if (dm__.size(0) != n || dm__.size(1) != n || blocks < 1 || blocks > 4 ||
            (type_.indexb().full_j() && blocks != 1)) {
            RTE_THROW("LAPW XC atom: incompatible density matrix");
        }
        for (int s = 0; s < blocks; s++) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    auto value = dm__(i, j, s);
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
                        ((!spinor || s < 2) &&
                         std::abs(value - std::conj(dm__(j, i, s))) > 1e-12 * (1 + std::abs(value)))) {
                        RTE_THROW("LAPW XC atom: nonfinite or non-Hermitian density matrix");
                    }
                }
            }
        }
        auto tangent       = kinetic_response(response__);
        auto small_tangent = small_ ? kinetic_response(response__, 1) : nullptr;
        std::array<std::vector<Flm>, 2> result;
        mdarray<std::complex<double>, 2> dm({n, n});
        for (int s = 0; s < ns; s++) {
            result[0].emplace_back(lmmax_, type_.radial_grid());
            result[1].emplace_back(lmmax_, type_.radial_grid());
            auto& rho = result[0].back();
            rho.zero();
            result[1].back().zero();
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    if (!spinor) {
                        dm(i, j) = dm__(i, j, small_ ? 0 : s);
                    } else if (s < 2) {
                        dm(i, j) = dm__(i, j, 0) + (1.0 - 2 * s) * dm__(i, j, 1);
                    } else {
                        auto du = dm__(i, j, 2), ud = std::conj(dm__(j, i, 2));
                        dm(i, j) = s == 2 ? du + ud : std::complex<double>(0, 1) * (du - ud);
                    }
                }
            }
            int component = small_ ? s : 0;
            kinetic_.add_charge_density(dm, rho, tangent.get(), component);
            kinetic_.add_density(dm, result[1].back(), tangent.get(), component);
            if (small_) {
                if (s) {
                    for (auto& value : dm) {
                        value = -value;
                    }
                }
                small_->add_charge_density(dm, rho, small_tangent.get(), s);
                small_->add_density(dm, result[1].back(), small_tangent.get(), s);
            }
        }
        return result;
    }

    /// Complex Hermitian matrix from already-weighted radial-sample covectors.
    /** No radial quadrature, r^2 factor or occupation is applied again.
     *  If a radial response is supplied, return the matrix variation at fixed covectors. */
    mdarray<std::complex<double>, 2>
    matrix_elements(Flm const& drho__, Flm const& dtau__, radial_basis_response_t const* response__ = nullptr,
                    int component__ = 0) const
    {
        if (drho__.angular_domain_size() != lmmax_ || drho__.radial_grid().hash() != type_.radial_grid().hash()) {
            RTE_THROW("LAPW XC atom: incompatible charge covector");
        }
        auto tangent = kinetic_response(response__);
        auto result  = kinetic_.matrix_elements_from_derivatives(dtau__, tangent.get(), component__);
        auto charge  = kinetic_.charge_matrix_elements_from_derivatives(drho__, tangent.get(), component__);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += charge[i];
        }
        if (small_) {
            auto small_tangent = kinetic_response(response__, 1);
            auto lower_tau     = small_->matrix_elements_from_derivatives(dtau__, small_tangent.get(), component__);
            auto lower_rho = small_->charge_matrix_elements_from_derivatives(drho__, small_tangent.get(), component__);
            for (size_t i = 0; i < result.size(); i++) {
                result[i] += (component__ ? -1.0 : 1.0) * (lower_rho[i] + lower_tau[i]);
            }
        }
        return result;
    }
};

} // namespace sirius

#endif
