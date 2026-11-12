/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPHERIC_KINETIC_DENSITY_HPP__
#define __SPHERIC_KINETIC_DENSITY_HPP__

#include <type_traits>
#include "core/sht/gaunt.hpp"
#include "core/sht/sht.hpp"
#include "function3d/spheric_function.hpp"
#include "unit_cell/radial_functions_index.hpp"
#include "unit_cell/basis_functions_index.hpp"

namespace sirius {

enum class radial_wave_function_t
{
    R,
    r_times_R
};

enum class spinor_harmonic_t
{
    kappa,
    minus_kappa
};

/// Atom-local valence rho/tau and their variational matrices in a spherical orbital or spinor basis.
/** T=double selects real harmonics (PAW), T=complex<double> complex harmonics
 *  (LAPW). The density and v_tau are always expanded in real harmonics.
 *  The Hermitian density matrix uses D_ij = sum_n f_n conj(c_ni) c_nj.
 *  Core-state tau must be supplied separately. PAW charge-compensation functions
 *  do not specify orbital gradients and are not included. The grid and basis
 *  index must outlive this class. For PAW SOC, the projected density matrix is
 *  already rotated into scalar spatial harmonics; j labels then select radial
 *  functions, while spin mixing resides in the density-matrix components.
 *  A full-j complex-harmonic basis contracts both components of R_l(r) Omega_kappa,mj,
 *  with separate radial functions for the j=l+/-1/2 channels.
 *  Its canonical field components are total/z/x/y, numbered 0/1/2/3.
 *  The small Dirac component uses minus_kappa with Q/r radial functions.
 *  The caller combines large and small contributions with the Dirac charge or magnetic weights;
 *  no Dirac phase is inserted into these block-diagonal bilinears.
 */
template <typename T>
class Spheric_kinetic_density_operator
{
  private:
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>);

    Radial_grid<double> const& grid_;
    basis_functions_index const& indexb_;
    int lmmax_;
    spinor_harmonic_t harmonic_;
    Gaunt_coefficients<T> gaunt_;
    mdarray<std::vector<gaunt_L3<T>>, 3> spinor_gaunt_;
    mdarray<double, 2> radial_;
    mdarray<double, 2> derivative_;
    mdarray<double, 2> angular_radial_;

    static int
    angular_l(angular_momentum am__, spinor_harmonic_t harmonic__)
    {
        return harmonic__ == spinor_harmonic_t::minus_kappa ? am__.two_j() - am__.l() : am__.l();
    }

    static int
    angular_lmax(basis_functions_index const& indexb__, spinor_harmonic_t harmonic__)
    {
        int result{-1};
        for (auto const& rf : indexb__.indexr()) {
            result = std::max(result, angular_l(rf.am, harmonic__));
        }
        return result;
    }

    static T
    gaunt(int l1, int l3, int l2, int m1, int m3, int m2)
    {
        if constexpr (std::is_same_v<T, double>) {
            return SHT::gaunt_rrr(l1, l3, l2, m1, m3, m2);
        } else {
            return SHT::gaunt_yry(l1, l3, l2, m1, m3, m2);
        }
    }

    std::vector<gaunt_L3<T>> const&
    angular_products(int i__, int j__, int component__) const
    {
        return indexb_.full_j() ? spinor_gaunt_(i__, j__, component__)
                                : gaunt_.gaunt_vector(indexb_[i__].lm, indexb_[j__].lm);
    }

    void
    check_component(int component__) const
    {
        if (component__ < 0 || component__ >= (indexb_.full_j() ? 4 : 1)) {
            RTE_THROW("atom-local kinetic density: invalid spin component");
        }
    }

    static T
    spin_weight(int component__, int bra__, int ket__)
    {
        switch (component__) {
            case 0:
                return bra__ == ket__ ? 1 : 0;
            case 1:
                return bra__ == ket__ ? 1 - 2 * bra__ : 0;
            case 2:
                return bra__ != ket__ ? 1 : 0;
            default:
                if constexpr (std::is_same_v<T, std::complex<double>>) {
                    return bra__ != ket__ ? T(0, 2 * bra__ - 1) : T(0);
                }
                return 0;
        }
    }

    double
    radial_product(int i__, int j__, int L__, int ir__, Spheric_kinetic_density_operator const* response__ = nullptr,
                   bool charge__ = false) const
    {
        auto const& a = indexb_[i__];
        auto const& b = indexb_[j__];
        if (charge__) {
            return response__ ? radial_(ir__, a.idxrf) * response__->radial_(ir__, b.idxrf) +
                                        response__->radial_(ir__, a.idxrf) * radial_(ir__, b.idxrf)
                              : radial_(ir__, a.idxrf) * radial_(ir__, b.idxrf);
        }
        int l1 = angular_l(a.am, harmonic_), l2 = angular_l(b.am, harmonic_);
        double angular = 0.5 * (l1 * (l1 + 1) + l2 * (l2 + 1) - L__ * (L__ + 1));
        if (response__) {
            return 0.5 * (derivative_(ir__, a.idxrf) * response__->derivative_(ir__, b.idxrf) +
                          response__->derivative_(ir__, a.idxrf) * derivative_(ir__, b.idxrf) +
                          angular * (angular_radial_(ir__, a.idxrf) * response__->angular_radial_(ir__, b.idxrf) +
                                     response__->angular_radial_(ir__, a.idxrf) * angular_radial_(ir__, b.idxrf)));
        }
        return 0.5 * (derivative_(ir__, a.idxrf) * derivative_(ir__, b.idxrf) +
                      angular * angular_radial_(ir__, a.idxrf) * angular_radial_(ir__, b.idxrf));
    }

    void
    check_field(Spheric_function<function_domain_t::spectral, double> const& f__) const
    {
        if (f__.angular_domain_size() != lmmax_ || f__.radial_grid().hash() != grid_.hash()) {
            RTE_THROW("atom-local kinetic density: incompatible spherical field");
        }
    }

  public:
    Spheric_kinetic_density_operator(Radial_grid<double> const& grid__, basis_functions_index const& indexb__,
                                     mdarray<double, 2> const& radial__, int lmax__,
                                     radial_wave_function_t storage__        = radial_wave_function_t::R,
                                     std::vector<int> const& radial_sizes__  = {},
                                     mdarray<double, 2> const& derivatives__ = {}, bool soc_rotated__ = false,
                                     spinor_harmonic_t harmonic__ = spinor_harmonic_t::kappa)
        : grid_(grid__)
        , indexb_(indexb__)
        , lmmax_(sf::lmmax(lmax__))
        , harmonic_(harmonic__)
        , gaunt_(angular_lmax(indexb__, harmonic__), lmax__, angular_lmax(indexb__, harmonic__), gaunt)
        , radial_({grid__.num_points(), indexb__.indexr().size()})
        , derivative_({grid__.num_points(), indexb__.indexr().size()})
        , angular_radial_({grid__.num_points(), indexb__.indexr().size()})
    {
        if (harmonic_ != spinor_harmonic_t::kappa &&
            (harmonic_ != spinor_harmonic_t::minus_kappa || !indexb_.full_j())) {
            RTE_THROW("opposite-kappa kinetic density requires a full-j basis");
        }
        if (indexb_.full_j()) {
            if (!std::is_same_v<T, std::complex<double>> || soc_rotated__) {
                RTE_THROW("a full-j kinetic-density basis requires unrotated complex spinor harmonics");
            }
            spinor_gaunt_ = mdarray<std::vector<gaunt_L3<T>>, 3>({indexb_.size(), indexb_.size(), 4});
            auto l_by_lm  = sf::l_by_lm(lmax__);
            for (int j = 0; j < indexb_.size(); j++) {
                for (int i = 0; i <= j; i++) {
                    for (int component = 0; component < 4; component++) {
                        std::vector<T> coefficients(lmmax_, 0);
                        for (int bra = 0; bra < 2; bra++) {
                            auto [lm1, c1] =
                                    indexb_[i].spinor_component(bra, harmonic_ == spinor_harmonic_t::minus_kappa);
                            for (int ket = 0; ket < 2; ket++) {
                                auto [lm2, c2] =
                                        indexb_[j].spinor_component(ket, harmonic_ == spinor_harmonic_t::minus_kappa);
                                auto weight = spin_weight(component, bra, ket);
                                if (c1 == 0 || c2 == 0 || weight == T(0)) {
                                    continue;
                                }
                                for (auto const& g : gaunt_.gaunt_vector(lm1, lm2)) {
                                    coefficients[g.lm3] += weight * c1 * c2 * g.coef;
                                }
                            }
                        }
                        for (int lm = 0; lm < lmmax_; lm++) {
                            if (std::abs(coefficients[lm]) != 0) {
                                spinor_gaunt_(i, j, component).push_back({lm, l_by_lm[lm], coefficients[lm]});
                            }
                        }
                    }
                }
            }
        }
        if (grid_.num_points() < 4 || grid_.first() < 0 || radial__.size(0) != grid_.num_points() ||
            radial__.size(1) != indexb_.indexr().size() ||
            (!radial_sizes__.empty() && radial_sizes__.size() != radial__.size(1))) {
            RTE_THROW("atom-local kinetic density: invalid radial functions or grid");
        }
        if (derivatives__.size() &&
            (storage__ != radial_wave_function_t::R || derivatives__.size(0) != radial__.size(0) ||
             derivatives__.size(1) != radial__.size(1))) {
            RTE_THROW("atom-local kinetic density: incompatible supplied radial derivatives");
        }
        radial_.zero();
        derivative_.zero();
        angular_radial_.zero();
        for (auto const& rf : indexb_.indexr()) {
            int l = angular_l(rf.am, harmonic_);
            if (rf.am.s() != 0 && !indexb_.full_j() && !(soc_rotated__ && std::is_same_v<T, double>)) {
                RTE_THROW("spin-orbit partial waves require a spinor kinetic-density operator");
            }
            int nr = radial_sizes__.empty() ? grid_.num_points() : radial_sizes__[rf.idxrf];
            if (nr < 4 || nr > grid_.num_points()) {
                RTE_THROW("atom-local kinetic density: invalid radial support");
            }
            // LAPW already stores the derivatives used by the kinetic Hamiltonian.
            if (derivatives__.size()) {
                for (int ir = 0; ir < nr; ir++) {
                    double value = radial__(ir, rf.idxrf), derivative = derivatives__(ir, rf.idxrf);
                    if (!std::isfinite(value) || !std::isfinite(derivative) ||
                        (grid_[ir] == 0 && l > 0 && value != 0)) {
                        RTE_THROW("atom-local kinetic density: invalid supplied radial values or derivatives");
                    }
                    derivative_(ir, rf.idxrf)     = derivative;
                    radial_(ir, rf.idxrf)         = value;
                    angular_radial_(ir, rf.idxrf) = l == 0           ? 0
                                                    : grid_[ir] != 0 ? value / grid_[ir]
                                                    : l == 1         ? derivative
                                                                     : 0;
                }
                continue;
            }
            // Differentiate only the supplied partial wave, not its artificial zero-padded tail.
            Radial_grid_ext<double> radial_grid(nr, grid_.x().at(memory_t::host));
            Spline<double> f(radial_grid);
            for (int ir = 0; ir < nr; ir++) {
                f(ir) = radial__(ir, rf.idxrf);
            }
            if (storage__ == radial_wave_function_t::r_times_R) {
                if (grid_.first() == 0 && f(0) != 0) {
                    RTE_THROW("r R(r) must vanish at the origin");
                }
                double origin{0};
                if (grid_.first() == 0) {
                    origin = f.interpolate().deriv(1, 0);
                }
                for (int ir = 0; ir < nr; ir++) {
                    f(ir) = grid_[ir] == 0 ? (l == 0 ? origin : 0) : f(ir) / grid_[ir];
                }
            }
            if (grid_.first() == 0 && l > 0 && f(0) != 0) {
                RTE_THROW("regular radial functions with l > 0 must vanish at the origin");
            }
            f.interpolate();
            for (int ir = 0; ir < nr; ir++) {
                radial_(ir, rf.idxrf)     = f(ir);
                derivative_(ir, rf.idxrf) = f.deriv(1, ir);
                if (grid_[ir] == 0 && l > 1) {
                    derivative_(ir, rf.idxrf) = 0;
                }
                // The angular gradient vanishes for s waves. For regular p waves R/r -> R'(0).
                angular_radial_(ir, rf.idxrf) = l == 0           ? 0
                                                : grid_[ir] != 0 ? f(ir) / grid_[ir]
                                                : l == 1         ? derivative_(ir, rf.idxrf)
                                                                 : 0;
            }
        }
    }

  private:
    void
    add_density_impl(mdarray<std::complex<double>, 2> const& dm__,
                     Spheric_function<function_domain_t::spectral, double>& field__,
                     Spheric_kinetic_density_operator const* response__, int component__, bool charge__) const
    {
        check_component(component__);
        check_field(field__);
        if (response__ && (&response__->indexb_ != &indexb_ || response__->grid_.hash() != grid_.hash() ||
                           response__->lmmax_ != lmmax_ || response__->harmonic_ != harmonic_)) {
            RTE_THROW("atom-local kinetic density: incompatible radial response");
        }
        if (dm__.size(0) != indexb_.size() || dm__.size(1) != indexb_.size()) {
            RTE_THROW("atom-local kinetic density: incompatible density matrix");
        }
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : angular_products(i, j, component__)) {
                    double w = (i == j ? 1 : 2) * std::real(dm__(i, j) * g.coef);
                    for (int ir = 0; ir < grid_.num_points(); ir++) {
                        field__(g.lm3, ir) += w * radial_product(i, j, g.l3, ir, response__, charge__);
                    }
                }
            }
        }
    }

  public:
    /// Accumulate scalar-spin or full-j spinor tau, with occupations already in the density matrix.
    /** An optional radial response accumulates the variation at fixed density matrix instead. */
    void
    add_density(mdarray<std::complex<double>, 2> const& dm__,
                Spheric_function<function_domain_t::spectral, double>& tau__,
                Spheric_kinetic_density_operator const* response__ = nullptr, int component__ = 0) const
    {
        add_density_impl(dm__, tau__, response__, component__, false);
    }

    /// Charge or Pauli-spin density with the same angular products and exact radial response as tau.
    void
    add_charge_density(mdarray<std::complex<double>, 2> const& dm__,
                       Spheric_function<function_domain_t::spectral, double>& rho__,
                       Spheric_kinetic_density_operator const* response__ = nullptr, int component__ = 0) const
    {
        add_density_impl(dm__, rho__, response__, component__, true);
    }

    /// Matrix of 1/2 integral v_tau grad(phi_i)^* . grad(phi_j), without integration by parts.
    /** Uses the same radial quadrature as inner() for spherical fields, including
     *  its r^2 weight before spline interpolation. No boundary condition is imposed
     *  on the partial waves at the sphere surface.
     */
    mdarray<T, 2>
    matrix_elements(Spheric_function<function_domain_t::spectral, double> const& vtau__, int component__ = 0) const
    {
        check_component(component__);
        check_field(vtau__);
        mdarray<T, 2> result({indexb_.size(), indexb_.size()});
        Spline<T> integrand(grid_);
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (int ir = 0; ir < grid_.num_points(); ir++) {
                    integrand(ir) = 0;
                    for (auto const& g : angular_products(i, j, component__)) {
                        integrand(ir) += vtau__(g.lm3, ir) * g.coef * radial_product(i, j, g.l3, ir);
                    }
                    integrand(ir) *= grid_[ir] * grid_[ir];
                }
                result(i, j) = integrand.interpolate().integrate(0);
                result(j, i) = sirius::conj(result(i, j));
            }
        }
        return result;
    }

  private:
    mdarray<T, 2>
    matrix_elements_from_derivatives_impl(Spheric_function<function_domain_t::spectral, double> const& field__,
                                          Spheric_kinetic_density_operator const* response__, int component__,
                                          bool charge__) const
    {
        check_component(component__);
        check_field(field__);
        if (response__ && (&response__->indexb_ != &indexb_ || response__->grid_.hash() != grid_.hash() ||
                           response__->lmmax_ != lmmax_ || response__->harmonic_ != harmonic_)) {
            RTE_THROW("atom-local kinetic density: incompatible radial response");
        }
        mdarray<T, 2> result({indexb_.size(), indexb_.size()});
        result.zero();
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : angular_products(i, j, component__)) {
                    for (int ir = 0; ir < grid_.num_points(); ir++) {
                        result(i, j) +=
                                field__(g.lm3, ir) * g.coef * radial_product(i, j, g.l3, ir, response__, charge__);
                    }
                }
                result(j, i) = sirius::conj(result(i, j));
            }
        }
        return result;
    }

  public:
    /// Matrix derivative from discrete spherical tau covectors, without quadrature.
    /** Use for derivatives returned by an atom-grid interpolation transpose.
     *  Unlike matrix_elements(), d_tau__ already contains the complete discrete
     *  energy weights. No radial integration, r^2 or occupation factor is added.
     *  An optional response operator supplies changes of the radial functions and
     *  their derivatives, returning the matrix variation at fixed covector.
     */
    mdarray<T, 2>
    matrix_elements_from_derivatives(Spheric_function<function_domain_t::spectral, double> const& d_tau__,
                                     Spheric_kinetic_density_operator const* response__ = nullptr,
                                     int component__                                    = 0) const
    {
        return matrix_elements_from_derivatives_impl(d_tau__, response__, component__, false);
    }

    /// Exact transpose of add_charge_density for already-weighted spherical sample covectors.
    mdarray<T, 2>
    charge_matrix_elements_from_derivatives(Spheric_function<function_domain_t::spectral, double> const& d_rho__,
                                            Spheric_kinetic_density_operator const* response__ = nullptr,
                                            int component__                                    = 0) const
    {
        return matrix_elements_from_derivatives_impl(d_rho__, response__, component__, true);
    }
};

} // namespace sirius

#endif
