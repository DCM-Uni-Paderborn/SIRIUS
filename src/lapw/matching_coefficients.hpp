/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file matching_coefficients.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Matching_coefficients class.
 */

#ifndef __MATCHING_COEFFICIENTS_HPP__
#define __MATCHING_COEFFICIENTS_HPP__

#include <gsl/gsl_sf_bessel.h>
#include "unit_cell/unit_cell.hpp"
#include "core/fft/gvec.hpp"
#include "core/la/linalg.hpp"
#include "core/sht/sht.hpp"

namespace sirius {

/// Orthonormal positive-energy free Dirac spinors, in Hartree with the rest energy subtracted.
class Dirac_plane_wave
{
  private:
    r3::vector<double> q_;
    double energy_, ratio_, normalization_;

  public:
    using spinors_t = std::array<std::array<std::complex<double>, 4>, 2>;

    explicit Dirac_plane_wave(r3::vector<double> const& q__)
        : q_(q__)
    {
        for (double q : q_) {
            if (!std::isfinite(q)) {
                RTE_THROW("Dirac plane wave: nonfinite momentum");
            }
        }
        double d       = std::hypot(1.0, q_.length() / speed_of_light);
        energy_        = q_.length2() / (d + 1);
        ratio_         = 1 / (speed_of_light * (d + 1));
        normalization_ = std::sqrt((d + 1) / (2 * d));
        if (!std::isfinite(energy_)) {
            RTE_THROW("Dirac plane wave: momentum exceeds the finite energy range");
        }
    }

    double
    energy() const
    {
        return energy_;
    }

    double
    energy_response(r3::vector<double> const& dq__) const
    {
        for (double dq : dq__) {
            if (!std::isfinite(dq)) {
                RTE_THROW("Dirac plane wave: nonfinite momentum direction");
            }
        }
        return dot(q_, dq__) / (1 + energy_ / (speed_of_light * speed_of_light));
    }

    spinors_t
    spinors() const
    {
        spinors_t result{};
        result[0] = {normalization_, 0, normalization_ * ratio_ * q_[2],
                     normalization_ * ratio_ * std::complex<double>(q_[0], q_[1])};
        result[1] = {0, normalization_, normalization_ * ratio_ * std::complex<double>(q_[0], -q_[1]),
                     -normalization_ * ratio_ * q_[2]};
        return result;
    }

    spinors_t
    response(r3::vector<double> const& dq__) const
    {
        double de     = energy_response(dq__);
        double dratio = -ratio_ * de / (energy_ + 2 * speed_of_light * speed_of_light);
        double dn = -std::pow(normalization_, 3) * (ratio_ * dratio * q_.length2() + ratio_ * ratio_ * dot(q_, dq__));
        double scale = dn * ratio_ + normalization_ * dratio;
        auto small   = q_ * scale + dq__ * (normalization_ * ratio_);
        spinors_t result{};
        result[0] = {dn, 0, small[2], std::complex<double>(small[0], small[1])};
        result[1] = {0, dn, std::complex<double>(small[0], -small[1]), -small[2]};
        return result;
    }
};

/** The following matching conditions must be fulfilled:
 *  \f[
 *   \frac{\partial^j}{\partial r^j} \sum_{L \nu} A_{L \nu}^{\bf k}({\bf G})u_{\ell \nu}(r)
 *   Y_{L}(\hat {\bf r}) \bigg|_{R^{MT}}  = \frac{\partial^j}{\partial r^j} \frac{4 \pi}{\sqrt \Omega}
 *   e^{i{\bf (G+k)\tau}} \sum_{L}i^{\ell} j_{\ell}(|{\bf G+k}|r) Y_{L}^{*}(\widehat {\bf G+k}) Y_{L}(\hat {\bf r})
 *   \bigg|_{R^{MT}}
 *  \f]
 *  where \f$ L = \{ \ell, m \} \f$. Dropping sum over L we arrive to the following system of linear equations:
 *  \f[ \sum_{\nu} \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}} A_{L \nu}^{\bf k}({\bf G})
 *   = \frac{4 \pi}{\sqrt \Omega} e^{i{\bf (G+k)\tau}} i^{\ell} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j}
 *      \bigg|_{R^{MT}} Y_{L}^{*}(\widehat {\bf G+k})
 *  \f]
 *  The matching coefficients are then equal to:
 *  \f[
 *   A_{L \nu}^{\bf k}({\bf G}) = \sum_{j} \bigg[ \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}}
 *   \bigg]_{\nu j}^{-1} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j} \bigg|_{R^{MT}}
 *   \frac{4 \pi}{\sqrt \Omega} i^{\ell} e^{i{\bf (G+k)\tau}} Y_{L}^{*}(\widehat {\bf G+k})
 *  \f]
 */
class Matching_coefficients // TODO: compute on GPU
{
  private:
    /// Description of the unit cell.
    Unit_cell const& unit_cell_;

    /// Description of the G+k vectors.
    fft::Gvec const& gkvec_;

    /// Length of G+k vectors.
    std::vector<double> gkvec_len_;

    /// Spherical harmonics Ylm(theta, phi) of the G+k vectors.
    mdarray<std::complex<double>, 2> gkvec_ylm_;

    /// Precomputed values for the linear equations for matching coefficients.
    mdarray<std::complex<double>, 4> alm_b_;

    /// Matching coefficients for all atoms.
    /** For convenience, conjugated coefficients are stored */
    mdarray<std::complex<double>, 2> alm_all_atoms_;

    /// Offset in the alm_all_atoms array.
    std::vector<int> mt_aw_offset_;

    /// True if matching coefficients for all atoms are computed.
    bool all_atoms_{false};

    struct plane_wave_response_t
    {
        mdarray<std::complex<double>, 3> bessel;
        mdarray<std::complex<double>, 2> harmonics;
        std::vector<double> phase;
    };

    plane_wave_response_t
    plane_wave_geometry_response(Atom const& atom__, r3::vector<double> const& displacement__,
                                 r3::matrix<double> const& strain__, int lmax__) const
    {
        int lmmax = sf::lmmax(lmax__), iat = atom__.type().id();
        plane_wave_response_t response{mdarray<std::complex<double>, 3>({3, gkvec_.count(), lmax__ + 1}),
                                       mdarray<std::complex<double>, 2>({gkvec_.count(), lmmax}),
                                       std::vector<double>(gkvec_.count())};
        response.bessel.zero();
        response.harmonics.zero();
        double trace  = strain__(0, 0) + strain__(1, 1) + strain__(2, 2);
        auto position = unit_cell_.get_cartesian_coordinates(atom__.position());
        double R      = atom__.type().mt_radius();
        std::vector<double> bessel(lmax__ + 3), drlm(lmmax);
        std::vector<std::complex<double>> dy(lmmax);
        mdarray<double, 2> grad_rlm({lmmax, 3});
        for (int igk = 0; igk < gkvec_.count(); igk++) {
            auto q              = gkvec_.gkvec_cart(gvec_index_t::local(igk));
            auto dq             = dot(transpose(strain__), q) * -1.0;
            response.phase[igk] = dot(q, displacement__) + dot(dq, position);
            double length       = gkvec_len_[igk];
            if (length > 1e-12) {
                double dlength = dot(q, dq) / length;
                gsl_sf_bessel_jl_array(lmax__ + 2, R * length, bessel.data());
                sf::dRlm_dr(lmax__, q, grad_rlm);
                for (int lm = 0; lm < lmmax; lm++) {
                    drlm[lm] = 0;
                    for (int x = 0; x < 3; x++) {
                        drlm[lm] += grad_rlm(lm, x) * dq[x];
                    }
                }
                // convert transforms expansion coefficients, so conjugate to obtain Ylm values.
                SHT::convert(lmax__, drlm.data(), dy.data());
                for (int lm = 0; lm < lmmax; lm++) {
                    response.harmonics(igk, lm) = std::conj(dy[lm]);
                }
                for (int l = 0; l <= lmax__; l++) {
                    double jp  = l / (R * length) * bessel[l] - bessel[l + 1];
                    double j1p = (l + 1) / (R * length) * bessel[l + 1] - bessel[l + 2];
                    auto scale = std::pow(std::complex<double>(0, 1), l) * (fourpi / std::sqrt(unit_cell_.omega())) *
                                 dlength;
                    response.bessel(0, igk, l) = scale * R * jp;
                    response.bessel(1, igk, l) = scale * (-bessel[l + 1] - length * R * j1p + l * jp);
                    response.bessel(2, igk, l) =
                            scale * (2 * bessel[l + 1] / R + 2 * length * j1p - 2 * length * bessel[l] +
                                     ((l - 1) * l / R - length * length * R) * jp);
                }
            }
            for (int l = 0; l <= lmax__; l++) {
                for (int j = 0; j < 3; j++) {
                    response.bessel(j, igk, l) -= 0.5 * trace * alm_b_(j, igk, l, iat);
                }
            }
        }
        return response;
    }

    /// Generate matching coefficients for a specific \f$ \ell \f$ and order.
    /** \param [in] ngk           Number of G+k vectors.
     *  \param [in] phase_factors Phase factors of G+k vectors.
     *  \param [in] iat           Index of atom type.
     *  \param [in] l             Orbital quantum nuber.
     *  \param [in] lm            Composite l,m index.
     *  \param [in] nu            Order of radial function \f$ u_{\ell \nu}(r) \f$ for which coefficients are generated.
     *  \param [in] A             Inverse matrix of radial derivatives.
     *  \param [out] alm          Pointer to alm coefficients.
     */
    template <int N, bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    inline void
    generate(int ngk, std::vector<std::complex<double>> const& phase_factors__, int iat, int l, int lm, int nu,
             r3::matrix<double> const& A, T* alm, mdarray<std::complex<double>, 3> const* db__ = nullptr,
             mdarray<std::complex<double>, 2> const* dylm__ = nullptr,
             std::vector<double> const* dphase__            = nullptr) const
    {
        std::complex<double> zt;

        for (int igk = 0; igk < ngk; igk++) {
            switch (N) {
                case 1: {
                    zt = alm_b_(0, igk, l, iat) * A(0, 0);
                    break;
                }
                case 2: {
                    zt = alm_b_(0, igk, l, iat) * A(nu, 0) + alm_b_(1, igk, l, iat) * A(nu, 1);
                    break;
                }
                case 3: {
                    zt = alm_b_(0, igk, l, iat) * A(nu, 0) + alm_b_(1, igk, l, iat) * A(nu, 1) +
                         alm_b_(2, igk, l, iat) * A(nu, 2);
                    break;
                }
            }
            if (db__) {
                std::complex<double> dzt{0};
                for (int j = 0; j < N; j++) {
                    dzt += (*db__)(j, igk, l) * A(nu, j);
                }
                auto derivative = phase_factors__[igk] * ((dzt + std::complex<double>(0, (*dphase__)[igk]) * zt) *
                                                                  std::conj(gkvec_ylm_(igk, lm)) +
                                                          zt * std::conj((*dylm__)(igk, lm)));
                alm[igk]        = conjugate ? std::conj(derivative) : derivative;
            } else if (conjugate) {
                alm[igk] = std::conj(phase_factors__[igk] * zt) * gkvec_ylm_(igk, lm);
            } else {
                alm[igk] = phase_factors__[igk] * zt * std::conj(gkvec_ylm_(igk, lm));
            }
        }
    }

  public:
    /// Constructor
    Matching_coefficients(Unit_cell const& unit_cell__, fft::Gvec const& gkvec__, bool all_atoms__ = false)
        : unit_cell_(unit_cell__)
        , gkvec_(gkvec__)
        , all_atoms_(all_atoms__)
    {
        int lmax_apw = unit_cell__.lmax_apw();
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto const& type = unit_cell_.atom_type(iat);
            if (type.indexb().full_j()) {
                if (all_atoms_) {
                    RTE_THROW("full-j matching requires component-resolved coefficients");
                }
                lmax_apw = std::max(lmax_apw, type.lmax_apw() + 1);
            }
        }
        int lmmax_apw = sf::lmmax(lmax_apw);

        gkvec_ylm_ = mdarray<std::complex<double>, 2>({gkvec_.count(), lmmax_apw}, mdarray_label("gkvec_ylm"));
        gkvec_len_.resize(gkvec_.count());

        /* get length and Ylm harmonics of G+k vectors */
        #pragma omp parallel
        {
            std::vector<std::complex<double>> ylm(lmmax_apw);
            std::vector<double> rlm(lmmax_apw);

            int ngk = gkvec_.count();
            #pragma omp for
            for (int igloc = 0; igloc < ngk; igloc++) {
                auto gkvec_cart = gkvec_.gkvec_cart(gvec_index_t::local(igloc));
                /* get r, theta, phi */
                auto vs = r3::spherical_coordinates(gkvec_cart);

                gkvec_len_[igloc] = vs[0];
                // Cartesian evaluation retains angular resolution close to the polar axis.
                sf::spherical_harmonics(lmax_apw, gkvec_cart, rlm.data());
                SHT::convert(lmax_apw, rlm.data(), ylm.data());

                for (int lm = 0; lm < lmmax_apw; lm++) {
                    gkvec_ylm_(igloc, lm) = std::conj(ylm[lm]);
                }
            }
        }

        alm_b_ = mdarray<std::complex<double>, 4>({3, gkvec_.count(), lmax_apw + 1, unit_cell_.num_atom_types()});
        alm_b_.zero();

        #pragma omp parallel
        {
            /* value and first two derivatives of spherical Bessel functions */
            mdarray<double, 2> sbessel_mt({lmax_apw + 2, 3});

            #pragma omp for
            for (int igk = 0; igk < gkvec_.count(); igk++) {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                    double R = unit_cell_.atom_type(iat).mt_radius();

                    double RGk = R * gkvec_len_[igk];

                    /* compute values and first and second derivatives of the spherical Bessel functions
                       at the MT boundary */
                    gsl_sf_bessel_jl_array(lmax_apw + 1, RGk, &sbessel_mt(0, 0));

                    /* Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,1}]]
                     * Out[]= (n SphericalBesselJ[n,a x])/x-a SphericalBesselJ[1+n,a x]
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,2}]]
                     * Out[]= (((-1+n) n-a^2 x^2) SphericalBesselJ[n,a x]+2 a x SphericalBesselJ[1+n,a x])/x^2
                     */
                    for (int l = 0; l <= lmax_apw; l++) { // TODO: move to sbessel class
                        sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igk] + (l / R) * sbessel_mt(l, 0);
                        sbessel_mt(l, 2) = 2 * gkvec_len_[igk] * sbessel_mt(l + 1, 0) / R +
                                           ((l - 1) * l - std::pow(RGk, 2)) * sbessel_mt(l, 0) / std::pow(R, 2);
                    }

                    for (int l = 0; l <= lmax_apw; l++) {
                        std::complex<double> z = std::pow(std::complex<double>(0, 1), l);
                        double f               = fourpi / std::sqrt(unit_cell_.omega());
                        alm_b_(0, igk, l, iat) = z * f * sbessel_mt(l, 0);
                        alm_b_(1, igk, l, iat) = z * f * sbessel_mt(l, 1);
                        alm_b_(2, igk, l, iat) = z * f * sbessel_mt(l, 2);
                    }
                }
            }
        }

        if (all_atoms_) {
            alm_all_atoms_ = mdarray<std::complex<double>, 2>({gkvec_.count(), unit_cell_.mt_aw_basis_size()},
                                                              mdarray_label("alm_all_atoms"));
        }
    }

    inline auto const&
    gkvec() const
    {
        return gkvec_;
    }

    inline auto
    all_atoms() const
    {
        return all_atoms_;
    }

    /// Return pointer to the beginning of Alm array for a given atom.
    inline auto
    begin(int ia) const
    {
        if (!all_atoms_) {
            RTE_THROW("Matching coefficients for all atoms were not allocated");
        }
        return alm_all_atoms_.at(memory_t::host, 0, mt_aw_offset_[ia]);
    }

    /// Return pointer to the end of Alm array for a given atom.
    inline auto
    end(int ia) const
    {
        if (!all_atoms_) {
            RTE_THROW("Matching coefficients for all atoms were not allocated");
        }
        return this->begin(ia) + this->gkvec().count() * unit_cell_.atom(ia).type().mt_aw_basis_size();
    }

    /// Generate plane-wave matching coefficients for the radial solutions of a given atom.
    /** \param [in]  atom      Atom, for which matching coefficients are generated.
        \param [out] alm       Array of matching coefficients with dimension indices \f$ ({\bf G+k}, \xi) \f$.
     */
    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate(Atom const& atom__, mdarray<T, 2>& alm__) const
    {
        generate_impl<conjugate>(atom__, alm__, nullptr);
    }

    /// Differentiate matching coefficients for a radial-basis variation at fixed geometry.
    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate_potential_response(Atom const& atom__, radial_basis_response_t const& response__,
                                mdarray<T, 2>& alm__) const
    {
        auto const& sd = response__.surface_derivatives;
        if (sd.size(0) != 3 ||
            sd.size(1) != atom__.type().mt_radial_basis_size() - atom__.type().num_lo_descriptors()) {
            RTE_THROW("wrong dimensions of AW boundary response");
        }
        generate_impl<false>(atom__, alm__, &sd);
        if constexpr (conjugate) {
            for (auto& z : alm__) {
                z = std::conj(z);
            }
        }
    }

    /// Cartesian atom displacement and strain dA A^-1 at fixed radial functions and MT radius.
    /// Atom displacements are independent of strain, not implicitly affine.
    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate_geometry_response(Atom const& atom__, r3::vector<double> const& displacement__,
                               r3::matrix<double> const& strain__, mdarray<T, 2>& alm__) const
    {
        generate_impl<conjugate>(atom__, alm__, nullptr, &displacement__, &strain__);
    }

    /// Full-j coefficients indexed by (G+k, AW, 4 physical components or 2 positive-energy PW spinors).
    /** Match both Dirac components at the MT boundary; a third AW also matches the large-component derivative.
        The lower spinor is -i S Omega_-kappa, consistent with the radial Dirac solver and angular convention. */
    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate(Atom const& atom__, mdarray<T, 3>& alm__) const
    {
        generate_dirac_impl<conjugate>(atom__, alm__, nullptr);
    }

    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate_potential_response(Atom const& atom__, radial_basis_response_t const& response__,
                                mdarray<T, 3>& alm__) const
    {
        auto const& sd = response__.surface_derivatives;
        if (sd.size(0) != 6 ||
            sd.size(1) != atom__.type().mt_radial_basis_size() - atom__.type().num_lo_descriptors()) {
            RTE_THROW("wrong dimensions of Dirac AW boundary response");
        }
        generate_dirac_impl<conjugate>(atom__, alm__, &sd);
    }

    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate_geometry_response(Atom const& atom__, r3::vector<double> const& displacement__,
                               r3::matrix<double> const& strain__, mdarray<T, 3>& alm__) const
    {
        generate_dirac_impl<conjugate>(atom__, alm__, nullptr, &displacement__, &strain__);
    }

  private:
    template <bool conjugate, typename T>
    void
    generate_dirac_impl(Atom const& atom__, mdarray<T, 3>& alm__, mdarray<double, 2> const* response__,
                        r3::vector<double> const* displacement__ = nullptr,
                        r3::matrix<double> const* strain__       = nullptr) const
    {
        auto const& type = atom__.type();
        if (!type.indexb().full_j() || alm__.size(0) != gkvec_.count() || alm__.size(1) != type.mt_aw_basis_size() ||
            (alm__.size(2) != 4 && alm__.size(2) != 2)) {
            RTE_THROW("full-j matching requires dimensions (G+k, AW, 4 components or 2 PW spinors)");
        }
        if (alm__.size(2) == 2) {
            if constexpr (conjugate) {
                generate_dirac_impl<false>(atom__, alm__, response__, displacement__, strain__);
                for (auto& value : alm__) {
                    value = std::conj(value);
                }
                return;
            }
            mdarray<T, 3> components({gkvec_.count(), type.mt_aw_basis_size(), 4}), baseline;
            generate_dirac_impl<false>(atom__, components, response__, displacement__, strain__);
            if (strain__) {
                baseline = mdarray<T, 3>({gkvec_.count(), type.mt_aw_basis_size(), 4});
                generate_dirac_impl<false>(atom__, baseline, nullptr);
            }
            alm__.zero();
            for (int igk = 0; igk < gkvec_.count(); igk++) {
                auto q = gkvec_.gkvec_cart(gvec_index_t::local(igk));
                Dirac_plane_wave pw(q);
                auto spinors = pw.spinors();
                auto tangent =
                        strain__ ? pw.response(dot(transpose(*strain__), q) * -1.0) : Dirac_plane_wave::spinors_t{};
                for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                    for (int s : {0, 1}) {
                        std::complex<double> value{0};
                        for (int c = 0; c < 4; c++) {
                            value += static_cast<std::complex<double>>(components(igk, xi, c)) * spinors[s][c];
                            if (strain__) {
                                value += static_cast<std::complex<double>>(baseline(igk, xi, c)) * tangent[s][c];
                            }
                        }
                        alm__(igk, xi, s) = value;
                    }
                }
            }
            return;
        }
        std::map<int, r3::matrix<double>> inverse_boundary;
        constexpr std::array<int, 3> component{0, 1, 0}, derivative{0, 0, 1};
        for (auto const& rf : type.indexr()) {
            if (int(rf.idxlo) >= 0 || rf.order != 0) {
                continue;
            }
            int n = static_cast<int>(atom__.symmetry_class().aw_descriptor(rf.am).size());
            if (n < 2 || n > 3) {
                RTE_THROW("Dirac matching requires two or three augmented waves per kappa");
            }
            mdarray<double, 2> boundary({n, n}), inverse({n, n});
            inverse.zero();
            for (int row = 0; row < n; row++) {
                inverse(row, row) = 1;
                for (int order = 0; order < n; order++) {
                    boundary(row, order) =
                            atom__.symmetry_class().aw_surface_deriv(rf.am, order, derivative[row], component[row]);
                }
            }
            if (la::wrap(la::lib_t::lapack).gesv(n, n, boundary.at(memory_t::host), n, inverse.at(memory_t::host), n) !=
                0) {
                RTE_THROW("singular Dirac matching boundary matrix");
            }
            r3::matrix<double> A;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A(i, j) = inverse(i, j);
                }
            }
            if (response__) {
                r3::matrix<double> dA;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        for (int row = 0; row < n; row++) {
                            for (int order = 0; order < n; order++) {
                                auto idxrf = type.indexr().index_of(rf.am, order);
                                dA(i, j) -= A(i, row) * (*response__)(derivative[row] + 3 * component[row], idxrf) *
                                            A(order, j);
                            }
                        }
                    }
                }
                A = dA;
            }
            inverse_boundary.emplace(rf.am.kappa(), A);
        }
        plane_wave_response_t geometry;
        if (strain__) {
            geometry = plane_wave_geometry_response(atom__, *displacement__, *strain__, type.lmax_apw() + 1);
        }
        alm__.zero();
        for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
            auto const& basis = type.indexb(xi);
            auto const& A     = inverse_boundary.at(basis.am.kappa());
            int n             = static_cast<int>(atom__.symmetry_class().aw_descriptor(basis.am).size());
            for (int c = 0; c < 4; c++) {
                auto [lm, cg] = basis.spinor_component(c % 2, c >= 2);
                if (lm < 0) {
                    continue;
                }
                int l = c < 2 ? basis.am.l() : basis.am.two_j() - basis.am.l();
                for (int igk = 0; igk < gkvec_.count(); igk++) {
                    auto phase = std::exp(std::complex<double>(
                            0, twopi * dot(gkvec_.gkvec(gvec_index_t::local(igk)), atom__.position())));
                    std::complex<double> value{0};
                    for (int row = 0; row < n; row++) {
                        if (component[row] != c / 2) {
                            continue;
                        }
                        auto b   = alm_b_(derivative[row], igk, l, type.id());
                        auto rhs = b * std::conj(gkvec_ylm_(igk, lm));
                        if (strain__) {
                            rhs = (geometry.bessel(derivative[row], igk, l) +
                                   std::complex<double>(0, geometry.phase[igk]) * b) *
                                          std::conj(gkvec_ylm_(igk, lm)) +
                                  b * std::conj(geometry.harmonics(igk, lm));
                        }
                        value += A(basis.order, row) * rhs;
                    }
                    value *= cg * phase * (c < 2 ? std::complex<double>(1, 0) : std::complex<double>(0, 1));
                    alm__(igk, xi, c) = conjugate ? std::conj(value) : value;
                }
            }
        }
    }

    template <bool conjugate, typename T>
    void
    generate_impl(Atom const& atom__, mdarray<T, 2>& alm__, mdarray<double, 2> const* response__,
                  r3::vector<double> const* displacement__ = nullptr,
                  r3::matrix<double> const* strain__       = nullptr) const
    {
        auto& type = atom__.type();

        if (type.indexb().full_j()) {
            RTE_THROW("full-j matching requires component-resolved coefficients");
        }

        RTE_ASSERT(type.max_aw_order() <= 3);

        int iat = type.id();

        std::vector<std::complex<double>> phase_factors(gkvec_.count());
        for (auto it : gkvec_) {
            double phase            = twopi * dot(gkvec_.gkvec(it.igloc), atom__.position());
            phase_factors[it.igloc] = std::exp(std::complex<double>(0, phase));
        }

        const double eps{0.1};
        std::vector<r3::matrix<double>> Al(atom__.type().lmax_apw() + 1);
        /* create inverse matrix of radial derivatives for all values of \ell */
        for (int l = 0; l <= atom__.type().lmax_apw(); l++) {
            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());
            r3::matrix<double> A;
            /* create matrix of radial derivatives */
            for (int order = 0; order < num_aw; order++) {
                for (int dm = 0; dm < num_aw; dm++) {
                    A(dm, order) = atom__.symmetry_class().aw_surface_deriv(l, order, dm);
                }
            }

            /* invert matrix of radial derivatives */
            switch (num_aw) {
                case 1: {
                    if (unit_cell_.parameters().cfg().control().verification() >= 1) {
                        if (std::abs(A(0, 0)) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            RTE_WARNING(s.str());
                        }
                    }

                    A(0, 0) = 1.0 / A(0, 0);
                    break;
                }
                case 2: {
                    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

                    if (unit_cell_.parameters().cfg().control().verification() >= 1) {
                        if (std::abs(det) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            RTE_WARNING(s.str());
                        }
                    }

                    std::swap(A(0, 0), A(1, 1));
                    A(0, 0) /= det;
                    A(1, 1) /= det;
                    A(0, 1) = -A(0, 1) / det;
                    A(1, 0) = -A(1, 0) / det;
                    break;
                }
                case 3: {
                    A = inverse(A);
                    break;
                }
            }
            if (response__) {
                // A is the inverse boundary matrix, so dA = -A dB A.
                r3::matrix<double> derivative;
                for (int i = 0; i < num_aw; i++) {
                    for (int j = 0; j < num_aw; j++) {
                        for (int dm = 0; dm < num_aw; dm++) {
                            for (int order = 0; order < num_aw; order++) {
                                auto idxrf = type.indexr().index_of(angular_momentum(l), order);
                                derivative(i, j) -= A(i, dm) * (*response__)(dm, idxrf) * A(order, j);
                            }
                        }
                    }
                }
                A = derivative;
            }
            Al[l] = A;
        }

        plane_wave_response_t geometry;
        if (strain__) {
            geometry = plane_wave_geometry_response(atom__, *displacement__, *strain__, type.lmax_apw());
        }

        for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
            int l  = type.indexb(xi).am.l();
            int lm = type.indexb(xi).lm;
            int nu = type.indexb(xi).order;

            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());

            switch (num_aw) {
                /* APW */
                case 1: {
                    generate<1, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi),
                                           strain__ ? &geometry.bessel : nullptr, &geometry.harmonics, &geometry.phase);
                    break;
                }
                /* LAPW */
                case 2: {
                    generate<2, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi),
                                           strain__ ? &geometry.bessel : nullptr, &geometry.harmonics, &geometry.phase);
                    break;
                }
                /* Super LAPW */
                case 3: {
                    generate<3, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi),
                                           strain__ ? &geometry.bessel : nullptr, &geometry.harmonics, &geometry.phase);
                    break;
                }
                default: {
                    RTE_THROW("wrong order of augmented wave");
                }
            }
        }
    }

  public:
    /// Generate matching coefficients for all atoms.
    inline void
    generate()
    {
        PROFILE("sirius::Matching_coefficients::generate");

        int num_mt_aw{0};
        mt_aw_offset_ = std::vector<int>(unit_cell_.num_atoms());
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            mt_aw_offset_[ia] = num_mt_aw;
            num_mt_aw += unit_cell_.atom(ia).mt_aw_basis_size();
        }

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
            /* wrap matching coefficients of a single atom */
            mdarray<std::complex<double>, 2> alm_atom({this->gkvec().count(), type.mt_aw_basis_size()},
                                                      alm_all_atoms_.at(memory_t::host, 0, mt_aw_offset_[ia]),
                                                      mdarray_label("alm_atom"));
            /* generate conjugated LAPW matching coefficients on the CPU */
            this->generate<true>(atom, alm_atom);
        }
    }
};

} // namespace sirius

#endif // __MATCHING_COEFFICIENTS_HPP__
