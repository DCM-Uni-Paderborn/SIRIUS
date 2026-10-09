/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hamiltonian.hpp
 *
 *  \brief Contains declaration and definition of sirius::Hamiltonian class.
 */

#ifndef __HAMILTONIAN_HPP__
#define __HAMILTONIAN_HPP__

#include <memory>
#include <complex>
#include "core/wf/wave_functions.hpp"
#include "core/typedefs.hpp"
#include "core/fft/fft.hpp"
#include "core/la/dmatrix.hpp"
#include "local_operator.hpp"
#include "non_local_operator.hpp"

namespace sirius {
/* forward declaration */
class Atom;
class Simulation_context;
class Potential;
class Unit_cell;
class K_point_set;
struct radial_basis_response_t;
struct lapw_xc_field_adjoint_t;
struct xc_geometry_direction_t;
template <typename T>
class Periodic_function;

/// Hartree and XC matrix response at fixed geometry and radial basis.
/// Scalar/collinear states use spin channels; noncollinear states use total/z/x/y components.
/// Dirac states use four canonical Fourier columns and one intrinsic full-j atom matrix.
struct lapw_hamiltonian_response_t
{
    Simulation_context const* context{nullptr};
    mdarray<std::complex<double>, 2> scalar_pw, kinetic_pw;
    std::vector<mdarray<std::complex<double>, 3>> local;
    /// IORA mass response in the interstitial and muffin-tin overlap; empty for other modes.
    mdarray<std::complex<double>, 2> overlap_pw;
    std::vector<mdarray<std::complex<double>, 3>> overlap_local;
};

/// Atom-local radial-basis derivatives and the unperturbed overlap, in AW/LO order.
/// Native Dirac stores only local-field dH; its bare H/O response uses the full-j basis directly.
struct lapw_atom_basis_response_t
{
    mdarray<std::complex<double>, 3> h;
    mdarray<double, 2> o, overlap;
};

/// Moving interstitial boundary at fixed scalar-relativistic potential samples.
struct lapw_interstitial_geometry_response_t
{
    Simulation_context const* context{nullptr};
    std::vector<double> direction;
    std::vector<std::complex<double>> kinetic_pw, overlap_pw;
};
template <typename T>
class Local_operator;
template <typename T>
class D_operator;
template <typename T>
class Q_operator;
template <typename T>
class U_operator;
template <typename T>
class K_point;
template <typename T>
class Gaunt_coefficients;

/// Representation of Kohn-Sham Hamiltonian.
/** In general, Hamiltonian consists of kinetic term, local part of potential and non-local
    part of potential:
    \f[
      H = -\frac{1}{2} \nabla^2 + V_{loc}({\bf r}) + \sum_{\alpha} \sum_{\xi \xi'} |\beta_{\xi}^{\alpha} \rangle
        D_{\xi \xi'}^{\alpha} \langle \beta_{\xi'}^{\alpha}|
    \f]
*/

/* forward declaration */
template <typename T>
class Hamiltonian_k;
template <typename T>
class Hamiltonian0;

/// Complete frozen-basis LAPW spectrum and its generalized density-matrix response.
/** Uses every PW/LO eigenvector, not only the stored bands. Occupations and their
 *  energy derivatives are supplied explicitly for this spectrum. All calls are
 *  collective on the k-point communicator and leave the ground state unchanged.
 *  Native Dirac matrices retain both positive-energy PW blocks and the intrinsic
 *  full-j LO block in one spectrum, using the physical four-component overlap.
 */
class LAPW_eigensystem
{
  private:
    friend class LAPW_spinor_eigensystem;
    enum class space_t
    {
        pw_lo,
        first_variation,
        second_variation
    };
    Simulation_context& ctx_;
    K_point<double>& kp_;
    int dimension_;
    mdarray<std::complex<double>, 2> vectors_, overlap_, physical_overlap_;
    std::vector<double> energies_, physical_norms_;

    LAPW_eigensystem(Hamiltonian_k<double> const& hk__, la::dmatrix<std::complex<double>> const& h__,
                     la::dmatrix<std::complex<double>> const& s__, space_t space__);

    void
    validate_matrix(la::dmatrix<std::complex<double>> const& matrix__) const;

    void
    validate_occupations(std::vector<double> const& occupations__) const;

  public:
    struct perturbation_t
    {
        LAPW_eigensystem const* reference{nullptr};
        mdarray<std::complex<double>, 2> h, s;
        std::vector<double> energies, physical_norms;
    };

    LAPW_eigensystem(Hamiltonian_k<double> const& hk__, la::dmatrix<std::complex<double>> const& h__,
                     la::dmatrix<std::complex<double>> const& s__,
                     la::dmatrix<std::complex<double>> const* physical_overlap__ = nullptr);
    LAPW_eigensystem(LAPW_eigensystem const&) = delete;
    LAPW_eigensystem&
    operator=(LAPW_eigensystem const&) = delete;

    auto const&
    energies() const
    {
        return energies_;
    }

    auto const&
    overlap() const
    {
        return overlap_;
    }

    /// Transform dH/dS and compute energy directions without changing the eigenvectors.
    perturbation_t
    perturbation(la::dmatrix<std::complex<double>> const& h__, la::dmatrix<std::complex<double>> const& s__,
                 la::dmatrix<std::complex<double>> const* physical_overlap__ = nullptr) const;

    /// Coefficient density matrix C f C^H, including optional physical (IORA) normalization.
    /// IORA uses the ordinary overlap, as in normalize_for_iora, not its energy-dependent eigenproblem metric.
    mdarray<std::complex<double>, 2>
    density_matrix(std::vector<double> const& occupations__) const;

    /// Includes the generalized-overlap response and the degenerate spectral limit.
    /** Slopes are df/de, not df/dmu. mu_response__ is shared by the relevant
     *  charge/spin population and must be determined across k-points by the caller.
     */
    mdarray<std::complex<double>, 2>
    density_matrix_response(perturbation_t const& direction__, std::vector<double> const& occupations__,
                            std::vector<double> const& slopes__, double mu_response__ = 0) const;
};

/// Density response of the truncated first-/second-variation noncollinear LAPW problem.
/** Includes motion of the retained first-variation subspace into the excluded
 *  states. Matrices use canonical total/z/x/y components and a common overlap.
 *  All calls are k-point-collective and leave the ground state unchanged.
 */
class LAPW_spinor_eigensystem
{
  public:
    using matrix_set_t = std::array<la::dmatrix<std::complex<double>> const*, 4>;
    struct perturbation_t
    {
        LAPW_spinor_eigensystem const* reference{nullptr};
        mdarray<std::complex<double>, 2> frame;
        LAPW_eigensystem::perturbation_t spinor;
    };

  private:
    Simulation_context& ctx_;
    K_point<double>& kp_;
    std::unique_ptr<LAPW_eigensystem> auxiliary_, spinor_;
    mdarray<std::complex<double>, 2> frame_;
    std::array<mdarray<std::complex<double>, 2>, 4> h_;

    std::array<mdarray<std::complex<double>, 2>, 4>
    matrices(matrix_set_t const& h__) const;

  public:
    LAPW_spinor_eigensystem(Hamiltonian_k<double> const& hk__, matrix_set_t const& h__,
                            la::dmatrix<std::complex<double>> const& s__);

    auto const&
    energies() const
    {
        return spinor_->energies();
    }

    auto const&
    overlap() const
    {
        return auxiliary_->overlap();
    }

    perturbation_t
    perturbation(matrix_set_t const& h__, la::dmatrix<std::complex<double>> const& s__) const;

    /// Coefficient density matrices in independent PW/LO order, with four Pauli components.
    mdarray<std::complex<double>, 3>
    density_matrix(std::vector<double> const& occupations__) const;

    mdarray<std::complex<double>, 3>
    density_matrix_response(perturbation_t const& direction__, std::vector<double> const& occupations__,
                            std::vector<double> const& slopes__, double mu_response__ = 0) const;
};

/// Frozen LAPW density response with charge/spin constraints shared across k-points.
/** Caches the complete spectra once. The stored bands must describe the same
 *  Hamiltonian and remain unchanged during response calls. Returned matrices use
 *  independent PW/LO order and can be mapped to rho/tau with
 *  Density::lapw_density_matrix_fields. Radial-basis and geometry derivatives of
 *  that field map are separate contributions. All calls are context-collective.
 */
class LAPW_density_response
{
  private:
    K_point_set& ks_;
    std::vector<std::array<std::unique_ptr<LAPW_eigensystem>, 2>> spectra_;
    std::vector<std::unique_ptr<LAPW_spinor_eigensystem>> spinor_spectra_;
    std::vector<mdarray<double, 2>> bands_, occupations_;
    std::array<double, 2> fermi_;
    double width_, moment_, charge_;
    int smearing_;

    void
    validate_reference() const;

  public:
    struct matrix_direction_t
    {
        /// Noncollinear H uses total/z/x/y components and only S[0], the common metric.
        /// Native Dirac uses only H[0]/S[0], each containing both PW components and full-j LOs.
        std::array<la::dmatrix<std::complex<double>> const*, 4> h{};
        std::array<la::dmatrix<std::complex<double>> const*, 2> s{};
        /// Ordinary-overlap direction for IORA physical orbital normalization; absent in other modes.
        std::array<la::dmatrix<std::complex<double>> const*, 2> physical_s{};
    };

    LAPW_density_response(Hamiltonian0<double> const& h0__, K_point_set& ks__);

    /// Frozen valence matrices in local-k-point order, with spin channels, total/z/x/y or one full-j block last.
    std::vector<mdarray<std::complex<double>, 3>>
    density_matrices() const;

    /// Full occupied/virtual response, including overlap and global occupation derivatives.
    /** Optionally return the collectively summed smearing and valence eigenvalue-sum derivatives. */
    std::vector<mdarray<std::complex<double>, 3>>
    density_matrix_response(std::vector<matrix_direction_t> const& directions__, double* entropy__ = nullptr,
                            double* eval_sum__ = nullptr) const;
};

/// Represent the k-point independent part of Hamiltonian.
/** \tparam T   Precision of the wave-functions (float or double).
 */
template <typename T> // type is real type precision
class Hamiltonian0
{
  private:
    /// Simulation context.
    Simulation_context& ctx_;

    /// Alias for the potential.
    Potential* potential_{nullptr};

    /// Alias for unit cell.
    Unit_cell& unit_cell_;

    /// Local part of the Hamiltonian operator.
    std::unique_ptr<Local_operator<T>> local_op_;

    /// D operator (non-local part of Hamiltonian).
    std::unique_ptr<D_operator<T>> d_op_;

    /// Q operator (non-local part of S-operator).
    std::unique_ptr<Q_operator<T>> q_op_;

    /// Muffin-tin part of potential and magentic field.
    /** This quantities are k-point independent and can be precomputed when H0 is created. */
    std::vector<mdarray<std::complex<T>, 3>> hmt_;

    /// Effective potential after domain restriction (no second mask on discrete XC covectors).
    mdarray<std::complex<T>, 2> veff_pw_;

    /// Kinetic-density potential for exact PP or LAPW diagonalization.
    mdarray<std::complex<T>, 2> vtau_pw_;

    /// Plane-wave coefficients of the inverse relativistic mass weighted by the unit step-function.
    mdarray<std::complex<T>, 1> rm_inv_pw_;

    /// Plane-wave coefficients of the squared inverse relativistic mass weighted by the unit step-function.
    mdarray<std::complex<T>, 1> rm2_inv_pw_;

    /// Prepare canonical Fourier covectors and the full-j atom correction before any local k-point loop.
    void
    prepare_dirac_fields();

    std::vector<mdarray<std::complex<T>, 3>>
    dirac_atom_matrices(std::vector<radial_basis_response_t> const* response__ = nullptr) const;

    /* copy constructor is forbidden */
    Hamiltonian0(Hamiltonian0<T> const& src) = delete;
    /* copy assignment operator is forbidden */
    Hamiltonian0<T>&
    operator=(Hamiltonian0<T> const& src) = delete;

  public:
    /// Constructor.
    Hamiltonian0(Potential& potential__, bool precompute_lapw__, bool update_lapw_rf__ = true);

    ~Hamiltonian0();

    /// Default move constructor.
    Hamiltonian0(Hamiltonian0<T>&& src) = default;

    /// Return a Hamiltonian for the given k-point.
    inline Hamiltonian_k<T>
    operator()(K_point<T>& kp__) const;

    Simulation_context&
    ctx() const
    {
        return ctx_;
    }

    Potential&
    potential() const
    {
        return *potential_;
    }

    Local_operator<T>&
    local_op() const
    {
        if (!local_op_) {
            RTE_THROW("Dirac LAPW uses its four-component matrix operator, not the scalar local operator");
        }
        return *local_op_;
    }

    inline Q_operator<T> const&
    Q() const
    {
        return *q_op_;
    }

    inline D_operator<T> const&
    D() const
    {
        return *d_op_;
    }

    auto const&
    hmt(int ia__) const
    {
        return hmt_[ia__];
    }

    /// Atom-local Hamiltonian in scalar/collinear or total/z/x/y component order.
    /// Includes the second-variation SOC operator in the spinor components.
    std::complex<T>
    hmt_component(int ia__, int i__, int j__, int s__, bool spin_orbit_only__ = false) const;

    auto const&
    veff_pw(int ig__, int ispn__ = 0) const
    {
        return this->veff_pw_(ig__, ispn__);
    }

    auto const&
    vtau_pw(int ig__, int ispn__) const
    {
        return vtau_pw_(ig__, ispn__);
    }

    auto const&
    rm_inv_pw(int ig__) const
    {
        return this->rm_inv_pw_(ig__);
    }

    auto const&
    rm2_inv_pw(int ig__) const
    {
        return this->rm2_inv_pw_(ig__);
    }

    /// Apply the muffin-tin part of the Hamiltonian to the apw basis functions of an atom.
    /** The following matrix is computed:
     *  \f[
     *    b_{L_2 \nu_2}^{\alpha}({\bf G'}) = \sum_{L_1 \nu_1} \sum_{L_3}
     *      a_{L_1\nu_1}^{\alpha}({\bf G'})
     *      \langle u_{\ell_1\nu_1}^{\alpha} | h_{L3}^{\alpha} |  u_{\ell_2\nu_2}^{\alpha}
     *      \rangle  \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle
     *  \f]
     */
    void
    apply_hmt_to_apw(device_t pu__, int ia__, int j__, int ngv__, mdarray<std::complex<T>, 2> const& alm__,
                     mdarray<std::complex<T>, 2>& halm__, int stream_id__ = -1) const;

    /// Add correction to LAPW overlap arising in the infinite-order relativistic approximation (IORA).
    void
    add_o1mt_to_apw(Atom const& atom__, int num_gkvec__,
                    mdarray<std::complex<T>, 2>& alm__) const; // TODO: documentation

    /// Apply muffin-tin part of magnetic filed to the wave-functions.
    void
    apply_bmt(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const;

    /// Apply SO correction to the first-variational LAPW wave-functions.
    /** Raising and lowering operators:
     *  \f[
     *      L_{\pm} Y_{\ell m}= (L_x \pm i L_y) Y_{\ell m}  = \sqrt{\ell(\ell+1) - m(m \pm 1)} Y_{\ell m \pm 1}
     *  \f]
     */
    void
    apply_so_correction(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& hpsi__) const;

    void
    generate_pw_coefs(Potential const& potential__);

    /// Physical Dirac H/O in two positive-energy PW blocks followed by full-j LOs.
    /** Field gathers and atom contractions are performed once by the constructor on all context ranks.
     *  This call is collective only on the supplied G+k communicator. It does not use scalar FV matrices. */
    std::array<mdarray<std::complex<double>, 2>, 2>
    dirac_matrices(fft::Gvec const& gkvec__) const;

    /// Native Dirac dH at fixed radial basis and geometry, with no bare kinetic or overlap derivative.
    /// Uses covectors prepared collectively by lapw_field_response(); collective only on G+k ranks.
    mdarray<std::complex<double>, 2>
    dirac_field_response(fft::Gvec const& gkvec__, lapw_hamiltonian_response_t const& response__) const;

    /// Native Dirac dH/dO from radial functions and matching coefficients at fixed physical fields and geometry.
    /// Atom covector derivatives are prepared on the context by lapw_radial_basis_response().
    std::array<mdarray<std::complex<double>, 2>, 2>
    dirac_radial_response(fft::Gvec const& gkvec__, std::vector<radial_basis_response_t> const& response__,
                          std::vector<lapw_atom_basis_response_t> const& atoms__) const;

    /// Moving native Dirac H/O at fixed radial functions and Fourier/atom field covectors.
    /// Physical-field, sampling and radial-relaxation derivatives are separate contributions.
    std::array<mdarray<std::complex<double>, 2>, 2>
    dirac_geometry_response(fft::Gvec const& gkvec__, xc_geometry_direction_t const& direction__) const;

    /// Build atom-local dH/dS at fixed geometry and physical fields, without updating radial integrals.
    /// Native Dirac returns local-field dH with empty o/overlap; dirac_radial_response() assembles complete H/O.
    /// The SOC-only mode omits the overlap and all non-SOC terms.
    /// Collective on the context communicator, before looping over locally owned k-points.
    std::vector<lapw_atom_basis_response_t>
    lapw_radial_basis_response(std::vector<radial_basis_response_t> const& response__, bool spin_orbit_only__ = false,
                               bool physical_overlap__ = false) const;

    /// Prepare the Hartree/XC field contribution to dH and the IORA dS without changing ground-state fields.
    /// Collective on the context communicator, before looping over locally owned k-points.
    lapw_hamiltonian_response_t
    lapw_field_response(Periodic_function<double> const& hartree__, lapw_xc_field_adjoint_t const& xc__) const;

    /// ZORA/IORA mass-mask derivatives, collective on the context before local k-point loops.
    lapw_interstitial_geometry_response_t
    lapw_interstitial_geometry_response(xc_geometry_direction_t const& direction__) const;

    /// Frozen magnetic Hamiltonian in total/z/x/y order, with a zero total component.
    /// The SOC-only mode isolates the local spin-orbit operator.
    lapw_hamiltonian_response_t
    lapw_spin_fields(bool spin_orbit_only__ = false) const;
};

template <typename T>
class Hamiltonian_k
{
  private:
    /// K-point independent part of Hamiltonian.
    Hamiltonian0<T> const& H0_;

    /// K-point for which Hamiltonian is applied.
    K_point<T>& kp_;

    /// Hubbard correction.
    /** In general case it is a k-dependent matrix */
    std::shared_ptr<U_operator<T>> u_op_;

    /// Frozen matching transformation from independent PW/LO to atom AW/LO coefficients.
    void
    lapw_basis_coefficients(int ia__, mdarray<std::complex<T>, 2>& row__, mdarray<std::complex<T>, 2>& col__) const;

    /// Copy constructor is forbidden.
    Hamiltonian_k(Hamiltonian_k<T> const& src__) = delete;

    /// Assignment operator is forbidden.
    Hamiltonian_k<T>&
    operator=(Hamiltonian_k<T> const& src__) = delete;

  public:
    Hamiltonian_k(Hamiltonian0<T> const& H0__, K_point<T>& kp__);

    Hamiltonian_k(Hamiltonian_k<T>&& src__);

    ~Hamiltonian_k();

    Hamiltonian0<T> const&
    H0() const
    {
        return H0_;
    }

    template <typename F, int what>
    std::pair<mdarray<T, 2>, mdarray<T, 2>>
    get_h_o_diag_pw() const;

    template <int what>
    std::pair<mdarray<T, 2>, mdarray<T, 2>>
    get_h_o_diag_lapw() const;

    auto
    U() const -> U_operator<T> const&
    {
        return *u_op_;
    }

    K_point<T>&
    kp() const
    {
        return kp_;
    }

    /// Apply first-variational LAPW Hamiltonian and overlap matrices.
    /** Check the documentation of Hamiltonain::set_fv_h_o() for the expressions of Hamiltonian and overlap
     *  matrices and \ref basis for the definition of the LAPW+lo basis.
     *
     *  For the set of wave-functions expanded in LAPW+lo basis (k-point index is dropped for simplicity)
     *  \f[
     *      \psi_{i} = \sum_{\mu} \phi_{\mu} C_{\mu i}
     *  \f]
     *  where \f$ \mu = \{ {\bf G}, j \} \f$ is a combined index of LAPW and local orbitals we want to contrusct
     *  a subspace Hamiltonian and overlap matrices:
     *  \f[
     *      H_{i' i} = \langle \psi_{i'} | \hat H | \psi_i \rangle =
     *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i} =
     *          \sum_{\mu'} C_{\mu' i'}^{*} h_{\mu' i}(\psi)
     *  \f]
     *  \f[
     *      O_{i' i} = \langle \psi_{i'} | \psi_i \rangle =
     *          \sum_{\mu' \mu} C_{\mu' i'}^{*} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i} =
     *          \sum_{\mu'} C_{\mu' i'}^{*} o_{\mu' i}(\psi)
     *  \f]
     *  where
     *  \f[
     *      h_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \hat H | \phi_{\mu} \rangle C_{\mu i}
     *  \f]
     *  and
     *  \f[
     *      o_{\mu' i}(\psi) = \sum_{\mu} \langle \phi_{\mu'} | \phi_{\mu} \rangle C_{\mu i}
     *  \f]
     *  For the APW block of \f$  h_{\mu' i}(\psi)  \f$ and \f$  o_{\mu' i}(\psi)  \f$ we have:
     *  \f[
     *       h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i}
     *  \f]
     *  \f[
     *       o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i}
     *  \f]
     *  and for the lo block:
     *  \f[
     *       h_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i}
     *  \f]
     *  \f[
     *       o_{j' i}(\psi) = \sum_{{\bf G}} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} +
     *          \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i}
     *  \f]
     *
     *  APW-APW contribution, muffin-tin part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) b_{\xi'}^{\alpha}({\bf G})
     *           C_{{\bf G} i}
     *  \f]
     *  \f[
     *      o_{{\bf G'} i}(\psi) = \sum_{{\bf G}} \langle \phi_{\bf G'} | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *          \sum_{{\bf G}} \sum_{\alpha} \sum_{\xi'} a_{\xi'}^{\alpha *}({\bf G'}) a_{\xi'}^{\alpha}({\bf G})
     *           C_{{\bf G} i}
     *  \f]
     *  APW-APW contribution, interstitial effective potential part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} V({\bf r}) \psi_{i}({\bf r}) d{\bf r}
     *  \f]
     *  This is done by transforming \f$ \psi_i({\bf G}) \f$ to the real space, multiplying by effectvive potential
     *  and step function and transforming the result back to the \f$ {\bf G} \f$ domain.
     *
     *  APW-APW contribution, interstitial kinetic energy part:
     *  \f[
     *      h_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \Big( -\frac{1}{2} \nabla \Big)
     *          \Big( \nabla \psi_{i}({\bf r}) \Big) d{\bf r}
     *  \f]
     *  and the gradient of the wave-function is computed with FFT as:
     *  \f[
     *      \Big( \nabla \psi_{i}({\bf r}) \Big) = \sum_{\bf G} i{\bf G}e^{i{\bf G}{\bf r}}\psi_i({\bf G})
     *  \f]
     *
     *  APW-APW contribution, interstitial overlap:
     *  \f[
     *      o_{{\bf G'} i}(\psi) = \int \Theta({\bf r}) e^{-i{\bf G'}{\bf r}} \psi_{i}({\bf r}) d{\bf r}
     *  \f]
     *
     *  APW-lo contribution:
     *  \f[
     *      h_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \hat H | \phi_{j} \rangle C_{j i} =
     *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'})
     *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j}
     * Y_{\ell_j m_j} \rangle =
     *      \sum_{j} C_{j i} \sum_{\xi'} a_{\xi'}^{\alpha_j *}({\bf G'}) h_{\xi' \xi_j}^{\alpha_j}
     *  \f]
     *  \f[
     *      o_{{\bf G'} i}(\psi) =  \sum_{j} \langle \phi_{\bf G'} | \phi_{j} \rangle C_{j i} =
     *      \sum_{j} C_{j i}   \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'})
     *          \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}| \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j} \rangle
     * =
     *      \sum_{j} C_{j i} \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'}) o_{\nu' \zeta_j \ell_j}^{\alpha_j}
     *  \f]
     *  lo-APW contribution:
     *  \f[
     *     h_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} | \hat H | \phi_{\bf G} \rangle C_{{\bf G} i} =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'}
     * m_{j'}}
     *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{\xi} h_{\xi_{j'} \xi}^{\alpha_{j'}} a_{\xi}^{\alpha_{j'}}({\bf G})
     *  \f]
     *  \f[
     *     o_{j' i}(\psi) = \sum_{\bf G} \langle \phi_{j'} |  \phi_{\bf G} \rangle C_{{\bf G} i} =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'}
     * m_{j'}}
     *          | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G}) =
     *      \sum_{\bf G} C_{{\bf G} i} \sum_{\nu} o_{\zeta_{j'} \nu \ell_{j'}}^{\alpha_{j'}} a_{\ell_{j'} m_{j'}
     * \nu}^{\alpha_{j'}}({\bf G})
     *  \f]
     *  lo-lo contribution:
     *  \f[
     *      h_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \hat H | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i}
     * h_{\xi_{j'} \xi_j}^{\alpha_j}
     *          \delta_{\alpha_j \alpha_{j'}}
     *  \f]
     *  \f[
     *      o_{j' i}(\psi) = \sum_{j} \langle \phi_{j'} | \phi_{j} \rangle C_{j i} = \sum_{j} C_{j i}
     *          o_{\zeta_{j'} \zeta_{j} \ell_j}^{\alpha_j}
     *            \delta_{\alpha_j \alpha_{j'}} \delta_{\ell_j \ell_{j'}} \delta_{m_j m_{j'}}
     *  \f]
     *
     *  \param [in]  apw_only   True if only APW-APW block of H and O are applied.
     *  \param [in]  phi_is_lo  True if input wave-functions are pure local orbitals.
     *  \param [in]  b          Range of bands.
     *  \param [in]  phi        Input wave-functions.
     *  \param [out] hphi       Result of Hamiltonian, applied to wave-functions.
     *  \param [out] ophi       Result of overlap operator, applied to wave-functions.
     */
    void
    apply_fv_h_o(bool apw_only__, bool phi_is_lo__, wf::band_range b__, wf::Wave_functions<T>& phi__,
                 wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* ophi__) const;

    /// Setup the Hamiltonian and overlap matrices in APW+lo basis
    /** Starting from the LAPW+lo basis functions definition in \ref lapw_basis_definition, the Hamiltonian
     *  matrix has the following expression:
     *  \f[
     *      H_{\mu' \mu}=\langle \varphi_{\mu' } | \hat H | \varphi_{\mu } \rangle  =
     *      \left( \begin{array}{cc}
     *         H_{\bf G'G} & H_{{\bf G'}j} \\
     *         H_{j'{\bf G}} & H_{j'j}
     *      \end{array} \right)
     *  \f]
     *  APW-APW block:
     *  \f{eqnarray*}{
     *      H_{{\bf G'} {\bf G}}^{\bf k} &=& \sum_{\alpha} \sum_{L'\nu', L\nu} a_{L'\nu'}^{\alpha *}({\bf G'+k})
     *      \langle  u_{\ell' \nu'}^{\alpha}Y_{\ell' m'}|\hat h^{\alpha} | u_{\ell \nu}^{\alpha}Y_{\ell m}  \rangle
     *       a_{L\nu}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde V_{eff}({\bf
     * G - G'}) \\
     *          &=& \sum_{\alpha} \sum_{\xi' } a_{\xi'}^{\alpha *}({\bf G'+k})
     *              b_{\xi'}^{\alpha}({\bf G+k}) + \frac{1}{2}{\bf G'} {\bf G} \cdot \Theta({\bf G - G'}) + \tilde
     * V_{eff}({\bf G - G'})
     *  \f}
     *  APW-lo block:
     *  \f[
     *      H_{{\bf G'} j}^{\bf k} = \sum_{L'\nu'} a_{L'\nu'}^{\alpha_j *}({\bf G'+k})
     *      \langle  u_{\ell' \nu'}^{\alpha_j}Y_{\ell' m'}|\hat h^{\alpha_j} |  \phi_{\ell_j}^{\zeta_j \alpha_j}
     * Y_{\ell_j m_j}  \rangle
     *  \f]
     *  lo-APW block:
     *  \f[
     *      H_{j' {\bf G}}^{\bf k} = \sum_{L\nu} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}}
     *          |\hat h^{\alpha_{j'}} | u_{\ell \nu}^{\alpha_{j'}}Y_{\ell m}  \rangle a_{L\nu}^{\alpha_{j'}}({\bf G+k})
     *  \f]
     *  lo-lo block:
     *  \f[
     *      H_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} Y_{\ell_{j'} m_{j'}}
     *          |\hat h^{\alpha_{j}} |  \phi_{\ell_j}^{\zeta_j \alpha_j} Y_{\ell_j m_j}  \rangle  \delta_{\alpha_j
     * \alpha_{j'}}
     *  \f]
     *
     *  The overlap matrix has the following expression:
     *  \f[
     *      O_{\mu' \mu} = \langle \varphi_{\mu'} | \varphi_{\mu} \rangle
     *  \f]
     *  APW-APW block:
     *  \f[
     *      O_{{\bf G'} {\bf G}}^{\bf k} = \sum_{\alpha} \sum_{L\nu} a_{L\nu}^{\alpha *}({\bf G'+k})
     *      a_{L\nu}^{\alpha}({\bf G+k}) + \Theta({\bf G-G'})
     *  \f]
     *
     *  APW-lo block:
     *  \f[
     *      O_{{\bf G'} j}^{\bf k} = \sum_{\nu'} a_{\ell_j m_j \nu'}^{\alpha_j *}({\bf G'+k})
     *      \langle u_{\ell_j \nu'}^{\alpha_j} | \phi_{\ell_j}^{\zeta_j \alpha_j} \rangle
     *  \f]
     *
     *  lo-APW block:
     *  \f[
     *      O_{j' {\bf G}}^{\bf k} =
     *      \sum_{\nu'} \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} | u_{\ell_{j'} \nu'}^{\alpha_{j'}} \rangle
     *      a_{\ell_{j'} m_{j'} \nu'}^{\alpha_{j'}}({\bf G+k})
     *  \f]
     *
     *  lo-lo block:
     *  \f[
     *      O_{j' j}^{\bf k} = \langle \phi_{\ell_{j'}}^{\zeta_{j'} \alpha_{j'}} |
     *      \phi_{\ell_{j}}^{\zeta_{j} \alpha_{j}} \rangle \delta_{\alpha_{j'} \alpha_j}
     *      \delta_{\ell_{j'} \ell_j} \delta_{m_{j'} m_j}
     *  \f]
     *
     *  For IORA there is additional term that arises after linearisation of 1/M over energy:
     *  \f[
     *    \frac{1}{M} = \frac{1}{M_0} - \frac{\alpha^2}{2} \frac{1}{M_0^2} E
     *  \f]
     *
     *  If we now insert it into kinetic energy \f$ \hat {\bf P} \frac{1}{2M} \hat {\bf P} \f$ we obtain
     *  correction linear to energy which has to be added to the overlap matrix:
     *  \f[
     *   \hat {\bf P} \frac{\alpha^2}{2} \frac{1}{2M_0^2} \hat {\bf P}
     *  \f]
     */
    void
    set_fv_h_o(int ispn__, la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__,
               bool physical_overlap__ = false) const;

    /// Native full-j Dirac H/O, dimension 2*num_gkvec + mt_lo_basis_size, collective on the k-point.
    void
    set_dirac_h_o(la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const;

    /// Fixed-basis Dirac Hartree/XC dH in the native positive-energy PW/full-j LO order.
    void
    set_dirac_h_field_response(lapw_hamiltonian_response_t const& response__, la::dmatrix<std::complex<T>>& h__) const;

    void
    set_dirac_h_o_radial_response(std::vector<radial_basis_response_t> const& response__,
                                  std::vector<lapw_atom_basis_response_t> const& atoms__,
                                  la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__) const;

    /// Native full-j moving-basis response at fixed radial functions and field covectors.
    void
    set_dirac_h_o_geometry_response(xc_geometry_direction_t const& direction__, la::dmatrix<std::complex<T>>& h__,
                                    la::dmatrix<std::complex<T>>& o__) const;

    /// Radial-basis response of the complete PW/LO matrices, including AW matching.
    /** Atom-local derivatives are prepared once by H0().lapw_radial_basis_response().
     *  The interstitial domain, physical potentials and independent PW/LO coefficients stay fixed.
     *  Collective on the k-point communicator.
     */
    void
    set_fv_h_o_radial_response(int ispn__, std::vector<radial_basis_response_t> const& response__,
                               std::vector<lapw_atom_basis_response_t> const& atoms__,
                               la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__,
                               bool spin_orbit_only__ = false) const;

    /// Basis, interstitial boundary and kinetic-metric contribution to the PW/LO geometry response.
    /** Radial functions, MT radii, local matrices and potential Fourier covectors stay fixed.
     *  Physical-field derivatives are separate. ZORA/IORA require the mass-mask response
     *  prepared collectively by Hamiltonian0. Collective on the k-point communicator.
     */
    void
    set_fv_h_o_geometry_response(int ispn__, xc_geometry_direction_t const& direction__,
                                 la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__,
                                 bool spin_orbit_only__                              = false,
                                 lapw_interstitial_geometry_response_t const* mass__ = nullptr,
                                 bool physical_overlap__                             = false) const;

    /// Hartree/XC field contribution to the complete PW/LO dH, at fixed basis and geometry.
    /// For spinors, ispn__ selects a total/z/x/y Pauli component, not an up/down spin block.
    /// Collective on the k-point communicator. An overlap output is required for an IORA mass response.
    void
    set_fv_h_field_response(int ispn__, lapw_hamiltonian_response_t const& response__,
                            la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>* o__ = nullptr) const;

    /// Add interstitial contribution to apw-apw block of Hamiltonian and overlap.
    void
    set_fv_h_o_it(int ispn__, la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__,
                  bool physical_overlap__ = false) const;

    /// Setup lo-lo block of Hamiltonian and overlap matrices.
    void
    set_fv_h_o_lo_lo(int ispn__, la::dmatrix<std::complex<T>>& h__, la::dmatrix<std::complex<T>>& o__,
                     bool physical_overlap__ = false) const;

    /// Setup apw-lo and lo-apw blocks of LAPW Hamiltonian and overlap matrices.
    void
    set_fv_h_o_apw_lo(Atom const& atom, int ia, int isnp__, mdarray<std::complex<T>, 2>& alm_row,
                      mdarray<std::complex<T>, 2>& alm_col, mdarray<std::complex<T>, 2>& h,
                      mdarray<std::complex<T>, 2>& o, bool physical_overlap__ = false) const;

    /// Apply pseudopotential H and S operators to the wavefunctions.
    /** \tparam F    Type of the subspace matrix.
     *  \param [in]  spins Range of spins.
     *  \param [in]  br    Range of bands.
     *  \param [in]  phi   Input wave-functions [storage: CPU && GPU].
     *  \param [out] hphi  Result of Hamiltonian, applied to wave-functions [storage: CPU || GPU].
     *  \param [out] sphi  Result of S-operator, applied to wave-functions [storage: CPU || GPU].
     *
     *  In non-collinear case (spins in [0,1]) the Hamiltonian and S operator are applied to both components of spinor
     *  wave-functions. Otherwise they are applied to a single component.
     */
    template <typename F>
    std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
    apply_h_s(wf::spin_range spins__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
              wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* sphi__) const
    {
        PROFILE("sirius::Hamiltonian_k::apply_h_s");

        auto pcs = env::print_checksum();

        if (hphi__ != nullptr) {
            /* apply local part of Hamiltonian */
            H0().local_op().apply_h(reinterpret_cast<fft::spfft_transform_type<T>&>(kp_.spfft_transform()),
                                    kp_.gkvec_fft_sptr(), spins__, phi__, *hphi__, br__);
        }

        auto mem = H0().ctx().processing_unit_memory_t();

        if (pcs) {
            auto cs = phi__.checksum(mem, br__);
            print_checksum("phi", cs, RTE_OUT(H0().ctx().out()));
            if (hphi__) {
                auto cs = hphi__->checksum(mem, br__);
                print_checksum("hloc_phi", cs, RTE_OUT(H0().ctx().out()));
            }
        }

        /* set initial sphi */
        if (sphi__ != nullptr) {
            for (auto s = spins__.begin(); s != spins__.end(); s++) {
                auto sp = phi__.actual_spin_index(s);
                wf::copy(mem, phi__, sp, br__, *sphi__, sp, br__);
            }
        }

        /* return if there are no beta-projectors */
        if (H0().ctx().unit_cell().max_mt_basis_size()) {
            auto bp_generator = kp_.beta_projectors().make_generator();
            auto beta_coeffs  = bp_generator.prepare();
            apply_non_local_D_Q<T, F>(mem, spins__, br__, bp_generator, beta_coeffs, phi__, &H0().D(), hphi__,
                                      &H0().Q(), sphi__);
        }

        /* apply the hubbard potential if relevant */
        if (H0().ctx().hubbard_correction() && !H0().ctx().gamma_point() && hphi__) {
            /* apply the hubbard potential */
            apply_U_operator(H0().ctx(), spins__, br__, kp_.hubbard_wave_functions_S(), phi__, this->U(), *hphi__);
        }

        if (pcs) {
            if (hphi__) {
                auto cs = hphi__->checksum(mem, br__);
                print_checksum("hphi", cs, RTE_OUT(H0().ctx().out()));
            }
            if (sphi__) {
                auto cs = sphi__->checksum(mem, br__);
                print_checksum("hsphi", cs, RTE_OUT(H0().ctx().out()));
            }
        }
    }

    template <typename F>
    std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
    apply_h_s(wf::spin_range spins__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
              wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* sphi__) const
    {
        RTE_THROW("implement this");
    }

    /// apply S operator
    /** \tparam F    Type of the subspace matrix.
     *  \param [in]  spins Range of spins.
     *  \param [in]  br    Range of bands.
     *  \param [in]  phi   Input wave-functions [storage: CPU && GPU].
     *  \param [out] sphi  Result of S-operator, applied to wave-functions [storage: CPU || GPU].
     *
     *  In non-collinear case (spins in [0,1]) the S operator is applied to both components of spinor
     *  wave-functions. Otherwise they are applied to a single component.
     */
    template <typename F>
    std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
    apply_s(wf::spin_range spin__, wf::band_range br__, wf::Wave_functions<T> const& phi__,
            wf::Wave_functions<T>& sphi__) const
    {
        auto mem       = H0().ctx().processing_unit_memory_t();
        auto bp_gen    = kp_.beta_projectors().make_generator();
        auto bp_coeffs = bp_gen.prepare();
        apply_S_operator<T, F>(mem, spin__, br__, bp_gen, bp_coeffs, phi__, &H0().Q(), sphi__);
    }

    /// Apply magnetic field to first-variational LAPW wave-functions.
    void
    apply_b(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const;
};

template <typename T>
Hamiltonian_k<T>
Hamiltonian0<T>::operator()(K_point<T>& kp__) const
{
    return Hamiltonian_k<T>(*this, kp__);
}

} // namespace sirius

#endif
