/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_RADIAL_BASIS_HPP__
#define __LAPW_RADIAL_BASIS_HPP__

#include "unit_cell/unit_cell.hpp"
#include "core/la/eigensolver.hpp"
#include "core/mpi/pstdout.hpp"
#include "radial/radial_core_solver.hpp"

namespace sirius {

struct radial_solver_result_t;

/// Potential derivatives of the radial basis, AW boundary data and spherical core fields.
struct radial_basis_response_t
{
    mdarray<double, 3> radial_functions;
    mdarray<double, 2> surface_derivatives;
    /// Spin-summed {rho, tau} response on the MT grid, without radial weights.
    std::array<std::vector<double>, 2> core_fields;
    /// Occupation-weighted derivative of the core eigenvalue sum for one atom.
    double core_eval_sum{0};
    /// Total derivative of the SOC radial integrals, including basis and potential motion.
    mdarray<double, 3> spin_orbit_integrals;

    explicit radial_basis_response_t(Atom_type const& type__)
        : radial_functions({type__.num_mt_points(), type__.mt_radial_basis_size(), type__.indexb().full_j() ? 4 : 2})
        , surface_derivatives(
                  {type__.indexb().full_j() ? 6 : 3, type__.mt_radial_basis_size() - type__.num_lo_descriptors()})
    {
        radial_functions.zero();
        surface_derivatives.zero();
        for (auto& field : core_fields) {
            field.resize(type__.num_mt_points(), 0);
        }
        if (type__.parameters().so_correction() && !type__.indexb().full_j()) {
            int order            = type__.indexr().max_order();
            spin_orbit_integrals = mdarray<double, 3>({type__.indexr().lmax() + 1, order, order});
            spin_orbit_integrals.zero();
        }
    }
};

/// Data and methods specific to the symmetry class of the atom.
/** Atoms transforming into each other under symmetry opeartions belong to the same symmetry class. They have the
 *  same spherical part of the on-site potential and, as a consequence, the same radial functions.
 */
class lapw_radial_basis_t
{
  public:
    /// Radial channels keyed by l (scalar) or kappa (Dirac).
    using core_states_t = std::map<int, std::vector<radial_core_state_t>>;

  private:
    /// Symmetry class id in the range [0, N_class).
    int id_;

    /// Reference to atom type.
    Atom_type const& atom_type_;

    /// Spherical part of the effective potential.
    /** Used by the LAPW radial solver. Actual value is stored, not the Y00 component. */
    std::vector<double> spherical_potential_;

    /// Raw spherical density/tau derivatives, also used by the radial core solver.
    std::array<std::vector<double>, 2> spherical_xc_derivatives_;

    using enu_response_t = std::map<std::pair<int, int>, double>;

    mutable std::map<int, radial_core_reference_t> core_reference_;
    mutable std::map<int, std::shared_ptr<Radial_dirac_xc_solver const>> dirac_core_reference_;

    /// List of radial functions for the LAPW basis.
    /** This array stores all the radial functions (AW and LO) and their derivatives. Radial derivatives of functions
     *  are multiplied by \f$ x \f$.\n
     *  1-st dimension: index of radial point \n
     *  2-nd dimension: index of radial function \n
     *  3-rd dimension: {R, rR'} and, for a Dirac basis, {S, rS'} as well. */
    mdarray<double, 3> radial_functions_;

    /// Surface derivatives of AW radial functions.
    mdarray<double, 2> surface_derivatives_;

    /// Spherical part of radial integral.
    mdarray<double, 2> h_spherical_integrals_;

    /// Overlap integrals.
    mdarray<double, 3> o_radial_integrals_;

    /// Overlap integrals for IORA relativistic treatment.
    mdarray<double, 2> o1_radial_integrals_;

    /// Spin-orbit interaction integrals.
    mdarray<double, 3> so_radial_integrals_;

    std::map<std::pair<int, int>, enu_search_t> enu_search_;

    /// List of radial descriptor sets used to construct augmented waves.
    mutable std::vector<radial_solution_descriptor_set> aw_descriptors_;
    mutable std::map<int, radial_solution_descriptor_set> dirac_aw_descriptors_;

    /// List of radial descriptor sets used to construct local orbitals.
    mutable std::vector<local_orbital_descriptor> lo_descriptors_;

    void
    update_enu_descriptors();

    std::vector<angular_momentum>
    aw_channels() const;

    void
    validate_radial_relativity(relativity_t rel__) const;

    core_states_t
    generate_core_radial_states(std::vector<double> const* direction__ = nullptr, core_states_t* response__ = nullptr,
                                std::array<std::vector<double>, 2> const& xc_directions__ = {},
                                std::array<std::vector<double>, 2>* fields_response__     = nullptr,
                                double* energy_response__                                 = nullptr) const;

    radial_solver_result_t
    solve_radial(relativity_t rel__, int dme__, angular_momentum am__, double energy__, core_states_t const& core__,
                 std::vector<double> const* direction__ = nullptr, radial_solver_result_t* response__ = nullptr,
                 core_states_t const* core_response__ = nullptr, double energy_direction__ = 0) const;

    /// Generate radial functions for augmented waves
    int
    generate_aw_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__, mdarray<double, 2>& sd__,
                                 core_states_t const& core__, std::vector<double> const* direction__ = nullptr,
                                 radial_basis_response_t* response__  = nullptr,
                                 core_states_t const* core_response__ = nullptr,
                                 enu_response_t const* enu_response__ = nullptr) const;

    /// Generate local orbital raidal functions
    int
    generate_lo_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__, core_states_t const& core__,
                                 std::vector<double> const* direction__ = nullptr,
                                 radial_basis_response_t* response__    = nullptr,
                                 core_states_t const* core_response__   = nullptr,
                                 enu_response_t const* enu_response__   = nullptr) const;

    /// Orthogonalize the radial functions.
    void
    orthogonalize_radial_functions(mdarray<double, 3>& rf__, mdarray<double, 3>* response__ = nullptr) const;

  public:
    /// Constructor
    relativity_t rel_;

    lapw_radial_basis_t(int id__, Atom_type const& atom_type__, relativity_t rel__, std::vector<double> potential__);

    lapw_radial_basis_t(int id__, Atom_type const& atom_type__)
        : lapw_radial_basis_t(id__, atom_type__, atom_type__.parameters().valence_relativity(), {})
    {
    }

    int
    generate_radial_functions()
    {
        return generate_radial_functions(rel_);
    }

    void
    generate_radial_integrals()
    {
        if (rel_ != relativity_t::dirac) {
            generate_radial_integrals(rel_);
        }
    }

    /// Set the spherical component of the potential
    /** Atoms belonging to the same symmetry class have the same spherical potential. */
    void
    set_spherical_potential(std::vector<double> const& vs__);

    auto const&
    spherical_potential() const
    {
        return spherical_potential_;
    }

    void
    set_spherical_xc_derivatives(std::array<std::vector<double>, 2> const& fields__);

    auto const&
    spherical_xc_derivatives() const
    {
        return spherical_xc_derivatives_;
    }

    /// Preserve immutable nuclear projectors when geometry changes the symmetry classes.
    void
    inherit_dirac_core_reference(lapw_radial_basis_t const& previous__)
    {
        if (&atom_type_ != &previous__.atom_type_) {
            RTE_THROW("Dirac core projector reuse requires the same atom type");
        }
        for (auto const& [channel, reference] : previous__.dirac_core_reference_) {
            dirac_core_reference_.try_emplace(channel, reference);
        }
    }

    /// Share core solver references between density generation and LAPW core projection.
    /** The angular channel is l for scalar cores and kappa for Dirac cores. */
    std::vector<radial_core_state_t>
    solve_core_states(Radial_grid<double> const& grid__, int channel__, std::vector<double> const& potential__,
                      int count__, std::array<std::vector<double>, 2> const& fields__, relativity_t relativity__,
                      std::vector<double> const* direction__                    = nullptr,
                      std::vector<radial_core_state_t>* response__              = nullptr,
                      std::array<std::vector<double>, 2> const& xc_directions__ = {}) const;

    /// Solve the channels required by the configured occupied core levels, retaining both Dirac components.
    /** States and responses have unit occupation and are indexed by n-l-1. Potential directions use the extended grid, while raw XC
     *  covectors and their directions use the MT grid without additional quadrature weights. */
    core_states_t
    solve_occupied_core_states(Radial_grid<double> const& grid__, std::vector<double> const& potential__,
                               std::array<std::vector<double>, 2> const& fields__, relativity_t relativity__,
                               std::vector<double> const* direction__ = nullptr, core_states_t* response__ = nullptr,
                               std::array<std::vector<double>, 2> const& xc_directions__ = {}) const;

    /// Save spherical potential for debugging purposes.
    void
    save_spherical_potential() const;

    /// Save radial functions.
    void
    save_radial_functions(std::string const& fname__) const;

    /// Generate APW and LO radial functions.
    int
    generate_radial_functions(relativity_t rel__, bool update_enu__ = true);

    /// Differentiate the AW/LO construction without changing the stored basis or energies.
    /** Directions are changes in the spherical potential and optional raw density/tau XC covectors on the MT
     *  grid. Empty XC directions mean zero. Core projection includes the induced core-orbital response without
     *  adding quadrature weights to the covectors. Automatic linearisation energies follow the potential
     *  unless update_enu__ is false. The free-atom reference remains fixed. Dirac valence functions
     *  retain both components and require Dirac states for XC-dependent core projection. */
    radial_basis_response_t
    radial_basis_potential_response(std::vector<double> const& direction__,
                                    std::array<std::vector<double>, 2> const& xc_directions__ = {},
                                    bool update_enu__                                         = true) const;

    void
    sync_radial_functions(mpi::Communicator const& comm__, int const rank__);

    void
    sync_radial_integrals(mpi::Communicator const& comm__, int const rank__);

    /// Check if local orbitals are linearly independent
    std::vector<int>
    check_lo_linear_independence(double etol__) const;

    /// Find linearization energy.
    int
    find_enu(relativity_t rel__);

    template <typename T>
    inline void
    write_enu(T& pout) const
    {
        pout << "Atom : " << atom_type_.symbol() << ", class id : " << id_ << std::endl;
        pout << "augmented waves" << std::endl;
        for (auto am : aw_channels()) {
            for (auto const& rsd : aw_descriptor(am)) {
                if (rsd.auto_enu) {
                    if (am.s()) {
                        pout << "kappa=" << am.kappa() << " ";
                    }
                    pout << rsd << std::endl;
                }
            }
        }

        pout << "local orbitals" << std::endl;
        for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
            for (size_t order = 0; order < lo_descriptor(idxlo).rsd_set.size(); order++) {
                auto& rsd = lo_descriptor(idxlo).rsd_set[order];
                if (rsd.auto_enu) {
                    pout << rsd << std::endl;
                }
            }
        }
        pout << std::endl;
    }

    /// Generate radial overlap and SO integrals
    /** In the case of spin-orbit interaction the following integrals are computed:
     *  \f[
     *      \int f_{p}(r) \Big( \frac{1}{(2 M c)^2} \frac{1}{r} \frac{d V}{d r} \Big) f_{p'}(r) r^2 dr
     *  \f]
     *
     *  Relativistic mass M is defined as
     *  \f[
     *      M = 1 - \frac{1}{2 c^2} V
     *  \f]
     */
    void
    generate_radial_integrals(relativity_t rel__);

    /// Physical spherical {H, O} in the full-j radial basis, diagonal in kappa and m_j.
    /** Uses both Dirac components and the rest-energy-subtracted Hamiltonian in Hartree.
     *  The kinetic bilinear is Hermitian; sphere boundary fluxes cancel only after matching
     *  to an interstitial operator with the same convention. No Y00 factor is included.
     *  A response returns dH/dO, including an optional direction of the physical potential. */
    std::array<mdarray<double, 2>, 2>
    dirac_spherical_matrices(radial_basis_response_t const* response__ = nullptr,
                             std::vector<double> const& direction__    = {}) const;

    /// Get m-th order radial derivative of AW functions at the MT surface.
    inline double
    aw_surface_deriv(int l__, int order__, int dm__) const
    {
        RTE_ASSERT(dm__ <= 2);
        auto idxrf = atom_type_.indexr().index_of(angular_momentum(l__), order__);
        return surface_derivatives_(dm__, idxrf);
    }

    /// Component-resolved boundary data of a full-j augmented wave.
    inline double
    aw_surface_deriv(angular_momentum am__, int order__, int dm__, int component__ = 0) const
    {
        RTE_ASSERT(dm__ >= 0 && dm__ <= 2 && component__ >= 0 && component__ < (am__.s() ? 2 : 1));
        return surface_derivatives_(dm__ + 3 * component__, atom_type_.indexr().index_of(am__, order__));
    }

    /// Set surface derivative of AW radial functions.
    inline void
    aw_surface_deriv(int l__, int order__, int dm__, double deriv__)
    {
        RTE_ASSERT(dm__ <= 2);
        surface_derivatives_(dm__, atom_type_.indexr().index_of(angular_momentum(l__), order__)) = deriv__;
    }

    /// Return symmetry class id.
    inline int
    id() const
    {
        return id_;
    }

    /// Get a value of the radial functions.
    inline double
    radial_function(int ir, int idx, int component__ = 0) const
    {
        RTE_ASSERT(component__ >= 0 && component__ < (atom_type_.indexb().full_j() ? 2 : 1));
        return radial_functions_(ir, idx, 2 * component__);
    }

    /// Set radial function.
    inline void
    radial_function(int idx__, std::vector<double> f__)
    {
        for (int ir = 0; ir < this->atom_type().num_mt_points(); ir++) {
            radial_functions_(ir, idx__, 0) = f__[ir];
        }
    }

    /// Get radial function derivative r*(du/dr).
    inline double
    radial_function_derivative(int ir__, int idx__, int component__ = 0) const
    {
        RTE_ASSERT(component__ >= 0 && component__ < (atom_type_.indexb().full_j() ? 2 : 1));
        return radial_functions_(ir__, idx__, 1 + 2 * component__);
    }

    /// Set radial function derivative r*(du/dr).
    inline void
    radial_function_derivative(int idx__, std::vector<double> f__)
    {
        for (int ir = 0; ir < this->atom_type().num_mt_points(); ir++) {
            radial_functions_(ir, idx__, 1) = f__[ir];
        }
    }

    inline double
    h_spherical_integral(int i1, int i2) const
    {
        return h_spherical_integrals_(i1, i2);
    }

    inline double const&
    o_radial_integral(int l, int order1, int order2) const
    {
        return o_radial_integrals_(l, order1, order2);
    }

    inline void
    set_o_radial_integral(int l, int order1, int order2, double oint__)
    {
        o_radial_integrals_(l, order1, order2) = oint__;
    }

    inline double const&
    o1_radial_integral(int xi1__, int xi2__) const
    {
        return o1_radial_integrals_(xi1__, xi2__);
    }

    inline void
    set_o1_radial_integral(int idxrf1__, int idxrf2__, double val__)
    {
        o1_radial_integrals_(idxrf1__, idxrf2__) = val__;
    }

    inline double
    so_radial_integral(int l, int order1, int order2) const
    {
        return so_radial_integrals_(l, order1, order2);
    }

    inline Atom_type const&
    atom_type() const
    {
        return atom_type_;
    }

    inline int
    num_aw_descriptors() const
    {
        return static_cast<int>(aw_descriptors_.size());
    }

    inline radial_solution_descriptor_set&
    aw_descriptor(int idx__) const
    {
        return aw_descriptor(angular_momentum(idx__));
    }

    inline radial_solution_descriptor_set&
    aw_descriptor(angular_momentum am__) const
    {
        if (bool(am__.s()) != atom_type_.indexb().full_j()) {
            RTE_THROW("AW descriptor requires the initialized scalar or full-j angular channel");
        }
        return am__.s() ? dirac_aw_descriptors_.at(am__.kappa()) : aw_descriptors_.at(am__.l());
    }

    inline int
    num_lo_descriptors() const
    {
        return static_cast<int>(lo_descriptors_.size());
    }

    inline local_orbital_descriptor&
    lo_descriptor(int idx__) const
    {
        return lo_descriptors_[idx__];
    }

    inline void
    set_aw_enu(int l, int order, double enu)
    {
        aw_descriptor(l).at(order).enu = enu;
    }

    inline double
    get_aw_enu(int l, int order) const
    {
        return aw_descriptor(l).at(order).enu;
    }

    inline void
    set_lo_enu(int idxlo, int order, double enu)
    {
        lo_descriptors_[idxlo].rsd_set[order].enu = enu;
    }

    inline double
    get_lo_enu(int idxlo, int order) const
    {
        return lo_descriptors_[idxlo].rsd_set[order].enu;
    }
};

class LAPW_radial_basis
{
  private:
    std::vector<lapw_radial_basis_t> radial_basis_of_symmetry_class;
    std::vector<int> class_of_atom_;
    Unit_cell const& unit_cell_;

    void
    sync_radial_functions(int ic__, int rank__)
    {
        auto& rb = radial_basis_of_symmetry_class[ic__];

        rb.sync_radial_functions(unit_cell_.comm(), rank__);
    }

    void
    sync_radial_integrals(int ic__, int rank__)
    {
        auto& rb = radial_basis_of_symmetry_class[ic__];

        rb.sync_radial_integrals(unit_cell_.comm(), rank__);
    }

  public:
    LAPW_radial_basis(Unit_cell const& unit_cell__, relativity_t rel__, std::vector<std::vector<double>> vs__,
                      std::vector<std::array<std::vector<double>, 2>> const& xc__ = {},
                      LAPW_radial_basis const* previous__                         = nullptr)
        : unit_cell_{unit_cell__}
    {
        for (int ia = 0; ia < unit_cell__.num_atoms(); ia++) {
            class_of_atom_.push_back(unit_cell__.atom(ia).symmetry_class().id());
        }
        for (int ic = 0; ic < unit_cell__.num_atom_symmetry_classes(); ic++) {
            radial_basis_of_symmetry_class.emplace_back(ic, unit_cell__.atom_symmetry_class(ic).atom_type(), rel__,
                                                        vs__[ic]);
            auto& radial = radial_basis_of_symmetry_class.back();
            if (!xc__.empty()) {
                radial.set_spherical_xc_derivatives(xc__.at(ic));
            }
            if (previous__) {
                int ia = unit_cell__.atom_symmetry_class(ic).atom_id(0);
                radial.inherit_dirac_core_reference(previous__->radial_basis(atom_index_t::global(ia)));
            }
        }

        generate_radial_functions();
        generate_radial_integrals();
    }

    void
    generate_radial_functions()
    {
        int failed{0};
        try {
            for (auto it : unit_cell_.spl_num_atom_symmetry_classes()) {
                failed |= radial_basis_of_symmetry_class[it.i].generate_radial_functions();
            }
        } catch (std::exception const&) {
            failed = 1;
        }
        unit_cell_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("LAPW radial functions could not be generated on a symmetry-class owner");
        }
        for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
            sync_radial_functions(
                    ic,
                    unit_cell_.spl_num_atom_symmetry_classes().location(atom_symmetry_class_index_t::global(ic)).ib);
        }
    }

    void
    generate_radial_integrals()
    {
        int failed{0};
        try {
            for (auto it : unit_cell_.spl_num_atom_symmetry_classes()) {
                radial_basis_of_symmetry_class[it.i].generate_radial_integrals();
            }
        } catch (std::exception const&) {
            failed = 1;
        }
        unit_cell_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("LAPW radial integrals could not be generated on a symmetry-class owner");
        }
        for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
            sync_radial_integrals(
                    ic,
                    unit_cell_.spl_num_atom_symmetry_classes().location(atom_symmetry_class_index_t::global(ic)).ib);
        }
    }

    Unit_cell const&
    unit_cell() const
    {
        return unit_cell_;
    }

    lapw_radial_basis_t const&
    radial_basis_for_class(int ic__) const
    {
        return radial_basis_of_symmetry_class.at(ic__);
    }

    lapw_radial_basis_t&
    radial_basis_for_class(int ic__)
    {
        return radial_basis_of_symmetry_class.at(ic__);
    }

    lapw_radial_basis_t const&
    radial_basis(typename atom_index_t::global ia__) const
    {
        int icls = class_of_atom_.at(ia__);
        return radial_basis_of_symmetry_class[icls];
    }

    lapw_radial_basis_t&
    radial_basis(typename atom_index_t::global ia__)
    {
        return radial_basis_of_symmetry_class.at(class_of_atom_.at(ia__));
    }

    inline void
    write_enu(std::ostream& out__) const
    {
        if (unit_cell_.parameters().verbosity() >= 2) {
            mpi::pstdout pout(unit_cell_.comm());
            if (unit_cell_.comm().rank() == 0) {
                pout << std::endl << "Linearization energies" << std::endl;
            }

            for (auto it : unit_cell_.spl_num_atom_symmetry_classes()) {
                radial_basis_of_symmetry_class[it.i].write_enu(pout);
            }
            RTE_OUT(out__) << pout.flush(0);
        }
    }
};

} // namespace sirius

#endif
