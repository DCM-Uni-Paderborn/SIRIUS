/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hamiltonian.cpp
 *
 *  \brief Contains definition of sirius::Hamiltonian0 class.
 */

#include <memory>
#include "potential/potential.hpp"
#include "lapw/interstitial_functions.hpp"
#include "lapw/step_function.hpp"
#include "density/lapw_xc_atom.hpp"
#include "lapw/dirac_basis.hpp"
#include "local_operator.hpp"
#include "hamiltonian.hpp"

namespace sirius {
namespace {

double
dirac_field_component(int ns, int component, int spin)
{
    return ns == 4 ? double(component == spin)
                   : (ns == 1 ? double(component == 0)
                              : (component < 2 ? 0.5 * (component == 0 ? 1 : 1 - 2 * spin) : 0.0));
}

std::vector<double>
lapw_radial_product_weights(Radial_grid<double> const& grid)
{
    Spline<double> prototype(grid);
    mdarray<double, 2> integrals({grid.num_points(), 4});
    integrals.zero();
    for (int ir = 0; ir < grid.num_points() - 1; ir++) {
        double r = grid[ir], h = grid[ir + 1] - r;
        for (int j = 0; j < 4; j++) {
            integrals(ir, j) = r * r * std::pow(h, j + 1) / (j + 1) + 2 * r * std::pow(h, j + 2) / (j + 2) +
                               std::pow(h, j + 3) / (j + 3);
        }
    }
    return prototype.interpolation_adjoint(integrals);
}

std::complex<double>
lapw_so_angular(Atom_type const& type, int i, int j, int s)
{
    auto const& a = type.indexb(i);
    auto const& b = type.indexb(j);
    int l         = a.am.l();
    if (s == 0 || l != b.am.l() || l > type.lmax_apw()) {
        return 0;
    }
    if (s == 1) {
        return a.m == b.m ? a.m : 0;
    }
    double up_down = a.m == b.m + 1 ? std::sqrt(l * (l + 1.0) - a.m * (a.m - 1.0)) : 0;
    double down_up = b.m == a.m + 1 ? std::sqrt(l * (l + 1.0) - b.m * (b.m - 1.0)) : 0;
    return s == 2 ? std::complex<double>{0.5 * (up_down + down_up), 0}
                  : std::complex<double>{0, 0.5 * (up_down - down_up)};
}

} // namespace

// TODO: radial integrals for the potential should be computed here; the problem is that they also can be set
//       externally by the host code

template <typename T>
Hamiltonian0<T>::Hamiltonian0(Potential& potential__, bool precompute_lapw__, bool update_lapw_rf__)
    : ctx_(potential__.ctx())
    , potential_(&potential__)
    , unit_cell_(potential__.ctx().unit_cell())
{
    PROFILE("sirius::Hamiltonian0");

    if (ctx_.full_potential() && ctx_.valence_relativity() == relativity_t::dirac) {
        int invalid = !std::is_same_v<T, double> || ctx_.cfg().control().use_second_variation() ||
                      ctx_.so_correction() || ctx_.hubbard_correction();
        ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("Dirac LAPW matrix assembly requires fp64 full variation without a separate SOC or Hubbard "
                      "operator");
        }
        if (precompute_lapw__) {
            potential_->update_atomic_potential();
            if (update_lapw_rf__) {
                unit_cell_.generate_radial_functions(ctx_.out());
            }
        }
        prepare_dirac_fields();
        return;
    }

    local_op_ =
            std::make_unique<Local_operator<T>>(ctx_, ctx_.spfft_coarse<T>(), ctx_.gvec_coarse_fft_sptr(), potential__);

    if (!ctx_.full_potential()) {
        d_op_ = std::make_unique<D_operator<T>>(potential__);
        q_op_ = std::make_unique<Q_operator<T>>(ctx_);
        if (ctx_.cfg().iterative_solver().type() == "exact") {
            this->generate_pw_coefs(potential__);
        }
    }
    if (ctx_.full_potential()) {
        if (precompute_lapw__) {
            if (ctx_.cfg().iterative_solver().type() == "exact") {
                this->generate_pw_coefs(potential__);
            }
            potential_->update_atomic_potential();
            if (update_lapw_rf__) {
                ctx_.unit_cell().generate_radial_functions(ctx_.out());
            }
            ctx_.unit_cell().generate_radial_integrals();
        }
        auto pu = ctx_.processing_unit();

        hmt_ = std::vector<mdarray<std::complex<T>, 3>>(ctx_.unit_cell().num_atoms());
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            auto& atom = ctx_.unit_cell().atom(ia);
            auto& type = atom.type();

            int nmt  = type.mt_basis_size();
            hmt_[ia] = mdarray<std::complex<T>, 3>({nmt, nmt, ctx_.num_mag_dims() + 1}, mdarray_label("hmt"));
        }

        struct mt_constraint_t
        {
            int l;
            std::vector<std::vector<std::vector<double>>> matrix;
        };

        /* each atom might have several constraints */
        std::vector<std::vector<mt_constraint_t>> mt_constraints;
        if (ctx_.hubbard_constrained_calculation() && ctx_.num_mag_dims() == 1 &&
            !ctx_.cfg().control().use_second_variation() &&
            ctx_.num_constraints_applied() < ctx_.cfg().hubbard().constraint().maxiter()) {

            mt_constraints = std::vector<std::vector<mt_constraint_t>>(ctx_.unit_cell().num_atoms());

            for (int i = 0; i < ctx_.cfg().hubbard().constraint().local().size(); i++) {
                auto const& constraint = ctx_.cfg().hubbard().constraint().local(i);
                int ia                 = constraint.atom_index();
                int l                  = constraint.l();
                if (ia < 0 || ia >= ctx_.unit_cell().num_atoms()) {
                    RTE_THROW("wrong atom index in local Hubbard constraint");
                }
                if (l < 0) {
                    RTE_THROW("wrong angular momentum in local Hubbard constraint");
                }

                int mmax = 2 * l + 1;

                auto matrix = constraint.occupancy();

                for (auto const& spin_block : matrix) {
                    if (static_cast<int>(spin_block.size()) != mmax) {
                        RTE_THROW("wrong number of rows in local Hubbard constraint");
                    }
                    for (auto const& row : spin_block) {
                        if (static_cast<int>(row.size()) != mmax) {
                            RTE_THROW("wrong number of columns in local Hubbard constraint");
                        }
                    }
                }
                mt_constraints[ia].push_back({l, std::move(matrix)});
            }
            ctx_.num_constraints_applied(1);
        }

        #pragma omp parallel for
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            auto& atom = ctx_.unit_cell().atom(ia);
            auto& type = atom.type();

            int nmt = type.mt_basis_size();

            /* compute muffin-tin Hamiltonian */
            for (int j2 = 0; j2 < nmt; j2++) {
                int lm2    = type.indexb(j2).lm;
                int idxrf2 = type.indexb(j2).idxrf;
                for (int j1 = 0; j1 < nmt; j1++) {
                    int lm1    = type.indexb(j1).lm;
                    int idxrf1 = type.indexb(j1).idxrf;
                    switch (ctx_.num_mag_dims()) {
                        case 3: {
                            // spin-block index is consistent with non-local pseudopotential operator
                            // 0, 1: V, Bz in second variation; V + Bz, V - Bz otherwise
                            // 2: Bx - i By
                            // 3: Bx + i By

                            // Bx - i By
                            hmt_[ia](j1, j2, 2) = atom.template radial_integrals_sum_L3<4>(
                                    {0, 0, 1, -1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                            // Bx + i By
                            hmt_[ia](j1, j2, 3) = atom.template radial_integrals_sum_L3<4>(
                                    {0, 0, 1, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                        }
                        case 1: {
                            if (ctx_.cfg().control().use_second_variation()) {
                                hmt_[ia](j1, j2, 0) = atom.template radial_integrals_sum_L3<2>(
                                        {1, 0}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                                hmt_[ia](j1, j2, 1) = atom.template radial_integrals_sum_L3<2>(
                                        {0, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                            } else {
                                hmt_[ia](j1, j2, 0) = atom.template radial_integrals_sum_L3<2>(
                                        {1, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                                hmt_[ia](j1, j2, 1) = atom.template radial_integrals_sum_L3<2>(
                                        {1, -1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));

                                if (!mt_constraints.empty()) {
                                    /* add constraints */
                                    int l1 = type.indexb(j1).am.l();
                                    int l2 = type.indexb(j2).am.l();
                                    for (auto const& constraint : mt_constraints[ia]) {
                                        if (l1 == constraint.l && l2 == constraint.l) {
                                            int m1     = type.indexb(j1).m;
                                            int m2     = type.indexb(j2).m;
                                            int order1 = type.indexb(j1).order;
                                            int order2 = type.indexb(j2).order;
                                            double ori = atom.symmetry_class().o_radial_integral(constraint.l, order1,
                                                                                                 order2);
                                            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                                                hmt_[ia](j1, j2, ispn) +=
                                                        constraint.matrix[ispn][constraint.l + m1][constraint.l + m2] *
                                                        ori;
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 0: {
                            hmt_[ia](j1, j2, 0) = atom.template radial_integrals_sum_L3<1>(
                                    {1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
                        }
                    }
                }
            }
        }
        if (auto const* d = potential__.lapw_xc_derivatives()) {
            // Contract after any radial-basis update, using raw sample covectors.
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                LAPW_xc_atom op(unit_cell_.atom(ia), ctx_.lmax_rho());
                bool spinor = ctx_.num_mag_dims() == 3;
                for (int s = 0; s < (spinor ? 4 : ctx_.num_spins()); s++) {
                    auto matrix = op.matrix_elements(d->local[ia][0][s], d->local[ia][1][s]);
                    for (int j = 0; j < unit_cell_.atom(ia).type().mt_basis_size(); j++) {
                        for (int i = 0; i < unit_cell_.atom(ia).type().mt_basis_size(); i++) {
                            if (spinor) {
                                if (s < 2) {
                                    hmt_[ia](i, j, s) += static_cast<std::complex<T>>(matrix(i, j));
                                } else {
                                    auto value = static_cast<std::complex<T>>(matrix(i, j));
                                    hmt_[ia](i, j, 2) += s == 2 ? value : std::complex<T>(0, -1) * value;
                                    hmt_[ia](i, j, 3) += s == 2 ? value : std::complex<T>(0, 1) * value;
                                }
                            } else if (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()) {
                                hmt_[ia](i, j, 0) += static_cast<std::complex<T>>(0.5 * matrix(i, j));
                                hmt_[ia](i, j, 1) += static_cast<std::complex<T>>(0.5 * (1 - 2 * s) * matrix(i, j));
                            } else {
                                hmt_[ia](i, j, s) += static_cast<std::complex<T>>(matrix(i, j));
                            }
                        }
                    }
                }
            }
        }
        if (pu == device_t::GPU) {
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                hmt_[ia].allocate(memory_t::device).copy_to(memory_t::device);
            }
        }
    }
}

template <typename T>
Hamiltonian0<T>::~Hamiltonian0()
{
}

template <typename T>
void
Hamiltonian0<T>::prepare_dirac_fields()
{
    int invalid{0};
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
        auto const& cls = unit_cell_.atom_symmetry_class(ic);
        invalid |= cls.spherical_potential().size() != static_cast<size_t>(cls.atom_type().num_mt_points());
        for (double value : cls.spherical_potential()) {
            invalid |= !std::isfinite(value);
        }
    }
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Dirac LAPW fields require an initialized finite spherical atomic potential");
    }
    int ng   = ctx_.gvec().num_gvec();
    veff_pw_ = mdarray<std::complex<T>, 2>({ng, 4});
    vtau_pw_ = mdarray<std::complex<T>, 2>({ng, 4});
    veff_pw_.zero();
    vtau_pw_.zero();
    Smooth_periodic_function<double> masked(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    for (int s = 0; s <= ctx_.num_mag_dims(); s++) {
        for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
            masked.value(ir) = ctx_.theta(ir) * potential_->component(s).rg().value(ir);
        }
        masked.fft_transform(-1);
        auto values = masked.gather_f_pw();
        for (int ig = 0; ig < ng; ig++) {
            veff_pw_(ig, s) = static_cast<std::complex<T>>(values[ig]);
        }
    }
    auto const* xc = potential_->lapw_xc_derivatives();
    int ns         = ctx_.num_mag_dims() == 3 ? 4 : ctx_.num_spins();
    if (xc) {
        for (int s = 0; s < ns; s++) {
            auto rho = xc->rho[s].gather_f_pw(), tau = xc->tau[s].gather_f_pw();
            for (int component = 0; component < 4; component++) {
                double factor = dirac_field_component(ns, component, s);
                for (int ig = 0; ig < ng; ig++) {
                    // These covectors already contain the quadrature and interstitial restriction.
                    veff_pw_(ig, component) += static_cast<std::complex<T>>(factor * rho[ig]);
                    vtau_pw_(ig, component) += static_cast<std::complex<T>>(factor * tau[ig]);
                }
            }
        }
    }
    hmt_ = dirac_atom_matrices();
}

template <typename T>
std::vector<mdarray<std::complex<T>, 3>>
Hamiltonian0<T>::dirac_atom_matrices(std::vector<radial_basis_response_t> const* response__) const
{
    auto const* xc = potential_->lapw_xc_derivatives();
    int ns         = ctx_.num_mag_dims() == 3 ? 4 : ctx_.num_spins();
    int lmmax      = sf::lmmax(std::max(ctx_.lmax_pot(), ctx_.lmax_rho()));
    std::vector<mdarray<std::complex<T>, 3>> matrices;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto const& atom = unit_cell_.atom(ia);
        auto const& type = atom.type();
        auto const& grid = type.radial_grid();
        int nr = grid.num_points(), nb = type.mt_basis_size();
        int owner = unit_cell_.spl_num_atoms().location(atom_index_t::global(ia)).ib;
        mdarray<double, 3> values({ctx_.lmmax_pot(), nr, ctx_.num_mag_dims() + 1});
        if (ctx_.comm().rank() == owner) {
            for (int s = 0; s <= ctx_.num_mag_dims(); s++) {
                std::copy_n(potential_->component(s).mt()[ia].at(memory_t::host), ctx_.lmmax_pot() * nr,
                            &values(0, 0, s));
            }
        }
        ctx_.comm().bcast(values.at(memory_t::host), values.size(), owner);
        // Exact adjoint of integrating a sampled-product cubic spline with r^2 dr.
        auto weights = lapw_radial_product_weights(grid);
        LAPW_xc_atom op(atom, sf::lmax(lmmax));
        matrices.emplace_back(std::array<index_range, 3>{nb, nb, 1});
        matrices.back().zero();
        for (int component = 0; component < 4; component++) {
            Flm rho(lmmax, grid), tau(lmmax, grid);
            rho.zero();
            tau.zero();
            if (component <= ctx_.num_mag_dims()) {
                for (int ir = 0; ir < nr; ir++) {
                    for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                        double value = values(lm, ir, component);
                        if (component == 0 && lm == 0) {
                            value -= atom.symmetry_class().spherical_potential()[ir] / y00;
                        }
                        rho(lm, ir) = weights[ir] * value;
                    }
                }
            }
            if (xc) {
                for (int s = 0; s < ns; s++) {
                    double factor = dirac_field_component(ns, component, s);
                    for (int ir = 0; ir < nr; ir++) {
                        for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                            rho(lm, ir) += factor * xc->local[ia][0][s](lm, ir);
                            tau(lm, ir) += factor * xc->local[ia][1][s](lm, ir);
                        }
                    }
                }
            }
            auto matrix = op.matrix_elements(
                    rho, tau, response__ ? &(*response__)[atom.symmetry_class().id()] : nullptr, component);
            for (size_t i = 0; i < matrix.size(); i++) {
                matrices.back()[i] += static_cast<std::complex<T>>(matrix[i]);
            }
        }
    }
    return matrices;
}

template <typename T>
std::array<mdarray<std::complex<double>, 2>, 2>
Hamiltonian0<T>::dirac_matrices(fft::Gvec const& gkvec__) const
{
    int invalid = !std::is_same_v<T, double> || !ctx_.full_potential() ||
                  ctx_.valence_relativity() != relativity_t::dirac || local_op_ != nullptr ||
                  hmt_.size() != static_cast<size_t>(unit_cell_.num_atoms());
    gkvec__.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Dirac matrices require the prepared native full-j Hamiltonian");
    }
    if constexpr (std::is_same_v<T, double>) {
        Dirac_lapw_basis basis(unit_cell_, gkvec__);
        std::vector<mdarray<std::complex<double>, 2>> local;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int nb = unit_cell_.atom(ia).type().mt_basis_size();
            local.emplace_back(std::array<index_range, 2>{nb, nb},
                               const_cast<std::complex<double>*>(hmt_[ia].at(memory_t::host)));
        }
        auto h = basis.hamiltonian();
        auto v = basis.field_operator(ctx_.gvec(), veff_pw_, vtau_pw_, local);
        for (size_t i = 0; i < h.size(); i++) {
            h[i] += v[i];
        }
        return {std::move(h), basis.overlap()};
    }
    RTE_THROW("Dirac matrices require fp64");
}

template <typename T>
mdarray<std::complex<double>, 2>
Hamiltonian0<T>::dirac_field_response(fft::Gvec const& gkvec__, lapw_hamiltonian_response_t const& response__) const
{
    int invalid = !std::is_same_v<T, double> || !ctx_.full_potential() ||
                  ctx_.valence_relativity() != relativity_t::dirac || response__.context != &ctx_ ||
                  response__.scalar_pw.size(1) != 4 || response__.kinetic_pw.size(1) != 4 ||
                  response__.local.size() != static_cast<size_t>(unit_cell_.num_atoms()) ||
                  response__.overlap_pw.size() || !response__.overlap_local.empty();
    if (!invalid) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int nb = unit_cell_.atom(ia).type().mt_basis_size();
            invalid |= response__.local[ia].size(0) != static_cast<size_t>(nb) ||
                       response__.local[ia].size(1) != static_cast<size_t>(nb) || response__.local[ia].size(2) != 1 ||
                       !response__.local[ia].on_host();
        }
    }
    gkvec__.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Dirac field response requires prepared canonical covectors and full-j atom matrices");
    }
    Dirac_lapw_basis basis(unit_cell_, gkvec__);
    std::vector<mdarray<std::complex<double>, 2>> local;
    for (auto const& atom : response__.local) {
        local.emplace_back(std::array<index_range, 2>{atom.size(0), atom.size(1)},
                           const_cast<std::complex<double>*>(atom.at(memory_t::host)));
    }
    return basis.field_operator(ctx_.gvec(), response__.scalar_pw, response__.kinetic_pw, local);
}

template <typename T>
std::array<mdarray<std::complex<double>, 2>, 2>
Hamiltonian0<T>::dirac_radial_response(fft::Gvec const& gkvec__, std::vector<radial_basis_response_t> const& response__,
                                       std::vector<lapw_atom_basis_response_t> const& atoms__) const
{
    int invalid = !std::is_same_v<T, double> || !ctx_.full_potential() ||
                  ctx_.valence_relativity() != relativity_t::dirac ||
                  atoms__.size() != static_cast<size_t>(unit_cell_.num_atoms()) || atoms__.size() != hmt_.size();
    if (!invalid) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int n         = unit_cell_.atom(ia).type().mt_basis_size();
            auto const& h = atoms__[ia].h;
            invalid |= h.size(0) != static_cast<size_t>(n) || h.size(1) != static_cast<size_t>(n) || h.size(2) != 1 ||
                       !h.on_host();
        }
    }
    gkvec__.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Dirac radial response requires native full-j atom matrices");
    }
    if constexpr (std::is_same_v<T, double>) {
        Dirac_lapw_basis basis(unit_cell_, gkvec__);
        std::vector<mdarray<std::complex<double>, 2>> local, tangent;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int n = unit_cell_.atom(ia).type().mt_basis_size();
            local.emplace_back(std::array<index_range, 2>{n, n},
                               const_cast<std::complex<double>*>(hmt_[ia].at(memory_t::host)));
            tangent.emplace_back(std::array<index_range, 2>{n, n},
                                 const_cast<std::complex<double>*>(atoms__[ia].h.at(memory_t::host)));
        }
        // The spherical auxiliary potential cancels against the atom-local subtraction.
        // Only radial functions and matching coefficients change here; physical field changes are separate.
        auto h = basis.hamiltonian(&response__);
        auto v = basis.field_operator(ctx_.gvec(), veff_pw_, vtau_pw_, local, &response__, &tangent);
        for (size_t i = 0; i < h.size(); i++) {
            h[i] += v[i];
        }
        return {std::move(h), basis.overlap(&response__)};
    }
    RTE_THROW("Dirac radial response requires fp64");
}

template <typename T>
std::array<mdarray<std::complex<double>, 2>, 2>
Hamiltonian0<T>::dirac_geometry_response(fft::Gvec const& gkvec__, xc_geometry_direction_t const& direction__) const
{
    int invalid = !std::is_same_v<T, double> || !ctx_.full_potential() ||
                  ctx_.valence_relativity() != relativity_t::dirac || local_op_ != nullptr ||
                  hmt_.size() != static_cast<size_t>(unit_cell_.num_atoms()) ||
                  direction__.atoms.size() != static_cast<size_t>(unit_cell_.num_atoms());
    gkvec__.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Dirac geometry response requires a prepared full-j Hamiltonian and matching atom directions");
    }
    if constexpr (std::is_same_v<T, double>) {
        mdarray<double, 2> atoms({3, unit_cell_.num_atoms()});
        r3::matrix<double> lattice;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            for (int x = 0; x < 3; x++) {
                atoms(x, ia) = direction__.atoms[ia][x];
            }
        }
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                lattice(x, y) = direction__.lattice[x + 3 * y];
            }
        }
        auto strain = dot(lattice, unit_cell_.inverse_lattice_vectors());
        Dirac_lapw_basis basis(unit_cell_, gkvec__);
        std::vector<mdarray<std::complex<double>, 2>> local;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int n = unit_cell_.atom(ia).type().mt_basis_size();
            local.emplace_back(std::array<index_range, 2>{n, n},
                               const_cast<std::complex<double>*>(hmt_[ia].at(memory_t::host)));
        }
        auto result = basis.geometry_response(atoms, strain);
        auto v      = basis.field_operator(ctx_.gvec(), veff_pw_, vtau_pw_, local, nullptr, nullptr, &atoms, &strain);
        for (size_t i = 0; i < v.size(); i++) {
            result[0][i] += v[i];
        }
        return result;
    }
    RTE_THROW("Dirac geometry response requires fp64");
}

template <typename T>
void
Hamiltonian0<T>::apply_hmt_to_apw(device_t pu__, int ia__, int ispn__, int ngv__,
                                  mdarray<std::complex<T>, 2> const& alm__, mdarray<std::complex<T>, 2>& halm__,
                                  int stream_id__) const
{
    auto& type = ctx_.unit_cell().atom(ia__).type();

    auto la  = (pu__ == device_t::CPU) ? la::lib_t::blas : la::lib_t::gpublas;
    auto mem = (pu__ == device_t::CPU) ? memory_t::host : memory_t::device;

    la::wrap(la).gemm('N', 'T', ngv__, type.mt_aw_basis_size(), type.mt_aw_basis_size(),
                      &la::constant<std::complex<T>>::one(), alm__.at(mem), alm__.ld(),
                      hmt_[ia__].at(mem, 0, 0, ispn__), hmt_[ia__].ld(), &la::constant<std::complex<T>>::zero(),
                      halm__.at(mem), halm__.ld(), acc::stream_id(stream_id__));
}

template <typename T>
void
Hamiltonian0<T>::add_o1mt_to_apw(Atom const& atom__, int num_gkvec__, mdarray<std::complex<T>, 2>& alm__) const
{
    // TODO: optimize for the loop layout using blocks of G-vectors
    auto& type = atom__.type();
    std::vector<std::complex<T>> alm(type.mt_aw_basis_size());
    std::vector<std::complex<T>> oalm(type.mt_aw_basis_size());
    for (int ig = 0; ig < num_gkvec__; ig++) {
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            alm[j] = oalm[j] = alm__(ig, j);
        }
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            int l     = type.indexb(j).am.l();
            int lm    = type.indexb(j).lm;
            int idxrf = type.indexb(j).idxrf;
            for (int order = 0; order < type.aw_order(l); order++) {
                int j1     = type.indexb().index_by_lm_order(lm, order);
                int idxrf1 = type.indexr().index_of(angular_momentum(l), order);
                oalm[j] += static_cast<const T>(atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1)) * alm[j1];
            }
        }
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            alm__(ig, j) = oalm[j];
        }
    }
}

template <typename T>
void
Hamiltonian0<T>::apply_bmt(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const
{
    for (auto it : psi__.spl_num_atoms()) {
        auto ia           = it.i;
        auto& atom        = unit_cell_.atom(ia);
        int mt_basis_size = atom.type().mt_basis_size();

        auto const& zm = hmt_[ia];

        /* compute bwf = B_z*|wf_j> */
        la::wrap(la::lib_t::blas)
                .hemm('L', 'U', mt_basis_size, ctx_.num_fv_states(), &la::constant<std::complex<T>>::one(),
                      zm.at(memory_t::host, 0, 0, 1), zm.ld(),
                      &psi__.mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), psi__.ld(),
                      &la::constant<std::complex<T>>::zero(),
                      &bpsi__[0].mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), bpsi__[0].ld());

        /* compute bwf = (B_x - iB_y)|wf_j> */
        if (bpsi__.size() == 3) {
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', mt_basis_size, ctx_.num_fv_states(), mt_basis_size,
                          &la::constant<std::complex<T>>::one(), zm.at(memory_t::host, 0, 0, 2), zm.ld(),
                          &psi__.mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), psi__.ld(),
                          &la::constant<std::complex<T>>::zero(),
                          &bpsi__[2].mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), bpsi__[2].ld());
        }
    }
}

template <typename T>
void
Hamiltonian0<T>::apply_so_correction(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& hpsi__) const
{
    PROFILE("sirius::Hamiltonian0::apply_so_correction");

    wf::spin_index s(0);

    for (auto it : psi__.spl_num_atoms()) {
        auto ia    = it.i;
        auto& atom = unit_cell_.atom(ia);
        auto a     = it.li;

        for (int l = 0; l <= atom.type().lmax_apw(); l++) {
            /* number of radial functions for this l */
            int nrf = atom.type().indexr().max_order(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                for (int order2 = 0; order2 < nrf; order2++) {
                    T sori = atom.symmetry_class().so_radial_integral(l, order1, order2);

                    for (int m = -l; m <= l; m++) {
                        int idx1 = atom.type().indexb_by_l_m_order(l, m, order1);
                        int idx2 = atom.type().indexb_by_l_m_order(l, m, order2);
                        int idx3 = (m + l != 0) ? atom.type().indexb_by_l_m_order(l, m - 1, order2) : 0;
                        // int idx4 = (m - l != 0) ? atom.type().indexb_by_l_m_order(l, m + 1, order2) : 0;

                        for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
                            wf::band_index b(ist);
                            auto z1 = psi__.mt_coeffs(idx2, a, s, b) * T(m) * sori;
                            /* u-u part */
                            hpsi__[0].mt_coeffs(idx1, a, s, b) += z1;
                            /* d-d part */
                            hpsi__[1].mt_coeffs(idx1, a, s, b) -= z1;
                            /* apply L_{-} operator; u-d part */
                            if (m + l) {
                                hpsi__[2].mt_coeffs(idx1, a, s, b) +=
                                        psi__.mt_coeffs(idx3, a, s, b) * sori * std::sqrt(T(l * (l + 1) - m * (m - 1)));
                            }
                            /* for the d-u part */
                            ///* apply L_{+} operator */
                            // if (m - l) {
                            //    hpsi[3].mt_coeffs(0).prime().at(memory_t::host, offset + idx1, ist) +=
                            //        fv_states__.mt_coeffs(0).prime().at(memory_t::host, offset + idx4, ist) *
                            //            sori * std::sqrt(double(l * (l + 1) - m * (m + 1)));
                            //}
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void
Hamiltonian0<T>::generate_pw_coefs(Potential const& potential__)
{
    PROFILE("sirius::Hamiltonian0::generate_pw_coefs");

    if (ctx_.full_potential() && ctx_.valence_relativity() == relativity_t::dirac) {
        RTE_THROW("Dirac LAPW fields are prepared by the four-component Hamiltonian constructor");
    }
    if (!ctx_.full_potential()) {
        // These gathers involve all context ranks, including ranks with no k-point.
        auto gather = [&](auto const& scalar, auto const* magnetic, auto& dest) {
            dest   = mdarray<std::complex<T>, 2>({ctx_.gvec().num_gvec(), ctx_.num_spins()});
            auto v = scalar.gather_f_pw();
            std::vector<std::complex<double>> b;
            if (magnetic) {
                b = magnetic->gather_f_pw();
            }
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
                    dest(ig, s) = static_cast<std::complex<T>>(v[ig] + (b.empty() ? 0.0 : double(1 - 2 * s) * b[ig]));
                }
            }
        };
        gather(potential__.effective_potential().rg(),
               ctx_.num_mag_dims() == 1 ? &potential__.effective_magnetic_field(0).rg() : nullptr, veff_pw_);
        if (potential__.has_kinetic_potential()) {
            auto const& t = potential__.kinetic_potential();
            gather(t.scalar().rg(), ctx_.num_mag_dims() == 1 ? &t.vector(0).rg() : nullptr, vtau_pw_);
        }
        return;
    }

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            rm2_inv_pw_ = mdarray<std::complex<T>, 1>({ctx_.gvec().num_gvec()});
        }
        case relativity_t::zora: {
            rm_inv_pw_ = mdarray<std::complex<T>, 1>({ctx_.gvec().num_gvec()});
        }
        default: {
            if (ctx_.cfg().control().use_second_variation()) {
                veff_pw_ = mdarray<std::complex<T>, 2>({ctx_.gvec().num_gvec(), 1});
            } else {
                veff_pw_ = mdarray<std::complex<T>, 2>({ctx_.gvec().num_gvec(), ctx_.num_mag_dims() + 1});
            }
        }
    }

    int gv_count = this->ctx_.gvec_fft().count();

    auto& fft = this->ctx_.spfft<T>();

    /* temporaty output buffer */
    mdarray<std::complex<T>, 1> fpw_fft({gv_count});

    auto get_fft_result = [&](std::complex<T>* ptr) {
        fft.forward(SPFFT_PU_HOST, reinterpret_cast<T*>(&fpw_fft[0]), SPFFT_FULL_SCALING);
        ctx_.gvec_fft().gather_pw_global(&fpw_fft[0], ptr);
    };

    switch (ctx_.valence_relativity()) {
        case relativity_t::iora: {
            auto m = interstitial_mass<2>(potential__);
            fft::spfft_input<T>(fft, m);
            get_fft_result(&rm2_inv_pw_[0]);
        }
        case relativity_t::zora: {
            auto m = interstitial_mass<1>(potential__);
            fft::spfft_input<T>(fft, m);
            get_fft_result(&rm_inv_pw_[0]);
        }
        default: {
        }
    }
    if (ctx_.cfg().control().use_second_variation() || ctx_.num_mag_dims() == 0) {
        auto v = interstitial_potential(potential__, 0);
        fft::spfft_input<T>(fft, v);
        get_fft_result(&veff_pw_[0]);
    } else {
        switch (ctx_.num_mag_dims()) {
            case 3: {
                // spin-block index always has this order
                // 0: V + Bz
                // 1: V - Bz
                // 2: Bx - i By
                // 3: Bx + i By
                auto v2 = interstitial_canonical_potential(potential__, std::complex<double>(0, -1));
                fft::spfft_input<T>(fft, v2);
                get_fft_result(&veff_pw_(0, 2));

                auto v3 = interstitial_canonical_potential(potential__, std::complex<double>(0, 1));
                fft::spfft_input<T>(fft, v3);
                get_fft_result(&veff_pw_(0, 3));
            }
            case 1: {
                auto v0 = interstitial_canonical_potential(potential__, 1.0);
                fft::spfft_input<T>(fft, v0);
                get_fft_result(&veff_pw_(0, 0));

                auto v1 = interstitial_canonical_potential(potential__, -1.0);
                fft::spfft_input<T>(fft, v1);
                get_fft_result(&veff_pw_(0, 1));
                break;
            }
            default: {
                break;
            }
        }
    }
    if (auto const* d = potential__.lapw_xc_derivatives()) {
        bool second = ctx_.cfg().control().use_second_variation();
        bool spinor = ctx_.num_mag_dims() == 3;
        vtau_pw_ = mdarray<std::complex<T>, 2>({ctx_.gvec().num_gvec(), spinor ? 4 : (second ? 1 : ctx_.num_spins())});
        vtau_pw_.zero();
        for (int s = 0; s < (spinor ? 4 : ctx_.num_spins()); s++) {
            auto v       = d->rho[s].gather_f_pw();
            auto t       = d->tau[s].gather_f_pw();
            int column   = spinor ? s : (second ? 0 : s);
            double scale = second && !spinor ? 1.0 / ctx_.num_spins() : 1.0;
            for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
                // These Fourier covectors already exclude the MT quadrature points.
                if (!spinor || s == 0) {
                    veff_pw_(ig, spinor ? 0 : column) += static_cast<std::complex<T>>(scale * v[ig]);
                }
                vtau_pw_(ig, column) += static_cast<std::complex<T>>(scale * t[ig]);
            }
        }
    }
}

template <typename T>
std::complex<T>
Hamiltonian0<T>::hmt_component(int ia__, int i__, int j__, int s__, bool spin_orbit_only__) const
{
    if (ctx_.valence_relativity() == relativity_t::dirac) {
        RTE_THROW("Dirac LAPW atom matrices do not use scalar spin blocks or second-variation SOC");
    }
    auto const& h = hmt_[ia__];
    if (ctx_.num_mag_dims() != 3) {
        return spin_orbit_only__ ? std::complex<T>(0) : h(i__, j__, s__);
    }
    std::complex<T> result{0};
    if (!spin_orbit_only__ && s__ < 2) {
        result = ctx_.cfg().control().use_second_variation()
                         ? h(i__, j__, s__)
                         : T(0.5) * (h(i__, j__, 0) + T(1 - 2 * s__) * h(i__, j__, 1));
    } else if (!spin_orbit_only__) {
        result = s__ == 2 ? T(0.5) * (h(i__, j__, 2) + h(i__, j__, 3))
                          : std::complex<T>(0, -0.5) * (h(i__, j__, 3) - h(i__, j__, 2));
    }
    if (ctx_.so_correction() && s__ != 0) {
        auto const& atom = unit_cell_.atom(ia__);
        auto const& a    = atom.type().indexb(i__);
        auto const& b    = atom.type().indexb(j__);
        result += static_cast<std::complex<T>>(lapw_so_angular(atom.type(), i__, j__, s__)) *
                  static_cast<T>(atom.symmetry_class().so_radial_integral(a.am.l(), a.order, b.order));
    }
    return result;
}

template <typename T>
std::vector<lapw_atom_basis_response_t>
Hamiltonian0<T>::lapw_radial_basis_response(std::vector<radial_basis_response_t> const& response__,
                                            bool spin_orbit_only__, bool physical_overlap__) const
{
    auto const& uc = unit_cell_;
    bool spinor    = ctx_.num_mag_dims() == 3;
    bool second    = ctx_.cfg().control().use_second_variation();
    auto rel       = ctx_.valence_relativity();
    bool dirac     = rel == relativity_t::dirac;
    int invalid    = !std::is_same_v<T, double> || !ctx_.full_potential() || (ctx_.so_correction() && !spinor) ||
                  ctx_.hubbard_correction() ||
                  (dirac ? (second || ctx_.so_correction() || spin_orbit_only__ || physical_overlap__)
                         : (spinor ? !second : (ctx_.num_spins() == 2 && second))) ||
                  response__.size() != uc.num_atom_symmetry_classes();
    int requested = int(spin_orbit_only__) + 2 * int(physical_overlap__);
    int mode      = requested;
    ctx_.comm().bcast(&mode, 1, 0);
    invalid |= mode != requested;
    if (!invalid) {
        for (int ic = 0; ic < uc.num_atom_symmetry_classes(); ic++) {
            auto const& type = uc.atom_symmetry_class(ic).atom_type();
            auto const& d    = response__[ic];
            invalid |= d.radial_functions.size(0) != type.num_mt_points() ||
                       d.radial_functions.size(1) != type.mt_radial_basis_size() ||
                       d.radial_functions.size(2) != (dirac ? 4 : 2) ||
                       d.surface_derivatives.size(0) != (dirac ? 6 : 3) ||
                       d.surface_derivatives.size(1) != type.mt_radial_basis_size() - type.num_lo_descriptors() ||
                       !d.radial_functions.on_host() || !d.surface_derivatives.on_host();
            if (d.radial_functions.on_host()) {
                for (size_t i = 0; i < d.radial_functions.size(); i++) {
                    invalid |= !std::isfinite(d.radial_functions[i]);
                }
            }
            if (d.surface_derivatives.on_host()) {
                for (size_t i = 0; i < d.surface_derivatives.size(); i++) {
                    invalid |= !std::isfinite(d.surface_derivatives[i]);
                }
            }
            if (ctx_.so_correction()) {
                int order       = type.indexr().max_order();
                auto const& soc = d.spin_orbit_integrals;
                invalid |= soc.size(0) != type.indexr().lmax() + 1 || soc.size(1) != order || soc.size(2) != order ||
                           !soc.on_host();
                if (soc.on_host()) {
                    for (size_t i = 0; i < soc.size(); i++) {
                        invalid |= !std::isfinite(soc[i]);
                    }
                }
            }
        }
    }
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW radial matrices require compatible fp64 states and finite directions");
    }
    std::vector<lapw_atom_basis_response_t> matrices;
    if (dirac) {
        auto local = dirac_atom_matrices(&response__);
        for (auto& h : local) {
            matrices.push_back({std::move(h), {}, {}});
        }
        return matrices;
    }
    bool simple       = ctx_.cfg().settings().simple_lapw_ri();
    double alpha_half = rel == relativity_t::none ? 0 : 0.5 / (speed_of_light * speed_of_light);
    int ns = spinor ? 4 : ctx_.num_spins(), lmmax = sf::lmmax(ctx_.lmax_pot());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        auto const& atom      = uc.atom(ia);
        auto const& type      = atom.type();
        auto const& cls       = atom.symmetry_class();
        auto const& grid      = type.radial_grid();
        auto const& spherical = cls.spherical_potential();
        auto const& drf       = response__[cls.id()].radial_functions;
        int n = type.mt_basis_size(), nrf = type.indexr().size();
        lapw_atom_basis_response_t matrix{mdarray<std::complex<double>, 3>({n, n, ns}), mdarray<double, 2>({n, n}),
                                          mdarray<double, 2>({n, n})};
        matrix.h.zero();
        matrix.o.zero();
        matrix.overlap.zero();
        if (ctx_.so_correction()) {
            auto const& soc = response__[cls.id()].spin_orbit_integrals;
            for (int j = 0; j < n; j++) {
                auto const& b = type.indexb(j);
                for (int i = 0; i < n; i++) {
                    auto const& a = type.indexb(i);
                    for (int s = 1; s < ns; s++) {
                        matrix.h(i, j, s) = lapw_so_angular(type, i, j, s) * soc(a.am.l(), a.order, b.order);
                    }
                }
            }
        }
        if (spin_orbit_only__) {
            matrices.push_back(std::move(matrix));
            continue;
        }
        std::vector<Spline<double>> radial, tangent;
        for (int i = 0; i < nrf; i++) {
            radial.emplace_back(grid);
            tangent.emplace_back(grid);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                radial.back()(ir)  = cls.radial_function(ir, i);
                tangent.back()(ir) = drf(ir, i, 0);
            }
            radial.back().interpolate();
            tangent.back().interpolate();
        }
        mdarray<double, 4> dh({lmmax, nrf, nrf, ns});
        mdarray<double, 2> ds({nrf, nrf});
        dh.zero();
        ds.zero();
        Spline<double> product(grid), integral(grid), field(grid);
        auto integrate_field = [&](int i, int j) {
            if (simple) {
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    integral(ir) = product(ir) * field(ir) * grid[ir] * grid[ir];
                }
                return integral.interpolate().integrate(0);
            }
            field.interpolate();
            return inner(tangent[j], radial[i] * field, 2) + inner(radial[j], tangent[i] * field, 2);
        };
        for (int j = 0; j < nrf; j++) {
            for (int i = 0; i <= j; i++) {
                int l1 = type.indexr(i).am.l(), l2 = type.indexr(j).am.l();
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    product(ir) = tangent[i](ir) * radial[j](ir) + radial[i](ir) * tangent[j](ir);
                }
                if (l1 == l2 && i != j &&
                    (i >= nrf - type.num_lo_descriptors() || j >= nrf - type.num_lo_descriptors())) {
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        integral(ir) = product(ir) * (simple ? grid[ir] * grid[ir] : 1);
                    }
                    ds(i, j) = ds(j, i) = integral.interpolate().integrate(simple ? 0 : 2);
                }
                if (rel == relativity_t::iora && !physical_overlap__ && l1 == l2) {
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        double minv = 1 / (1 - alpha_half * spherical[ir]);
                        integral(ir) =
                                0.5 * alpha_half * minv * minv *
                                (drf(ir, i, 1) * cls.radial_function_derivative(ir, j) +
                                 cls.radial_function_derivative(ir, i) * drf(ir, j, 1) + l1 * (l1 + 1) * product(ir));
                    }
                    ds(i, j) += integral.interpolate().integrate(0);
                    ds(j, i) = ds(i, j);
                }
                for (int lm = 0; lm < lmmax; lm++) {
                    double value{0};
                    if (lm == 0 && l1 == l2) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            double minv = 1 / (1 - alpha_half * spherical[ir]);
                            integral(ir) =
                                    0.5 * minv *
                                            (drf(ir, i, 1) * cls.radial_function_derivative(ir, j) +
                                             cls.radial_function_derivative(ir, i) * drf(ir, j, 1)) +
                                    product(ir) * (0.5 * minv * l1 * (l1 + 1) + spherical[ir] * grid[ir] * grid[ir]);
                        }
                        value = integral.interpolate().integrate(0) / y00;
                    } else if (lm != 0) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            field(ir) = potential_->effective_potential().mt()[ia](lm, ir);
                        }
                        value = integrate_field(i, j);
                    }
                    if (spinor) {
                        dh(lm, i, j, 0) = dh(lm, j, i, 0) = value;
                        for (int s = 1; s < ns; s++) {
                            for (int ir = 0; ir < grid.num_points(); ir++) {
                                field(ir) = potential_->effective_magnetic_field(s - 1).mt()[ia](lm, ir);
                            }
                            dh(lm, i, j, s) = dh(lm, j, i, s) = integrate_field(i, j);
                        }
                    } else {
                        double magnetic{0};
                        if (ns == 2) {
                            for (int ir = 0; ir < grid.num_points(); ir++) {
                                field(ir) = potential_->effective_magnetic_field(0).mt()[ia](lm, ir);
                            }
                            magnetic = integrate_field(i, j);
                        }
                        for (int s = 0; s < ns; s++) {
                            dh(lm, i, j, s) = dh(lm, j, i, s) = value + (1 - 2 * s) * magnetic;
                        }
                    }
                }
            }
        }
        for (int j = 0; j < n; j++) {
            auto const& b = type.indexb(j);
            for (int i = 0; i < n; i++) {
                auto const& a = type.indexb(i);
                if (a.lm == b.lm) {
                    bool aw              = i < type.mt_aw_basis_size() && j < type.mt_aw_basis_size();
                    matrix.overlap(i, j) = aw ? (i == j ? 1 : 0) : cls.o_radial_integral(a.am.l(), a.order, b.order);
                    matrix.o(i, j) = aw && (rel != relativity_t::iora || physical_overlap__) ? 0 : ds(a.idxrf, b.idxrf);
                    if (rel == relativity_t::iora && !physical_overlap__) {
                        matrix.overlap(i, j) += cls.o1_radial_integral(a.idxrf, b.idxrf);
                    }
                }
                for (auto const& g : type.gaunt_coefs().gaunt_vector(a.lm, b.lm)) {
                    if (g.lm3 < lmmax) {
                        for (int s = 0; s < ns; s++) {
                            matrix.h(i, j, s) += g.coef * dh(g.lm3, a.idxrf, b.idxrf, s);
                        }
                    }
                }
            }
        }
        if (auto const* xc = potential_->lapw_xc_derivatives()) {
            LAPW_xc_atom op(atom, ctx_.lmax_rho());
            for (int s = 0; s < ns; s++) {
                auto d = op.matrix_elements(xc->local[ia][0][s], xc->local[ia][1][s], &response__[cls.id()]);
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        matrix.h(i, j, s) += d(i, j);
                    }
                }
            }
        }
        matrices.push_back(std::move(matrix));
    }

    return matrices;
}

template <typename T>
lapw_hamiltonian_response_t
Hamiltonian0<T>::lapw_spin_fields(bool spin_orbit_only__) const
{
    int invalid =
            !std::is_same_v<T, double> || !ctx_.full_potential() || ctx_.num_mag_dims() != 3 ||
            !ctx_.cfg().control().use_second_variation() || ctx_.cfg().parameters().precision_wf() != "fp64" ||
            (ctx_.valence_relativity() != relativity_t::none && ctx_.valence_relativity() != relativity_t::zora) ||
            ctx_.hubbard_correction();
    int mode = spin_orbit_only__;
    ctx_.comm().bcast(&mode, 1, 0);
    invalid |= mode != spin_orbit_only__;
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spin matrices require noncollinear NR/ZORA fp64 second variation");
    }
    auto const* xc = potential_->lapw_xc_derivatives();
    if (xc) {
        potential_->validate_lapw_xc_derivatives(*xc);
    }
    lapw_hamiltonian_response_t result;
    result.context    = &ctx_;
    result.scalar_pw  = mdarray<std::complex<double>, 2>({ctx_.gvec().num_gvec(), 4});
    result.kinetic_pw = mdarray<std::complex<double>, 2>({ctx_.gvec().num_gvec(), 4});
    result.scalar_pw.zero();
    result.kinetic_pw.zero();
    Smooth_periodic_function<double> masked(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    for (int s = 1; s < 4; s++) {
        if (spin_orbit_only__) {
            continue;
        }
        for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
            masked.value(ir) = potential_->component(s).rg().value(ir) * ctx_.theta(ir);
        }
        masked.fft_transform(-1);
        auto magnetic = masked.gather_f_pw();
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
            result.scalar_pw(ig, s) = magnetic[ig];
        }
        if (xc) {
            auto rho = xc->rho[s].gather_f_pw(), tau = xc->tau[s].gather_f_pw();
            for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
                // Discrete XC covectors already contain the interstitial restriction.
                result.scalar_pw(ig, s) += rho[ig];
                result.kinetic_pw(ig, s) = tau[ig];
            }
        }
    }
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int nb = unit_cell_.atom(ia).type().mt_basis_size();
        result.local.emplace_back(std::array<index_range, 3>{nb, nb, 4});
        result.local.back().zero();
        for (int j = 0; j < nb; j++) {
            for (int i = 0; i < nb; i++) {
                for (int s = 1; s < 4; s++) {
                    result.local.back()(i, j, s) =
                            static_cast<std::complex<double>>(hmt_component(ia, i, j, s, spin_orbit_only__));
                }
            }
        }
    }
    return result;
}

template <typename T>
lapw_hamiltonian_response_t
Hamiltonian0<T>::lapw_field_response(Periodic_function<double> const& hartree__,
                                     lapw_xc_field_adjoint_t const& xc__) const
{
    bool spinor = ctx_.num_mag_dims() == 3;
    int lmmax = ctx_.lmmax_pot(), ns = spinor ? 4 : ctx_.num_spins();
    auto rel          = ctx_.valence_relativity();
    bool dirac        = rel == relativity_t::dirac;
    double alpha_half = rel == relativity_t::none ? 0 : 0.5 / (speed_of_light * speed_of_light);
    int invalid       = !std::is_same_v<T, double> || !ctx_.full_potential() || (ctx_.so_correction() && !spinor) ||
                  ctx_.hubbard_correction() || (!spinor && ns == 2 && ctx_.cfg().control().use_second_variation()) ||
                  (dirac && (ctx_.so_correction() || ctx_.cfg().control().use_second_variation())) ||
                  &hartree__.ctx() != &ctx_;
    if (!invalid) {
        invalid |= hartree__.mt().atoms().size() != static_cast<size_t>(unit_cell_.num_atoms());
        for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
            invalid |= !std::isfinite(hartree__.rg().value(ir));
            if (rel == relativity_t::zora || rel == relativity_t::iora) {
                double mass = 1 - alpha_half * potential_->effective_potential().rg().value(ir);
                invalid |= !std::isfinite(mass) || mass <= 0;
            }
        }
        if (alpha_half && !dirac) {
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                auto const& cls = unit_cell_.atom_symmetry_class(ic);
                invalid |= cls.spherical_potential().size() != cls.atom_type().num_mt_points();
                for (double v : cls.spherical_potential()) {
                    double mass = 1 - alpha_half * v;
                    invalid |= !std::isfinite(mass) || mass <= 0;
                }
            }
        }
        if (!invalid) {
            for (auto it : unit_cell_.spl_num_atoms()) {
                auto const& v    = hartree__.mt()[it.i];
                auto const& grid = unit_cell_.atom(it.i).radial_grid();
                invalid |= v.angular_domain_size() != lmmax || v.radial_grid().hash() != grid.hash() ||
                           v.size() != static_cast<size_t>(lmmax * grid.num_points()) || !v.on_host();
                if (v.on_host()) {
                    for (size_t i = 0; i < v.size(); i++) {
                        invalid |= !std::isfinite(v[i]);
                    }
                }
            }
        }
    }
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW field matrices require finite Hartree fields, positive scalar masses and fp64 states");
    }
    potential_->validate_lapw_xc_derivatives(xc__);
    lapw_hamiltonian_response_t result;
    result.context    = &ctx_;
    result.scalar_pw  = mdarray<std::complex<double>, 2>({ctx_.gvec().num_gvec(), dirac ? 4 : ns});
    result.kinetic_pw = mdarray<std::complex<double>, 2>({ctx_.gvec().num_gvec(), dirac ? 4 : ns});
    result.scalar_pw.zero();
    result.kinetic_pw.zero();
    if (rel == relativity_t::iora) {
        result.overlap_pw = mdarray<std::complex<double>, 2>({ctx_.gvec().num_gvec(), ns});
        result.overlap_pw.zero();
    }
    Smooth_periodic_function<double> masked(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        masked.value(ir) = hartree__.rg().value(ir) * ctx_.theta(ir);
    }
    masked.fft_transform(-1);
    auto hartree_pw = masked.gather_f_pw();
    if (dirac) {
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
            result.scalar_pw(ig, 0) = hartree_pw[ig];
        }
    }
    for (int s = 0; s < ns; s++) {
        auto rho = xc__.rho[s].gather_f_pw();
        auto tau = xc__.tau[s].gather_f_pw();
        for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
            // The XC covectors already carry their interstitial restriction.
            if (dirac) {
                for (int component = 0; component < 4; component++) {
                    double factor = dirac_field_component(ns, component, s);
                    result.scalar_pw(ig, component) += factor * rho[ig];
                    result.kinetic_pw(ig, component) += factor * tau[ig];
                }
            } else {
                result.scalar_pw(ig, s)  = ((!spinor || s == 0) ? hartree_pw[ig] : std::complex<double>{0}) + rho[ig];
                result.kinetic_pw(ig, s) = tau[ig];
            }
        }
    }
    if (dirac) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& atom = unit_cell_.atom(ia);
            auto const& grid = atom.radial_grid();
            int nr = grid.num_points(), nb = atom.type().mt_basis_size();
            int owner = unit_cell_.spl_num_atoms().location(atom_index_t::global(ia)).ib;
            mdarray<double, 2> values({lmmax, nr});
            if (ctx_.comm().rank() == owner) {
                std::copy_n(hartree__.mt()[ia].at(memory_t::host), values.size(), values.at(memory_t::host));
            }
            ctx_.comm().bcast(values.at(memory_t::host), values.size(), owner);
            auto weights = lapw_radial_product_weights(grid);
            int nfield   = sf::lmmax(std::max(ctx_.lmax_pot(), ctx_.lmax_rho()));
            LAPW_xc_atom op(atom, sf::lmax(nfield));
            result.local.emplace_back(std::array<index_range, 3>{nb, nb, 1});
            result.local.back().zero();
            for (int component = 0; component < 4; component++) {
                Flm rho(nfield, grid), tau(nfield, grid);
                rho.zero();
                tau.zero();
                for (int ir = 0; ir < nr; ir++) {
                    if (component == 0) {
                        for (int lm = 0; lm < lmmax; lm++) {
                            rho(lm, ir) = weights[ir] * values(lm, ir);
                        }
                    }
                    for (int s = 0; s < ns; s++) {
                        double factor = dirac_field_component(ns, component, s);
                        for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                            rho(lm, ir) += factor * xc__.local[ia][0][s](lm, ir);
                            tau(lm, ir) += factor * xc__.local[ia][1][s](lm, ir);
                        }
                    }
                }
                auto matrix = op.matrix_elements(rho, tau, nullptr, component);
                for (size_t i = 0; i < matrix.size(); i++) {
                    result.local.back()[i] += matrix[i];
                }
            }
        }
        return result;
    }
    if (rel == relativity_t::zora || rel == relativity_t::iora) {
        // Discrete XC is a separate operator. Only the raw electrostatic field changes the bare mass.
        for (int part = 0; part < (rel == relativity_t::iora ? 2 : 1); part++) {
            for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                double minv      = 1 / (1 - alpha_half * potential_->effective_potential().rg().value(ir));
                masked.value(ir) = ctx_.theta(ir) * alpha_half * hartree__.rg().value(ir) * minv * minv;
                if (part) {
                    masked.value(ir) *= 2 * alpha_half * minv;
                }
            }
            masked.fft_transform(-1);
            auto mass = masked.gather_f_pw();
            for (int s = 0; s < (spinor ? 1 : ns); s++) {
                for (int ig = 0; ig < ctx_.gvec().num_gvec(); ig++) {
                    if (part) {
                        result.overlap_pw(ig, s) = mass[ig];
                    } else {
                        result.kinetic_pw(ig, s) += mass[ig];
                    }
                }
            }
        }
    }
    bool simple  = ctx_.cfg().settings().simple_lapw_ri();
    auto l_by_lm = sf::l_by_lm(ctx_.lmax_pot());
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto const& atom = unit_cell_.atom(ia);
        auto const& type = atom.type();
        auto const& grid = type.radial_grid();
        auto const& cls  = atom.symmetry_class();
        int nr = grid.num_points(), nrf = type.indexr().size(), nb = type.mt_basis_size();
        int owner = unit_cell_.spl_num_atoms().location(atom_index_t::global(ia)).ib;
        mdarray<double, 2> values({lmmax, nr});
        if (ctx_.comm().rank() == owner) {
            std::copy_n(hartree__.mt()[ia].at(memory_t::host), values.size(), values.at(memory_t::host));
        }
        ctx_.comm().bcast(values.at(memory_t::host), values.size(), owner);
        std::vector<Spline<double>> radial;
        for (int i = 0; i < nrf; i++) {
            radial.emplace_back(grid);
            for (int ir = 0; ir < nr; ir++) {
                radial.back()(ir) = cls.radial_function(ir, i);
            }
            radial.back().interpolate();
        }
        mdarray<double, 3> integrals({lmmax, nrf, nrf});
        integrals.zero();
        mdarray<double, 2> overlap;
        if (rel == relativity_t::iora) {
            overlap = mdarray<double, 2>({nrf, nrf});
            overlap.zero();
        }
        Spline<double> field(grid), product(grid);
        for (int lm = 0; lm < lmmax; lm++) {
            for (int ir = 0; ir < nr; ir++) {
                field(ir) = values(lm, ir);
            }
            field.interpolate();
            for (int j = 0; j < nrf; j++) {
                for (int i = 0; i <= j; i++) {
                    if ((l_by_lm[lm] + type.indexr(i).am.l() + type.indexr(j).am.l()) % 2) {
                        continue;
                    }
                    double value;
                    if (lm == 0 || simple) {
                        // The spherical Hamiltonian always uses a sampled-product spline.
                        for (int ir = 0; ir < nr; ir++) {
                            product(ir) = field(ir) * radial[i](ir) * radial[j](ir) * grid[ir] * grid[ir];
                            if (lm == 0 && alpha_half && type.indexr(i).am.l() == type.indexr(j).am.l()) {
                                int l       = type.indexr(i).am.l();
                                double minv = 1 / (1 - alpha_half * cls.spherical_potential()[ir]);
                                double kinetic =
                                        cls.radial_function_derivative(ir, i) * cls.radial_function_derivative(ir, j) +
                                        l * (l + 1) * radial[i](ir) * radial[j](ir);
                                product(ir) += 0.5 * alpha_half * minv * minv * field(ir) * kinetic;
                            }
                        }
                        value = product.interpolate().integrate(0);
                    } else {
                        value = inner(radial[j], radial[i] * field, 2);
                    }
                    integrals(lm, i, j) = integrals(lm, j, i) = value;
                    if (lm == 0 && rel == relativity_t::iora && type.indexr(i).am.l() == type.indexr(j).am.l()) {
                        int l = type.indexr(i).am.l();
                        for (int ir = 0; ir < nr; ir++) {
                            double minv = 1 / (1 - alpha_half * cls.spherical_potential()[ir]);
                            product(ir) =
                                    alpha_half * alpha_half * y00 * field(ir) * minv * minv * minv *
                                    (cls.radial_function_derivative(ir, i) * cls.radial_function_derivative(ir, j) +
                                     l * (l + 1) * radial[i](ir) * radial[j](ir));
                        }
                        overlap(i, j) = overlap(j, i) = product.interpolate().integrate(0);
                    }
                }
            }
        }
        result.local.emplace_back(std::array<index_range, 3>{nb, nb, ns});
        auto& h = result.local.back();
        h.zero();
        if (rel == relativity_t::iora) {
            result.overlap_local.emplace_back(std::array<index_range, 3>{nb, nb, ns});
            result.overlap_local.back().zero();
        }
        for (int j = 0; j < nb; j++) {
            auto const& b = type.indexb(j);
            for (int i = 0; i < nb; i++) {
                auto const& a = type.indexb(i);
                if (rel == relativity_t::iora && a.lm == b.lm) {
                    for (int s = 0; s < (spinor ? 1 : ns); s++) {
                        result.overlap_local.back()(i, j, s) = overlap(a.idxrf, b.idxrf);
                    }
                }
                for (auto const& g : type.gaunt_coefs().gaunt_vector(a.lm, b.lm)) {
                    if (g.lm3 < lmmax) {
                        for (int s = 0; s < (spinor ? 1 : ns); s++) {
                            h(i, j, s) += g.coef * integrals(g.lm3, a.idxrf, b.idxrf);
                        }
                    }
                }
            }
        }
        LAPW_xc_atom op(atom, ctx_.lmax_rho());
        for (int s = 0; s < ns; s++) {
            auto xc = op.matrix_elements(xc__.local[ia][0][s], xc__.local[ia][1][s]);
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < nb; i++) {
                    h(i, j, s) += xc(i, j);
                }
            }
        }
    }
    return result;
}

template <typename T>
lapw_interstitial_geometry_response_t
Hamiltonian0<T>::lapw_interstitial_geometry_response(xc_geometry_direction_t const& direction__) const
{
    auto rel    = ctx_.valence_relativity();
    int invalid = !std::is_same_v<T, double> || !ctx_.full_potential() ||
                  (rel != relativity_t::zora && rel != relativity_t::iora) ||
                  direction__.atoms.size() != unit_cell_.num_atoms();
    lapw_interstitial_geometry_response_t result;
    result.context = &ctx_;
    result.direction.assign(direction__.lattice.begin(), direction__.lattice.end());
    for (auto const& atom : direction__.atoms) {
        result.direction.insert(result.direction.end(), atom.begin(), atom.end());
    }
    double alpha_half = 0.5 / (speed_of_light * speed_of_light);
    for (double value : result.direction) {
        invalid |= !std::isfinite(value);
    }
    for (double value : potential_->effective_potential().rg().values()) {
        double mass = 1 - alpha_half * value;
        invalid |= !std::isfinite(mass) || mass <= 0;
    }
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW mass geometry response requires finite replicated directions and positive ZORA/IORA masses");
    }
    auto root = result.direction;
    ctx_.comm().bcast(root.data(), root.size(), 0);
    invalid = root != result.direction;
    ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW mass geometry directions differ between context ranks");
    }
    r3::matrix<double> lattice;
    mdarray<double, 2> atoms({3, unit_cell_.num_atoms()});
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            lattice(x, y) = direction__.lattice[x + 3 * y];
        }
    }
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            atoms(x, ia) = direction__.atoms[ia][x];
        }
    }
    auto theta = step_function_geometry_response(unit_cell_, ctx_.gvec(), atoms,
                                                 dot(lattice, unit_cell_.inverse_lattice_vectors()));
    Smooth_periodic_function<double> boundary(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
        boundary.f_pw_local(ig) = theta[ctx_.gvec().offset() + ig];
    }
    boundary.fft_transform(1);
    Smooth_periodic_function<double> masked(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    for (int part = 0; part < (rel == relativity_t::iora ? 2 : 1); part++) {
        for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
            double minv      = 1 / (1 - alpha_half * potential_->effective_potential().rg().value(ir));
            masked.value(ir) = boundary.value(ir) * minv * (part ? alpha_half * minv : 1);
        }
        masked.fft_transform(-1);
        (part ? result.overlap_pw : result.kinetic_pw) = masked.gather_f_pw();
    }
    return result;
}

template class Hamiltonian0<double>;
#ifdef SIRIUS_USE_FP32
template class Hamiltonian0<float>;
#endif

} // namespace sirius
