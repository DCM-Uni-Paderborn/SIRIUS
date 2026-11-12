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

/// Generate radial Hamiltonian and effective magnetic field integrals
/** Hamiltonian operator has the following representation inside muffin-tins:
 *  \f[
 *      \hat H = -\frac{1}{2}\nabla^2 + \sum_{\ell m} V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) =
 *        \underbrace{-\frac{1}{2} \nabla^2+V_{00}(r)R_{00}}_{H_{s}(r)} +\sum_{\ell=1} \sum_{m=-\ell}^{\ell}
 *         V_{\ell m}(r) R_{\ell m}(\hat {\bf r}) = \sum_{\ell m} \widetilde V_{\ell m}(r) R_{\ell m}(\hat {\bf r})
 *  \f]
 *  where
 *  \f[
 *      \widetilde V_{\ell m}(r) = \left\{ \begin{array}{ll}
 *        \frac{H_{s}(r)}{R_{00}} & \ell = 0 \\
 *        V_{\ell m}(r) & \ell > 0 \end{array} \right.
 *  \f]
 */
auto
generate_h_L(Atom_type const& type__, lapw_radial_basis_t const& rb__,
             std::vector<Spheric_function_set<double, atom_index_t>*> const& vns__, int ia__)
{
    PROFILE("sirius::generate_h_L");

    int lmax_pot     = type__.parameters().lmax_pot();
    int lmmax        = sf::lmmax(lmax_pot);
    int nmtp         = type__.num_mt_points();
    int nrf          = type__.indexr().size();
    int num_mag_dims = type__.parameters().num_mag_dims();

    mdarray<double, 4> h_L({lmmax, num_mag_dims + 1, nrf, nrf});
    h_L.zero();

    std::array<int, 3> idx_map({1, 0, 0});
    if (num_mag_dims == 3) {
        idx_map = std::array<int, 3>({3, 1, 2});
    }

    auto l_by_lm = sf::l_by_lm(lmax_pot);

    bool simple = type__.parameters().cfg().settings().simple_lapw_ri();
    std::vector<Spline<double>> radial;
    if (!simple) {
        for (int i = 0; i < nrf; i++) {
            radial.emplace_back(type__.radial_grid());
            for (int ir = 0; ir < nmtp; ir++) {
                radial.back()(ir) = rb__.radial_function(ir, i);
            }
            radial.back().interpolate();
        }
    }

#pragma omp parallel for
    for (int lm = 0; lm < lmmax; lm++) {
        int l = l_by_lm[lm];
        for (int component = 0; component <= num_mag_dims; component++) {
            int field_index = component ? idx_map[component - 1] : 0;
            Spline<double> field(type__.radial_grid()), integrand(type__.radial_grid());
            if (!simple) {
                for (int ir = 0; ir < nmtp; ir++) {
                    field(ir) = (*vns__[field_index])[ia__](lm, ir);
                }
                field.interpolate();
            }
            for (int i2 = 0; i2 < nrf; i2++) {
                int l2 = type__.indexr(i2).am.l();
                for (int i1 = 0; i1 <= i2; i1++) {
                    int l1 = type__.indexr(i1).am.l();
                    if ((l + l1 + l2) % 2 != 0) {
                        continue;
                    }
                    double value;
                    if (!component && !lm) {
                        value = rb__.h_spherical_integral(i1, i2);
                    } else if (simple) {
                        for (int ir = 0; ir < nmtp; ir++) {
                            integrand(ir) = (*vns__[field_index])[ia__](lm, ir) * rb__.radial_function(ir, i1) *
                                            rb__.radial_function(ir, i2) * std::pow(type__.radial_grid(ir), 2);
                        }
                        value = integrand.interpolate().integrate(0);
                    } else {
                        value = inner(radial[i1], radial[i2] * field, 2);
                    }
                    h_L(lm, component, i1, i2) = h_L(lm, component, i2, i1) = value;
                }
            }
        }
    }
    return h_L;
}

/** Compute the following kinds of sums for different components of the Hamiltonian:
 *  \f[
 *      \sum_{L_3} \langle Y_{L_1} u_{\ell_1 \nu_1} | R_{L_3} h_{L_3} | Y_{L_2} u_{\ell_2 \nu_2} \rangle =
 *      \sum_{L_3} \langle u_{\ell_1 \nu_1} | h_{L_3} | u_{\ell_2 \nu_2} \rangle
 *                 \langle Y_{L_1} | R_{L_3} | Y_{L_2} \rangle
 *  \f]
 *
 *  For the non-collinear case, up-dn block corresponds to Bx - i By and dn-up block corresponds to Bx + i By
 */
template <int N>
auto
radial_integrals_sum_L3(std::array<std::complex<double>, N> const& coefficients__, double const* h_L__, int ld__,
                        std::vector<gaunt_L3<std::complex<double>>> const& gnt__)
{
    static_assert(N == 1 || N == 2 || N == 4, "wrong size of coefficients array");
    std::complex<double> result{0};

    for (auto const& g : gnt__) {
        std::complex<double> z{0};

        for (int j = 0; j < N; j++) {
            z += coefficients__[j] * h_L__[g.lm3 + j * ld__];
        }
        result += z * g.coef;
    }
    return result;
}

//template <typename T>
//Hamiltonian0<T>::Hamiltonian0(Potential& potential__)
//    : ctx_(potential__.ctx())
//    , potential_(&potential__)
//    , unit_cell_(potential__.ctx().unit_cell())
//{
//    PROFILE("sirius::Hamiltonian0");
//
//    bool precompute_lapw__ = false;
//    bool update_lapw_rf__ = false;
//
//    local_op_ =
//            std::make_unique<Local_operator<T>>(ctx_, ctx_.spfft_coarse<T>(), ctx_.gvec_coarse_fft_sptr(), potential__);
//
//    if (!ctx_.full_potential()) {
//        d_op_ = std::make_unique<D_operator<T>>(potential__);
//        q_op_ = std::make_unique<Q_operator<T>>(ctx_);
//    }
//    if (ctx_.full_potential()) {
//        if (precompute_lapw__) {
//            if (ctx_.cfg().iterative_solver().type() == "exact") {
//                this->generate_pw_coefs(potential__);
//            }
//            potential_->update_atomic_potential();
//            if (update_lapw_rf__) {
//                ctx_.unit_cell().generate_radial_functions(ctx_.out());
//            }
//            ctx_.unit_cell().generate_radial_integrals();
//        }
//        auto pu = ctx_.processing_unit();
//
//        hmt_ = std::vector<mdarray<std::complex<T>, 3>>(ctx_.unit_cell().num_atoms());
//        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
//            auto& atom = ctx_.unit_cell().atom(ia);
//            auto& type = atom.type();
//
//            int nmt  = type.mt_basis_size();
//            hmt_[ia] = mdarray<std::complex<T>, 3>({nmt, nmt, ctx_.num_mag_dims() + 1}, mdarray_label("hmt"));
//        }
//
//        struct mt_constraint_t
//        {
//            int l;
//            std::vector<std::vector<std::vector<double>>> matrix;
//        };
//
//        /* each atom might have several constraints */
//        std::vector<std::vector<mt_constraint_t>> mt_constraints;
//        if (ctx_.hubbard_constrained_calculation() && ctx_.num_mag_dims() == 1 &&
//            !ctx_.cfg().control().use_second_variation() &&
//            ctx_.num_constraints_applied() < ctx_.cfg().hubbard().constraint().maxiter()) {
//
//            mt_constraints = std::vector<std::vector<mt_constraint_t>>(ctx_.unit_cell().num_atoms());
//
//            for (int i = 0; i < ctx_.cfg().hubbard().constraint().local().size(); i++) {
//                auto const& constraint = ctx_.cfg().hubbard().constraint().local(i);
//                int ia                 = constraint.atom_index();
//                int l                  = constraint.l();
//                if (ia < 0 || ia >= ctx_.unit_cell().num_atoms()) {
//                    RTE_THROW("wrong atom index in local Hubbard constraint");
//                }
//                if (l < 0) {
//                    RTE_THROW("wrong angular momentum in local Hubbard constraint");
//                }
//
//                int mmax = 2 * l + 1;
//
//                auto matrix = constraint.occupancy();
//
//                for (auto const& spin_block : matrix) {
//                    if (static_cast<int>(spin_block.size()) != mmax) {
//                        RTE_THROW("wrong number of rows in local Hubbard constraint");
//                    }
//                    for (auto const& row : spin_block) {
//                        if (static_cast<int>(row.size()) != mmax) {
//                            RTE_THROW("wrong number of columns in local Hubbard constraint");
//                        }
//                    }
//                }
//                mt_constraints[ia].push_back({l, std::move(matrix)});
//            }
//            ctx_.num_constraints_applied(1);
//        }
//
//        #pragma omp parallel for
//        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
//            auto& atom = ctx_.unit_cell().atom(ia);
//            auto& type = atom.type();
//
//            int nmt = type.mt_basis_size();
//
//            /* compute muffin-tin Hamiltonian */
//            for (int j2 = 0; j2 < nmt; j2++) {
//                int lm2    = type.indexb(j2).lm;
//                int idxrf2 = type.indexb(j2).idxrf;
//                for (int j1 = 0; j1 < nmt; j1++) {
//                    int lm1    = type.indexb(j1).lm;
//                    int idxrf1 = type.indexb(j1).idxrf;
//                    switch (ctx_.num_mag_dims()) {
//                        case 3: {
//                            // spin-block index is consistent with non-local pseudopotential operator
//                            // 0: V - Bz
//                            // 1: V + Bz
//                            // 2: Bx - i By
//                            // 3: Bx + i By
//
//                            // Bx - i By
//                            hmt_[ia](j1, j2, 2) = atom.radial_integrals_sum_L3<4>(
//                                    {0, 0, 1, -1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                            // Bx + i By
//                            hmt_[ia](j1, j2, 3) = atom.radial_integrals_sum_L3<4>(
//                                    {0, 0, 1, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                        }
//                        case 1: {
//                            if (ctx_.cfg().control().use_second_variation()) {
//                                hmt_[ia](j1, j2, 0) = atom.radial_integrals_sum_L3<2>(
//                                        {1, 0}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                                hmt_[ia](j1, j2, 1) = atom.radial_integrals_sum_L3<2>(
//                                        {0, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                            } else {
//                                hmt_[ia](j1, j2, 0) = atom.radial_integrals_sum_L3<2>(
//                                        {1, 1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                                hmt_[ia](j1, j2, 1) = atom.radial_integrals_sum_L3<2>(
//                                        {1, -1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//
//                                if (!mt_constraints.empty()) {
//                                    /* add constraints */
//                                    int l1 = type.indexb(j1).am.l();
//                                    int l2 = type.indexb(j2).am.l();
//                                    for (auto const& constraint : mt_constraints[ia]) {
//                                        if (l1 == constraint.l && l2 == constraint.l) {
//                                            int m1     = type.indexb(j1).m;
//                                            int m2     = type.indexb(j2).m;
//                                            int order1 = type.indexb(j1).order;
//                                            int order2 = type.indexb(j2).order;
//                                            double ori = atom.symmetry_class().o_radial_integral(constraint.l, order1,
//                                                                                                 order2);
//                                            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
//                                                hmt_[ia](j1, j2, ispn) +=
//                                                        constraint.matrix[ispn][constraint.l + m1][constraint.l + m2] *
//                                                        ori;
//                                            }
//                                        }
//                                    }
//                                }
//                            }
//                            break;
//                        }
//                        case 0: {
//                            hmt_[ia](j1, j2, 0) = atom.radial_integrals_sum_L3<1>(
//                                    {1}, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
//                        }
//                    }
//                }
//            }
//        }
//        if (pu == device_t::GPU) {
//            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
//                hmt_[ia].allocate(memory_t::device).copy_to(memory_t::device);
//            }
//        }
//    }
//}

template <typename T>
Hamiltonian0<T>::Hamiltonian0(Potential& potential__, std::shared_ptr<LAPW_radial_basis> lapw_basis__)
    : ctx_(potential__.ctx())
    , potential_(&potential__)
    , lapw_basis_(lapw_basis__)
    , unit_cell_(potential__.ctx().unit_cell())
{
    PROFILE("sirius::Hamiltonian0");

    if (ctx_.full_potential() && !lapw_basis_) {
        lapw_basis_ = potential__.create_lapw_basis();
    }
    if (ctx_.full_potential() && ctx_.valence_relativity() == relativity_t::dirac) {
        int invalid = !std::is_same_v<T, double> || ctx_.cfg().control().use_second_variation() ||
                      ctx_.so_correction() || ctx_.hubbard_correction();
        ctx_.comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("Dirac LAPW matrix assembly requires fp64 full variation without a separate SOC or Hubbard "
                      "operator");
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
    } else {
        if (!lapw_basis_) {
            RTE_THROW("LAPW radial basis is not provided");
        }

        if (ctx_.cfg().iterative_solver().type() == "exact") {
            this->generate_pw_coefs(potential__);
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

        auto vns = potential__.mt_components();
        #pragma omp parallel for
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            auto aidx  = atom_index_t::local(ialoc);
            int ia     = unit_cell_.spl_num_atoms().global_index(aidx);
            auto& atom = ctx_.unit_cell().atom(ia);
            auto& type = atom.type();

            auto const& rb = lapw_basis_->radial_basis(atom_index_t::global(ia));

            auto h_L = generate_h_L(type, rb, vns, ia);

            int nmt = type.mt_basis_size();

            /* compute muffin-tin Hamiltonian */
            for (int j2 = 0; j2 < nmt; j2++) {
                int lm2    = type.indexb(j2).lm;
                int idxrf2 = type.indexb(j2).idxrf;
                for (int j1 = 0; j1 < nmt; j1++) {
                    int lm1             = type.indexb(j1).lm;
                    int idxrf1          = type.indexb(j1).idxrf;
                    auto const* h_L_ptr = h_L.at(memory_t::host, 0, 0, idxrf1, idxrf2);
                    auto const& gnt     = type.gaunt_coefs().gaunt_vector(lm1, lm2);
                    switch (ctx_.num_mag_dims()) {
                        case 3: {
                            // spin-block index is consistent with non-local pseudopotential operator
                            // 0, 1: V, Bz in second variation; V + Bz, V - Bz otherwise
                            // 2: Bx - i By
                            // 3: Bx + i By

                            // Bx - i By
                            hmt_[ia](j1, j2, 2) = radial_integrals_sum_L3<4>(
                                    {0.0, 0.0, 1.0, std::complex<double>{0.0, -1.0}}, h_L_ptr, h_L.ld(), gnt);
                            // Bx + i By
                            hmt_[ia](j1, j2, 3) = radial_integrals_sum_L3<4>(
                                    {0.0, 0.0, 1.0, std::complex<double>{0.0, 1.0}}, h_L_ptr, h_L.ld(), gnt);
                        }
                        case 1: {
                            if (ctx_.cfg().control().use_second_variation()) {
                                hmt_[ia](j1, j2, 0) = radial_integrals_sum_L3<2>({1.0, 0.0}, h_L_ptr, h_L.ld(), gnt);
                                hmt_[ia](j1, j2, 1) = radial_integrals_sum_L3<2>({0.0, 1.0}, h_L_ptr, h_L.ld(), gnt);
                            } else {
                                hmt_[ia](j1, j2, 0) = radial_integrals_sum_L3<2>({1.0, 1.0}, h_L_ptr, h_L.ld(), gnt);
                                hmt_[ia](j1, j2, 1) = radial_integrals_sum_L3<2>({1.0, -1.0}, h_L_ptr, h_L.ld(), gnt);

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
                                            double ori = rb.o_radial_integral(constraint.l, order1, order2);
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
                            hmt_[ia](j1, j2, 0) = radial_integrals_sum_L3<1>({1.0}, h_L_ptr, h_L.ld(), gnt);
                        }
                    }
                }
            }
        }
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int rank = unit_cell_.spl_num_atoms().location(typename atom_index_t::global(ia)).ib;
            unit_cell_.comm().bcast(hmt_[ia].at(memory_t::host), static_cast<int>(hmt_[ia].size()), rank);
        }
        if (auto const* d = potential__.lapw_xc_derivatives()) {
            // Contract after any radial-basis update, using raw sample covectors.
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                LAPW_xc_atom op(unit_cell_.atom(ia), lapw_basis().radial_basis(atom_index_t::global(ia)),
                                ctx_.lmax_rho());
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
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                hmt_[ia].allocate(memory_t::device).copy_to(memory_t::device);
            }
        }
    }
}

template <typename T>
Hamiltonian0<T>::Hamiltonian0(Potential& potential__, bool precompute_lapw__, bool update_lapw_rf__)
    : Hamiltonian0(potential__, !potential__.ctx().full_potential()     ? nullptr
                                : precompute_lapw__ && update_lapw_rf__ ? potential__.create_lapw_basis()
                                                                        : potential__.lapw_basis_sptr())
{
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
        auto const& cls = lapw_basis().radial_basis_for_class(ic);
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
        LAPW_xc_atom op(atom, lapw_basis().radial_basis(atom_index_t::global(ia)), sf::lmax(lmmax));
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
                            value -=
                                    lapw_basis().radial_basis(atom_index_t::global(ia)).spherical_potential()[ir] / y00;
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
        Dirac_lapw_basis basis(unit_cell_, gkvec__, lapw_basis());
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
Hamiltonian0<T>::add_o1mt_to_apw(Atom const& atom__, lapw_radial_basis_t const& rb__, int num_gkvec__,
                                 mdarray<std::complex<T>, 2>& alm__) const
{
    // TODO: optimize for the loop layout using blocks of G-vectors
    auto const& type = atom__.type();
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
                oalm[j] += alm[j1] * rb__.o1_radial_integral(idxrf, idxrf1);
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
        auto ia        = it.i;
        auto const& rb = lapw_basis_->radial_basis(it.i);
        auto& atom     = unit_cell_.atom(ia);
        auto a         = it.li;

        for (int l = 0; l <= atom.type().lmax_apw(); l++) {
            /* number of radial functions for this l */
            int nrf = atom.type().indexr().max_order(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                for (int order2 = 0; order2 < nrf; order2++) {
                    T sori = rb.so_radial_integral(l, order1, order2);

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

    auto& fft = ctx_.spfft<T>();

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
                  static_cast<T>(lapw_basis()
                                         .radial_basis(atom_index_t::global(ia__))
                                         .so_radial_integral(a.am.l(), a.order, b.order));
    }
    return result;
}

template class Hamiltonian0<double>;
#ifdef SIRIUS_USE_FP32
template class Hamiltonian0<float>;
#endif

} // namespace sirius
