/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file force.cpp
 *
 *  \brief Contains implementation of sirius::Force class.
 */

#include "force.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "density/density.hpp"
#include "density/augmentation_operator.hpp"
#include "density/lapw_xc_atom.hpp"
#include "potential/potential.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "beta_projectors/beta_projectors_gradient.hpp"
#include "non_local_functor.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "symmetry/symmetrize_forces.hpp"
#include "lapw/step_function.hpp"
#include <string>
#include <iomanip>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>

namespace sirius {

Force::Force(Simulation_context& ctx__, Density& density__, Potential& potential__, K_point_set& kset__)
    : ctx_(ctx__)
    , density_(density__)
    , potential_(potential__)
    , kset_(kset__)
{
}

double
Force::lapw_radial_basis_derivative(std::vector<std::vector<double>> const& directions__,
                                    std::vector<std::array<std::vector<double>, 2>> const& xc_directions__)
{
    auto& uc    = ctx_.unit_cell();
    int invalid = !ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU ||
                  ctx_.valence_relativity() != relativity_t::none || ctx_.so_correction() || ctx_.num_mag_dims() == 3 ||
                  ctx_.hubbard_correction() ||
                  (uc.num_core_electrons() != 0 && ctx_.core_relativity() != relativity_t::none) ||
                  (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()) ||
                  directions__.size() != uc.num_atom_symmetry_classes() ||
                  (!xc_directions__.empty() && xc_directions__.size() != directions__.size());
    if (!invalid) {
        for (int ic = 0; ic < uc.num_atom_symmetry_classes(); ic++) {
            invalid |= directions__[ic].size() != uc.atom_symmetry_class(ic).atom_type().num_mt_points() ||
                       !std::all_of(directions__[ic].begin(), directions__[ic].end(),
                                    [](double x) { return std::isfinite(x); });
            if (!xc_directions__.empty()) {
                auto const& cls = uc.atom_symmetry_class(ic);
                for (auto const& field : xc_directions__[ic]) {
                    invalid |=
                            !field.empty() &&
                            (field.size() != directions__[ic].size() ||
                             !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); }) ||
                             (cls.atom_type().num_core_electrons() != 0 && cls.spherical_xc_derivatives()[0].empty()));
                }
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW radial basis derivative requires nonrelativistic scalar/collinear valence states and valid "
                  "directions");
    }

    // Differentiate the stored basis without updating its potential or energies.
    Hamiltonian0<double> h0(potential_, false);
    std::vector<radial_basis_response_t> response;
    for (int ic = 0; ic < uc.num_atom_symmetry_classes(); ic++) {
        response.push_back(uc.atom_symmetry_class(ic).radial_basis_potential_response(
                directions__[ic],
                xc_directions__.empty() ? std::array<std::vector<double>, 2>{} : xc_directions__[ic]));
    }
    struct atom_response_t
    {
        mdarray<std::complex<double>, 3> h;
        mdarray<double, 2> o, overlap;
    };
    std::vector<atom_response_t> matrices;
    bool simple = ctx_.cfg().settings().simple_lapw_ri();
    int ns = ctx_.num_spins(), lmmax = sf::lmmax(ctx_.lmax_pot());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        auto const& atom      = uc.atom(ia);
        auto const& type      = atom.type();
        auto const& cls       = atom.symmetry_class();
        auto const& grid      = type.radial_grid();
        auto const& spherical = cls.spherical_potential();
        auto const& drf       = response[cls.id()].radial_functions;
        int n = type.mt_basis_size(), nrf = type.indexr().size();
        atom_response_t matrix{mdarray<std::complex<double>, 3>({n, n, ns}), mdarray<double, 2>({n, n}),
                               mdarray<double, 2>({n, n})};
        matrix.h.zero();
        matrix.o.zero();
        matrix.overlap.zero();
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
                if (l1 == l2 && i != j) {
                    for (int ir = 0; ir < grid.num_points(); ir++) {
                        integral(ir) = product(ir) * (simple ? grid[ir] * grid[ir] : 1);
                    }
                    ds(i, j) = ds(j, i) = integral.interpolate().integrate(simple ? 0 : 2);
                }
                for (int lm = 0; lm < lmmax; lm++) {
                    double value{0};
                    if (lm == 0 && l1 == l2) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            integral(ir) = 0.5 * (drf(ir, i, 1) * cls.radial_function_derivative(ir, j) +
                                                  cls.radial_function_derivative(ir, i) * drf(ir, j, 1)) +
                                           product(ir) * (0.5 * l1 * (l1 + 1) + spherical[ir] * grid[ir] * grid[ir]);
                        }
                        value = integral.interpolate().integrate(0) / y00;
                    } else if (lm != 0) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            field(ir) = potential_.effective_potential().mt()[ia](lm, ir);
                        }
                        value = integrate_field(i, j);
                    }
                    double magnetic{0};
                    if (ns == 2) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            field(ir) = potential_.effective_magnetic_field(0).mt()[ia](lm, ir);
                        }
                        magnetic = integrate_field(i, j);
                    }
                    for (int s = 0; s < ns; s++) {
                        dh(lm, i, j, s) = dh(lm, j, i, s) = value + (1 - 2 * s) * magnetic;
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
                    matrix.o(i, j)       = aw ? 0 : ds(a.idxrf, b.idxrf);
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
        if (auto const* xc = potential_.lapw_xc_derivatives()) {
            LAPW_xc_atom op(atom, ctx_.lmax_rho());
            for (int s = 0; s < ns; s++) {
                auto d = op.matrix_elements(xc->local[ia][0][s], xc->local[ia][1][s], &response[cls.id()]);
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        matrix.h(i, j, s) += d(i, j);
                    }
                }
            }
        }
        matrices.push_back(std::move(matrix));
    }

    double result{0};
    for (auto it : kset_.spl_num_kpoints()) {
        auto& kp = *kset_.get<double>(it.i);
        Matching_coefficients matching(uc, kp.gkvec());
        std::vector<mdarray<std::complex<double>, 2>> alm, dalm;
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            auto const& atom = uc.atom(ia);
            std::array<index_range, 2> shape{kp.gkvec().count(), atom.type().mt_aw_basis_size()};
            alm.emplace_back(shape);
            dalm.emplace_back(shape);
            matching.generate<false>(atom, alm.back());
            matching.generate_potential_response<false>(atom, response[atom.symmetry_class().id()], dalm.back());
        }
        for (int s = 0; s < ns; s++) {
            for (int band = 0; band < ctx_.num_bands(); band++) {
                double weight = kp.weight() * kp.band_occupancy(band, s) / kp.comm().size();
                if (weight == 0) {
                    continue;
                }
                auto all   = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(band));
                int offset = kp.gkvec().num_gvec();
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    auto const& type = uc.atom(ia).type();
                    int n = type.mt_basis_size(), naw = type.mt_aw_basis_size();
                    std::vector<std::complex<double>> c(n), dc(n);
                    for (int i = 0; i < naw; i++) {
                        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                            c[i] += alm[ia](ig, i) * all[kp.gkvec().offset() + ig];
                            dc[i] += dalm[ia](ig, i) * all[kp.gkvec().offset() + ig];
                        }
                    }
                    kp.comm().allreduce(c.data(), naw);
                    kp.comm().allreduce(dc.data(), naw);
                    for (int i = naw; i < n; i++) {
                        c[i] = all[offset + i];
                    }
                    auto const& d = matrices[ia];
                    double energy = kp.band_energy(band, s);
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            result +=
                                    weight * std::real(2.0 * std::conj(dc[i]) *
                                                               (h0.hmt(ia)(i, j, s) - energy * d.overlap(i, j)) * c[j] +
                                                       std::conj(c[i]) * (d.h(i, j, s) - energy * d.o(i, j)) * c[j]);
                        }
                    }
                    offset += n;
                }
            }
        }
    }
    ctx_.comm().allreduce(&result, 1);
    return result;
}

void
Force::calc_forces_dftd3()
{
    forces_dftd3_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_dftd3_.zero();
    if (ctx_.cfg().parameters().dftd3_correction()) {
        auto& f = potential_.dftd3_ctx().forces();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_dftd3_(x, ia) = f(x, ia);
            }
        }
    }
}

void
Force::calc_forces_dftd4()
{
    forces_dftd4_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_dftd4_.zero();
    if (ctx_.cfg().parameters().dftd4_correction()) {
        auto& f = potential_.dftd4_ctx().forces();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_dftd4_(x, ia) = f(x, ia);
            }
        }
    }
}

template <typename T>
void
Force::calc_forces_nonloc_aux()
{
    forces_nonloc_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_nonloc_.zero();

    auto& spl_num_kp = kset_.spl_num_kpoints();

    for (auto it : spl_num_kp) {
        auto* kp = kset_.get<T>(it.i);

        if (ctx_.gamma_point()) {
            add_k_point_contribution<T, T>(*kp, forces_nonloc_);
        } else {
            add_k_point_contribution<T, std::complex<T>>(*kp, forces_nonloc_);
        }
    }

    ctx_.comm().allreduce(&forces_nonloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_nonloc_);
}

mdarray<double, 2> const&
Force::calc_forces_nonloc()
{
    PROFILE("sirius::Force::calc_forces_nonloc");

    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
        this->calc_forces_nonloc_aux<float>();
#endif
    } else {
        this->calc_forces_nonloc_aux<double>();
    }
    return forces_nonloc_;
}

template <typename T, typename F>
void
Force::add_k_point_contribution(K_point<T>& kp__, mdarray<double, 2>& forces__) const
{
    /* if there are no beta projectors then get out there */
    if (ctx_.unit_cell().max_mt_basis_size() == 0) {
        return;
    }

    Beta_projectors_gradient<T> bp_grad(ctx_, kp__.gkvec(), kp__.beta_projectors());
    auto mem = ctx_.processing_unit_memory_t();
    auto mg  = kp__.spinor_wave_functions().memory_guard(mem, wf::copy_to::device);

    mdarray<real_type<F>, 2> f({3, ctx_.unit_cell().num_atoms()});
    f.zero();

    add_k_point_contribution_nonlocal<T, F>(potential_, bp_grad, kp__, f);

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            forces__(x, ia) += f(x, ia);
        }
    }
}

void
Force::compute_dmat(K_point<double>* kp__, la::dmatrix<std::complex<double>>& dm__) const
{
    dm__.zero();

    /* trivial case */
    if (!ctx_.need_sv()) {
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            dm__.set(i, i, std::complex<double>(kp__->band_occupancy(i, 0), 0));
        }
    } else {
        if (ctx_.num_mag_dims() != 3) {
            la::dmatrix<std::complex<double>> ev1(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                                  ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                auto& ev = kp__->sv_eigen_vectors(ispn);
                /* multiply second-variational eigen-vectors with band occupancies */
                for (int j = 0; j < ev.num_cols_local(); j++) {
                    /* up- or dn- band index */
                    int jb = ev.icol(j);
                    for (int i = 0; i < ev.num_rows_local(); i++) {
                        ev1(i, j) = std::conj(ev(i, j)) * kp__->band_occupancy(jb, ispn);
                    }
                }

                la::wrap(la::lib_t::scalapack)
                        .gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                              &la::constant<std::complex<double>>::one(), ev1, 0, 0, ev, 0, 0,
                              &la::constant<std::complex<double>>::one(), dm__, 0, 0);
            }
        } else {
            la::dmatrix<std::complex<double>> ev1(ctx_.num_bands(), ctx_.num_bands(), ctx_.blacs_grid(),
                                                  ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            auto& ev = kp__->sv_eigen_vectors(0);
            /* multiply second-variational eigen-vectors with band occupancies */
            for (int j = 0; j < ev.num_cols_local(); j++) {
                /* band index */
                int jb = ev.icol(j);
                for (int i = 0; i < ev.num_rows_local(); i++) {
                    ev1(i, j) = std::conj(ev(i, j)) * kp__->band_occupancy(jb, 0);
                }
            }
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                int offs = ispn * ctx_.num_fv_states();

                la::wrap(la::lib_t::scalapack)
                        .gemm('N', 'T', ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.num_bands(),
                              &la::constant<std::complex<double>>::one(), ev1, offs, 0, ev, offs, 0,
                              &la::constant<std::complex<double>>::one(), dm__, 0, 0);
            }
        }
    }
}
mdarray<double, 2> const&
Force::calc_forces_total()
{
    return calc_forces_total(true /*add scf correction*/);
}

mdarray<double, 2> const&
Force::calc_forces_total(bool add_scf_corr)
{
    xc_geometry_derivatives_t xc_geometry(ctx_.unit_cell().num_atoms());
    if (ctx_.cb().external_xc_) {
        if (ctx_.full_potential()) {
            RTE_THROW("external LAPW XC basis and boundary derivatives are not connected to forces yet");
        }
        xc_geometry = potential_.external_xc_geometry(density_);
    }
    if (ctx_.meta_gga() && !ctx_.cb().external_xc_) {
        RTE_THROW("meta-GGA geometry derivatives are not connected to forces yet");
    }
    forces_xc_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            forces_xc_(x, ia) = -xc_geometry.atoms[ia][x];
        }
    }
    forces_total_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});

    calc_forces_dftd3();
    calc_forces_dftd4();

    if (ctx_.full_potential()) {
        calc_forces_rho();
        calc_forces_hf();
        calc_forces_ibs();
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_total_(x, ia) = forces_ibs_(x, ia) + forces_hf_(x, ia) + forces_rho_(x, ia);
            }
        }
    } else {
        calc_forces_vloc();
        calc_forces_us();
        calc_forces_nonloc();
        calc_forces_core();
        calc_forces_ewald();
        if (add_scf_corr) {
            calc_forces_scf_corr();
        }
        calc_forces_hubbard();

        forces_total_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                forces_total_(x, ia) = forces_vloc_(x, ia) + forces_us_(x, ia) + forces_nonloc_(x, ia) +
                                       forces_core_(x, ia) + forces_ewald_(x, ia) + forces_hubbard_(x, ia) +
                                       forces_dftd3_(x, ia) + forces_dftd4_(x, ia);
                if (add_scf_corr) {
                    forces_total_(x, ia) += forces_scf_corr_(x, ia);
                }
                // Projector response is already in the nonlocal force through D_ij.
                forces_total_(x, ia) += forces_xc_(x, ia);
            }
        }
    }
    return forces_total_;
}

mdarray<double, 2> const&
Force::calc_forces_ibs()
{
    forces_ibs_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_ibs_.zero();

    bool second = ctx_.cfg().control().use_second_variation();
    if (!second && ctx_.valence_relativity() != relativity_t::none) {
        RTE_THROW("full-variational IBS forces require nonrelativistic valence states");
    }

    mdarray<double, 2> ffac({ctx_.unit_cell().num_atom_types(), ctx_.gvec().num_shells()});
    #pragma omp parallel for
    for (int igs = 0; igs < ctx_.gvec().num_shells(); igs++) {
        for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
            ffac(iat, igs) = unit_step_function_form_factors(ctx_.unit_cell().atom_type(iat).mt_radius(),
                                                             ctx_.gvec().shell_len(igs));
        }
    }

    Hamiltonian0<double> H0(potential_, false);
    for (auto it : kset_.spl_num_kpoints()) {
        auto hk = H0(*kset_.get<double>(it.i));
        for (int s = 0; s < (second ? 1 : ctx_.num_spins()); s++) {
            add_ibs_force(kset_.get<double>(it.i), hk, ffac, forces_ibs_, s);
        }
    }
    ctx_.comm().allreduce(&forces_ibs_(0, 0), (int)forces_ibs_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_ibs_);

    return forces_ibs_;
}

mdarray<double, 2> const&
Force::calc_forces_rho()
{

    forces_rho_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_rho_.zero();
    for (auto it : ctx_.unit_cell().spl_num_atoms()) {
        auto g = gradient(density_.density_mt(it.li));
        for (int x = 0; x < 3; x++) {
            forces_rho_(x, it.i) = inner(potential_.effective_potential_mt(it.li), g[x]);
        }
    }
    ctx_.comm().allreduce(&forces_rho_(0, 0), (int)forces_rho_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_rho_);

    return forces_rho_;
}

mdarray<double, 2> const&
Force::calc_forces_hf()
{
    forces_hf_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_hf_.zero();

    for (auto it : ctx_.unit_cell().spl_num_atoms()) {
        auto g = gradient(potential_.hartree_potential_mt(it.li));
        for (int x = 0; x < 3; x++) {
            forces_hf_(x, it.i) = ctx_.unit_cell().atom(it.i).zn() * g[x](0, 0) * y00;
        }
    }
    ctx_.comm().allreduce(&forces_hf_(0, 0), (int)forces_hf_.size());
    symmetrize_forces(ctx_.unit_cell(), forces_hf_);

    return forces_hf_;
}

mdarray<double, 2> const&
Force::calc_forces_hubbard()
{
    PROFILE("sirius::Force::hubbard_force");
    forces_hubbard_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_hubbard_.zero();

    if (ctx_.hubbard_correction()) {
        /* recompute the hubbard potential */
        //::sirius::generate_potential(density_.occupation_matrix(), potential_.hubbard_potential());

        Q_operator<double> q_op(ctx_);

        for (auto it : kset_.spl_num_kpoints()) {

            auto kp  = kset_.get<double>(it.i);
            auto mg1 = kp->spinor_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg2 =
                    kp->hubbard_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg3 = kp->atomic_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
            auto mg4 = kp->atomic_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);

            if (ctx_.num_mag_dims() == 3) {
                RTE_THROW("Hubbard forces are only implemented for the simple hubbard correction.");
            }
            hubbard_force_add_k_contribution_collinear(*kp, q_op, forces_hubbard_);
        }

        /* global reduction */
        kset_.comm().allreduce(forces_hubbard_.at(memory_t::host), 3 * ctx_.unit_cell().num_atoms());
    }

    symmetrize_forces(ctx_.unit_cell(), forces_hubbard_);
    return forces_hubbard_;
}

mdarray<double, 2> const&
Force::calc_forces_ewald()
{
    PROFILE("sirius::Force::calc_forces_ewald");

    forces_ewald_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_ewald_.zero();

    Unit_cell& unit_cell = ctx_.unit_cell();

    double alpha = ctx_.ewald_lambda();

    double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

    mdarray<std::complex<double>, 1> rho_tmp({ctx_.gvec().count()});
    rho_tmp.zero();
    int igloc0 = ctx_.gvec().skip_g0();
    int ngv    = ctx_.gvec().count();
    #pragma omp parallel for schedule(static)
    for (int igloc = igloc0; igloc < ngv; igloc++) {

        auto ig = ctx_.gvec().global_index(gvec_index_t::local(igloc));

        double g2 = std::pow(ctx_.gvec().gvec_len(gvec_index_t::local(igloc)), 2);

        std::complex<double> rho(0, 0);

        for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
            rho += ctx_.gvec_phase_factor(ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
        }

        rho_tmp[igloc] = prefac * std::conj(rho) * std::exp(-g2 / (4 * alpha)) / g2;
    }

    #pragma omp parallel for
    for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
        for (int igloc = igloc0; igloc < ngv; igloc++) {

            auto ig = ctx_.gvec().global_index(gvec_index_t::local(igloc));

            /* cartesian coordinates for getting cartesian force components */
            auto gvec_cart = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));

            double scalar_part = (rho_tmp[igloc] * ctx_.gvec_phase_factor(ig, ja)).imag() *
                                 static_cast<double>(unit_cell.atom(ja).zn());

            for (int x : {0, 1, 2}) {
                forces_ewald_(x, ja) += scalar_part * gvec_cart[x];
            }
        }
    }

    ctx_.comm().allreduce(&forces_ewald_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    double invpi = 1.0 / pi;

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
            int ja = unit_cell.nearest_neighbour(i, ia).atom_id;

            double d  = unit_cell.nearest_neighbour(i, ia).distance;
            double d2 = d * d;

            /* rc = r_j + T - r_i, but we need the inverse */
            auto t = -1 * unit_cell.nearest_neighbour(i, ia).rc;

            double scalar_part =
                    static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                    (std::erfc(std::sqrt(alpha) * d) / d + 2.0 * std::sqrt(alpha * invpi) * std::exp(-d2 * alpha));

            for (int x : {0, 1, 2}) {
                forces_ewald_(x, ia) += scalar_part * t[x];
            }
        }
    }

    symmetrize_forces(ctx_.unit_cell(), forces_ewald_);

    return forces_ewald_;
}

mdarray<double, 2> const&
Force::calc_forces_us()
{
    PROFILE("sirius::Force::calc_forces_us");

    forces_us_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_us_.zero();

    if (ctx_.cb().external_xc_) {
        potential_.xc_potential().rg().fft_transform(-1);
    }

    //potential_.fft_transform(-1);

    auto const& unit_cell = ctx_.unit_cell();

    double reduce_g_fact = ctx_.gvec().reduced() ? 2.0 : 1.0;

    la::lib_t la{la::lib_t::none};

    memory_pool* mp{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            mp = &get_memory_pool(memory_t::host);
            la = la::lib_t::blas;
            break;
        }
        case device_t::GPU: {
            mp = &get_memory_pool(memory_t::host_pinned);
            la = la::lib_t::spla;
            break;
        }
    }

    /* over atom types */
    for (int iat = 0; iat < unit_cell.num_atom_types(); iat++) {
        auto& atom_type = unit_cell.atom_type(iat);

        if (!ctx_.unit_cell().atom_type(iat).augment()) {
            continue;
        }

        auto& aug_op = ctx_.augmentation_op(iat);

        int nbf = atom_type.mt_basis_size();

        /* get auxiliary density matrix */
        auto dm = density_.density_matrix_aux(atom_type);

        mdarray<double, 2> v_tmp({atom_type.num_atoms(), ctx_.gvec().count() * 2}, *mp);
        mdarray<double, 2> tmp({nbf * (nbf + 1) / 2, atom_type.num_atoms()}, *mp);

        /* over spin components, can be from 1 to 4*/
        for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++) {
            /* over 3 components of the force/G - vectors */
            for (int ivec = 0; ivec < 3; ivec++) {
                /* over local rank G vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                    int ig   = ctx_.gvec().offset() + igloc;
                    auto gvc = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
                    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                        /* here we write in v_tmp  -i * G * exp[ iGRn] Veff(G)
                         * but in formula we have   i * G * exp[-iGRn] Veff*(G)
                         * the differences because we unfold complex array in the real one
                         * and need negative imagine part due to a multiplication law of complex numbers */
                        auto v = potential_.component(ispin).rg().f_pw_local(igloc);
                        if (ctx_.cb().external_xc_) {
                            // Reconstructed XC excludes the Fourier Q augmentation,
                            // just as in the corresponding PAW nonlocal operator.
                            v -= ispin == 0 ? potential_.xc_potential().rg().f_pw_local(igloc)
                                            : potential_.effective_magnetic_field(ispin - 1).rg().f_pw_local(igloc);
                        }
                        auto z                   = std::complex<double>(0, -gvc[ivec]) *
                                                   ctx_.gvec_phase_factor(ig, atom_type.atom_id(ia)) * v;
                        v_tmp(ia, 2 * igloc)     = z.real();
                        v_tmp(ia, 2 * igloc + 1) = z.imag();
                    }
                }

                /* multiply tmp matrices, or sum over G */
                la::wrap(la).gemm('N', 'T', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                  &la::constant<double>::one(), aug_op.q_pw().at(memory_t::host), aug_op.q_pw().ld(),
                                  v_tmp.at(memory_t::host), v_tmp.ld(), &la::constant<double>::zero(),
                                  tmp.at(memory_t::host), tmp.ld());

                #pragma omp parallel for
                for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                    for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                        forces_us_(ivec, atom_type.atom_id(ia)) += ctx_.unit_cell().omega() * reduce_g_fact *
                                                                   dm(i, ia, ispin) * aug_op.sym_weight(i) * tmp(i, ia);
                    }
                }
            }
        }
    }

    ctx_.comm().allreduce(&forces_us_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_us_);

    return forces_us_;
}

mdarray<double, 2> const&
Force::calc_forces_scf_corr()
{
    PROFILE("sirius::Force::calc_forces_scf_corr");

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().ps_rho_->values(q, ctx_.comm());

    forces_scf_corr_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_scf_corr_.zero();

    /* get main arrays */
    auto const& dveff = potential_.dveff();

    auto const& unit_cell = ctx_.unit_cell();

    auto const& gvec = ctx_.gvec();

    int gvec_count  = gvec.count();
    int gvec_offset = gvec.offset();

    double fact = gvec.reduced() ? 2.0 : 1.0;

    int ig0 = ctx_.gvec().skip_g0();

    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        auto const& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (int igloc = ig0; igloc < gvec_count; igloc++) {
            int ig   = gvec_offset + igloc;
            int igsh = ctx_.gvec().shell(ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvec.gvec_cart(gvec_index_t::local(igloc));

            /* scalar part of a force without multipying by G-vector */
            std::complex<double> z =
                    fact * fourpi * ff(igsh, iat) * std::conj(dveff.f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_scf_corr_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_scf_corr_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_scf_corr_);

    return forces_scf_corr_;
}

mdarray<double, 2> const&
Force::calc_forces_core()
{
    PROFILE("sirius::Force::calc_forces_core");

    forces_core_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_core_.zero();
    // External PAW XC includes the AE core in its atom-local geometry derivative,
    // not a pseudo-core density in the smooth XC field. NC NLCC is rejected.
    if (ctx_.cb().external_xc_) {
        return forces_core_;
    }

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().ps_core_->values(q, ctx_.comm());

    /* get main arrays */
    auto& xc_pot = potential_.xc_potential();

    /* transform from real space to reciprocal */
    xc_pot.rg().fft_transform(-1);

    auto const& unit_cell = ctx_.unit_cell();

    auto const& gvecs = ctx_.gvec();

    int gvec_count  = gvecs.count();
    int gvec_offset = gvecs.offset();

    double fact = gvecs.reduced() ? 2.0 : 1.0;

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        auto const& atom = unit_cell.atom(ia);
        if (atom.type().ps_core_charge_density().empty()) {
            continue;
        }
        int iat = atom.type_id();

        for (int igloc = ctx_.gvec().skip_g0(); igloc < gvec_count; igloc++) {
            int ig    = gvec_offset + igloc;
            auto igsh = ctx_.gvec().shell(ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvecs.gvec_cart(gvec_index_t::local(igloc));

            /* scalar part of a force without multipying by G-vector */
            std::complex<double> z = fact * fourpi * ff(igsh, iat) *
                                     std::conj(xc_pot.rg().f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

            /* get force components multiplying by cartesian G-vector */
            for (int x : {0, 1, 2}) {
                forces_core_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }
    ctx_.comm().allreduce(&forces_core_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_core_);

    return forces_core_;
}

void
Force::hubbard_force_add_k_contribution_collinear(K_point<double>& kp__, Q_operator<double>& q_op__,
                                                  mdarray<double, 2>& forceh__)
{
    auto r = ctx_.unit_cell().num_hubbard_wf();

    mdarray<std::complex<double>, 5> dn({r.first, r.first, ctx_.num_spins(), 3, ctx_.unit_cell().num_atoms()});

    potential_.U().compute_occupancies_derivatives(kp__, q_op__, dn);

    #pragma omp parallel for
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        /* compute the derivative of the occupancies numbers */
        for (int dir = 0; dir < 3; dir++) {
            std::complex<double> d{0.0};
            for (int at_lvl = 0; at_lvl < static_cast<int>(potential_.hubbard_potential().num_atomic_levels());
                 at_lvl++) {
                const int ia1    = potential_.hubbard_potential().atomic_orbital(at_lvl).first;
                const auto& atom = ctx_.unit_cell().atom(ia1);
                const int lo     = potential_.hubbard_potential().atomic_orbital(at_lvl).second;
                if (atom.type().lo_descriptor_hub(lo).use_for_calculation()) {
                    int const lmax_at = 2 * atom.type().lo_descriptor_hub(lo).l() + 1;
                    const int offset  = potential_.hubbard_potential().offset(at_lvl);
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                d += potential_.hubbard_potential().local(at_lvl)(m1, m2, ispn) *
                                     dn(offset + m2, offset + m1, ispn, dir, ia);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
                auto nl = ctx_.cfg().hubbard().nonlocal(i);
                int ia1 = nl.atom_pair()[0];
                int ja  = nl.atom_pair()[1];
                // consider the links that involve atom i
                int il  = nl.l()[0];
                int jl  = nl.l()[1];
                int in  = nl.n()[0];
                int jn  = nl.n()[1];
                auto Tr = nl.T();

                auto z1           = std::exp(std::complex<double>(0, -twopi * dot(r3::vector<int>(Tr), kp__.vk())));
                const int at_lvl1 = potential_.hubbard_potential().find_orbital_index(ia1, in, il);
                const int at_lvl2 = potential_.hubbard_potential().find_orbital_index(ja, jn, jl);
                const int offset1 = potential_.hubbard_potential().offset(at_lvl1);
                const int offset2 = potential_.hubbard_potential().offset(at_lvl2);
                for (int is = 0; is < ctx_.num_spins(); is++) {
                    for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                        for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                            auto result1_ = z1 * std::conj(dn(offset2 + m2, offset1 + m1, is, dir, ia)) *
                                            potential_.hubbard_potential().nonlocal(i)(m1, m2, is);
                            d += std::real(result1_);
                        }
                    }
                }
            }
            forceh__(dir, ia) -= std::real(d);
        }
    }
}

mdarray<double, 2> const&
Force::calc_forces_vloc()
{
    PROFILE("sirius::Force::calc_forces_vloc");

    auto q = ctx_.gvec().shells_len();
    /* get form-factors for all G shells */
    auto ff = ctx_.ri().vloc_->values(q, ctx_.comm());

    forces_vloc_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    forces_vloc_.zero();

    auto& valence_rho = density_.rho();

    Unit_cell& unit_cell = ctx_.unit_cell();

    auto const& gvecs = ctx_.gvec();

    double fact = valence_rho.rg().gvec().reduced() ? 2.0 : 1.0;

    /* here the calculations are in lattice vectors space */
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
        Atom& atom = unit_cell.atom(ia);

        int iat = atom.type_id();

        for (auto it : gvecs) {
            int igsh = ctx_.gvec().shell(it.ig);

            /* cartesian form for getting cartesian force components */
            auto gvec_cart = gvecs.gvec_cart(it.igloc);

            /* scalar part of a force without multiplying by G-vector */
            std::complex<double> z = fact * fourpi * ff(igsh, iat) * std::conj(valence_rho.rg().f_pw_local(it.igloc)) *
                                     std::conj(ctx_.gvec_phase_factor(it.ig, ia));

            /* get force components multiplying by cartesian G-vector  */
            for (int x : {0, 1, 2}) {
                forces_vloc_(x, ia) -= (gvec_cart[x] * z).imag();
            }
        }
    }

    ctx_.comm().allreduce(&forces_vloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

    symmetrize_forces(ctx_.unit_cell(), forces_vloc_);

    return forces_vloc_;
}

mdarray<double, 2> const&
Force::calc_forces_usnl()
{
    calc_forces_us();
    calc_forces_nonloc();

    forces_usnl_ = mdarray<double, 2>({3, ctx_.unit_cell().num_atoms()});
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x : {0, 1, 2}) {
            forces_usnl_(x, ia) = forces_us_(x, ia) + forces_nonloc_(x, ia);
        }
    }

    return forces_usnl_;
}

void
Force::add_ibs_force(K_point<double>* kp__, Hamiltonian_k<double>& Hk__, mdarray<double, 2>& ffac__,
                     mdarray<double, 2>& forcek__, int ispn__) const
{
    PROFILE("sirius::Force::ibs_force");

    auto& uc = ctx_.unit_cell();

    auto& bg = ctx_.blacs_grid();

    auto bs = ctx_.cyclic_block_size();

    bool second = ctx_.cfg().control().use_second_variation();
    auto nfv    = second ? ctx_.num_fv_states() : ctx_.num_bands();

    auto ngklo = kp__->gklo_basis_size();

    /* compute density matrix for a k-point */
    la::dmatrix<std::complex<double>> dm(nfv, nfv, bg, bs, bs);
    la::dmatrix<std::complex<double>> canonical_evec;
    if (second) {
        compute_dmat(kp__, dm);
    } else {
        dm.zero();
        canonical_evec = la::dmatrix<std::complex<double>>(ngklo, nfv, bg, bs, bs);
        // Actual LAPW orbitals store AW and LO coefficients, while the eigenproblem
        // contains only PW and LO coefficients. AW coefficients are not independent.
        for (int b = 0; b < nfv; b++) {
            dm.set(b, b, kp__->band_occupancy(b, ispn__));
            auto c = kp__->spinor_wave_functions().gather(wf::spin_index(ispn__), wf::band_index(b));
            int ng = kp__->gkvec().num_gvec();
            for (int ig = 0; ig < ng; ig++) {
                canonical_evec.set(ig, b, c[ig]);
            }
            int row = ng, offset = ng;
            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                auto const& type = uc.atom(ia).type();
                for (int xi = 0; xi < type.mt_lo_basis_size(); xi++) {
                    canonical_evec.set(row++, b, c[offset + type.mt_aw_basis_size() + xi]);
                }
                offset += type.mt_basis_size();
            }
        }
    }

    auto& fv_evec = second ? kp__->fv_eigen_vectors() : canonical_evec;

    la::dmatrix<std::complex<double>> h(ngklo, ngklo, bg, bs, bs);
    la::dmatrix<std::complex<double>> o(ngklo, ngklo, bg, bs, bs);

    la::dmatrix<std::complex<double>> h1(ngklo, ngklo, bg, bs, bs);
    la::dmatrix<std::complex<double>> o1(ngklo, ngklo, bg, bs, bs);

    la::dmatrix<std::complex<double>> zm1(ngklo, nfv, bg, bs, bs);
    la::dmatrix<std::complex<double>> zf(nfv, nfv, bg, bs, bs);

    mdarray<std::complex<double>, 2> alm_row({kp__->num_gkvec_row(), uc.max_mt_aw_basis_size()});
    mdarray<std::complex<double>, 2> alm_col({kp__->num_gkvec_col(), uc.max_mt_aw_basis_size()});
    mdarray<std::complex<double>, 2> halm_col({kp__->num_gkvec_col(), uc.max_mt_aw_basis_size()});

    auto multiply = [&](char trans, int m, int n, int k, auto const& a, auto const& b, std::complex<double> beta,
                        auto& c) {
        auto const& one = la::constant<std::complex<double>>::one();
        if (bg.comm().size() == 1) {
            la::wrap(la::lib_t::blas)
                    .gemm(trans, 'N', m, n, k, &one, a.at(memory_t::host), a.ld(), b.at(memory_t::host), b.ld(), &beta,
                          c.at(memory_t::host), c.ld());
        } else {
            la::wrap(la::lib_t::scalapack).gemm(trans, 'N', m, n, k, &one, a, 0, 0, b, 0, 0, &beta, c, 0, 0);
        }
    };
    // Sequential eigensolvers replicate their matrices on the band communicator.
    double weight = kp__->weight() * bg.comm().size() / kp__->comm().size();

    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        h.zero();
        o.zero();

        auto& atom = uc.atom(ia);
        auto& type = atom.type();

        /* generate matching coefficients for current atom */
        kp__->alm_coeffs_row().generate<true>(atom, alm_row);
        kp__->alm_coeffs_col().generate<false>(atom, alm_col);

        /* setup apw-lo and lo-apw blocks */
        Hk__.set_fv_h_o_apw_lo(atom, ia, ispn__, alm_row, alm_col, h, o);

        /* apply MT Hamiltonian to column coefficients */
        Hk__.H0().apply_hmt_to_apw(device_t::CPU, ia, ispn__, kp__->num_gkvec_col(), alm_col, halm_col);

        /* apw-apw block of the overlap matrix */
        la::wrap(la::lib_t::blas)
                .gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                      &la::constant<std::complex<double>>::one(), alm_row.at(memory_t::host), alm_row.ld(),
                      alm_col.at(memory_t::host), alm_col.ld(), &la::constant<std::complex<double>>::zero(),
                      o.at(memory_t::host), o.ld());

        /* apw-apw block of the Hamiltonian matrix */
        la::wrap(la::lib_t::blas)
                .gemm('N', 'T', kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                      &la::constant<std::complex<double>>::one(), alm_row.at(memory_t::host), alm_row.ld(),
                      halm_col.at(memory_t::host), halm_col.ld(), &la::constant<std::complex<double>>::zero(),
                      h.at(memory_t::host), h.ld());

        int iat = type.id();

        for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
            auto gvec_col       = kp__->gkvec_col().gvec(gvec_index_t::local(igk_col));
            auto gkvec_col_cart = kp__->gkvec_col().gkvec_cart(gvec_index_t::local(igk_col));
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // for each column loop over rows
                auto gvec_row       = kp__->gkvec_row().gvec(gvec_index_t::local(igk_row));
                auto gkvec_row_cart = kp__->gkvec_row().gkvec_cart(gvec_index_t::local(igk_row));

                int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);

                int igs = ctx_.gvec().shell(ig12);

                auto zt = std::conj(ctx_.gvec_phase_factor(ig12, ia)) * ffac__(iat, igs) * fourpi / uc.omega();

                double t1 = 0.5 * dot(gkvec_row_cart, gkvec_col_cart);

                h(igk_row, igk_col) -= t1 * zt;
                o(igk_row, igk_col) -= zt;
            }
        }

        for (int x = 0; x < 3; x++) {
            /* zero lo-lo block */
            for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                for (int irow = 0; irow < kp__->num_lo_row(); irow++) {
                    h1(irow + kp__->num_gkvec_row(), icol + kp__->num_gkvec_col()) =
                            la::constant<std::complex<double>>::zero();
                    o1(irow + kp__->num_gkvec_row(), icol + kp__->num_gkvec_col()) =
                            la::constant<std::complex<double>>::zero();
                }
            }

            for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
                auto gvec_col = kp__->gkvec_col().gvec(gvec_index_t::local(igk_col));
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // loop over rows
                    auto gvec_row = kp__->gkvec_row().gvec(gvec_index_t::local(igk_row));
                    /* compute index of G-G' */
                    int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);
                    /* get G-G' */
                    auto vg = ctx_.gvec().gvec_cart(gvec_index_t::global(ig12));
                    /* multiply by i(G-G') */
                    h1(igk_row, igk_col) = std::complex<double>(0.0, vg[x]) * h(igk_row, igk_col);
                    /* multiply by i(G-G') */
                    o1(igk_row, igk_col) = std::complex<double>(0.0, vg[x]) * o(igk_row, igk_col);
                }
            }

            for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) {
                    auto gkvec_row_cart = kp__->gkvec_row().gkvec_cart(gvec_index_t::local(igk_row));
                    /* multiply by i(G+k) */
                    h1(igk_row, icol + kp__->num_gkvec_col()) =
                            std::complex<double>(0.0, gkvec_row_cart[x]) * h(igk_row, icol + kp__->num_gkvec_col());
                    /* multiply by i(G+k) */
                    o1(igk_row, icol + kp__->num_gkvec_col()) =
                            std::complex<double>(0.0, gkvec_row_cart[x]) * o(igk_row, icol + kp__->num_gkvec_col());
                }
            }

            for (int irow = 0; irow < kp__->num_lo_row(); irow++) {
                for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) {
                    auto gkvec_col_cart = kp__->gkvec_col().gkvec_cart(gvec_index_t::local(igk_col));
                    /* multiply by i(G+k) */
                    h1(irow + kp__->num_gkvec_row(), igk_col) =
                            std::complex<double>(0.0, -gkvec_col_cart[x]) * h(irow + kp__->num_gkvec_row(), igk_col);
                    /* multiply by i(G+k) */
                    o1(irow + kp__->num_gkvec_row(), igk_col) =
                            std::complex<double>(0.0, -gkvec_col_cart[x]) * o(irow + kp__->num_gkvec_row(), igk_col);
                }
            }

            /* zm1 = dO * V */
            multiply('N', ngklo, nfv, ngklo, o1, fv_evec, 0.0, zm1);
            /* multiply by energy: zm1 = E * (dO * V)  */
            for (int i = 0; i < zm1.num_cols_local(); i++) {
                int ist = zm1.icol(i);
                for (int j = 0; j < kp__->gklo_basis_size_row(); j++) {
                    zm1(j, i) *= second ? kp__->fv_eigen_value(ist) : kp__->band_energy(ist, ispn__);
                }
            }
            /* compute zm1 = dH * V - E * (dO * V) */
            multiply('N', ngklo, nfv, ngklo, h1, fv_evec, -1.0, zm1);

            /* compute zf = V^{+} * zm1 = V^{+} * (dH * V - E * (dO * V)) */
            multiply('C', nfv, nfv, ngklo, fv_evec, zm1, 0.0, zf);

            for (int i = 0; i < dm.num_cols_local(); i++) {
                for (int j = 0; j < dm.num_rows_local(); j++) {
                    forcek__(x, ia) += weight * std::real(dm(j, i) * zf(j, i));
                }
            }
        } // x
    } // ia
}

void
Force::print_info(std::ostream& out__, int verbosity__)
{
    auto print_forces = [&](std::string label__, mdarray<double, 2> const& forces) {
        out__ << "==== " << label__ << " =====" << std::endl;
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            out__ << "atom: " << std::setw(4) << ia << ", force: " << ffmt(15, 7) << forces(0, ia) << ffmt(15, 7)
                  << forces(1, ia) << ffmt(15, 7) << forces(2, ia) << std::endl;
        }
    };

    if (verbosity__ >= 1) {
        out__ << std::endl;
        print_forces("total Forces in Ha/bohr", forces_total());
    }

    if (verbosity__ >= 2) {
        if (ctx_.full_potential()) {
            print_forces("contribution from density", forces_rho());

            print_forces("contribution from Hartree potential", forces_hf());

            print_forces("contribution from IBS", forces_ibs());
        } else {
            print_forces("ultrasoft contribution from Qij", forces_us());

            print_forces("non-local contribution from Beta-projector", forces_nonloc());

            print_forces("contribution from local potential", forces_vloc());

            print_forces("contribution from core density", forces_core());

            if (ctx_.cb().external_xc_) {
                print_forces("explicit contribution from external XC", forces_xc_);
            }

            print_forces("Ewald forces from ions", forces_ewald());

            if (ctx_.hubbard_correction()) {
                print_forces("contribution from Hubbard correction", forces_hubbard());
            }

            if (ctx_.cfg().parameters().dftd3_correction()) {
                print_forces("contribution from the dftd3 dispersion correction", potential_.dftd3_ctx().forces());
            }

            if (ctx_.cfg().parameters().dftd4_correction()) {
                print_forces("contribution from the dftd4 dispersion correction", potential_.dftd4_ctx().forces());
            }
        }
    }
}

template void
Force::calc_forces_nonloc_aux<double>();
#if defined(SIRIUS_USE_FP32)
template void
Force::calc_forces_nonloc_aux<float>();
#endif

} // namespace sirius
