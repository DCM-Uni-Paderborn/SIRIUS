/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <chrono>
#include <filesystem>
#include "density/lapw_xc_atom.hpp"
#include "density/lapw_xc_field_map.hpp"
#include "hamiltonian/local_operator.hpp"
#include "hamiltonian/diagonalize_fp.hpp"
#include "lapw/dirac_basis.hpp"
#include "mixer/mixer_functions.hpp"

using namespace sirius;

namespace {

std::complex<double>
orbital(int i, int ia, int s, bool direction)
{
    double x = 0.37 * (i + 1) + 0.23 * ia + 0.19 * s;
    return direction ? std::complex<double>(0.13 * std::sin(2 * x), -0.09 * std::cos(3 * x))
                     : std::complex<double>(0.27 * std::cos(x), 0.21 * std::sin(1.7 * x));
}

double
occupation(int s)
{
    return 0.8 / (s + 1);
}

std::complex<double>
fourier_coefficient(r3::vector<int> g, int s, bool tau, bool direction)
{
    if (r3::dot(g, g) == 0) {
        return (direction ? 0.1 : 0.7) / (s + 1) * (tau ? 0.6 : 1.0);
    }
    if (g[2] != 0 || std::abs(g[0]) + std::abs(g[1]) != 1) {
        return 0;
    }
    return std::complex<double>(direction ? -0.012 : 0.06, (direction ? 0.009 : -0.04) * (g[0] + g[1])) *
           (tau ? 0.4 : 1.0) / double(s + 1);
}

int
test_lapw_hamiltonian(Simulation_context& ctx, lapw_xc_field_adjoint_t full_data,
                      lapw_xc_field_adjoint_t interstitial_data)
{
    auto const& comm = ctx.comm();
    int failures{0};
    Test_check check(failures, comm, "LAPW Hamiltonian");
    for (auto& atom : interstitial_data.local) {
        for (auto& part : atom) {
            for (auto& field : part) {
                field.zero();
            }
        }
    }
    for (auto& atom : interstitial_data.core) {
        for (auto& field : atom) {
            std::fill(field.begin(), field.end(), 0);
        }
    }
    Potential base(ctx), full(ctx), interstitial(ctx);
    for (auto* p : {&base, &full, &interstitial}) {
        p->zero();
        p->fft_transform(-1);
    }
    full.set_lapw_xc_derivatives(std::move(full_data));
    interstitial.set_lapw_xc_derivatives(std::move(interstitial_data));
    Density density(ctx);
    check.reject("no incomplete XC total energy", [&] { full.energy_exc(density); });
    check.reject("no incomplete rho contraction", [&] { full.energy_vxc(density); });
    check.reject("no incomplete tau contraction", [&] { full.energy_vtau(density); });
    check.reject("no incomplete magnetic contraction", [&] { energy_bxc(density, full); });
    check.reject("no incomplete effective-potential contraction", [&] { energy_veff(density, full); });
    check.reject("no partial operator copy from staged source", [&] { copy(full, base); });
    check.reject("no stale staged operator after copy", [&] { copy(base, full); });
    auto const& d = *full.lapw_xc_derivatives();
    bool spinor   = ctx.num_mag_dims() == 3;
    int nf        = spinor ? 4 : ctx.num_spins();
    std::vector<std::vector<std::complex<double>>> vrho, vtau;
    for (int s = 0; s < nf; s++) {
        vrho.push_back(d.rho[s].gather_f_pw());
        vtau.push_back(d.tau[s].gather_f_pw());
    }
    bool second    = ctx.cfg().control().use_second_variation();
    auto component = [&](auto const& fields, int g, int j, bool canonical) {
        if (spinor) {
            return j < 2 ? fields[j][g] : fields[2][g] + std::complex<double>(0, 2 * j - 5) * fields[3][g];
        }
        if (canonical || ctx.num_spins() == 1) {
            return fields[j][g];
        }
        return 0.5 * (fields[0][g] + double(1 - 2 * j) * fields[1][g]);
    };
    Hamiltonian0<double> h0(base, density.lapw_basis_sptr());
    Hamiltonian0<double> h1(full, density.lapw_basis_sptr());
    Hamiltonian0<double> hi(interstitial, density.lapw_basis_sptr());
    double error{0};
    for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
        for (int s = 0; s < (second ? 1 : ctx.num_spins()); s++) {
            error = std::max(error, std::abs(h1.veff_pw(ig, s) - h0.veff_pw(ig, s) - component(vrho, ig, s, !second)));
            error = std::max(error, std::abs(h1.vtau_pw(ig, s) - component(vtau, ig, s, !second)));
        }
    }
    check("unmasked exact Fourier covectors", error, 2e-13);
    auto mt_error = [&](Hamiltonian0<double> const& a, Hamiltonian0<double> const& b) {
        double error{0};
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            LAPW_xc_atom op(
                    ctx.unit_cell().atom(ia),
                    density.lapw_basis().radial_basis_for_class((ctx.unit_cell().atom(ia)).symmetry_class().id()),
                    ctx.lmax_rho());
            std::vector<mdarray<std::complex<double>, 2>> matrices;
            for (int s = 0; s < nf; s++) {
                matrices.push_back(op.matrix_elements(d.local[ia][0][s], d.local[ia][1][s]));
            }
            int n = ctx.unit_cell().atom(ia).type().mt_basis_size();
            for (int s = 0; s < nf; s++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        auto expected = matrices[s](i, j);
                        if (spinor && s >= 2) {
                            expected = matrices[2](i, j) + std::complex<double>(0, 2 * s - 5) * matrices[3](i, j);
                        } else if (!spinor && second && ctx.num_spins() == 2) {
                            expected = 0.5 * (matrices[0](i, j) + double(1 - 2 * s) * matrices[1](i, j));
                        }
                        error = std::max(error, std::abs(b.hmt(ia)(i, j, s) - a.hmt(ia)(i, j, s) - expected));
                    }
                }
            }
        }
        return error;
    };
    check("AW/LO local matrices and spin conversion", mt_error(h0, h1), 2e-13);

    K_point_set ks(ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 1);
    ks.initialize();
    double local_error{0}, matrix_error{0}, full_matrix_error{0}, full_apply_error{0}, boundary_error{0};
    double magnetic_mt_error{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto k0 = h0(*kp);
        auto k1 = hi(*kp);
        auto k2 = h1(*kp);
        int n = kp->gklo_basis_size(), bs = ctx.cyclic_block_size();
        la::dmatrix<std::complex<double>> m0(n, n, ctx.blacs_grid(), bs, bs), m1(n, n, ctx.blacs_grid(), bs, bs);
        la::dmatrix<std::complex<double>> o0(n, n, ctx.blacs_grid(), bs, bs), o1(n, n, ctx.blacs_grid(), bs, bs);
        la::dmatrix<std::complex<double>> m2(n, n, ctx.blacs_grid(), bs, bs), o2(n, n, ctx.blacs_grid(), bs, bs);
        std::vector<mdarray<std::complex<double>, 2>> mt_basis;
        std::vector<int> num_lo, lo_offset;
        int next_lo = kp->gkvec().num_gvec();
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx.unit_cell().atom(ia);
            int naw = atom.type().mt_aw_basis_size(), nmt = atom.type().mt_basis_size();
            mdarray<std::complex<double>, 2> alm({kp->gkvec().count(), naw});
            kp->alm_coeffs_loc().generate<false>(
                    atom, alm, density.lapw_basis().radial_basis_for_class(atom.symmetry_class().id()));
            auto centre   = r3::dot(ctx.unit_cell().lattice_vectors(), atom.position());
            double radius = atom.type().mt_radius();
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto q      = kp->gkvec().gkvec_cart(gvec_index_t::local(ig));
                auto angles = r3::spherical_coordinates(q);
                std::vector<std::complex<double>> ylm(sf::lmmax(atom.type().lmax_apw()));
                sf::spherical_harmonics(atom.type().lmax_apw(), angles[1], angles[2], ylm.data());
                for (int l = 0; l <= atom.type().lmax_apw(); l++) {
                    double bessel     = gsl_sf_bessel_jl(l, angles[0] * radius);
                    double derivative = l / radius * bessel - angles[0] * gsl_sf_bessel_jl(l + 1, angles[0] * radius);
                    for (int m = -l; m <= l; m++) {
                        auto factor = fourpi / std::sqrt(ctx.unit_cell().omega()) *
                                      std::polar(1.0, r3::dot(q, centre)) * std::pow(std::complex<double>(0, 1), l) *
                                      std::conj(ylm[sf::lm(l, m)]);
                        for (int deriv = 0; deriv <= 1; deriv++) {
                            std::complex<double> value{0};
                            for (int order = 0; order < static_cast<int>(atom.type().aw_descriptor(l).size());
                                 order++) {
                                value += density.lapw_basis_sptr()
                                                 ->radial_basis_for_class(atom.symmetry_class().id())
                                                 .aw_surface_deriv(l, order, deriv) *
                                         alm(ig, atom.type().indexb_by_l_m_order(l, m, order));
                            }
                            boundary_error =
                                    std::max(boundary_error, std::abs(value - factor * (deriv ? derivative : bessel)));
                        }
                    }
                }
            }
            mt_basis.push_back(mdarray<std::complex<double>, 2>({nmt, n}));
            auto& a = mt_basis.back();
            a.zero();
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                for (int xi = 0; xi < naw; xi++) {
                    a(xi, kp->gkvec().offset() + ig) = alm(ig, xi);
                }
            }
            kp->comm().allreduce(a.at(memory_t::host), a.size());
            num_lo.push_back(nmt - naw);
            lo_offset.push_back(next_lo);
            for (int ilo = 0; ilo < nmt - naw; ilo++) {
                a(naw + ilo, next_lo++) = 1;
            }
        }
        mdarray<std::complex<double>, 2> scalar_delta({n, n});
        scalar_delta.zero();
        for (int s = 0; s < (second ? 1 : ctx.num_spins()); s++) {
            k0.set_fv_h_o(s, m0, o0);
            k1.set_fv_h_o(s, m1, o1);
            k2.set_fv_h_o(s, m2, o2);
            std::vector<mdarray<std::complex<double>, 2>> local_matrices;
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                LAPW_xc_atom op(
                        ctx.unit_cell().atom(ia),
                        density.lapw_basis().radial_basis_for_class((ctx.unit_cell().atom(ia)).symmetry_class().id()),
                        ctx.lmax_rho());
                auto matrix = op.matrix_elements(d.local[ia][0][s], d.local[ia][1][s]);
                if (!spinor && second && ctx.num_spins() == 2) {
                    auto down = op.matrix_elements(d.local[ia][0][1], d.local[ia][1][1]);
                    for (size_t i = 0; i < matrix.size(); i++) {
                        matrix[i] = 0.5 * (matrix[i] + down[i]);
                    }
                }
                local_matrices.push_back(std::move(matrix));
            }
            for (int j = 0; j < m0.num_cols_local(); j++) {
                for (int i = 0; i < m0.num_rows_local(); i++) {
                    std::complex<double> expected{0};
                    if (m0.irow(i) < kp->gkvec().num_gvec() && m0.icol(j) < kp->gkvec().num_gvec()) {
                        auto gi = kp->gkvec().gvec(gvec_index_t::global(m0.irow(i)));
                        auto gj = kp->gkvec().gvec(gvec_index_t::global(m0.icol(j)));
                        auto qi = kp->gkvec().gkvec_cart(gvec_index_t::global(m0.irow(i)));
                        auto qj = kp->gkvec().gkvec_cart(gvec_index_t::global(m0.icol(j)));
                        int g   = ctx.gvec().index_g12(gi, gj);
                        expected =
                                component(vrho, g, s, !second) + 0.5 * r3::dot(qi, qj) * component(vtau, g, s, !second);
                    }
                    matrix_error = std::max(
                            {matrix_error, std::abs(m1(i, j) - m0(i, j) - expected), std::abs(o1(i, j) - o0(i, j))});
                    // Independently transform each atomic matrix into the full PW/LO basis.
                    for (size_t ia = 0; ia < mt_basis.size(); ia++) {
                        auto const& a = mt_basis[ia];
                        auto const& h = local_matrices[ia];
                        for (int v = 0; v < static_cast<int>(h.size(1)); v++) {
                            for (int u = 0; u < static_cast<int>(h.size(0)); u++) {
                                expected += std::conj(a(u, m0.irow(i))) * h(u, v) * a(v, m0.icol(j));
                            }
                        }
                    }
                    full_matrix_error = std::max({full_matrix_error, std::abs(m2(i, j) - m0(i, j) - expected),
                                                  std::abs(o2(i, j) - o0(i, j))});
                    if (s == 0) {
                        scalar_delta(m0.irow(i), m0.icol(j)) = expected;
                    }
                }
            }
        }
        // LAPACK uses a replicated exact matrix, even with distributed orbitals.
        ctx.blacs_grid().comm().allreduce(scalar_delta.at(memory_t::host), scalar_delta.size());
        // The iterative first-variation operator uses the scalar channel.
        if (second || ctx.num_spins() == 1) {
            wf::Wave_functions<double> phi(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                           memory_t::host);
            wf::Wave_functions<double> a(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                         memory_t::host);
            wf::Wave_functions<double> b(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                         memory_t::host);
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                phi.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) =
                        orbital(kp->gkvec().offset() + ig, 0, 0, false);
            }
            for (auto atom : phi.spl_num_atoms()) {
                for (int ilo = 0; ilo < num_lo[atom.i]; ilo++) {
                    phi.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)) =
                            orbital(lo_offset[atom.i] + ilo, 0, 0, false);
                }
            }
            k0.apply_fv_h_o(false, false, wf::band_range(0, 1), phi, &a, nullptr);
            k2.apply_fv_h_o(false, false, wf::band_range(0, 1), phi, &b, nullptr);
            auto residual = [&](int i, std::complex<double> value) {
                std::complex<double> expected{0};
                for (int j = 0; j < n; j++) {
                    expected += scalar_delta(i, j) * orbital(j, 0, 0, false);
                }
                full_apply_error = std::max(full_apply_error, std::abs(value - expected));
            };
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                residual(kp->gkvec().offset() + ig, b.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) -
                                                            a.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)));
            }
            for (auto atom : phi.spl_num_atoms()) {
                for (int ilo = 0; ilo < num_lo[atom.i]; ilo++) {
                    residual(lo_offset[atom.i] + ilo,
                             b.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)) -
                                     a.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)));
                }
            }
        }
        wf::Wave_functions<double> phi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> a(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> b(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> za(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> zb(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> xa(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> xb(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        auto c = [&](int ig) { return orbital(ig, 0, 0, false); };
        for (int ig = 0; ig < kp->gkvec().count(); ig++) {
            phi.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) = c(kp->gkvec().offset() + ig);
        }
        h0.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), wf::band_range(0, 1), phi, &a, nullptr,
                                   ctx.num_spins() == 2 ? &za : nullptr, spinor ? &xa : nullptr);
        hi.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), wf::band_range(0, 1), phi, &b, nullptr,
                                   ctx.num_spins() == 2 ? &zb : nullptr, spinor ? &xb : nullptr);
        for (int s = 0; s < (spinor ? 3 : ctx.num_spins()); s++) {
            auto const& wa = s == 0 ? a : s == 1 ? za : xa;
            auto const& wb = s == 0 ? b : s == 1 ? zb : xb;
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto gi = kp->gkvec().gvec(gvec_index_t::local(ig));
                auto qi = kp->gkvec().gkvec_cart(gvec_index_t::local(ig));
                std::complex<double> expected{0};
                for (int jg = 0; jg < kp->gkvec().num_gvec(); jg++) {
                    auto gj = kp->gkvec().gvec(gvec_index_t::global(jg));
                    auto qj = kp->gkvec().gkvec_cart(gvec_index_t::global(jg));
                    int g   = ctx.gvec().index_g12(gi, gj);
                    expected += (component(vrho, g, s, false) + 0.5 * r3::dot(qi, qj) * component(vtau, g, s, false)) *
                                c(jg);
                }
                local_error = std::max(local_error,
                                       std::abs(wb.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) -
                                                wa.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) - expected));
            }
        }
        if (spinor) {
            std::vector<int> nmt;
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                nmt.push_back(ctx.unit_cell().atom(ia).type().mt_basis_size());
            }
            int nb = ctx.num_fv_states();
            wf::Wave_functions<double> psi(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(nb),
                                           memory_t::host);
            std::vector<wf::Wave_functions<double>> ba, bb;
            for (int s = 0; s < 3; s++) {
                ba.emplace_back(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(nb), memory_t::host);
                bb.emplace_back(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(nb), memory_t::host);
            }
            for (auto atom : psi.spl_num_atoms()) {
                for (int b = 0; b < nb; b++) {
                    for (int i = 0; i < nmt[atom.i]; i++) {
                        psi.mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b)) = orbital(i, atom.i, b, false);
                    }
                }
            }
            h0.apply_bmt(psi, ba);
            h1.apply_bmt(psi, bb);
            for (auto atom : psi.spl_num_atoms()) {
                LAPW_xc_atom op(ctx.unit_cell().atom(atom.i),
                                density.lapw_basis().radial_basis_for_class(
                                        (ctx.unit_cell().atom(atom.i)).symmetry_class().id()),
                                ctx.lmax_rho());
                auto z = op.matrix_elements(d.local[atom.i][0][1], d.local[atom.i][1][1]);
                auto x = op.matrix_elements(d.local[atom.i][0][2], d.local[atom.i][1][2]);
                auto y = op.matrix_elements(d.local[atom.i][0][3], d.local[atom.i][1][3]);
                for (int b = 0; b < nb; b++) {
                    for (int i = 0; i < nmt[atom.i]; i++) {
                        for (int s : {0, 2}) {
                            std::complex<double> expected{0};
                            for (int j = 0; j < nmt[atom.i]; j++) {
                                auto matrix = s == 0 ? z(i, j) : x(i, j) - std::complex<double>(0, 1) * y(i, j);
                                expected += matrix * orbital(j, atom.i, b, false);
                            }
                            magnetic_mt_error = std::max(
                                    magnetic_mt_error,
                                    std::abs(bb[s].mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b)) -
                                             ba[s].mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b)) -
                                             expected));
                        }
                    }
                }
            }
        }
    }
    check("exact LAPW assembly, shifted k and unchanged overlap", matrix_error, 3e-12);
    check("matrix-free scalar/magnetic rho and tau", local_error, 3e-12);
    if (spinor) {
        check("MT longitudinal and complex transverse operator application", magnetic_mt_error, 3e-12);
    }
    check("complete PW/AW/LO assembly", full_matrix_error, 3e-12);
    check("complete matrix-free PW/AW/LO application", full_apply_error, 3e-12);
    check("LAPW boundary value and radial slope", boundary_error, 3e-12);

    // Change a basis function after XC evaluation. Cached matrices would now be stale.
    auto& atom = ctx.unit_cell().atom(0);
    std::vector<double> saved(atom.type().num_mt_points()), changed(saved.size());
    for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
        saved[ir] =
                density.lapw_basis_sptr()->radial_basis_for_class(atom.symmetry_class().id()).radial_function(ir, 0);
        changed[ir] = 1.07 * saved[ir];
    }
    auto set_radial = [&](std::vector<double> const& values) {
        density.lapw_basis_sptr()->radial_basis_for_class(atom.symmetry_class().id()).radial_function(0, values);
        Spline<double> spline(atom.type().radial_grid(), values);
        spline.interpolate();
        std::vector<double> derivative(values.size());
        for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
            derivative[ir] = atom.type().radial_grid(ir) * spline.deriv(1, ir);
        }
        density.lapw_basis_sptr()
                ->radial_basis_for_class(atom.symmetry_class().id())
                .radial_function_derivative(0, derivative);
        for (int order = 0; order <= 2; order++) {
            density.lapw_basis_sptr()
                    ->radial_basis_for_class(atom.symmetry_class().id())
                    .aw_surface_deriv(0, 0, order, spline.deriv(order, values.size() - 1));
        }
    };
    set_radial(changed);
    Hamiltonian0<double> updated_base(base, density.lapw_basis_sptr()), updated_full(full, density.lapw_basis_sptr());
    check("reprojection after radial-basis update", mt_error(updated_base, updated_full), 2e-13);
    double change = std::abs(updated_full.hmt(0)(0, 0, 0) - updated_base.hmt(0)(0, 0, 0) - h1.hmt(0)(0, 0, 0) +
                             h0.hmt(0)(0, 0, 0));
    check("basis update changes the XC matrix", change > 1e-8 ? 0 : 1, 0);
    set_radial(saved);
    return failures;
}

int
test_dirac_core_map()
{
    auto const& comm = mpi::Communicator::world();
    int failures{0};
    Test_check check(failures, comm, "Dirac core with scalar valence");
    for (int mag : {0, 1, 3}) {
        auto config                          = R"({"parameters": {
            "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
            "pw_cutoff": 4.0, "gk_cutoff": 0.5, "num_bands": 4, "xc_functionals": ["XC_LDA_X"],
            "core_relativity": "dirac", "valence_relativity": "none", "auto_rmt": 0,
            "lmax_apw": 1, "lmax_rho": 2, "lmax_pot": 2
        }, "control": {"processing_unit": "CPU", "verbosity": 0}})"_json;
        config["parameters"]["num_mag_dims"] = mag;
        Simulation_context ctx(config);
        ctx.unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 8, 0}, {0, 0, 8}});
        auto& type = ctx.unit_cell().add_atom_type("Ne");
        type.zn(10);
        std::vector<double> radii{1e-6, 2e-6, 5e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.8};
        type.set_radial_grid(static_cast<int>(radii.size()), radii.data());
        auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 100, 1e-7, 20.0, 3.0);
        type.free_atom_density(std::vector<double>(free_grid.num_points(), 0));
        type.set_free_atom_radial_grid(std::move(free_grid));
        type.set_configuration(1, 0, 1, 2.0, true);
        for (int l : {0, 1}) {
            for (int order : {0, 1}) {
                type.add_aw_descriptor(l + 1, l, -0.5, order, 0);
            }
        }
        ctx.unit_cell().add_atom("Ne", {0, 0, 0});
        ctx.initialize();
        Density density(ctx, true);
        density.zero();
        density.kinetic_density().zero();
        double gamma = std::sqrt(1 - std::pow(type.zn() / speed_of_light, 2));
        std::array<double, 2> powers{2 * gamma - 2, 2 * gamma - 4};
        Periodic_function<double> metric_field(ctx, [](int) { return lmax_t(0); });
        metric_field.zero();
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            metric_field.mt()[0](0, ir) = std::pow(radii[ir] / radii.back(), powers[1]);
        }
        auto tau_property = mixer::periodic_function_property_tau();
        double norm       = tau_property.inner(metric_field, metric_field);
        check("finite Coulomb-regularized tau mixer norm", std::isfinite(norm) ? 0 : 1, 0);
        check("Coulomb tau norm excludes origin singularity", std::abs(norm - std::pow(radii.back(), 3) / 3), 2e-12);
        auto value = [](double r, double power) { return 1e-8 * std::pow(r, power) * (1 + 0.4 * r + 0.3 * r * r); };
        std::vector<double> knots(3, radii.front());
        knots.insert(knots.end(), radii.begin(), radii.end());
        knots.insert(knots.end(), 3, radii.back());
        bspline_basis<4> basis(std::move(knots));
        std::vector<double> controls(basis.size(), value(radii.front(), 0));
        for (int i = 3; i < basis.size() - 1; i++) {
            double r        = (basis.knot(i + 1) + basis.knot(i + 2) + basis.knot(i + 3)) / 3;
            int left        = type.radial_grid().index_of(r);
            double fraction = (r - radii[left]) / (radii[left + 1] - radii[left]);
            controls[i]     = (1 - fraction) * value(radii[left], 0) + fraction * value(radii[left + 1], 0);
        }
        controls.back()             = value(radii.back(), 0);
        auto sampled_value_gradient = [&](double r, double power) {
            double regular{0}, slope{0};
            if (r < radii.front()) {
                regular = controls.front();
            } else {
                // Sum the complete basis, independently of the production segment-local implementation.
                for (int i = 0; i < basis.size(); i++) {
                    regular += controls[i] * basis(i, r);
                    slope += controls[i] * basis.deriv(i, r);
                }
            }
            double sampled = std::pow(r, power) * regular;
            return std::array<double, 2>{sampled, power * sampled / r + std::pow(r, power) * slope};
        };
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            density.rho().mt()[0](0, ir)                      = value(radii[ir], powers[0]) / y00;
            density.kinetic_density().scalar().mt()[0](0, ir) = value(radii[ir], powers[1]) / y00;
        }
        int nf = mag == 3 ? 4 : ctx.num_spins();
        std::vector<Smooth_periodic_function<double>> rho, tau;
        for (int s = 0; s < nf; s++) {
            rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
            tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
            rho.back().f_pw_local().zero();
            tau.back().f_pw_local().zero();
        }
        std::vector<r3::vector<double>> points;
        std::array<r3::vector<double>, 3> all{{{2e-7, 0, 0}, {3.4e-6, 0, 0}, {0.01, 0.02, 0.03}}};
        for (int p = 0; p < static_cast<int>(all.size()); p++) {
            if (p % comm.size() == comm.rank()) {
                points.push_back(all[p]);
            }
        }
        LAPW_xc_field_map map(density, points);
        auto fields = map.sample(rho, tau);
        xc_point_fields_t covectors(points.size(), nf);
        double error{0}, forward{0};
        for (int p = 0; p < static_cast<int>(points.size()); p++) {
            double r = points[p].length();
            for (int s = 0; s < nf; s++) {
                double spin       = mag == 3 ? (s == 0 ? 1 : 0) : 1.0 / nf;
                auto sampled      = sampled_value_gradient(r, powers[0]);
                double expected   = spin * sampled[0];
                double derivative = spin * sampled[1];
                auto compare      = [&](double actual, double reference) {
                    error = std::max(error, std::abs(actual - reference) / std::max(1.0, std::abs(reference)));
                };
                compare(fields.rho(p, s), expected);
                compare(fields.tau(p, s), spin * sampled_value_gradient(r, powers[1])[0]);
                covectors.rho(p, s) = r * r;
                covectors.tau(p, s) = r * r * r * r;
                forward += covectors.rho(p, s) * fields.rho(p, s) + covectors.tau(p, s) * fields.tau(p, s);
                for (int x = 0; x < 3; x++) {
                    compare(fields.gradient(p, x, s), derivative * points[p][x] / r);
                    covectors.gradient(p, x, s) = (x + 1) * r * r * r;
                    forward += covectors.gradient(p, x, s) * fields.gradient(p, x, s);
                }
            }
        }
        auto reverse = map.adjoint(covectors);
        double backward{0};
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            for (int part : {0, 1}) {
                backward += reverse.core[0][part][ir] * value(radii[ir], powers[part]);
            }
        }
        comm.allreduce(&forward, 1);
        check("Coulomb core values and charge gradients", error, 2e-10);
        check("Coulomb core sampling transpose", std::abs(forward - backward), 2e-12);
    }
    return failures;
}

int
test_lapw_map(int num_mag_dims, bool replicated_fft, bool second_variation)
{
    auto const& comm                          = mpi::Communicator::world();
    auto config                               = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo",
        "use_symmetry": false, "pw_cutoff": 4.0, "gk_cutoff": 1.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"], "core_relativity": "none", "valence_relativity": "none",
        "auto_rmt": 0, "lmax_apw": 2, "lmax_rho": 4, "lmax_pot": 4
    }, "control": {"processing_unit": "CPU", "verbosity": 0},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false}})"_json;
    config["parameters"]["num_mag_dims"]      = num_mag_dims;
    config["control"]["mpi_grid_dims"]        = {replicated_fft ? 1 : comm.size(), 1};
    config["control"]["use_second_variation"] = second_variation;
    config["iterative_solver"]["type"]        = "exact";
    Simulation_context ctx(config);
    ctx.unit_cell().set_lattice_vectors({{8, 0.7, 0.2}, {0, 9, 0.4}, {0, 0, 10}});
    for (auto label : {"Li", "H"}) {
        auto& type = ctx.unit_cell().add_atom_type(label);
        type.zn(std::string(label) == "Li" ? 3 : 1);
        type.set_radial_grid(radial_grid_t::power, 501, 1e-4, 2.0, 2.0);
        auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
        type.free_atom_density(std::vector<double>(free_grid.num_points(), 0));
        type.set_free_atom_radial_grid(std::move(free_grid));
        for (int l = 0; l <= 2; l++) {
            type.add_aw_descriptor(l + 1, l, -0.5, 0, 0);
            type.add_aw_descriptor(l + 1, l, -0.5, 1, 0);
        }
        for (int l = 0; l <= 1; l++) {
            type.add_lo_descriptor(l, l + 2, l, -0.3, 0, 0);
            type.add_lo_descriptor(l, l + 2, l, -0.3, 1, 0);
        }
        if (std::string(label) == "Li") {
            type.set_configuration(1, 0, 1, 2.0, true);
        }
    }
    ctx.unit_cell().add_atom("Li", {0.15, 0.2, 0.25});
    ctx.unit_cell().add_atom("H", {0.65, 0.7, 0.75});
    ctx.initialize();
    Density density(ctx, true);
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx.unit_cell().atom(ia);
        for (auto const& rf : atom.type().indexr()) {
            std::vector<double> values(atom.type().num_mt_points());
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                double r   = atom.type().radial_grid(ir);
                values[ir] = std::pow(r, rf.am.l()) * std::exp(-(0.4 + 0.1 * rf.idxrf) * r * r);
                if (rf.idxlo >= 0) {
                    values[ir] *= std::pow(1 - r * r / 4, 2);
                }
            }
            density.lapw_basis_sptr()
                    ->radial_basis_for_class(atom.symmetry_class().id())
                    .radial_function(rf.idxrf, values);
            Spline<double> spline(atom.type().radial_grid(), values);
            spline.interpolate();
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                values[ir] = atom.type().radial_grid(ir) * spline.deriv(1, ir);
            }
            density.lapw_basis_sptr()
                    ->radial_basis_for_class(atom.symmetry_class().id())
                    .radial_function_derivative(rf.idxrf, values);
            if (rf.idxlo < 0) {
                for (int d = 0; d <= 2; d++) {
                    density.lapw_basis_sptr()
                            ->radial_basis_for_class(atom.symmetry_class().id())
                            .aw_surface_deriv(rf.am.l(), rf.order, d, spline.deriv(d, atom.type().num_mt_points() - 1));
                }
            }
        }
    }
    bool spinor = num_mag_dims == 3;
    int nf      = spinor ? 4 : ctx.num_spins();
    int failures{0};
    Test_check check(failures, comm,
                     "LAPW map spins=" + std::to_string(nf) + " replicated FFT=" + std::to_string(replicated_fft) +
                             " second variation=" + std::to_string(second_variation));
    check.reject("core required before tau construction", [&] { density.generate_mt_kinetic_density(true); });
    check.reject("stored tau required for field sampling", [&] {
        Density without_tau(ctx);
        LAPW_xc_field_map map(without_tau, {});
    });
    bool fields_initialized{false};
    auto update_mt_fields = [&] {
        for (auto it : ctx.unit_cell().spl_num_atoms()) {
            auto local = LAPW_xc_atom(ctx.unit_cell().atom(it.i),
                                      density.lapw_basis().radial_basis_for_class(
                                              (ctx.unit_cell().atom(it.i)).symmetry_class().id()),
                                      ctx.lmax_rho())
                                 .fields(density.density_matrix(it.i));
            for (size_t i = 0; i < local[0][0].size(); i++) {
                if (spinor) {
                    for (int s = 0; s < nf; s++) {
                        density.component(s).mt()[it.i][i] = local[0][s][i];
                    }
                    continue;
                }
                double up = local[0][0][i], dn = nf == 2 ? local[0][1][i] : 0;
                density.rho().mt()[it.i][i] = up + dn;
                if (nf == 2) {
                    density.mag(0).mt()[it.i][i] = up - dn;
                }
            }
            for (int ir = 0; ir < ctx.unit_cell().atom(it.i).num_mt_points(); ir++) {
                density.rho().mt()[it.i](0, ir) += density.core_charge_density(it.i)[ir] / y00;
            }
        }
        for (int j = 0; j < nf; j++) {
            density.component(j).mt().sync(ctx.unit_cell().spl_num_atoms());
        }
        density.generate_mt_kinetic_density(true);
    };
    auto solve_core = [&](double scale) {
        std::vector<std::vector<double>> v(ctx.unit_cell().num_atom_symmetry_classes());
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& type = ctx.unit_cell().atom_symmetry_class(ic).atom_type();
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                v[ic].push_back(-scale * type.zn() / type.radial_grid(ir));
            }
        }
        density.generate_core_charge_density(v);
        if (fields_initialized) {
            update_mt_fields();
        }
    };
    solve_core(1);
    auto set_dm = [&](double step, double angle = 0) {
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto& dm          = density.density_matrix(ia);
            auto coefficients = [&](int i) {
                std::array<std::complex<double>, 2> c{};
                for (int s = 0; s < ctx.num_spins(); s++) {
                    c[s] = orbital(i, ia, s, false) + step * orbital(i, ia, s, true);
                }
                double a = std::cos(angle / 2), b = std::sin(angle / 2);
                return std::array<std::complex<double>, 2>{a * c[0] - b * c[1], b * c[0] + a * c[1]};
            };
            for (int j = 0; j < static_cast<int>(dm.size(1)); j++) {
                for (int i = 0; i < static_cast<int>(dm.size(0)); i++) {
                    auto ci = coefficients(i), cj = coefficients(j);
                    for (int s = 0; s < ctx.num_spins(); s++) {
                        dm(i, j, s) = occupation(spinor ? 0 : s) * std::conj(ci[s]) * cj[s];
                    }
                    if (spinor) {
                        dm(i, j, 2) = occupation(0) * std::conj(ci[1]) * cj[0];
                    }
                }
            }
        }
        update_mt_fields();
        fields_initialized = true;
    };
    set_dm(0);
    auto const& metric_field = density.kinetic_density().scalar();
    check("nonrelativistic tau mixer norm is unchanged",
          std::abs(mixer::periodic_function_property_tau().inner(metric_field, metric_field) -
                   inner(metric_field, metric_field)),
          0);
    std::vector<Smooth_periodic_function<double>> rho, tau;
    for (int s = 0; s < nf; s++) {
        rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
    }
    auto set_pw = [&](double step) {
        for (int s = 0; s < nf; s++) {
            for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                auto g = ctx.gvec().gvec(gvec_index_t::local(ig));
                rho[s].f_pw_local(ig) =
                        fourier_coefficient(g, s, false, false) + step * fourier_coefficient(g, s, false, true);
                tau[s].f_pw_local(ig) =
                        fourier_coefficient(g, s, true, false) + step * fourier_coefficient(g, s, true, true);
            }
        }
    };
    set_pw(0);
    auto const& lattice = ctx.unit_cell().lattice_vectors();
    auto centre         = [&](int ia) { return r3::dot(lattice, ctx.unit_cell().atom(ia).position()); };
    // The exact surface belongs to MT, including when the point has a periodic image.
    std::vector<r3::vector<double>> all_points{centre(0) + r3::vector<double>(0.4, 0.2, -0.3),
                                               centre(1) + r3::vector<double>(-0.2, 0.33, 0.15),
                                               r3::dot(lattice, r3::vector<double>(0.4, 0.45, 0.5)),
                                               centre(0) + r3::vector<double>(0, 0, 2),
                                               centre(0) + r3::vector<double>(0, 0, 2 + 1e-6),
                                               centre(0) + r3::vector<double>(0, 0, 2 - 1e-6),
                                               centre(0) + r3::vector<double>(0.4, 0.2, -0.3) +
                                                       r3::dot(lattice, r3::vector<double>(2, -1, 1)),
                                               centre(1) + r3::vector<double>(1e-5, 2e-5, -1e-5)};
    std::vector<int> ids;
    std::vector<r3::vector<double>> points;
    for (int p = 0; p < static_cast<int>(all_points.size()); p++) {
        if ((replicated_fft ? 0 : p % comm.size()) == comm.rank()) {
            ids.push_back(p);
            points.push_back(all_points[p]);
        }
    }
    LAPW_xc_field_map map(density, points);
    auto fields = map.sample(rho, tau);
    double error{0};
    auto pw_values = Smooth_periodic_function_sampler(ctx.gvec(), points);
    for (int s = 0; s < nf; s++) {
        auto rpw = pw_values.sample(rho[s]);
        auto tpw = pw_values.sample(tau[s]);
        for (int p = 0; p < static_cast<int>(points.size()); p++) {
            int id = ids[p];
            std::array<double, 4> r;
            double t;
            if (id == 2 || id == 4) {
                r = rpw[p];
                t = tpw[p][0];
            } else {
                int ia     = id == 1 || id == 7 ? 1 : 0;
                auto d     = all_points[id == 6 ? 0 : id] - centre(ia);
                auto local = LAPW_xc_atom(ctx.unit_cell().atom(ia),
                                          density.lapw_basis().radial_basis_for_class(
                                                  (ctx.unit_cell().atom(ia)).symmetry_class().id()),
                                          ctx.lmax_rho())
                                     .fields(density.density_matrix(ia));
                r = Spheric_function_interpolator(local[0][s], spheric_inner_extension_t::charge).value_gradient(d);
                t = Spheric_function_interpolator(local[1][s], spheric_inner_extension_t::kinetic).value_gradient(d)[0];
                Flm core(1, ctx.unit_cell().atom(ia).type().radial_grid());
                for (int ir = 0; ir < core.radial_grid().num_points(); ir++) {
                    core(0, ir) =
                            (!spinor || s == 0) ? density.core_charge_density(ia)[ir] / ((spinor ? 1 : nf) * y00) : 0;
                }
                auto cr = Spheric_function_interpolator(core, spheric_inner_extension_t::charge).value_gradient(d);
                for (int x = 0; x < 4; x++) {
                    r[x] += cr[x];
                }
                for (int ir = 0; ir < core.radial_grid().num_points(); ir++) {
                    core(0, ir) =
                            (!spinor || s == 0) ? density.core_kinetic_density(ia)[ir] / ((spinor ? 1 : nf) * y00) : 0;
                }
                t += Spheric_function_interpolator(core, spheric_inner_extension_t::charge).value_gradient(d)[0];
            }
            error = std::max({error, std::abs(fields.rho(p, s) - r[0]), std::abs(fields.tau(p, s) - t)});
            for (int x = 0; x < 3; x++) {
                error = std::max(error, std::abs(fields.gradient(p, x, s) - r[x + 1]));
            }
        }
    }
    check("piecewise values, gradients, core and periodic images", error, 3e-11);

    std::vector<r3::vector<double>> surface_points{all_points[3], all_points[3],
                                                   all_points[3] + r3::dot(lattice, r3::vector<double>(2, -1, 1))};
    for (int p : {1, 2}) {
        surface_points[p][2] = std::nextafter(surface_points[p][2], std::numeric_limits<double>::infinity());
    }
    auto surface = LAPW_xc_field_map(density, surface_points).sample(rho, tau);
    error        = 0;
    for (int s = 0; s < nf; s++) {
        for (int p : {1, 2}) {
            error = std::max({error, std::abs(surface.rho(p, s) - surface.rho(0, s)),
                              std::abs(surface.tau(p, s) - surface.tau(0, s))});
            for (int x = 0; x < 3; x++) {
                error = std::max(error, std::abs(surface.gradient(p, x, s) - surface.gradient(0, x, s)));
            }
        }
    }
    check("surface ownership survives coordinate and periodic-image roundoff", error, 3e-11);

    // Independent orbital values test the complex Gaunt contraction and DM convention.
    error = 0;
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto const& atom = ctx.unit_cell().atom(ia);
        auto local       = LAPW_xc_atom(atom, density.lapw_basis().radial_basis_for_class((atom).symmetry_class().id()),
                                        ctx.lmax_rho())
                                   .fields(density.density_matrix(ia));
        auto native_tau  = density.generate_mt_valence_kinetic_density(ia);
        SHT sht(device_t::CPU, ctx.lmax_rho());
        for (int s = 0; s < nf; s++) {
            auto values = transform(sht, local[0][s]);
            for (int ir : {0, 200, 500}) {
                for (int tp = 0; tp < sht.num_points(); tp++) {
                    auto angular = r3::spherical_coordinates(sht.coord(tp));
                    std::vector<std::complex<double>> ylm(sf::lmmax(2));
                    sf::spherical_harmonics(2, angular[1], angular[2], ylm.data());
                    std::array<std::complex<double>, 2> psi{};
                    for (int spin = 0; spin < (spinor ? 2 : 1); spin++) {
                        for (auto const& b : atom.type().indexb()) {
                            psi[spin] += orbital(b.xi, ia, spinor ? spin : s, false) *
                                         density.lapw_basis_sptr()
                                                 ->radial_basis_for_class(atom.symmetry_class().id())
                                                 .radial_function(ir, b.idxrf) *
                                         ylm[b.lm];
                        }
                    }
                    std::array<double, 4> target{
                            std::norm(psi[0]) + std::norm(psi[1]), std::norm(psi[0]) - std::norm(psi[1]),
                            2 * std::real(std::conj(psi[0]) * psi[1]), 2 * std::imag(std::conj(psi[0]) * psi[1])};
                    error = std::max(error,
                                     std::abs(values(tp, ir) - occupation(spinor ? 0 : s) * target[spinor ? s : 0]));
                }
            }
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                    error = std::max(error, std::abs(local[1][s](lm, ir) - native_tau[s](lm, ir)));
                }
            }
        }
    }
    check("charge from complex orbitals and native MT tau", error, 2e-12);

    // A single nonlinear nonlocal energy spans both domains and both spins.
    // Point 0 has no energy weight but participates in the descriptor moment.
    auto energy = [&](LAPW_xc_field_map const& m, xc_point_fields_t* derivatives) {
        auto f = m.sample(rho, tau);
        double moment{0}, e{0};
        for (int p = 0; p < static_cast<int>(ids.size()); p++) {
            double w = ids[p] == 0 ? 0 : 0.02 + 0.003 * ids[p];
            double d = 0.06 + 0.005 * ids[p];
            for (int s = 0; s < nf; s++) {
                double r = f.rho(p, s), t = f.tau(p, s);
                moment += d * (r + 0.7 * t);
                e += w * (0.3 * r * r + 0.2 * t * t + 0.13 * r * t);
                for (int x = 0; x < 3; x++) {
                    e += 0.1 * w * std::pow(f.gradient(p, x, s), 2);
                }
            }
        }
        comm.allreduce(&moment, 1);
        comm.allreduce(&e, 1);
        if (derivatives) {
            for (int p = 0; p < static_cast<int>(ids.size()); p++) {
                double w = ids[p] == 0 ? 0 : 0.02 + 0.003 * ids[p];
                double d = 0.06 + 0.005 * ids[p];
                for (int s = 0; s < nf; s++) {
                    derivatives->rho(p, s) = w * (0.6 * f.rho(p, s) + 0.13 * f.tau(p, s)) + 0.04 * moment * d;
                    derivatives->tau(p, s) = w * (0.4 * f.tau(p, s) + 0.13 * f.rho(p, s)) + 0.028 * moment * d;
                    for (int x = 0; x < 3; x++) {
                        derivatives->gradient(p, x, s) = 0.2 * w * f.gradient(p, x, s);
                    }
                }
            }
        }
        return e + 0.02 * moment * moment;
    };
    xc_point_fields_t derivatives(points.size(), nf);
    double xc_energy = energy(map, &derivatives);
    for (int p = 0; p < static_cast<int>(ids.size()); p++) {
        if (ids[p] == 0 && std::abs(derivatives.rho(p, 0)) < 1e-4) {
            failures++;
        }
    }
    auto reverse          = map.adjoint(derivatives);
    auto local_derivative = [&](lapw_xc_field_adjoint_t const& r, bool real_only) {
        double value{0};
        for (size_t ia = 0; ia < r.local.size(); ia++) {
            LAPW_xc_atom op(
                    ctx.unit_cell().atom(ia),
                    density.lapw_basis().radial_basis_for_class((ctx.unit_cell().atom(ia)).symmetry_class().id()),
                    ctx.lmax_rho());
            for (int s = 0; s < nf; s++) {
                auto matrix = op.matrix_elements(r.local[ia][0][s], r.local[ia][1][s]);
                for (int j = 0; j < static_cast<int>(matrix.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(matrix.size(0)); i++) {
                        auto h = real_only ? std::complex<double>(matrix(i, j).real()) : matrix(i, j);
                        if (!spinor) {
                            value += 2 * occupation(s) *
                                     std::real(std::conj(orbital(i, ia, s, true)) * h * orbital(j, ia, s, false));
                        } else {
                            std::complex<double> coherence{0};
                            for (int a = 0; a < 2; a++) {
                                for (int b = 0; b < 2; b++) {
                                    std::complex<double> pauli{0};
                                    if (s == 0 && a == b)
                                        pauli = 1;
                                    if (s == 1 && a == b)
                                        pauli = 1 - 2 * a;
                                    if (s == 2 && a != b)
                                        pauli = 1;
                                    if (s == 3 && a != b)
                                        pauli = std::complex<double>(0, 2 * a - 1);
                                    coherence += std::conj(orbital(i, ia, a, true)) * pauli * orbital(j, ia, b, false);
                                }
                            }
                            value += 2 * occupation(0) * std::real(h * coherence);
                        }
                    }
                }
            }
        }
        return value;
    };
    double interstitial{0};
    for (int s = 0; s < nf; s++) {
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            auto g              = ctx.gvec().gvec(gvec_index_t::local(ig));
            double multiplicity = ctx.gvec().reduced() && r3::dot(g, g) != 0 ? 2 : 1;
            interstitial +=
                    ctx.gvec().omega() * multiplicity *
                    std::real(std::conj(reverse.rho[s].f_pw_local(ig)) * fourier_coefficient(g, s, false, true) +
                              std::conj(reverse.tau[s].f_pw_local(ig)) * fourier_coefficient(g, s, true, true));
        }
    }
    comm.allreduce(&interstitial, 1);
    double local    = local_derivative(reverse, false);
    double analytic = local + interstitial;
    if (comm.rank() == 0) {
        std::cout << "analytic derivatives MT=" << local << " interstitial=" << interstitial << '\n';
    }
    for (double h : {1e-4, 5e-5}) {
        set_dm(h);
        set_pw(h);
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        set_dm(-h);
        set_pw(-h);
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        check("joint complex-orbital/PW derivative", std::abs((plus - minus) / (2 * h) - analytic), 2e-9);
    }
    set_dm(0);
    set_pw(0);
    for (bool mt_only : {false, true}) {
        double h = 1e-4;
        if (mt_only) {
            set_dm(h);
        } else {
            set_pw(h);
        }
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        if (mt_only) {
            set_dm(-h);
        } else {
            set_pw(-h);
        }
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        check(mt_only ? "MT-only orbital derivative" : "interstitial-only PW derivative",
              std::abs((plus - minus) / (2 * h) - (mt_only ? local : interstitial)), 2e-9);
        set_dm(0);
        set_pw(0);
    }
    check("interstitial derivative is exercised", std::abs(interstitial) > 1e-5 ? 0 : 1, 0);
    check("imaginary local matrix is exercised", std::abs(local - local_derivative(reverse, true)) > 1e-5 ? 0 : 1, 0);
    auto no_tau = xc_point_fields_t(points.size(), nf);
    std::copy(derivatives.rho.begin(), derivatives.rho.end(), no_tau.rho.begin());
    std::copy(derivatives.gradient.begin(), derivatives.gradient.end(), no_tau.gradient.begin());
    no_tau.tau.zero();
    check("local tau derivative is exercised",
          std::abs(local - local_derivative(map.adjoint(no_tau), false)) > 1e-5 ? 0 : 1, 0);

    // Differentiate the sampled core fields, without claiming a tau-dependent core solver.
    for (double h : {1e-4, 5e-5}) {
        solve_core(1 + h);
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        std::vector<std::array<std::vector<double>, 2>> core_plus;
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            core_plus.push_back({density.core_charge_density(ia), density.core_kinetic_density(ia)});
        }
        solve_core(1 - h);
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        double core_derivative{0};
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            std::array<std::vector<double> const*, 2> values{&density.core_charge_density(ia),
                                                             &density.core_kinetic_density(ia)};
            for (int part = 0; part < 2; part++) {
                for (size_t ir = 0; ir < values[part]->size(); ir++) {
                    core_derivative +=
                            reverse.core[ia][part][ir] * (core_plus[ia][part][ir] - (*values[part])[ir]) / (2 * h);
                }
            }
        }
        check("core-field adjoint", std::abs(core_derivative - (plus - minus) / (2 * h)), 2e-7);
        check("core response is not silently discarded", std::abs(core_derivative) > 1e-3 ? 0 : 1, 0);
    }
    solve_core(1);

    // Geometry partials keep the physical fields fixed and do not cross MT surfaces.
    // The Hamiltonian's first/second-variation choice does not change this map.
    if (!second_variation || spinor) {
        auto geometry_points = points;
        for (size_t p = 0; p < ids.size(); p++) {
            if (ids[p] >= 3 && ids[p] <= 5) {
                geometry_points[p] = centre(0) + r3::vector<double>(0, 0, ids[p] == 4 ? 2.5 : 1.5);
            } else if (ids[p] == 7) {
                geometry_points[p] = centre(1) + r3::vector<double>(0.15, -0.2, 0.25);
            }
        }
        LAPW_xc_field_map geometry_map(density, geometry_points);
        xc_point_fields_t covectors(geometry_points.size(), nf);
        energy(geometry_map, &covectors);
        auto geometry = geometry_map.geometry_adjoint(rho, tau, covectors);
        auto cell = lattice, reciprocal = ctx.gvec().lattice_vectors(), inverse = r3::inverse(cell);
        std::vector<r3::vector<double>> fractional_atoms, atoms;
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            fractional_atoms.push_back(ctx.unit_cell().atom(ia).position());
            atoms.push_back(centre(ia));
        }
        auto set_cell = [&](r3::matrix<double> const& trial) {
            ctx.unit_cell().set_lattice_vectors(trial);
            ctx.gvec_sptr()->lattice_vectors(twopi * r3::transpose(r3::inverse(trial)));
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                ctx.unit_cell().atom(ia).set_position(r3::dot(r3::inverse(trial), atoms[ia]));
            }
        };
        auto fd_error = [](double fd, double ad) {
            return std::isfinite(fd) && std::isfinite(ad) ? std::abs(fd - ad) / std::max(1.0, std::abs(ad))
                                                          : std::numeric_limits<double>::infinity();
        };
        double point_error{0}, atom_error{0}, cell_error{0};
        for (int x = 0; x < 3; x++) {
            double dp{0};
            for (size_t p = 0; p < ids.size(); p++) {
                dp += geometry.points[p][x] * (0.13 + 0.02 * ids[p]);
            }
            comm.allreduce(&dp, 1);
            for (double h : {1e-5, 5e-6}) {
                auto displaced = [&](double step) {
                    auto trial = geometry_points;
                    for (size_t p = 0; p < ids.size(); p++) {
                        trial[p][x] += step * (0.13 + 0.02 * ids[p]);
                    }
                    return energy(LAPW_xc_field_map(density, trial), nullptr);
                };
                double plus = displaced(h), minus = displaced(-h);
                point_error = std::max(point_error, fd_error((plus - minus) / (2 * h), dp));
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto shifted_atom = [&](double step) {
                        auto trial = atoms[ia];
                        trial[x] += step;
                        ctx.unit_cell().atom(ia).set_position(r3::dot(inverse, trial));
                        return energy(LAPW_xc_field_map(density, geometry_points), nullptr);
                    };
                    plus       = shifted_atom(h);
                    minus      = shifted_atom(-h);
                    atom_error = std::max(atom_error, fd_error((plus - minus) / (2 * h), geometry.atoms(x, ia)));
                    ctx.unit_cell().atom(ia).set_position(fractional_atoms[ia]);
                }
                for (int y = 0; y < 3; y++) {
                    auto strained = [&](double step) {
                        auto trial = cell;
                        trial(x, y) += step;
                        set_cell(trial);
                        return energy(LAPW_xc_field_map(density, geometry_points), nullptr);
                    };
                    plus       = strained(h);
                    minus      = strained(-h);
                    cell_error = std::max(cell_error, fd_error((plus - minus) / (2 * h), geometry.lattice(x, y)));
                    set_cell(cell);
                }
            }
        }
        ctx.gvec_sptr()->lattice_vectors(reciprocal);
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            ctx.unit_cell().atom(ia).set_position(fractional_atoms[ia]);
        }
        check("MT/interstitial geometry point FD", point_error, 2e-7);
        check("geometry atom FD including physical core fields", atom_error, 2e-7);
        check("geometry all nine fixed-Cartesian cell components FD", cell_error, 2e-7);

        std::vector<r3::vector<double>> point_directions;
        mdarray<double, 2> atom_directions({3, ctx.unit_cell().num_atoms()});
        r3::matrix<double> lattice_direction;
        for (size_t p = 0; p < ids.size(); p++) {
            point_directions.emplace_back(0.13 + 0.02 * ids[p], -0.04, 0.07);
        }
        for (int x = 0; x < 3; x++) {
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                atom_directions(x, ia) = 0.03 * std::sin(x + ia + 0.7);
            }
            for (int y = 0; y < 3; y++) {
                lattice_direction(x, y) = 0.02 * std::cos(x + 2 * y + 0.4);
            }
        }
        auto response = geometry_map.geometry_response(rho, tau, point_directions, atom_directions, lattice_direction);
        auto reverse_response =
                geometry_map.geometry_adjoint_response(covectors, point_directions, atom_directions, lattice_direction);
        double forward{0}, backward{0};
        for (size_t i = 0; i < response.rho.size(); i++) {
            forward += response.rho[i] * covectors.rho[i] + response.tau[i] * covectors.tau[i];
        }
        for (size_t i = 0; i < response.gradient.size(); i++) {
            forward += response.gradient[i] * covectors.gradient[i];
        }
        for (size_t p = 0; p < ids.size(); p++) {
            backward += r3::dot(geometry.points[p], point_directions[p]);
        }
        comm.allreduce(&forward, 1);
        comm.allreduce(&backward, 1);
        for (size_t i = 0; i < atom_directions.size(); i++) {
            backward += geometry.atoms[i] * atom_directions[i];
        }
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                backward += geometry.lattice(x, y) * lattice_direction(x, y);
            }
        }
        check("geometry forward/adjoint duality", fd_error(forward, backward), 2e-10);
        for (double h : {1e-5, 5e-6}) {
            auto shifted = [&](double step) {
                auto trial_cell   = cell;
                auto trial_points = geometry_points;
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        trial_cell(x, y) += step * lattice_direction(x, y);
                    }
                }
                set_cell(trial_cell);
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto atom = atoms[ia];
                    for (int x = 0; x < 3; x++) {
                        atom[x] += step * atom_directions(x, ia);
                    }
                    ctx.unit_cell().atom(ia).set_position(r3::dot(r3::inverse(trial_cell), atom));
                }
                for (size_t p = 0; p < ids.size(); p++) {
                    trial_points[p] += step * point_directions[p];
                }
                LAPW_xc_field_map trial(density, trial_points);
                auto result = std::make_pair(trial.sample(rho, tau), trial.adjoint(covectors));
                set_cell(cell);
                return result;
            };
            auto plus = shifted(h), minus = shifted(-h);
            double response_error{0};
            auto compare = [&](auto const& d, auto const& p, auto const& m) {
                double peak{0}, error{0};
                for (size_t i = 0; i < d.size(); i++) {
                    peak  = std::max(peak, std::abs(d[i]));
                    error = std::max(error, std::abs(d[i] - (p[i] - m[i]) / (2 * h)));
                }
                response_error = std::max(response_error, error / std::max(1.0, peak));
            };
            compare(response.rho, plus.first.rho, minus.first.rho);
            compare(response.tau, plus.first.tau, minus.first.tau);
            compare(response.gradient, plus.first.gradient, minus.first.gradient);
            for (int s = 0; s < nf; s++) {
                compare(reverse_response.rho[s].f_pw_local(), plus.second.rho[s].f_pw_local(),
                        minus.second.rho[s].f_pw_local());
                compare(reverse_response.tau[s].f_pw_local(), plus.second.tau[s].f_pw_local(),
                        minus.second.tau[s].f_pw_local());
            }
            for (size_t ia = 0; ia < reverse_response.local.size(); ia++) {
                for (int part = 0; part < 2; part++) {
                    for (int s = 0; s < nf; s++) {
                        compare(reverse_response.local[ia][part][s], plus.second.local[ia][part][s],
                                minus.second.local[ia][part][s]);
                    }
                    compare(reverse_response.core[ia][part], plus.second.core[ia][part], minus.second.core[ia][part]);
                }
            }
            check("joint field-sampling/transpose geometry response FD", response_error, 2e-7);
        }
    }
    if (spinor) {
        Density initial(ctx, true);
        std::array<double, 4> constant{0.8, 0.12, 0.16, 0.2};
        for (int j = 0; j < 4; j++) {
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                initial.component(j).rg().value(ir) = constant[j];
            }
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                initial.component(j).mt()[ia].zero();
                for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                    initial.component(j).mt()[ia](0, ir) = constant[j] / y00;
                }
            }
        }
        initial.initial_kinetic_density();
        double m  = std::sqrt(0.08);
        auto tf   = [&](double n) { return 0.3 * std::pow(6 * pi * pi, 2.0 / 3.0) * std::pow(n, 5.0 / 3.0); };
        double up = tf((constant[0] + m) / 2), dn = tf((constant[0] - m) / 2), guess_error{0};
        for (int j = 0; j < 4; j++) {
            double target     = j == 0 ? up + dn : (up - dn) * constant[j] / m;
            auto const& field = initial.kinetic_density().component(j);
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                guess_error = std::max(guess_error, std::abs(field.rg().value(ir) - target));
            }
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                auto const& mt = field.mt()[ia];
                for (size_t i = 0; i < mt.size(); i++) {
                    guess_error =
                            std::max(guess_error, std::abs(mt[i] - (i % ctx.lmmax_rho() == 0 ? target / y00 : 0)));
                }
            }
        }
        check("initial LAPW spinor tau in both domains", guess_error, 2e-12);
        double angle = 0.71, c = std::cos(angle), s = std::sin(angle);
        set_dm(0, angle);
        for (auto* part : {&rho, &tau}) {
            for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                auto z = (*part)[1].f_pw_local(ig), x = (*part)[2].f_pw_local(ig);
                (*part)[1].f_pw_local(ig) = c * z - s * x;
                (*part)[2].f_pw_local(ig) = s * z + c * x;
            }
        }
        auto rotated = LAPW_xc_field_map(density, points).sample(rho, tau);
        double rotation_error{0}, origin_gradient_error{0};
        for (int p = 0; p < static_cast<int>(points.size()); p++) {
            auto compare = [&](auto value, auto target, bool gradient = false) {
                std::array<double, 4> expected{target(0), c * target(1) - s * target(2), s * target(1) + c * target(2),
                                               target(3)};
                for (int j = 0; j < 4; j++) {
                    auto& error = gradient && ids[p] == 7 ? origin_gradient_error : rotation_error;
                    error       = std::max(error, std::abs(value(j) - expected[j]) /
                                                          std::max({1.0, std::abs(value(j)), std::abs(expected[j])}));
                }
            };
            compare([&](int j) { return rotated.rho(p, j); }, [&](int j) { return fields.rho(p, j); });
            compare([&](int j) { return rotated.tau(p, j); }, [&](int j) { return fields.tau(p, j); });
            for (int x = 0; x < 3; x++) {
                compare([&](int j) { return rotated.gradient(p, x, j); },
                        [&](int j) { return fields.gradient(p, x, j); }, true);
            }
        }
        check("global spin rotation of both LAPW domains and scalar core", rotation_error, 2e-12);
        // Angular gradients amplify coefficient roundoff by 1/r at the near-nucleus target.
        check("near-origin gradient spin rotation", origin_gradient_error, 1e-10);
        set_dm(0);
        set_pw(0);

        K_point_set ks(ctx);
        ks.add_kpoint({0.13, -0.17, 0.07}, 1);
        ks.initialize();
        std::array<std::complex<double>, 2> amplitudes{1, std::complex<double>(0.6, 0.3)};
        double occupied = ctx.unit_cell().num_valence_electrons();
        for (int b = 0; b < ctx.num_bands(); b++) {
            ks.get<double>(0)->band_occupancy(b, 0, b == 0 ? occupied : 0);
        }
        for (auto it : ks.spl_num_kpoints()) {
            auto kp   = ks.get<double>(it.i);
            auto& psi = kp->spinor_wave_functions();
            psi.zero(memory_t::host);
            for (int spin = 0; spin < 2; spin++) {
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    if (kp->gkvec().gvec(gvec_index_t::local(ig)) == r3::vector<int>(1, 0, 0)) {
                        psi.pw_coeffs(ig, wf::spin_index(spin), wf::band_index(0)) = amplitudes[spin];
                    }
                }
                for (auto ia : psi.spl_num_atoms()) {
                    for (int xi = 0; xi < ctx.unit_cell().atom(ia.i).type().mt_basis_size(); xi++) {
                        psi.mt_coeffs(xi, atom_index_t::local(ia.li), wf::spin_index(spin), wf::band_index(0)) =
                                orbital(xi, ia.i, spin, false);
                    }
                }
            }
        }
        auto q = r3::dot(ctx.unit_cell().reciprocal_lattice_vectors(), r3::vector<double>(1.13, -0.17, 0.07));
        std::array<double, 4> smooth{1.45, 0.55, 1.2, 0.6};
        for (bool core : {false, true}) {
            density.generate<double>(ks, false, core, true);
            double generation_error{0};
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                auto const& dm = density.density_matrix(ia);
                for (int j = 0; j < static_cast<int>(dm.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(dm.size(0)); i++) {
                        for (int block = 0; block < 3; block++) {
                            int a = block == 2 ? 1 : block, b = block == 2 ? 0 : block;
                            auto expected = occupied * std::conj(orbital(i, ia, a, false)) * orbital(j, ia, b, false);
                            generation_error = std::max(generation_error, std::abs(dm(i, j, block) - expected));
                        }
                    }
                }
                auto local = LAPW_xc_atom(ctx.unit_cell().atom(ia),
                                          density.lapw_basis().radial_basis_for_class(
                                                  (ctx.unit_cell().atom(ia)).symmetry_class().id()),
                                          ctx.lmax_rho())
                                     .fields(dm);
                for (int j = 0; j < 4; j++) {
                    for (size_t i = 0; i < local[0][j].size(); i++) {
                        for (int part : {0, 1}) {
                            auto const& stored = part == 0 ? density.component(j).mt()[ia]
                                                           : density.kinetic_density().component(j).mt()[ia];
                            double expected    = local[part][j][i];
                            if (core && j == 0 && i % ctx.lmmax_rho() == 0) {
                                auto const& radial =
                                        part == 0 ? density.core_charge_density(ia) : density.core_kinetic_density(ia);
                                expected += radial[i / ctx.lmmax_rho()] / y00;
                            }
                            generation_error = std::max(generation_error, std::abs(stored[i] - expected));
                        }
                    }
                }
            }
            for (int j = 0; j < 4; j++) {
                double expected = occupied * 0.5 * r3::dot(q, q) * smooth[j] / ctx.unit_cell().omega();
                for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                    generation_error =
                            std::max(generation_error,
                                     std::abs(density.kinetic_density().component(j).rg().value(ir) - expected));
                }
            }
            check(core ? "native LAPW spinor rho/tau plus scalar core" : "native LAPW spinor valence rho/tau",
                  generation_error, 3e-12);
        }
        set_dm(0);
    }
    failures += test_lapw_hamiltonian(ctx, map.adjoint(derivatives), map.adjoint(derivatives));

    // The physical fields, not the current density matrix, define the energy
    // contraction. Compare the production contraction to the original point sum.
    for (int ig = 0; ig < ctx.gvec().count(); ig++) {
        density.rho().rg().f_pw_local(ig)                      = rho[0].f_pw_local(ig);
        density.kinetic_density().scalar().rg().f_pw_local(ig) = tau[0].f_pw_local(ig);
        if (spinor) {
            for (int j = 1; j < nf; j++) {
                density.component(j).rg().f_pw_local(ig)                   = rho[j].f_pw_local(ig);
                density.kinetic_density().component(j).rg().f_pw_local(ig) = tau[j].f_pw_local(ig);
            }
        } else if (ctx.num_spins() == 2) {
            density.rho().rg().f_pw_local(ig) += rho[1].f_pw_local(ig);
            density.kinetic_density().scalar().rg().f_pw_local(ig) += tau[1].f_pw_local(ig);
            density.mag(0).rg().f_pw_local(ig)                      = rho[0].f_pw_local(ig) - rho[1].f_pw_local(ig);
            density.kinetic_density().vector(0).rg().f_pw_local(ig) = tau[0].f_pw_local(ig) - tau[1].f_pw_local(ig);
        }
    }
    density.fft_transform(1);
    density.kinetic_density().fft_transform(1);
    std::array<double, 3> point_sum{0, 0, 0};
    for (int p = 0; p < static_cast<int>(points.size()); p++) {
        double r = fields.rho(p, 0);
        std::array<double, 3> g;
        for (int x = 0; x < 3; x++) {
            g[x] = fields.gradient(p, x, 0);
        }
        if (!spinor && ctx.num_spins() == 2) {
            r = 0.5 * (r + fields.rho(p, 1));
            for (int x = 0; x < 3; x++) {
                g[x] = 0.5 * (g[x] + fields.gradient(p, x, 1));
            }
        }
        for (int s = 0; s < nf; s++) {
            double scalar = spinor && s > 0 ? 0 : r;
            point_sum[0] += derivatives.rho(p, s) * scalar;
            point_sum[1] += derivatives.rho(p, s) * (fields.rho(p, s) - scalar);
            point_sum[2] += derivatives.tau(p, s) * fields.tau(p, s);
            for (int x = 0; x < 3; x++) {
                double scalar_gradient = spinor && s > 0 ? 0 : g[x];
                point_sum[0] += derivatives.gradient(p, x, s) * scalar_gradient;
                point_sum[1] += derivatives.gradient(p, x, s) * (fields.gradient(p, x, s) - scalar_gradient);
            }
        }
    }
    comm.allreduce(point_sum.data(), point_sum.size());
    check("homogeneous functional Euler identity", std::abs(point_sum[0] + point_sum[1] + point_sum[2] - 2 * xc_energy),
          2e-12);

    Potential staged(ctx);
    staged.zero();
    staged.xc_potential().zero();
    for (int j = 0; j < nf; j++) {
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            staged.component(j).rg().value(ir) = 0.17 / (j + 1);
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                staged.component(j).mt()[ia](0, ir) = 0.23 / (j + 1);
            }
        }
    }
    staged.fft_transform(-1);
    double electrostatic = energy_veff(density, staged), ordinary_magnetic = energy_bxc(density, staged);
    staged.set_lapw_xc_derivatives(map.adjoint(derivatives), xc_energy);
    auto core_fields        = staged.get_core_xc_derivatives();
    auto const& atom_fields = staged.lapw_xc_derivatives()->core;
    double class_error{0};
    for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
        auto const& cls = ctx.unit_cell().atom_symmetry_class(ic);
        for (int part = 0; part < 2; part++) {
            for (size_t ir = 0; ir < core_fields[ic][part].size(); ir++) {
                double average{0};
                for (int i = 0; i < cls.num_atoms(); i++) {
                    average += atom_fields[cls.atom_id(i)][part][ir] / cls.num_atoms();
                }
                class_error = std::max(class_error, std::abs(average - core_fields[ic][part][ir]));
            }
        }
    }
    check("class-resolved spherical core covectors", class_error, 0);
    auto contractions = staged.lapw_xc_contractions(density);
    check("discrete scalar-rho energy contraction", std::abs(contractions.rho - point_sum[0]), 2e-12);
    check("discrete magnetic energy contraction", std::abs(contractions.magnetic - point_sum[1]), 2e-12);
    check("discrete tau energy contraction", std::abs(contractions.tau - point_sum[2]), 2e-12);
    check("functional energy associated with its covectors", std::abs(staged.energy_exc(density) - xc_energy), 0);
    check("XC scalar contraction API", std::abs(staged.energy_vxc(density) - point_sum[0]), 2e-12);
    check("XC tau contraction API", std::abs(staged.energy_vtau(density) - point_sum[2]), 2e-12);
    check("XC addition to electrostatic contraction",
          std::abs(energy_veff(density, staged) - electrostatic - point_sum[0]), 2e-12);
    check("XC addition to magnetic contraction",
          std::abs(energy_bxc(density, staged) - ordinary_magnetic - point_sum[1]), 2e-12);
    check.reject("no pseudo-core contraction in LAPW", [&] { staged.energy_vxc_core(density); });
    check.reject("no LAPW energy density for nonlocal XC", [&] { staged.xc_energy_density(); });
    check.reject("energy contraction requires stored tau", [&] {
        Density without_tau(ctx);
        staged.energy_vtau(without_tau);
    });
    auto const* original = staged.lapw_xc_derivatives();
    for (int invalid : {0, 1, 2, 3, 4}) {
        auto bad = map.adjoint(derivatives);
        if (comm.rank() == 0) {
            switch (invalid) {
                case 0:
                    bad.rho.clear();
                    break;
                case 1:
                    bad.local[0][0][0] = Flm(1, ctx.unit_cell().atom(0).type().radial_grid());
                    break;
                case 2:
                    bad.core[0][1][0] = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 3:
                    bad.tau[0].f_pw_local(0) = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 4:
                    bad.core[0][1][0] += 0.1;
                    break;
            }
        }
        check.reject("collective invalid Hamiltonian covectors",
                     [&] { staged.set_lapw_xc_derivatives(std::move(bad)); });
        check("rejected update preserves the staged operator", staged.lapw_xc_derivatives() == original ? 0 : 1, 0);
        check("rejected update preserves functional energy", std::abs(staged.energy_exc(density) - xc_energy), 0);
    }
    auto bad_energy = map.adjoint(derivatives);
    check.reject("collective nonfinite XC energy", [&] {
        staged.set_lapw_xc_derivatives(std::move(bad_energy),
                                       comm.rank() == 0 ? std::numeric_limits<double>::quiet_NaN() : xc_energy);
    });
    if (comm.size() > 1) {
        auto bad_presence = map.adjoint(derivatives);
        check.reject("collective mismatched XC energy presence", [&] {
            staged.set_lapw_xc_derivatives(std::move(bad_presence),
                                           comm.rank() == 0 ? std::optional<double>(xc_energy) : std::nullopt);
        });
        auto bad_value = map.adjoint(derivatives);
        check.reject("collective inconsistent XC energy",
                     [&] { staged.set_lapw_xc_derivatives(std::move(bad_value), xc_energy + 0.1 * comm.rank()); });
    }
    check("rejected energy preserves the staged operator", staged.lapw_xc_derivatives() == original ? 0 : 1, 0);
    check("rejected energy preserves functional energy", std::abs(staged.energy_exc(density) - xc_energy), 0);
    staged.set_lapw_xc_derivatives(map.adjoint(derivatives));
    check.reject("operator-only restaging clears previous energy", [&] { staged.energy_exc(density); });
    if (spinor) {
        return failures;
    }

    for (int region : {0, 1, 2}) {
        std::vector<r3::vector<double>> selected;
        if (comm.rank() == 0 && region) {
            selected.push_back(all_points[region == 1 ? 1 : 2]);
        }
        LAPW_xc_field_map small(density, selected);
        auto f = small.sample(rho, tau);
        xc_point_fields_t zero(selected.size(), ctx.num_spins());
        zero.rho.zero();
        zero.gradient.zero();
        zero.tau.zero();
        auto r        = small.adjoint(zero);
        auto geometry = small.geometry_adjoint(rho, tau, zero);
        double maximum{0};
        for (auto const& p : geometry.points) {
            maximum = std::max(maximum, p.length());
        }
        for (size_t i = 0; i < geometry.atoms.size(); i++) {
            maximum = std::max(maximum, std::abs(geometry.atoms[i]));
        }
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                maximum = std::max(maximum, std::abs(geometry.lattice(x, y)));
            }
        }
        for (auto const& atom : r.local) {
            for (auto const& part : atom) {
                for (auto const& field : part) {
                    for (size_t i = 0; i < field.size(); i++) {
                        maximum = std::max(maximum, std::abs(field[i]));
                    }
                }
            }
        }
        for (auto const& atom : r.core) {
            for (auto const& field : atom) {
                for (double value : field) {
                    maximum = std::max(maximum, std::abs(value));
                }
            }
        }
        for (auto const* fields : {&r.rho, &r.tau}) {
            for (auto const& field : *fields) {
                for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                    maximum = std::max(maximum, std::abs(field.f_pw_local(ig)));
                }
            }
        }
        check("empty, MT-only and interstitial-only target sets", maximum, 0);
    }
    auto bad_points = points;
    if (comm.rank() == 0) {
        bad_points.push_back(centre(0));
    }
    check.reject("collective exact-nucleus rejection", [&] { LAPW_xc_field_map bad(density, bad_points); });
    bad_points = points;
    if (comm.rank() == 0) {
        bad_points.push_back({std::numeric_limits<double>::quiet_NaN(), 0, 0});
    }
    check.reject("collective nonfinite-coordinate rejection", [&] { LAPW_xc_field_map bad(density, bad_points); });
    for (bool tau_field : {false, true}) {
        auto& f      = tau_field ? density.kinetic_density().scalar().mt()[0] : density.rho().mt()[0];
        double saved = f(0, 0);
        if (comm.rank() == 0) {
            f(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("collective nonfinite physical-field rejection", [&] { LAPW_xc_field_map bad(density, points); });
        f(0, 0) = saved;
    }
    auto& atom = ctx.unit_cell().atom(0);
    std::vector<double> radial(atom.type().num_mt_points());
    for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
        radial[ir] =
                density.lapw_basis_sptr()->radial_basis_for_class(atom.symmetry_class().id()).radial_function(ir, 0);
    }
    auto changed = radial;
    for (auto& value : changed) {
        value *= 1.7;
    }
    density.lapw_basis_sptr()->radial_basis_for_class(atom.symmetry_class().id()).radial_function(0, changed);
    density.density_matrix(0).zero();
    auto unchanged = LAPW_xc_field_map(density, points).sample(rho, tau);
    error          = 0;
    for (size_t i = 0; i < fields.rho.size(); i++) {
        error = std::max(
                {error, std::abs(unchanged.rho[i] - fields.rho[i]), std::abs(unchanged.tau[i] - fields.tau[i])});
    }
    for (size_t i = 0; i < fields.gradient.size(); i++) {
        error = std::max(error, std::abs(unchanged.gradient[i] - fields.gradient[i]));
    }
    check("physical fields survive radial-basis and density-matrix updates", error, 2e-12);
    density.lapw_basis_sptr()->radial_basis_for_class(atom.symmetry_class().id()).radial_function(0, radial);
    set_dm(0);

    // Physical fields, including the core contribution, must be mixed and restarted as fields.
    Density saved(ctx, true), restored(ctx, true);
    copy(density, saved);
    auto state = [&](Density const& d) {
        std::vector<double> values;
        for (auto const* f : {static_cast<Field4D const*>(&d), &d.kinetic_density()}) {
            for (int j = 0; j < ctx.num_spins(); j++) {
                for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                    values.push_back(f->component(j).rg().value(ir));
                }
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto const& mt = f->component(j).mt()[ia];
                    values.insert(values.end(), mt.begin(), mt.end());
                }
            }
        }
        return values;
    };
    auto compare = [&](std::vector<double> const& a, std::vector<double> const& b) {
        double delta{0};
        for (size_t i = 0; i < a.size(); i++) {
            delta = std::max(delta, std::abs(a[i] - b[i]));
        }
        return delta;
    };
    check("copy preserves physical MT tau and core metadata", compare(state(density), state(saved)), 0);
    check("copy preserves solved core tau",
          std::abs(saved.core_kinetic_density(0)[20] - density.core_kinetic_density(0)[20]), 0);
    Density without_tau(ctx);
    check.reject("copy rejects incompatible tau storage", [&] { copy(density, without_tau); });
    density.mixer_init(ctx.cfg().mixer());
    auto previous = state(density);
    for (auto* f : {static_cast<Field4D*>(&density), &density.kinetic_density()}) {
        for (int j = 0; j < ctx.num_spins(); j++) {
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                f->component(j).rg().value(ir) = 0.04 * (j + 1) + 0.7 * f->component(j).rg().value(ir);
            }
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                auto& mt = f->component(j).mt()[ia];
                for (size_t i = 0; i < mt.size(); i++) {
                    mt[i] = 0.7 * mt[i] + 0.001 * (j + 1) * std::cos(0.17 * i);
                }
            }
        }
        f->fft_transform(-1);
    }
    auto expected = state(density);
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] = 0.4 * expected[i] + 0.6 * previous[i];
    }
    density.mix();
    check("SCF mixes charge and tau on every owned and replicated sphere", compare(state(density), expected), 2e-12);
    auto mixed_points        = LAPW_xc_field_map(density, points).sample(rho, tau);
    unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    comm.bcast(&stamp, 1, 0);
    auto path = std::filesystem::temp_directory_path() / ("sirius-lapw-density-" + std::to_string(stamp) + ".h5");
    ctx.create_storage_file(path.string());
    density.save(path.string());
    comm.barrier();
    restored.load(path.string());
    check("restart preserves physical charge and tau", compare(state(density), state(restored)), 2e-12);
    auto restart_points = LAPW_xc_field_map(restored, points).sample(rho, tau);
    error               = 0;
    for (size_t i = 0; i < mixed_points.rho.size(); i++) {
        error = std::max({error, std::abs(mixed_points.rho[i] - restart_points.rho[i]),
                          std::abs(mixed_points.tau[i] - restart_points.tau[i])});
    }
    for (size_t i = 0; i < mixed_points.gradient.size(); i++) {
        error = std::max(error, std::abs(mixed_points.gradient[i] - restart_points.gradient[i]));
    }
    check("restarted point fields need neither a radial-basis rebuild nor a new core solve", error, 0);
    comm.barrier();
    if (comm.rank() == 0) {
        std::filesystem::remove(path);
    }

    density.zero();
    for (int j = 0; j < ctx.num_spins(); j++) {
        double value = j == 0 ? 0.8 : 0.2;
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            density.component(j).rg().value(ir) = value;
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                density.component(j).mt()[ia](0, ir) = value / y00;
            }
        }
    }
    density.initial_kinetic_density();
    error = 0;
    double total{0}, difference{0};
    for (int s = 0; s < ctx.num_spins(); s++) {
        double n = ctx.num_spins() == 1 ? 0.8 : 0.5 * (0.8 + (1 - 2 * s) * 0.2);
        double t = 0.3 * std::pow(3 * pi * pi * ctx.num_spins(), 2.0 / 3.0) * std::pow(n, 5.0 / 3.0);
        total += t;
        difference += (1 - 2 * s) * t;
    }
    for (int j = 0; j < ctx.num_spins(); j++) {
        double value  = j == 0 ? total : difference;
        auto const& f = density.kinetic_density().component(j);
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(f.rg().value(ir) - value));
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                    error = std::max(error, std::abs(f.mt()[ia](lm, ir) - (lm == 0 ? value / y00 : 0)));
                }
            }
        }
    }
    check("initial tau populates both real-space domains and spin channels", error, 2e-12);
    copy(saved, density);

    K_point_set density_ks(ctx);
    std::array<r3::vector<double>, 2> kpoints{{{0.13, -0.17, 0.07}, {-0.21, 0.11, 0.16}}};
    std::array<double, 2> weights{0.3, 0.7};
    for (int ik = 0; ik < 2; ik++) {
        density_ks.add_kpoint(kpoints[ik], weights[ik]);
    }
    density_ks.initialize();
    auto amplitude = [](int ik, int s) { return 1 + 0.1 * ik + 0.2 * s; };
    for (int ik = 0; ik < 2; ik++) {
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ib = 0; ib < ctx.num_bands(); ib++) {
                density_ks.get<double>(ik)->band_occupancy(ib, s, ib == 0 ? 2.0 / ctx.num_spins() : 0);
            }
        }
    }
    for (auto it : density_ks.spl_num_kpoints()) {
        auto kp   = density_ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        psi.zero(memory_t::host);
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                if (g[0] == 1 && g[1] == 0 && g[2] == 0) {
                    psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0)) = amplitude(it.i, s);
                }
            }
            for (auto ia : psi.spl_num_atoms()) {
                auto local_atom = atom_index_t::local(ia.li);
                for (int xi = 0; xi < ctx.unit_cell().atom(ia.i).type().mt_basis_size(); xi++) {
                    psi.mt_coeffs(xi, local_atom, wf::spin_index(s), wf::band_index(0)) =
                            amplitude(it.i, s) * orbital(xi, ia.i, s, false);
                }
            }
        }
    }
    for (bool add_core : {false, true}) {
        density.generate<double>(density_ks, false, add_core, true);
        error = 0;
        std::vector<double> norms(ctx.num_spins()), smooth_tau(ctx.num_spins());
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ik = 0; ik < 2; ik++) {
                double weight = weights[ik] * 2.0 / ctx.num_spins() * std::pow(amplitude(ik, s), 2);
                auto q        = r3::dot(ctx.unit_cell().reciprocal_lattice_vectors(),
                                        kpoints[ik] + r3::vector<double>(1, 0, 0));
                norms[s] += weight;
                smooth_tau[s] += weight * 0.5 * r3::dot(q, q) / ctx.unit_cell().omega();
            }
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& dm = density.density_matrix(ia);
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (int j = 0; j < static_cast<int>(dm.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(dm.size(0)); i++) {
                        auto target = norms[s] * std::conj(orbital(i, ia, s, false)) * orbital(j, ia, s, false);
                        error       = std::max(error, std::abs(dm(i, j, s) - target));
                    }
                }
            }
            auto local = LAPW_xc_atom(ctx.unit_cell().atom(ia),
                                      density.lapw_basis().radial_basis_for_class(
                                              (ctx.unit_cell().atom(ia)).symmetry_class().id()),
                                      ctx.lmax_rho())
                                 .fields(dm);
            for (int j = 0; j < ctx.num_spins(); j++) {
                for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                    for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                        double r = local[0][0](lm, ir), t = local[1][0](lm, ir);
                        if (ctx.num_spins() == 2) {
                            r += (1 - 2 * j) * local[0][1](lm, ir);
                            t += (1 - 2 * j) * local[1][1](lm, ir);
                        }
                        if (add_core && lm == 0 && j == 0) {
                            r += density.core_charge_density(ia)[ir] / y00;
                            t += density.core_kinetic_density(ia)[ir] / y00;
                        }
                        error = std::max({error, std::abs(density.component(j).mt()[ia](lm, ir) - r),
                                          std::abs(density.kinetic_density().component(j).mt()[ia](lm, ir) - t)});
                    }
                }
            }
        }
        for (int j = 0; j < ctx.num_spins(); j++) {
            double value = smooth_tau[0] + (ctx.num_spins() == 2 ? (1 - 2 * j) * smooth_tau[1] : 0);
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) - value));
            }
        }
        check(add_core ? "orbital density update stores valence plus core rho/tau"
                       : "orbital density update stores valence rho/tau",
              error, 3e-12);
    }
    copy(saved, density);
    if (comm.rank() == 0) {
        rho[0].f_pw_local(0) = std::numeric_limits<double>::quiet_NaN();
    }
    check.reject("collective nonfinite-PW rejection", [&] { map.sample(rho, tau); });
    set_pw(0);
    std::vector<Smooth_periodic_function<double>> empty;
    check.reject("collective wrong-spin rejection", [&] { map.sample(comm.rank() == 0 ? empty : rho, tau); });
    check.reject("geometry collective wrong-spin rejection",
                 [&] { map.geometry_adjoint(comm.rank() == 0 ? empty : rho, tau, derivatives); });
    std::vector<r3::vector<double>> zero_points(points.size());
    mdarray<double, 2> zero_atoms({3, ctx.unit_cell().num_atoms()});
    zero_atoms.zero();
    r3::matrix<double> zero_lattice;
    auto bad_directions = zero_points;
    if (comm.rank() == 0) {
        bad_directions.emplace_back();
    }
    check.reject("geometry response collective wrong-buffer rejection",
                 [&] { map.geometry_response(rho, tau, bad_directions, zero_atoms, zero_lattice); });
    check.reject("geometry transpose response collective wrong-buffer rejection",
                 [&] { map.geometry_adjoint_response(derivatives, bad_directions, zero_atoms, zero_lattice); });
    if (comm.size() > 1) {
        zero_atoms(0, 0) = comm.rank() == 0 ? 1 : 0;
        check.reject("geometry response inconsistent atom-direction rejection",
                     [&] { map.geometry_response(rho, tau, zero_points, zero_atoms, zero_lattice); });
        zero_atoms.zero();
    }
    xc_point_fields_t bad(points.size() + (comm.rank() == 0 ? 1 : 0), ctx.num_spins());
    bad.rho.zero();
    bad.tau.zero();
    bad.gradient.zero();
    check.reject("collective wrong-buffer rejection", [&] { map.adjoint(bad); });
    check.reject("geometry collective wrong-buffer rejection", [&] { map.geometry_adjoint(rho, tau, bad); });
    if (comm.rank() == 0) {
        derivatives.gradient(0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    check.reject("collective nonfinite-covector rejection", [&] { map.adjoint(derivatives); });
    check.reject("geometry collective nonfinite-covector rejection",
                 [&] { map.geometry_adjoint(rho, tau, derivatives); });
    return failures;
}

int
test_dirac_native_hamiltonian(int mag, bool independent_kpoints, bool gpu)
{
    auto const& comm                     = mpi::Communicator::world();
    auto config                          = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
        "pw_cutoff": 4.0, "gk_cutoff": 1.0, "num_bands": 4,
        "xc_functionals": ["XC_LDA_X"], "core_relativity": "none", "valence_relativity": "dirac",
        "auto_rmt": 0, "lmax_apw": 2, "lmax_rho": 4, "lmax_pot": 4
    }, "control": {"processing_unit": "CPU", "verbosity": 0, "use_second_variation": false},
    "iterative_solver": {"type": "exact"}})"_json;
    config["parameters"]["num_mag_dims"] = mag;
    config["control"]["mpi_grid_dims"]   = {independent_kpoints ? 1 : comm.size(), 1};
    config["control"]["processing_unit"] = gpu ? "GPU" : "CPU";
    Simulation_context ctx(config);
    auto& cell = ctx.unit_cell();
    cell.set_lattice_vectors({{8, 0.7, 0.2}, {0, 9, 0.4}, {0, 0, 10}});
    for (int z : {1, 2}) {
        auto& type = cell.add_atom_type(z == 1 ? "H" : "He");
        type.zn(z);
        type.set_radial_grid(radial_grid_t::power, 301, 1e-6, 1.5, 2.0);
        auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 1000, 1e-7, 20.0, 3.0);
        type.free_atom_density(std::vector<double>(free_grid.num_points(), 0));
        type.set_free_atom_radial_grid(std::move(free_grid));
        for (int l : {0, 1, 2}) {
            for (int order : {0, 1}) {
                type.add_aw_descriptor(l + 1, l, -0.3, order, 0);
            }
        }
        for (int order : {0, 1, 2}) {
            type.add_lo_descriptor(0, 2, 0, -0.2, order, 0);
        }
    }
    cell.add_atom("H", {0.15, 0.2, 0.25});
    cell.add_atom("He", {0.65, 0.7, 0.75});
    ctx.initialize();
    int failures{0};
    Test_check check(failures, comm,
                     "Native Dirac H/O mag=" + std::to_string(mag) +
                             " independent k-points=" + std::to_string(independent_kpoints));
    check("one coherent, single-occupancy Dirac band",
          std::abs(ctx.num_bands() - 4) + std::abs(ctx.num_spinors() - 1) + std::abs(ctx.max_occupancy() - 1) +
                  std::abs(ctx.num_mag_comp() - 1),
          0);
    Potential plain(ctx), full(ctx);
    for (auto* p : {&plain, &full}) {
        p->zero();
        for (int s = 0; s <= mag; s++) {
            for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                p->component(s).rg().f_pw_local(ig) =
                        fourier_coefficient(ctx.gvec().gvec(gvec_index_t::local(ig)), s, false, false);
            }
            for (auto it : cell.spl_num_atoms()) {
                auto const& atom = cell.atom(it.i);
                auto& field      = p->component(s).mt()[it.i];
                for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                    double r     = atom.type().radial_grid(ir);
                    field(0, ir) = (s == 0 ? -atom.type().zn() / r + 0.1 * std::exp(-r) : 0.03 * s) / y00;
                    field(3, ir) = 0.019 * (s + 1) * r * std::exp(-r);
                }
            }
            p->component(s).mt().sync(cell.spl_num_atoms());
        }
        p->fft_transform(1);
    }
    auto radial_basis = plain.create_lapw_basis();
    lapw_xc_field_adjoint_t data;
    int nf = mag == 3 ? 4 : ctx.num_spins();
    for (int s = 0; s < nf; s++) {
        data.rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        data.tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            auto g                         = ctx.gvec().gvec(gvec_index_t::local(ig));
            data.rho.back().f_pw_local(ig) = 0.17 * fourier_coefficient(g, s, false, false);
            data.tau.back().f_pw_local(ig) = 0.23 * fourier_coefficient(g, s, true, false);
        }
        data.rho.back().fft_transform(1);
        data.tau.back().fft_transform(1);
    }
    data.local.resize(cell.num_atoms());
    data.core.resize(cell.num_atoms());
    for (int ia = 0; ia < cell.num_atoms(); ia++) {
        auto const& grid = cell.atom(ia).type().radial_grid();
        for (int part : {0, 1}) {
            for (int s = 0; s < nf; s++) {
                data.local[ia][part].emplace_back(ctx.lmmax_rho(), grid);
                auto& v = data.local[ia][part].back();
                v.zero();
                for (int ir : {17, 93, 218}) {
                    for (int lm : {0, 2, 6}) {
                        v(lm, ir) = 0.007 * (ia + 1) * (part + 1) * (s + 1) * (lm == 2 ? -1 : 1);
                    }
                }
            }
            data.core[ia][part].resize(grid.num_points());
            for (int ir = 0; ir < grid.num_points(); ir++) {
                for (int s = 0; s < (mag == 3 ? 1 : nf); s++) {
                    data.core[ia][part][ir] += data.local[ia][part][s](0, ir) / ((mag == 3 ? 1 : nf) * y00);
                }
            }
        }
    }
    full.set_lapw_xc_derivatives(std::move(data));
    auto const& d = *full.lapw_xc_derivatives();
    Hamiltonian0<double> h0(plain, radial_basis), h1(full, radial_basis);
    check.reject("scalar local operator is not a Dirac operator", [&] { h1.local_op(); });
    check.reject("scalar PW fields cannot overwrite Dirac fields", [&] { h1.generate_pw_coefs(full); });
    check.reject("scalar spin blocks rejected for Dirac atoms", [&] { h1.hmt_component(0, 0, 0, 1); });
    // The dense FFT spans the context, even when k-points use independent band communicators.
    auto const& fft = ctx.spfft<double>();
    int grid_size   = fft::spfft_grid_size(fft);
    auto gather     = [&](Smooth_periodic_function<double> const& field) {
        std::vector<double> values(grid_size);
        mpi::Communicator(fft.communicator())
                .allgather(field.values().at(memory_t::host), values.data(), fft.local_slice_size(),
                           fft.dim_x() * fft.dim_y() * fft.local_z_offset());
        return values;
    };
    std::vector<double> theta(grid_size), local_theta(fft.local_slice_size());
    for (int ir = 0; ir < fft.local_slice_size(); ir++) {
        local_theta[ir] = ctx.theta(ir);
    }
    mpi::Communicator(fft.communicator())
            .allgather(local_theta.data(), theta.data(), fft.local_slice_size(),
                       fft.dim_x() * fft.dim_y() * fft.local_z_offset());
    std::vector<std::vector<double>> physical, rho_xc, tau_xc;
    for (int s = 0; s <= mag; s++) {
        physical.push_back(gather(plain.component(s).rg()));
    }
    for (int s = 0; s < nf; s++) {
        rho_xc.push_back(gather(d.rho[s]));
        tau_xc.push_back(gather(d.tau[s]));
    }
    K_point_set ks(ctx);
    ks.add_kpoint({0.12, -0.06, 0.04}, 1);
    ks.initialize();
    double matrix_error{0}, energy_error{0}, hermitian_error{0};
    double eigen_error{0}, norm_error{0}, matching_error{0};
    std::array<double, 4> occupations{0.8, 0.7, 0.9, 0.6};
    std::array<std::vector<std::vector<double>>, 2> expected_fields;
    for (auto& part : expected_fields) {
        part.resize(mag + 1, std::vector<double>(grid_size));
    }
    std::vector<mdarray<std::complex<double>, 3>> expected_dm;
    for (int ia = 0; ia < cell.num_atoms(); ia++) {
        int nm = cell.atom(ia).mt_basis_size();
        expected_dm.emplace_back(std::array<index_range, 3>{nm, nm, 1});
        expected_dm.back().zero();
    }
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        Dirac_lapw_basis basis(cell, kp->gkvec(), *radial_basis);
        int n = basis.size(), ng = kp->num_gkvec(), bs = ctx.cyclic_block_size();
        la::dmatrix<std::complex<double>> h(n, n, ctx.blacs_grid(), bs, bs), o(n, n, ctx.blacs_grid(), bs, bs);
        auto hk = h1(*kp);
        hk.set_dirac_h_o(h, o);
        Test_check kcheck(failures, kp->comm());
        kcheck.reject("scalar first variation rejected", [&] { hk.set_fv_h_o(0, h, o); });
        kcheck.reject("aliased Dirac H/O rejected", [&] { hk.set_dirac_h_o(h, h); });
        la::dmatrix<std::complex<double>> wrong(n - 1, n - 1, ctx.blacs_grid(), bs, bs);
        kcheck.reject("scalar-size Dirac matrix rejected", [&] { hk.set_dirac_h_o(wrong, o); });
        auto dense_h = h.get_full_matrix(), dense_o = o.get_full_matrix();
        auto expected = h1.dirac_matrices(kp->gkvec());
        auto base     = h0.dirac_matrices(kp->gkvec());
        auto bare = basis.hamiltonian(), metric = basis.overlap();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                matrix_error    = std::max({matrix_error, std::abs(dense_h(i, j) - expected[0](i, j)),
                                            std::abs(dense_o(i, j) - metric(i, j))});
                hermitian_error = std::max({hermitian_error, std::abs(dense_h(i, j) - std::conj(dense_h(j, i))),
                                            std::abs(dense_o(i, j) - std::conj(dense_o(j, i)))});
            }
        }
        diagonalize_fp(hk, *kp, 1e-12);
        auto& psi = kp->spinor_wave_functions();
        mdarray<std::complex<double>, 2> coefficients({n, ctx.num_bands()});
        coefficients.zero();
        for (int b = 0; b < ctx.num_bands(); b++) {
            kp->band_occupancy(b, 0, occupations[b]);
            for (int s : {0, 1}) {
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    coefficients(s * ng + kp->gkvec().offset() + ig, b) =
                            psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b));
                }
            }
            for (auto atom : psi.spl_num_atoms()) {
                int offset = 2 * ng;
                for (int ia = 0; ia < atom.i; ia++) {
                    offset += cell.atom(ia).mt_lo_basis_size();
                }
                int naw = cell.atom(atom.i).mt_aw_basis_size();
                for (int lo = 0; lo < cell.atom(atom.i).mt_lo_basis_size(); lo++) {
                    coefficients(offset + lo, b) =
                            psi.mt_coeffs(naw + lo, atom.li, wf::spin_index(0), wf::band_index(b));
                }
            }
        }
        kp->comm().allreduce(coefficients.at(memory_t::host), coefficients.size());
        mdarray<std::complex<double>, 2> wrong_vectors({n - 1, ctx.num_bands()});
        kcheck.reject("scalar-size Dirac eigenvectors rejected",
                      [&] { kp->generate_dirac_wave_functions(wrong_vectors, *radial_basis); });
        auto finite        = coefficients(0, 0);
        coefficients(0, 0) = std::numeric_limits<double>::quiet_NaN();
        kcheck.reject("nonfinite Dirac eigenvectors rejected",
                      [&] { kp->generate_dirac_wave_functions(coefficients, *radial_basis); });
        coefficients(0, 0) = finite;
        for (int b = 0; b < ctx.num_bands(); b++) {
            std::complex<double> norm{0};
            for (int i = 0; i < n; i++) {
                std::complex<double> residual{0};
                for (int j = 0; j < n; j++) {
                    norm += std::conj(coefficients(i, b)) * dense_o(i, j) * coefficients(j, b);
                    residual += (dense_h(i, j) - kp->band_energy(b, 0) * dense_o(i, j)) * coefficients(j, b);
                }
                eigen_error = std::max(eigen_error, std::abs(residual));
            }
            norm_error = std::max(norm_error, std::abs(norm - 1.0));
            for (int ia = 0; ia < cell.num_atoms(); ia++) {
                auto const& a = basis.atom_coefficients(ia);
                int nm        = cell.atom(ia).mt_basis_size();
                std::vector<std::complex<double>> local(nm);
                for (int i = 0; i < nm; i++) {
                    for (int j = 0; j < n; j++) {
                        local[i] += a(i, j) * coefficients(j, b);
                    }
                }
                for (int j = 0; j < nm; j++) {
                    for (int i = 0; i < nm; i++) {
                        expected_dm[ia](i, j, 0) += kp->weight() * occupations[b] * std::conj(local[i]) * local[j] /
                                                    double(kp->comm().size());
                    }
                }
                for (auto atom : psi.spl_num_atoms()) {
                    if (atom.i == ia) {
                        for (int i = 0; i < nm; i++) {
                            matching_error = std::max(
                                    {matching_error,
                                     std::abs(psi.mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b)) -
                                              local[i]),
                                     std::abs(psi.mt_coeffs(i, atom.li, wf::spin_index(1), wf::band_index(b)))});
                        }
                    }
                }
            }
        }
        std::vector<std::complex<double>> c(n);
        for (int i = 0; i < n; i++) {
            c[i] = 0.4 * orbital(i, 0, 0, false);
        }
        // Independent physical real-space contraction, without using the assembled field matrices.
        std::array<double, 2> interstitial{};
        double mt_plain{0}, mt_xc{0};
        auto bilinear = [](auto const& psi) {
            std::array<double, 4> value{};
            for (int lower : {0, 1}) {
                auto up = psi[2 * lower], down = psi[2 * lower + 1], cross = std::conj(up) * down;
                double sign = lower ? -1 : 1;
                value[0] += std::norm(up) + std::norm(down);
                value[1] += sign * (std::norm(up) - std::norm(down));
                value[2] += sign * 2 * cross.real();
                value[3] += sign * 2 * cross.imag();
            }
            return value;
        };
        auto spin_field = [mag, nf](auto const& fields, int s) {
            return mag == 3 ? fields[s] : (nf == 1 ? fields[0] : 0.5 * (fields[0] + (1 - 2 * s) * fields[1]));
        };
        for (int ir = 0; ir < grid_size; ir++) {
            r3::vector<double> frac{double(ir % fft.dim_x()) / fft.dim_x(),
                                    double((ir / fft.dim_x()) % fft.dim_y()) / fft.dim_y(),
                                    double(ir / (fft.dim_x() * fft.dim_y())) / fft.dim_z()};
            auto point = cell.get_cartesian_coordinates(frac);
            std::array<std::complex<double>, 4> psi{};
            std::array<std::array<std::complex<double>, 4>, 3> gradient{};
            for (int ig = 0; ig < ng; ig++) {
                auto q     = kp->gkvec().gkvec_cart(gvec_index_t::global(ig));
                auto u     = Dirac_plane_wave(q).spinors();
                auto phase = std::polar(1 / std::sqrt(cell.omega()), r3::dot(q, point));
                for (int a = 0; a < 4; a++) {
                    auto v = phase * (c[ig] * u[0][a] + c[ng + ig] * u[1][a]);
                    psi[a] += v;
                    for (int x = 0; x < 3; x++) {
                        gradient[x][a] += std::complex<double>(0, q[x]) * v;
                    }
                }
            }
            auto rho = bilinear(psi);
            std::array<double, 4> tau{};
            for (int x = 0; x < 3; x++) {
                auto t = bilinear(gradient[x]);
                for (int s = 0; s < 4; s++) {
                    tau[s] += 0.5 * t[s];
                }
            }
            for (int s = 0; s <= mag; s++) {
                interstitial[0] += theta[ir] * physical[s][ir] * rho[s];
            }
            for (int s = 0; s < nf; s++) {
                interstitial[1] += rho_xc[s][ir] * spin_field(rho, s) + tau_xc[s][ir] * spin_field(tau, s);
            }
            for (int b = 0; b < ctx.num_bands(); b++) {
                std::array<std::complex<double>, 4> state{};
                std::array<std::array<std::complex<double>, 4>, 3> derivative{};
                for (int ig = 0; ig < ng; ig++) {
                    auto q     = kp->gkvec().gkvec_cart(gvec_index_t::global(ig));
                    auto u     = Dirac_plane_wave(q).spinors();
                    auto phase = std::polar(1 / std::sqrt(cell.omega()), r3::dot(q, point));
                    for (int a = 0; a < 4; a++) {
                        auto v = phase * (coefficients(ig, b) * u[0][a] + coefficients(ng + ig, b) * u[1][a]);
                        state[a] += v;
                        for (int x = 0; x < 3; x++) {
                            derivative[x][a] += std::complex<double>(0, q[x]) * v;
                        }
                    }
                }
                auto r = bilinear(state);
                for (int s = 0; s <= mag; s++) {
                    double t{0};
                    for (int x = 0; x < 3; x++) {
                        t += 0.5 * bilinear(derivative[x])[s];
                    }
                    double weight = kp->weight() * occupations[b] / kp->comm().size();
                    expected_fields[0][s][ir] += weight * r[s];
                    expected_fields[1][s][ir] += weight * t;
                }
            }
        }
        for (auto& value : interstitial) {
            value *= cell.omega() / grid_size;
        }
        for (int ia = 0; ia < cell.num_atoms(); ia++) {
            auto const& atom = cell.atom(ia);
            auto const& a    = basis.atom_coefficients(ia);
            int nb           = atom.type().mt_basis_size();
            std::vector<std::complex<double>> atomic(nb);
            for (int i = 0; i < nb; i++) {
                for (int j = 0; j < n; j++) {
                    atomic[i] += a(i, j) * c[j];
                }
            }
            mdarray<std::complex<double>, 3> dm({nb, nb, 1});
            for (int j = 0; j < nb; j++) {
                for (int i = 0; i < nb; i++) {
                    dm(i, j, 0) = std::conj(atomic[i]) * atomic[j];
                }
            }
            auto fields      = LAPW_xc_atom(atom, radial_basis->radial_basis_for_class((atom).symmetry_class().id()),
                                            ctx.lmax_rho())
                                       .fields(dm);
            auto const& grid = atom.type().radial_grid();
            Spline<double> integrand(grid);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                integrand(ir) = 0;
                for (int lm = 0; lm < ctx.lmmax_pot(); lm++) {
                    for (int s = 0; s <= mag; s++) {
                        double value = plain.component(s).mt()[ia](lm, ir);
                        if (s == 0 && lm == 0) {
                            value -= radial_basis->radial_basis_for_class(atom.symmetry_class().id())
                                             .spherical_potential()[ir] /
                                     y00;
                        }
                        integrand(ir) += value * fields[0][s](lm, ir);
                    }
                    for (int s = 0; s < nf; s++) {
                        for (int part : {0, 1}) {
                            std::array<double, 4> value{};
                            for (int j = 0; j < 4; j++) {
                                value[j] = fields[part][j](lm, ir);
                            }
                            mt_xc += d.local[ia][part][s](lm, ir) * spin_field(value, s);
                        }
                    }
                }
            }
            mt_plain += integrand.interpolate().integrate(2);
        }
        auto quadratic = [&](auto const& matrix) {
            std::complex<double> value{0};
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    value += std::conj(c[i]) * matrix(i, j) * c[j];
                }
            }
            return value.real();
        };
        energy_error =
                std::max({energy_error, std::abs(quadratic(base[0]) - quadratic(bare) - interstitial[0] - mt_plain),
                          std::abs(quadratic(dense_h) - quadratic(base[0]) - interstitial[1] - mt_xc)});
    }
    comm.allreduce<double, mpi::op_t::max>(&matrix_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&hermitian_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&energy_error, 1);
    check("native distributed Dirac H/O", matrix_error, 2e-13);
    check("native Hermitian Dirac matrices", hermitian_error, 2e-13);
    check("physical potential and raw XC energy contraction", energy_error, 2e-12);
    comm.allreduce<double, mpi::op_t::max>(&eigen_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&norm_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&matching_error, 1);
    check("native Dirac eigenvalue residual", eigen_error, 2e-10);
    check("physical Dirac eigenvector norm", norm_error, 2e-11);
    check("full-j matching without duplicated spinor coefficients", matching_error, 2e-12);
    ks.sync_band<double, sync_band_t::energy>();
    ks.sync_band<double, sync_band_t::occupancy>();
    for (auto& part : expected_fields) {
        for (auto& field : part) {
            comm.allreduce(field.data(), field.size());
        }
    }
    for (auto& dm : expected_dm) {
        comm.allreduce(dm.at(memory_t::host), dm.size());
    }
    Density density(ctx, true);
    density.set_lapw_basis(radial_basis);
    density.generate<double>(ks, false, false, true);
    double field_error{0}, dm_error{0}, mt_error{0};
    for (int s = 0; s <= mag; s++) {
        for (int ir = 0; ir < fft.local_slice_size(); ir++) {
            int global  = fft.dim_x() * fft.dim_y() * fft.local_z_offset() + ir;
            field_error = std::max(
                    {field_error, std::abs(density.component(s).rg().value(ir) - expected_fields[0][s][global]),
                     std::abs(density.kinetic_density().component(s).rg().value(ir) - expected_fields[1][s][global])});
        }
    }
    for (int ia = 0; ia < cell.num_atoms(); ia++) {
        auto const& dm = density.density_matrix(ia);
        for (size_t i = 0; i < dm.size(); i++) {
            dm_error = std::max(dm_error, std::abs(dm[i] - expected_dm[ia][i]));
        }
        auto fields =
                LAPW_xc_atom(cell.atom(ia), radial_basis->radial_basis_for_class((cell.atom(ia)).symmetry_class().id()),
                             ctx.lmax_rho())
                        .fields(expected_dm[ia]);
        for (int s = 0; s <= mag; s++) {
            auto const& rho = density.component(s).mt()[ia];
            auto const& tau = density.kinetic_density().component(s).mt()[ia];
            for (size_t i = 0; i < rho.size(); i++) {
                // Nuclear near-origin tau is large; bound each sample with both absolute and relative accuracy.
                mt_error = std::max({mt_error,
                                     std::abs(rho[i] - fields[0][s][i]) / std::max(1.0, std::abs(fields[0][s][i])),
                                     std::abs(tau[i] - fields[1][s][i]) / std::max(1.0, std::abs(fields[1][s][i]))});
            }
        }
    }
    comm.allreduce<double, mpi::op_t::max>(&field_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&dm_error, 1);
    comm.allreduce<double, mpi::op_t::max>(&mt_error, 1);
    check("physical four-component Dirac density and tau", field_error, 2e-12);
    check("intrinsic full-j occupied density matrix", dm_error, 2e-12);
    check("native full-j muffin-tin fields", mt_error, 2e-11);
    check("physical Dirac electron count", std::abs(density.rho().integrate().total - 3.0), 2e-10);
    auto checksum = [&] {
        std::vector<std::array<std::complex<double>, 4>> values;
        for (auto it : ks.spl_num_kpoints()) {
            auto& psi = ks.get<double>(it.i)->spinor_wave_functions();
            values.push_back({psi.checksum_pw(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx.num_bands())),
                              psi.checksum_pw(memory_t::host, wf::spin_index(1), wf::band_range(0, ctx.num_bands())),
                              psi.checksum_mt(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx.num_bands())),
                              psi.checksum_mt(memory_t::host, wf::spin_index(1), wf::band_range(0, ctx.num_bands()))});
        }
        return values;
    };
    auto saved               = checksum();
    unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    comm.bcast(&stamp, 1, 0);
    auto path =
            std::filesystem::temp_directory_path() / ("sirius-dirac-wave-functions-" + std::to_string(stamp) + ".h5");
    ks.save(path.string());
    for (auto it : ks.spl_num_kpoints()) {
        for (int s : {0, 1}) {
            ks.get<double>(it.i)->spinor_wave_functions().zero(memory_t::host, wf::spin_index(s),
                                                               wf::band_range(0, ctx.num_bands()));
        }
    }
    ks.load(path.string());
    auto restored = checksum();
    double restart_error{0};
    for (size_t ik = 0; ik < saved.size(); ik++) {
        for (int s = 0; s < 4; s++) {
            restart_error = std::max(restart_error, std::abs(saved[ik][s] - restored[ik][s]));
        }
    }
    comm.allreduce<double, mpi::op_t::max>(&restart_error, 1);
    check("Dirac restart preserves both positive PW blocks and intrinsic MT coefficients", restart_error, 0);
    comm.barrier();
    if (comm.rank() == 0) {
        std::filesystem::remove(path);
    }
    ks.find_band_occupancies<double>();
    double electrons{0}, capacity_error{0};
    for (int ik = 0; ik < ks.num_kpoints(); ik++) {
        auto kp = ks.get<double>(ik);
        for (int b = 0; b < ctx.num_bands(); b++) {
            double f = kp->band_occupancy(b, 0);
            electrons += kp->weight() * f;
            capacity_error = std::max({capacity_error, -f, f - 1});
        }
    }
    check("Dirac Fermi occupations retain the prescribed electron count", std::abs(electrons - 3), 2e-10);
    check("each Dirac spinor carries at most one electron", capacity_error, 2e-12);
    comm.allreduce<int, mpi::op_t::max>(&failures, 1);
    return failures;
}

int
test_dirac_xc_atom()
{
    int failures{0};
    Test_check check(failures, mpi::Communicator::world(), "Dirac XC atom");
    Simulation_parameters parameters;
    parameters.valence_relativity("dirac");
    parameters.core_relativity("dirac");
    parameters.lmax_apw(1);
    parameters.lmax_rho(4);
    parameters.lmax_pot(4);
    parameters.cfg().parameters().auto_rmt(0);
    parameters.cfg().parameters().use_symmetry(false);
    Unit_cell cell(parameters, mpi::Communicator::world());
    cell.set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    auto& type = cell.add_atom_type("Ne");
    type.zn(10);
    type.set_radial_grid(radial_grid_t::power, 161, 1e-6, 1.8, 2.0);
    for (int l : {0, 1}) {
        for (int order : {0, 1}) {
            type.add_aw_descriptor(2, l, -0.3, order, 0);
        }
    }
    cell.add_atom("Ne", {0.23, 0.31, 0.43});
    cell.initialize();
    auto& atom       = cell.atom(0);
    auto const& grid = type.radial_grid();
    int n = type.indexb().size(), nr = grid.num_points(), lmmax = sf::lmmax(4);
    std::vector<double> potential(nr), direction(nr);
    for (int ir = 0; ir < nr; ir++) {
        potential[ir] = -10 / grid[ir] + 0.2 * std::exp(-grid[ir]);
        direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
    }
    LAPW_radial_basis radial_basis(cell, parameters.valence_relativity(), {potential});
    auto& cls     = radial_basis.radial_basis_for_class(0);
    auto generate = [&](double h) {
        auto v = potential;
        for (int ir = 0; ir < nr; ir++) {
            v[ir] += h * direction[ir];
        }
        cls.set_spherical_potential(v);
        radial_basis.generate_radial_functions();
    };
    generate(0);
    LAPW_xc_atom op(atom, cls, 4);
    std::vector<std::complex<double>> c(n), dc(n);
    for (int i = 0; i < n; i++) {
        c[i]  = orbital(i, 0, 0, false);
        dc[i] = orbital(i, 0, 0, true);
    }
    auto density_matrix = [&](auto const& coefficients) {
        mdarray<std::complex<double>, 3> dm({n, n, 1});
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                dm(i, j, 0) = std::conj(coefficients[i]) * coefficients[j];
            }
        }
        return dm;
    };
    auto dm     = density_matrix(c);
    auto fields = op.fields(dm);
    SHT sht(device_t::CPU, 6);
    std::array<std::vector<Spheric_function<function_domain_t::spatial, double>>, 2> values;
    for (int part : {0, 1}) {
        for (int s = 0; s < 4; s++) {
            values[part].push_back(transform(sht, fields[part][s]));
        }
    }
    auto bilinear = [](std::array<std::complex<double>, 4> const& psi) {
        std::array<double, 4> result{};
        for (int lower : {0, 1}) {
            auto up = psi[2 * lower], down = psi[2 * lower + 1], cross = std::conj(up) * down;
            double sign = lower ? -1 : 1;
            result[0] += std::norm(up) + std::norm(down);
            result[1] += sign * (std::norm(up) - std::norm(down));
            result[2] += sign * 2 * cross.real();
            result[3] += sign * 2 * cross.imag();
        }
        return result;
    };
    using Spinor = std::array<std::complex<double>, 4>;
    auto spinors = [&](int ir, int tp) {
        auto unit = sht.coord(tp), point = unit * grid[ir];
        std::vector<double> rlm(sf::lmmax(2));
        std::vector<std::complex<double>> ylm(rlm.size());
        sf::spherical_harmonics(2, point, rlm.data());
        SHT::convert(2, rlm.data(), ylm.data());
        mdarray<double, 2> grad_rlm({sf::lmmax(2), 3});
        sf::dRlm_dr(2, point, grad_rlm);
        std::array<std::vector<std::complex<double>>, 3> grad_ylm;
        for (int x = 0; x < 3; x++) {
            grad_ylm[x].resize(rlm.size());
            SHT::convert(2, &grad_rlm(0, x), grad_ylm[x].data());
        }
        std::vector<Spinor> psi(n);
        std::vector<std::array<Spinor, 3>> gradient(n);
        for (int i = 0; i < n; i++) {
            auto const& basis = type.indexb(i);
            for (int component = 0; component < 4; component++) {
                int lower     = component / 2;
                auto [lm, cg] = basis.spinor_component(component % 2, lower);
                if (lm < 0) {
                    continue;
                }
                auto coefficient  = cg * (lower ? std::complex<double>(0, -1) : 1.0);
                double radial     = cls.radial_function(ir, basis.idxrf, lower);
                double derivative = cls.radial_function_derivative(ir, basis.idxrf, lower) / grid[ir];
                psi[i][component] = coefficient * radial * std::conj(ylm[lm]);
                for (int x = 0; x < 3; x++) {
                    gradient[i][x][component] = coefficient * (derivative * unit[x] * std::conj(ylm[lm]) +
                                                               radial * std::conj(grad_ylm[x][lm]));
                }
            }
        }
        return std::pair{std::move(psi), std::move(gradient)};
    };
    std::array<double, 2> error{}, peak{};
    double lower_peak{0}, positivity_error{0};
    for (int ir : {nr / 8, nr / 2, nr - 1}) {
        for (int tp = 0; tp < sht.num_points(); tp++) {
            auto basis = spinors(ir, tp);
            Spinor psi{};
            std::array<Spinor, 3> gradient{};
            for (int i = 0; i < n; i++) {
                for (int component = 0; component < 4; component++) {
                    psi[component] += c[i] * basis.first[i][component];
                    for (int x = 0; x < 3; x++) {
                        gradient[x][component] += c[i] * basis.second[i][x][component];
                    }
                }
            }
            std::array<std::array<double, 4>, 2> reference{bilinear(psi), {}};
            for (int x = 0; x < 3; x++) {
                auto tau = bilinear(gradient[x]);
                for (int s = 0; s < 4; s++) {
                    reference[1][s] += 0.5 * tau[s];
                }
            }
            lower_peak = std::max(lower_peak, std::norm(psi[2]) + std::norm(psi[3]));
            for (int part : {0, 1}) {
                double moment{0};
                for (int s = 0; s < 4; s++) {
                    error[part] = std::max(error[part], std::abs(values[part][s](tp, ir) - reference[part][s]));
                    peak[part]  = std::max(peak[part], std::abs(reference[part][s]));
                    if (s) {
                        moment += std::pow(values[part][s](tp, ir), 2);
                    }
                }
                positivity_error = std::max(positivity_error, (std::sqrt(moment) - values[part][0](tp, ir)) /
                                                                      std::max(1e-12, reference[part][0]));
            }
        }
    }
    check("nonzero lower-component reference", lower_peak > 1e-8 ? 0 : 1, 0);
    check("four-component Cartesian charge and beta-Sigma magnetization", error[0] / peak[0], 2e-11);
    check("four-component Cartesian positive and spin kinetic density", error[1] / peak[1], 2e-11);
    check("positive spin-channel rho and tau", positivity_error, 2e-11);

    int nrf = type.indexr().size();
    std::vector<int> representatives;
    for (auto const& rf : type.indexr()) {
        representatives.push_back(type.indexb().index_by_j_mj_order(rf.am, 1, rf.order));
    }
    std::array<std::vector<Spline<std::complex<double>>>, 2> reference_matrices;
    for (auto& part : reference_matrices) {
        for (int i = 0; i < nrf * nrf; i++) {
            part.emplace_back(grid);
        }
    }
    auto inner_spinor = [](Spinor const& a, Spinor const& b) {
        std::complex<double> result{0};
        for (int component = 0; component < 4; component++) {
            result += std::conj(a[component]) * b[component];
        }
        return result;
    };
    for (int ir = 0; ir < nr; ir++) {
        std::array<std::vector<std::complex<double>>, 2> angular{std::vector<std::complex<double>>(nrf * nrf),
                                                                 std::vector<std::complex<double>>(nrf * nrf)};
        for (int tp = 0; tp < sht.num_points(); tp++) {
            auto basis = spinors(ir, tp);
            std::vector<Spinor> action(nrf);
            std::complex<double> imaginary(0, 1), minus_ic(0, -speed_of_light);
            for (int j = 0; j < nrf; j++) {
                auto const& psi = basis.first[representatives[j]];
                auto const& g   = basis.second[representatives[j]];
                action[j]       = {potential[ir] * psi[0] + minus_ic * (g[0][3] - imaginary * g[1][3] + g[2][2]),
                                   potential[ir] * psi[1] + minus_ic * (g[0][2] + imaginary * g[1][2] - g[2][3]),
                                   (potential[ir] - 2 * speed_of_light * speed_of_light) * psi[2] +
                                           minus_ic * (g[0][1] - imaginary * g[1][1] + g[2][0]),
                                   (potential[ir] - 2 * speed_of_light * speed_of_light) * psi[3] +
                                           minus_ic * (g[0][0] + imaginary * g[1][0] - g[2][1])};
            }
            for (int j = 0; j < nrf; j++) {
                for (int i = 0; i < nrf; i++) {
                    auto const& left  = basis.first[representatives[i]];
                    auto const& right = basis.first[representatives[j]];
                    double weight     = fourpi * sht.weight(tp);
                    angular[0][i + nrf * j] +=
                            weight * 0.5 * (inner_spinor(left, action[j]) + inner_spinor(action[i], right));
                    angular[1][i + nrf * j] += weight * inner_spinor(left, right);
                }
            }
        }
        for (int part : {0, 1}) {
            for (int i = 0; i < nrf * nrf; i++) {
                reference_matrices[part][i](ir) = angular[part][i];
            }
        }
    }
    auto spherical = cls.dirac_spherical_matrices();
    for (int part : {0, 1}) {
        double matrix_error{0}, matrix_peak{0};
        for (int i = 0; i < nrf * nrf; i++) {
            auto reference = reference_matrices[part][i].interpolate().integrate(2);
            matrix_error   = std::max(matrix_error, std::abs(spherical[part][i] - reference));
            matrix_peak    = std::max(matrix_peak, std::abs(reference));
        }
        check(part ? "Cartesian four-component spherical overlap" : "Cartesian Dirac alpha.p spherical Hamiltonian",
              matrix_error / matrix_peak, 2e-11);
    }

    std::array<std::vector<Flm>, 2> covectors;
    for (int part : {0, 1}) {
        for (int s = 0; s < 4; s++) {
            covectors[part].emplace_back(lmmax, grid);
            for (int ir = 0; ir < nr; ir++) {
                for (int lm = 0; lm < lmmax; lm++) {
                    double weight                  = std::pow(grid[ir], 3) / (nr * lmmax);
                    covectors[part].back()(lm, ir) = weight * (part ? 0.01 * std::cos(0.3 * ir + 0.5 * lm + s)
                                                                    : 0.02 * std::sin(ir + 0.7 * lm + s));
                }
            }
        }
    }
    auto contraction = [&](auto const& f) {
        double result{0};
        for (int part : {0, 1}) {
            for (int s = 0; s < 4; s++) {
                for (size_t i = 0; i < f[part][s].size(); i++) {
                    result += f[part][s][i] * covectors[part][s][i];
                }
            }
        }
        return result;
    };
    std::vector<mdarray<std::complex<double>, 2>> matrices;
    mdarray<std::complex<double>, 2> matrix({n, n});
    matrix.zero();
    double trace{0}, hermitian_error{0};
    for (int s = 0; s < 4; s++) {
        matrices.push_back(op.matrix_elements(covectors[0][s], covectors[1][s], nullptr, s));
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                auto value = matrices.back()(i, j);
                matrix(i, j) += value;
                trace += std::real(dm(i, j, 0) * value);
                hermitian_error = std::max(hermitian_error, std::abs(value - std::conj(matrices.back()(j, i))));
            }
        }
    }
    double moment = contraction(fields), dmoment{0};
    // Keep the nonlinear feedback comparable to the linear term despite quadrature cancellation.
    double curvature       = 1 / std::max(1e-8, std::abs(moment));
    auto energy_derivative = [&](double value) { return 2 + curvature * value; };
    std::vector<std::complex<double>> action(n), hessian(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            action[i] += matrix(i, j) * c[j];
        }
        dmoment += 2 * std::real(std::conj(dc[i]) * action[i]);
    }
    for (int i = 0; i < n; i++) {
        hessian[i] = curvature * dmoment * action[i];
        for (int j = 0; j < n; j++) {
            hessian[i] += energy_derivative(moment) * matrix(i, j) * dc[j];
        }
    }
    check("Dirac field/matrix duality", std::abs(moment - trace), 2e-13);
    check("Dirac covector matrices Hermitian", hermitian_error, 1e-14);
    check("nonzero Dirac XC orbital derivative", std::abs(energy_derivative(moment) * dmoment) > 1e-6 ? 0 : 1, 0);
    double nonlinear_peak{0};
    for (auto value : action) {
        nonlinear_peak = std::max(nonlinear_peak, std::abs(curvature * dmoment * value));
    }
    check("nonzero Dirac nonlocal Hessian feedback", nonlinear_peak > 1e-8 ? 0 : 1, 0);
    for (double h : {1e-4, 5e-5}) {
        auto energy = [&](double step) {
            auto shifted = c;
            for (int i = 0; i < n; i++) {
                shifted[i] += step * dc[i];
            }
            return std::pair{contraction(op.fields(density_matrix(shifted))), shifted};
        };
        auto plus = energy(h), minus = energy(-h);
        double fd = ((2 * plus.first + 0.5 * curvature * std::pow(plus.first, 2)) -
                     (2 * minus.first + 0.5 * curvature * std::pow(minus.first, 2))) /
                    (2 * h);
        check("Dirac nonlocal orbital derivative", std::abs(fd - energy_derivative(moment) * dmoment), 2e-9);
        double hvp_error{0};
        for (int i = 0; i < n; i++) {
            std::complex<double> fd_gradient{0};
            for (int j = 0; j < n; j++) {
                fd_gradient += matrix(i, j) *
                               (energy_derivative(plus.first) * plus.second[j] -
                                energy_derivative(minus.first) * minus.second[j]) /
                               (2 * h);
            }
            hvp_error = std::max(hvp_error, std::abs(fd_gradient - hessian[i]));
        }
        check("Dirac nonlocal orbital Hessian direction", hvp_error, 2e-9);
    }
    auto response = cls.radial_basis_potential_response(direction);
    auto dfields  = op.fields(dm, &response);
    std::vector<mdarray<std::complex<double>, 2>> dmatrices;
    for (int s = 0; s < 4; s++) {
        dmatrices.push_back(op.matrix_elements(covectors[0][s], covectors[1][s], &response, s));
    }
    auto sample = [&](double h) {
        generate(h);
        LAPW_xc_atom shifted(atom, cls, 4);
        std::vector<mdarray<std::complex<double>, 2>> m;
        for (int s = 0; s < 4; s++) {
            m.push_back(shifted.matrix_elements(covectors[0][s], covectors[1][s], nullptr, s));
        }
        return std::pair{shifted.fields(dm), std::move(m)};
    };
    for (double h : {1e-3, 5e-4}) {
        auto plus = sample(h), minus = sample(-h);
        std::array<double, 3> derivative_error{}, derivative_peak{};
        for (int s = 0; s < 4; s++) {
            for (int part : {0, 1}) {
                for (size_t i = 0; i < dfields[part][s].size(); i++) {
                    derivative_error[part] =
                            std::max(derivative_error[part],
                                     std::abs(dfields[part][s][i] -
                                              (plus.first[part][s][i] - minus.first[part][s][i]) / (2 * h)));
                    derivative_peak[part] = std::max(derivative_peak[part], std::abs(dfields[part][s][i]));
                }
            }
            for (size_t i = 0; i < dmatrices[s].size(); i++) {
                derivative_error[2] =
                        std::max(derivative_error[2],
                                 std::abs(dmatrices[s][i] - (plus.second[s][i] - minus.second[s][i]) / (2 * h)));
                derivative_peak[2] = std::max(derivative_peak[2], std::abs(dmatrices[s][i]));
            }
        }
        for (int part : {0, 1, 2}) {
            check("nonzero Dirac radial field/matrix response", derivative_peak[part] > 1e-8 ? 0 : 1, 0);
            check(part == 0 ? "Dirac rho radial response"
                            : (part == 1 ? "Dirac tau radial response" : "Dirac XC matrix radial response"),
                  derivative_error[part] / derivative_peak[part], 2e-6);
        }
    }
    mdarray<std::complex<double>, 3> bad_dm({n, n, 2});
    bad_dm.zero();
    check.reject("scalar spin blocks rejected for full-j orbitals", [&] { op.fields(bad_dm); });
    return failures;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    if (argc > 1 && std::string(argv[1]) == "dirac-core") {
        result = call_test("Dirac core field sampling", test_dirac_core_map);
        sirius::finalize();
        return result;
    }
    if (argc > 1 && std::string(argv[1]) == "dirac-atom") {
        result = call_test("Dirac XC atom", test_dirac_xc_atom);
        sirius::finalize();
        return result;
    }
    if (argc > 1 && std::string(argv[1]) == "dirac-native") {
        result = call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 3, false, false);
        result += call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 1, true, false);
        result += call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 0, false, false);
        sirius::finalize();
        return result;
    }
    if (argc > 1 && std::string(argv[1]) == "dirac-native-gpu") {
        result = call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 3, false, true);
        sirius::finalize();
        return result;
    }
    if (argc > 1 && std::string(argv[1]) == "spinor") {
        result = call_test("LAPW spinor XC field map", test_lapw_map, 3, false, true);
        sirius::finalize();
        return result;
    }
    // Pairwise coverage of spin, FFT layout and first/second variation.
    for (int mag : {0, 1}) {
        for (bool second : {false, true}) {
            result += call_test("LAPW XC field map", test_lapw_map, mag, mag != second, second);
        }
    }
    result += call_test("LAPW spinor XC field map", test_lapw_map, 3, false, true);
    result += call_test("Dirac core field sampling", test_dirac_core_map);
    result += call_test("Dirac XC atom", test_dirac_xc_atom);
    result += call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 3, false, false);
    result += call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 1, true, false);
    result += call_test("Native Dirac Hamiltonian", test_dirac_native_hamiltonian, 0, false, false);
    sirius::finalize();
    return result;
}
