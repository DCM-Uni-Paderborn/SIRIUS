/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dft_ground_state.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include <sstream>

namespace sirius {

std::array<double, 2>
DFT_ground_state::check_lapw_xc_response(double step__) const
{
    int invalid = !ctx_.full_potential() || !ctx_.cb().external_xc_response_ || !std::isfinite(step__) || step__ <= 0;
    double hmin = step__, hmax = step__;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    ctx_.comm().allreduce<double, mpi::op_t::min>(&hmin, 1);
    ctx_.comm().allreduce<double, mpi::op_t::max>(&hmax, 1);
    if (invalid || hmin != hmax) {
        RTE_THROW("LAPW XC response check requires a point-response callback and a common positive step");
    }
    Density direction(ctx_), plus(ctx_), minus(ctx_);
    direction.zero();
    direction.kinetic_density().zero();
    for (int j = 0; j <= ctx_.num_mag_dims(); j++) {
        axpy(0.3 + 0.1 * j, density_.component(j), direction.component(j));
        axpy(-0.2 + 0.05 * j, density_.kinetic_density().component(j), direction.kinetic_density().component(j));
    }
    auto response = potential_.lapw_external_xc_response(density_, direction);
    std::vector<std::vector<double>> radial_scalar;
    std::vector<std::array<std::vector<double>, 2>> radial_xc;
    potential_.lapw_radial_potential_response(density_, direction, radial_scalar, radial_xc);
    Potential plus_potential(ctx_), minus_potential(ctx_);
    int nf     = ctx_.num_mag_dims() + 1;
    bool dirac = ctx_.valence_relativity() == relativity_t::dirac;
    std::unique_ptr<Hamiltonian0<double>> h0;
    if (dirac || (nf == 4 && ctx_.processing_unit() == device_t::CPU && !ctx_.so_correction())) {
        // Field matrices use the frozen basis and supplied covectors, not the reference potential.
        h0 = std::make_unique<Hamiltonian0<double>>(plus_potential, false);
    }
    auto matrix_samples = [&](Periodic_function<double> const& hartree, lapw_xc_field_adjoint_t const& xc,
                              Potential* actual = nullptr) {
        std::vector<std::complex<double>> values;
        lapw_hamiltonian_response_t prepared;
        std::unique_ptr<Hamiltonian0<double>> shifted;
        if (dirac && actual) {
            // Rebuild the production operator for the stencil, independently of its derivative assembler.
            // Its fixed spherical and kinetic terms cancel between the two samples.
            copy(hartree, actual->effective_potential());
            shifted = std::make_unique<Hamiltonian0<double>>(*actual, false);
        } else {
            prepared = h0->lapw_field_response(hartree, xc);
        }
        for (auto it : kset_.spl_num_kpoints()) {
            auto& kp = *kset_.get<double>(it.i);
            if (dirac) {
                auto matrix = shifted ? std::move(shifted->dirac_matrices(kp.gkvec())[0])
                                      : h0->dirac_field_response(kp.gkvec(), prepared);
                values.insert(values.end(), matrix.begin(), matrix.end());
                continue;
            }
            Hamiltonian_k<double> hk(*h0, kp);
            int n = kp.gklo_basis_size(), bs = ctx_.cyclic_block_size();
            la::dmatrix<std::complex<double>> matrix(n, n, ctx_.blacs_grid(), bs, bs);
            for (int s = 0; s < nf; s++) {
                hk.set_fv_h_field_response(s, prepared, matrix);
                if (matrix.size_local()) {
                    values.insert(values.end(), matrix.at(memory_t::host),
                                  matrix.at(memory_t::host) + matrix.size_local());
                }
            }
        }
        return values;
    };
    std::vector<std::complex<double>> matrices;
    if (h0) {
        matrices = matrix_samples(*potential_.hartree_response(direction.rho()), response);
    }
    std::array<double, 2> result{};
    std::vector<std::complex<double>> previous;
    for (int k = 0; k < 3; k++) {
        double h = std::ldexp(step__, -k);
        copy(density_, plus);
        copy(density_, minus);
        for (int j = 0; j <= ctx_.num_mag_dims(); j++) {
            for (auto* field : {&plus, &minus}) {
                double sign = field == &plus ? 1 : -1;
                axpy(sign * h, direction.component(j), field->component(j));
                axpy(sign * h, direction.kinetic_density().component(j), field->kinetic_density().component(j));
            }
        }
        plus_potential.xc(plus);
        minus_potential.xc(minus);
        auto plus_hartree  = potential_.hartree_response(plus.rho());
        auto minus_hartree = potential_.hartree_response(minus.rho());
        auto plus_core     = plus_potential.get_core_xc_derivatives();
        auto minus_core    = minus_potential.get_core_xc_derivatives();
        auto const& p      = *plus_potential.lapw_xc_derivatives();
        auto const& m      = *minus_potential.lapw_xc_derivatives();
        double error{0}, peak{0};
        size_t cursor{0};
        auto compare = [&](auto r, auto u, auto v) {
            auto fd = (u - v) / (2 * h);
            // Richardson extrapolation cancels the leading central-stencil error.
            double e = k ? std::abs(r - (4.0 * fd - previous[cursor]) / 3.0) : 0;
            double a = std::abs(r);
            if (k) {
                previous[cursor] = fd;
            } else {
                previous.push_back(fd);
            }
            cursor++;
            if (!std::isfinite(e) || !std::isfinite(a)) {
                error = std::numeric_limits<double>::infinity();
            } else {
                error = std::max(error, e);
                peak  = std::max(peak, a);
            }
        };
        auto finish_group = [&](char const* label) {
            ctx_.comm().allreduce<double, mpi::op_t::max>(&error, 1);
            ctx_.comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            if (k) {
                result[k - 1] = std::max(result[k - 1], error / std::max(1e-12, peak));
                if (ctx_.verbosity() >= 2 && ctx_.comm().rank() == 0) {
                    ctx_.out() << "LAPW response " << label << " h=" << h << " error=" << error << " peak=" << peak
                               << '\n';
                }
            }
            error = peak = 0;
        };
        for (int s = 0; s < nf; s++) {
            for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
                compare(response.rho[s].f_pw_local(ig), p.rho[s].f_pw_local(ig), m.rho[s].f_pw_local(ig));
                compare(response.tau[s].f_pw_local(ig), p.tau[s].f_pw_local(ig), m.tau[s].f_pw_local(ig));
            }
        }
        for (size_t ia = 0; ia < response.local.size(); ia++) {
            for (int part = 0; part < 2; part++) {
                for (int s = 0; s < nf; s++) {
                    for (size_t i = 0; i < response.local[ia][part][s].size(); i++) {
                        compare(response.local[ia][part][s][i], p.local[ia][part][s][i], m.local[ia][part][s][i]);
                    }
                }
                for (size_t i = 0; i < response.core[ia][part].size(); i++) {
                    compare(response.core[ia][part][i], p.core[ia][part][i], m.core[ia][part][i]);
                }
            }
        }
        finish_group("field covectors");
        for (int ic = 0; ic < ctx_.unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& cls = ctx_.unit_cell().atom_symmetry_class(ic);
            int nr          = cls.atom_type().num_mt_points();
            for (int ir = 0; ir < nr; ir++) {
                for (int part : {0, 1}) {
                    compare(radial_xc[ic][part][ir], plus_core[ic][part][ir], minus_core[ic][part][ir]);
                }
            }
        }
        finish_group("core covectors");
        for (int ic = 0; ic < ctx_.unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& cls = ctx_.unit_cell().atom_symmetry_class(ic);
            int ia = cls.atom_id(0), nr = cls.atom_type().num_mt_points();
            int owner = ctx_.unit_cell().spl_num_atoms().location(atom_index_t::global(ia)).ib;
            if (ctx_.comm().rank() == owner) {
                for (int ir = 0; ir < nr; ir++) {
                    compare(radial_scalar[ic][ir], y00 * plus_hartree->mt()[ia](0, ir),
                            y00 * minus_hartree->mt()[ia](0, ir));
                }
            }
        }
        finish_group("Hartree");
        if (h0) {
            auto plus  = matrix_samples(*plus_hartree, p, &plus_potential);
            auto minus = matrix_samples(*minus_hartree, m, &minus_potential);
            for (size_t i = 0; i < matrices.size(); i++) {
                compare(matrices[i], plus[i], minus[i]);
            }
            finish_group(dirac ? "Dirac Hamiltonian matrices" : "Pauli Hamiltonian matrices");
        }
    }
    if (ctx_.cb().external_xc_geometry_response_ && ctx_.cb().external_xc_geometry_) {
        auto point_errors = potential_.check_external_xc_point_hessian(density_, step__);
        for (int k = 0; k < 2; k++) {
            result[k] = std::max(result[k], point_errors[k]);
        }
    }
    return result;
}

std::array<std::array<double, 3>, 2>
DFT_ground_state::check_lapw_xc_derivative(double step__)
{
    bool spinor = ctx_.num_mag_dims() == 3;
    bool dirac  = ctx_.valence_relativity() == relativity_t::dirac;
    int nsc     = dirac ? 2 : ctx_.num_spins();
    int nf      = ctx_.num_mag_dims() + 1;
    int invalid = !ctx_.full_potential() || !potential_.lapw_xc_derivatives() || ctx_.hubbard_correction() ||
                  ctx_.cfg().parameters().precision_wf() != "fp64" || !std::isfinite(step__) || step__ <= 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    double hmin = step__, hmax = step__;
    ctx_.comm().allreduce<double, mpi::op_t::min>(&hmin, 1);
    ctx_.comm().allreduce<double, mpi::op_t::max>(&hmax, 1);
    if (invalid || hmin != hmax) {
        RTE_THROW("LAPW XC derivative check requires fp64 operators and a common positive step");
    }
    using Wf   = wf::Wave_functions<double>;
    auto bands = wf::band_range(0, ctx_.num_bands());
    std::vector<int> nmt, nlo;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        nmt.push_back(unit_cell_.atom(ia).type().mt_basis_size());
        nlo.push_back(unit_cell_.atom(ia).type().mt_lo_basis_size());
    }
    std::vector<std::unique_ptr<Wf>> saved, tangent;
    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<double>(it.i);
        for (auto* storage : {&saved, &tangent}) {
            storage->push_back(std::make_unique<Wf>(kp->gkvec_sptr(), nmt,
                                                    wf::num_mag_dims(dirac ? 3 : ctx_.num_mag_dims()),
                                                    wf::num_bands(ctx_.num_bands()), memory_t::host));
        }
        if (kp->spinor_wave_functions().pw_coeffs(wf::spin_index(0)).on_device()) {
            kp->spinor_wave_functions().copy_to(memory_t::host);
        }
        for (int s = 0; s < nsc; s++) {
            wf::copy(memory_t::host, kp->spinor_wave_functions(), wf::spin_index(s), bands, *saved.back(),
                     wf::spin_index(s), bands);
        }
    }
    auto restore = [&] {
        size_t ik{0};
        for (auto it : kset_.spl_num_kpoints()) {
            auto kp = kset_.get<double>(it.i);
            for (int s = 0; s < nsc; s++) {
                wf::copy(memory_t::host, *saved[ik], wf::spin_index(s), bands, kp->spinor_wave_functions(),
                         wf::spin_index(s), bands);
            }
            ik++;
        }
    };
    auto dirac_coefficients = [&](K_point<double> const& kp, Wf const& psi, int b) {
        int ng = kp.num_gkvec();
        std::vector<std::complex<double>> c(2 * ng + unit_cell_.mt_lo_basis_size());
        std::vector<std::complex<double>> all;
        for (int s : {1, 0}) {
            all = psi.gather(wf::spin_index(s), wf::band_index(b));
            std::copy_n(all.data(), ng, c.data() + s * ng);
        }
        int offset = ng, column = 2 * ng;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int naw = unit_cell_.atom(ia).type().mt_aw_basis_size();
            std::copy_n(all.data() + offset + naw, nlo[ia], c.data() + column);
            offset += nmt[ia];
            column += nlo[ia];
        }
        return c;
    };
    // Perturb the independent PW/LO coefficients, then use the production
    // matching transformation for both the orbitals and their tangents.
    auto set_orbitals = [&](int direction, double t, bool derivative) {
        size_t ik{0};
        for (auto it : kset_.spl_num_kpoints()) {
            auto kp = kset_.get<double>(it.i);
            if (dirac) {
                int ng = kp->num_gkvec();
                mdarray<std::complex<double>, 2> evec({2 * ng + unit_cell_.mt_lo_basis_size(), ctx_.num_bands()});
                for (int b = 0; b < ctx_.num_bands(); b++) {
                    auto c = dirac_coefficients(*kp, *saved[ik], b);
                    for (int s : {0, 1}) {
                        for (int ig = 0; ig < ng; ig++) {
                            auto g               = kp->gkvec().gvec(gvec_index_t::global(ig));
                            double phase         = direction == 0
                                                           ? (1 + 0.17 * s) * std::sin(0.41 * g[0] + 0.73 * g[1] + 0.19 * g[2])
                                                           : 0;
                            auto z               = c[s * ng + ig] * std::polar(1.0, t * phase);
                            evec(s * ng + ig, b) = derivative ? std::complex<double>(0, phase) * z : z;
                        }
                    }
                    int column = 2 * ng;
                    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                        for (int j = 0; j < nlo[ia]; j++, column++) {
                            double phase    = direction == 1 ? std::sin(0.43 * (ia + 1) + 0.29 * (j + 1)) : 0;
                            auto z          = c[column] * std::polar(1.0, t * phase);
                            evec(column, b) = derivative ? std::complex<double>(0, phase) * z : z;
                        }
                    }
                }
                kp->generate_dirac_wave_functions(evec, derivative ? tangent[ik].get() : nullptr);
                ik++;
                continue;
            }
            Wf evec(kp->gkvec_sptr(), nlo, wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
            auto& target = derivative ? *tangent[ik] : kp->spinor_wave_functions();
            auto rotate  = [&](std::complex<double> c, double phase) {
                auto z = c * std::polar(1.0, t * phase);
                return derivative ? std::complex<double>(0, phase) * z : z;
            };
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int b = 0; b < ctx_.num_bands(); b++) {
                    auto band = wf::band_index(b);
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g       = kp->gkvec().gvec(gvec_index_t::local(ig));
                        double phase = direction == 0 ? std::sin(0.41 * g[0] + 0.73 * g[1] + 0.19 * g[2]) : 0;
                        phase *= spinor ? 1 + 0.17 * s : 1;
                        evec.pw_coeffs(ig, wf::spin_index(0), band) =
                                rotate(saved[ik]->pw_coeffs(ig, wf::spin_index(s), band), phase);
                    }
                    for (auto atom : evec.spl_num_atoms()) {
                        int naw = unit_cell_.atom(atom.i).type().mt_aw_basis_size();
                        for (int j = 0; j < nlo[atom.i]; j++) {
                            double phase = direction == 1 ? std::sin(0.43 * (atom.i + 1) + 0.29 * (j + 1)) : 0;
                            phase *= spinor ? 1 + 0.17 * s : 1;
                            evec.mt_coeffs(j, atom.li, wf::spin_index(0), band) =
                                    rotate(saved[ik]->mt_coeffs(naw + j, atom.li, wf::spin_index(s), band), phase);
                        }
                    }
                }
                kp->generate_lapw_wave_functions(evec, target, s);
            }
            ik++;
        }
    };

    std::array<std::array<double, 3>, 2> result{};
    try {
        Density density(ctx_);
        copy(density_, density);
        Potential potential(ctx_);
        auto expectation = [&] {
            // Do not regenerate radial functions or update atom-potential pointers.
            // Subtracting the zero-XC operator cancels the fixed non-XC terms.
            Hamiltonian0<double> h0(potential, false);
            double derivative{0};
            size_t ik{0};
            for (auto it : kset_.spl_num_kpoints()) {
                auto kp = kset_.get<double>(it.i);
                if (dirac) {
                    auto matrices = h0.dirac_matrices(kp->gkvec());
                    auto const& h = matrices[0];
                    for (int b = 0; b < kp->num_occupied_bands(0); b++) {
                        auto c        = dirac_coefficients(*kp, kp->spinor_wave_functions(), b);
                        auto d        = dirac_coefficients(*kp, *tangent[ik], b);
                        double weight = 2 * kp->weight() * kp->band_occupancy(b, 0) / kp->comm().size();
                        for (size_t j = 0; j < c.size(); j++) {
                            for (size_t i = 0; i < c.size(); i++) {
                                derivative += weight * std::real(std::conj(d[i]) * h(i, j) * c[j]);
                            }
                        }
                    }
                    ik++;
                    continue;
                }
                h0.local_op().prepare_k(*kp->gkvec_fft_sptr());
                Wf phi(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf hphi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf bphi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf other(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf xyphi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    int nb = kp->num_occupied_bands(spinor ? 0 : s);
                    if (!nb) {
                        continue;
                    }
                    auto occupied = wf::band_range(0, nb);
                    wf::copy(memory_t::host, kp->spinor_wave_functions(), wf::spin_index(s), occupied, phi,
                             wf::spin_index(0), occupied);
                    {
                        auto input = phi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
                        auto h     = hphi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::host);
                        auto z     = bphi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::host);
                        h0.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), occupied, phi, &hphi,
                                                   nullptr, ctx_.num_spins() == 2 ? &bphi : nullptr, nullptr);
                    }
                    if (spinor) {
                        wf::copy(memory_t::host, kp->spinor_wave_functions(), wf::spin_index(1 - s), occupied, other,
                                 wf::spin_index(0), occupied);
                        auto input = other.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
                        auto xy    = xyphi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::host);
                        h0.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), occupied, other,
                                                   nullptr, nullptr, nullptr, &xyphi, s == 1);
                    }
                    for (int b = 0; b < nb; b++) {
                        auto band     = wf::band_index(b);
                        double weight = 2 * kp->weight() * kp->band_occupancy(b, spinor ? 0 : s);
                        for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                            auto h = hphi.pw_coeffs(ig, wf::spin_index(0), band);
                            if (ctx_.num_spins() == 2) {
                                h += double(1 - 2 * s) * bphi.pw_coeffs(ig, wf::spin_index(0), band);
                            }
                            if (spinor) {
                                h += xyphi.pw_coeffs(ig, wf::spin_index(0), band);
                            }
                            derivative += weight *
                                          std::real(std::conj(tangent[ik]->pw_coeffs(ig, wf::spin_index(s), band)) * h);
                        }
                        for (auto atom : phi.spl_num_atoms()) {
                            auto const& hmt = h0.hmt(atom.i);
                            for (int i = 0; i < nmt[atom.i]; i++) {
                                std::complex<double> h{0};
                                for (int j = 0; j < nmt[atom.i]; j++) {
                                    auto matrix = hmt(i, j, s);
                                    if (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()) {
                                        matrix = hmt(i, j, 0) + double(1 - 2 * s) * hmt(i, j, 1);
                                    }
                                    h += matrix * phi.mt_coeffs(j, atom.li, wf::spin_index(0), band);
                                    if (spinor) {
                                        h += hmt(i, j, s + 2) * other.mt_coeffs(j, atom.li, wf::spin_index(0), band);
                                    }
                                }
                                derivative += weight * std::real(std::conj(tangent[ik]->mt_coeffs(
                                                                         i, atom.li, wf::spin_index(s), band)) *
                                                                 h);
                            }
                        }
                    }
                }
                ik++;
            }
            ctx_.comm().allreduce(&derivative, 1);
            return derivative;
        };
        Density plus_fields(ctx_), minus_fields(ctx_);
        std::vector<std::pair<std::complex<double>, double>> coarse_difference;
        auto field_error = [&](Density const& response, double h, bool refined) {
            double error{0}, peak{0}, relative{0};
            std::array<double, 3> result{};
            size_t index{0};
            auto compare = [&](auto r, auto p, auto m) {
                std::complex<double> difference = (p - m) / (2 * h);
                double roundoff = 32 * std::numeric_limits<double>::epsilon() * (std::abs(p) + std::abs(m)) / (2 * h);
                if (refined) {
                    auto const& coarse = coarse_difference.at(index++);
                    difference         = (4.0 * difference - coarse.first) / 3.0;
                    roundoff           = (4 * roundoff + coarse.second) / 3;
                } else {
                    coarse_difference.emplace_back(difference, roundoff);
                }
                double e = std::abs(r - difference), a = std::abs(r);
                if (!std::isfinite(e) || !std::isfinite(a)) {
                    error = std::numeric_limits<double>::infinity();
                } else {
                    // Weak responses can be below the cancellation noise of the base fields.
                    e     = std::max(0.0, e - roundoff);
                    error = std::max(error, e);
                    peak  = std::max(peak, a);
                }
            };
            for (int part : {0, 1}) {
                for (int j = 0; j <= ctx_.num_mag_dims(); j++) {
                    auto const& r = part == 0 ? response.component(j) : response.kinetic_density().component(j);
                    auto const& p = part == 0 ? plus_fields.component(j) : plus_fields.kinetic_density().component(j);
                    auto const& m = part == 0 ? minus_fields.component(j) : minus_fields.kinetic_density().component(j);
                    for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
                        compare(r.rg().f_pw_local(ig), p.rg().f_pw_local(ig), m.rg().f_pw_local(ig));
                    }
                    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                        compare(r.rg().value(ir), p.rg().value(ir), m.rg().value(ir));
                    }
                    for (auto it : unit_cell_.spl_num_atoms()) {
                        for (size_t i = 0; i < r.mt()[it.i].size(); i++) {
                            compare(r.mt()[it.i][i], p.mt()[it.i][i], m.mt()[it.i][i]);
                        }
                    }
                }
                ctx_.comm().allreduce<double, mpi::op_t::max>(&error, 1);
                ctx_.comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                double scaled_error = error / std::max(1e-12, peak);
                if (scaled_error > relative) {
                    relative = scaled_error;
                    result   = {relative, error, peak};
                }
                error = peak = 0;
            }
            return result;
        };
        for (int direction = 0; direction < 2; direction++) {
            coarse_difference.clear();
            constexpr double offset{0.17};
            auto evaluate = [&](double t) {
                set_orbitals(direction, offset + t, false);
                density.generate<double>(kset_, false, true, true);
                potential.generate(density, false, true);
                return potential.energy_exc(density);
            };
            auto base_energy = evaluate(0);
            if (direction == 0) {
                auto repeat_energy = evaluate(0);
                if (std::abs(base_energy - repeat_energy) > 1e-10 * std::max(1.0, std::abs(base_energy))) {
                    RTE_THROW("LAPW XC callback energy is not repeatable at fixed orbital fields");
                }
            }
            set_orbitals(direction, offset, true);
            result[direction][0] = expectation();
            std::vector<Wf*> orbital_directions;
            for (auto const& d : tangent) {
                orbital_directions.push_back(d.get());
            }
            auto fields             = density_.lapw_orbital_fields_response(kset_, orbital_directions);
            auto contraction        = potential.lapw_xc_contractions(*fields);
            double field_derivative = contraction.rho + contraction.magnetic + contraction.tau;
            // Remove only XC, leaving the identical non-XC operator in place.
            lapw_xc_field_adjoint_t zero;
            for (int s = 0; s < nf; s++) {
                zero.rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                zero.tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                zero.rho.back().zero();
                zero.tau.back().zero();
            }
            zero.local.resize(unit_cell_.num_atoms());
            zero.core.resize(unit_cell_.num_atoms());
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto const& grid = unit_cell_.atom(ia).type().radial_grid();
                for (int part = 0; part < 2; part++) {
                    zero.core[ia][part].resize(grid.num_points());
                    for (int s = 0; s < nf; s++) {
                        zero.local[ia][part].emplace_back(sf::lmmax(ctx_.lmax_rho()), grid);
                        zero.local[ia][part].back().zero();
                    }
                }
            }
            potential.set_lapw_xc_derivatives(std::move(zero), 0);
            result[direction][0] -= expectation();
            if (std::abs(result[direction][0] - field_derivative) >
                1e-10 * std::max({1.0, std::abs(result[direction][0]), std::abs(field_derivative)})) {
                RTE_THROW("LAPW orbital density/tau response disagrees with the XC Hamiltonian derivative");
            }
            for (int j = 1; j <= 2; j++) {
                double h    = step__ / j;
                double plus = evaluate(h);
                plus_fields.generate<double>(kset_, false, false, true);
                double minus = evaluate(-h);
                minus_fields.generate<double>(kset_, false, false, true);
                result[direction][j] = (plus - minus) / (2 * h);
                // Both existing stencils give a fourth-order field derivative.
                auto error       = field_error(*fields, h, j == 2);
                double tolerance = std::max(5e-9, 8 * std::pow(step__, 4));
                if (!std::isfinite(error[0]) || (j == 2 && error[0] > tolerance)) {
                    std::ostringstream message;
                    message << std::scientific << "LAPW orbital rho/tau response fails finite differences: direction "
                            << direction + 1 << ", step " << h << ", relative error " << error[0] << ", tolerance "
                            << tolerance << ", roundoff-adjusted absolute error " << error[1] << ", response scale "
                            << error[2];
                    RTE_THROW(message.str());
                }
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    return result;
}

} // namespace sirius
