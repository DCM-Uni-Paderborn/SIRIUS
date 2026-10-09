/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_lapw_wave_functions.cpp
 *
 *  \brief Contains implementation of sirius::K_point::generate_lapw_wave_functions method.
 */

#include "k_point.hpp"
#include "lapw/generate_alm_block.hpp"
#include "lapw/dirac_basis.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::generate_lapw_wave_functions(wf::Wave_functions<T> const& evec__, wf::Wave_functions<T>& wf__, int ispn__)
{
    PROFILE("sirius::K_point::generate_lapw_wave_functions");

    if (!ctx_.full_potential()) {
        return;
    }
    if (ctx_.valence_relativity() == relativity_t::dirac) {
        RTE_THROW("scalar LAPW matching cannot reconstruct a Dirac spinor");
    }

    auto const& blacs_grid = ctx_.blacs_grid_band();

    auto const& uc = ctx_.unit_cell();

    auto pcs = env::print_checksum();

    auto bs        = ctx_.cyclic_block_size();
    int num_states = evec__.num_wf().get();
    /* store the result of Alm(G) * C_i(G) product */
    la::dmatrix<std::complex<T>> alm_fv(uc.mt_aw_basis_size(), num_states, blacs_grid, bs, bs);

    int atom_begin{0};
    int mt_aw_offset{0};

    auto evec_mg = evec__.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);

    /* loop over blocks of atoms */
    for (auto na : split_in_blocks(uc.num_atoms(), ctx_.cfg().control().max_atom_chunk_size())) {
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        for (int i = 0; i < na; i++) {
            int ia     = atom_begin + i;
            auto& type = uc.atom(ia).type();
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<true, T>(ctx_, atom_begin, na, this->alm_coeffs_loc());
        if (pcs) {
            auto cs = alm.checksum();
            print_checksum("alm", cs, RTE_OUT(this->out(0)));
        }

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for the block of atoms */
        spla::pgemm_ssb(num_mt_aw, num_states, this->gkvec().count(), SPLA_OP_CONJ_TRANSPOSE, 1.0,
                        alm.at(ctx_.processing_unit_memory_t()), alm.ld(),
                        evec__.pw_coeffs(wf::spin_index(0)).at(ctx_.processing_unit_memory_t()), evec__.ld(), 0.0,
                        alm_fv.at(memory_t::host), alm_fv.ld(), mt_aw_offset, 0, alm_fv.spla_distribution(),
                        ctx_.spla_context());

        atom_begin += na;
        mt_aw_offset += num_mt_aw;
    }

    std::vector<int> num_mt_apw_coeffs(uc.num_atoms());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        num_mt_apw_coeffs[ia] = uc.atom(ia).mt_aw_basis_size();
    }
    wf::Wave_functions_mt<T> alm_fv_slab(this->comm(), num_mt_apw_coeffs, wf::num_mag_dims(0),
                                         wf::num_bands(num_states), memory_t::host);

    auto& one  = la::constant<std::complex<T>>::one();
    auto& zero = la::constant<std::complex<T>>::zero();

    auto layout_in  = alm_fv.grid_layout(0, 0, uc.mt_aw_basis_size(), num_states);
    auto layout_out = alm_fv_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, num_states));
    costa::transform(layout_in, layout_out, 'N', one, zero, this->comm().native());

    #pragma omp parallel for
    for (int i = 0; i < num_states; i++) {
        /* G+k block */
        auto in_ptr  = &evec__.pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        auto out_ptr = &wf__.pw_coeffs(0, wf::spin_index(ispn__), wf::band_index(i));
        std::copy(in_ptr, in_ptr + gkvec().count(), out_ptr);

        for (auto it : alm_fv_slab.spl_num_atoms()) {
            int num_mt_aw = uc.atom(it.i).type().mt_aw_basis_size();
            /* aw part of the muffin-tin coefficients */
            for (int xi = 0; xi < num_mt_aw; xi++) {
                wf__.mt_coeffs(xi, it.li, wf::spin_index(ispn__), wf::band_index(i)) =
                        alm_fv_slab.mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
            /* lo part of muffin-tin coefficients: copy from evec */
            for (int xi = 0; xi < uc.atom(it.i).type().mt_lo_basis_size(); xi++) {
                wf__.mt_coeffs(num_mt_aw + xi, it.li, wf::spin_index(ispn__), wf::band_index(i)) =
                        evec__.mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
        }
    }
    if (pcs) {
        auto z1 = wf__.checksum_pw(memory_t::host, wf::spin_index(ispn__), wf::band_range(0, num_states));
        auto z2 = wf__.checksum_mt(memory_t::host, wf::spin_index(ispn__), wf::band_range(0, num_states));
        print_checksum("wf_pw", z1, RTE_OUT(this->out(0)));
        print_checksum("wf_mt", z2, RTE_OUT(this->out(0)));
    }
}

template <typename T>
void
K_point<T>::generate_dirac_wave_functions(mdarray<std::complex<double>, 2> const& evec__,
                                          wf::Wave_functions<T>* target__)
{
    if constexpr (!std::is_same_v<T, double>) {
        RTE_THROW("Dirac LAPW wave functions require fp64");
    } else {
        int ng = num_gkvec(), n = 2 * ng + unit_cell_.mt_lo_basis_size(), nb = ctx_.num_bands();
        auto& psi   = target__ ? *target__ : spinor_wave_functions();
        int invalid = !ctx_.full_potential() || ctx_.valence_relativity() != relativity_t::dirac ||
                      evec__.size(0) != static_cast<size_t>(n) || evec__.size(1) != static_cast<size_t>(nb) ||
                      !evec__.on_host() || &psi.gkvec() != &gkvec() || psi.num_sc() != 2 || psi.num_wf().get() < nb ||
                      psi.num_mt_coeffs() != spinor_wave_functions().num_mt_coeffs();
        if (!invalid) {
            for (size_t i = 0; i < evec__.size(); i++) {
                invalid |= !std::isfinite(evec__[i].real()) || !std::isfinite(evec__[i].imag());
            }
        }
        comm().template allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("Dirac wave-function reconstruction requires finite positive-PW/full-j eigenvectors");
        }
        Dirac_lapw_basis basis(unit_cell_, gkvec());
        for (int s : {0, 1}) {
            for (int b = 0; b < nb; b++) {
                for (int ig = 0; ig < gkvec().count(); ig++) {
                    psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)) = evec__(s * ng + gkvec().offset() + ig, b);
                }
            }
        }
        auto const& one  = la::constant<std::complex<double>>::one();
        auto const& zero = la::constant<std::complex<double>>::zero();
        for (auto atom : psi.spl_num_atoms()) {
            auto const& a = basis.atom_coefficients(atom.i);
            int nm        = unit_cell_.atom(atom.i).mt_basis_size();
            mdarray<std::complex<double>, 2> local({nm, nb});
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', nm, nb, n, &one, a.at(memory_t::host), a.ld(), evec__.at(memory_t::host),
                          evec__.ld(), &zero, local.at(memory_t::host), local.ld());
            for (int b = 0; b < nb; b++) {
                for (int xi = 0; xi < nm; xi++) {
                    psi.mt_coeffs(xi, atom.li, wf::spin_index(0), wf::band_index(b)) = local(xi, b);
                    psi.mt_coeffs(xi, atom.li, wf::spin_index(1), wf::band_index(b)) = 0;
                }
            }
        }
    }
}

template void
K_point<double>::generate_dirac_wave_functions(mdarray<std::complex<double>, 2> const&, wf::Wave_functions<double>*);
#if defined(SIRIUS_USE_FP32)
template void
K_point<float>::generate_dirac_wave_functions(mdarray<std::complex<double>, 2> const&, wf::Wave_functions<float>*);
#endif

template void
K_point<double>::generate_lapw_wave_functions(wf::Wave_functions<double> const& evec__,
                                              wf::Wave_functions<double>& wf__, int ispn__);
#if defined(SIRIUS_USE_FP32)
template void
K_point<float>::generate_lapw_wave_functions(wf::Wave_functions<float> const& evec__, wf::Wave_functions<float>& wf__,
                                             int ispn__);
#endif

} // namespace sirius
