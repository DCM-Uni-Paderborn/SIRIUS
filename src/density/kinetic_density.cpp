/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "density.hpp"
#include "core/wf/kinetic_density_operator.hpp"
#include "function3d/kinetic_density.hpp"
#include "paw_xc_correction.hpp"
#include "lapw_xc_atom.hpp"
#include "lapw/matching_coefficients.hpp"
#include <gsl/gsl_spline.h>

namespace sirius {

namespace {

template <typename T>
std::vector<Spheric_function<function_domain_t::spectral, double>>
atom_kinetic_density(Atom_type const& type__, mdarray<double, 2> const& radial__, radial_wave_function_t storage__,
                     mdarray<std::complex<double>, 3> const& dm__, int num_spins__, int lmax__,
                     std::vector<int> const& radial_sizes__ = {}, mdarray<double, 2> const& derivatives__ = {})
{
    Spheric_kinetic_density_operator<T> op(type__.radial_grid(), type__.indexb(), radial__, lmax__, storage__,
                                           radial_sizes__, derivatives__);
    std::vector<Spheric_function<function_domain_t::spectral, double>> result;
    int n = type__.indexb().size();
    mdarray<std::complex<double>, 2> spin_dm({n, n});
    for (int ispn = 0; ispn < num_spins__; ispn++) {
        result.emplace_back(sf::lmmax(lmax__), type__.radial_grid());
        result.back().zero();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                // LAPW stores <c_i^* c_j>; PAW uses the transposed projector convention.
                spin_dm(i, j) = std::is_same_v<T, double> ? dm__(j, i, ispn) : dm__(i, j, ispn);
            }
        }
        op.add_density(spin_dm, result.back());
    }
    return result;
}

} // namespace

void
Density::initial_kinetic_density()
{
    bool add_von_weizsaecker = ctx_.meta_gga() || external_xc_registered_;
    auto& tau                = kinetic_density();
    tau.zero();
    auto guess = [&](double n) {
        return 0.3 * std::pow(3 * pi * pi * ctx_.num_spins(), 2.0 / 3.0) * std::pow(std::max(0.0, n), 5.0 / 3.0);
    };
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        for (int s = 0; s < ctx_.num_spins(); s++) {
            double n = rho().rg().value(ir);
            if (ctx_.num_spins() == 2) {
                n = 0.5 * (n + (1 - 2 * s) * mag(0).rg().value(ir));
            }
            double t = guess(n);
            tau.scalar().rg().value(ir) += t;
            if (ctx_.num_spins() == 2) {
                tau.vector(0).rg().value(ir) += (1 - 2 * s) * t;
            }
        }
    }
    if (add_von_weizsaecker) {
        // A Thomas-Fermi guess alone can violate tau >= |grad n|^2/(8 n).
        // Add the von Weizsaecker term only to the initial guess, not orbital tau.
        for (int s = 0; s < ctx_.num_spins(); s++) {
            Smooth_periodic_function<double> n(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                n.value(ir) = rho().rg().value(ir);
                if (ctx_.num_spins() == 2) {
                    n.value(ir) = 0.5 * (n.value(ir) + (1 - 2 * s) * mag(0).rg().value(ir));
                }
            }
            n.fft_transform(-1);
            auto grad = to_rg(gradient(n));
            for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                double sigma{0};
                for (int x = 0; x < 3; x++) {
                    sigma += std::pow(grad[x].value(ir), 2);
                }
                double tw = n.value(ir) > 0 ? sigma / (8 * n.value(ir)) : 0;
                tau.scalar().rg().value(ir) += tw;
                if (ctx_.num_spins() == 2) {
                    tau.vector(0).rg().value(ir) += (1 - 2 * s) * tw;
                }
            }
        }
    }
    if (ctx_.full_potential()) {
        SHT sht(device_t::CPU, ctx_.lmax_rho());
        for (auto it : unit_cell_.spl_num_atoms()) {
            auto n = transform(sht, rho().mt()[it.i]);
            Ftp m(sht.num_points(), n.radial_grid());
            m.zero();
            if (ctx_.num_spins() == 2) {
                transform(sht, mag(0).mt()[it.i], m);
            }
            std::vector<Ftp> grad_n, grad_m;
            if (add_von_weizsaecker) {
                auto g = gradient(rho().mt()[it.i]);
                for (int x = 0; x < 3; x++) {
                    grad_n.push_back(transform(sht, g[x]));
                }
                if (ctx_.num_spins() == 2) {
                    auto gm = gradient(mag(0).mt()[it.i]);
                    for (int x = 0; x < 3; x++) {
                        grad_m.push_back(transform(sht, gm[x]));
                    }
                }
            }
            Ftp t(sht.num_points(), n.radial_grid()), delta(sht.num_points(), n.radial_grid());
            t.zero();
            delta.zero();
            for (size_t i = 0; i < n.size(); i++) {
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    double spin_n = ctx_.num_spins() == 1 ? n[i] : 0.5 * (n[i] + (1 - 2 * s) * m[i]);
                    double value  = guess(spin_n);
                    if (add_von_weizsaecker && spin_n > 0) {
                        double sigma{0};
                        for (int x = 0; x < 3; x++) {
                            double g = grad_n[x][i];
                            if (!grad_m.empty()) {
                                g = 0.5 * (g + (1 - 2 * s) * grad_m[x][i]);
                            }
                            sigma += g * g;
                        }
                        value += sigma / (8 * spin_n);
                    }
                    t[i] += value;
                    delta[i] += (1 - 2 * s) * value;
                }
            }
            transform(sht, t, tau.scalar().mt()[it.i]);
            if (ctx_.num_spins() == 2) {
                transform(sht, delta, tau.vector(0).mt()[it.i]);
            }
        }
        for (int j = 0; j < ctx_.num_spins(); j++) {
            tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
        }
    }
    tau.fft_transform(-1);
}

void
Density::generate_mt_kinetic_density(bool add_core__)
{
    if (!ctx_.full_potential() || !has_kinetic_density() || (add_core__ && !core_kinetic_density_ready_)) {
        RTE_THROW("LAPW kinetic density requires enabled storage and, when requested, solved core states");
    }
    auto& tau = kinetic_density();
    for (auto it : unit_cell_.spl_num_atoms()) {
        auto spin_tau = generate_mt_valence_kinetic_density(it.i);
        for (size_t i = 0; i < spin_tau[0].size(); i++) {
            double up = spin_tau[0][i], dn = ctx_.num_spins() == 2 ? spin_tau[1][i] : 0;
            tau.scalar().mt()[it.i][i] = up + dn;
            if (ctx_.num_spins() == 2) {
                tau.vector(0).mt()[it.i][i] = up - dn;
            }
        }
        if (add_core__) {
            auto const& core = core_kinetic_density(it.i);
            for (int ir = 0; ir < static_cast<int>(core.size()); ir++) {
                tau.scalar().mt()[it.i](0, ir) += core[ir] / y00;
            }
        }
    }
    for (int j = 0; j <= ctx_.num_mag_dims(); j++) {
        tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
    }
}

std::vector<Spheric_function<function_domain_t::spectral, double>>
Density::generate_mt_valence_kinetic_density(int ia__) const
{
    if (!ctx_.full_potential()) {
        RTE_THROW("muffin-tin kinetic density requires LAPW");
    }
    auto const& atom = unit_cell_.atom(ia__);
    auto const& type = atom.type();
    mdarray<double, 2> radial({type.num_mt_points(), type.indexr().size()});
    mdarray<double, 2> derivative({type.num_mt_points(), type.indexr().size()});
    for (auto const& rf : type.indexr()) {
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            radial(ir, rf.idxrf) = lapw_basis().radial_basis(atom_index_t::global(ia__)).radial_function(ir, rf.idxrf);
            derivative(ir, rf.idxrf) =
                    lapw_basis().radial_basis(atom_index_t::global(ia__)).radial_function_derivative(ir, rf.idxrf) /
                    type.radial_grid(ir);
        }
    }
    return atom_kinetic_density<std::complex<double>>(type, radial, radial_wave_function_t::R, density_matrix(ia__),
                                                      ctx_.num_spins(), ctx_.lmax_rho(), {}, derivative);
}

std::array<std::vector<Spheric_function<function_domain_t::spectral, double>>, 2>
Density::generate_paw_valence_kinetic_density(int ia__) const
{
    auto const& type = unit_cell_.atom(ia__).type();
    if (ctx_.full_potential() || !type.is_paw() || (type.spin_orbit_coupling() && ctx_.num_mag_dims() != 3)) {
        RTE_THROW("PAW kinetic density requires a pseudopotential context and spinors for SOC partial waves");
    }
    int lmax = 2 * type.indexr().lmax();
    std::vector<int> ae_sizes, ps_sizes;
    for (auto const& rf : type.indexr()) {
        ae_sizes.push_back(static_cast<int>(type.ae_paw_wf(rf.idxrf).size()));
        ps_sizes.push_back(static_cast<int>(type.ps_paw_wf(rf.idxrf).size()));
    }
    return {atom_kinetic_density<double>(type, type.ae_paw_wfs_array(), radial_wave_function_t::r_times_R,
                                         density_matrix(ia__), ctx_.num_spins(), lmax, ae_sizes),
            atom_kinetic_density<double>(type, type.ps_paw_wfs_array(), radial_wave_function_t::r_times_R,
                                         density_matrix(ia__), ctx_.num_spins(), lmax, ps_sizes)};
}

std::array<std::vector<Flm>, 2>
Density::generate_paw_xc_correction(int ia__) const
{
    if (ctx_.full_potential()) {
        RTE_THROW("PAW XC correction requires a pseudopotential context");
    }
    return PAW_xc_correction(unit_cell_.atom(ia__).type()).fields(density_matrix(ia__));
}

template <typename T>
std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density(K_point_set const& ks__) const
{
    PROFILE("sirius::Density::generate_smooth_kinetic_density");
    if (&ks__.ctx() != &ctx_) {
        RTE_THROW("kinetic density requires k-points from the same context");
    }

    int components = ctx_.num_spins();
    std::vector<Smooth_periodic_function<double>> coarse, result;
    for (int j = 0; j < components; j++) {
        coarse.emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
        result.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    }
    int nr = ctx_.spfft_coarse<T>().local_slice_size();
    std::array<std::vector<T>, 4> tau;
    for (int j = 0; j < components; j++) {
        tau[j].resize(nr);
    }
    for (auto it : ks__.spl_num_kpoints()) {
        auto kp = ks__.get<T>(it.i);
        wf::Kinetic_density_operator<T> op(kp->spfft_transform(), *kp->gkvec_fft_sptr());
        std::array<wf::Wave_functions_fft<T>, 2> psi;
        for (int j = 0; j < components; j++) {
            std::fill(tau[j].begin(), tau[j].end(), T(0));
        }
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            if (nbnd == 0) {
                continue;
            }
            psi[ispn] =
                    wf::Wave_functions_fft<T>(kp->gkvec_fft_sptr(), kp->spinor_wave_functions(), wf::spin_index(ispn),
                                              wf::band_range(0, nbnd), wf::shuffle_to::fft_layout);
            for (int ib = 0; ib < psi[ispn].num_wf_local(); ib++) {
                auto band = psi[ispn].spl_num_wf().global_index(ib);
                T weight  = kp->weight() * kp->band_occupancy(band, ispn) / unit_cell_.omega();
                op.add_density(weight, psi[ispn].at(memory_t::host, 0, wf::band_index(ib)), tau[ispn].data());
            }
        }
        for (int j = 0; j < components; j++) {
            for (int ir = 0; ir < nr; ir++) {
                coarse[j].value(ir) += tau[j][ir];
            }
        }
    }

    // Bands and k-points are distributed orthogonally to the real-space FFT slabs.
    auto const& comm = ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft();
    for (int j = 0; j < components; j++) {
        auto ptr = nr ? &coarse[j].value(0) : nullptr;
        comm.allreduce(ptr, nr);
        coarse[j].fft_transform(-1);
        for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
            result[j].f_pw_local(ctx_.gvec().gvec_base_mapping(ig)) = coarse[j].f_pw_local(ig);
        }
        result[j].fft_transform(1);
    }
    return result;
}

template std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density<double>(K_point_set const&) const;
#ifdef SIRIUS_USE_FP32
template std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density<float>(K_point_set const&) const;
#endif

} // namespace sirius
