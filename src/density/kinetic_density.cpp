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
#include <gsl/gsl_spline.h>

namespace sirius {

void
Density::generate_pseudo_core_fields()
{
    int invalid{0};
    try {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto const& type = unit_cell_.atom_type(iat);
            type.check_ps_core_kinetic_density();
            invalid |= type.has_ps_core_kinetic_density() &&
                       type.num_mt_points() < static_cast<int>(gsl_interp_type_min_size(gsl_interp_steffen));
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("meta-GGA nonlinear core correction requires setup-supplied positive pseudo-core tau");
    }

    rho_pseudo_core_->zero();
    tau_pseudo_core_->zero();
    auto const& fft = ctx_.spfft<double>();
    std::array<int, 3> dims{fft.dim_x(), fft.dim_y(), fft.dim_z()};
    auto wrap = [](int i, int n) { return (i % n + n) % n; };
    try {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto const& type = unit_cell_.atom_type(iat);
            if (!type.has_ps_core_kinetic_density()) {
                continue;
            }
            std::vector<double> grid(type.num_mt_points());
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                grid[ir] = type.radial_grid(ir);
            }
            auto core_rho = type.ps_core_charge_density();
            core_rho.resize(grid.size(), 0);
            std::unique_ptr<gsl_spline, decltype(&gsl_spline_free)> rho(
                    gsl_spline_alloc(gsl_interp_steffen, grid.size()), gsl_spline_free);
            std::unique_ptr<gsl_spline, decltype(&gsl_spline_free)> tau(
                    gsl_spline_alloc(gsl_interp_steffen, grid.size()), gsl_spline_free);
            std::unique_ptr<gsl_interp_accel, decltype(&gsl_interp_accel_free)> accel(gsl_interp_accel_alloc(),
                                                                                      gsl_interp_accel_free);
            if (!rho || !tau || !accel) {
                RTE_THROW("failed to allocate pseudo-core interpolation");
            }
            gsl_spline_init(rho.get(), grid.data(), core_rho.data(), grid.size());
            gsl_spline_init(tau.get(), grid.data(), type.ps_core_kinetic_density().data(), grid.size());
            // Sample the fixed radial fields directly. Fourier truncation of a positive
            // core can produce negative tails; monotone interpolation avoids this without clipping.
            for (int i = 0; i < type.num_atoms(); i++) {
                auto position = unit_cell_.atom(type.atom_id(i)).position();
                std::array<int, 3> first, last;
                for (int j = 0; j < 3; j++) {
                    position[j] -= std::floor(position[j]);
                    double length2{0};
                    for (int k = 0; k < 3; k++) {
                        length2 += std::pow(unit_cell_.inverse_lattice_vectors()(j, k), 2);
                    }
                    double extent = grid.back() * std::sqrt(length2);
                    first[j]      = static_cast<int>(std::floor(dims[j] * (position[j] - extent)));
                    last[j]       = static_cast<int>(std::ceil(dims[j] * (position[j] + extent)));
                }
                // Unwrapped grid indices enumerate every periodic image, including overlapping cores.
                for (int z = first[2]; z <= last[2]; z++) {
                    int iz = wrap(z, dims[2]) - fft.local_z_offset();
                    if (iz < 0 || iz >= fft.local_z_length()) {
                        continue;
                    }
                    for (int y = first[1]; y <= last[1]; y++) {
                        for (int x = first[0]; x <= last[0]; x++) {
                            r3::vector<double> dr(double(x) / dims[0] - position[0], double(y) / dims[1] - position[1],
                                                  double(z) / dims[2] - position[2]);
                            double r = unit_cell_.get_cartesian_coordinates(dr).length();
                            if (r > grid.back()) {
                                continue;
                            }
                            int ir = ctx_.fft_grid().index_by_coord(wrap(x, dims[0]), wrap(y, dims[1]), iz);
                            // The tabulated origin value also defines the unresolved interval below r_min.
                            r = std::max(r, grid.front());
                            // Preserve tabulated endpoints exactly, including a zero at the outer boundary.
                            auto sample = [&](gsl_spline const* spline) {
                                if (r == grid.front()) {
                                    return spline->y[0];
                                }
                                if (r == grid.back()) {
                                    return spline->y[grid.size() - 1];
                                }
                                return gsl_spline_eval(spline, r, accel.get());
                            };
                            rho_pseudo_core_->value(ir) += sample(rho.get());
                            tau_pseudo_core_->value(ir) += sample(tau.get());
                        }
                    }
                }
            }
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("failed to sample pseudo-core density and kinetic density");
    }
    rho_pseudo_core_->fft_transform(-1);
    tau_pseudo_core_->fft_transform(-1);
}

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
    for (int j = 0; j < ctx_.num_spins(); j++) {
        tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
    }
}

std::vector<Spheric_function<function_domain_t::spectral, double>>
Density::generate_mt_valence_kinetic_density(int ia__) const
{
    if (!ctx_.full_potential() || ctx_.num_mag_dims() == 3) {
        RTE_THROW("muffin-tin kinetic density requires nonmagnetic or collinear LAPW");
    }
    auto const& atom = unit_cell_.atom(ia__);
    auto const& type = atom.type();
    mdarray<double, 2> radial({type.num_mt_points(), type.indexr().size()});
    mdarray<double, 2> derivative({type.num_mt_points(), type.indexr().size()});
    for (auto const& rf : type.indexr()) {
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            radial(ir, rf.idxrf) = atom.symmetry_class().radial_function(ir, rf.idxrf);
            derivative(ir, rf.idxrf) =
                    atom.symmetry_class().radial_function_derivative(ir, rf.idxrf) / type.radial_grid(ir);
        }
    }
    return atom_kinetic_density<std::complex<double>>(type, radial, radial_wave_function_t::R, density_matrix(ia__),
                                                      ctx_.num_spins(), ctx_.lmax_rho(), {}, derivative);
}

std::array<std::vector<Spheric_function<function_domain_t::spectral, double>>, 2>
Density::generate_paw_valence_kinetic_density(int ia__) const
{
    auto const& type = unit_cell_.atom(ia__).type();
    if (ctx_.full_potential() || !type.is_paw() || ctx_.num_mag_dims() == 3 || type.spin_orbit_coupling()) {
        RTE_THROW("PAW kinetic density requires nonmagnetic or collinear scalar partial waves");
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
    if (ctx_.full_potential() || ctx_.num_mag_dims() == 3) {
        RTE_THROW("PAW XC correction requires a nonmagnetic or collinear pseudopotential context");
    }
    return PAW_xc_correction(unit_cell_.atom(ia__).type()).fields(density_matrix(ia__));
}

std::vector<std::array<std::vector<Flm>, 2>>
Density::lapw_radial_fields_response(K_point_set& ks__, std::vector<radial_basis_response_t> const& response__) const
{
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU ||
                  ctx_.cfg().parameters().precision_wf() != "fp64" || ctx_.num_mag_dims() == 3 ||
                  ctx_.so_correction() || ctx_.valence_relativity() != relativity_t::none ||
                  (unit_cell_.num_core_electrons() != 0 && ctx_.core_relativity() != relativity_t::none) ||
                  response__.size() != unit_cell_.num_atom_symmetry_classes();
    if (!invalid) {
        for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
            auto const& cls  = unit_cell_.atom_symmetry_class(ic);
            auto const& type = cls.atom_type();
            auto const& d    = response__[ic];
            invalid |= type.num_core_electrons() != 0 && cls.spherical_xc_derivatives()[0].empty();
            invalid |= d.radial_functions.size(0) != type.num_mt_points() ||
                       d.radial_functions.size(1) != type.mt_radial_basis_size() || d.radial_functions.size(2) != 2 ||
                       d.surface_derivatives.size(0) != 3 ||
                       d.surface_derivatives.size(1) != type.mt_radial_basis_size() - type.num_lo_descriptors();
            for (size_t i = 0; i < d.radial_functions.size(); i++) {
                invalid |= !std::isfinite(d.radial_functions[i]);
            }
            for (size_t i = 0; i < d.surface_derivatives.size(); i++) {
                invalid |= !std::isfinite(d.surface_derivatives[i]);
            }
            for (auto const& field : d.core_fields) {
                invalid |= field.size() != type.num_mt_points() ||
                           !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); });
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW radial fields response requires scalar CPU fp64 orbitals and compatible finite responses");
    }
    int ns = ctx_.num_spins();
    std::vector<mdarray<std::complex<double>, 3>> dm, ddm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, ns});
        ddm.emplace_back(std::array<index_range, 3>{n, n, ns});
        dm.back().zero();
        ddm.back().zero();
    }
    for (auto it : ks__.spl_num_kpoints()) {
        auto const& kp = *ks__.get<double>(it.i);
        Matching_coefficients matching(unit_cell_, kp.gkvec());
        std::vector<mdarray<std::complex<double>, 2>> alm, dalm;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& atom = unit_cell_.atom(ia);
            std::array<index_range, 2> shape{kp.gkvec().count(), atom.type().mt_aw_basis_size()};
            alm.emplace_back(shape);
            dalm.emplace_back(shape);
            matching.generate<false>(atom, alm.back());
            matching.generate_potential_response<false>(atom, response__[atom.symmetry_class().id()], dalm.back());
        }
        for (int s = 0; s < ns; s++) {
            for (int band = 0; band < ctx_.num_bands(); band++) {
                double weight = kp.weight() * kp.band_occupancy(band, s) / kp.comm().size();
                if (weight == 0) {
                    continue;
                }
                auto all   = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(band));
                int offset = kp.gkvec().num_gvec();
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    auto const& type = unit_cell_.atom(ia).type();
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
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            dm[ia](i, j, s) += weight * std::conj(c[i]) * c[j];
                            ddm[ia](i, j, s) += weight * (std::conj(dc[i]) * c[j] + std::conj(c[i]) * dc[j]);
                        }
                    }
                    offset += n;
                }
            }
        }
    }
    std::vector<std::array<std::vector<Flm>, 2>> result;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ctx_.comm().allreduce(dm[ia].at(memory_t::host), dm[ia].size());
        ctx_.comm().allreduce(ddm[ia].at(memory_t::host), ddm[ia].size());
        auto const& atom = unit_cell_.atom(ia);
        auto const& d    = response__[atom.symmetry_class().id()];
        LAPW_xc_atom op(atom, ctx_.lmax_rho());
        auto radial = op.fields(dm[ia], &d);
        result.push_back(op.fields(ddm[ia]));
        for (int part : {0, 1}) {
            for (int s = 0; s < ns; s++) {
                auto& field = result.back()[part][s];
                for (size_t i = 0; i < field.size(); i++) {
                    field[i] += radial[part][s][i];
                }
                for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                    field(0, ir) += d.core_fields[part][ir] / (ns * y00);
                }
            }
        }
    }
    return result;
}

std::unique_ptr<Density>
Density::lapw_orbital_fields_response(K_point_set& ks__, std::vector<wf::Wave_functions<double>*> const& directions__,
                                      std::vector<radial_basis_response_t> const& radial__,
                                      std::vector<mdarray<double, 2>> const& occupation_directions__) const
{
    int occupation_response = !occupation_directions__.empty();
    ctx_.comm().allreduce<int, mpi::op_t::max>(&occupation_response, 1);
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || !has_kinetic_density() ||
                  ctx_.processing_unit() != device_t::CPU ||
                  ctx_.spfft_coarse<double>().processing_unit() != SPFFT_PU_HOST ||
                  ctx_.cfg().parameters().precision_wf() != "fp64" || ctx_.num_mag_dims() == 3 ||
                  ctx_.so_correction() || ctx_.valence_relativity() != relativity_t::none ||
                  directions__.size() != ks__.spl_num_kpoints().local_size() ||
                  (occupation_response && occupation_directions__.size() != directions__.size());
    if (!invalid) {
        size_t ik{0};
        for (auto it : ks__.spl_num_kpoints()) {
            auto const& psi = ks__.get<double>(it.i)->spinor_wave_functions();
            auto const* d   = directions__[ik];
            if (occupation_response) {
                auto const& df = occupation_directions__[ik];
                if (df.size(0) != ctx_.num_bands() || df.size(1) != ctx_.num_spins() || !df.on_host()) {
                    invalid = 1;
                } else {
                    for (size_t i = 0; i < df.size(); i++) {
                        invalid |= !std::isfinite(df[i]);
                    }
                }
            }
            ik++;
            if (!d || &d->gkvec() != &psi.gkvec() || d->num_sc() != psi.num_sc() ||
                d->num_wf().get() < ctx_.num_bands() || d->num_mt_coeffs() != psi.num_mt_coeffs()) {
                invalid = 1;
                continue;
            }
            if (psi.ld() == 0) {
                continue;
            }
            for (int s = 0; s < ctx_.num_spins(); s++) {
                if (!psi.pw_coeffs(wf::spin_index(s)).on_host() || !d->pw_coeffs(wf::spin_index(s)).on_host()) {
                    invalid = 1;
                    continue;
                }
                for (int b = 0; b < ctx_.num_bands(); b++) {
                    auto p = psi.at(memory_t::host, 0, wf::spin_index(s), wf::band_index(b));
                    auto q = d->at(memory_t::host, 0, wf::spin_index(s), wf::band_index(b));
                    for (int i = 0; i < psi.ld(); i++) {
                        invalid |= !std::isfinite(p[i].real()) || !std::isfinite(p[i].imag()) ||
                                   !std::isfinite(q[i].real()) || !std::isfinite(q[i].imag());
                    }
                }
            }
        }
    }
    int radial_min = !radial__.empty(), radial_max = radial_min;
    ctx_.comm().allreduce<int, mpi::op_t::min>(&radial_min, 1);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&radial_max, 1);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid || radial_min != radial_max) {
        RTE_THROW("LAPW orbital fields response requires matching finite CPU fp64 orbitals and valid directions");
    }
    auto result = std::make_unique<Density>(ctx_);
    result->zero();
    result->kinetic_density().zero();
    int ns = ctx_.num_spins(), nr = ctx_.spfft_coarse<double>().local_slice_size();
    std::array<std::vector<Smooth_periodic_function<double>>, 2> coarse;
    for (int part : {0, 1}) {
        for (int s = 0; s < ns; s++) {
            coarse[part].emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
        }
    }
    std::vector<mdarray<std::complex<double>, 3>> dm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, ns});
        dm.back().zero();
    }
    size_t ik{0};
    for (auto it : ks__.spl_num_kpoints()) {
        auto& kp       = *ks__.get<double>(it.i);
        auto& d        = *directions__[ik];
        auto const* df = occupation_response ? &occupation_directions__[ik] : nullptr;
        ik++;
        auto& fft     = kp.spfft_transform();
        bool real_fft = fft.type() == SPFFT_TRANS_R2C;
        wf::Kinetic_density_operator<double> kinetic(fft, *kp.gkvec_fft_sptr());
        std::vector<std::complex<double>> base(nr);
        std::vector<double> tau(nr);
        for (int s = 0; s < ns; s++) {
            int nb = kp.num_occupied_bands(s);
            if (df) {
                for (int b = nb; b < ctx_.num_bands(); b++) {
                    if ((*df)(b, s) != 0) {
                        nb = b + 1;
                    }
                }
            }
            if (nb == 0) {
                continue;
            }
            wf::Wave_functions_fft<double> psi(kp.gkvec_fft_sptr(), kp.spinor_wave_functions(), wf::spin_index(s),
                                               wf::band_range(0, nb), wf::shuffle_to::fft_layout);
            wf::Wave_functions_fft<double> delta(kp.gkvec_fft_sptr(), d, wf::spin_index(s), wf::band_range(0, nb),
                                                 wf::shuffle_to::fft_layout);
            std::fill(tau.begin(), tau.end(), 0);
            for (int b = 0; b < psi.num_wf_local(); b++) {
                auto band     = psi.spl_num_wf().global_index(b);
                double weight = kp.weight() * kp.band_occupancy(band, s) / unit_cell_.omega();
                double dw     = df ? kp.weight() * (*df)(band, s) / unit_cell_.omega() : 0;
                auto p        = psi.at(memory_t::host, 0, wf::band_index(b));
                auto q        = delta.at(memory_t::host, 0, wf::band_index(b));
                fft.backward(reinterpret_cast<double const*>(p), SPFFT_PU_HOST);
                auto values = fft.space_domain_data(SPFFT_PU_HOST);
                for (int ir = 0; ir < nr; ir++) {
                    base[ir] = real_fft ? std::complex<double>(values[ir], 0)
                                        : reinterpret_cast<std::complex<double> const*>(values)[ir];
                }
                fft.backward(reinterpret_cast<double const*>(q), SPFFT_PU_HOST);
                values = fft.space_domain_data(SPFFT_PU_HOST);
                for (int ir = 0; ir < nr; ir++) {
                    auto value = real_fft ? std::complex<double>(values[ir], 0)
                                          : reinterpret_cast<std::complex<double> const*>(values)[ir];
                    coarse[0][s].value(ir) +=
                            2 * weight * std::real(std::conj(base[ir]) * value) + dw * std::norm(base[ir]);
                }
                kinetic.add_density_response(weight, p, q, tau.data());
                if (dw != 0) {
                    kinetic.add_density(dw, p, tau.data());
                }
            }
            for (int ir = 0; ir < nr; ir++) {
                coarse[1][s].value(ir) += tau[ir];
            }
            for (int b = 0; b < nb; b++) {
                double weight = kp.weight() * kp.band_occupancy(b, s) / kp.comm().size();
                double dw     = df ? kp.weight() * (*df)(b, s) / kp.comm().size() : 0;
                auto p        = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(b));
                auto q        = d.gather(wf::spin_index(s), wf::band_index(b));
                int offset    = kp.gkvec().num_gvec();
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    int n = unit_cell_.atom(ia).type().mt_basis_size();
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            dm[ia](i, j, s) += weight * (std::conj(q[offset + i]) * p[offset + j] +
                                                         std::conj(p[offset + i]) * q[offset + j]) +
                                               dw * std::conj(p[offset + i]) * p[offset + j];
                        }
                    }
                    offset += n;
                }
            }
        }
    }
    for (int part : {0, 1}) {
        auto& target = part == 0 ? static_cast<Field4D&>(*result) : result->kinetic_density();
        for (int s = 0; s < ns; s++) {
            auto ptr = nr ? &coarse[part][s].value(0) : nullptr;
            ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft().allreduce(ptr, nr);
            coarse[part][s].fft_transform(-1);
            for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
                int fine   = ctx_.gvec().gvec_base_mapping(ig);
                auto value = coarse[part][s].f_pw_local(ig);
                target.scalar().rg().f_pw_local(fine) += value;
                if (ns == 2) {
                    target.vector(0).rg().f_pw_local(fine) += double(1 - 2 * s) * value;
                }
            }
        }
        target.fft_transform(1);
    }
    auto radial = radial__.empty() ? std::vector<std::array<std::vector<Flm>, 2>>{}
                                   : lapw_radial_fields_response(ks__, radial__);
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ctx_.comm().allreduce(dm[ia].at(memory_t::host), dm[ia].size());
        auto fields = LAPW_xc_atom(unit_cell_.atom(ia), ctx_.lmax_rho()).fields(dm[ia]);
        for (int part : {0, 1}) {
            auto& target = part == 0 ? static_cast<Field4D&>(*result) : result->kinetic_density();
            for (int s = 0; s < ns; s++) {
                if (!radial.empty()) {
                    fields[part][s] += radial[ia][part][s];
                }
                target.scalar().mt()[ia] += fields[part][s];
                if (ns == 2) {
                    for (size_t i = 0; i < fields[part][s].size(); i++) {
                        target.vector(0).mt()[ia][i] += (1 - 2 * s) * fields[part][s][i];
                    }
                }
            }
        }
    }
    return result;
}

std::unique_ptr<Density>
Density::lapw_density_matrix_fields(K_point_set& ks__,
                                    std::vector<mdarray<std::complex<double>, 3>> const& matrices__) const
{
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || !has_kinetic_density() ||
                  ctx_.processing_unit() != device_t::CPU || ctx_.cfg().parameters().precision_wf() != "fp64" ||
                  ctx_.num_mag_dims() == 3 || ctx_.so_correction() || ctx_.valence_relativity() != relativity_t::none ||
                  ctx_.spfft<double>().processing_unit() != SPFFT_PU_HOST || ctx_.gvec_coarse().reduced() ||
                  matrices__.size() != ks__.spl_num_kpoints().local_size();
    if (!invalid) {
        size_t ik{0};
        for (auto it : ks__.spl_num_kpoints()) {
            auto const& p = matrices__[ik++];
            int n         = ks__.get<double>(it.i)->gklo_basis_size();
            if (p.size(0) != n || p.size(1) != n || p.size(2) != ctx_.num_spins() || !p.on_host()) {
                invalid = 1;
                continue;
            }
            double peak{0}, error{0};
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        auto value = p(i, j, s);
                        invalid |= !std::isfinite(value.real()) || !std::isfinite(value.imag());
                        peak  = std::max(peak, std::abs(value));
                        error = std::max(error, std::abs(value - std::conj(p(j, i, s))));
                    }
                }
            }
            invalid |= error > 1e-11 * std::max(1.0, peak);
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW coefficient fields require finite Hermitian scalar/collinear CPU fp64 PW/LO matrices");
    }
    size_t matrix_k{0};
    for (auto it : ks__.spl_num_kpoints()) {
        auto const& kp = *ks__.get<double>(it.i);
        auto const& p  = matrices__[matrix_k++];
        if (kp.comm().size() > 1) {
            std::vector<std::complex<double>> reference(p.at(memory_t::host), p.at(memory_t::host) + p.size());
            kp.comm().bcast(reference.data(), reference.size(), 0);
            for (size_t i = 0; i < p.size(); i++) {
                invalid |= std::abs(p[i] - reference[i]) > 1e-12 * std::max(1.0, std::abs(reference[i]));
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW coefficient density matrices must agree across each k-point communicator");
    }
    int ns = ctx_.num_spins(), ng = ctx_.gvec_coarse().num_gvec();
    std::array<mdarray<std::complex<double>, 2>, 2> fourier{mdarray<std::complex<double>, 2>({ng, ns}),
                                                            mdarray<std::complex<double>, 2>({ng, ns})};
    for (auto& field : fourier) {
        field.zero();
    }
    std::vector<mdarray<std::complex<double>, 3>> dm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, ns});
        dm.back().zero();
    }
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    size_t ik{0};
    for (auto it : ks__.spl_num_kpoints()) {
        auto const& kp = *ks__.get<double>(it.i);
        auto const& p  = matrices__[ik++];
        int n = kp.gklo_basis_size(), npw = kp.gkvec().num_gvec();
        for (int i = 0; i < kp.gkvec().count(); i++) {
            int row = kp.gkvec().offset() + i;
            auto gi = kp.gkvec().gvec(gvec_index_t::local(i));
            auto qi = kp.gkvec().gkvec_cart(gvec_index_t::local(i));
            for (int j = 0; j < npw; j++) {
                int ig = ctx_.gvec_coarse().index_by_gvec(gi - kp.gkvec().gvec(gvec_index_t::global(j)));
                if (ig < 0) {
                    invalid = 1;
                    continue;
                }
                double kinetic = 0.5 * dot(qi, kp.gkvec().gkvec_cart(gvec_index_t::global(j)));
                for (int s = 0; s < ns; s++) {
                    auto value = kp.weight() * p(row, j, s) / unit_cell_.omega();
                    fourier[0](ig, s) += value;
                    fourier[1](ig, s) += kinetic * value;
                }
            }
        }
        Matching_coefficients matching(unit_cell_, kp.gkvec());
        int lo_offset = npw;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& atom = unit_cell_.atom(ia);
            int mt = atom.type().mt_basis_size(), naw = atom.type().mt_aw_basis_size();
            mdarray<std::complex<double>, 2> alm({kp.gkvec().count(), naw}), map({mt, n});
            matching.generate<false>(atom, alm);
            map.zero();
            for (int i = 0; i < naw; i++) {
                for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                    map(i, kp.gkvec().offset() + ig) = alm(ig, i);
                }
            }
            kp.comm().allreduce(map.at(memory_t::host), map.size());
            for (int i = naw; i < mt; i++) {
                map(i, lo_offset++) = 1;
            }
            if (ia % kp.comm().size() != kp.comm().rank()) {
                continue;
            }
            mdarray<std::complex<double>, 2> work({mt, n}), local({mt, mt});
            for (int s = 0; s < ns; s++) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', mt, n, n, &one, map.at(memory_t::host), map.ld(), p.at(memory_t::host, 0, 0, s),
                              p.ld(), &zero, work.at(memory_t::host), work.ld());
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'C', mt, mt, n, &one, work.at(memory_t::host), work.ld(), map.at(memory_t::host),
                              map.ld(), &zero, local.at(memory_t::host), local.ld());
                for (int j = 0; j < mt; j++) {
                    for (int i = 0; i < mt; i++) {
                        // The atom operator stores conj(c_i) c_j, the transpose of ket/bra P.
                        dm[ia](i, j, s) += kp.weight() * local(j, i);
                    }
                }
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW PW products lie outside the density reciprocal grid");
    }
    auto result = std::make_unique<Density>(ctx_);
    result->zero();
    result->kinetic_density().zero();
    for (int part : {0, 1}) {
        ctx_.comm().allreduce(fourier[part].at(memory_t::host), fourier[part].size());
        for (size_t i = 0; i < fourier[part].size(); i++) {
            invalid |= !std::isfinite(fourier[part][i].real()) || !std::isfinite(fourier[part][i].imag());
        }
        if (invalid) {
            RTE_THROW("nonfinite LAPW coefficient Fourier fields");
        }
        auto& target = part == 0 ? static_cast<Field4D&>(*result) : result->kinetic_density();
        for (int s = 0; s < ns; s++) {
            for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
                int fine   = ctx_.gvec().gvec_base_mapping(ig);
                auto value = fourier[part](ctx_.gvec_coarse().offset() + ig, s);
                target.scalar().rg().f_pw_local(fine) += value;
                if (ns == 2) {
                    target.vector(0).rg().f_pw_local(fine) += double(1 - 2 * s) * value;
                }
            }
        }
        target.fft_transform(1);
    }
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        ctx_.comm().allreduce(dm[ia].at(memory_t::host), dm[ia].size());
        auto fields = LAPW_xc_atom(unit_cell_.atom(ia), ctx_.lmax_rho()).fields(dm[ia]);
        for (int part : {0, 1}) {
            auto& target = part == 0 ? static_cast<Field4D&>(*result) : result->kinetic_density();
            for (int s = 0; s < ns; s++) {
                target.scalar().mt()[ia] += fields[part][s];
                if (ns == 2) {
                    for (size_t i = 0; i < fields[part][s].size(); i++) {
                        target.vector(0).mt()[ia][i] += (1 - 2 * s) * fields[part][s][i];
                    }
                }
            }
        }
    }
    return result;
}

template <typename T>
std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density(K_point_set const& ks__) const
{
    PROFILE("sirius::Density::generate_smooth_kinetic_density");
    if (ctx_.processing_unit() != device_t::CPU || ctx_.spfft_coarse<T>().processing_unit() != SPFFT_PU_HOST) {
        RTE_THROW("smooth kinetic-energy density currently requires CPU orbitals and FFTs");
    }
    if (ctx_.num_mag_dims() == 3) {
        RTE_THROW("noncollinear kinetic-energy density is not implemented");
    }

    std::vector<Smooth_periodic_function<double>> coarse, result;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        coarse.emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
        result.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    }
    int nr = ctx_.spfft_coarse<T>().local_slice_size();
    std::vector<T> tau(nr);
    for (auto it : ks__.spl_num_kpoints()) {
        auto kp = ks__.get<T>(it.i);
        wf::Kinetic_density_operator<T> op(kp->spfft_transform(), *kp->gkvec_fft_sptr());
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            if (nbnd == 0) {
                continue;
            }
            wf::Wave_functions_fft<T> psi(kp->gkvec_fft_sptr(), kp->spinor_wave_functions(), wf::spin_index(ispn),
                                          wf::band_range(0, nbnd), wf::shuffle_to::fft_layout);
            std::fill(tau.begin(), tau.end(), T(0));
            for (int ib = 0; ib < psi.num_wf_local(); ib++) {
                auto band = psi.spl_num_wf().global_index(ib);
                T weight  = kp->weight() * kp->band_occupancy(band, ispn) / unit_cell_.omega();
                op.add_density(weight, psi.at(memory_t::host, 0, wf::band_index(ib)), tau.data());
            }
            for (int ir = 0; ir < nr; ir++) {
                coarse[ispn].value(ir) += tau[ir];
            }
        }
    }

    // Bands and k-points are distributed orthogonally to the real-space FFT slabs.
    auto const& comm = ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft();
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        auto ptr = nr ? &coarse[ispn].value(0) : nullptr;
        comm.allreduce(ptr, nr);
        coarse[ispn].fft_transform(-1);
        for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
            result[ispn].f_pw_local(ctx_.gvec().gvec_base_mapping(ig)) = coarse[ispn].f_pw_local(ig);
        }
        result[ispn].fft_transform(1);
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
