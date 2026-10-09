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
#include "lapw/dirac_basis.hpp"
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
    if (ctx_.num_mag_dims() == 3) {
        auto spinor_guess = [&](double n, std::array<double, 3> const& m,
                                std::array<std::array<double, 4>, 3> const& grad) {
            double length = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
            double up = guess(0.5 * (n + length)), dn = guess(0.5 * (n - length));
            std::array<double, 4> result{up + dn, 0, 0, 0};
            std::array<double, 3> axis{};
            for (int j = 0; j < 3; j++) {
                axis[j]       = length ? m[j] / length : 0;
                result[j + 1] = (up - dn) * axis[j];
            }
            if (add_von_weizsaecker) {
                // For R=(n I+m.sigma)/2, use sum_x (d_x sqrt(R))^2/2.
                // This positive matrix recovers both spin-channel guesses and includes spin-texture gradients.
                double su = std::sqrt(std::max(0.0, 0.5 * (n + length)));
                double sd = std::sqrt(std::max(0.0, 0.5 * (n - length)));
                double b  = n >= length ? (su + sd > 0 ? 0.5 / (su + sd) : 0) : (length ? 0.5 * (su - sd) / length : 0);
                for (int x = 0; x < 3; x++) {
                    double dm = std::inner_product(axis.begin(), axis.end(), grad[x].begin() + 1, 0.0);
                    double du = su > 0 ? 0.25 * (grad[x][0] + dm) / su : 0;
                    double dd = sd > 0 ? 0.25 * (grad[x][0] - dm) / sd : 0;
                    double a  = 0.5 * (du + dd);
                    result[0] += a * a;
                    for (int j = 0; j < 3; j++) {
                        double v = 0.5 * (du - dd) * axis[j] + b * (grad[x][j + 1] - axis[j] * dm);
                        result[0] += v * v;
                        result[j + 1] += 2 * a * v;
                    }
                }
            }
            return result;
        };
        std::vector<Smooth_periodic_vector_function<double>> grad;
        if (add_von_weizsaecker) {
            for (int j = 0; j < 4; j++) {
                grad.push_back(to_rg(gradient(component(j).rg())));
            }
        }
        for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
            std::array<std::array<double, 4>, 3> g{};
            for (size_t j = 0; j < grad.size(); j++) {
                for (int x = 0; x < 3; x++) {
                    g[x][j] = grad[j][x].value(ir);
                }
            }
            auto value = spinor_guess(rho().rg().value(ir),
                                      {mag(0).rg().value(ir), mag(1).rg().value(ir), mag(2).rg().value(ir)}, g);
            for (int j = 0; j < 4; j++) {
                tau.component(j).rg().value(ir) = value[j];
            }
        }
        if (ctx_.full_potential()) {
            SHT sht(device_t::CPU, ctx_.lmax_rho());
            for (auto it : unit_cell_.spl_num_atoms()) {
                std::vector<Ftp> fields;
                std::vector<std::vector<Ftp>> grad(4);
                for (int j = 0; j < 4; j++) {
                    fields.push_back(transform(sht, component(j).mt()[it.i]));
                    if (add_von_weizsaecker) {
                        auto g = gradient(component(j).mt()[it.i]);
                        for (int x = 0; x < 3; x++) {
                            grad[j].push_back(transform(sht, g[x]));
                        }
                    }
                }
                for (size_t i = 0; i < fields[0].size(); i++) {
                    std::array<std::array<double, 4>, 3> g{};
                    if (add_von_weizsaecker) {
                        for (int j = 0; j < 4; j++) {
                            for (int x = 0; x < 3; x++) {
                                g[x][j] = grad[j][x][i];
                            }
                        }
                    }
                    auto value = spinor_guess(fields[0][i], {fields[1][i], fields[2][i], fields[3][i]}, g);
                    for (int j = 0; j < 4; j++) {
                        fields[j][i] = value[j];
                    }
                }
                for (int j = 0; j < 4; j++) {
                    transform(sht, fields[j], tau.component(j).mt()[it.i]);
                }
            }
            for (int j = 0; j < 4; j++) {
                tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
            }
        }
        tau.fft_transform(-1);
        return;
    }
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
            if (ctx_.num_mag_dims() == 3 || ctx_.valence_relativity() == relativity_t::dirac) {
                for (int j = 0; j <= ctx_.num_mag_dims(); j++) {
                    tau.component(j).mt()[it.i][i] = spin_tau[j][i];
                }
                continue;
            }
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
    if (ctx_.num_mag_dims() == 3 || ctx_.valence_relativity() == relativity_t::dirac) {
        auto fields = LAPW_xc_atom(atom, ctx_.lmax_rho()).fields(density_matrix(ia__));
        return std::move(fields[1]);
    }
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
    if (ctx_.full_potential() || !type.is_paw() || (type.spin_orbit_coupling() && ctx_.num_mag_dims() != 3)) {
        RTE_THROW("PAW kinetic density requires a pseudopotential context and spinors for SOC partial waves");
    }
    if (ctx_.num_mag_dims() == 3) {
        auto ae = PAW_local_fields(type, true).fields(density_matrix(ia__), false);
        auto ps = PAW_local_fields(type, false, false).fields(density_matrix(ia__), false);
        return {std::move(ae[1]), std::move(ps[1])};
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

std::vector<std::array<std::vector<Flm>, 2>>
Density::lapw_radial_fields_response(K_point_set& ks__, std::vector<radial_basis_response_t> const& response__) const
{
    bool spinor = ctx_.num_mag_dims() == 3;
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || ctx_.cfg().parameters().precision_wf() != "fp64" ||
                  (ctx_.so_correction() && !spinor) || ctx_.valence_relativity() != relativity_t::none ||
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
        RTE_THROW("LAPW radial fields response requires nonrelativistic fp64 orbitals and compatible finite "
                  "responses");
    }
    int ns = spinor ? 4 : ctx_.num_spins(), blocks = spinor ? 3 : ns;
    std::vector<mdarray<std::complex<double>, 3>> dm, ddm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, blocks});
        ddm.emplace_back(std::array<index_range, 3>{n, n, blocks});
        dm.back().zero();
        ddm.back().zero();
    }
    for (auto it : ks__.spl_num_kpoints()) {
        auto& kp  = *ks__.get<double>(it.i);
        auto& psi = kp.spinor_wave_functions();
        if (psi.pw_coeffs(wf::spin_index(0)).on_device()) {
            psi.copy_to(memory_t::host);
        }
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
        for (int s = 0; s < ctx_.num_spinors(); s++) {
            for (int band = 0; band < ctx_.num_bands(); band++) {
                double weight = kp.weight() * kp.band_occupancy(band, s) / kp.comm().size();
                if (weight == 0) {
                    continue;
                }
                std::array<std::vector<std::complex<double>>, 2> all;
                for (int c = 0; c < ctx_.num_spinor_comp(); c++) {
                    all[c] = kp.spinor_wave_functions().gather(wf::spin_index(spinor ? c : s), wf::band_index(band));
                }
                int offset = kp.gkvec().num_gvec();
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    auto const& type = unit_cell_.atom(ia).type();
                    int n = type.mt_basis_size(), naw = type.mt_aw_basis_size();
                    std::array<std::vector<std::complex<double>>, 2> c, dc;
                    for (int comp = 0; comp < ctx_.num_spinor_comp(); comp++) {
                        c[comp].resize(n);
                        dc[comp].resize(n);
                        for (int i = 0; i < naw; i++) {
                            for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                                c[comp][i] += alm[ia](ig, i) * all[comp][kp.gkvec().offset() + ig];
                                dc[comp][i] += dalm[ia](ig, i) * all[comp][kp.gkvec().offset() + ig];
                            }
                        }
                        kp.comm().allreduce(c[comp].data(), naw);
                        kp.comm().allreduce(dc[comp].data(), naw);
                        for (int i = naw; i < n; i++) {
                            c[comp][i] = all[comp][offset + i];
                        }
                    }
                    for (int block = 0; block < (spinor ? 3 : 1); block++) {
                        int left = block == 2 ? 1 : block, right = block == 2 ? 0 : block;
                        for (int j = 0; j < n; j++) {
                            for (int i = 0; i < n; i++) {
                                dm[ia](i, j, spinor ? block : s) += weight * std::conj(c[left][i]) * c[right][j];
                                ddm[ia](i, j, spinor ? block : s) += weight * (std::conj(dc[left][i]) * c[right][j] +
                                                                               std::conj(c[left][i]) * dc[right][j]);
                            }
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
                if (!spinor || s == 0) {
                    for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                        field(0, ir) += d.core_fields[part][ir] / ((spinor ? 1 : ns) * y00);
                    }
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
    bool spinor             = ctx_.num_mag_dims() == 3;
    bool dirac              = ctx_.valence_relativity() == relativity_t::dirac;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&occupation_response, 1);
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || !has_kinetic_density() ||
                  ctx_.cfg().parameters().precision_wf() != "fp64" || (ctx_.so_correction() && !spinor) ||
                  (dirac && !radial__.empty()) ||
                  (!dirac && spinor && ctx_.valence_relativity() != relativity_t::none &&
                   ctx_.valence_relativity() != relativity_t::zora) ||
                  directions__.size() != ks__.spl_num_kpoints().local_size() ||
                  (occupation_response && occupation_directions__.size() != directions__.size());
    if (!invalid) {
        size_t ik{0};
        for (auto it : ks__.spl_num_kpoints()) {
            auto& psi = ks__.get<double>(it.i)->spinor_wave_functions();
            auto* d   = directions__[ik];
            invalid |= spinor && ks__.get<double>(it.i)->spfft_transform().type() != SPFFT_TRANS_C2C;
            if (occupation_response) {
                auto const& df = occupation_directions__[ik];
                if (df.size(0) != ctx_.num_bands() || df.size(1) != ctx_.num_spinors() || !df.on_host()) {
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
            if (psi.pw_coeffs(wf::spin_index(0)).on_device()) {
                psi.copy_to(memory_t::host);
            }
            if (d->pw_coeffs(wf::spin_index(0)).on_device()) {
                d->copy_to(memory_t::host);
            }
            for (int s = 0; s < (dirac ? 2 : ctx_.num_spins()); s++) {
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
        RTE_THROW("LAPW orbital fields response requires matching finite fp64 orbitals and valid directions");
    }
    auto result = std::make_unique<Density>(ctx_);
    result->zero();
    result->kinetic_density().zero();
    if (dirac) {
        result->generate_dirac_valence(ks__, &directions__, occupation_response ? &occupation_directions__ : nullptr);
        result->generate_mt_kinetic_density(false);
        result->fft_transform(1);
        result->kinetic_density().fft_transform(1);
        for (int s = 0; s <= ctx_.num_mag_dims(); s++) {
            result->component(s).mt().sync(unit_cell_.spl_num_atoms());
        }
        return result;
    }
    int ns = spinor ? 4 : ctx_.num_spins(), nr = ctx_.spfft_coarse<double>().local_slice_size();
    std::array<std::vector<Smooth_periodic_function<double>>, 2> coarse;
    for (int part : {0, 1}) {
        for (int s = 0; s < ns; s++) {
            coarse[part].emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
        }
    }
    std::vector<mdarray<std::complex<double>, 3>> dm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, spinor ? 3 : ns});
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
        if (spinor) {
            int nb = kp.num_occupied_bands();
            if (df) {
                for (int b = nb; b < ctx_.num_bands(); b++) {
                    if ((*df)(b, 0) != 0) {
                        nb = b + 1;
                    }
                }
            }
            if (!nb) {
                continue;
            }
            std::array<wf::Wave_functions_fft<double>, 2> psi, delta;
            std::array<std::vector<std::complex<double>>, 2> base, direction;
            for (int s = 0; s < 2; s++) {
                psi[s]   = wf::Wave_functions_fft<double>(kp.gkvec_fft_sptr(), kp.spinor_wave_functions(),
                                                          wf::spin_index(s), wf::band_range(0, nb),
                                                          wf::shuffle_to::fft_layout);
                delta[s] = wf::Wave_functions_fft<double>(kp.gkvec_fft_sptr(), d, wf::spin_index(s),
                                                          wf::band_range(0, nb), wf::shuffle_to::fft_layout);
                if (psi[s].on_device()) {
                    psi[s].copy_to(memory_t::host);
                }
                if (delta[s].on_device()) {
                    delta[s].copy_to(memory_t::host);
                }
                base[s].resize(nr);
                direction[s].resize(nr);
            }
            for (int b = 0; b < psi[0].num_wf_local(); b++) {
                int band  = psi[0].spl_num_wf().global_index(b);
                double w  = kp.weight() * kp.band_occupancy(band, 0) / unit_cell_.omega();
                double dw = df ? kp.weight() * (*df)(band, 0) / unit_cell_.omega() : 0;
                std::array<std::complex<double> const*, 2> p, q;
                for (int s = 0; s < 2; s++) {
                    p[s] = psi[s].at(memory_t::host, 0, wf::band_index(b));
                    q[s] = delta[s].at(memory_t::host, 0, wf::band_index(b));
                    for (auto pair : {std::make_pair(p[s], &base[s]), std::make_pair(q[s], &direction[s])}) {
                        fft.backward(reinterpret_cast<double const*>(pair.first), SPFFT_PU_HOST);
                        auto ptr = reinterpret_cast<std::complex<double> const*>(fft.space_domain_data(SPFFT_PU_HOST));
                        if (nr) {
                            std::copy(ptr, ptr + nr, pair.second->begin());
                        }
                    }
                }
                for (int ir = 0; ir < nr; ir++) {
                    double uu =
                            2 * w * std::real(std::conj(base[0][ir]) * direction[0][ir]) + dw * std::norm(base[0][ir]);
                    double dd =
                            2 * w * std::real(std::conj(base[1][ir]) * direction[1][ir]) + dw * std::norm(base[1][ir]);
                    auto cross = 2 * w *
                                         (std::conj(direction[0][ir]) * base[1][ir] +
                                          std::conj(base[0][ir]) * direction[1][ir]) +
                                 2 * dw * std::conj(base[0][ir]) * base[1][ir];
                    coarse[0][0].value(ir) += uu + dd;
                    coarse[0][1].value(ir) += uu - dd;
                    coarse[0][2].value(ir) += cross.real();
                    coarse[0][3].value(ir) += cross.imag();
                }
                std::array<double*, 4> t;
                for (int j = 0; j < 4; j++) {
                    t[j] = nr ? &coarse[1][j].value(0) : nullptr;
                }
                kinetic.add_spinor_density_response(w, p, q, t);
                if (dw) {
                    kinetic.add_spinor_density(dw, p, t);
                }
            }
            for (int b = 0; b < nb; b++) {
                double w  = kp.weight() * kp.band_occupancy(b, 0) / kp.comm().size();
                double dw = df ? kp.weight() * (*df)(b, 0) / kp.comm().size() : 0;
                std::array<std::vector<std::complex<double>>, 2> p, q;
                for (int s = 0; s < 2; s++) {
                    p[s] = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(b));
                    q[s] = d.gather(wf::spin_index(s), wf::band_index(b));
                }
                int offset = kp.gkvec().num_gvec();
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    int n = unit_cell_.atom(ia).type().mt_basis_size();
                    for (int block = 0; block < 3; block++) {
                        int a = block == 2 ? 1 : block, c = block == 2 ? 0 : block;
                        for (int j = 0; j < n; j++) {
                            for (int i = 0; i < n; i++) {
                                dm[ia](i, j, block) += w * (std::conj(q[a][offset + i]) * p[c][offset + j] +
                                                            std::conj(p[a][offset + i]) * q[c][offset + j]) +
                                                       dw * std::conj(p[a][offset + i]) * p[c][offset + j];
                            }
                        }
                    }
                    offset += n;
                }
            }
            continue;
        }
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
            if (psi.on_device()) {
                psi.copy_to(memory_t::host);
            }
            if (delta.on_device()) {
                delta.copy_to(memory_t::host);
            }
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
                target.component(spinor ? s : 0).rg().f_pw_local(fine) += value;
                if (!spinor && ns == 2) {
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
                target.component(spinor ? s : 0).mt()[ia] += fields[part][s];
                if (!spinor && ns == 2) {
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
    return lapw_density_matrix_fields_impl(ks__, matrices__, nullptr, nullptr);
}

std::unique_ptr<Density>
Density::lapw_density_matrix_radial_response(K_point_set& ks__,
                                             std::vector<mdarray<std::complex<double>, 3>> const& matrices__,
                                             std::vector<radial_basis_response_t> const& radial__) const
{
    return lapw_density_matrix_fields_impl(ks__, matrices__, &radial__, nullptr);
}

std::unique_ptr<Density>
Density::lapw_density_matrix_geometry_response(K_point_set& ks__,
                                               std::vector<mdarray<std::complex<double>, 3>> const& matrices__,
                                               xc_geometry_direction_t const& direction__) const
{
    return lapw_density_matrix_fields_impl(ks__, matrices__, nullptr, &direction__);
}

std::unique_ptr<Density>
Density::lapw_density_matrix_fields_impl(K_point_set& ks__,
                                         std::vector<mdarray<std::complex<double>, 3>> const& matrices__,
                                         std::vector<radial_basis_response_t> const* radial__,
                                         xc_geometry_direction_t const* geometry__) const
{
    bool spinor = ctx_.num_mag_dims() == 3;
    bool dirac  = ctx_.valence_relativity() == relativity_t::dirac;
    int ns      = dirac ? ctx_.num_mag_dims() + 1 : (spinor ? 4 : ctx_.num_spins());
    int blocks  = dirac ? 1 : ns;
    int invalid = &ks__.ctx() != &ctx_ || !ctx_.full_potential() || !has_kinetic_density() ||
                  ctx_.cfg().parameters().precision_wf() != "fp64" || (ctx_.so_correction() && !spinor) ||
                  (radial__ && geometry__) ||
                  (spinor && !dirac && ctx_.valence_relativity() != relativity_t::none &&
                   ctx_.valence_relativity() != relativity_t::zora) ||
                  ctx_.gvec_coarse().reduced() || matrices__.size() != ks__.spl_num_kpoints().local_size();
    std::vector<double> geometry_values;
    if (geometry__) {
        invalid |= geometry__->atoms.size() != unit_cell_.num_atoms();
        geometry_values.assign(geometry__->lattice.begin(), geometry__->lattice.end());
        for (auto const& atom : geometry__->atoms) {
            geometry_values.insert(geometry_values.end(), atom.begin(), atom.end());
        }
        for (auto x : geometry_values) {
            invalid |= !std::isfinite(x);
        }
    }
    if (radial__) {
        invalid |= (unit_cell_.num_core_electrons() != 0 && ctx_.core_relativity() != relativity_t::none &&
                    ctx_.core_relativity() != relativity_t::zora && ctx_.core_relativity() != relativity_t::dirac) ||
                   radial__->size() != unit_cell_.num_atom_symmetry_classes();
        if (!invalid) {
            for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
                auto const& type = unit_cell_.atom_symmetry_class(ic).atom_type();
                auto const& d    = (*radial__)[ic];
                if (!d.radial_functions.on_host() || !d.surface_derivatives.on_host()) {
                    invalid = 1;
                    continue;
                }
                invalid |= d.radial_functions.size(0) != type.num_mt_points() ||
                           d.radial_functions.size(1) != type.mt_radial_basis_size() ||
                           d.radial_functions.size(2) != (dirac ? 4 : 2) ||
                           d.surface_derivatives.size(0) != (dirac ? 6 : 3) ||
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
    }
    if (!invalid) {
        size_t ik{0};
        for (auto it : ks__.spl_num_kpoints()) {
            auto const& p  = matrices__[ik++];
            auto const& kp = *ks__.get<double>(it.i);
            int n          = dirac ? 2 * kp.num_gkvec() + unit_cell_.mt_lo_basis_size() : kp.gklo_basis_size();
            if (p.size(0) != n || p.size(1) != n || p.size(2) != blocks || !p.on_host()) {
                invalid = 1;
                continue;
            }
            double peak{0}, error{0};
            for (int s = 0; s < blocks; s++) {
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
        RTE_THROW("LAPW coefficient fields require finite Hermitian host fp64 PW/LO matrices "
                  "and matching finite radial/geometry directions");
    }
    r3::matrix<double> strain;
    mdarray<double, 2> atom_directions;
    if (geometry__) {
        auto root = geometry_values;
        ctx_.comm().bcast(root.data(), root.size(), 0);
        invalid = root != geometry_values;
        ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("LAPW field geometry directions must agree across the context communicator");
        }
        r3::matrix<double> lattice;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                lattice(x, y) = geometry__->lattice[x + 3 * y];
            }
        }
        strain          = dot(lattice, unit_cell_.inverse_lattice_vectors());
        atom_directions = mdarray<double, 2>({3, unit_cell_.num_atoms()});
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            for (int x = 0; x < 3; x++) {
                atom_directions(x, ia) = geometry__->atoms[ia][x];
            }
        }
    }
    double volume_response = strain(0, 0) + strain(1, 1) + strain(2, 2);
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
    int ng = ctx_.gvec_coarse().num_gvec();
    std::array<mdarray<std::complex<double>, 2>, 2> fourier{mdarray<std::complex<double>, 2>({ng, ns}),
                                                            mdarray<std::complex<double>, 2>({ng, ns})};
    for (auto& field : fourier) {
        field.zero();
    }
    std::vector<mdarray<std::complex<double>, 3>> dm, ddm;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int n = unit_cell_.atom(ia).type().mt_basis_size();
        dm.emplace_back(std::array<index_range, 3>{n, n, blocks});
        dm.back().zero();
        if (radial__ || geometry__) {
            ddm.emplace_back(std::array<index_range, 3>{n, n, blocks});
            ddm.back().zero();
        }
    }
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    size_t ik{0};
    for (auto it : ks__.spl_num_kpoints()) {
        auto const& kp = *ks__.get<double>(it.i);
        auto const& p  = matrices__[ik++];
        int npw        = kp.gkvec().num_gvec();
        int n          = dirac ? 2 * npw + unit_cell_.mt_lo_basis_size() : kp.gklo_basis_size();
        std::unique_ptr<Dirac_lapw_basis> dirac_basis;
        if (dirac) {
            dirac_basis = std::make_unique<Dirac_lapw_basis>(unit_cell_, kp.gkvec());
        }
        if (dirac && !radial__) {
            std::vector<std::array<std::complex<double>, 4>> spinors(2 * npw), tangent(2 * npw);
            for (int s : {0, 1}) {
                for (int ig = 0; ig < npw; ig++) {
                    auto q = kp.gkvec().gkvec_cart(gvec_index_t::global(ig));
                    Dirac_plane_wave pw(q);
                    spinors[s * npw + ig] = pw.spinors()[s];
                    if (geometry__) {
                        tangent[s * npw + ig] = pw.response(dot(transpose(strain), q) * -1.0)[s];
                    }
                }
            }
            for (int i = 0; i < kp.gkvec().count(); i++) {
                int row  = kp.gkvec().offset() + i;
                auto gi  = kp.gkvec().gvec(gvec_index_t::local(i));
                auto qi  = kp.gkvec().gkvec_cart(gvec_index_t::local(i));
                auto dqi = geometry__ ? dot(transpose(strain), qi) * -1.0 : r3::vector<double>{0, 0, 0};
                for (int j = 0; j < npw; j++) {
                    int ig = ctx_.gvec_coarse().index_by_gvec(gi - kp.gkvec().gvec(gvec_index_t::global(j)));
                    if (ig < 0) {
                        invalid = 1;
                        continue;
                    }
                    auto qj        = kp.gkvec().gkvec_cart(gvec_index_t::global(j));
                    auto dqj       = geometry__ ? dot(transpose(strain), qj) * -1.0 : r3::vector<double>{0, 0, 0};
                    double kinetic = 0.5 * dot(qi, qj), dkinetic = 0.5 * (dot(dqi, qj) + dot(qi, dqj));
                    for (int a : {0, 1}) {
                        for (int b : {0, 1}) {
                            auto const& u   = spinors[a * npw + row];
                            auto const& v   = spinors[b * npw + j];
                            auto weight     = kp.weight() * p(a * npw + row, b * npw + j, 0) / unit_cell_.omega();
                            auto accumulate = [&](auto const& left, auto const& right, double rho, double tau) {
                                for (int lower : {0, 1}) {
                                    int up = 2 * lower, down = up + 1;
                                    double sign = lower ? -1 : 1;
                                    auto uu = left[up] * std::conj(right[up]), dd = left[down] * std::conj(right[down]);
                                    auto ud = left[up] * std::conj(right[down]), du = left[down] * std::conj(right[up]);
                                    std::array<std::complex<double>, 4> fields{
                                            uu + dd, sign * (uu - dd), sign * (ud + du),
                                            sign * std::complex<double>(0, 1) * (ud - du)};
                                    for (int s = 0; s < ns; s++) {
                                        fourier[0](ig, s) += rho * weight * fields[s];
                                        fourier[1](ig, s) += tau * weight * fields[s];
                                    }
                                }
                            };
                            if (geometry__) {
                                accumulate(u, v, -volume_response, dkinetic - volume_response * kinetic);
                                accumulate(tangent[a * npw + row], v, 1.0, kinetic);
                                accumulate(u, tangent[b * npw + j], 1.0, kinetic);
                            } else {
                                accumulate(u, v, 1.0, kinetic);
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; !dirac && !radial__ && i < kp.gkvec().count(); i++) {
            int row  = kp.gkvec().offset() + i;
            auto gi  = kp.gkvec().gvec(gvec_index_t::local(i));
            auto qi  = kp.gkvec().gkvec_cart(gvec_index_t::local(i));
            auto dqi = geometry__ ? dot(transpose(strain), qi) * -1.0 : r3::vector<double>{0, 0, 0};
            for (int j = 0; j < npw; j++) {
                int ig = ctx_.gvec_coarse().index_by_gvec(gi - kp.gkvec().gvec(gvec_index_t::global(j)));
                if (ig < 0) {
                    invalid = 1;
                    continue;
                }
                auto qj         = kp.gkvec().gkvec_cart(gvec_index_t::global(j));
                double kinetic  = 0.5 * dot(qi, qj);
                double dkinetic = geometry__ ? 0.5 * (dot(dqi, qj) - dot(qi, dot(transpose(strain), qj))) : 0;
                for (int s = 0; s < ns; s++) {
                    auto value = kp.weight() * p(row, j, s) / unit_cell_.omega();
                    fourier[0](ig, s) += (geometry__ ? -volume_response : 1.0) * value;
                    fourier[1](ig, s) += (geometry__ ? dkinetic - volume_response * kinetic : kinetic) * value;
                }
            }
        }
        Matching_coefficients matching(unit_cell_, kp.gkvec());
        int lo_offset = npw;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& atom = unit_cell_.atom(ia);
            int mt = atom.type().mt_basis_size(), naw = atom.type().mt_aw_basis_size();
            mdarray<std::complex<double>, 2> alm({kp.gkvec().count(), naw}), map({mt, n});
            if (!dirac) {
                matching.generate<false>(atom, alm);
            }
            mdarray<std::complex<double>, 2> delta_map;
            if (radial__ || geometry__) {
                if (dirac) {
                    delta_map = geometry__
                                        ? dirac_basis->atom_coefficients_geometry_response(ia, atom_directions, strain)
                                        : dirac_basis->atom_coefficients_response(ia, *radial__);
                } else {
                    mdarray<std::complex<double>, 2> dalm({kp.gkvec().count(), naw});
                    if (radial__) {
                        matching.generate_potential_response<false>(atom, (*radial__)[atom.symmetry_class().id()],
                                                                    dalm);
                    } else {
                        auto const& d = geometry__->atoms[ia];
                        matching.generate_geometry_response<false>(atom, {d[0], d[1], d[2]}, strain, dalm);
                    }
                    delta_map = mdarray<std::complex<double>, 2>({mt, n});
                    delta_map.zero();
                    for (int i = 0; i < naw; i++) {
                        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                            delta_map(i, kp.gkvec().offset() + ig) = dalm(ig, i);
                        }
                    }
                    kp.comm().allreduce(delta_map.at(memory_t::host), delta_map.size());
                }
            }
            map.zero();
            if (dirac) {
                auto const& full = dirac_basis->atom_coefficients(ia);
                std::copy_n(full.at(memory_t::host), full.size(), map.at(memory_t::host));
            } else {
                for (int i = 0; i < naw; i++) {
                    for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                        map(i, kp.gkvec().offset() + ig) = alm(ig, i);
                    }
                }
                kp.comm().allreduce(map.at(memory_t::host), map.size());
                for (int i = naw; i < mt; i++) {
                    map(i, lo_offset++) = 1;
                }
            }
            if (ia % kp.comm().size() != kp.comm().rank()) {
                continue;
            }
            mdarray<std::complex<double>, 2> work({mt, n}), local({mt, mt});
            for (int s = 0; s < blocks; s++) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', mt, n, n, &one, map.at(memory_t::host), map.ld(), p.at(memory_t::host, 0, 0, s),
                              p.ld(), &zero, work.at(memory_t::host), work.ld());
                if (!geometry__) {
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
                if (radial__ || geometry__) {
                    la::wrap(la::lib_t::blas)
                            .gemm('N', 'C', mt, mt, n, &one, work.at(memory_t::host), work.ld(),
                                  delta_map.at(memory_t::host), delta_map.ld(), &zero, local.at(memory_t::host),
                                  local.ld());
                    for (int j = 0; j < mt; j++) {
                        for (int i = 0; i < mt; i++) {
                            ddm[ia](i, j, s) += kp.weight() * (local(j, i) + std::conj(local(i, j)));
                        }
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
                target.component(spinor || dirac ? s : 0).rg().f_pw_local(fine) += value;
                if (!spinor && !dirac && ns == 2) {
                    target.vector(0).rg().f_pw_local(fine) += double(1 - 2 * s) * value;
                }
            }
        }
        target.fft_transform(1);
    }
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& matrix = geometry__ ? ddm[ia] : dm[ia];
        ctx_.comm().allreduce(matrix.at(memory_t::host), matrix.size());
        auto const& atom = unit_cell_.atom(ia);
        LAPW_xc_atom op(atom, ctx_.lmax_rho());
        auto fields = op.fields(matrix, radial__ ? &(*radial__)[atom.symmetry_class().id()] : nullptr);
        if (radial__) {
            ctx_.comm().allreduce(ddm[ia].at(memory_t::host), ddm[ia].size());
            auto matching = op.fields(ddm[ia]);
            for (int part : {0, 1}) {
                for (int s = 0; s < ns; s++) {
                    fields[part][s] += matching[part][s];
                    if (!(spinor || dirac) || s == 0) {
                        for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                            fields[part][s](0, ir) += (*radial__)[atom.symmetry_class().id()].core_fields[part][ir] /
                                                      (((spinor || dirac) ? 1 : ns) * y00);
                        }
                    }
                }
            }
        }
        for (int part : {0, 1}) {
            auto& target = part == 0 ? static_cast<Field4D&>(*result) : result->kinetic_density();
            for (int s = 0; s < ns; s++) {
                target.component(spinor || dirac ? s : 0).mt()[ia] += fields[part][s];
                if (!spinor && !dirac && ns == 2) {
                    for (size_t i = 0; i < fields[part][s].size(); i++) {
                        target.vector(0).mt()[ia][i] += (1 - 2 * s) * fields[part][s][i];
                    }
                }
            }
        }
    }
    return result;
}

void
Density::generate_dirac_valence(K_point_set const& ks__, std::vector<wf::Wave_functions<double>*> const* directions__,
                                std::vector<mdarray<double, 2>> const* occupations__)
{
    if (&ks__.ctx() != &ctx_ || ctx_.valence_relativity() != relativity_t::dirac || !ctx_.full_potential() ||
        ctx_.cfg().parameters().precision_wf() != "fp64") {
        RTE_THROW("Dirac density requires fp64 full-j wave functions from the same context");
    }
    int nc = ctx_.num_mag_dims() + 1, nr = ctx_.spfft_coarse<double>().local_slice_size();
    density_matrix_->zero();
    for (int s = 0; s < nc; s++) {
        rho_mag_coarse_[s]->zero();
    }
    std::vector<Smooth_periodic_function<double>> tau_coarse;
    if (kinetic_density_) {
        kinetic_density_->zero();
        for (int s = 0; s < nc; s++) {
            tau_coarse.emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
            tau_coarse.back().zero();
        }
    }
    size_t ik{0};
    for (auto it : ks__.spl_num_kpoints()) {
        auto kp         = ks__.get<double>(it.i);
        auto& psi       = kp->spinor_wave_functions();
        auto* direction = directions__ ? (*directions__)[ik] : nullptr;
        auto* df        = occupations__ ? &(*occupations__)[ik] : nullptr;
        ik++;
        int nb = kp->num_occupied_bands(0);
        if (df) {
            for (int b = nb; b < ctx_.num_bands(); b++) {
                if ((*df)(b, 0) != 0) {
                    nb = b + 1;
                }
            }
        }
        // MT coefficients contain the whole full-j spinor once, independently of the PW slots.
        for (auto atom : psi.spl_num_atoms()) {
            auto& dm = density_matrix(atom.i);
            int nm   = unit_cell_.atom(atom.i).mt_basis_size();
            for (int b = 0; b < nb; b++) {
                double weight = kp->weight() * kp->band_occupancy(b, 0);
                double dw     = df ? kp->weight() * (*df)(b, 0) : 0;
                for (int j = 0; j < nm; j++) {
                    auto cj = psi.mt_coeffs(j, atom.li, wf::spin_index(0), wf::band_index(b));
                    for (int i = 0; i < nm; i++) {
                        auto ci = psi.mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b));
                        if (direction) {
                            auto di = direction->mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b));
                            auto dj = direction->mt_coeffs(j, atom.li, wf::spin_index(0), wf::band_index(b));
                            dm(i, j, 0) += weight * (std::conj(di) * cj + std::conj(ci) * dj) + dw * std::conj(ci) * cj;
                        } else {
                            dm(i, j, 0) += weight * std::conj(ci) * cj;
                        }
                    }
                }
            }
        }
        if (!nb) {
            continue;
        }
        auto guard = psi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
        std::array<wf::Wave_functions_fft<double>, 2> shuffled, delta;
        for (int s : {0, 1}) {
            shuffled[s] = wf::Wave_functions_fft<double>(kp->gkvec_fft_sptr(), psi, wf::spin_index(s),
                                                         wf::band_range(0, nb), wf::shuffle_to::fft_layout);
            if (shuffled[s].on_device()) {
                shuffled[s].copy_to(memory_t::host);
            }
            if (direction) {
                delta[s] = wf::Wave_functions_fft<double>(kp->gkvec_fft_sptr(), *direction, wf::spin_index(s),
                                                          wf::band_range(0, nb), wf::shuffle_to::fft_layout);
                if (delta[s].on_device()) {
                    delta[s].copy_to(memory_t::host);
                }
            }
        }
        auto& fft      = kp->spfft_transform();
        auto const& gk = *kp->gkvec_fft_sptr();
        if (fft.local_slice_size() != nr) {
            RTE_THROW("incompatible Dirac wave-function and density FFT slabs");
        }
        wf::Kinetic_density_operator<double> op(fft, gk);
        std::array<std::vector<std::complex<double>>, 4> pw, real, dpw, dreal;
        std::array<std::vector<double>, 4> tau;
        for (int a = 0; a < 4; a++) {
            pw[a].resize(gk.count());
            real[a].resize(nr);
            tau[a].resize(nr);
            if (direction) {
                dpw[a].resize(gk.count());
                dreal[a].resize(nr);
            }
        }
        for (int ib = 0; ib < shuffled[0].num_wf_local(); ib++) {
            auto band     = shuffled[0].spl_num_wf().global_index(ib);
            double weight = kp->weight() * kp->band_occupancy(band, 0) / unit_cell_.omega();
            double dw     = df ? kp->weight() * (*df)(band, 0) / unit_cell_.omega() : 0;
            for (int ig = 0; ig < gk.count(); ig++) {
                auto u = Dirac_plane_wave(gk.gkvec_cart(ig)).spinors();
                for (int a = 0; a < 4; a++) {
                    pw[a][ig] = shuffled[0].pw_coeffs(ig, wf::band_index(ib)) * u[0][a] +
                                shuffled[1].pw_coeffs(ig, wf::band_index(ib)) * u[1][a];
                    if (direction) {
                        dpw[a][ig] = delta[0].pw_coeffs(ig, wf::band_index(ib)) * u[0][a] +
                                     delta[1].pw_coeffs(ig, wf::band_index(ib)) * u[1][a];
                    }
                }
            }
            for (int a = 0; a < 4; a++) {
                fft.backward(reinterpret_cast<double const*>(pw[a].data()), SPFFT_PU_HOST);
                auto ptr = reinterpret_cast<std::complex<double> const*>(fft.space_domain_data(SPFFT_PU_HOST));
                if (nr) {
                    std::copy_n(ptr, nr, real[a].data());
                }
                if (direction) {
                    fft.backward(reinterpret_cast<double const*>(dpw[a].data()), SPFFT_PU_HOST);
                    ptr = reinterpret_cast<std::complex<double> const*>(fft.space_domain_data(SPFFT_PU_HOST));
                    if (nr) {
                        std::copy_n(ptr, nr, dreal[a].data());
                    }
                }
            }
            for (int lower : {0, 1}) {
                double sign = lower ? -1 : 1;
                for (int ir = 0; ir < nr; ir++) {
                    auto up = real[2 * lower][ir], down = real[2 * lower + 1][ir];
                    auto cross = std::conj(up) * down;
                    std::array<double, 4> value{std::norm(up) + std::norm(down),
                                                sign * (std::norm(up) - std::norm(down)), sign * 2 * cross.real(),
                                                sign * 2 * cross.imag()};
                    for (int s = 0; s < nc; s++) {
                        rho_mag_coarse_[s]->value(ir) += (direction ? dw : weight) * value[s];
                    }
                    if (direction) {
                        auto du = dreal[2 * lower][ir], dd = dreal[2 * lower + 1][ir];
                        double uu = 2 * std::real(std::conj(up) * du);
                        double dn = 2 * std::real(std::conj(down) * dd);
                        auto dc   = std::conj(du) * down + std::conj(up) * dd;
                        std::array<double, 4> response{uu + dn, sign * (uu - dn), sign * 2 * dc.real(),
                                                       sign * 2 * dc.imag()};
                        for (int s = 0; s < nc; s++) {
                            rho_mag_coarse_[s]->value(ir) += weight * response[s];
                        }
                    }
                }
                if (kinetic_density_) {
                    for (auto& v : tau) {
                        std::fill(v.begin(), v.end(), 0);
                    }
                    std::array<std::complex<double> const*, 2> p{pw[2 * lower].data(), pw[2 * lower + 1].data()};
                    std::array<double*, 4> t{tau[0].data(), tau[1].data(), tau[2].data(), tau[3].data()};
                    if (direction) {
                        op.add_spinor_density_response(weight, p, {dpw[2 * lower].data(), dpw[2 * lower + 1].data()},
                                                       t);
                        if (dw) {
                            op.add_spinor_density(dw, p, t);
                        }
                    } else {
                        op.add_spinor_density(weight, p, t);
                    }
                    for (int s = 0; s < nc; s++) {
                        for (int ir = 0; ir < nr; ir++) {
                            tau_coarse[s].value(ir) += (s ? sign : 1) * tau[s][ir];
                        }
                    }
                }
            }
        }
    }
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& dm = density_matrix(ia);
        ctx_.comm().allreduce(dm.at(memory_t::host), dm.size());
    }
    auto const& comm = ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft();
    auto finish      = [&](auto& coarse, auto& fine) {
        comm.allreduce(nr ? &coarse.value(0) : nullptr, nr);
        coarse.fft_transform(-1);
        for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
            fine.f_pw_local(ctx_.gvec().gvec_base_mapping(ig)) = coarse.f_pw_local(ig);
        }
    };
    for (int s = 0; s < nc; s++) {
        finish(*rho_mag_coarse_[s], component(s).rg());
        if (kinetic_density_) {
            finish(tau_coarse[s], kinetic_density_->component(s).rg());
        }
    }
    generate_valence_mt();
}

template <typename T>
std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density(K_point_set const& ks__) const
{
    PROFILE("sirius::Density::generate_smooth_kinetic_density");
    if (&ks__.ctx() != &ctx_) {
        RTE_THROW("kinetic density requires k-points from the same context");
    }
    if (ctx_.valence_relativity() == relativity_t::dirac) {
        RTE_THROW("use native Dirac density generation to reconstruct all four physical components");
    }

    bool spinor    = ctx_.num_mag_dims() == 3;
    int components = spinor ? 4 : ctx_.num_spins();
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
            if (psi[ispn].on_device()) {
                psi[ispn].copy_to(memory_t::host);
            }
            if (!spinor) {
                for (int ib = 0; ib < psi[ispn].num_wf_local(); ib++) {
                    auto band = psi[ispn].spl_num_wf().global_index(ib);
                    T weight  = kp->weight() * kp->band_occupancy(band, ispn) / unit_cell_.omega();
                    op.add_density(weight, psi[ispn].at(memory_t::host, 0, wf::band_index(ib)), tau[ispn].data());
                }
            }
        }
        if (spinor && kp->num_occupied_bands()) {
            for (int ib = 0; ib < psi[0].num_wf_local(); ib++) {
                auto band = psi[0].spl_num_wf().global_index(ib);
                T weight  = kp->weight() * kp->band_occupancy(band, 0) / unit_cell_.omega();
                op.add_spinor_density(weight,
                                      {psi[0].at(memory_t::host, 0, wf::band_index(ib)),
                                       psi[1].at(memory_t::host, 0, wf::band_index(ib))},
                                      {tau[0].data(), tau[1].data(), tau[2].data(), tau[3].data()});
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
