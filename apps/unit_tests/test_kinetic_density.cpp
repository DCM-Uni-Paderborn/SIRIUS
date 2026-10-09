/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <chrono>
#include <filesystem>
#include <sirius.hpp>
#include <testing.hpp>
#include "core/wf/kinetic_density_operator.hpp"
#include "hamiltonian/local_operator.hpp"

using namespace sirius;

template <typename T>
int
test_kinetic_density(bool gamma__, SpfftProcessingUnitType processing_unit__)
{
    auto const& comm = mpi::Communicator::world();
    r3::matrix<double> reciprocal{{1.1, 0.2, 0.1}, {0.0, 0.9, 0.15}, {0.0, 0.0, 1.2}};
    r3::vector<double> k = gamma__ ? r3::vector<double>(0, 0, 0) : r3::vector<double>(0.23, -0.17, 0.11);
    fft::Gvec gv(k, reciprocal, 2.5, comm, gamma__);
    fft::Gvec_fft gvp(gv, comm, mpi::Communicator::self());
    int const n = 12;
    auto spl_z  = fft::split_z_dimension(n, comm);
    fft::spfft_grid_type<T> grid(n, n, n, gvp.zcol_count(), spl_z.local_size(), processing_unit__, 1, comm.native(),
                                 SPFFT_EXCH_DEFAULT);
    fft::spfft_transform_type<T> transform(grid.create_transform(
            processing_unit__, gamma__ ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C, n, n, n, spl_z.local_size(), gvp.count(),
            SPFFT_INDEX_TRIPLETS, gvp.gvec_array().at(memory_t::host)));
    wf::Kinetic_density_operator<T> op(transform, gvp);

    auto coefficient = [&](r3::vector<int> const& g, int shift) -> std::complex<T> {
        if (dot(g, g) > 3) {
            return 0;
        }
        if (dot(g, g) == 0) {
            return T(0.13 + 0.02 * shift);
        }
        return {T(0.09 * std::cos(g[0] + 2 * g[1] + 3 * g[2] + shift)),
                T(0.07 * std::sin(2 * g[0] - g[1] + g[2] + shift))};
    };
    std::vector<std::complex<T>> phi(gvp.count()), direction(gvp.count()), hphi(gvp.count()), hd(gvp.count());
    for (int ig = 0; ig < gvp.count(); ig++) {
        r3::vector<int> g(&gvp.gvec_array()(0, ig));
        phi[ig]       = coefficient(g, 0);
        direction[ig] = coefficient(g, 1);
    }

    double const volume     = std::pow(twopi, 3) / std::abs(reciprocal.det());
    double const occupation = 0.73;
    T weight                = occupation / volume;
    double const dv         = volume / (n * n * n);
    int nr                  = transform.local_slice_size();
    std::vector<T> tau(nr, 0), potential(nr), response(nr, 0);
    op.add_density(weight, phi.data(), tau.data());
    op.add_density_response(weight, phi.data(), direction.data(), response.data());
    std::array<std::vector<T>, 4> spin_tau, spin_potential, spin_response;
    for (int j = 0; j < 4; j++) {
        spin_tau[j].resize(nr, 0);
        spin_potential[j].resize(nr, 0);
        spin_response[j].resize(nr, 0);
    }
    if (!gamma__) {
        op.add_spinor_density(weight, {phi.data(), direction.data()},
                              {spin_tau[0].data(), spin_tau[1].data(), spin_tau[2].data(), spin_tau[3].data()});
        auto rotated = direction;
        for (auto& value : rotated) {
            value *= std::complex<T>(0, 1);
        }
        op.add_spinor_density(weight, {phi.data(), rotated.data()},
                              {spin_potential[0].data(), spin_potential[1].data(), spin_potential[2].data(),
                               spin_potential[3].data()});
        rotated = phi;
        for (auto& value : rotated) {
            value *= std::complex<T>(0, 1);
        }
        op.add_spinor_density_response(
                weight, {phi.data(), direction.data()}, {direction.data(), rotated.data()},
                {spin_response[0].data(), spin_response[1].data(), spin_response[2].data(), spin_response[3].data()});
    }

    double error{0}, response_error{0}, spin_error{0}, rotation_error{0}, spin_response_error{0};
    for (int iz = 0; iz < transform.local_z_length(); iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                int ir = ix + n * (iy + n * iz);
                r3::vector<double> r(double(ix) / n, double(iy) / n, double(iz + transform.local_z_offset()) / n);
                std::array<std::complex<double>, 3> grad{}, delta_grad{};
                for (int ig = 0; ig < gv.num_gvec(); ig++) {
                    auto g     = gv.gvec(gvec_index_t::global(ig));
                    auto q     = dot(reciprocal, g + k);
                    auto c     = static_cast<std::complex<double>>(coefficient(g, 0));
                    auto d     = static_cast<std::complex<double>>(coefficient(g, 1));
                    auto phase = std::exp(std::complex<double>(0, twopi * dot(g, r)));
                    for (int x = 0; x < 3; x++) {
                        auto z  = std::complex<double>(0, q[x]) * c * phase;
                        auto dz = std::complex<double>(0, q[x]) * d * phase;
                        grad[x] += gamma__ ? std::complex<double>(2 * z.real(), 0) : z;
                        delta_grad[x] += gamma__ ? std::complex<double>(2 * dz.real(), 0) : dz;
                    }
                }
                double reference{0};
                for (auto z : grad) {
                    reference += 0.5 * weight * std::norm(z);
                }
                double delta_reference{0};
                for (int x = 0; x < 3; x++) {
                    delta_reference += weight * std::real(std::conj(grad[x]) * delta_grad[x]);
                }
                response_error = std::max(response_error, std::abs(response[ir] - delta_reference));
                error          = std::max(error, std::abs(tau[ir] - reference));
                potential[ir]  = 0.8 + 0.3 * std::cos(twopi * r[0]) - 0.2 * std::sin(twopi * r[2]);
                if (!gamma__) {
                    std::array<double, 4> reference{}, delta{};
                    for (int x = 0; x < 3; x++) {
                        double up = 0.5 * weight * std::norm(grad[x]), dn = 0.5 * weight * std::norm(delta_grad[x]);
                        auto cross = double(weight) * std::conj(grad[x]) * delta_grad[x];
                        reference[0] += up + dn;
                        reference[1] += up - dn;
                        reference[2] += cross.real();
                        reference[3] += cross.imag();
                        double du = weight * std::real(std::conj(grad[x]) * delta_grad[x]);
                        double dd = weight * std::real(std::conj(delta_grad[x]) * std::complex<double>(0, 1) * grad[x]);
                        delta[0] += du + dd;
                        delta[1] += du - dd;
                        delta[2] += weight * std::norm(delta_grad[x]);
                        delta[3] += weight * std::norm(grad[x]);
                    }
                    std::array<double, 4> rotated{spin_tau[0][ir], spin_tau[1][ir], -spin_tau[3][ir], spin_tau[2][ir]};
                    for (int j = 0; j < 4; j++) {
                        spin_error          = std::max(spin_error, std::abs(spin_tau[j][ir] - reference[j]));
                        rotation_error      = std::max(rotation_error, std::abs(spin_potential[j][ir] - rotated[j]));
                        spin_response_error = std::max(spin_response_error, std::abs(spin_response[j][ir] - delta[j]));
                    }
                    spin_potential[0][ir] = potential[ir];
                    spin_potential[1][ir] = 0.2 * std::cos(twopi * r[1]);
                    spin_potential[2][ir] = 0.1 * std::sin(twopi * r[0]);
                    spin_potential[3][ir] = 0.3 + 0.1 * std::sin(twopi * r[2]);
                }
            }
        }
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    comm.allreduce<double, mpi::op_t::max>(&response_error, 1);
    double const tolerance = std::is_same_v<T, double> ? 2e-11 : 2e-6;
    int failures{0};
    Test_check check(failures, comm, gamma__ ? "Gamma" : "k-point");
    transform.set_execution_mode(SPFFT_EXEC_ASYNCHRONOUS);
    check.reject("asynchronous host-buffer use", [&] { wf::Kinetic_density_operator<T> invalid(transform, gvp); });
    check.reject("changed FFT execution mode", [&] { op.add_density(weight, phi.data(), tau.data()); });
    transform.set_execution_mode(SPFFT_EXEC_SYNCHRONOUS);
    check("analytic tau", error, tolerance);
    check("analytic tau response", response_error, tolerance);

    auto inner_pw = [&](std::vector<std::complex<T>> const& a, std::vector<std::complex<T>> const& b) {
        std::complex<double> value{0};
        for (int ig = 0; ig < gvp.count(); ig++) {
            auto z = std::conj(static_cast<std::complex<double>>(a[ig])) * static_cast<std::complex<double>>(b[ig]);
            auto q = gvp.gkvec_cart(ig);
            value += gamma__ ? std::complex<double>((q.length() == 0 ? 1 : 2) * z.real(), 0) : z;
        }
        comm.allreduce(&value, 1);
        return value;
    };

    if (!gamma__) {
        comm.allreduce<double, mpi::op_t::max>(&spin_error, 1);
        comm.allreduce<double, mpi::op_t::max>(&rotation_error, 1);
        comm.allreduce<double, mpi::op_t::max>(&spin_response_error, 1);
        check("analytic spinor tau", spin_error, tolerance);
        check("analytic spinor tau response", spin_response_error, tolerance);
        check("spinor tau rotation including transverse y sign", rotation_error, tolerance);
        std::array<std::vector<std::complex<T>>, 2> hspin, hswap;
        for (int s = 0; s < 2; s++) {
            hspin[s].resize(phi.size(), 0);
            hswap[s].resize(phi.size(), 0);
        }
        std::array<T const*, 4> v{spin_potential[0].data(), spin_potential[1].data(), spin_potential[2].data(),
                                  spin_potential[3].data()};
        op.add_spinor_potential(v, {phi.data(), direction.data()}, {hspin[0].data(), hspin[1].data()});
        op.add_spinor_potential(v, {direction.data(), phi.data()}, {hswap[0].data(), hswap[1].data()});
        check("spinor tau operator Hermiticity",
              std::abs(inner_pw(direction, hspin[0]) + inner_pw(phi, hspin[1]) -
                       std::conj(inner_pw(phi, hswap[0]) + inner_pw(direction, hswap[1]))),
              tolerance);
        double energy{0};
        for (int ir = 0; ir < nr; ir++) {
            for (int j = 0; j < 4; j++) {
                energy += dv * spin_potential[j][ir] * spin_tau[j][ir];
            }
        }
        comm.allreduce(&energy, 1);
        check("spinor tau density/operator normalization",
              std::abs(energy - occupation * (inner_pw(phi, hspin[0]) + inner_pw(direction, hspin[1])).real()),
              tolerance);
        check.reject("spinor in-place application", [&] {
            op.add_spinor_potential(v, {phi.data(), direction.data()}, {phi.data(), hspin[1].data()});
        });
    } else {
        check.reject("real FFT for a complex spinor", [&] {
            op.add_spinor_density(weight, {phi.data(), direction.data()},
                                  {spin_tau[0].data(), spin_tau[1].data(), spin_tau[2].data(), spin_tau[3].data()});
        });
    }

    check.reject("rank-local in-place application",
                 [&] { op.add_potential(potential.data(), phi.data(), comm.rank() == 0 ? phi.data() : hphi.data()); });
    op.add_potential(potential.data(), phi.data(), hphi.data());
    op.add_potential(potential.data(), direction.data(), hd.data());
    check("Hermiticity", std::abs(inner_pw(direction, hphi) - std::conj(inner_pw(phi, hd))), tolerance);
    double energy{0};
    for (int ir = 0; ir < nr; ir++) {
        energy += dv * potential[ir] * tau[ir];
    }
    comm.allreduce(&energy, 1);
    check("density/operator normalization", std::abs(energy - occupation * inner_pw(phi, hphi).real()), tolerance);

    std::fill(hphi.begin(), hphi.end(), std::complex<T>(0));
    std::fill(potential.begin(), potential.end(), T(1.7));
    op.add_potential(potential.data(), phi.data(), hphi.data());
    error = 0;
    for (int ig = 0; ig < gvp.count(); ig++) {
        auto q = gvp.gkvec_cart(ig);
        error  = std::max(error, double(std::abs(hphi[ig] - T(0.5 * 1.7 * dot(q, q)) * phi[ig])));
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    check("constant v_tau", error, tolerance);

    // E[tau] = integral tau^2/2 gives a nonuniform v_tau = tau.
    std::fill(hphi.begin(), hphi.end(), std::complex<T>(0));
    op.add_potential(tau.data(), phi.data(), hphi.data());
    auto trial_energy = [&](T step) {
        std::vector<std::complex<T>> c(phi.size());
        for (size_t i = 0; i < c.size(); i++) {
            c[i] = phi[i] + step * direction[i];
        }
        std::vector<T> t(nr, 0);
        op.add_density(weight, c.data(), t.data());
        double e{0};
        for (auto value : t) {
            e += 0.5 * dv * double(value) * value;
        }
        comm.allreduce(&e, 1);
        return e;
    };
    T step            = std::is_same_v<T, double> ? 1e-4 : 1e-2;
    double fd         = (trial_energy(step) - trial_energy(-step)) / (2 * step);
    double derivative = 2 * occupation * inner_pw(direction, hphi).real();
    check("nonlinear energy derivative", std::abs(fd - derivative), tolerance);

    std::fill(tau.begin(), tau.end(), T(0));
    op.add_density(weight, phi.data(), tau.data());
    op.add_density(2 * weight, phi.data(), tau.data());
    std::vector<T> tau_sum(nr, 0);
    op.add_density(3 * weight, phi.data(), tau_sum.data());
    error = 0;
    for (int ir = 0; ir < nr; ir++) {
        error = std::max(error, double(std::abs(tau[ir] - tau_sum[ir])));
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    check("occupation accumulation", error, tolerance);
    return failures;
}

int
test_kinetic_density_collection(bool gamma__, int num_mag_dims__, bool gpu__ = false)
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method": "pseudopotential",
        "use_symmetry": false, "pw_cutoff": 6.0, "gk_cutoff": 2.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"]
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    conf["parameters"]["gamma_point"]  = gamma__;
    conf["parameters"]["num_mag_dims"] = num_mag_dims__;
    conf["control"]["processing_unit"] = gpu__ ? "GPU" : "CPU";
    conf["mixer"]                      = {{"type", "linear"}, {"beta", 0.4}, {"use_hartree", false}};
    if (num_mag_dims__ == 3) {
        conf["parameters"]["extra_charge"] = -1.54;
    }
    auto ctx = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    K_point_set ks(*ctx);
    std::vector<r3::vector<double>> kpoints =
            gamma__ ? std::vector<r3::vector<double>>{{0, 0, 0}}
                    : std::vector<r3::vector<double>>{{0.23, -0.1, 0.05}, {-0.15, 0.2, 0.3}};
    std::vector<double> weights = gamma__ ? std::vector<double>{1.0} : std::vector<double>{0.3, 0.7};
    for (size_t ik = 0; ik < kpoints.size(); ik++) {
        ks.add_kpoint(kpoints[ik], weights[ik]);
    }
    ks.initialize();
    auto occupancy = [](int band, int spin, int ik) { return (0.6 + 0.2 * band + 0.1 * ik) / (spin + 1); };
    bool spinor    = num_mag_dims__ == 3;
    std::vector<wf::device_memory_guard> guards;
    for (auto it : ks.spl_num_kpoints()) {
        auto kp   = ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        for (int ispn = 0; ispn < ctx->num_spins(); ispn++) {
            for (int ib = 0; ib < ctx->num_bands(); ib++) {
                if (!spinor || ispn == 0) {
                    kp->band_occupancy(ib, ispn, ib < 2 ? occupancy(ib, ispn, it.i) : 0);
                }
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    auto g        = kp->gkvec().gvec(gvec_index_t::global(kp->gkvec().offset() + ig));
                    int axis      = spinor ? (ib + ispn) % 2 : ib;
                    bool selected = ib < 2 && g[(axis + 1) % 2] == 0 && g[2] == 0 &&
                                    (gamma__ ? std::abs(g[axis]) == 1 : g[axis] == 1);
                    std::complex<double> amplitude = gamma__ ? std::sqrt(0.5) : 1.0;
                    if (spinor) {
                        amplitude = ispn == 0 ? std::complex<double>(std::sqrt(0.6), 0)
                                              : std::complex<double>(0, std::sqrt(0.4));
                    }
                    psi.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(ib)) = selected ? amplitude : 0.0;
                }
            }
        }
        if (gpu__) {
            guards.emplace_back(psi.memory_guard(memory_t::device, wf::copy_to::device));
        }
    }
    ks.sync_band<double, sync_band_t::occupancy>();
    Density density(*ctx, spinor);
    auto tau = density.generate_smooth_kinetic_density<double>(ks);
    guards.clear();
    auto const& fft = ctx->spfft<double>();
    double error{0};
    if (spinor) {
        density.generate<double>(ks, false, false, true);
    }
    for (int ispn = 0; ispn < (spinor ? 4 : ctx->num_spins()); ispn++) {
        for (int iz = 0; iz < fft.local_z_length(); iz++) {
            for (int iy = 0; iy < fft.dim_y(); iy++) {
                for (int ix = 0; ix < fft.dim_x(); ix++) {
                    int ir = ix + fft.dim_x() * (iy + fft.dim_y() * iz);
                    r3::vector<double> r(double(ix) / fft.dim_x(), double(iy) / fft.dim_y(),
                                         double(iz + fft.local_z_offset()) / fft.dim_z());
                    double reference{0};
                    for (size_t ik = 0; ik < kpoints.size(); ik++) {
                        for (int ib = 0; ib < 2; ib++) {
                            if (spinor) {
                                r3::vector<double> gu(0, 0, 0), gd(0, 0, 0);
                                gu[ib] = gd[1 - ib] = 1;
                                auto reciprocal     = ctx->unit_cell().reciprocal_lattice_vectors();
                                auto qu = dot(reciprocal, gu + kpoints[ik]), qd = dot(reciprocal, gd + kpoints[ik]);
                                double weight = weights[ik] * occupancy(ib, 0, ik) / ctx->unit_cell().omega();
                                double up = 0.3 * weight * dot(qu, qu), dn = 0.2 * weight * dot(qd, qd);
                                auto cross = std::complex<double>(0, std::sqrt(0.24)) * weight * dot(qu, qd) *
                                             std::exp(std::complex<double>(0, twopi * dot(gd - gu, r)));
                                std::array<double, 4> fields{up + dn, up - dn, cross.real(), cross.imag()};
                                reference += fields[ispn];
                                continue;
                            }
                            r3::vector<double> g(0, 0, 0);
                            g[ib]            = 1;
                            auto q           = dot(ctx->unit_cell().reciprocal_lattice_vectors(), g + kpoints[ik]);
                            double amplitude = gamma__ ? 2 * std::pow(std::sin(twopi * r[ib]), 2) : 1;
                            reference += 0.5 * weights[ik] * occupancy(ib, ispn, ik) * dot(q, q) * amplitude /
                                         ctx->unit_cell().omega();
                        }
                    }
                    if (!std::isfinite(tau[ispn].value(ir))) {
                        error = 1;
                    }
                    error = std::max(error, std::abs(tau[ispn].value(ir) - reference));
                    if (spinor) {
                        error = std::max(error, std::abs(density.kinetic_density().component(ispn).rg().value(ir) -
                                                         tau[ispn].value(ir)));
                    }
                }
            }
        }
    }
    if (spinor) {
        Potential potential(*ctx, true);
        potential.zero();
        auto& vtau = potential.kinetic_potential();
        for (int iz = 0; iz < fft.local_z_length(); iz++) {
            double angle = twopi * (iz + fft.local_z_offset()) / fft.dim_z();
            for (int ir = iz * fft.dim_x() * fft.dim_y(); ir < (iz + 1) * fft.dim_x() * fft.dim_y(); ir++) {
                std::array<double, 4> v{0.7, 0.2, 0.15 * std::cos(angle), -0.11 * std::sin(angle)};
                for (int j = 0; j < 4; j++) {
                    vtau.component(j).rg().value(ir) = v[j];
                }
            }
        }
        vtau.fft_transform(-1);
        Local_operator<double> local(*ctx, ctx->spfft_coarse<double>(), ctx->gvec_coarse_fft_sptr(), potential);
        error = std::max({error, std::abs(local.vtau0(0) - 0.9), std::abs(local.vtau0(1) - 0.5)});
        double expectation{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            local.prepare_k(*kp->gkvec_fft_sptr());
            wf::Wave_functions<double> hpsi(kp->gkvec_sptr(), wf::num_mag_dims(3), wf::num_bands(2), memory_t::host);
            {
                auto psi_guard =
                        kp->spinor_wave_functions().memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::device);
                auto guard = hpsi.memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::host);
                local.apply_h(kp->spfft_transform(), kp->gkvec_fft_sptr(), wf::spin_range(0, 2),
                              kp->spinor_wave_functions(), hpsi, wf::band_range(0, 2));
            }
            for (int ib = 0; ib < 2; ib++) {
                for (int s = 0; s < 2; s++) {
                    r3::vector<int> own(0, 0, 0), other(0, 0, 0);
                    own[(ib + s) % 2] = other[(ib + 1 - s) % 2] = 1;
                    auto amplitude =
                            s == 0 ? std::complex<double>(std::sqrt(0.6), 0) : std::complex<double>(0, std::sqrt(0.4));
                    auto opposite =
                            s == 1 ? std::complex<double>(std::sqrt(0.6), 0) : std::complex<double>(0, std::sqrt(0.4));
                    auto reciprocal = ctx->unit_cell().reciprocal_lattice_vectors();
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                        auto q = kp->gkvec().gkvec_cart(gvec_index_t::local(ig));
                        std::complex<double> reference{0}, free{0};
                        if (g == own) {
                            free      = 0.5 * dot(q, q) * amplitude;
                            reference = (s == 0 ? 1.9 : 1.5) * free;
                        }
                        if (g[0] == other[0] && g[1] == other[1] && std::abs(g[2]) == 1) {
                            auto input_q = dot(reciprocal, other + kp->vk());
                            double v     = (s == 0) == (g[2] == 1) ? 0.13 : 0.02;
                            reference += 0.5 * dot(q, input_q) * v * opposite;
                        }
                        auto value = hpsi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(ib));
                        error      = std::max(error, std::abs(value - reference));
                        auto psi   = kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(ib));
                        expectation +=
                                kp->weight() * kp->band_occupancy(ib, 0) * std::real(std::conj(psi) * (value - free));
                    }
                }
            }
        }
        ctx->comm().allreduce(&expectation, 1);
        error = std::max(error, std::abs(expectation - potential.energy_vtau(density)));
        Density replica(*ctx, true);
        copy(density, replica);
        density.mixer_init(ctx->cfg().mixer());
        for (int j = 0; j < 4; j++) {
            for (int ir = 0; ir < fft.local_slice_size(); ir++) {
                error = std::max(error,
                                 std::abs(replica.kinetic_density().component(j).rg().value(ir) - tau[j].value(ir)));
                density.kinetic_density().component(j).rg().value(ir) += (j + 1) * 1e-4;
            }
        }
        density.mix();
        for (int j = 0; j < 4; j++) {
            for (int ir = 0; ir < fft.local_slice_size(); ir++) {
                error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) -
                                                 tau[j].value(ir) - 0.4 * (j + 1) * 1e-4));
            }
        }
        unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        ctx->comm().bcast(&stamp, 1, 0);
        auto path = std::filesystem::temp_directory_path() / ("sirius-spinor-tau-" + std::to_string(stamp) + ".h5");
        ctx->create_storage_file(path.string());
        density.save(path.string());
        ctx->comm().barrier();
        replica.kinetic_density().zero();
        replica.load(path.string());
        for (int j = 0; j < 4; j++) {
            for (int ir = 0; ir < fft.local_slice_size(); ir++) {
                error = std::max(error, std::abs(replica.kinetic_density().component(j).rg().value(ir) -
                                                 density.kinetic_density().component(j).rg().value(ir)));
            }
        }
        ctx->comm().barrier();
        if (ctx->comm().rank() == 0) {
            std::filesystem::remove(path);
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    if (ctx->comm().rank() == 0) {
        std::cout << "smooth density collection, gamma=" << gamma__ << ", magnetic dimensions=" << num_mag_dims__
                  << ": " << error << '\n';
    }
    return error > 1e-12;
}

int
test_density_copy()
{
    auto conf  = R"({"parameters": {
        "electronic_structure_method": "pseudopotential", "use_symmetry": false,
        "pw_cutoff": 6.0, "gk_cutoff": 2.0, "num_bands": 4,
        "num_mag_dims": 1, "xc_functionals": ["XC_LDA_X"]
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    auto ctx   = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    auto other = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    int failures{0};
    for (bool tau : {false, true}) {
        Density source(*ctx, tau), same(*ctx, tau), different(*other, tau), mismatch(*ctx, !tau);
        for (int j = 0; j < 2; j++) {
            source.component(j).rg().values()     = [j]() { return double(j + 1); };
            source.component(j).rg().f_pw_local() = [j]() { return std::complex<double>(j + 2, j + 3); };
            if (tau) {
                source.kinetic_density().component(j).rg().values()     = [j]() { return double(j + 4); };
                source.kinetic_density().component(j).rg().f_pw_local() = [j]() {
                    return std::complex<double>(j + 5, j + 6);
                };
            }
        }
        for (auto* dest : {&same, &different, &mismatch}) {
            bool rejected{false};
            try {
                copy(source, *dest);
            } catch (std::exception const&) {
                rejected = true;
            }
            bool expected = dest == &mismatch || (tau && dest == &different);
            failures += rejected != expected;
            if (!rejected && !expected) {
                for (int j = 0; j < 2; j++) {
                    for (size_t i = 0; i < source.component(j).rg().values().size(); i++) {
                        failures += dest->component(j).rg().value(i) != source.component(j).rg().value(i);
                        if (tau) {
                            failures += dest->kinetic_density().component(j).rg().value(i) !=
                                        source.kinetic_density().component(j).rg().value(i);
                        }
                    }
                    for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                        failures += dest->component(j).rg().f_pw_local(ig) != source.component(j).rg().f_pw_local(ig);
                        if (tau) {
                            failures += dest->kinetic_density().component(j).rg().f_pw_local(ig) !=
                                        source.kinetic_density().component(j).rg().f_pw_local(ig);
                        }
                    }
                }
            }
        }
    }
    return failures;
}

int
test_spinor_initial_kinetic_density(bool gpu__)
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method": "pseudopotential", "num_mag_dims": 3,
        "use_symmetry": false, "pw_cutoff": 6.0, "gk_cutoff": 2.0,
        "num_bands": 4, "xc_functionals": []
    }, "control": {"mpi_grid_dims": [1, 1], "verbosity": 0}})"_json;
    conf["control"]["processing_unit"] = gpu__ ? "GPU" : "CPU";
    auto ctx    = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    auto points = std::make_shared<xc_point_set_t>(*ctx);
    if (ctx->comm().rank() == 0) {
        points->coordinates        = {{1, 2, 3}};
        points->energy_weights     = {1};
        points->descriptor_weights = {1};
        points->block_sizes        = {1};
        points->block_centres      = {{1, 2, 3}};
    }
    ctx->cb().external_xc_points_ = points;
    ctx->cb().external_xc_spinor_ = true;
    ctx->cb().external_xc_        = [](auto const&, auto const&, auto&) {
        RTE_THROW("initial-guess test must not evaluate XC");
    };
    Density density(*ctx, true);
    auto const& fft = ctx->spfft<double>();
    double error{0}, q = twopi / 8, tf = 0.3 * std::pow(6 * pi * pi, 2.0 / 3.0);
    for (int mode = 0; mode < 5; mode++) {
        std::array<std::vector<double>, 4> reference;
        for (auto& component : reference) {
            component.resize(fft.local_slice_size());
        }
        for (int iz = 0; iz < fft.local_z_length(); iz++) {
            for (int iy = 0; iy < fft.dim_y(); iy++) {
                for (int ix = 0; ix < fft.dim_x(); ix++) {
                    int ir              = ix + fft.dim_x() * (iy + fft.dim_y() * iz);
                    double angle        = twopi * ix / fft.dim_x();
                    double n            = mode >= 3 ? 0.8 : 0.8 + 0.2 * std::sin(angle);
                    double dn           = mode >= 3 ? 0 : 0.2 * q * std::cos(angle);
                    double polarization = mode == 2 ? 0 : (mode == 4 ? 0.2 * std::sin(angle) : 0.2);
                    std::array<double, 3> axis{1, 0, 0};
                    if (mode == 1) {
                        axis = {1 / std::sqrt(14.0), 2 / std::sqrt(14.0), 3 / std::sqrt(14.0)};
                    } else if (mode == 3) {
                        axis = {0, std::cos(angle), std::sin(angle)};
                    }
                    double up = 0.5 * (1 + polarization) * n, down = 0.5 * (1 - polarization) * n;
                    double tu = tf * std::pow(up, 5.0 / 3.0) + 0.5 * (1 + polarization) * dn * dn / (8 * n);
                    double td = tf * std::pow(down, 5.0 / 3.0) + 0.5 * (1 - polarization) * dn * dn / (8 * n);
                    if (mode == 4) {
                        // At a zero of m, its gradient still contributes to both spin channels.
                        double dm = 0.2 * n * q * std::cos(angle);
                        tu += dm * dm / (32 * up);
                        td += dm * dm / (32 * down);
                    }
                    density.rho().rg().value(ir) = n;
                    reference[0][ir]             = tu + td;
                    if (mode == 3) {
                        reference[0][ir] += std::pow(std::sqrt(up) - std::sqrt(down), 2) * q * q / 4;
                    }
                    for (int j = 0; j < 3; j++) {
                        density.mag(j).rg().value(ir) = polarization * n * axis[j];
                        reference[j + 1][ir]          = (tu - td) * axis[j];
                    }
                }
            }
        }
        density.fft_transform(-1);
        density.initial_kinetic_density();
        for (int j = 0; j < 4; j++) {
            for (int ir = 0; ir < fft.local_slice_size(); ir++) {
                error = std::max(error,
                                 std::abs(density.kinetic_density().component(j).rg().value(ir) - reference[j][ir]));
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    int failures{0};
    Test_check check(failures, ctx->comm(), "spinor initial kinetic density");
    check("collinear, unpolarized, rotated and transverse-gradient starting guesses", error, 2e-11);
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(true);
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
    if (argc == 2 && std::string(argv[1]) == "gpu") {
        int result =
                call_test("GPU kinetic density, complex k-point", test_kinetic_density<double>, false, SPFFT_PU_GPU);
        result += call_test("GPU kinetic density, real Gamma", test_kinetic_density<double>, true, SPFFT_PU_GPU);
        result += call_test("GPU spinor kinetic density collection", test_kinetic_density_collection, false, 3, true);
        result += call_test("GPU spinor initial kinetic density", test_spinor_initial_kinetic_density, true);
        sirius::finalize();
        return std::min(result, 1);
    }
#endif
    int result = call_test("density copy with optional kinetic fields", test_density_copy);
    result += call_test("spinor initial kinetic density", test_spinor_initial_kinetic_density, false);
    result += call_test("kinetic density, complex k-point", test_kinetic_density<double>, false, SPFFT_PU_HOST);
    result += call_test("kinetic density, real Gamma", test_kinetic_density<double>, true, SPFFT_PU_HOST);
#ifdef SIRIUS_USE_FP32
    result += call_test("kinetic density FP32, complex k-point", test_kinetic_density<float>, false, SPFFT_PU_HOST);
    result += call_test("kinetic density FP32, real Gamma", test_kinetic_density<float>, true, SPFFT_PU_HOST);
#endif
    for (auto gamma : {false, true}) {
        for (int num_mag_dims : {0, 1}) {
            result += call_test("smooth kinetic density collection", test_kinetic_density_collection, gamma,
                                num_mag_dims, false);
        }
    }
    result += call_test("spinor kinetic density collection", test_kinetic_density_collection, false, 3, false);
    sirius::finalize();
    return std::min(result, 1);
}
