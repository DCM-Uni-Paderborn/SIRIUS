/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "xc_derivative_checks.hpp"
#include "api/error_codes.hpp"
#include "core/any_ptr.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include <iostream>

namespace sirius::testing {

std::array<std::array<double, 3>, 2>
check_external_xc_derivative(DFT_ground_state& gs__, double step__)
{
    auto& kset_ = gs__.k_point_set();
    auto& ctx_ = kset_.ctx();
    int invalid = !ctx_.cb().external_xc_ || ctx_.cfg().parameters().precision_wf() != "fp64" ||
                  ctx_.hubbard_correction() || !std::isfinite(step__) || step__ <= 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("XC derivative check requires an external evaluator, fp64 orbitals, no Hubbard term and a "
                  "positive step");
    }
    validate_external_xc_context(ctx_);
    double lo = step__, hi = step__;
    ctx_.comm().allreduce<double, mpi::op_t::min>(&lo, 1);
    ctx_.comm().allreduce<double, mpi::op_t::max>(&hi, 1);
    if (lo != hi) {
        RTE_THROW("XC derivative check requires the same step on every rank");
    }
    if (ctx_.full_potential()) {
        return check_lapw_xc_derivative(gs__, step__);
    }

    bool spinor = ctx_.num_mag_dims() == 3;
    std::vector<std::vector<std::complex<double>>> saved;
    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<double>(it.i);
        saved.emplace_back();
        for (int s = 0; s < ctx_.num_spins(); s++) {
            for (int b = 0; b < kp->num_occupied_bands(spinor ? 0 : s); b++) {
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    saved.back().push_back(
                            kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)));
                }
            }
        }
    }
    // Odd real phases preserve conjugate symmetry for reduced-G Gamma states.
    // Nonlinear G dependence changes the shape, not just the position, of an orbital.
    auto phase = [](r3::vector<int> const& g, int direction) {
        return direction == 0 ? std::sin(0.41 * g[0] + 0.73 * g[1] + 0.19 * g[2])
                              : std::sin(0.67 * g[0] - 0.23 * g[1] + 0.53 * g[2]);
    };
    auto set_orbitals = [&](int direction, double t) {
        size_t ik{0};
        for (auto it : kset_.spl_num_kpoints()) {
            auto kp = kset_.get<double>(it.i);
            size_t index{0};
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int b = 0; b < kp->num_occupied_bands(spinor ? 0 : s); b++) {
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                        auto c = saved[ik][index++];
                        kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)) =
                                t == 0 ? c : c * std::polar(1.0, t * phase(g, direction));
                    }
                }
            }
            ik++;
        }
    };
    auto evaluator = ctx_.cb().external_xc_;
    auto restore   = [&]() {
        set_orbitals(0, 0);
        ctx_.cb().external_xc_ = evaluator;
    };
    std::array<std::array<double, 3>, 2> result{};
    try {
        Density density(ctx_);
        Potential potential(ctx_);
        for (int direction = 0; direction < 2; direction++) {
            constexpr double offset{0.17};
            auto evaluate = [&](double t) {
                set_orbitals(direction, offset + t);
                density.generate<double>(kset_, false, true, true);
                potential.generate(density, false, true);
                return potential.energy_exc(density);
            };
            evaluate(0);
            auto expectation = [&]() {
                Hamiltonian0<double> H0(potential, false);
                double derivative{0};
                for (auto it : kset_.spl_num_kpoints()) {
                    auto kp = kset_.get<double>(it.i);
                    auto Hk = H0(*kp);
                    wf::Wave_functions<double> hphi(kp->gkvec_sptr(), wf::num_mag_dims(ctx_.num_mag_dims()),
                                                    wf::num_bands(ctx_.num_bands()), memory_t::host);
                    for (int s = 0; s < ctx_.num_spins(); s++) {
                        int nb = kp->num_occupied_bands(spinor ? 0 : s);
                        if (nb == 0) {
                            continue;
                        }
                        if (!spinor || s == 0) {
                            auto psi_guard  = kp->spinor_wave_functions().memory_guard(ctx_.processing_unit_memory_t(),
                                                                                       wf::copy_to::device);
                            auto hphi_guard = hphi.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::host);
                            if (ctx_.gamma_point()) {
                                Hk.apply_h_s<double>(wf::spin_range(s), wf::band_range(0, nb),
                                                     kp->spinor_wave_functions(), &hphi, nullptr);
                            } else {
                                auto spins = spinor ? wf::spin_range(0, 2) : wf::spin_range(s);
                                Hk.apply_h_s<std::complex<double>>(spins, wf::band_range(0, nb),
                                                                   kp->spinor_wave_functions(), &hphi, nullptr);
                            }
                        }
                        for (int b = 0; b < nb; b++) {
                            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                                auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                                auto c =
                                        kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(b));
                                auto d              = std::complex<double>(0, phase(g, direction)) * c;
                                auto h              = hphi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b));
                                double multiplicity = kp->gkvec().reduced() && dot(g, g) ? 2 : 1;
                                derivative += 2 * kp->weight() * kp->band_occupancy(b, spinor ? 0 : s) * multiplicity *
                                              std::real(std::conj(d) * h);
                            }
                        }
                    }
                }
                ctx_.comm().allreduce(&derivative, 1);
                return derivative;
            };
            result[direction][0] = expectation();
            // Subtract the same-density Hamiltonian with zero XC. This retains
            // precisely the scalar, smooth-Q, local PAW and tau XC operators.
            ctx_.cb().external_xc_ = [](auto const&, auto const&, auto&) {};
            potential.generate(density, false, true);
            result[direction][0] -= expectation();
            ctx_.cb().external_xc_ = evaluator;
            for (int j = 1; j <= 2; j++) {
                double h             = step__ / j;
                double plus          = evaluate(h);
                double minus         = evaluate(-h);
                result[direction][j] = (plus - minus) / (2 * h);
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    return result;
}

} // namespace sirius::testing

namespace {

template <typename F>
int
check_ground_state(void* handle__, double* values__, F&& check__) noexcept
{
    try {
        if (!handle__ || !values__) {
            RTE_THROW("Missing ground-state handle or result buffer in XC check");
        }
        auto& gs = static_cast<sirius::any_ptr*>(handle__)->get<sirius::DFT_ground_state>();
        check__(gs);
        return SIRIUS_SUCCESS;
    } catch (std::runtime_error const& e) {
        std::cerr << e.what() << '\n';
        return SIRIUS_ERROR_RUNTIME;
    } catch (std::exception const& e) {
        std::cerr << e.what() << '\n';
        return SIRIUS_ERROR_EXCEPTION;
    } catch (...) {
        return SIRIUS_ERROR_UNKNOWN;
    }
}

} // namespace

extern "C" int
sirius_test_external_xc_derivative(void* handle__, double step__, double* derivatives__) noexcept
{
    return check_ground_state(handle__, derivatives__, [&](auto& gs) {
        auto result = sirius::testing::check_external_xc_derivative(gs, step__);
        for (int d = 0; d < 2; d++) {
            std::copy(result[d].begin(), result[d].end(), derivatives__ + 3 * d);
        }
    });
}

extern "C" int
sirius_test_external_xc_response(void* handle__, double step__, double* errors__) noexcept
{
    return check_ground_state(handle__, errors__, [&](auto& gs) {
        auto result = sirius::testing::check_lapw_xc_response(gs, step__);
        std::copy(result.begin(), result.end(), errors__);
    });
}
