/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SMOOTH_PERIODIC_FUNCTION_SAMPLER_HPP__
#define __SMOOTH_PERIODIC_FUNCTION_SAMPLER_HPP__

#include "function3d/smooth_periodic_function.hpp"

namespace sirius {

struct smooth_sampling_geometry_adjoint_t
{
    std::vector<r3::vector<double>> points;
    r3::matrix<double> lattice;
};

/// Fourier sampling of a real field and its Cartesian gradient at arbitrary points.
/** Points are Cartesian bohr and are owned exactly once across gvec.comm(),
 *  independently of the FFT layout. In particular, replicated FFT slabs are
 *  not independent point sets. All operations are collective, including empty
 *  ranks. G-vector ordering and cell must remain fixed for this object's lifetime.
 *  The Fourier sum costs O(N_points * N_G), without a dense sampling matrix.
 *  Integer G components allow phase factors to be reused within bounded tiles.
 */
class Smooth_periodic_function_sampler
{
  private:
    fft::Gvec const& gvec_;
    std::vector<r3::vector<double>> fractional_points_;
    std::vector<r3::vector<double>> unwrapped_fractional_points_;
    std::vector<r3::vector<int>> phase_indices_;
    std::vector<r3::vector<double>> cartesian_g_;
    std::array<int, 3> bounds_{}, offsets_{};
    int num_phases_{0};
    static constexpr size_t point_tile{256};
    static constexpr int gvec_tile{128};

    mdarray<std::complex<double>, 2>
    phases(size_t first__, int count__) const
    {
        mdarray<std::complex<double>, 2> result({static_cast<size_t>(count__), static_cast<size_t>(num_phases_)});
        for (int x = 0; x < 3; x++) {
            int zero = offsets_[x] + bounds_[x];
            for (int p = 0; p < count__; p++) {
                auto step = std::exp(std::complex<double>(0, twopi * fractional_points_[first__ + p][x]));
                std::complex<double> phase{1};
                result(p, zero) = phase;
                for (int n = 1; n <= bounds_[x]; n++) {
                    phase *= step;
                    result(p, zero + n) = phase;
                    result(p, zero - n) = std::conj(phase);
                }
            }
        }
        return result;
    }

    void
    check(bool failed__, char const* message__) const
    {
        int failed = failed__;
        gvec_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW(message__);
        }
    }

    void
    check_geometry_direction(std::vector<r3::vector<double>> const& points__, r3::matrix<double> const& lattice__) const
    {
        bool failed = points__.size() != fractional_points_.size();
        for (auto const& p : points__) {
            for (int x = 0; x < 3; x++) {
                failed |= !std::isfinite(p[x]);
            }
        }
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                failed |= !std::isfinite(lattice__(x, y));
            }
        }
        check(failed, "Fourier geometry response: incompatible or nonfinite direction");
        auto lo = lattice__, hi = lattice__;
        gvec_.comm().allreduce<double, mpi::op_t::min>(&lo(0, 0), 9);
        gvec_.comm().allreduce<double, mpi::op_t::max>(&hi(0, 0), 9);
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                failed |= lo(x, y) != hi(x, y);
            }
        }
        check(failed, "Fourier geometry response: lattice direction differs between ranks");
    }

  public:
    Smooth_periodic_function_sampler(fft::Gvec const& gvec__, std::vector<r3::vector<double>> const& points__)
        : gvec_(gvec__)
    {
        bool failed  = gvec_.vk().length() != 0 || !std::isfinite(gvec_.omega()) || gvec_.omega() <= 0;
        auto inverse = r3::inverse(gvec_.unit_cell_lattice_vectors());
        for (auto const& p : points__) {
            auto f = r3::dot(inverse, p);
            unwrapped_fractional_points_.push_back(f);
            for (int x = 0; x < 3; x++) {
                failed |= !std::isfinite(f[x]);
                f[x] -= std::floor(f[x]);
            }
            fractional_points_.push_back(f);
        }
        check(failed, "Fourier sampling requires finite points, a nonsingular cell and unshifted G vectors");
        for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
            auto g = gvec_.gvec(gvec_index_t::global(ig));
            phase_indices_.push_back(g);
            cartesian_g_.push_back(gvec_.gvec_cart(gvec_index_t::global(ig)));
            for (int x = 0; x < 3; x++) {
                bounds_[x] = std::max(bounds_[x], std::abs(g[x]));
            }
        }
        for (int x = 0; x < 3; x++) {
            offsets_[x] = num_phases_;
            num_phases_ += 2 * bounds_[x] + 1;
        }
        for (auto& g : phase_indices_) {
            for (int x = 0; x < 3; x++) {
                g[x] += offsets_[x] + bounds_[x];
            }
        }
    }

    /// Read current PW coefficients; no hidden transform of the real-space buffer.
    std::vector<std::array<double, 4>>
    sample(Smooth_periodic_function<double> const& field__) const
    {
        PROFILE("sirius::Smooth_periodic_function_sampler::sample");
        bool failed = &field__.gvec() != &gvec_;
        for (auto c : field__.f_pw_local()) {
            failed |= !std::isfinite(c.real()) || !std::isfinite(c.imag());
        }
        check(failed, "Fourier sampling: incompatible G vectors or nonfinite coefficients");
        std::vector<std::complex<double>> coefficients(gvec_.num_gvec());
        gvec_.comm().allgather(field__.f_pw_local().at(memory_t::host), coefficients.data(), gvec_.count(),
                               gvec_.offset());
        std::vector<std::array<double, 4>> result(fractional_points_.size());
        #pragma omp parallel for schedule(static)
        for (size_t first = 0; first < fractional_points_.size(); first += point_tile) {
            int count  = static_cast<int>(std::min(point_tile, fractional_points_.size() - first));
            auto phase = phases(first, count);
            for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                auto const& g       = phase_indices_[ig];
                auto const& cart    = cartesian_g_[ig];
                double multiplicity = gvec_.reduced() && r3::dot(cart, cart) != 0 ? 2 : 1;
                for (int p = 0; p < count; p++) {
                    auto z = coefficients[ig] * phase(p, g[0]) * phase(p, g[1]) * phase(p, g[2]);
                    result[first + p][0] += multiplicity * z.real();
                    for (int x = 0; x < 3; x++) {
                        result[first + p][x + 1] -= multiplicity * cart[x] * z.imag();
                    }
                }
            }
        }
        return result;
    }

    /// Geometry direction at fixed Fourier coefficients, with Cartesian point directions.
    std::vector<std::array<double, 4>>
    geometry_response(Smooth_periodic_function<double> const& field__,
                      std::vector<r3::vector<double>> const& point_directions__,
                      r3::matrix<double> const& lattice_direction__) const
    {
        check_geometry_direction(point_directions__, lattice_direction__);
        bool failed = &field__.gvec() != &gvec_;
        for (auto c : field__.f_pw_local()) {
            failed |= !std::isfinite(c.real()) || !std::isfinite(c.imag());
        }
        check(failed, "Fourier geometry response: incompatible field or nonfinite coefficients");
        std::vector<std::complex<double>> coefficients(gvec_.num_gvec());
        gvec_.comm().allgather(field__.f_pw_local().at(memory_t::host), coefficients.data(), gvec_.count(),
                               gvec_.offset());
        auto inverse_transpose = r3::transpose(r3::inverse(gvec_.unit_cell_lattice_vectors()));
        std::vector<std::array<double, 4>> result(fractional_points_.size());
        #pragma omp parallel for schedule(static)
        for (size_t first = 0; first < fractional_points_.size(); first += point_tile) {
            int count  = static_cast<int>(std::min(point_tile, fractional_points_.size() - first));
            auto phase = phases(first, count);
            for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                auto const& g    = phase_indices_[ig];
                auto const& cart = cartesian_g_[ig];
                auto dg          = -1.0 * r3::dot(inverse_transpose, r3::dot(r3::transpose(lattice_direction__), cart));
                double multiplicity = gvec_.reduced() && r3::dot(cart, cart) != 0 ? 2 : 1;
                for (int p = 0; p < count; p++) {
                    auto z            = coefficients[ig] * phase(p, g[0]) * phase(p, g[1]) * phase(p, g[2]);
                    auto displacement = point_directions__[first + p] -
                                        r3::dot(lattice_direction__, unwrapped_fractional_points_[first + p]);
                    double dphase     = r3::dot(cart, displacement);
                    result[first + p][0] -= multiplicity * z.imag() * dphase;
                    for (int x = 0; x < 3; x++) {
                        result[first + p][x + 1] -= multiplicity * (dg[x] * z.imag() + cart[x] * z.real() * dphase);
                    }
                }
            }
        }
        return result;
    }

    /// Geometry derivatives of value/gradient sampling at fixed Fourier coefficients.
    /** Raw covectors include all model and quadrature weights. Point derivatives
     *  are local Cartesian dE/dr. The globally summed lattice derivative is at
     *  fixed Cartesian points, with lattice vectors in columns. No volume or
     *  strain normalization is applied. Changes in coefficients, quadrature and
     *  atom-local fields must be differentiated separately by the caller.
     */
    smooth_sampling_geometry_adjoint_t
    geometry_adjoint(Smooth_periodic_function<double> const& field__,
                     std::vector<std::array<double, 4>> const& covectors__) const
    {
        bool failed = &field__.gvec() != &gvec_ || covectors__.size() != fractional_points_.size();
        for (auto c : field__.f_pw_local()) {
            failed |= !std::isfinite(c.real()) || !std::isfinite(c.imag());
        }
        for (auto const& d : covectors__) {
            for (double v : d) {
                failed |= !std::isfinite(v);
            }
        }
        check(failed, "Fourier geometry adjoint: incompatible buffers or nonfinite inputs");
        std::vector<std::complex<double>> coefficients(gvec_.num_gvec());
        gvec_.comm().allgather(field__.f_pw_local().at(memory_t::host), coefficients.data(), gvec_.count(),
                               gvec_.offset());
        auto inverse = r3::inverse(gvec_.unit_cell_lattice_vectors());
        smooth_sampling_geometry_adjoint_t result;
        result.points.resize(fractional_points_.size());
        std::vector<r3::matrix<double>> lattice((fractional_points_.size() + point_tile - 1) / point_tile);
        #pragma omp parallel for schedule(static)
        for (size_t first = 0; first < fractional_points_.size(); first += point_tile) {
            int count  = static_cast<int>(std::min(point_tile, fractional_points_.size() - first));
            auto phase = phases(first, count);
            std::vector<r3::vector<double>> gradient_covectors(count);
            for (int p = 0; p < count; p++) {
                auto const& d         = covectors__[first + p];
                gradient_covectors[p] = r3::dot(inverse, r3::vector<double>({d[1], d[2], d[3]}));
            }
            auto& cell = lattice[first / point_tile];
            for (int ig = 0; ig < gvec_.num_gvec(); ig++) {
                auto const& g       = phase_indices_[ig];
                auto const& cart    = cartesian_g_[ig];
                double multiplicity = gvec_.reduced() && r3::dot(cart, cart) != 0 ? 2 : 1;
                for (int p = 0; p < count; p++) {
                    auto z        = coefficients[ig] * phase(p, g[0]) * phase(p, g[1]) * phase(p, g[2]);
                    auto const& d = covectors__[first + p];
                    double a      = d[0] * z.imag() + (cart[0] * d[1] + cart[1] * d[2] + cart[2] * d[3]) * z.real();
                    for (int x = 0; x < 3; x++) {
                        result.points[first + p][x] -= multiplicity * cart[x] * a;
                        for (int y = 0; y < 3; y++) {
                            cell(x, y) += multiplicity * cart[x] *
                                          (unwrapped_fractional_points_[first + p][y] * a +
                                           gradient_covectors[p][y] * z.imag());
                        }
                    }
                }
            }
        }
        for (auto const& cell : lattice) {
            result.lattice += cell;
        }
        gvec_.comm().allreduce(&result.lattice(0, 0), 9);
        return result;
    }

    /// Map raw value/gradient covectors to volume-normalized potential coefficients.
    /** The point covectors already include all energy/descriptor weights. The
     *  output satisfies dE = Omega Re sum_G conj(v_G) df_G, with multiplicity
     *  two for nonzero reduced G vectors. No additional point weight belongs
     *  here. This normalization lets existing smooth Hamiltonian and Q-matrix
     *  contractions consume the result. Only the PW buffer is overwritten.
     */
    void
    adjoint(std::vector<std::array<double, 4>> const& covectors__, Smooth_periodic_function<double>& potential__) const
    {
        PROFILE("sirius::Smooth_periodic_function_sampler::adjoint");
        bool failed = covectors__.size() != fractional_points_.size() || &potential__.gvec() != &gvec_;
        for (auto const& d : covectors__) {
            for (double v : d) {
                failed |= !std::isfinite(v);
            }
        }
        check(failed, "Fourier adjoint: incompatible buffers or nonfinite covectors");
        std::vector<std::complex<double>> coefficients(gvec_.num_gvec());
        #pragma omp parallel for schedule(static)
        for (int begin = 0; begin < gvec_.num_gvec(); begin += gvec_tile) {
            int end = std::min(begin + gvec_tile, gvec_.num_gvec());
            for (size_t first = 0; first < fractional_points_.size(); first += point_tile) {
                int count  = static_cast<int>(std::min(point_tile, fractional_points_.size() - first));
                auto phase = phases(first, count);
                for (int ig = begin; ig < end; ig++) {
                    auto const& g    = phase_indices_[ig];
                    auto const& cart = cartesian_g_[ig];
                    auto value       = coefficients[ig];
                    for (int p = 0; p < count; p++) {
                        auto const& d   = covectors__[first + p];
                        double gradient = cart[0] * d[1] + cart[1] * d[2] + cart[2] * d[3];
                        value += std::complex<double>(d[0], -gradient) *
                                 std::conj(phase(p, g[0]) * phase(p, g[1]) * phase(p, g[2]));
                    }
                    coefficients[ig] = value;
                }
            }
            for (int ig = begin; ig < end; ig++) {
                coefficients[ig] /= gvec_.omega();
            }
        }
        gvec_.comm().allreduce(coefficients.data(), coefficients.size());
        for (int ig = 0; ig < gvec_.count(); ig++) {
            potential__.f_pw_local(ig) = coefficients[gvec_.offset() + ig];
        }
    }

    /// Geometry derivative of the adjoint, including its inverse-volume normalization.
    void
    geometry_adjoint_response(std::vector<std::array<double, 4>> const& covectors__,
                              std::vector<r3::vector<double>> const& point_directions__,
                              r3::matrix<double> const& lattice_direction__,
                              Smooth_periodic_function<double>& potential__) const
    {
        check_geometry_direction(point_directions__, lattice_direction__);
        bool failed = covectors__.size() != fractional_points_.size() || &potential__.gvec() != &gvec_;
        for (auto const& d : covectors__) {
            for (double v : d) {
                failed |= !std::isfinite(v);
            }
        }
        check(failed, "Fourier adjoint response: incompatible buffers or nonfinite covectors");
        auto inverse       = r3::inverse(gvec_.unit_cell_lattice_vectors());
        auto deformation   = r3::dot(inverse, lattice_direction__);
        double dlog_volume = deformation(0, 0) + deformation(1, 1) + deformation(2, 2);
        std::vector<std::complex<double>> coefficients(gvec_.num_gvec());
        #pragma omp parallel for schedule(static)
        for (int begin = 0; begin < gvec_.num_gvec(); begin += gvec_tile) {
            int end = std::min(begin + gvec_tile, gvec_.num_gvec());
            for (size_t first = 0; first < fractional_points_.size(); first += point_tile) {
                int count  = static_cast<int>(std::min(point_tile, fractional_points_.size() - first));
                auto phase = phases(first, count);
                for (int ig = begin; ig < end; ig++) {
                    auto const& g    = phase_indices_[ig];
                    auto const& cart = cartesian_g_[ig];
                    auto dg = -1.0 * r3::dot(r3::transpose(inverse), r3::dot(r3::transpose(lattice_direction__), cart));
                    auto value = coefficients[ig];
                    for (int p = 0; p < count; p++) {
                        auto const& d = covectors__[first + p];
                        r3::vector<double> gradient({d[1], d[2], d[3]});
                        auto displacement = point_directions__[first + p] -
                                            r3::dot(lattice_direction__, unwrapped_fractional_points_[first + p]);
                        double dphase     = r3::dot(cart, displacement);
                        auto raw          = std::complex<double>(d[0], -r3::dot(cart, gradient));
                        value += (raw * std::complex<double>(-dlog_volume, -dphase) -
                                  std::complex<double>(0, r3::dot(dg, gradient))) *
                                 std::conj(phase(p, g[0]) * phase(p, g[1]) * phase(p, g[2]));
                    }
                    coefficients[ig] = value;
                }
            }
            for (int ig = begin; ig < end; ig++) {
                coefficients[ig] /= gvec_.omega();
            }
        }
        gvec_.comm().allreduce(coefficients.data(), coefficients.size());
        for (int ig = 0; ig < gvec_.count(); ig++) {
            potential__.f_pw_local(ig) = coefficients[gvec_.offset() + ig];
        }
    }
};

} // namespace sirius

#endif
