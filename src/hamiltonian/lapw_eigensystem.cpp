/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "hamiltonian.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include <sstream>

namespace sirius {
namespace {

bool
finite_hermitian(mdarray<std::complex<double>, 2> const& matrix__, int n__)
{
    if (matrix__.size(0) != n__ || matrix__.size(1) != n__ || !matrix__.on_host()) {
        return false;
    }
    double error{0}, peak{0};
    for (int j = 0; j < n__; j++) {
        for (int i = 0; i < n__; i++) {
            auto value = matrix__(i, j);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
                return false;
            }
            peak  = std::max(peak, std::abs(value));
            error = std::max(error, std::abs(value - std::conj(matrix__(j, i))));
        }
    }
    return error <= 1e-11 * std::max(1.0, peak);
}

mdarray<std::complex<double>, 2>
project(mdarray<std::complex<double>, 2> const& matrix__, mdarray<std::complex<double>, 2> const& left__,
        mdarray<std::complex<double>, 2> const& right__)
{
    int n = matrix__.size(0), nl = left__.size(1), nr = right__.size(1);
    mdarray<std::complex<double>, 2> work({n, nr}), result({nl, nr});
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', n, nr, n, &one, matrix__.at(memory_t::host), matrix__.ld(), right__.at(memory_t::host),
                  right__.ld(), &zero, work.at(memory_t::host), work.ld());
    la::wrap(la::lib_t::blas)
            .gemm('C', 'N', nl, nr, n, &one, left__.at(memory_t::host), left__.ld(), work.at(memory_t::host), work.ld(),
                  &zero, result.at(memory_t::host), result.ld());
    return result;
}

mdarray<std::complex<double>, 2>
reconstruct(mdarray<std::complex<double>, 2> const& matrix__, mdarray<std::complex<double>, 2> const& left__,
            mdarray<std::complex<double>, 2> const& right__)
{
    int n = left__.size(0), nl = left__.size(1), nr = right__.size(1);
    mdarray<std::complex<double>, 2> work({n, nr}), result({n, n});
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', n, nr, nl, &one, left__.at(memory_t::host), left__.ld(), matrix__.at(memory_t::host),
                  matrix__.ld(), &zero, work.at(memory_t::host), work.ld());
    la::wrap(la::lib_t::blas)
            .gemm('N', 'C', n, n, nr, &one, work.at(memory_t::host), work.ld(), right__.at(memory_t::host),
                  right__.ld(), &zero, result.at(memory_t::host), result.ld());
    return result;
}

mdarray<std::complex<double>, 2>
similarity(mdarray<std::complex<double>, 2> const& matrix__, mdarray<std::complex<double>, 2> const& vectors__,
           bool to_eigenbasis__)
{
    return to_eigenbasis__ ? project(matrix__, vectors__, vectors__) : reconstruct(matrix__, vectors__, vectors__);
}

mdarray<std::complex<double>, 2>
spinor_matrix(std::array<mdarray<std::complex<double>, 2>, 4> const& pauli__)
{
    int nf = pauli__[0].size(0);
    mdarray<std::complex<double>, 2> result({2 * nf, 2 * nf});
    for (int j = 0; j < nf; j++) {
        for (int i = 0; i < nf; i++) {
            result(i, j)           = pauli__[0](i, j) + pauli__[1](i, j);
            result(nf + i, nf + j) = pauli__[0](i, j) - pauli__[1](i, j);
            result(i, nf + j)      = pauli__[2](i, j) - std::complex<double>(0, 1) * pauli__[3](i, j);
            result(nf + i, j)      = pauli__[2](i, j) + std::complex<double>(0, 1) * pauli__[3](i, j);
        }
    }
    return result;
}

mdarray<std::complex<double>, 2>
pauli_density(mdarray<std::complex<double>, 2> const& matrix__, int component__)
{
    int nf = matrix__.size(0) / 2;
    mdarray<std::complex<double>, 2> result({nf, nf});
    for (int j = 0; j < nf; j++) {
        for (int i = 0; i < nf; i++) {
            result(i, j) =
                    component__ < 2
                            ? matrix__(i, j) + (component__ == 0 ? 1.0 : -1.0) * matrix__(nf + i, nf + j)
                            : (component__ == 2 ? std::complex<double>(1, 0) : std::complex<double>(0, 1)) *
                                      (matrix__(i, nf + j) + (component__ == 2 ? 1.0 : -1.0) * matrix__(nf + i, j));
        }
    }
    return result;
}

} // namespace

void
LAPW_eigensystem::validate_matrix(la::dmatrix<std::complex<double>> const& matrix__) const
{
    int n       = dimension_;
    int invalid = matrix__.num_rows() != n || matrix__.num_cols() != n ||
                  matrix__.bs_row() != ctx_.cyclic_block_size() || matrix__.bs_col() != ctx_.cyclic_block_size() ||
                  !matrix__.has_blacs_grid() || &matrix__.blacs_grid() != &ctx_.blacs_grid() ||
                  (matrix__.size_local() && !matrix__.on_host());
    if (!invalid) {
        for (int j = 0; j < matrix__.num_cols_local(); j++) {
            for (int i = 0; i < matrix__.num_rows_local(); i++) {
                auto value = matrix__(i, j);
                invalid |= !std::isfinite(value.real()) || !std::isfinite(value.imag());
            }
        }
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spectrum response requires a finite compatible matrix layout");
    }
}

void
LAPW_eigensystem::validate_occupations(std::vector<double> const& occupations__) const
{
    int invalid = occupations__.size() != energies_.size();
    for (double value : occupations__) {
        invalid |= !std::isfinite(value);
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW occupations must be finite and cover the complete spectrum");
    }
    auto minimum = occupations__, maximum = occupations__;
    kp_.comm().allreduce<double, mpi::op_t::min>(minimum.data(), minimum.size());
    kp_.comm().allreduce<double, mpi::op_t::max>(maximum.data(), maximum.size());
    if (minimum != maximum) {
        RTE_THROW("LAPW occupations and slopes must agree across the k-point communicator");
    }
}

LAPW_eigensystem::LAPW_eigensystem(Hamiltonian_k<double> const& hk__, la::dmatrix<std::complex<double>> const& h__,
                                   la::dmatrix<std::complex<double>> const& s__,
                                   la::dmatrix<std::complex<double>> const* physical_overlap__)
    : LAPW_eigensystem(hk__, h__, s__, space_t::pw_lo)
{
    int mode = physical_overlap__ != nullptr, minimum = mode, maximum = mode;
    kp_.comm().allreduce<int, mpi::op_t::min>(&minimum, 1);
    kp_.comm().allreduce<int, mpi::op_t::max>(&maximum, 1);
    if (minimum != maximum) {
        RTE_THROW("LAPW physical normalization mode must agree across k-point ranks");
    }
    if (physical_overlap__) {
        validate_matrix(*physical_overlap__);
        auto metric       = physical_overlap__->get_full_matrix();
        int invalid       = ctx_.valence_relativity() != relativity_t::iora || !finite_hermitian(metric, dimension_);
        physical_overlap_ = similarity(metric, vectors_, true);
        physical_norms_.resize(dimension_);
        for (int i = 0; i < dimension_; i++) {
            physical_norms_[i] = physical_overlap_(i, i).real();
            invalid |= !std::isfinite(physical_norms_[i]) || physical_norms_[i] <= 0;
        }
        kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("IORA orbital normalization requires a positive finite physical metric");
        }
    }
}

LAPW_eigensystem::LAPW_eigensystem(Hamiltonian_k<double> const& hk__, la::dmatrix<std::complex<double>> const& h__,
                                   la::dmatrix<std::complex<double>> const& s__, space_t space__)
    : ctx_(hk__.H0().ctx())
    , kp_(hk__.kp())
    , dimension_(space__ == space_t::second_variation ? 2 * ctx_.num_fv_states()
                 : ctx_.valence_relativity() == relativity_t::dirac
                         ? 2 * kp_.num_gkvec() + ctx_.unit_cell().mt_lo_basis_size()
                         : kp_.gklo_basis_size())
{
    bool dirac  = ctx_.valence_relativity() == relativity_t::dirac;
    int invalid = !ctx_.full_potential() || ctx_.cfg().parameters().precision_wf() != "fp64" ||
                  (ctx_.valence_relativity() != relativity_t::none && ctx_.valence_relativity() != relativity_t::zora &&
                   space__ != space_t::pw_lo) ||
                  (ctx_.so_correction() && space__ == space_t::pw_lo) || ctx_.hubbard_correction();
    if (space__ == space_t::pw_lo) {
        invalid |= dirac ? ctx_.cfg().control().use_second_variation()
                         : (ctx_.num_mag_dims() == 3 ||
                            (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()));
    } else {
        invalid |= ctx_.num_mag_dims() != 3 || !ctx_.cfg().control().use_second_variation() ||
                   ctx_.num_fv_states() < 1 || ctx_.num_fv_states() > kp_.gklo_basis_size() ||
                   ctx_.num_bands() != 2 * ctx_.num_fv_states();
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spectrum response requires a compatible scalar or native Dirac fp64 representation");
    }
    validate_matrix(h__);
    validate_matrix(s__);
    auto full_h = h__.get_full_matrix();
    overlap_    = s__.get_full_matrix();
    int n = dimension_, bs = ctx_.cyclic_block_size();
    invalid = !finite_hermitian(full_h, n) || !finite_hermitian(overlap_, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW eigensystem matrices must be Hermitian");
    }
    bool standard = space__ == space_t::second_variation;
    if (standard) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                invalid |= std::abs(overlap_(i, j) - (i == j ? 1.0 : 0.0)) > 1e-12;
            }
        }
        kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW("LAPW second-variation response requires an orthonormal first-variation frame");
        }
    }
    auto& solver = standard ? ctx_.std_evp_solver() : ctx_.gen_evp_solver();
    la::dmatrix<std::complex<double>> h(n, n, ctx_.blacs_grid(), bs, bs, get_memory_pool(solver.host_memory_t()));
    la::dmatrix<std::complex<double>> s(n, n, ctx_.blacs_grid(), bs, bs, get_memory_pool(solver.host_memory_t()));
    la::dmatrix<std::complex<double>> z(n, n, ctx_.blacs_grid(), bs, bs, get_memory_pool(solver.host_memory_t()));
    for (int j = 0; j < h.num_cols_local(); j++) {
        for (int i = 0; i < h.num_rows_local(); i++) {
            h(i, j) = h__(i, j);
            s(i, j) = s__(i, j);
        }
    }
    energies_.resize(n);
    if (is_device_memory(solver.data_memory_t())) {
        // GPU eigensolvers stage the host matrices and return host eigenvectors.
        auto& pool = get_memory_pool(solver.data_memory_t());
        h.allocate(pool);
        s.allocate(pool);
        z.allocate(pool);
    }
    int failed = (standard ? solver.solve(n, n, h, energies_.data(), z)
                           : solver.solve(n, n, h, s, energies_.data(), z)) != 0;
    kp_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("failed to solve the complete LAPW spectral eigenproblem");
    }
    vectors_ = z.get_full_matrix();
    // Replicated serial solvers may choose different gauges within a degenerate space.
    kp_.comm().bcast(vectors_.at(memory_t::host), vectors_.size(), 0);
    kp_.comm().bcast(energies_.data(), energies_.size(), 0);
    for (double energy : energies_) {
        invalid |= !std::isfinite(energy);
    }
    for (size_t i = 0; i < vectors_.size(); i++) {
        invalid |= !std::isfinite(vectors_[i].real()) || !std::isfinite(vectors_[i].imag());
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite complete LAPW eigensystem");
    }
}

LAPW_eigensystem::perturbation_t
LAPW_eigensystem::perturbation(la::dmatrix<std::complex<double>> const& h__,
                               la::dmatrix<std::complex<double>> const& s__,
                               la::dmatrix<std::complex<double>> const* physical_overlap__) const
{
    int invalid = bool(physical_overlap__) != !physical_norms_.empty();
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("IORA response must include the ordinary-overlap direction on every rank");
    }
    validate_matrix(h__);
    validate_matrix(s__);
    auto h = h__.get_full_matrix(), s = s__.get_full_matrix();
    int n   = energies_.size();
    invalid = !finite_hermitian(h, n) || !finite_hermitian(s, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW matrix directions must be Hermitian");
    }
    perturbation_t result;
    result.reference = this;
    // Remove the common metric-proportional shift before projection. Otherwise tiny
    // off-diagonal roundoff in C^dagger O C can be amplified by nearly degenerate gaps.
    double shift = h(0, 0).real() / overlap_(0, 0).real();
    bool uniform = true;
    for (size_t i = 0; i < h.size(); i++) {
        auto common     = shift * overlap_[i];
        double roundoff = 8 * std::numeric_limits<double>::epsilon() * (std::abs(h[i]) + std::abs(common));
        uniform &= std::abs(h[i] - common) <= roundoff;
        h[i] -= common;
    }
    if (uniform) {
        // A representable constant shift is exactly diagonal, including within degenerate subspaces.
        result.h = mdarray<std::complex<double>, 2>({n, n});
        result.h.zero();
    } else {
        result.h = similarity(h, vectors_, true);
    }
    result.s = similarity(s, vectors_, true);
    result.energies.resize(n);
    for (int i = 0; i < n; i++) {
        result.h(i, i) += shift;
        result.energies[i] = std::real(result.h(i, i) - energies_[i] * result.s(i, i));
        invalid |= !std::isfinite(result.energies[i]);
    }
    if (physical_overlap__) {
        validate_matrix(*physical_overlap__);
        auto metric = physical_overlap__->get_full_matrix();
        invalid |= !finite_hermitian(metric, n);
        auto transformed = similarity(metric, vectors_, true);
        result.physical_norms.resize(n);
        double eps = std::numeric_limits<double>::epsilon();
        for (int i = 0; i < n; i++) {
            double norm = transformed(i, i).real() - physical_norms_[i] * result.s(i, i).real();
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    continue;
                }
                double gap   = energies_[i] - energies_[j];
                double scale = std::max({1.0, std::abs(energies_[i]), std::abs(energies_[j])});
                if (std::abs(gap) <= 64 * eps * scale) {
                    invalid |= std::abs(physical_overlap_(i, j)) >
                               std::sqrt(eps) * std::sqrt(physical_norms_[i] * physical_norms_[j]);
                    continue;
                }
                auto mixing = (result.h(j, i) - energies_[i] * result.s(j, i)) / gap;
                norm += 2 * std::real(physical_overlap_(i, j) * mixing);
            }
            result.physical_norms[i] = norm;
            invalid |= !std::isfinite(norm);
        }
    }
    invalid |= !finite_hermitian(result.h, n) || !finite_hermitian(result.s, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite or non-Hermitian LAPW spectral perturbation");
    }
    return result;
}

mdarray<std::complex<double>, 2>
LAPW_eigensystem::density_matrix(std::vector<double> const& occupations__) const
{
    validate_occupations(occupations__);
    int n = energies_.size();
    mdarray<std::complex<double>, 2> diagonal({n, n});
    diagonal.zero();
    for (int i = 0; i < n; i++) {
        diagonal(i, i) = occupations__[i] / (physical_norms_.empty() ? 1 : physical_norms_[i]);
    }
    auto result = similarity(diagonal, vectors_, false);
    int invalid = !finite_hermitian(result, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite or non-Hermitian LAPW density matrix");
    }
    return result;
}

mdarray<std::complex<double>, 2>
LAPW_eigensystem::density_matrix_response(perturbation_t const& direction__, std::vector<double> const& occupations__,
                                          std::vector<double> const& slopes__, double mu_response__) const
{
    validate_occupations(occupations__);
    validate_occupations(slopes__);
    int n       = energies_.size();
    int invalid = direction__.reference != this || !finite_hermitian(direction__.h, n) ||
                  !finite_hermitian(direction__.s, n) || !std::isfinite(mu_response__) ||
                  direction__.physical_norms.size() != physical_norms_.size();
    for (double value : direction__.physical_norms) {
        invalid |= !std::isfinite(value);
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW density response requires a finite perturbation of the same eigensystem");
    }
    double minimum_mu = mu_response__, maximum_mu = mu_response__;
    kp_.comm().allreduce<double, mpi::op_t::min>(&minimum_mu, 1);
    kp_.comm().allreduce<double, mpi::op_t::max>(&maximum_mu, 1);
    if (minimum_mu != maximum_mu) {
        RTE_THROW("LAPW chemical-potential response must agree across the k-point communicator");
    }
    mdarray<std::complex<double>, 2> response({n, n});
    auto occupations = occupations__, slopes = slopes__;
    if (!physical_norms_.empty()) {
        for (int i = 0; i < n; i++) {
            occupations[i] /= physical_norms_[i];
            slopes[i] /= physical_norms_[i];
        }
    }
    double eps = std::numeric_limits<double>::epsilon();
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double gap        = energies_[i] - energies_[j],
                   scale      = std::max({1.0, std::abs(energies_[i]), std::abs(energies_[j])});
            double difference = occupations[i] - occupations[j], slope = 0.5 * (slopes[i] + slopes[j]);
            double weight = std::max({1.0, std::abs(occupations[i]), std::abs(occupations[j])});
            if (std::abs(gap) <= 64 * eps * scale) {
                invalid |= std::abs(difference) > 64 * eps * weight ||
                           std::abs(slopes[i] - slopes[j]) >
                                   std::sqrt(eps) * std::max({1.0, std::abs(slopes[i]), std::abs(slopes[j])});
            } else if (std::abs(gap) > std::sqrt(eps) * scale ||
                       std::abs(difference - slope * gap) > 64 * eps * weight) {
                slope = difference / gap;
            }
            auto effective_h = direction__.h(i, j) - 0.5 * (energies_[i] + energies_[j]) * direction__.s(i, j);
            response(i, j)   = slope * effective_h - 0.5 * (occupations[i] + occupations[j]) * direction__.s(i, j);
            if (i == j) {
                // Subtract the common potential shift before multiplying the occupation slope.
                double de      = direction__.h(i, i).real() - energies_[i] * direction__.s(i, i).real();
                response(i, i) = slopes[i] * (de - mu_response__) - occupations[i] * direction__.s(i, i).real();
                if (!physical_norms_.empty()) {
                    response(i, i) -= occupations[i] * direction__.physical_norms[i] / physical_norms_[i];
                }
            }
        }
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW degenerate levels require common occupations and energy slopes");
    }
    auto result = similarity(response, vectors_, false);
    invalid     = !finite_hermitian(result, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite or non-Hermitian LAPW density-matrix response");
    }
    return result;
}

std::array<mdarray<std::complex<double>, 2>, 4>
LAPW_spinor_eigensystem::matrices(matrix_set_t const& h__) const
{
    int invalid{0};
    for (auto* matrix : h__) {
        invalid |= matrix == nullptr;
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spinor response requires all four Pauli matrices");
    }
    std::array<mdarray<std::complex<double>, 2>, 4> result;
    for (int s = 0; s < 4; s++) {
        auxiliary_->validate_matrix(*h__[s]);
        result[s] = h__[s]->get_full_matrix();
        invalid |= !finite_hermitian(result[s], kp_.gklo_basis_size());
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spinor Pauli matrices must be Hermitian");
    }
    return result;
}

LAPW_spinor_eigensystem::LAPW_spinor_eigensystem(Hamiltonian_k<double> const& hk__, matrix_set_t const& h__,
                                                 la::dmatrix<std::complex<double>> const& s__)
    : ctx_(hk__.H0().ctx())
    , kp_(hk__.kp())
{
    int invalid = h__[0] == nullptr;
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spinor response requires a first-variation Hamiltonian");
    }
    auxiliary_.reset(new LAPW_eigensystem(hk__, *h__[0], s__, LAPW_eigensystem::space_t::first_variation));
    h_    = matrices(h__);
    int n = kp_.gklo_basis_size(), nf = ctx_.num_fv_states(), nb = 2 * nf, bs = ctx_.cyclic_block_size();
    frame_ = mdarray<std::complex<double>, 2>({n, nf});
    for (int j = 0; j < nf; j++) {
        for (int i = 0; i < n; i++) {
            frame_(i, j) = auxiliary_->vectors_(i, j);
        }
    }
    std::array<mdarray<std::complex<double>, 2>, 4> projected;
    for (int s = 0; s < 4; s++) {
        projected[s] = project(h_[s], frame_, frame_);
    }
    auto full_h = spinor_matrix(projected);
    la::dmatrix<std::complex<double>> h(nb, nb, ctx_.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> overlap(nb, nb, ctx_.blacs_grid(), bs, bs);
    for (int j = 0; j < h.num_cols_local(); j++) {
        for (int i = 0; i < h.num_rows_local(); i++) {
            h(i, j)       = full_h(h.irow(i), h.icol(j));
            overlap(i, j) = h.irow(i) == h.icol(j) ? 1.0 : 0.0;
        }
    }
    spinor_.reset(new LAPW_eigensystem(hk__, h, overlap, LAPW_eigensystem::space_t::second_variation));
}

LAPW_spinor_eigensystem::perturbation_t
LAPW_spinor_eigensystem::perturbation(matrix_set_t const& h__, la::dmatrix<std::complex<double>> const& s__) const
{
    auto fields    = matrices(h__);
    auto direction = auxiliary_->perturbation(*h__[0], s__);
    int n = kp_.gklo_basis_size(), nf = ctx_.num_fv_states(), nb = 2 * nf, bs = ctx_.cyclic_block_size();
    int invalid{0};
    mdarray<std::complex<double>, 2> coefficients({n, nf});
    // Parallel transport within the retained subspace avoids denominators at its
    // internal degeneracies. Only the excluded-space motion needs energy gaps.
    auto const& e = auxiliary_->energies();
    for (int j = 0; j < nf; j++) {
        for (int i = 0; i < n; i++) {
            coefficients(i, j) = -0.5 * direction.s(i, j);
            if (i >= nf) {
                double gap = e[j] - e[i], scale = std::max({1.0, std::abs(e[i]), std::abs(e[j])});
                if (std::abs(gap) <= 64 * std::numeric_limits<double>::epsilon() * scale) {
                    invalid = 1;
                } else {
                    coefficients(i, j) = (direction.h(i, j) - e[j] * direction.s(i, j)) / gap;
                }
            }
        }
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW first-variation response requires a separated retained/excluded subspace");
    }
    perturbation_t result;
    result.reference = this;
    result.frame     = mdarray<std::complex<double>, 2>({n, nf});
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', n, nf, n, &one, auxiliary_->vectors_.at(memory_t::host), auxiliary_->vectors_.ld(),
                  coefficients.at(memory_t::host), coefficients.ld(), &zero, result.frame.at(memory_t::host),
                  result.frame.ld());
    std::array<mdarray<std::complex<double>, 2>, 4> projected;
    for (int s = 0; s < 4; s++) {
        projected[s] = project(fields[s], frame_, frame_);
        auto left = project(h_[s], result.frame, frame_), right = project(h_[s], frame_, result.frame);
        for (size_t i = 0; i < projected[s].size(); i++) {
            projected[s][i] += left[i] + right[i];
        }
    }
    auto full_h = spinor_matrix(projected);
    la::dmatrix<std::complex<double>> h(nb, nb, ctx_.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> overlap(nb, nb, ctx_.blacs_grid(), bs, bs);
    overlap.zero();
    for (int j = 0; j < h.num_cols_local(); j++) {
        for (int i = 0; i < h.num_rows_local(); i++) {
            h(i, j) = full_h(h.irow(i), h.icol(j));
        }
    }
    result.spinor = spinor_->perturbation(h, overlap);
    return result;
}

mdarray<std::complex<double>, 3>
LAPW_spinor_eigensystem::density_matrix(std::vector<double> const& occupations__) const
{
    auto spinor = spinor_->density_matrix(occupations__);
    int n       = kp_.gklo_basis_size(), invalid{0};
    mdarray<std::complex<double>, 3> result({n, n, 4});
    for (int s = 0; s < 4; s++) {
        auto block = reconstruct(pauli_density(spinor, s), frame_, frame_);
        invalid |= !finite_hermitian(block, n);
        std::copy(block.at(memory_t::host), block.at(memory_t::host) + block.size(), &result(0, 0, s));
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite or non-Hermitian LAPW spinor density matrix");
    }
    return result;
}

mdarray<std::complex<double>, 3>
LAPW_spinor_eigensystem::density_matrix_response(perturbation_t const& direction__,
                                                 std::vector<double> const& occupations__,
                                                 std::vector<double> const& slopes__, double mu_response__) const
{
    int n = kp_.gklo_basis_size(), nf = ctx_.num_fv_states();
    int invalid = direction__.reference != this || direction__.frame.size(0) != static_cast<size_t>(n) ||
                  direction__.frame.size(1) != static_cast<size_t>(nf) || !direction__.frame.on_host();
    if (!invalid) {
        for (size_t i = 0; i < direction__.frame.size(); i++) {
            invalid |= !std::isfinite(direction__.frame[i].real()) || !std::isfinite(direction__.frame[i].imag());
        }
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spinor density response requires a finite perturbation of the same eigensystem");
    }
    auto spinor   = spinor_->density_matrix(occupations__);
    auto response = spinor_->density_matrix_response(direction__.spinor, occupations__, slopes__, mu_response__);
    mdarray<std::complex<double>, 3> result({n, n, 4});
    for (int s = 0; s < 4; s++) {
        auto q     = pauli_density(spinor, s);
        auto block = reconstruct(pauli_density(response, s), frame_, frame_);
        auto left = reconstruct(q, direction__.frame, frame_), right = reconstruct(q, frame_, direction__.frame);
        for (size_t i = 0; i < block.size(); i++) {
            block[i] += left[i] + right[i];
        }
        invalid |= !finite_hermitian(block, n);
        std::copy(block.at(memory_t::host), block.at(memory_t::host) + block.size(), &result(0, 0, s));
    }
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("nonfinite or non-Hermitian LAPW spinor density-matrix response");
    }
    return result;
}

LAPW_density_response::LAPW_density_response(Hamiltonian0<double> const& h0__, K_point_set& ks__)
    : ks_(ks__)
    , fermi_{ks__.energy_fermi_fixed_mag(0), ks__.energy_fermi_fixed_mag(ks__.ctx().num_spinors() - 1)}
    , width_(ks__.ctx().smearing_width())
    , moment_(ks__.ctx().cfg().parameters().fixed_mag())
    , charge_(ks__.ctx().cfg().parameters().extra_charge())
    , smearing_(static_cast<int>(ks__.ctx().smearing()))
{
    auto& ctx   = ks_.ctx();
    bool dirac  = ctx.valence_relativity() == relativity_t::dirac;
    bool spinor = ctx.num_mag_dims() == 3 && !dirac;
    int ns = ctx.num_spinors(), nb = ctx.num_bands();
    // SCF updates the mixed potential after solving the bands. Resolve that
    // finite residual to the requested response accuracy, not a fixed roundoff bound.
    double tolerance =
            std::max(std::sqrt(std::numeric_limits<double>::epsilon()), ctx.cb().external_xc_response_tolerance_);
    double spectrum_error{0};
    int invalid = &ctx != &h0__.ctx() || !ctx.full_potential() || ctx.cfg().parameters().precision_wf() != "fp64" ||
                  (spinor && ctx.valence_relativity() != relativity_t::none &&
                   ctx.valence_relativity() != relativity_t::zora) ||
                  (ctx.so_correction() && !spinor) || ctx.hubbard_correction() ||
                  (!spinor && ns == 2 && ctx.cfg().control().use_second_variation()) ||
                  (spinor && !ctx.cfg().control().use_second_variation()) ||
                  (dirac && ctx.cfg().control().use_second_variation());
    for (int s = 0; s < ns; s++) {
        invalid |= !std::isfinite(fermi_[s]);
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW density response requires a compatible scalar, NR/ZORA spinor or native Dirac fp64 reference");
    }
    lapw_hamiltonian_response_t magnetic;
    if (spinor) {
        magnetic = h0__.lapw_spin_fields();
    }
    for (auto it : ks_.spl_num_kpoints()) {
        auto& kp = *ks_.get<double>(it.i);
        int n    = dirac ? 2 * kp.num_gkvec() + ctx.unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
        int bs   = ctx.cyclic_block_size();
        bands_.emplace_back(std::array<index_range, 2>{nb, ns});
        occupations_.emplace_back(std::array<index_range, 2>{nb, ns});
        spectra_.emplace_back();
        spinor_spectra_.emplace_back();
        try {
            Hamiltonian_k<double> hk(h0__, kp);
            auto& pool = get_memory_pool(ctx.gen_evp_solver().host_memory_t());
            la::dmatrix<std::complex<double>> h(n, n, ctx.blacs_grid(), bs, bs, pool);
            la::dmatrix<std::complex<double>> s(n, n, ctx.blacs_grid(), bs, bs, pool);
            std::unique_ptr<la::dmatrix<std::complex<double>>> physical_s;
            if (ctx.valence_relativity() == relativity_t::iora) {
                physical_s = std::make_unique<la::dmatrix<std::complex<double>>>(n, n, ctx.blacs_grid(), bs, bs);
                hk.set_fv_h_o(0, h, *physical_s, true);
            }
            if (spinor) {
                hk.set_fv_h_o(0, h, s);
                std::array<std::unique_ptr<la::dmatrix<std::complex<double>>>, 3> spin_fields;
                LAPW_spinor_eigensystem::matrix_set_t pauli{&h};
                for (int j = 0; j < 3; j++) {
                    spin_fields[j] =
                            std::make_unique<la::dmatrix<std::complex<double>>>(n, n, ctx.blacs_grid(), bs, bs);
                    hk.set_fv_h_field_response(j + 1, magnetic, *spin_fields[j]);
                    pauli[j + 1] = spin_fields[j].get();
                }
                spinor_spectra_.back() = std::make_unique<LAPW_spinor_eigensystem>(hk, pauli, s);
            }
            for (int spin = 0; spin < ns; spin++) {
                if (!spinor) {
                    if (dirac) {
                        hk.set_dirac_h_o(h, s);
                    } else {
                        hk.set_fv_h_o(spin, h, s);
                    }
                    spectra_.back()[spin] = std::make_unique<LAPW_eigensystem>(hk, h, s, physical_s.get());
                }
                auto const& energies = spinor ? spinor_spectra_.back()->energies() : spectra_.back()[spin]->energies();
                invalid |= energies.size() < static_cast<size_t>(nb);
                for (int b = 0; b < nb; b++) {
                    double energy = kp.band_energy(b, spin), f = kp.band_occupancy(b, spin);
                    bands_.back()(b, spin)       = energy;
                    occupations_.back()(b, spin) = f;
                    invalid |= !std::isfinite(energy) || !std::isfinite(f);
                    if (static_cast<size_t>(b) < energies.size()) {
                        double exact   = energies[b];
                        double error   = std::abs(energy - exact) / std::max({1.0, std::abs(energy), std::abs(exact)});
                        spectrum_error = std::max(spectrum_error, error);
                        invalid |= error > tolerance;
                    }
                }
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    ctx.comm().allreduce<double, mpi::op_t::max>(&spectrum_error, 1);
    RTE_OUT(ctx.out(2)) << "LAPW response reference: relative band mismatch " << spectrum_error << ", tolerance "
                        << tolerance << '\n';
    if (invalid) {
        std::stringstream message;
        message << "stored bands do not match the complete frozen LAPW Hamiltonian (relative error " << spectrum_error
                << ", tolerance " << tolerance << ')';
        RTE_THROW(message.str());
    }
}

void
LAPW_density_response::validate_reference() const
{
    auto& ctx   = ks_.ctx();
    int invalid = spectra_.size() != ks_.spl_num_kpoints().local_size() || spinor_spectra_.size() != spectra_.size() ||
                  width_ != ctx.smearing_width() || moment_ != ctx.cfg().parameters().fixed_mag() ||
                  charge_ != ctx.cfg().parameters().extra_charge() || smearing_ != static_cast<int>(ctx.smearing()) ||
                  static_cast<bool>(ctx.band_occ_callback());
    for (int s = 0; s < ctx.num_spinors(); s++) {
        invalid |= fermi_[s] != ks_.energy_fermi_fixed_mag(s);
    }
    size_t ik{0};
    if (!invalid) {
        for (auto it : ks_.spl_num_kpoints()) {
            auto const& kp = *ks_.get<double>(it.i);
            invalid |= bands_[ik].size(0) != ctx.num_bands() || bands_[ik].size(1) != ctx.num_spinors() ||
                       static_cast<bool>(spinor_spectra_[ik]) !=
                               (ctx.num_mag_dims() == 3 && ctx.valence_relativity() != relativity_t::dirac);
            if (!invalid) {
                for (int s = 0; s < ctx.num_spinors(); s++) {
                    for (int b = 0; b < ctx.num_bands(); b++) {
                        invalid |= bands_[ik](b, s) != kp.band_energy(b, s) ||
                                   occupations_[ik](b, s) != kp.band_occupancy(b, s);
                    }
                }
            }
            ik++;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW density response reference changed; rebuild its frozen spectra");
    }
}

std::vector<mdarray<std::complex<double>, 3>>
LAPW_density_response::density_matrices() const
{
    validate_reference();
    auto& ctx = ks_.ctx();
    std::vector<mdarray<std::complex<double>, 3>> result;
    int invalid{0};
    for (size_t ik = 0; ik < spectra_.size(); ik++) {
        try {
            if (spinor_spectra_[ik]) {
                std::vector<double> f(ctx.num_bands());
                std::copy_n(occupations_[ik].at(memory_t::host), f.size(), f.data());
                result.push_back(spinor_spectra_[ik]->density_matrix(f));
                continue;
            }
            int n = spectra_[ik][0]->energies().size();
            result.emplace_back(std::array<index_range, 3>{n, n, ctx.num_spinors()});
            for (int s = 0; s < ctx.num_spinors(); s++) {
                std::vector<double> f(n, 0);
                for (int b = 0; b < ctx.num_bands(); b++) {
                    f[b] = occupations_[ik](b, s);
                }
                auto p = spectra_[ik][s]->density_matrix(f);
                std::copy_n(p.at(memory_t::host), p.size(), result.back().at(memory_t::host) + s * p.size());
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("failed to construct the frozen LAPW density matrices");
    }
    return result;
}

std::vector<mdarray<std::complex<double>, 3>>
LAPW_density_response::density_matrix_response(std::vector<matrix_direction_t> const& directions__, double* entropy__,
                                               double* eval_sum__) const
{
    validate_reference();
    auto& ctx   = ks_.ctx();
    bool dirac  = ctx.valence_relativity() == relativity_t::dirac;
    bool spinor = ctx.num_mag_dims() == 3 && !dirac;
    int emin = (entropy__ != nullptr) + 2 * (eval_sum__ != nullptr), emax = emin;
    ctx.comm().allreduce<int, mpi::op_t::min>(&emin, 1);
    ctx.comm().allreduce<int, mpi::op_t::max>(&emax, 1);
    int invalid = directions__.size() != spectra_.size() || emin != emax;
    for (auto const& d : directions__) {
        for (int s = 0; s < (spinor ? 4 : ctx.num_spinors()); s++) {
            invalid |= !d.h[s];
        }
        for (int s = 0; s < ctx.num_spinors(); s++) {
            invalid |= !d.s[s];
            invalid |= bool(d.physical_s[s]) != (ctx.valence_relativity() == relativity_t::iora);
        }
        invalid |= spinor && d.s[1] != nullptr;
        invalid |= dirac && (d.h[1] || d.h[2] || d.h[3] || d.s[1] || d.physical_s[1]);
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW density response requires H/S directions for every local k-point and spin");
    }
    std::vector<std::array<LAPW_eigensystem::perturbation_t, 2>> perturbations(spectra_.size());
    std::vector<LAPW_spinor_eigensystem::perturbation_t> spinor_perturbations(spinor ? spectra_.size() : 0);
    std::vector<mdarray<double, 2>> energy_directions;
    for (size_t ik = 0; ik < spectra_.size(); ik++) {
        energy_directions.emplace_back(std::array<index_range, 2>{ctx.num_bands(), ctx.num_spinors()});
        try {
            if (spinor) {
                spinor_perturbations[ik] =
                        spinor_spectra_[ik]->perturbation(directions__[ik].h, *directions__[ik].s[0]);
                std::copy_n(spinor_perturbations[ik].spinor.energies.data(), ctx.num_bands(),
                            energy_directions.back().at(memory_t::host));
                continue;
            }
            for (int s = 0; s < ctx.num_spinors(); s++) {
                perturbations[ik][s] = spectra_[ik][s]->perturbation(*directions__[ik].h[s], *directions__[ik].s[s],
                                                                     directions__[ik].physical_s[s]);
                for (int b = 0; b < ctx.num_bands(); b++) {
                    energy_directions.back()(b, s) = perturbations[ik][s].energies[b];
                }
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("invalid LAPW Hamiltonian or overlap direction");
    }
    auto population = ks_.band_occupation_response(energy_directions);
    std::vector<mdarray<std::complex<double>, 3>> result;
    for (size_t ik = 0; ik < spectra_.size(); ik++) {
        try {
            if (spinor) {
                std::vector<double> f(ctx.num_bands()), slopes(ctx.num_bands());
                std::copy_n(occupations_[ik].at(memory_t::host), f.size(), f.data());
                std::copy_n(population.slopes[ik].at(memory_t::host), slopes.size(), slopes.data());
                result.push_back(spinor_spectra_[ik]->density_matrix_response(spinor_perturbations[ik], f, slopes,
                                                                              population.fermi[0]));
                continue;
            }
            int n = spectra_[ik][0]->energies().size();
            result.emplace_back(std::array<index_range, 3>{n, n, ctx.num_spinors()});
            for (int s = 0; s < ctx.num_spinors(); s++) {
                // Virtual states participate in mixing without extending the stored band population.
                std::vector<double> f(n, 0), slopes(n, 0);
                for (int b = 0; b < ctx.num_bands(); b++) {
                    f[b]      = occupations_[ik](b, s);
                    slopes[b] = population.slopes[ik](b, s);
                }
                auto p = spectra_[ik][s]->density_matrix_response(perturbations[ik][s], f, slopes, population.fermi[s]);
                std::copy_n(p.at(memory_t::host), p.size(), result.back().at(memory_t::host) + s * p.size());
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("failed to construct the constrained LAPW density response");
    }
    if (entropy__ || eval_sum__) {
        std::array<double, 2> value{};
        size_t ik{0};
        for (auto it : ks_.spl_num_kpoints()) {
            auto const& kp = *ks_.get<double>(it.i);
            for (int s = 0; s < ctx.num_spinors(); s++) {
                for (int b = 0; b < ctx.num_bands(); b++) {
                    double weight = kp.weight() / kp.comm().size();
                    double df     = population.occupations[ik](b, s);
                    value[0] -= weight * kp.band_energy(b, s) * df;
                    value[1] +=
                            weight * (occupations_[ik](b, s) * energy_directions[ik](b, s) + kp.band_energy(b, s) * df);
                }
            }
            ik++;
        }
        ctx.comm().allreduce(value.data(), value.size());
        if (!std::isfinite(value[0]) || !std::isfinite(value[1])) {
            RTE_THROW("nonfinite LAPW occupation or eigenvalue-sum response");
        }
        if (entropy__) {
            *entropy__ = value[0];
        }
        if (eval_sum__) {
            *eval_sum__ = value[1];
        }
    }
    return result;
}

} // namespace sirius
