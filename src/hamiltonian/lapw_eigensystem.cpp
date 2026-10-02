/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "hamiltonian.hpp"
#include "k_point/k_point.hpp"

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
similarity(mdarray<std::complex<double>, 2> const& matrix__, mdarray<std::complex<double>, 2> const& vectors__,
           bool to_eigenbasis__)
{
    int n = matrix__.size(0);
    mdarray<std::complex<double>, 2> work({n, n}), result({n, n});
    auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
    auto const& left  = to_eigenbasis__ ? matrix__ : vectors__;
    auto const& right = to_eigenbasis__ ? vectors__ : matrix__;
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', n, n, n, &one, left.at(memory_t::host), left.ld(), right.at(memory_t::host), right.ld(),
                  &zero, work.at(memory_t::host), work.ld());
    if (to_eigenbasis__) {
        la::wrap(la::lib_t::blas)
                .gemm('C', 'N', n, n, n, &one, vectors__.at(memory_t::host), vectors__.ld(), work.at(memory_t::host),
                      work.ld(), &zero, result.at(memory_t::host), result.ld());
    } else {
        la::wrap(la::lib_t::blas)
                .gemm('N', 'C', n, n, n, &one, work.at(memory_t::host), work.ld(), vectors__.at(memory_t::host),
                      vectors__.ld(), &zero, result.at(memory_t::host), result.ld());
    }
    return result;
}

} // namespace

void
LAPW_eigensystem::validate_matrix(la::dmatrix<std::complex<double>> const& matrix__) const
{
    int n       = kp_.gklo_basis_size();
    int invalid = matrix__.num_rows() != n || matrix__.num_cols() != n ||
                  matrix__.num_rows_local() != kp_.gklo_basis_size_row() ||
                  matrix__.num_cols_local() != kp_.gklo_basis_size_col() ||
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
        RTE_THROW("LAPW spectrum response requires a finite compatible PW/LO matrix layout");
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
                                   la::dmatrix<std::complex<double>> const& s__)
    : ctx_(hk__.H0().ctx())
    , kp_(hk__.kp())
{
    int invalid = !ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU ||
                  ctx_.valence_relativity() != relativity_t::none || ctx_.so_correction() || ctx_.num_mag_dims() == 3 ||
                  ctx_.hubbard_correction() || (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation());
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW spectrum response requires nonrelativistic scalar/collinear CPU fp64 matrices");
    }
    validate_matrix(h__);
    validate_matrix(s__);
    auto full_h = h__.get_full_matrix();
    overlap_    = s__.get_full_matrix();
    int n = kp_.gklo_basis_size(), bs = ctx_.cyclic_block_size();
    invalid = !finite_hermitian(full_h, n) || !finite_hermitian(overlap_, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW eigensystem matrices must be Hermitian");
    }
    auto& solver = ctx_.gen_evp_solver();
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
    int failed = solver.solve(n, n, h, s, energies_.data(), z) != 0;
    kp_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("failed to solve the complete LAPW generalized eigenproblem");
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
                               la::dmatrix<std::complex<double>> const& s__) const
{
    validate_matrix(h__);
    validate_matrix(s__);
    auto h = h__.get_full_matrix(), s = s__.get_full_matrix();
    int n       = energies_.size();
    int invalid = !finite_hermitian(h, n) || !finite_hermitian(s, n);
    kp_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW matrix directions must be Hermitian");
    }
    perturbation_t result;
    result.reference = this;
    result.h         = similarity(h, vectors_, true);
    result.s         = similarity(s, vectors_, true);
    result.energies.resize(n);
    for (int i = 0; i < n; i++) {
        result.energies[i] = std::real(result.h(i, i) - energies_[i] * result.s(i, i));
        invalid |= !std::isfinite(result.energies[i]);
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
        diagonal(i, i) = occupations__[i];
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
                  !finite_hermitian(direction__.s, n) || !std::isfinite(mu_response__);
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
    double eps = std::numeric_limits<double>::epsilon();
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double gap        = energies_[i] - energies_[j],
                   scale      = std::max({1.0, std::abs(energies_[i]), std::abs(energies_[j])});
            double difference = occupations__[i] - occupations__[j], slope = 0.5 * (slopes__[i] + slopes__[j]);
            double weight = std::max({1.0, std::abs(occupations__[i]), std::abs(occupations__[j])});
            if (std::abs(gap) <= 64 * eps * scale) {
                invalid |= std::abs(difference) > 64 * eps * weight ||
                           std::abs(slopes__[i] - slopes__[j]) >
                                   std::sqrt(eps) * std::max({1.0, std::abs(slopes__[i]), std::abs(slopes__[j])});
            } else if (std::abs(gap) > std::sqrt(eps) * scale ||
                       std::abs(difference - slope * gap) > 64 * eps * weight) {
                slope = difference / gap;
            }
            auto effective_h = direction__.h(i, j) - 0.5 * (energies_[i] + energies_[j]) * direction__.s(i, j);
            response(i, j)   = slope * effective_h - 0.5 * (occupations__[i] + occupations__[j]) * direction__.s(i, j);
            if (i == j) {
                response(i, i) -= slopes__[i] * mu_response__;
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

} // namespace sirius
