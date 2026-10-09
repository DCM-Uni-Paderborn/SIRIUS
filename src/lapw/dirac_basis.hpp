/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DIRAC_BASIS_HPP__
#define __DIRAC_BASIS_HPP__

#include "lapw/matching_coefficients.hpp"
#include "core/fft/fft.hpp"
#include "lapw/step_function.hpp"

namespace sirius {

/// Frozen physical Dirac LAPW basis: two positive-energy PW blocks followed by atom-local full-j LOs.
/** Atom transformations and matrices are replicated across the G+k communicator, as in LAPW spectral response.
 *  Both physical spinor components enter the metric. The cell, G+k set and radial functions must remain unchanged
 *  while this frozen basis is used. This is not the scalar first-variation PW/LO layout. */
class Dirac_lapw_basis
{
  private:
    Unit_cell const& cell_;
    fft::Gvec const& gkvec_;
    Matching_coefficients matching_;
    std::vector<mdarray<std::complex<double>, 2>> atoms_;

    void
    check(bool invalid__, char const* message__) const
    {
        int invalid = invalid__;
        gkvec_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        if (invalid) {
            RTE_THROW(message__);
        }
    }

    mdarray<std::complex<double>, 2>
    atom_coefficients(int ia__, radial_basis_response_t const* response__,
                      mdarray<double, 2> const* displacements__ = nullptr,
                      r3::matrix<double> const* strain__        = nullptr) const
    {
        auto const& atom = cell_.atom(ia__);
        auto const& type = atom.type();
        int ng = gkvec_.num_gvec(), naw = type.mt_aw_basis_size();
        mdarray<std::complex<double>, 3> local({gkvec_.count(), naw, 2}), full({ng, naw, 2});
        if (displacements__) {
            r3::vector<double> d{(*displacements__)(0, ia__), (*displacements__)(1, ia__), (*displacements__)(2, ia__)};
            matching_.generate_geometry_response<false>(atom, d, *strain__, local);
        } else if (response__) {
            matching_.generate_potential_response<false>(atom, *response__, local);
        } else {
            matching_.generate<false>(atom, local);
        }
        for (int s : {0, 1}) {
            for (int xi = 0; xi < naw; xi++) {
                auto send = gkvec_.count() ? &local(0, xi, s) : nullptr;
                gkvec_.comm().allgather(send, &full(0, xi, s), gkvec_.count(), gkvec_.offset());
            }
        }
        mdarray<std::complex<double>, 2> result({type.mt_basis_size(), size()});
        result.zero();
        for (int s : {0, 1}) {
            for (int ig = 0; ig < ng; ig++) {
                for (int xi = 0; xi < naw; xi++) {
                    result(xi, s * ng + ig) = full(ig, xi, s);
                }
            }
        }
        if (!response__ && !displacements__) {
            int offset = 2 * ng;
            for (int i = 0; i < ia__; i++) {
                offset += cell_.atom(i).mt_lo_basis_size();
            }
            for (int lo = 0; lo < type.mt_lo_basis_size(); lo++) {
                result(naw + lo, offset + lo) = 1;
            }
        }
        return result;
    }

    void
    validate_geometry(mdarray<double, 2> const* displacements__, r3::matrix<double> const* strain__) const
    {
        int mode = (displacements__ ? 1 : 0) + (strain__ ? 2 : 0), minimum = mode, maximum = mode;
        gkvec_.comm().allreduce<int, mpi::op_t::min>(&minimum, 1);
        gkvec_.comm().allreduce<int, mpi::op_t::max>(&maximum, 1);
        bool invalid = minimum != maximum || (mode != 0 && mode != 3);
        if (displacements__) {
            invalid |= displacements__->size(0) != 3 ||
                       displacements__->size(1) != static_cast<size_t>(cell_.num_atoms()) ||
                       !displacements__->on_host();
        }
        check(invalid, "Dirac LAPW geometry: matching atom and strain directions must agree across ranks");
        if (!mode) {
            return;
        }
        std::vector<double> values(displacements__->begin(), displacements__->end());
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                values.push_back((*strain__)(x, y));
            }
        }
        for (double value : values) {
            invalid |= !std::isfinite(value);
        }
        check(invalid, "Dirac LAPW geometry requires finite directions");
        auto root = values;
        gkvec_.comm().bcast(root.data(), root.size(), 0);
        check(values != root, "Dirac LAPW geometry directions differ between G+k ranks");
    }

    void
    add_projection(mdarray<std::complex<double>, 2> const& left__, mdarray<std::complex<double>, 2> const& matrix__,
                   mdarray<std::complex<double>, 2> const& right__, mdarray<std::complex<double>, 2>& out__) const
    {
        int nb = matrix__.size(0), n = size();
        mdarray<std::complex<double>, 2> work({nb, n});
        auto const one = la::constant<std::complex<double>>::one(), zero = la::constant<std::complex<double>>::zero();
        la::wrap(la::lib_t::blas)
                .gemm('N', 'N', nb, n, nb, &one, matrix__.at(memory_t::host), matrix__.ld(), right__.at(memory_t::host),
                      right__.ld(), &zero, work.at(memory_t::host), work.ld());
        la::wrap(la::lib_t::blas)
                .gemm('C', 'N', n, n, nb, &one, left__.at(memory_t::host), left__.ld(), work.at(memory_t::host),
                      work.ld(), &one, out__.at(memory_t::host), out__.ld());
    }

    mdarray<std::complex<double>, 2>
    spherical_matrix(mdarray<double, 2> const& radial__, int ia__) const
    {
        auto const& type = cell_.atom(ia__).type();
        int nb           = type.mt_basis_size();
        mdarray<std::complex<double>, 2> matrix({nb, nb});
        matrix.zero();
        for (int j = 0; j < nb; j++) {
            for (int i = 0; i < nb; i++) {
                auto const& a = type.indexb(i);
                auto const& b = type.indexb(j);
                if (a.am == b.am && a.two_mj == b.two_mj) {
                    matrix(i, j) = radial__(a.idxrf, b.idxrf);
                }
            }
        }
        return matrix;
    }

    void
    validate_response(std::vector<radial_basis_response_t> const* response__) const
    {
        int mode = response__ != nullptr, minimum = mode, maximum = mode;
        gkvec_.comm().allreduce<int, mpi::op_t::min>(&minimum, 1);
        gkvec_.comm().allreduce<int, mpi::op_t::max>(&maximum, 1);
        check(minimum != maximum ||
                      (response__ && response__->size() != static_cast<size_t>(cell_.num_atom_symmetry_classes())),
              "Dirac LAPW basis: response mode and symmetry-class count must agree across ranks");
        bool invalid{false};
        if (response__) {
            for (int ic = 0; ic < cell_.num_atom_symmetry_classes(); ic++) {
                auto const& type = cell_.atom_symmetry_class(ic).atom_type();
                auto const& d    = (*response__)[ic];
                invalid |= d.radial_functions.size(0) != static_cast<size_t>(type.num_mt_points()) ||
                           d.radial_functions.size(1) != static_cast<size_t>(type.mt_radial_basis_size()) ||
                           d.radial_functions.size(2) != 4 || d.surface_derivatives.size(0) != 6 ||
                           d.surface_derivatives.size(1) !=
                                   static_cast<size_t>(type.mt_radial_basis_size() - type.num_lo_descriptors());
                for (double value : d.radial_functions) {
                    invalid |= !std::isfinite(value);
                }
                for (double value : d.surface_derivatives) {
                    invalid |= !std::isfinite(value);
                }
            }
        }
        check(invalid, "Dirac LAPW basis: incompatible or nonfinite radial response");
    }

    mdarray<std::complex<double>, 2>
    interstitial_matrix(bool kinetic__, mdarray<double, 2> const* displacements__ = nullptr,
                        r3::matrix<double> const* strain__ = nullptr) const
    {
        int ng = gkvec_.num_gvec();
        mdarray<std::complex<double>, 2> result({size(), size()});
        result.zero();
        for (int j = 0; j < ng; j++) {
            auto qj = gkvec_.gkvec_cart(gvec_index_t::global(j));
            Dirac_plane_wave pwj(qj);
            auto uj  = pwj.spinors();
            auto dqj = strain__ ? dot(transpose(*strain__), qj) * -1.0 : r3::vector<double>{0, 0, 0};
            auto duj = strain__ ? pwj.response(dqj) : Dirac_plane_wave::spinors_t{};
            for (int i = 0; i < ng; i++) {
                auto qi = gkvec_.gkvec_cart(gvec_index_t::global(i));
                Dirac_plane_wave pwi(qi);
                auto ui  = pwi.spinors();
                auto dqi = strain__ ? dot(transpose(*strain__), qi) * -1.0 : r3::vector<double>{0, 0, 0};
                auto dui = strain__ ? pwi.response(dqi) : Dirac_plane_wave::spinors_t{};
                auto g   = qi - qj;
                std::complex<double> theta = i == j ? 1 : 0;
                std::complex<double> dtheta{0};
                for (int ia = 0; ia < cell_.num_atoms(); ia++) {
                    auto const& atom = cell_.atom(ia);
                    auto position    = cell_.get_cartesian_coordinates(atom.position());
                    auto phase       = fourpi / cell_.omega() * std::exp(std::complex<double>(0, -dot(g, position)));
                    double f         = unit_step_function_form_factors(atom.mt_radius(), g.length());
                    theta -= phase * f;
                    if (strain__) {
                        auto dg      = dqi - dqj;
                        double dl    = g.length() > 1e-12 ? dot(g, dg) / g.length() : 0;
                        double df    = unit_step_function_form_factor_derivative(atom.mt_radius(), g.length()) * dl;
                        double trace = (*strain__)(0, 0) + (*strain__)(1, 1) + (*strain__)(2, 2);
                        r3::vector<double> dr{(*displacements__)(0, ia), (*displacements__)(1, ia),
                                              (*displacements__)(2, ia)};
                        dtheta -= phase * (df - (trace + std::complex<double>(0, dot(dg, position) + dot(g, dr))) * f);
                    }
                }
                for (int sj : {0, 1}) {
                    for (int si : {0, 1}) {
                        std::complex<double> norm{0}, dnorm{0};
                        for (int c = 0; c < 4; c++) {
                            norm += std::conj(ui[si][c]) * uj[sj][c];
                            dnorm += std::conj(dui[si][c]) * uj[sj][c] + std::conj(ui[si][c]) * duj[sj][c];
                        }
                        // Symmetrized c alpha.p + (beta-I)c^2, with the rest energy subtracted.
                        double energy = kinetic__ ? 0.5 * (pwi.energy() + pwj.energy()) : 1.0;
                        double de =
                                kinetic__ && strain__ ? 0.5 * (pwi.energy_response(dqi) + pwj.energy_response(dqj)) : 0;
                        result(si * ng + i, sj * ng + j) =
                                strain__ ? (dtheta * norm + theta * dnorm) * energy + theta * norm * de
                                         : theta * norm * energy;
                    }
                }
            }
        }
        return result;
    }

    mdarray<std::complex<double>, 2>
    assemble_geometry(int part__, mdarray<double, 2> const& displacements__, r3::matrix<double> const& strain__) const
    {
        auto result = interstitial_matrix(part__ == 0, &displacements__, &strain__);
        for (int ia = 0; ia < cell_.num_atoms(); ia++) {
            auto radial = cell_.atom(ia).symmetry_class().dirac_spherical_matrices();
            auto matrix = spherical_matrix(radial[part__], ia);
            auto da     = atom_coefficients(ia, nullptr, &displacements__, &strain__);
            add_projection(da, matrix, atoms_[ia], result);
            add_projection(atoms_[ia], matrix, da, result);
        }
        return result;
    }

    mdarray<std::complex<double>, 2>
    assemble_spherical(int part__, std::vector<radial_basis_response_t> const* response__,
                       std::vector<std::vector<double>> const& direction__) const
    {
        validate_response(response__);
        int mode = !direction__.empty(), minimum = mode, maximum = mode;
        gkvec_.comm().allreduce<int, mpi::op_t::min>(&minimum, 1);
        gkvec_.comm().allreduce<int, mpi::op_t::max>(&maximum, 1);
        bool invalid =
                minimum != maximum ||
                (!direction__.empty() && (!response__ || direction__.size() != static_cast<size_t>(cell_.num_atoms())));
        if (!invalid && !direction__.empty()) {
            for (int ia = 0; ia < cell_.num_atoms(); ia++) {
                invalid |= direction__[ia].size() != static_cast<size_t>(cell_.atom(ia).num_mt_points());
                for (double value : direction__[ia]) {
                    invalid |= !std::isfinite(value);
                }
            }
        }
        check(invalid, "Dirac LAPW Hamiltonian: incompatible spherical potential direction");
        auto result =
                response__ ? mdarray<std::complex<double>, 2>({size(), size()}) : interstitial_matrix(part__ == 0);
        if (response__) {
            result.zero();
        }
        for (int ia = 0; ia < cell_.num_atoms(); ia++) {
            auto const& atom = cell_.atom(ia);
            auto radial      = atom.symmetry_class().dirac_spherical_matrices();
            auto matrix      = spherical_matrix(radial[part__], ia);
            auto const& a    = atoms_[ia];
            if (response__) {
                auto const& response = (*response__)[atom.symmetry_class().id()];
                auto da              = atom_coefficients(ia, &response);
                auto dradial         = atom.symmetry_class().dirac_spherical_matrices(
                        &response, direction__.empty() ? std::vector<double>{} : direction__[ia]);
                auto dmatrix = spherical_matrix(dradial[part__], ia);
                add_projection(da, matrix, a, result);
                add_projection(a, matrix, da, result);
                add_projection(a, dmatrix, a, result);
            } else {
                add_projection(a, matrix, a, result);
            }
        }
        return result;
    }

  public:
    Dirac_lapw_basis(Unit_cell const& cell__, fft::Gvec const& gkvec__)
        : cell_(cell__)
        , gkvec_(gkvec__)
        , matching_(cell__, gkvec__)
    {
        bool invalid = gkvec_.num_gvec() < 1 || gkvec_.reduced() || !std::isfinite(cell_.omega()) || cell_.omega() <= 0;
        for (int ia = 0; ia < cell_.num_atoms(); ia++) {
            invalid |= !cell_.atom(ia).type().indexb().full_j();
        }
        check(invalid, "Dirac LAPW basis requires full-j atoms and unreduced plane waves");
        for (int ia = 0; ia < cell_.num_atoms(); ia++) {
            atoms_.push_back(atom_coefficients(ia, nullptr));
        }
    }

    int
    size() const
    {
        return 2 * gkvec_.num_gvec() + cell_.mt_lo_basis_size();
    }

    mdarray<std::complex<double>, 2> const&
    atom_coefficients(int ia__) const
    {
        return atoms_.at(ia__);
    }

    /// Matching-coefficient derivative for radial responses in symmetry-class order; LO identities stay fixed.
    mdarray<std::complex<double>, 2>
    atom_coefficients_response(int ia__, std::vector<radial_basis_response_t> const& response__) const
    {
        validate_response(&response__);
        return atom_coefficients(ia__, &response__[cell_.atom(ia__).symmetry_class().id()]);
    }

    /// Matching derivative at fixed radial functions and MT radii. Cartesian atom motion is independent of strain.
    mdarray<std::complex<double>, 2>
    atom_coefficients_geometry_response(int ia__, mdarray<double, 2> const& displacements__,
                                        r3::matrix<double> const& strain__) const
    {
        validate_geometry(&displacements__, &strain__);
        return atom_coefficients(ia__, nullptr, &displacements__, &strain__);
    }

    /// Moving-basis bare H/O, including the interstitial mask, PW spinors and full-j matching, at fixed radial functions.
    std::array<mdarray<std::complex<double>, 2>, 2>
    geometry_response(mdarray<double, 2> const& displacements__, r3::matrix<double> const& strain__) const
    {
        validate_geometry(&displacements__, &strain__);
        return {assemble_geometry(0, displacements__, strain__), assemble_geometry(1, displacements__, strain__)};
    }

    /// Physical overlap, or its potential derivative for radial responses in symmetry-class order.
    mdarray<std::complex<double>, 2>
    overlap(std::vector<radial_basis_response_t> const* response__ = nullptr) const
    {
        return assemble_spherical(1, response__, {});
    }

    /// Symmetrized free Dirac operator in the interstitial plus the spherical atomic Hamiltonians.
    /** Both-component matching and confined LOs cancel the Schlosser-Marcus surface terms in retained angular
     *  channels. Additional scalar, magnetic and XC fields enter through field_operator(), without adding the
     *  spherical potential a second time. A radial response returns the total potential derivative, including
     *  optional physical spherical potential directions in atom order. */
    mdarray<std::complex<double>, 2>
    hamiltonian(std::vector<radial_basis_response_t> const* response__ = nullptr,
                std::vector<std::vector<double>> const& direction__    = {}) const
    {
        return assemble_spherical(0, response__, direction__);
    }

    /// Additive variational operator from interstitial Fourier covectors and full-j atom matrices.
    /** Fourier columns use canonical total/z/x/y (1, 2 or 4 columns), coupling through I_4 and beta Sigma.
     *  They already include interstitial restriction and quadrature. No extra MT mask is applied. Atom matrices
     *  include nonspherical scalar/magnetic fields and XC raw-sample contractions, but exclude the spherical
     *  potential already present in hamiltonian(). For a radial response, local_response is the corresponding
     *  matrix derivative at fixed field covectors; the interstitial contribution then has zero variation.
     *  Alternatively, displacements/strain return the moving-basis derivative at fixed Fourier and atom covectors.
     *  Changes in physical field samples, quadrature and the interstitial restriction must be added separately. */
    mdarray<std::complex<double>, 2>
    field_operator(fft::Gvec const& fields__, mdarray<std::complex<double>, 2> const& rho__,
                   mdarray<std::complex<double>, 2> const& tau__,
                   std::vector<mdarray<std::complex<double>, 2>> const& local__,
                   std::vector<radial_basis_response_t> const* response__                = nullptr,
                   std::vector<mdarray<std::complex<double>, 2>> const* local_response__ = nullptr,
                   mdarray<double, 2> const* displacements__                             = nullptr,
                   r3::matrix<double> const* strain__                                    = nullptr) const
    {
        validate_response(response__);
        validate_geometry(displacements__, strain__);
        int nc = rho__.size(1), ng = gkvec_.num_gvec();
        int minimum = nc, maximum = nc;
        gkvec_.comm().allreduce<int, mpi::op_t::min>(&minimum, 1);
        gkvec_.comm().allreduce<int, mpi::op_t::max>(&maximum, 1);
        bool invalid = fields__.num_gvec() < 1 || fields__.reduced() || minimum != maximum ||
                       (nc != 1 && nc != 2 && nc != 4) || rho__.size(0) != static_cast<size_t>(fields__.num_gvec()) ||
                       tau__.size(0) != rho__.size(0) || tau__.size(1) != rho__.size(1) || !rho__.on_host() ||
                       !tau__.on_host() || local__.size() != static_cast<size_t>(cell_.num_atoms()) ||
                       (response__ != nullptr) != (local_response__ != nullptr) || (response__ && strain__) ||
                       (local_response__ && local_response__->size() != local__.size());
        auto finite = [](auto value) { return std::isfinite(value.real()) && std::isfinite(value.imag()); };
        if (!invalid) {
            for (auto const* pw : {&rho__, &tau__}) {
                for (auto value : *pw) {
                    invalid |= !finite(value);
                }
            }
            for (int ia = 0; ia < cell_.num_atoms(); ia++) {
                for (auto const* matrix : {&local__[ia], local_response__ ? &(*local_response__)[ia] : &local__[ia]}) {
                    int nb         = cell_.atom(ia).type().mt_basis_size();
                    bool bad_shape = matrix->size(0) != static_cast<size_t>(nb) ||
                                     matrix->size(1) != static_cast<size_t>(nb) || !matrix->on_host();
                    invalid |= bad_shape;
                    if (!bad_shape) {
                        for (int j = 0; j < nb; j++) {
                            for (int i = 0; i < nb; i++) {
                                auto value = (*matrix)(i, j);
                                invalid |= !finite(value) ||
                                           std::abs(value - std::conj((*matrix)(j, i))) > 1e-11 * (1 + std::abs(value));
                            }
                        }
                    }
                }
            }
        }
        check(invalid, "Dirac LAPW field operator: incompatible, nonfinite or non-Hermitian fields");
        auto lower = fields__.gvec(gvec_index_t::global(0)), upper = lower;
        for (int ig = 1; ig < fields__.num_gvec(); ig++) {
            auto g = fields__.gvec(gvec_index_t::global(ig));
            for (int x = 0; x < 3; x++) {
                lower[x] = std::min(lower[x], g[x]);
                upper[x] = std::max(upper[x], g[x]);
            }
        }
        mdarray<int, 2> indices({ng, ng});
        for (int j = 0; j < ng; j++) {
            for (int i = 0; i < ng; i++) {
                auto g       = gkvec_.gvec(gvec_index_t::global(i)) - gkvec_.gvec(gvec_index_t::global(j));
                bool outside = false;
                for (int x = 0; x < 3; x++) {
                    outside |= g[x] < lower[x] || g[x] > upper[x];
                }
                int ig        = outside ? -1 : fields__.index_by_gvec(g);
                indices(i, j) = ig;
                invalid |= ig < 0;
                if (ig >= 0) {
                    auto dq = gkvec_.gkvec_cart(gvec_index_t::global(i)) - gkvec_.gkvec_cart(gvec_index_t::global(j));
                    invalid |= (fields__.gvec_cart(gvec_index_t::global(ig)) - dq).length() > 1e-10;
                }
            }
        }
        check(invalid, "Dirac LAPW field operator: Fourier grid must cover all PW differences in the same cell");
        for (int j = 0; j < ng; j++) {
            for (int i = 0; i < ng; i++) {
                for (int s = 0; s < nc; s++) {
                    for (auto const* pw : {&rho__, &tau__}) {
                        auto value = (*pw)(indices(i, j), s);
                        invalid |= std::abs(value - std::conj((*pw)(indices(j, i), s))) > 1e-11 * (1 + std::abs(value));
                    }
                }
            }
        }
        check(invalid, "Dirac LAPW field operator: Fourier covectors must represent real fields");
        mdarray<std::complex<double>, 2> result({size(), size()});
        result.zero();
        if (!response__) {
            std::complex<double> imaginary(0, 1);
            auto components = [&](auto const& left, auto const& right) {
                std::array<std::complex<double>, 4> result{};
                for (int block : {0, 1}) {
                    auto up = std::conj(left[2 * block]), down = std::conj(left[2 * block + 1]);
                    auto vup = right[2 * block], vdown = right[2 * block + 1];
                    double sign = block ? -1.0 : 1.0;
                    result[0] += up * vup + down * vdown;
                    result[1] += sign * (up * vup - down * vdown);
                    result[2] += sign * (up * vdown + down * vup);
                    result[3] += sign * imaginary * (down * vup - up * vdown);
                }
                return result;
            };
            for (int j = 0; j < ng; j++) {
                auto qj  = gkvec_.gkvec_cart(gvec_index_t::global(j));
                auto uj  = Dirac_plane_wave(qj).spinors();
                auto dqj = strain__ ? dot(transpose(*strain__), qj) * -1.0 : r3::vector<double>{0, 0, 0};
                auto duj = strain__ ? Dirac_plane_wave(qj).response(dqj) : Dirac_plane_wave::spinors_t{};
                for (int i = 0; i < ng; i++) {
                    auto qi  = gkvec_.gkvec_cart(gvec_index_t::global(i));
                    auto ui  = Dirac_plane_wave(qi).spinors();
                    auto dqi = strain__ ? dot(transpose(*strain__), qi) * -1.0 : r3::vector<double>{0, 0, 0};
                    auto dui = strain__ ? Dirac_plane_wave(qi).response(dqi) : Dirac_plane_wave::spinors_t{};
                    int ig   = indices(i, j);
                    for (int sj : {0, 1}) {
                        for (int si : {0, 1}) {
                            auto base = components(ui[si], uj[sj]);
                            auto left = components(dui[si], uj[sj]), right = components(ui[si], duj[sj]);
                            double kinetic = 0.5 * dot(qi, qj), dkinetic = 0.5 * (dot(dqi, qj) + dot(qi, dqj));
                            for (int s = 0; s < nc; s++) {
                                auto coefficient = rho__(ig, s) + kinetic * tau__(ig, s);
                                result(si * ng + i, sj * ng + j) += strain__ ? coefficient * (left[s] + right[s]) +
                                                                                       dkinetic * tau__(ig, s) * base[s]
                                                                             : coefficient * base[s];
                            }
                        }
                    }
                }
            }
        }
        for (int ia = 0; ia < cell_.num_atoms(); ia++) {
            auto const& a = atoms_[ia];
            if (response__ || strain__) {
                auto da = atom_coefficients(ia,
                                            response__ ? &(*response__)[cell_.atom(ia).symmetry_class().id()] : nullptr,
                                            displacements__, strain__);
                add_projection(da, local__[ia], a, result);
                add_projection(a, local__[ia], da, result);
                if (local_response__) {
                    add_projection(a, (*local_response__)[ia], a, result);
                }
            } else {
                add_projection(a, local__[ia], a, result);
            }
        }
        return result;
    }
};

} // namespace sirius

#endif
