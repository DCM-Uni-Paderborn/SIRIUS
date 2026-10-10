/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_POINT_SYMMETRY_HPP__
#define __XC_POINT_SYMMETRY_HPP__

#include "context/simulation_context.hpp"
#include "function3d/xc_point_fields.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include <map>

namespace sirius {

/// Group projector and its discrete transpose on a symmetry-closed atom quadrature.
/** Complete blocks may reside on different ranks. Both weight arrays must be
 *  preserved. Scalar, collinear and total/z/x/y fields use the same point
 *  permutations, with Cartesian gradient and magnetic-spin rotations.
 *  The real-space quadrature is not reduced or replaced by orbit weights.
 */
class XC_point_symmetry
{
    Simulation_context const& ctx_;
    std::vector<int> counts_, offsets_;
    std::vector<std::vector<int>> forward_, inverse_;
    int nr_, ns_, total_;

    std::vector<double>
    gather(std::vector<double> const& local, int stride) const
    {
        auto counts = counts_, offsets = offsets_;
        for (size_t i = 0; i < counts.size(); i++) {
            counts[i] *= stride;
            offsets[i] *= stride;
        }
        std::vector<double> global(total_ * stride);
        ctx_.comm().allgather(local.data(), local.size(), global.data(), counts.data(), offsets.data());
        return global;
    }

    double
    spin(int g, int s, int t, bool adjoint) const
    {
        if (adjoint) {
            std::swap(s, t);
        }
        auto const& S = ctx_.unit_cell().symmetry()[g].spin_rotation;
        if (ns_ == 1) {
            return 1;
        }
        if (ns_ == 2) {
            return 0.5 * (1 + (s == t ? 1 : -1) * S(2, 2));
        }
        static constexpr int xyz[]{2, 0, 1};
        return s == 0 || t == 0 ? double(s == t) : S(xyz[s - 1], xyz[t - 1]);
    }

    std::vector<double>
    gather(xc_point_fields_t const& f) const
    {
        std::vector<double> local(nr_ * ns_ * 5);
        for (int p = 0; p < nr_; p++) {
            for (int s = 0; s < ns_; s++) {
                int k        = 5 * (s + ns_ * p);
                local[k]     = f.rho(p, s);
                local[k + 1] = f.tau(p, s);
                for (int x = 0; x < 3; x++) {
                    local[k + 2 + x] = f.gradient(p, x, s);
                }
            }
        }
        return gather(local, ns_ * 5);
    }

    r3::matrix<double>
    rotation(int g) const
    {
        auto const& A = ctx_.unit_cell().lattice_vectors();
        return r3::dot(A, r3::dot(ctx_.unit_cell().symmetry()[g].spg_op.R, r3::inverse(A)));
    }

    r3::matrix<double>
    rotation_direction(int g, r3::matrix<double> const& dA) const
    {
        auto const& op = ctx_.unit_cell().symmetry()[g].spg_op;
        auto inv       = r3::inverse(ctx_.unit_cell().lattice_vectors());
        return r3::dot(dA, r3::dot(op.R, inv)) + r3::dot(rotation(g), r3::dot(dA, inv)) * -1.0;
    }

  public:
    XC_point_symmetry(Simulation_context const& ctx, xc_point_set_t const& q)
        : ctx_(ctx)
        , counts_(ctx.comm().size())
        , offsets_(ctx.comm().size())
        , nr_(q.coordinates.size())
        , ns_(ctx.num_mag_dims() + 1)
        , total_(0)
    {
        q.validate(ctx);
        counts_[ctx.comm().rank()] = nr_;
        ctx.comm().allreduce(counts_.data(), counts_.size());
        for (size_t i = 0; i < counts_.size(); i++) {
            offsets_[i] = total_;
            if (counts_[i] > std::numeric_limits<int>::max() / std::max(8, 5 * ns_) - total_) {
                RTE_THROW("XC point symmetry exceeds the MPI count range");
            }
            total_ += counts_[i];
        }
        std::vector<double> local(nr_ * 8), starts(nr_);
        int offset{0};
        for (size_t b = 0; b < q.block_sizes.size(); b++) {
            starts[offset] = 1;
            for (int j = 0; j < q.block_sizes[b]; j++, offset++) {
                for (int x = 0; x < 3; x++) {
                    local[8 * offset + x]     = q.coordinates[offset][x] - q.block_centres[b][x];
                    local[8 * offset + 3 + x] = q.block_centres[b][x];
                }
                local[8 * offset + 6] = q.energy_weights[offset];
                local[8 * offset + 7] = q.descriptor_weights[offset];
            }
        }
        auto rows       = gather(local, 8);
        auto boundaries = gather(starts, 1);
        using Key       = std::array<long long, 3>;
        struct Block
        {
            r3::vector<double> centre;
            std::map<Key, std::vector<int>> bins;
        };
        std::vector<Block> blocks;
        auto key = [](r3::vector<double> const& r) {
            return Key{std::llround(r[0] * 1e6), std::llround(r[1] * 1e6), std::llround(r[2] * 1e6)};
        };
        auto vector = [&](int p, int part) {
            return r3::vector<double>(rows[8 * p + part], rows[8 * p + part + 1], rows[8 * p + part + 2]);
        };
        for (int p = 0; p < total_; p++) {
            auto c = vector(p, 3);
            if (boundaries[p]) {
                blocks.push_back({c, {}});
            }
            blocks.back().bins[key(vector(p, 0))].push_back(p);
        }
        auto Ainv       = r3::inverse(ctx.unit_cell().lattice_vectors());
        auto const& sym = ctx.unit_cell().symmetry();
        for (int g = 0; g < sym.size(); g++) {
            std::vector<int> permutation(total_, -1), transpose(total_, -1);
            for (auto const& source : blocks) {
                auto target_centre = r3::dot(sym[g].spg_op.R, r3::dot(Ainv, source.centre)) + sym[g].spg_op.t;
                Block const* target{nullptr};
                for (auto const& b : blocks) {
                    auto delta = target_centre - r3::dot(Ainv, b.centre);
                    for (int x = 0; x < 3; x++) {
                        delta[x] -= std::round(delta[x]);
                    }
                    if (delta.length() < 1e-9) {
                        if (target) {
                            RTE_THROW("XC symmetry requires unique atom blocks modulo lattice translations");
                        }
                        target = &b;
                    }
                }
                if (!target) {
                    RTE_THROW("XC atom blocks are not closed under the crystal symmetry");
                }
                for (auto const& [unused, points] : source.bins) {
                    for (int p : points) {
                        auto r   = r3::dot(sym[g].spg_op.Rc, vector(p, 0));
                        auto bin = key(r);
                        int match{-1};
                        for (int z = -1; z <= 1; z++) {
                            for (int y = -1; y <= 1; y++) {
                                for (int x = -1; x <= 1; x++) {
                                    auto it = target->bins.find(Key{bin[0] + x, bin[1] + y, bin[2] + z});
                                    if (it == target->bins.end()) {
                                        continue;
                                    }
                                    for (int j : it->second) {
                                        if ((r - vector(j, 0)).length() < 1e-9 * std::max(1.0, r.length())) {
                                            bool weights{true};
                                            for (int w : {6, 7}) {
                                                double a = rows[8 * p + w], b = rows[8 * j + w];
                                                weights &= std::abs(a - b) <=
                                                           1e-10 * std::max({1e-14, std::abs(a), std::abs(b)});
                                            }
                                            if (weights && transpose[j] < 0) {
                                                match = j;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (match < 0) {
                            RTE_THROW("XC points and both weight arrays must be closed under the crystal symmetry");
                        }
                        permutation[p]   = match;
                        transpose[match] = p;
                    }
                }
            }
            int start = offsets_[ctx.comm().rank()];
            forward_.emplace_back(permutation.begin() + start, permutation.begin() + start + nr_);
            inverse_.emplace_back(transpose.begin() + start, transpose.begin() + start + nr_);
        }
    }

    xc_point_fields_t
    apply(xc_point_fields_t const& f, bool adjoint = false, r3::matrix<double> const* dA = nullptr) const
    {
        auto input = gather(f);
        xc_point_fields_t result(nr_, ns_);
        result.rho.zero();
        result.tau.zero();
        result.gradient.zero();
        auto const& sym = ctx_.unit_cell().symmetry();
        for (int g = 0; g < sym.size(); g++) {
            auto R = dA ? rotation_direction(g, *dA) : rotation(g);
            for (int p = 0; p < nr_; p++) {
                int j = adjoint ? forward_[g][p] : inverse_[g][p];
                for (int s = 0; s < ns_; s++) {
                    for (int t = 0; t < ns_; t++) {
                        double w = spin(g, s, t, adjoint) / sym.size();
                        int k    = 5 * (t + ns_ * j);
                        if (!dA) {
                            result.rho(p, s) += w * input[k];
                            result.tau(p, s) += w * input[k + 1];
                        }
                        for (int x = 0; x < 3; x++) {
                            for (int y = 0; y < 3; y++) {
                                result.gradient(p, x, s) += w * (adjoint ? R(y, x) : R(x, y)) * input[k + 2 + y];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    r3::matrix<double>
    geometry_adjoint(xc_point_fields_t const& fields, xc_point_fields_t const& covectors) const
    {
        r3::matrix<double> result;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                r3::matrix<double> direction;
                direction(x, y) = 1;
                auto delta      = apply(fields, false, &direction);
                for (size_t i = 0; i < delta.gradient.size(); i++) {
                    result(x, y) += delta.gradient[i] * covectors.gradient[i];
                }
            }
        }
        ctx_.comm().allreduce(&result(0, 0), 9);
        return result;
    }
};

} // namespace sirius
#endif
