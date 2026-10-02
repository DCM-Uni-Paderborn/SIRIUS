/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "lapw_response.hpp"
#include "potential/potential.hpp"
#include "mixer/anderson_stable_mixer.hpp"

namespace sirius {
namespace {

template <typename Operation>
void
fields(Density& x, Operation&& operation)
{
    for (int s = 0; s < x.ctx().num_spins(); s++) {
        operation(x.component(s));
        operation(x.kinetic_density().component(s));
    }
}

void
synchronize(Density& x)
{
    fields(x, [&](auto& f) {
        f.rg().fft_transform(-1);
        f.mt().sync(x.ctx().unit_cell().spl_num_atoms());
    });
}

void
add(double alpha, Density const& x, Density& y)
{
    for (int s = 0; s < x.ctx().num_spins(); s++) {
        axpy(alpha, x.component(s), y.component(s));
        axpy(alpha, x.kinetic_density().component(s), y.kinetic_density().component(s));
    }
}

double
field_residual(Density const& rhs, Density const& x, Density const& y)
{
    auto const& ctx = x.ctx();
    double result{0};
    for (int part : {0, 1}) {
        for (int s = 0; s < ctx.num_spins(); s++) {
            auto const& r = part == 0 ? rhs.component(s) : rhs.kinetic_density().component(s);
            auto const& a = part == 0 ? x.component(s) : x.kinetic_density().component(s);
            auto const& b = part == 0 ? y.component(s) : y.kinetic_density().component(s);
            std::array<double, 2> norm{};
            auto sample = [&](double ri, double ai, double bi) {
                if (!std::isfinite(ri) || !std::isfinite(ai) || !std::isfinite(bi)) {
                    norm[0] = std::numeric_limits<double>::infinity();
                } else {
                    norm[0] = std::max(norm[0], std::abs(bi - ai));
                    norm[1] = std::max({norm[1], std::abs(ri), std::abs(ai), std::abs(bi)});
                }
            };
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                sample(r.rg().value(ir), a.rg().value(ir), b.rg().value(ir));
            }
            for (auto atom : ctx.unit_cell().spl_num_atoms()) {
                for (size_t i = 0; i < a.mt()[atom.i].size(); i++) {
                    sample(r.mt()[atom.i][i], a.mt()[atom.i][i], b.mt()[atom.i][i]);
                }
            }
            ctx.comm().allreduce<double, mpi::op_t::max>(norm.data(), norm.size());
            result = std::max(result, norm[0] / std::max(1e-12, norm[1]));
        }
    }
    return result;
}

} // namespace

LAPW_response::LAPW_response(K_point_set& ks__, Density const& density__, Potential& potential__)
    : ks_(ks__)
    , density_(density__)
    , potential_(potential__)
{
    auto& ctx   = ks_.ctx();
    int invalid = !ctx.full_potential() || &density_.ctx() != &ctx || &potential_.ctx() != &ctx ||
                  !density_.has_kinetic_density() || !ctx.cb().external_xc_response_ ||
                  (ctx.unit_cell().num_core_electrons() != 0 && ctx.core_relativity() != relativity_t::none);
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW response requires matching physical fields and nonrelativistic core XC response");
    }
    h0_ = std::make_unique<Hamiltonian0<double>>(potential_, false);
    h0_->generate_pw_coefs(potential_);
    spectrum_ = std::make_unique<LAPW_density_response>(*h0_, ks_);
    matrices_ = spectrum_->density_matrices();
}

std::unique_ptr<Density>
LAPW_response::apply(Density const& direction__) const
{
    auto& ctx   = ks_.ctx();
    int invalid = &direction__.ctx() != &ctx || !direction__.has_kinetic_density();
    if (!invalid) {
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (auto const* f : {&direction__.component(s), &direction__.kinetic_density().component(s)}) {
                for (auto z : f->rg().f_pw_local()) {
                    invalid |= !std::isfinite(z.real()) || !std::isfinite(z.imag());
                }
                for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                    invalid |= !std::isfinite(f->rg().value(ir));
                }
                for (auto atom : ctx.unit_cell().spl_num_atoms()) {
                    for (size_t i = 0; i < f->mt()[atom.i].size(); i++) {
                        invalid |= !std::isfinite(f->mt()[atom.i][i]);
                    }
                }
            }
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW response requires finite physical rho/tau directions from the same context");
    }
    auto xc      = potential_.lapw_external_xc_response(density_, direction__);
    auto hartree = potential_.hartree_response(direction__.rho());
    std::vector<std::vector<double>> scalar;
    std::vector<std::array<std::vector<double>, 2>> core;
    potential_.lapw_radial_potential_response(*hartree, xc, scalar, core);
    std::vector<radial_basis_response_t> radial;
    try {
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            radial.push_back(
                    ctx.unit_cell().atom_symmetry_class(ic).radial_basis_potential_response(scalar[ic], core[ic]));
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW radial response failed");
    }
    auto atoms     = h0_->lapw_radial_basis_response(radial);
    auto field     = h0_->lapw_field_response(*hartree, xc);
    using matrix_t = la::dmatrix<std::complex<double>>;
    std::vector<std::array<std::unique_ptr<matrix_t>, 2>> dh, ds;
    std::vector<LAPW_density_response::matrix_direction_t> directions;
    for (auto it : ks_.spl_num_kpoints()) {
        auto& kp = *ks_.get<double>(it.i);
        auto hk  = (*h0_)(kp);
        int n = kp.gklo_basis_size(), bs = ctx.cyclic_block_size();
        dh.emplace_back();
        ds.emplace_back();
        directions.emplace_back();
        try {
            matrix_t field_h(n, n, ctx.blacs_grid(), bs, bs);
            for (int s = 0; s < ctx.num_spins(); s++) {
                dh.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                ds.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                hk.set_fv_h_o_radial_response(s, radial, atoms, *dh.back()[s], *ds.back()[s]);
                hk.set_fv_h_field_response(s, field, field_h);
                for (int j = 0; j < field_h.num_cols_local(); j++) {
                    for (int i = 0; i < field_h.num_rows_local(); i++) {
                        (*dh.back()[s])(i, j) += field_h(i, j);
                    }
                }
                directions.back().h[s] = dh.back()[s].get();
                directions.back().s[s] = ds.back()[s].get();
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW Hamiltonian response failed");
    }
    auto dp     = spectrum_->density_matrix_response(directions);
    auto result = density_.lapw_density_matrix_fields(ks_, dp);
    auto basis  = density_.lapw_density_matrix_radial_response(ks_, matrices_, radial);
    add(1.0, *basis, *result);
    synchronize(*result);
    return result;
}

LAPW_response::result_t
LAPW_response::solve(Density const& rhs__, double tolerance__, int max_iterations__, int history__, double beta__) const
{
    auto& ctx = ks_.ctx();
    std::array<double, 4> controls{tolerance__, static_cast<double>(max_iterations__), static_cast<double>(history__),
                                   beta__};
    auto root = controls;
    ctx.comm().bcast(root.data(), root.size(), 0);
    int invalid = controls != root || !std::isfinite(tolerance__) || tolerance__ <= 0 || max_iterations__ < 1 ||
                  history__ < 2 || !std::isfinite(beta__) || beta__ <= 0 || beta__ > 1 || &rhs__.ctx() != &ctx ||
                  !rhs__.has_kinetic_density();
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW response solve requires common positive controls and matching rho/tau fields");
    }
    auto x = std::make_unique<Density>(ctx, true);
    x->zero();
    x->kinetic_density().zero();
    mixer::FunctionProperties<Density> properties(
            [&](Density const& a, Density const& b) {
                // A positive norm avoids Fourier sphere-mask ringing in the response mixer.
                double rg{0}, mt{0};
                for (int s = 0; s < ctx.num_spins(); s++) {
                    for (int part : {0, 1}) {
                        auto const& u = part == 0 ? a.component(s) : a.kinetic_density().component(s);
                        auto const& v = part == 0 ? b.component(s) : b.kinetic_density().component(s);
                        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                            rg += u.rg().value(ir) * v.rg().value(ir);
                        }
                        for (auto atom : ctx.unit_cell().spl_num_atoms()) {
                            for (size_t i = 0; i < u.mt()[atom.i].size(); i++) {
                                mt += u.mt()[atom.i][i] * v.mt()[atom.i][i];
                            }
                        }
                    }
                }
                double result = rg / (ctx.comm().size() / ctx.comm_fft().size()) + mt;
                ctx.comm().allreduce(&result, 1);
                return result;
            },
            [](double alpha, Density& a) { fields(a, [&](auto& f) { f *= alpha; }); },
            [](Density const& a, Density& b) { copy(a, b); },
            [](double alpha, Density const& a, Density& b) { add(alpha, a, b); },
            [](double c, double s, Density& a, Density& b) {
                for (int j = 0; j < a.ctx().num_spins(); j++) {
                    rotate(c, s, a.component(j), b.component(j));
                    rotate(c, s, a.kinetic_density().component(j), b.kinetic_density().component(j));
                }
            });
    mixer::Anderson_stable<Density> mixer(history__, beta__);
    mixer.initialize_function<0>(properties, *x, ctx, true);
    for (int iteration = 0; iteration < max_iterations__; iteration++) {
        auto y = apply(*x);
        add(1.0, rhs__, *y);
        double residual = field_residual(rhs__, *x, *y);
        if (!std::isfinite(residual)) {
            RTE_THROW("nonfinite coupled LAPW response residual");
        }
        if (residual <= tolerance__) {
            return {std::move(x), iteration + 1, residual};
        }
        mixer.set_input<0>(*y);
        mixer.mix(0);
        mixer.get_output<0>(*x);
        synchronize(*x);
    }
    RTE_THROW("coupled LAPW response did not reach the requested residual tolerance");
    return {};
}

} // namespace sirius
