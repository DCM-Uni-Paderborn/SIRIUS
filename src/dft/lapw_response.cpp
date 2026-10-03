/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "lapw_response.hpp"
#include "potential/potential.hpp"
#include "mixer/anderson_stable_mixer.hpp"
#include "lapw/step_function.hpp"

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
LAPW_response::apply(Density const& direction__, double* entropy__) const
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
    return potential_response(*hartree, xc, nullptr, entropy__);
}

std::unique_ptr<Density>
LAPW_response::geometry_rhs(xc_geometry_direction_t const& direction__, double* entropy__) const
{
    auto xc      = potential_.lapw_external_xc_geometry_response(density_, direction__);
    auto hartree = potential_.hartree_geometry_response(density_.rho(), direction__);
    return potential_response(*hartree, xc, &direction__, entropy__);
}

LAPW_response::radial_geometry_response_t
LAPW_response::radial_geometry_response(xc_geometry_direction_t const& direction__, double tolerance__,
                                        int max_iterations__) const
{
    double explicit_entropy{0};
    auto rhs    = geometry_rhs(direction__, &explicit_entropy);
    auto solved = solve(*rhs, tolerance__, max_iterations__);
    radial_geometry_response_t result, explicit_response;
    potential_.lapw_radial_potential_response(density_, *solved.fields, result.scalar, result.core);
    auto hartree = potential_.hartree_geometry_response(density_.rho(), direction__);
    auto xc      = potential_.lapw_external_xc_geometry_response(density_, direction__);
    potential_.lapw_radial_potential_response(*hartree, xc, explicit_response.scalar, explicit_response.core);
    for (size_t ic = 0; ic < result.scalar.size(); ic++) {
        for (size_t ir = 0; ir < result.scalar[ic].size(); ir++) {
            result.scalar[ic][ir] += explicit_response.scalar[ic][ir];
            for (int part = 0; part < 2; part++) {
                result.core[ic][part][ir] += explicit_response.core[ic][part][ir];
            }
        }
    }
    result.fields  = std::move(solved.fields);
    result.entropy = explicit_entropy + solved.entropy;
    return result;
}

double
LAPW_response::energy_geometry_derivative(xc_geometry_direction_t const& direction__,
                                          radial_geometry_response_t const& response__) const
{
    auto& ctx   = ks_.ctx();
    int invalid = !response__.fields || !std::isfinite(response__.entropy);
    if (response__.fields) {
        invalid |= &response__.fields->ctx() != &ctx || !response__.fields->has_kinetic_density();
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW energy geometry derivative requires a matching complete SCF field response");
    }
    auto const& fields = *response__.fields;
    auto potential     = potential_.hartree_response(fields.rho());
    auto geometry      = potential_.hartree_geometry_response(density_.rho(), direction__);
    axpy(1.0, *geometry, *potential);
    r3::matrix<double> lattice;
    mdarray<double, 2> atoms({3, ctx.unit_cell().num_atoms()});
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            lattice(x, y) = direction__.lattice[x + 3 * y];
        }
    }
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            atoms(x, ia) = direction__.atoms[ia][x];
        }
    }
    auto strain  = dot(lattice, ctx.unit_cell().inverse_lattice_vectors());
    double trace = strain(0, 0) + strain(1, 1) + strain(2, 2);
    auto theta   = step_function_geometry_response(ctx.unit_cell(), ctx.gvec(), atoms, strain);
    Smooth_periodic_function<double> measure(ctx.spfft<double>(), ctx.gvec_fft_sptr());
    for (int ig = 0; ig < ctx.gvec().count(); ig++) {
        measure.f_pw_local(ig) = theta[ctx.gvec().offset() + ig];
    }
    measure.fft_transform(1);
    for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
        measure.value(ir) += trace * ctx.theta(ir);
    }
    double result = fields.kinetic_density().scalar().integrate().total +
                    inner(density_.kinetic_density().scalar().rg(), measure) + response__.entropy;
    result += 0.5 * (inner(fields.rho(), potential_.hartree_potential()) + inner(density_.rho(), *potential) +
                     inner(density_.rho().rg(), potential_.hartree_potential().rg(),
                           [&](int ir) { return measure.value(ir); }));
    double nuclei{0};
    for (auto it : ctx.unit_cell().spl_num_atoms()) {
        nuclei -= 0.5 * ctx.unit_cell().atom(it.i).zn() * y00 * potential->mt()[it.i](0, 0);
    }
    ctx.comm().allreduce(&nuclei, 1);
    result += nuclei;
    auto xc = potential_.lapw_xc_contractions(fields);
    result += xc.rho + xc.magnetic + xc.tau;
    auto explicit_xc = potential_.external_xc_geometry(density_);
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            result += direction__.atoms[ia][x] * explicit_xc.atoms[ia][x];
        }
    }
    for (int i = 0; i < 9; i++) {
        result += direction__.lattice[i] * explicit_xc.lattice[i];
    }
    if (!std::isfinite(result)) {
        RTE_THROW("nonfinite LAPW free-energy geometry derivative");
    }
    return result;
}

std::unique_ptr<Density>
LAPW_response::potential_response(Periodic_function<double> const& hartree__, lapw_xc_field_adjoint_t const& xc__,
                                  xc_geometry_direction_t const* geometry__, double* entropy__) const
{
    auto& ctx = ks_.ctx();
    int invalid{0};
    std::vector<std::vector<double>> scalar;
    std::vector<std::array<std::vector<double>, 2>> core;
    potential_.lapw_radial_potential_response(hartree__, xc__, scalar, core);
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
    auto atoms = h0_->lapw_radial_basis_response(radial);
    auto field = h0_->lapw_field_response(hartree__, xc__);
    if (geometry__) {
        r3::matrix<double> lattice;
        mdarray<double, 2> atoms({3, ctx.unit_cell().num_atoms()});
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                lattice(x, y) = geometry__->lattice[x + 3 * y];
            }
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int x = 0; x < 3; x++) {
                atoms(x, ia) = geometry__->atoms[ia][x];
            }
        }
        auto strain = dot(lattice, ctx.unit_cell().inverse_lattice_vectors());
        auto theta  = step_function_geometry_response(ctx.unit_cell(), ctx.gvec(), atoms, strain);
        Smooth_periodic_function<double> boundary(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            boundary.f_pw_local(ig) = theta[ctx.gvec().offset() + ig];
        }
        boundary.fft_transform(1);
        // XC already differentiates its restriction; Hartree still needs the moving mask.
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            boundary.value(ir) *= potential_.hartree_potential().rg().value(ir);
        }
        boundary.fft_transform(-1);
        auto pw = boundary.gather_f_pw();
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
                field.scalar_pw(ig, s) += pw[ig];
            }
        }
    }
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
            std::unique_ptr<matrix_t> geometry_h, geometry_s;
            if (geometry__) {
                geometry_h = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                geometry_s = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
            }
            for (int s = 0; s < ctx.num_spins(); s++) {
                dh.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                ds.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                hk.set_fv_h_o_radial_response(s, radial, atoms, *dh.back()[s], *ds.back()[s]);
                hk.set_fv_h_field_response(s, field, field_h);
                if (geometry__) {
                    hk.set_fv_h_o_geometry_response(s, *geometry__, *geometry_h, *geometry_s);
                }
                for (int j = 0; j < field_h.num_cols_local(); j++) {
                    for (int i = 0; i < field_h.num_rows_local(); i++) {
                        (*dh.back()[s])(i, j) += field_h(i, j);
                        if (geometry__) {
                            (*dh.back()[s])(i, j) += (*geometry_h)(i, j);
                            (*ds.back()[s])(i, j) += (*geometry_s)(i, j);
                        }
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
    auto dp     = spectrum_->density_matrix_response(directions, entropy__);
    auto result = density_.lapw_density_matrix_fields(ks_, dp);
    auto basis  = density_.lapw_density_matrix_radial_response(ks_, matrices_, radial);
    add(1.0, *basis, *result);
    if (geometry__) {
        auto moved = density_.lapw_density_matrix_geometry_response(ks_, matrices_, *geometry__);
        add(1.0, *moved, *result);
    }
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
        double entropy{0};
        auto y = apply(*x, &entropy);
        add(1.0, rhs__, *y);
        double residual = field_residual(rhs__, *x, *y);
        if (!std::isfinite(residual)) {
            RTE_THROW("nonfinite coupled LAPW response residual");
        }
        if (residual <= tolerance__) {
            return {std::move(x), iteration + 1, residual, entropy};
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
