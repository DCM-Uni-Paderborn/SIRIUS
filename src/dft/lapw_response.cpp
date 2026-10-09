/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "lapw_response.hpp"
#include "potential/potential.hpp"
#include "mixer/anderson_stable_mixer.hpp"
#include "lapw/step_function.hpp"
#include <sstream>

namespace sirius {
namespace {

int
components(Simulation_context const& ctx)
{
    return ctx.num_mag_dims() == 3 ? 4 : ctx.num_spins();
}

template <typename Operation>
void
fields(Density& x, Operation&& operation)
{
    for (int s = 0; s < components(x.ctx()); s++) {
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
    for (int s = 0; s < components(x.ctx()); s++) {
        axpy(alpha, x.component(s), y.component(s));
        axpy(alpha, x.kinetic_density().component(s), y.kinetic_density().component(s));
    }
}

double
radial_norm_factor(Simulation_context const& ctx, Atom_type const& type, int part, int ir)
{
    if (ctx.valence_relativity() == relativity_t::none &&
        (type.num_core_electrons() == 0 || ctx.core_relativity() == relativity_t::none)) {
        return 1;
    }
    auto const& grid = type.radial_grid();
    double gamma     = std::sqrt(1 - std::pow(type.zn() / speed_of_light, 2));
    return std::pow(grid[ir] / grid.last(), (part == 0 ? 2 : 4) - 2 * gamma);
}

double
field_residual(Density const& rhs, Density const& x, Density const& y, std::array<std::array<double, 4>, 8>& norms)
{
    auto const& ctx = x.ctx();
    double result{0};
    for (int part : {0, 1}) {
        std::array<double, 2> error{}, scale{1e-12, 1e-12};
        for (int s = 0; s < components(ctx); s++) {
            auto const& r = part == 0 ? rhs.component(s) : rhs.kinetic_density().component(s);
            auto const& a = part == 0 ? x.component(s) : x.kinetic_density().component(s);
            auto const& b = part == 0 ? y.component(s) : y.kinetic_density().component(s);
            std::array<double, 4> norm{};
            auto sample = [&](double ri, double ai, double bi, double factor) {
                if (!std::isfinite(ri) || !std::isfinite(ai) || !std::isfinite(bi) || !std::isfinite(factor)) {
                    norm[0] = norm[2] = std::numeric_limits<double>::infinity();
                } else {
                    double e = std::abs(bi - ai), s = std::max({std::abs(ri), std::abs(ai), std::abs(bi)});
                    norm[0] = std::max(norm[0], e);
                    norm[1] = std::max(norm[1], s);
                    norm[2] = std::max(norm[2], factor * e);
                    norm[3] = std::max(norm[3], factor * s);
                }
            };
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                sample(r.rg().value(ir), a.rg().value(ir), b.rg().value(ir), 1);
            }
            for (auto atom : ctx.unit_cell().spl_num_atoms()) {
                auto const& type = ctx.unit_cell().atom(atom.i).type();
                for (int ir = 0; ir < type.num_mt_points(); ir++) {
                    double factor = radial_norm_factor(ctx, type, part, ir);
                    for (int lm = 0; lm < a.mt()[atom.i].angular_domain_size(); lm++) {
                        sample(r.mt()[atom.i](lm, ir), a.mt()[atom.i](lm, ir), b.mt()[atom.i](lm, ir), factor);
                    }
                }
            }
            ctx.comm().allreduce<double, mpi::op_t::max>(norm.data(), norm.size());
            norms[4 * part + s] = norm;
            for (int metric = 0; metric < 2; metric++) {
                error[metric] = std::max(error[metric], norm[2 * metric]);
                scale[metric] = std::max(scale[metric], norm[2 * metric + 1]);
            }
        }
        // Spin components share units; a vanishing magnetization must not set a sub-roundoff target.
        // The raw Coulomb peak must not hide unresolved response away from the nucleus.
        for (int metric = 0; metric < 2; metric++) {
            result = std::max(result, error[metric] / scale[metric]);
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
                  (ctx.unit_cell().num_core_electrons() != 0 && ctx.core_relativity() != relativity_t::none &&
                   ctx.core_relativity() != relativity_t::zora && ctx.core_relativity() != relativity_t::dirac);
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW response requires matching fields and NR, ZORA or Dirac core XC response");
    }
    h0_ = std::make_unique<Hamiltonian0<double>>(potential_, false);
    if (ctx.valence_relativity() != relativity_t::dirac) {
        h0_->generate_pw_coefs(potential_);
    }
    spectrum_ = std::make_unique<LAPW_density_response>(*h0_, ks_);
    matrices_ = spectrum_->density_matrices();
}

std::unique_ptr<Density>
LAPW_response::apply(Density const& direction__, double* entropy__, double* spin_orbit__, double* eval_sum__) const
{
    auto& ctx   = ks_.ctx();
    int invalid = &direction__.ctx() != &ctx || !direction__.has_kinetic_density();
    if (!invalid) {
        for (int s = 0; s < components(ctx); s++) {
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
    return potential_response(*hartree, xc, nullptr, entropy__, spin_orbit__, eval_sum__);
}

std::unique_ptr<Density>
LAPW_response::geometry_rhs(xc_geometry_direction_t const& direction__, double* entropy__, double* spin_orbit__,
                            double* eval_sum__) const
{
    auto xc      = potential_.lapw_external_xc_geometry_response(density_, direction__);
    auto hartree = potential_.hartree_geometry_response(density_.rho(), direction__);
    return potential_response(*hartree, xc, &direction__, entropy__, spin_orbit__, eval_sum__);
}

LAPW_response::radial_geometry_response_t
LAPW_response::radial_geometry_response(xc_geometry_direction_t const& direction__, double tolerance__,
                                        int max_iterations__) const
{
    double explicit_entropy{0}, explicit_eval_sum{0};
    auto hartree    = potential_.hartree_geometry_response(density_.rho(), direction__);
    auto xc         = potential_.lapw_external_xc_geometry_response(density_, direction__);
    auto rhs        = potential_response(*hartree, xc, &direction__, &explicit_entropy, nullptr, &explicit_eval_sum);
    auto solved     = solve(*rhs, tolerance__, max_iterations__);
    auto induced_xc = potential_.lapw_external_xc_response(density_, *solved.fields);
    auto induced_hartree = potential_.hartree_response(solved.fields->rho());
    radial_geometry_response_t result, explicit_response;
    potential_.lapw_radial_potential_response(*induced_hartree, induced_xc, result.scalar, result.core);
    potential_.lapw_radial_potential_response(*hartree, xc, explicit_response.scalar, explicit_response.core);
    auto explicit_contraction = potential_.lapw_xc_contractions(density_, xc);
    auto induced_contraction  = potential_.lapw_xc_contractions(density_, induced_xc);
    result.xc_contraction     = explicit_contraction.rho + explicit_contraction.magnetic + explicit_contraction.tau +
                            induced_contraction.rho + induced_contraction.magnetic + induced_contraction.tau;
    for (size_t ic = 0; ic < result.scalar.size(); ic++) {
        for (size_t ir = 0; ir < result.scalar[ic].size(); ir++) {
            result.scalar[ic][ir] += explicit_response.scalar[ic][ir];
            for (int part = 0; part < 2; part++) {
                result.core[ic][part][ir] += explicit_response.core[ic][part][ir];
            }
        }
    }
    result.fields   = std::move(solved.fields);
    result.entropy  = explicit_entropy + solved.entropy;
    result.eval_sum = explicit_eval_sum + solved.eval_sum;
    return result;
}

double
LAPW_response::energy_geometry_derivative(xc_geometry_direction_t const& direction__,
                                          radial_geometry_response_t const& response__,
                                          xc_geometry_derivatives_t const& explicit_xc__) const
{
    auto& ctx   = ks_.ctx();
    int invalid = !response__.fields || !std::isfinite(response__.entropy) || !std::isfinite(response__.eval_sum) ||
                  !std::isfinite(response__.xc_contraction) ||
                  explicit_xc__.atoms.size() != static_cast<size_t>(ctx.unit_cell().num_atoms());
    for (auto const& atom : explicit_xc__.atoms) {
        for (double value : atom) {
            invalid |= !std::isfinite(value);
        }
    }
    for (double value : explicit_xc__.lattice) {
        invalid |= !std::isfinite(value);
    }
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
    // Differentiate the same eigenvalue-sum identity used by energy_kin, including SOC and core states.
    double result = response__.eval_sum + response__.entropy - response__.xc_contraction;
    result -= 0.5 * (inner(fields.rho(), potential_.hartree_potential()) + inner(density_.rho(), *potential) +
                     inner(density_.rho().rg(), potential_.hartree_potential().rg(),
                           [&](int ir) { return measure.value(ir); }));
    for (int s = 0; s < ctx.num_mag_dims(); s++) {
        result -= inner(fields.mag(s), potential_.effective_magnetic_field(s)) +
                  inner(density_.mag(s).rg(), potential_.effective_magnetic_field(s).rg(),
                        [&](int ir) { return measure.value(ir); });
    }
    double nuclei{0};
    for (auto it : ctx.unit_cell().spl_num_atoms()) {
        nuclei -= 0.5 * ctx.unit_cell().atom(it.i).zn() * y00 * potential->mt()[it.i](0, 0);
    }
    ctx.comm().allreduce(&nuclei, 1);
    result += nuclei;
    // Reciprocal XC contractions contain the cell volume; MT covectors do not.
    result -= trace * potential_.lapw_xc_contractions(density_).interstitial;
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            result += direction__.atoms[ia][x] * explicit_xc__.atoms[ia][x];
        }
    }
    for (int i = 0; i < 9; i++) {
        result += direction__.lattice[i] * explicit_xc__.lattice[i];
    }
    if (!std::isfinite(result)) {
        RTE_THROW("nonfinite LAPW free-energy geometry derivative");
    }
    return result;
}

std::unique_ptr<Density>
LAPW_response::potential_response(Periodic_function<double> const& hartree__, lapw_xc_field_adjoint_t const& xc__,
                                  xc_geometry_direction_t const* geometry__, double* entropy__, double* spin_orbit__,
                                  double* eval_sum__) const
{
    auto& ctx  = ks_.ctx();
    bool dirac = ctx.valence_relativity() == relativity_t::dirac;
    int mode   = spin_orbit__ != nullptr;
    ctx.comm().bcast(&mode, 1, 0);
    int invalid = mode != (spin_orbit__ != nullptr);
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW SOC energy output must be requested on every context rank");
    }
    bool soc_energy = spin_orbit__ && ctx.so_correction();
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
    bool iora  = ctx.valence_relativity() == relativity_t::iora;
    auto physical_atoms =
            iora ? h0_->lapw_radial_basis_response(radial, false, true) : std::vector<lapw_atom_basis_response_t>{};
    auto field = h0_->lapw_field_response(hartree__, xc__);
    lapw_hamiltonian_response_t soc;
    std::vector<lapw_atom_basis_response_t> soc_atoms;
    if (soc_energy) {
        soc       = h0_->lapw_spin_fields(true);
        soc_atoms = h0_->lapw_radial_basis_response(radial, true);
    }
    std::optional<lapw_interstitial_geometry_response_t> mass_geometry;
    if (geometry__ &&
        (ctx.valence_relativity() == relativity_t::zora || ctx.valence_relativity() == relativity_t::iora)) {
        // Prepare the context-collective FFT before ranks enter their local k-point loops.
        mass_geometry = h0_->lapw_interstitial_geometry_response(*geometry__);
    }
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
        // XC already differentiates its restriction; raw fields still need the moving mask.
        Smooth_periodic_function<double> masked(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        for (int s = 0; s < components(ctx); s++) {
            auto const& raw = (dirac || ctx.num_mag_dims() == 3) && s != 0 ? potential_.component(s).rg()
                                                                           : potential_.hartree_potential().rg();
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                masked.value(ir) = boundary.value(ir) * raw.value(ir);
            }
            masked.fft_transform(-1);
            auto pw = masked.gather_f_pw();
            for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
                field.scalar_pw(ig, s) += pw[ig];
            }
        }
    }
    using matrix_t = la::dmatrix<std::complex<double>>;
    std::vector<std::array<std::unique_ptr<matrix_t>, 4>> dh, ds;
    std::vector<std::array<std::unique_ptr<matrix_t>, 2>> physical_ds;
    std::vector<LAPW_density_response::matrix_direction_t> directions;
    for (auto it : ks_.spl_num_kpoints()) {
        auto& kp = *ks_.get<double>(it.i);
        auto hk  = (*h0_)(kp);
        int n    = dirac ? 2 * kp.num_gkvec() + ctx.unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
        int bs   = ctx.cyclic_block_size();
        dh.emplace_back();
        ds.emplace_back();
        physical_ds.emplace_back();
        directions.emplace_back();
        try {
            matrix_t field_h(n, n, ctx.blacs_grid(), bs, bs);
            if (dirac) {
                dh.back()[0] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                ds.back()[0] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                hk.set_dirac_h_o_radial_response(radial, atoms, *dh.back()[0], *ds.back()[0]);
                hk.set_dirac_h_field_response(field, field_h);
                std::unique_ptr<matrix_t> geometry_h, geometry_s;
                if (geometry__) {
                    geometry_h = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                    geometry_s = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                    hk.set_dirac_h_o_geometry_response(*geometry__, *geometry_h, *geometry_s);
                }
                for (int j = 0; j < field_h.num_cols_local(); j++) {
                    for (int i = 0; i < field_h.num_rows_local(); i++) {
                        (*dh.back()[0])(i, j) += field_h(i, j);
                        if (geometry__) {
                            (*dh.back()[0])(i, j) += (*geometry_h)(i, j);
                            (*ds.back()[0])(i, j) += (*geometry_s)(i, j);
                        }
                    }
                }
                directions.back().h[0] = dh.back()[0].get();
                directions.back().s[0] = ds.back()[0].get();
                continue;
            }
            std::unique_ptr<matrix_t> field_s;
            std::unique_ptr<matrix_t> physical_h, physical_geometry;
            if (iora) {
                field_s    = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                physical_h = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                if (geometry__) {
                    physical_geometry = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                }
            }
            std::unique_ptr<matrix_t> geometry_h, geometry_s;
            if (geometry__) {
                geometry_h = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                geometry_s = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
            }
            for (int s = 0; s < components(ctx); s++) {
                dh.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                ds.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                hk.set_fv_h_o_radial_response(s, radial, atoms, *dh.back()[s], *ds.back()[s]);
                if (iora) {
                    physical_ds.back()[s] = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                    hk.set_fv_h_o_radial_response(s, radial, physical_atoms, *physical_h, *physical_ds.back()[s]);
                }
                hk.set_fv_h_field_response(s, field, field_h, field_s.get());
                if (geometry__) {
                    hk.set_fv_h_o_geometry_response(s, *geometry__, *geometry_h, *geometry_s, false,
                                                    mass_geometry ? &*mass_geometry : nullptr);
                    if (iora) {
                        hk.set_fv_h_o_geometry_response(s, *geometry__, *physical_h, *physical_geometry, false,
                                                        &*mass_geometry, true);
                    }
                }
                for (int j = 0; j < field_h.num_cols_local(); j++) {
                    for (int i = 0; i < field_h.num_rows_local(); i++) {
                        (*dh.back()[s])(i, j) += field_h(i, j);
                        if (field_s) {
                            (*ds.back()[s])(i, j) += (*field_s)(i, j);
                        }
                        if (geometry__) {
                            (*dh.back()[s])(i, j) += (*geometry_h)(i, j);
                            (*ds.back()[s])(i, j) += (*geometry_s)(i, j);
                            if (iora) {
                                (*physical_ds.back()[s])(i, j) += (*physical_geometry)(i, j);
                            }
                        }
                    }
                }
                directions.back().h[s] = dh.back()[s].get();
                if (ctx.num_mag_dims() != 3 || s == 0) {
                    directions.back().s[s]          = ds.back()[s].get();
                    directions.back().physical_s[s] = iora ? physical_ds.back()[s].get() : nullptr;
                }
            }
        } catch (std::exception const&) {
            invalid = 1;
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("coupled LAPW Hamiltonian response failed");
    }
    auto dp = spectrum_->density_matrix_response(directions, entropy__, eval_sum__);
    if (eval_sum__) {
        double core_energy{0};
        for (auto it : ctx.unit_cell().spl_num_atoms()) {
            core_energy += radial[ctx.unit_cell().atom(it.i).symmetry_class().id()].core_eval_sum;
        }
        ctx.comm().allreduce(&core_energy, 1);
        *eval_sum__ += core_energy;
        if (!std::isfinite(*eval_sum__)) {
            RTE_THROW("nonfinite total LAPW eigenvalue-sum response");
        }
    }
    if (spin_orbit__) {
        std::complex<double> energy{0};
        if (soc_energy) {
            size_t ik{0};
            for (auto it : ks_.spl_num_kpoints()) {
                auto& kp = *ks_.get<double>(it.i);
                auto hk  = (*h0_)(kp);
                int n = kp.gklo_basis_size(), bs = ctx.cyclic_block_size();
                try {
                    matrix_t h(n, n, ctx.blacs_grid(), bs, bs), dh(n, n, ctx.blacs_grid(), bs, bs);
                    matrix_t overlap(n, n, ctx.blacs_grid(), bs, bs);
                    std::unique_ptr<matrix_t> geometry_h, geometry_o;
                    if (geometry__) {
                        geometry_h = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                        geometry_o = std::make_unique<matrix_t>(n, n, ctx.blacs_grid(), bs, bs);
                    }
                    for (int s = 1; s < 4; s++) {
                        hk.set_fv_h_field_response(s, soc, h);
                        hk.set_fv_h_o_radial_response(s, radial, soc_atoms, dh, overlap, true);
                        if (geometry__) {
                            hk.set_fv_h_o_geometry_response(s, *geometry__, *geometry_h, *geometry_o, true);
                            for (int j = 0; j < dh.num_cols_local(); j++) {
                                for (int i = 0; i < dh.num_rows_local(); i++) {
                                    dh(i, j) += (*geometry_h)(i, j);
                                }
                            }
                        }
                        auto full_h = h.get_full_matrix(), full_dh = dh.get_full_matrix();
                        // Differentiate Tr(P H_SOC), including the moving AW/LO matching coefficients.
                        for (int j = 0; j < n; j++) {
                            for (int i = 0; i < n; i++) {
                                energy += kp.weight() / kp.comm().size() *
                                          (dp[ik](j, i, s) * full_h(i, j) + matrices_[ik](j, i, s) * full_dh(i, j));
                            }
                        }
                    }
                } catch (std::exception const&) {
                    invalid = 1;
                }
                ik++;
            }
        }
        ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
        ctx.comm().allreduce(&energy, 1);
        if (invalid || !std::isfinite(std::abs(energy)) ||
            std::abs(energy.imag()) > 1e-10 * (1 + std::abs(energy.real()))) {
            RTE_THROW("invalid LAPW SOC expectation derivative");
        }
        *spin_orbit__ = energy.real();
    }
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
                for (int s = 0; s < components(ctx); s++) {
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
                for (int j = 0; j < components(a.ctx()); j++) {
                    rotate(c, s, a.component(j), b.component(j));
                    rotate(c, s, a.kinetic_density().component(j), b.kinetic_density().component(j));
                }
            });
    mixer::Anderson_stable<Density> mixer(history__, beta__);
    mixer.initialize_function<0>(properties, *x, ctx, true);
    std::array<std::array<double, 4>, 8> norms{};
    double residual{0};
    for (int iteration = 0; iteration < max_iterations__; iteration++) {
        double entropy{0}, eval_sum{0};
        auto y = apply(*x, &entropy, nullptr, &eval_sum);
        add(1.0, rhs__, *y);
        residual = field_residual(rhs__, *x, *y, norms);
        if (!std::isfinite(residual)) {
            RTE_THROW("nonfinite coupled LAPW response residual");
        }
        if (residual <= tolerance__) {
            return {std::move(x), iteration + 1, residual, entropy, eval_sum};
        }
        mixer.set_input<0>(*y);
        mixer.mix(0);
        mixer.get_output<0>(*x);
        synchronize(*x);
    }
    std::ostringstream message;
    message << "coupled LAPW response residual " << residual << " exceeds tolerance " << tolerance__ << " after "
            << max_iterations__ << " iterations";
    for (int part : {0, 1}) {
        for (int s = 0; s < components(ctx); s++) {
            auto const& norm = norms[4 * part + s];
            message << "\n"
                    << (part == 0 ? "rho" : "tau") << " component " << s << ": error=" << norm[0]
                    << ", scale=" << norm[1] << ", regularized error=" << norm[2] << ", regularized scale=" << norm[3];
        }
    }
    RTE_THROW(message.str());
    return {};
}

} // namespace sirius
