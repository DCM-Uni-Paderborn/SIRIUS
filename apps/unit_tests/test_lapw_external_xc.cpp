/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/dft_ground_state.hpp"
#include "hamiltonian/diagonalize.hpp"
#include "hamiltonian/diagonalize_fp.hpp"
#include "dft/smearing.hpp"
#include "lapw/step_function.hpp"

using namespace sirius;

namespace {

double
functional_scale(Simulation_context const& ctx)
{
    return ctx.unit_cell().num_core_electrons() != 0 ? 1e-5 : 1.0;
}

void
functional(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out)
{
    if (!in.points) {
        RTE_THROW("LAPW fixture requires reconstructed point fields");
    }
    double moment{0}, energy{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j = p + in.num_points * s;
            moment += in.points->descriptor_weights[p] * (in.rho[j] + 0.3 * in.tau[j]);
        }
    }
    ctx.comm().allreduce(&moment, 1);
    double nonlocal = 0.01 * moment / ctx.unit_cell().omega();
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j    = p + in.num_points * s;
            double w = in.points->energy_weights[p], d = in.points->descriptor_weights[p];
            double r = in.rho[j], t = in.tau[j];
            double local = 0.02 * r * r + 0.01 * t * t + 0.005 * r * t;
            out.vrho[j]  = w * (0.04 * r + 0.005 * t) + d * nonlocal;
            out.vtau[j]  = w * (0.02 * t + 0.005 * r) + 0.3 * d * nonlocal;
            for (int x = 0; x < 3; x++) {
                int k = p + in.num_points * (x + 3 * s);
                local += 0.002 * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = 0.004 * w * in.gradient[k];
            }
            energy += w * local;
            if (out.geometry) {
                out.geometry->energy_weights[p] += local;
                out.geometry->descriptor_weights[p] += nonlocal * (r + 0.3 * t);
            }
        }
    }
    ctx.comm().allreduce(&energy, 1);
    out.energy = energy + 0.5 * nonlocal * moment;
    if (ctx.unit_cell().num_core_electrons() != 0) {
        // Keep the quadratic fixture's near-nucleus fields in the weak-response regime.
        double scale = functional_scale(ctx);
        out.energy *= scale;
        for (int j = 0; j < in.num_points * in.num_spins; j++) {
            out.vrho[j] *= scale;
            out.vtau[j] *= scale;
        }
        for (int j = 0; j < 3 * in.num_points * in.num_spins; j++) {
            out.vgradient[j] *= scale;
        }
        if (out.geometry) {
            for (auto* weights : {&out.geometry->energy_weights, &out.geometry->descriptor_weights}) {
                for (auto& w : *weights) {
                    w *= scale;
                }
            }
        }
    }
}

void
functional_response(Simulation_context const& ctx, xc_callback_input_t const& in,
                    xc_callback_direction_t const& direction, xc_callback_response_t& out)
{
    double moment{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j = p + in.num_points * s;
            moment += in.points->descriptor_weights[p] * (direction.rho[j] + 0.3 * direction.tau[j]);
        }
    }
    ctx.comm().allreduce(&moment, 1);
    double nonlocal = 0.01 * moment / ctx.unit_cell().omega(), scale = functional_scale(ctx);
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j    = p + in.num_points * s;
            double w = in.points->energy_weights[p], d = in.points->descriptor_weights[p];
            out.vrho[j] = scale * (w * (0.04 * direction.rho[j] + 0.005 * direction.tau[j]) + d * nonlocal);
            out.vtau[j] = scale * (w * (0.02 * direction.tau[j] + 0.005 * direction.rho[j]) + 0.3 * d * nonlocal);
            for (int x = 0; x < 3; x++) {
                int k            = p + in.num_points * (x + 3 * s);
                out.vgradient[k] = scale * 0.004 * w * direction.gradient[k];
            }
        }
    }
}

auto
context(int mag, bool distributed_fft, bool second_variation, bool core = false, bool relativistic_core = false,
        double displacement = 0, double pw_cutoff = 5.0, bool simple_radial_integrals = false, int auto_enu = 0)
{
    auto config                               = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
        "pw_cutoff": 5.0, "gk_cutoff": 1.5, "num_bands": 4, "xc_functionals": [],
        "core_relativity": "none", "valence_relativity": "none", "auto_rmt": 0,
        "lmax_apw": 3, "lmax_rho": 6, "lmax_pot": 6, "smearing_width": 0.01
    }, "control": {"processing_unit": "CPU", "verbosity": 0, "use_second_variation": true},
    "iterative_solver": {"type": "exact", "min_tolerance": 1e-11},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false}})"_json;
    config["parameters"]["num_mag_dims"]      = mag;
    config["parameters"]["fixed_mag"]         = mag ? 0.4 : 0.0;
    config["control"]["use_second_variation"] = second_variation;
    config["control"]["mpi_grid_dims"]        = {distributed_fft ? mpi::Communicator::world().size() : 1, 1};
    config["parameters"]["core_relativity"]   = relativistic_core ? "dirac" : "none";
    config["parameters"]["num_bands"]         = core ? 8 : 4;
    config["parameters"]["pw_cutoff"]         = pw_cutoff;
    config["settings"]["simple_lapw_ri"]      = simple_radial_integrals;
    auto ctx                                  = std::make_unique<Simulation_context>(config);
    ctx->unit_cell().set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    std::string label = core ? "Ne" : "He";
    auto& type        = ctx->unit_cell().add_atom_type(label);
    type.zn(core ? 10 : 2);
    type.set_radial_grid(radial_grid_t::power, core ? 3201 : 801, 1e-6, 1.8, 2.0);
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
    std::vector<double> rho(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        rho[ir] = core ? 2 * std::pow(9.0, 3) / pi * std::exp(-18 * grid[ir]) +
                                  8 * std::pow(1.6, 3) / pi * std::exp(-3.2 * grid[ir])
                       : 16 / pi * std::exp(-4 * grid[ir]);
    }
    type.free_atom_density(rho);
    type.set_free_atom_radial_grid(std::move(grid));
    for (int l = 0; l <= 3; l++) {
        type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, 0, auto_enu);
        type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, 1, auto_enu);
    }
    type.add_lo_descriptor(0, core ? 2 : 1, 0, -0.7, 0, auto_enu);
    type.add_lo_descriptor(0, core ? 2 : 1, 0, -0.7, 1, auto_enu);
    if (core) {
        type.set_configuration(1, 0, 1, 2.0, true);
    }
    ctx->unit_cell().add_atom(
            label, {0.23 + 0.23 * displacement / 6, 0.31 - 0.37 * displacement / 7, 0.43 + 0.19 * displacement / 8},
            {0, 0, mag ? 0.2 : 0.0});
    if (mag && !second_variation && !core) {
        ctx->unit_cell().add_atom(label, {0.71, 0.79, 0.86}, {0, 0, 0.2});
    }
    auto points     = std::make_shared<xc_point_set_t>(*ctx);
    constexpr int n = 10;
    // Replicated FFTs exercise empty model ranks; distributed FFTs also partition model rows.
    if (ctx->comm().rank() == 0 || distributed_fft) {
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    std::array<double, 3> p{6 * (x + 0.37) / n, 7 * (y + 0.29) / n, 8 * (z + 0.41) / n};
                    double r2 = std::pow(p[0] - 1.38, 2) + std::pow(p[1] - 2.17, 2) + std::pow(p[2] - 3.44, 2);
                    if (core && r2 <= 1.8 * 1.8) {
                        continue;
                    }
                    points->coordinates.push_back(p);
                    points->energy_weights.push_back(336.0 / (n * n * n));
                    points->descriptor_weights.push_back(336.0 / (n * n * n));
                }
            }
        }
        points->energy_weights[0] = 0;
        if (core) {
            // Radially resolved sphere quadrature is essential for core feedback.
            auto const& radial = type.radial_grid();
            Spline<double> prototype(radial);
            mdarray<double, 2> integral({radial.num_points(), 4});
            integral.zero();
            for (int ir = 0; ir < radial.num_points() - 1; ir++) {
                double h = radial[ir + 1] - radial[ir];
                for (int j = 0; j < 4; j++) {
                    integral(ir, j) = std::pow(h, j + 1) / (j + 1);
                }
            }
            auto weights = prototype.interpolation_adjoint(integral);
            for (int ir = 0; ir < radial.num_points(); ir++) {
                for (int axis = 0; axis < 3; axis++) {
                    for (int sign : {-1, 1}) {
                        std::array<double, 3> p{1.38, 2.17, 3.44};
                        p[axis] += sign * radial[ir];
                        points->coordinates.push_back(p);
                        double w = fourpi / 6 * radial[ir] * radial[ir] * weights[ir];
                        points->energy_weights.push_back(w);
                        points->descriptor_weights.push_back(w);
                    }
                }
            }
        }
        if (distributed_fft) {
            size_t local{0};
            for (size_t p = 0; p < points->coordinates.size(); p++) {
                if (p % ctx->comm().size() == static_cast<size_t>(ctx->comm().rank())) {
                    points->coordinates[local]          = points->coordinates[p];
                    points->energy_weights[local]       = points->energy_weights[p];
                    points->descriptor_weights[local++] = points->descriptor_weights[p];
                }
            }
            points->coordinates.resize(local);
            points->energy_weights.resize(local);
            points->descriptor_weights.resize(local);
        }
        if (!points->coordinates.empty()) {
            points->block_sizes   = {static_cast<int>(points->coordinates.size())};
            points->block_centres = {{1.38, 2.17, 3.44}};
        }
    }
    ctx->cb().external_xc_points_   = std::move(points);
    ctx->cb().external_xc_          = functional;
    ctx->cb().external_xc_response_ = functional_response;
    ctx->initialize();
    return ctx;
}

double
occupied_lapw_lagrangian(Simulation_context const& ctx, K_point<double> const& kp, int spin,
                         la::dmatrix<std::complex<double>> const& h, la::dmatrix<std::complex<double>> const& o)
{
    double value{0};
    for (int b = 0; b < ctx.num_bands(); b++) {
        auto all = kp.spinor_wave_functions().gather(wf::spin_index(spin), wf::band_index(b));
        int ng   = kp.gkvec().num_gvec();
        std::vector<std::complex<double>> c(all.begin(), all.begin() + ng);
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& type = ctx.unit_cell().atom(ia).type();
            c.insert(c.end(), all.begin() + ng + type.mt_aw_basis_size(), all.begin() + ng + type.mt_basis_size());
            ng += type.mt_basis_size();
        }
        double weight = kp.weight() * kp.band_occupancy(b, spin) * ctx.blacs_grid().comm().size() / kp.comm().size();
        for (int j = 0; j < h.num_cols_local(); j++) {
            for (int i = 0; i < h.num_rows_local(); i++) {
                value += weight * std::real(std::conj(c[h.irow(i)]) * (h(i, j) - kp.band_energy(b, spin) * o(i, j)) *
                                            c[h.icol(j)]);
            }
        }
    }
    return value;
}

std::vector<double>
potential_samples(Periodic_function<double> const& potential)
{
    std::vector<double> result;
    for (auto z : potential.rg().f_pw_local()) {
        result.push_back(z.real());
        result.push_back(z.imag());
    }
    for (auto x : potential.rg().values()) {
        result.push_back(x);
    }
    for (auto it : potential.ctx().unit_cell().spl_num_atoms()) {
        auto const& field = potential.mt()[it.i];
        for (size_t i = 0; i < field.size(); i++) {
            result.push_back(field[i]);
        }
    }
    return result;
}

double
potential_response_error(std::vector<double> const& response, std::vector<double> const& plus,
                         std::vector<double> const& minus, double step)
{
    double error{0}, peak{0};
    if (response.size() != plus.size() || response.size() != minus.size()) {
        return std::numeric_limits<double>::infinity();
    }
    for (size_t i = 0; i < response.size(); i++) {
        double fd = (plus[i] - minus[i]) / (2 * step);
        if (!std::isfinite(fd) || !std::isfinite(response[i])) {
            return std::numeric_limits<double>::infinity();
        }
        error = std::max(error, std::abs(fd - response[i]));
        peak  = std::max(peak, std::abs(response[i]));
    }
    return error / std::max(1e-8, peak);
}

std::vector<double>
xc_samples(lapw_xc_field_adjoint_t const& fields)
{
    std::vector<double> result;
    for (auto const* part : {&fields.rho, &fields.tau}) {
        for (auto const& spin : *part) {
            for (auto value : spin.f_pw_local()) {
                result.push_back(value.real());
                result.push_back(value.imag());
            }
        }
    }
    for (auto const& atom : fields.local) {
        for (auto const& part : atom) {
            for (auto const& spin : part) {
                for (size_t i = 0; i < spin.size(); i++) {
                    result.push_back(spin[i]);
                }
            }
        }
    }
    for (auto const& atom : fields.core) {
        for (auto const& part : atom) {
            result.insert(result.end(), part.begin(), part.end());
        }
    }
    return result;
}

std::vector<double>
radial_xc_response(Simulation_context& ctx, Potential const& potential, Density const& density,
                   std::vector<std::array<std::vector<Flm>, 2>> const& fields)
{
    Density direction(ctx);
    direction.zero();
    direction.kinetic_density().zero();
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        for (int part : {0, 1}) {
            auto& target = part == 0 ? static_cast<Field4D&>(direction) : direction.kinetic_density();
            for (int s = 0; s < ctx.num_spins(); s++) {
                target.scalar().mt()[ia] += fields[ia][part][s];
                if (ctx.num_spins() == 2) {
                    for (size_t i = 0; i < fields[ia][part][s].size(); i++) {
                        target.vector(0).mt()[ia][i] += (1 - 2 * s) * fields[ia][part][s][i];
                    }
                }
            }
        }
    }
    return xc_samples(potential.lapw_external_xc_response(density, direction));
}

std::vector<double>
radial_hartree_response(Simulation_context const& ctx, Potential const& potential,
                        std::vector<std::array<std::vector<Flm>, 2>> const& fields)
{
    Periodic_function<double> direction(
            ctx, [&](int) { return lmax_t(ctx.lmax_rho()); }, &ctx.unit_cell().spl_num_atoms());
    direction.zero();
    // Fixed PW/LO coefficients give no interstitial response. Sum physical spin densities in each sphere.
    for (auto it : ctx.unit_cell().spl_num_atoms()) {
        for (auto const& spin : fields[it.i][0]) {
            direction.mt()[it.i] += spin;
        }
    }
    return potential_samples(*potential.hartree_response(direction));
}

std::vector<double>
sample_radial_fields(Simulation_context& ctx, K_point_set& ks, std::vector<std::vector<double>> const& vs,
                     std::vector<std::array<std::vector<double>, 2>> const& core_xc,
                     std::vector<double>* hartree = nullptr, std::vector<double>* xc = nullptr)
{
    std::vector<int> nlo;
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        nlo.push_back(ctx.unit_cell().atom(ia).type().mt_lo_basis_size());
    }
    std::vector<std::complex<double>> saved;
    for (auto it : ks.spl_num_kpoints()) {
        auto kp   = ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        wf::Wave_functions<double> evec(kp->gkvec_sptr(), nlo, wf::num_mag_dims(0), wf::num_bands(ctx.num_bands()),
                                        memory_t::host);
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int b = 0; b < ctx.num_bands(); b++) {
                auto spin = wf::spin_index(s);
                auto band = wf::band_index(b);
                auto data = psi.at(memory_t::host, 0, spin, band);
                saved.insert(saved.end(), data, data + psi.ld());
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    evec.pw_coeffs(ig, wf::spin_index(0), band) = psi.pw_coeffs(ig, spin, band);
                }
                for (auto atom : evec.spl_num_atoms()) {
                    int naw = ctx.unit_cell().atom(atom.i).type().mt_aw_basis_size();
                    for (int j = 0; j < nlo[atom.i]; j++) {
                        evec.mt_coeffs(j, atom.li, wf::spin_index(0), band) =
                                psi.mt_coeffs(naw + j, atom.li, spin, band);
                    }
                }
            }
            kp->generate_lapw_wave_functions(evec, psi, s);
        }
    }
    auto restore = [&] {
        size_t offset{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto& psi = ks.get<double>(it.i)->spinor_wave_functions();
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (int b = 0; b < ctx.num_bands(); b++) {
                    std::copy_n(saved.data() + offset, psi.ld(),
                                psi.at(memory_t::host, 0, wf::spin_index(s), wf::band_index(b)));
                    offset += psi.ld();
                }
            }
        }
    };
    std::vector<double> result;
    try {
        Density density(ctx);
        density.generate_core_charge_density(vs, core_xc);
        density.generate<double>(ks, false, true, false);
        if (hartree) {
            Potential potential(ctx);
            // Omit the fixed nuclear potential before subtraction to avoid near-origin cancellation.
            *hartree = potential_samples(*potential.hartree_response(density.rho()));
        }
        if (xc) {
            Potential potential(ctx);
            potential.xc(density);
            *xc = xc_samples(*potential.lapw_xc_derivatives());
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (auto* fields : {static_cast<Field4D*>(&density), &density.kinetic_density()}) {
                for (int s = 0; s < ctx.num_spins(); s++) {
                    auto const& scalar = fields->scalar().mt()[ia];
                    for (size_t i = 0; i < scalar.size(); i++) {
                        result.push_back(ctx.num_spins() == 1
                                                 ? scalar[i]
                                                 : 0.5 * (scalar[i] + (1 - 2 * s) * fields->vector(0).mt()[ia][i]));
                    }
                }
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    return result;
}

double
radial_fields_error(std::vector<std::array<std::vector<Flm>, 2>> const& analytic, std::vector<double> const& plus,
                    std::vector<double> const& minus, double step)
{
    size_t offset{0};
    double error{0};
    for (auto const& atom : analytic) {
        for (auto const& part : atom) {
            for (auto const& spin : part) {
                double peak{0}, difference{0};
                for (size_t i = 0; i < spin.size(); i++, offset++) {
                    double fd = (plus.at(offset) - minus.at(offset)) / (2 * step);
                    if (!std::isfinite(fd) || !std::isfinite(spin[i])) {
                        return std::numeric_limits<double>::infinity();
                    }
                    peak       = std::max(peak, std::abs(spin[i]));
                    difference = std::max(difference, std::abs(fd - spin[i]));
                }
                error = std::max(error, difference / std::max(1e-8, peak));
            }
        }
    }
    if (offset != plus.size() || offset != minus.size()) {
        return std::numeric_limits<double>::infinity();
    }
    return error;
}

std::array<double, 4>
lapw_spectrum_checks(Hamiltonian_k<double> const& hk, la::dmatrix<std::complex<double>>& dh)
{
    auto& ctx = hk.H0().ctx();
    auto& kp  = hk.kp();
    int n = kp.gklo_basis_size(), bs = ctx.cyclic_block_size();
    la::dmatrix<std::complex<double>> h(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> s(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> ds(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> shifted_h(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> shifted_s(n, n, ctx.blacs_grid(), bs, bs);
    hk.set_fv_h_o(0, h, s);
    for (int j = 0; j < ds.num_cols_local(); j++) {
        for (int i = 0; i < ds.num_rows_local(); i++) {
            int row = ds.irow(i), col = ds.icol(j);
            ds(i, j) = 0.03 * s(i, j) + 0.002 * std::polar(1.0, 0.7 * (row - col)) / ((1.0 + row) * (1.0 + col));
        }
    }
    auto full_ds = ds.get_full_matrix();
    std::array<double, 4> errors{};
    for (bool degenerate : {false, true}) {
        if (degenerate) {
            for (int j = 0; j < h.num_cols_local(); j++) {
                for (int i = 0; i < h.num_rows_local(); i++) {
                    h(i, j) = 0.25 * s(i, j);
                }
            }
        }
        LAPW_eigensystem reference(hk, h, s);
        auto direction         = reference.perturbation(dh, ds);
        double mu              = degenerate ? 0.25 : reference.energies()[ctx.num_bands() / 2];
        constexpr double width = 0.4;
        auto occupations       = [&](auto const& spectrum, double chemical_potential) {
            std::vector<double> f;
            for (double energy : spectrum.energies()) {
                f.push_back(ctx.max_occupancy() * smearing::gaussian::occupancy(chemical_potential - energy, width));
            }
            return f;
        };
        auto f = occupations(reference, mu);
        std::vector<double> slopes;
        double slope_sum{0}, energy_sum{0};
        for (int i = 0; i < n; i++) {
            slopes.push_back(-ctx.max_occupancy() * smearing::gaussian::delta(mu - reference.energies()[i], width));
            slope_sum += slopes.back();
            energy_sum += slopes.back() * direction.energies[i];
        }
        auto p = reference.density_matrix(f);
        for (bool fixed_charge : {false, true}) {
            double dmu    = fixed_charge ? energy_sum / slope_sum : 0;
            auto response = reference.density_matrix_response(direction, f, slopes, dmu);
            std::complex<double> trace{0};
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    trace += response(i, j) * reference.overlap()(j, i) + p(i, j) * full_ds(j, i);
                }
            }
            errors[2]   = std::max(errors[2], std::abs(trace - (energy_sum - slope_sum * dmu)));
            auto sample = [&](double step) {
                for (int j = 0; j < h.num_cols_local(); j++) {
                    for (int i = 0; i < h.num_rows_local(); i++) {
                        shifted_h(i, j) = h(i, j) + step * dh(i, j);
                        shifted_s(i, j) = s(i, j) + step * ds(i, j);
                    }
                }
                LAPW_eigensystem shifted(hk, shifted_h, shifted_s);
                return shifted.density_matrix(occupations(shifted, mu + step * dmu));
            };
            for (double step : {1e-4, 5e-5}) {
                auto plus = sample(step), minus = sample(-step);
                double error{0}, peak{0};
                for (size_t i = 0; i < response.size(); i++) {
                    double e = std::abs(response[i] - (plus[i] - minus[i]) / (2 * step));
                    error    = std::isfinite(e) ? std::max(error, e) : std::numeric_limits<double>::infinity();
                    peak     = std::max(peak, std::abs(response[i]));
                }
                errors[degenerate] = std::max(errors[degenerate], error / std::max(1.0, peak));
            }
        }
        if (degenerate) {
            auto expect_rejection = [&](auto&& operation) {
                try {
                    operation();
                    errors[3]++;
                } catch (std::exception const&) {
                }
            };
            auto inconsistent = f;
            inconsistent[0] += 0.1;
            expect_rejection([&] { reference.density_matrix_response(direction, inconsistent, slopes); });
            auto incomplete = f;
            incomplete.pop_back();
            expect_rejection([&] { reference.density_matrix(incomplete); });
            expect_rejection([&] {
                reference.density_matrix_response(direction, f, slopes,
                                                  kp.comm().rank() == 0 ? std::numeric_limits<double>::quiet_NaN() : 0);
            });
            std::complex<double> saved{0};
            if (kp.comm().rank() == 0) {
                saved    = dh(0, 0);
                dh(0, 0) = std::numeric_limits<double>::quiet_NaN();
            }
            expect_rejection([&] { reference.perturbation(dh, ds); });
            if (kp.comm().rank() == 0) {
                dh(0, 0) = saved;
            }
            if (kp.comm().size() > 1) {
                auto different = f;
                if (kp.comm().rank() == 0) {
                    different[0] += 0.1;
                }
                expect_rejection([&] { reference.density_matrix(different); });
                expect_rejection([&] {
                    reference.density_matrix_response(direction, f, slopes, kp.comm().rank() == 0 ? 0.1 : 0);
                });
            }
        }
    }
    return errors;
}

int
test_lapw_external(int mag, bool distributed_fft, bool second_variation, bool core = false)
{
    bool simple = !mag && second_variation && !core;
    auto ctx    = context(mag, distributed_fft, second_variation, core, false, 0, 5.0, simple);
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        if (ctx->comm().rank() == 0) {
            std::cout << "external LAPW core=" << core << " spins=" << ctx->num_spins()
                      << " atoms=" << ctx->unit_cell().num_atoms() << " distributed FFT=" << distributed_fft
                      << " second variation=" << second_variation << " simple radial integrals=" << simple << " "
                      << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    auto reject = [&](char const* name, auto&& operation) {
        bool rejected{false};
        try {
            operation();
        } catch (std::exception const&) {
            rejected = true;
        }
        check(name, rejected ? 0 : 1, 0);
    };
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    {
        Periodic_function<double> direction(*ctx, [&](int) { return lmax_t(ctx->lmax_rho()); });
        direction.zero();
        auto saved          = potential_samples(potential.hartree_potential());
        double saved_energy = potential.energy_vha();
        std::vector<double> saved_origins;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            saved_origins.push_back(potential.vh_el(ia));
        }
        double peak{0};
        for (auto x : potential_samples(*potential.hartree_response(direction))) {
            peak = std::max(peak, std::abs(x));
        }
        check("zero Hartree response has no nuclear contribution", peak, 0);
        for (int ig = 0; ig < ctx->gvec().count(); ig++) {
            auto g = ctx->gvec().gvec(gvec_index_t::local(ig));
            if (std::abs(g[0]) + std::abs(g[1]) + std::abs(g[2]) == 1) {
                direction.rg().f_pw_local(ig) = {0.01, 0.02 * (g[0] + g[1] + g[2])};
            }
        }
        direction.rg().fft_transform(1);
        auto angular_l = sf::l_by_lm(ctx->lmax_rho());
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            auto const& grid = ctx->unit_cell().atom(ia).radial_grid();
            for (int lm = 0; lm < ctx->lmmax_rho(); lm++) {
                int l = angular_l[lm];
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    direction.mt()[ia](lm, ir) =
                            0.3 * (ia + 1) * std::pow(grid[ir], l) * std::exp(-grid[ir]) / (lm + 1);
                }
            }
        }
        auto response = potential.hartree_response(direction);
        auto samples  = potential_samples(*response);
        Potential displaced(*ctx);
        Density shifted(*ctx);
        for (double step : {0.125, 0.0625}) {
            auto sample = [&](double h) {
                copy(density.rho(), shifted.rho());
                axpy(h, direction, shifted.rho());
                displaced.poisson(shifted.rho());
                return potential_samples(displaced.hartree_potential());
            };
            auto plus = sample(step);
            check("Hartree response matches production Poisson derivative",
                  potential_response_error(samples, plus, sample(-step), step), 2e-8);
        }
        check("Hartree response preserves potential and energy",
              potential_samples(potential.hartree_potential()) != saved || potential.energy_vha() != saved_energy ? 1
                                                                                                                  : 0,
              0);
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            check("Hartree response preserves potential at nuclei", std::abs(potential.vh_el(ia) - saved_origins[ia]),
                  0);
        }
        if (ctx->comm().rank() == 0) {
            direction.rg().f_pw_local(0) = std::numeric_limits<double>::quiet_NaN();
        }
        reject("nonfinite Hartree direction rejected collectively", [&] { potential.hartree_response(direction); });
        direction.zero();
        for (auto it : ctx->unit_cell().spl_num_atoms()) {
            if (it.i == 0) {
                direction.mt()[it.i](0, 0) = std::numeric_limits<double>::quiet_NaN();
            }
        }
        reject("nonfinite MT Hartree direction rejected collectively", [&] { potential.hartree_response(direction); });
        Periodic_function<double> no_spheres(*ctx);
        no_spheres.zero();
        reject("missing MT Hartree direction rejected collectively", [&] { potential.hartree_response(no_spheres); });
    }
    if (!core) {
        Density direction(*ctx), shifted(*ctx);
        direction.zero();
        direction.kinetic_density().zero();
        auto saved          = xc_samples(*potential.lapw_xc_derivatives());
        double saved_energy = potential.energy_exc(density);
        double peak{0};
        for (double x : xc_samples(potential.lapw_external_xc_response(density, direction))) {
            peak = std::max(peak, std::abs(x));
        }
        check("zero XC kernel direction", peak, 0);
        for (int component = 0; component <= ctx->num_mag_dims(); component++) {
            for (int part : {0, 1}) {
                auto& field =
                        part == 0 ? direction.component(component) : direction.kinetic_density().component(component);
                for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                    auto g = ctx->gvec().gvec(gvec_index_t::local(ig));
                    if (std::abs(g[0]) + std::abs(g[1]) + std::abs(g[2]) == 1) {
                        field.rg().f_pw_local(ig) = std::complex<double>(0.003, 0.002 * (g[0] + g[1] + g[2])) *
                                                    double(1 + component + part);
                    }
                }
                field.rg().fft_transform(1);
                auto angular_l = sf::l_by_lm(ctx->lmax_rho());
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    auto const& grid = ctx->unit_cell().atom(ia).radial_grid();
                    for (int lm = 0; lm < ctx->lmmax_rho(); lm++) {
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            field.mt()[ia](lm, ir) = 0.02 * (1 + ia + component + part) *
                                                     std::pow(grid[ir], angular_l[lm]) * std::exp(-grid[ir]) / (lm + 1);
                        }
                    }
                }
            }
        }
        // Add base-dependent curvature to catch a kernel evaluated at the wrong fields.
        ctx->cb().external_xc_ = [](auto const& c, auto const& in, auto& out) {
            functional(c, in, out);
            double energy{0};
            for (int s = 0; s < in.num_spins; s++) {
                for (int p = 0; p < in.num_points; p++) {
                    int j    = p + in.num_points * s;
                    double w = 0.001 * in.points->energy_weights[p], r = in.rho[j], t = in.tau[j];
                    energy += w * r * r * r * t;
                    out.vrho[j] += 3 * w * r * r * t;
                    out.vtau[j] += w * r * r * r;
                    for (int x = 0; x < 3; x++) {
                        int k    = p + in.num_points * (x + 3 * s);
                        double g = in.gradient[k];
                        energy += 0.5 * w * std::pow(g, 4);
                        out.vgradient[k] += 2 * w * g * g * g;
                    }
                }
            }
            c.comm().allreduce(&energy, 1);
            out.energy += energy;
        };
        ctx->cb().external_xc_response_ = [](auto const& c, auto const& in, auto const& d, auto& out) {
            functional_response(c, in, d, out);
            for (int s = 0; s < in.num_spins; s++) {
                for (int p = 0; p < in.num_points; p++) {
                    int j    = p + in.num_points * s;
                    double w = 0.001 * in.points->energy_weights[p], r = in.rho[j], t = in.tau[j];
                    out.vrho[j] += w * (6 * r * t * d.rho[j] + 3 * r * r * d.tau[j]);
                    out.vtau[j] += 3 * w * r * r * d.rho[j];
                    for (int x = 0; x < 3; x++) {
                        int k = p + in.num_points * (x + 3 * s);
                        out.vgradient[k] += 6 * w * in.gradient[k] * in.gradient[k] * d.gradient[k];
                    }
                }
            }
        };
        auto response = xc_samples(potential.lapw_external_xc_response(density, direction));
        Potential displaced(*ctx);
        auto sample = [&](double step) {
            copy(density, shifted);
            for (int component = 0; component <= ctx->num_mag_dims(); component++) {
                axpy(step, direction.component(component), shifted.component(component));
                axpy(step, direction.kinetic_density().component(component),
                     shifted.kinetic_density().component(component));
            }
            displaced.xc(shifted);
            return xc_samples(*displaced.lapw_xc_derivatives());
        };
        for (double step : {1e-3, 5e-4}) {
            auto plus = sample(step);
            check("nonlinear nonlocal XC kernel matches production covectors",
                  potential_response_error(response, plus, sample(-step), step), 2e-7);
        }
        check("XC kernel preserves potential and energy",
              xc_samples(*potential.lapw_xc_derivatives()) != saved || potential.energy_exc(density) != saved_energy
                      ? 1
                      : 0,
              0);
        ctx->cb().external_xc_          = functional;
        ctx->cb().external_xc_response_ = nullptr;
        reject("missing XC response callback rejected collectively",
               [&] { potential.lapw_external_xc_response(density, direction); });
        ctx->cb().external_xc_response_ = [](auto const& c, auto const& in, auto const& d, auto& out) {
            functional_response(c, in, d, out);
            if (c.comm().rank() == 0) {
                out.vrho[0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        reject("nonfinite XC response rejected collectively",
               [&] { potential.lapw_external_xc_response(density, direction); });
        ctx->cb().external_xc_response_ = [](auto const& c, auto const&, auto const&, auto&) {
            if (c.comm().rank() == 0) {
                RTE_THROW("XC response fixture failure");
            }
        };
        reject("XC response callback failure propagated collectively",
               [&] { potential.lapw_external_xc_response(density, direction); });
        ctx->cb().external_xc_response_ = functional_response;
        if (ctx->comm().rank() == 0) {
            direction.rho().mt()[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        reject("nonfinite XC MT direction rejected collectively",
               [&] { potential.lapw_external_xc_response(density, direction); });
    }
    Density regenerated(*ctx);
    copy(density, regenerated);
    regenerated.generate<double>(ks, false, true, true);
    double initial_field_error{0};
    for (int component = 0; component <= ctx->num_mag_dims(); ++component) {
        for (auto fields : {std::make_pair(&density.component(component), &regenerated.component(component)),
                            std::make_pair(&density.kinetic_density().component(component),
                                           &regenerated.kinetic_density().component(component))}) {
            for (size_t i = 0; i < fields.first->rg().f_pw_local().size(); ++i) {
                initial_field_error = std::max(initial_field_error, std::abs(fields.first->rg().f_pw_local(i) -
                                                                             fields.second->rg().f_pw_local(i)));
            }
            for (auto it : ctx->unit_cell().spl_num_atoms()) {
                for (size_t i = 0; i < fields.first->mt()[it.i].size(); ++i) {
                    initial_field_error = std::max(
                            initial_field_error, std::abs(fields.first->mt()[it.i][i] - fields.second->mt()[it.i][i]));
                }
            }
        }
    }
    check("initial fields regenerated from occupied orbitals", initial_field_error, 1e-12);
    // Isolate the tau guess from angular truncation of the matched atomic density.
    Density initial(*ctx);
    initial.zero();
    for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
        auto const& grid = ctx->unit_cell().atom(ia).type().radial_grid();
        for (int ir = 0; ir < grid.num_points(); ir++) {
            initial.rho().mt()[ia](0, ir) = std::exp(-2 * grid[ir]) / y00;
            if (mag) {
                initial.mag(0).mt()[ia](0, ir) = 0.2 * initial.rho().mt()[ia](0, ir);
            }
        }
    }
    initial.initial_kinetic_density();
    SHT sht(device_t::CPU, ctx->lmax_rho());
    double initial_violation{0};
    for (auto it : ctx->unit_cell().spl_num_atoms()) {
        for (int s = 0; s < ctx->num_spins(); s++) {
            auto n = 1.0 * initial.rho().mt()[it.i];
            auto t = 1.0 * initial.kinetic_density().scalar().mt()[it.i];
            if (mag) {
                for (size_t i = 0; i < n.size(); i++) {
                    n[i] = 0.5 * (n[i] + (1 - 2 * s) * initial.mag(0).mt()[it.i][i]);
                    t[i] = 0.5 * (t[i] + (1 - 2 * s) * initial.kinetic_density().vector(0).mt()[it.i][i]);
                }
            }
            auto real_n = transform(sht, n), real_t = transform(sht, t);
            auto grad = gradient(n);
            auto gx = transform(sht, grad[0]), gy = transform(sht, grad[1]), gz = transform(sht, grad[2]);
            for (size_t i = 0; i < real_n.size(); i++) {
                double sigma      = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
                double product    = 8 * real_n[i] * real_t[i];
                initial_violation = std::max(initial_violation, (sigma - product) / std::max({1.0, sigma, product}));
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&initial_violation, 1);
    check("initial density/tau Cauchy inequality", initial_violation, 1e-12);
    if (failures) {
        return failures;
    }
    auto check_energy = [&] {
        double energy = potential.energy_exc(density);
        auto c        = potential.lapw_xc_contractions(density);
        check("callback Euler identity", std::abs(c.rho + c.magnetic + c.tau - 2 * energy), 2e-11);
        check("nonzero tau contraction", std::abs(c.tau) > 1e-6 ? 0 : 1, 0);
        if (mag) {
            check("nonzero magnetic contraction", std::abs(c.magnetic) / functional_scale(*ctx) > 1e-6 ? 0 : 1, 0);
        }
        check("XC operator is staged", potential.lapw_xc_derivatives() ? 0 : 1, 0);
    };
    check_energy();
    reject("point potential cannot be symmetrized independently", [&] { potential.generate(density, true, true); });

    auto result = gs.find(1e-9, 1e-10, 1e-9, 100, false);
    check("SCF convergence", result.at("converged").get<bool>() ? 0 : 1, 0);
    check_energy();
    density.generate<double>(ks, false, true, true);
    potential.generate(density, false, true);
    check_energy();

    auto orbital_snapshot = [&] {
        std::vector<std::complex<double>> values;
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    auto const& wf = kp->spinor_wave_functions();
                    auto data      = wf.at(memory_t::host, 0, wf::spin_index(s), wf::band_index(b));
                    values.insert(values.end(), data, data + wf.ld());
                }
            }
        }
        return values;
    };
    auto basis_snapshot = [&] {
        std::vector<double> values;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx->unit_cell().atom(ia);
            for (auto rf : atom.type().indexr()) {
                for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                    values.push_back(atom.symmetry_class().radial_function(ir, rf.idxrf));
                }
            }
        }
        return values;
    };
    auto saved_orbitals = orbital_snapshot();
    auto saved_basis    = basis_snapshot();
    double saved_energy = gs.total_energy(), saved_exc = potential.energy_exc(density);
    reject("incomplete external LAPW forces remain guarded", [&] { gs.forces().calc_forces_total(); });
    reject("incomplete external LAPW stress remains guarded", [&] { gs.stress().calc_stress_total(); });
    auto verify_restored = [&] {
        check("orbital diagnostic restores PW/AW/LO coefficients", orbital_snapshot() == saved_orbitals ? 0 : 1, 0);
        check("orbital diagnostic holds radial basis fixed", basis_snapshot() == saved_basis ? 0 : 1, 0);
        check("orbital diagnostic preserves ground-state energy", std::abs(gs.total_energy() - saved_energy), 0);
        check("orbital diagnostic preserves ground-state XC", std::abs(potential.energy_exc(density) - saved_exc), 0);
    };
    auto derivatives = gs.check_external_xc_derivative(1e-4);
    for (int direction = 0; direction < 2; direction++) {
        auto const& values = derivatives[direction];
        check(direction == 0 ? "matched PW/AW orbital derivative" : "local-orbital derivative",
              std::abs(values[0] - values[1]), 2e-8);
        check("orbital derivative at halved step", std::abs(values[0] - values[2]), 6e-9);
        check("nonzero orbital derivative", std::abs(values[0]) / functional_scale(*ctx) > 1e-7 ? 0 : 1, 0);
    }
    verify_restored();
    for (double error : gs.check_lapw_xc_response(1e-3)) {
        check("field-response diagnostic", error, 1e-8);
    }
    verify_restored();
    int evaluations{0};
    ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
        functional(c, in, out);
        if (++evaluations == 2 && c.comm().rank() == 0) {
            throw std::runtime_error("deliberate LAPW derivative-check callback failure");
        }
    };
    reject("orbital diagnostic propagates callback failure", [&] { gs.check_external_xc_derivative(1e-4); });
    verify_restored();
    potential.generate(density, false, true);
    check("orbital diagnostic restores evaluator", evaluations == 3 ? 0 : 1, 0);
    ctx->cb().external_xc_ = [](auto const& c, auto const& in, auto& out) {
        functional(c, in, out);
        for (int j = 0; j < in.num_points * in.num_spins; j++) {
            out.vtau[j] = 0;
        }
    };
    auto incomplete = gs.check_external_xc_derivative(1e-4);
    double missing_tau{0};
    for (auto const& values : incomplete) {
        missing_tau = std::max(missing_tau, std::abs(values[0] - values[2]));
    }
    check("orbital diagnostic detects missing tau", missing_tau / functional_scale(*ctx) > 1e-7 ? 0 : 1, 0);
    ctx->cb().external_xc_ = functional;
    verify_restored();

    Density displaced(*ctx);
    Potential probe(*ctx);
    auto c = potential.lapw_xc_contractions(density);
    std::vector<std::vector<double>> spherical_before_fields;
    std::vector<std::array<std::vector<double>, 2>> core_xc_before_fields;
    for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
        auto const& cls = ctx->unit_cell().atom_symmetry_class(ic);
        spherical_before_fields.push_back(cls.spherical_potential());
        core_xc_before_fields.push_back(cls.spherical_xc_derivatives());
    }
    auto restore_field_matrices = [&] {
        potential.update_atomic_potential();
        for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
            auto& cls = ctx->unit_cell().atom_symmetry_class(ic);
            cls.set_spherical_potential(spherical_before_fields[ic]);
            cls.set_spherical_xc_derivatives(core_xc_before_fields[ic]);
        }
        ctx->unit_cell().generate_radial_integrals();
    };
    bool spectrum_checked{false};
    std::array<double, 4> spectrum_errors{};
    auto matrix_snapshot = [&](Potential& p, lapw_hamiltonian_response_t const* response) {
        Hamiltonian0<double> h0(p, false);
        h0.generate_pw_coefs(p);
        std::vector<std::complex<double>> entries;
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            auto hk = h0(*kp);
            int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
            la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
            la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
            for (int s = 0; s < ctx->num_spins(); s++) {
                if (response) {
                    hk.set_fv_h_field_response(s, *response, h);
                    if (!core && !second_variation && !spectrum_checked && s == 0) {
                        auto errors = lapw_spectrum_checks(hk, h);
                        for (size_t i = 0; i < errors.size(); i++) {
                            spectrum_errors[i] = std::max(spectrum_errors[i], errors[i]);
                        }
                    }
                } else {
                    hk.set_fv_h_o(s, h, o);
                }
                for (int j = 0; j < h.num_cols_local(); j++) {
                    for (int i = 0; i < h.num_rows_local(); i++) {
                        entries.push_back(h(i, j));
                    }
                }
            }
        }
        return entries;
    };
    for (int part = 0; part < ctx->num_spins() + 1; part++) {
        std::vector<std::complex<double>> analytic_matrices, sampled_matrices;
        if (!mag || !second_variation) {
            Density direction(*ctx);
            direction.zero();
            direction.kinetic_density().zero();
            if (part == 0) {
                copy(density.rho(), direction.rho());
            } else if (part == 1) {
                for (int j = 0; j < ctx->num_spins(); j++) {
                    copy(density.kinetic_density().component(j), direction.kinetic_density().component(j));
                }
            } else {
                copy(density.mag(0), direction.mag(0));
            }
            auto hartree = potential.hartree_response(direction.rho());
            auto xc      = potential.lapw_external_xc_response(density, direction);
            Hamiltonian0<double> h0(potential, false);
            auto response     = h0.lapw_field_response(*hartree, xc);
            analytic_matrices = matrix_snapshot(potential, &response);
            if (!core && !second_variation && !spectrum_checked) {
                check("complete-spectrum density-matrix response", spectrum_errors[0], 3e-7);
                check("degenerate-spectrum density-matrix response", spectrum_errors[1], 3e-7);
                check("generalized density response preserves the charge constraint", spectrum_errors[2], 3e-10);
                check("invalid spectrum-response inputs rejected collectively", spectrum_errors[3], 0);
                spectrum_checked = true;
            }
            if (!mag && !second_variation && part == 0) {
                auto saved = xc.local[0][1][0][0];
                if (ctx->comm().rank() == 0) {
                    xc.local[0][1][0][0] = std::numeric_limits<double>::quiet_NaN();
                }
                reject("invalid XC field matrix input rejected collectively",
                       [&] { h0.lapw_field_response(*hartree, xc); });
                xc.local[0][1][0][0] = saved;
                if (ctx->comm().rank() == 0) {
                    saved               = hartree->mt()[0][0];
                    hartree->mt()[0][0] = std::numeric_limits<double>::quiet_NaN();
                }
                reject("invalid Hartree field matrix input rejected collectively",
                       [&] { h0.lapw_field_response(*hartree, xc); });
                if (ctx->comm().rank() == 0) {
                    hartree->mt()[0][0] = saved;
                }
                auto invalid_energy_fields = potential.lapw_external_xc_response(density, direction);
                reject("nonfinite staged LAPW energy rejected collectively", [&] {
                    probe.set_lapw_xc_derivatives(std::move(invalid_energy_fields),
                                                  ctx->comm().rank() == 0 ? std::numeric_limits<double>::quiet_NaN()
                                                                          : 0.0);
                });
            }
            verify_restored();
        }
        auto evaluate = [&](double step) {
            copy(density, displaced);
            if (part == 0) {
                displaced.rho() *= 1 + step;
            } else if (part == 1) {
                for (int j = 0; j < ctx->num_spins(); j++) {
                    displaced.kinetic_density().component(j) *= 1 + step;
                }
            } else {
                displaced.mag(0) *= 1 + step;
            }
            for (int j = 0; j < ctx->num_spins(); j++) {
                displaced.component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
                displaced.kinetic_density().component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
            }
            displaced.fft_transform(-1);
            displaced.kinetic_density().fft_transform(-1);
            if (!mag || !second_variation) {
                try {
                    probe.generate(displaced, false, true);
                    probe.update_atomic_potential();
                    ctx->unit_cell().generate_radial_integrals();
                    sampled_matrices = matrix_snapshot(probe, nullptr);
                } catch (...) {
                    restore_field_matrices();
                    throw;
                }
                restore_field_matrices();
            } else {
                probe.xc(displaced);
            }
            return probe.energy_exc(displaced);
        };
        // The quadratic fixture has an exactly linear Hamiltonian response. A larger
        // spin-only step resolves the weak core-fixture signal above subtraction roundoff.
        double field_step = core && part == 2 ? 0.1 : 1e-4;
        for (double h : {field_step, 0.5 * field_step}) {
            double plus        = evaluate(h);
            auto plus_matrices = sampled_matrices;
            double fd          = (plus - evaluate(-h)) / (2 * h);
            if (!mag || !second_variation) {
                double error{0}, peak{0};
                for (size_t i = 0; i < analytic_matrices.size(); i++) {
                    double e = std::abs(analytic_matrices[i] - (plus_matrices[i] - sampled_matrices[i]) / (2 * h));
                    error    = std::isfinite(e) ? std::max(error, e) : std::numeric_limits<double>::infinity();
                    peak     = std::max(peak, std::abs(analytic_matrices[i]));
                }
                ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                if (error > 3e-7 * std::max(1e-12, peak) && ctx->comm().rank() == 0) {
                    std::cout << "field matrix FD part=" << part << " step=" << h << " error=" << error
                              << " response=" << peak << '\n';
                }
                check("complete Hartree/XC Hamiltonian field response", error / std::max(1e-12, peak), 3e-7);
            }
            check("callback physical-field derivative",
                  std::abs(fd - (part == 0   ? c.rho
                                 : part == 1 ? c.tau
                                             : c.magnetic)),
                  2e-10);
        }
        verify_restored();
    }
    double kinetic = density.kinetic_density().scalar().integrate().total;
    double direct =
            kinetic + potential.energy_exc(density) + 0.5 * potential.energy_vha() + energy_enuc(*ctx, potential);
    double total = ks_energy(*ctx, ks, density, potential);
    check("kinetic energy from physical orbital tau", std::abs(kinetic - energy_kin(*ctx, ks, density, potential)),
          2e-7);
    check("direct total-energy identity", std::abs(total - direct), 2e-7);
    if (ctx->comm().rank() == 0) {
        std::cout << "external LAPW total=" << total << " direct=" << direct << " kinetic=" << kinetic << '\n';
    }
    if (core) {
        auto vs      = potential.get_spherical_potential();
        auto d       = potential.get_core_xc_derivatives();
        auto old_tau = density.core_kinetic_density(0);
        density.generate_core_charge_density(vs, d);
        auto tau    = density.core_kinetic_density(0);
        double peak = *std::max_element(tau.begin(), tau.end()), residual{0}, response{0};
        for (size_t ir = 0; ir < tau.size(); ir++) {
            residual = std::max(residual, std::abs(tau[ir] - old_tau[ir]) / peak);
        }
        check("self-consistent core tau residual", residual, 2e-6);
        check("core leakage", std::abs(density.core_leakage()), 1e-6);
        double derivative{0};
        auto const& cls = ctx->unit_cell().atom(0).symmetry_class();
        for (size_t ir = 0; ir < tau.size(); ir++) {
            derivative += density.core_charge_density(0)[ir] * d[cls.id()][0][ir] + tau[ir] * d[cls.id()][1][ir];
        }
        for (double h : {1e-3, 5e-4}) {
            auto energy = [&](double step) {
                auto shifted = d;
                for (auto& atom : shifted) {
                    for (auto& field : atom) {
                        for (auto& value : field) {
                            value *= 1 + step;
                        }
                    }
                }
                density.generate_core_charge_density(vs, shifted);
                return density.core_eval_sum();
            };
            double fd = (energy(h) - energy(-h)) / (2 * h);
            check("callback-to-core eigenvalue derivative", std::abs(fd - derivative), 2e-7);
        }
        density.generate_core_charge_density(vs);
        for (size_t ir = 0; ir < tau.size(); ir++) {
            response = std::max(response, std::abs(tau[ir] - density.core_kinetic_density(0)[ir]) / peak);
        }
        check("nonzero tau-dependent core response", response > 1e-5 ? 0 : 1, 0);
        density.generate_core_charge_density(vs, d);
        saved_energy = gs.total_energy();
    }
    if (!core && (!second_variation || !mag)) {
        auto const& forces = gs.forces().calc_forces_ibs();
        r3::vector<double> direction{0.23, -0.37, 0.19};
        auto position = ctx->unit_cell().atom(0).position();
        double analytic{0};
        for (int x = 0; x < 3; x++) {
            analytic -= direction[x] * forces(x, 0);
        }
        // Hold radial functions, local Hamiltonians, Fourier potentials and orbitals
        // fixed. Only matching coefficients and the interstitial boundary move.
        Hamiltonian0<double> h0(potential, false);
        h0.generate_pw_coefs(potential);
        auto evaluate = [&](double step) {
            auto shifted = position;
            for (int x = 0; x < 3; x++) {
                shifted[x] += step * direction[x] / ctx->unit_cell().lattice_vector(x)[x];
            }
            ctx->unit_cell().atom(0).set_position(shifted);
            double value{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto kp = ks.get<double>(it.i);
                kp->update();
                auto hk = h0(*kp);
                int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                for (int s = 0; s < ctx->num_spins(); s++) {
                    hk.set_fv_h_o(s, h, o);
                    // Translate the sphere's Fourier coefficients without rebuilding
                    // the context, which would discard the frozen radial basis.
                    for (int j = 0; j < kp->num_gkvec_col(); j++) {
                        auto qj = kp->gkvec_col().gkvec_cart(gvec_index_t::local(j));
                        for (int i = 0; i < kp->num_gkvec_row(); i++) {
                            auto qi    = kp->gkvec_row().gkvec_cart(gvec_index_t::local(i));
                            auto g     = qi - qj;
                            auto r     = ctx->unit_cell().get_cartesian_coordinates(position);
                            auto phase = std::exp(std::complex<double>(0, -dot(g, r)));
                            auto delta =
                                    -fourpi / ctx->unit_cell().omega() *
                                    unit_step_function_form_factors(ctx->unit_cell().atom(0).mt_radius(), g.length()) *
                                    phase * (std::exp(std::complex<double>(0, -step * dot(g, direction))) - 1.0);
                            o(i, j) += delta;
                            h(i, j) += 0.5 * dot(qi, qj) * delta;
                        }
                    }
                    value += occupied_lapw_lagrangian(*ctx, *kp, s, h, o);
                }
            }
            ctx->comm().allreduce(&value, 1);
            return value;
        };
        for (double step : {1e-3, 5e-4}) {
            double fd = (evaluate(step) - evaluate(-step)) / (2 * step);
            check("LAPW basis/boundary force", std::abs(fd - analytic), 2e-8);
            check("relative basis/boundary force error", std::abs((fd - analytic) / analytic), 2e-5);
        }
        evaluate(0);
        check("nonzero basis/boundary force", std::abs(analytic) > 1e-7 ? 0 : 1, 0);
        verify_restored();
    }
    if (!second_variation || !mag) {
        std::vector<std::vector<double>> spherical;
        std::vector<std::array<std::vector<double>, 2>> spherical_xc;
        for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& cls = ctx->unit_cell().atom_symmetry_class(ic);
            spherical.push_back(cls.spherical_potential());
            spherical_xc.push_back(cls.spherical_xc_derivatives());
        }
        std::vector<int> nmt;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            nmt.push_back(ctx->unit_cell().atom(ia).type().mt_basis_size());
        }
        std::vector<std::unique_ptr<wf::Wave_functions<double>>> zero_orbitals;
        std::vector<wf::Wave_functions<double>*> orbital_directions;
        for (auto it : ks.spl_num_kpoints()) {
            zero_orbitals.push_back(std::make_unique<wf::Wave_functions<double>>(
                    ks.get<double>(it.i)->gkvec_sptr(), nmt, wf::num_mag_dims(ctx->num_mag_dims()),
                    wf::num_bands(ctx->num_bands()), memory_t::host));
            zero_orbitals.back()->zero(memory_t::host);
            orbital_directions.push_back(zero_orbitals.back().get());
        }
        std::vector<mdarray<double, 2>> occupations, occupation_directions;
        for (auto it : ks.spl_num_kpoints()) {
            auto const& kp = *ks.get<double>(it.i);
            for (auto* values : {&occupations, &occupation_directions}) {
                values->emplace_back(std::array<index_range, 2>{ctx->num_bands(), ctx->num_spins()});
            }
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    occupations.back()(b, s)           = kp.band_occupancy(b, s);
                    occupation_directions.back()(b, s) = b == ctx->num_bands() - 1
                                                                 ? 0.031 * (s + 1)
                                                                 : 0.02 * std::cos(b + s) * kp.band_occupancy(b, s);
                }
            }
        }
        auto occupation_response =
                density.lapw_orbital_fields_response(ks, orbital_directions, {}, occupation_directions);
        auto set_occupations = [&](double step) {
            size_t ik{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto& kp = *ks.get<double>(it.i);
                for (int s = 0; s < ctx->num_spins(); s++) {
                    for (int b = 0; b < ctx->num_bands(); b++) {
                        kp.band_occupancy(b, s, occupations[ik](b, s) + step * occupation_directions[ik](b, s));
                    }
                }
                ik++;
            }
        };
        Density plus(*ctx), minus(*ctx);
        constexpr double occupation_step{1e-4};
        try {
            set_occupations(occupation_step);
            plus.generate<double>(ks, false, false, true);
            set_occupations(-occupation_step);
            minus.generate<double>(ks, false, false, true);
        } catch (...) {
            set_occupations(0);
            throw;
        }
        set_occupations(0);
        double occupation_error{0}, occupation_peak{0}, occupation_relative{0};
        auto compare_occupation = [&](auto r, auto p, auto m) {
            double error = std::abs(r - (p - m) / (2 * occupation_step)), peak = std::abs(r);
            if (!std::isfinite(error) || !std::isfinite(peak)) {
                occupation_error = std::numeric_limits<double>::infinity();
            } else {
                occupation_error = std::max(occupation_error, error);
                occupation_peak  = std::max(occupation_peak, peak);
            }
        };
        for (int p : {0, 1}) {
            for (int j = 0; j < ctx->num_spins(); j++) {
                auto const& r = p == 0 ? occupation_response->component(j)
                                       : occupation_response->kinetic_density().component(j);
                auto const& u = p == 0 ? plus.component(j) : plus.kinetic_density().component(j);
                auto const& v = p == 0 ? minus.component(j) : minus.kinetic_density().component(j);
                for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                    compare_occupation(r.rg().f_pw_local(ig), u.rg().f_pw_local(ig), v.rg().f_pw_local(ig));
                }
                for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                    compare_occupation(r.rg().value(ir), u.rg().value(ir), v.rg().value(ir));
                }
                for (auto it : ctx->unit_cell().spl_num_atoms()) {
                    for (size_t i = 0; i < r.mt()[it.i].size(); i++) {
                        compare_occupation(r.mt()[it.i][i], u.mt()[it.i][i], v.mt()[it.i][i]);
                    }
                }
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&occupation_error, 1);
            ctx->comm().allreduce<double, mpi::op_t::max>(&occupation_peak, 1);
            occupation_relative = std::max(occupation_relative, occupation_error / std::max(1e-12, occupation_peak));
            check("nonzero occupation response", occupation_peak > 1e-8 ? 0 : 1, 0);
            occupation_error = occupation_peak = 0;
        }
        check("occupation rho/tau response including empty bands", occupation_relative, 1e-8);
        verify_restored();
        for (int part : core ? std::vector<int>{0, 1, 2, 3} : std::vector<int>{0}) {
            auto radial_direction = spherical;
            std::vector<std::array<std::vector<double>, 2>> xc_direction(spherical.size());
            for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                auto const& grid = ctx->unit_cell().atom_symmetry_class(ic).atom_type().radial_grid();
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    // Resolve core response rather than subtracting a nearly constant potential shift.
                    radial_direction[ic][ir] = (part == 0 || part == 3)
                                                       ? (ic + 1) * (0.13 * std::exp(-(core ? 3.0 : 0.3) * grid[ir]) +
                                                                     0.04 * std::sin(grid[ir]))
                                                       : 0;
                }
                for (int j : {0, 1}) {
                    if (part == j + 1 || part == 3) {
                        xc_direction[ic][j] = spherical_xc[ic][j];
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            // Resolve both channels while keeping the stronger tau response linear.
                            xc_direction[ic][j][ir] *= (j == 0 ? 1000 : 10) * std::cos(grid[ir]);
                        }
                    }
                }
            }
            std::cout << "Radial basis Lagrangian response part=" << part << '\n';
            double basis_derivative = gs.forces().lapw_radial_basis_derivative(radial_direction, xc_direction);
            std::vector<radial_basis_response_t> radial_response;
            for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                radial_response.push_back(ctx->unit_cell().atom_symmetry_class(ic).radial_basis_potential_response(
                        radial_direction[ic], xc_direction[ic]));
            }
            Hamiltonian0<double> basis_h0(potential, false);
            auto atom_matrices = basis_h0.lapw_radial_basis_response(radial_response);
            std::array<std::vector<std::complex<double>>, 2> analytic_matrices, sampled_matrices;
            auto append_matrices = [](auto const& h, auto const& o, auto& values) {
                for (int j = 0; j < h.num_cols_local(); j++) {
                    for (int i = 0; i < h.num_rows_local(); i++) {
                        values[0].push_back(h(i, j));
                        values[1].push_back(o(i, j));
                    }
                }
            };
            int rejected_matrix_failures{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto kp = ks.get<double>(it.i);
                auto hk = basis_h0(*kp);
                int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                la::dmatrix<std::complex<double>> dh(n, n, ctx->blacs_grid(), bs, bs);
                la::dmatrix<std::complex<double>> ds(n, n, ctx->blacs_grid(), bs, bs);
                for (int s = 0; s < ctx->num_spins(); s++) {
                    hk.set_fv_h_o_radial_response(s, radial_response, atom_matrices, dh, ds);
                    append_matrices(dh, ds, analytic_matrices);
                }
                if (!mag && !second_variation && part == 0) {
                    auto expect_rejection = [&](auto&& operation) {
                        bool rejected{false};
                        try {
                            operation();
                        } catch (std::exception const&) {
                            rejected = true;
                        }
                        rejected_matrix_failures += !rejected;
                    };
                    expect_rejection([&] {
                        hk.set_fv_h_o_radial_response(kp->comm().rank() == 0 ? -1 : 0, radial_response, atom_matrices,
                                                      dh, ds);
                    });
                    la::dmatrix<std::complex<double>> no_grid(n, n);
                    expect_rejection(
                            [&] { hk.set_fv_h_o_radial_response(0, radial_response, atom_matrices, no_grid, ds); });
                    expect_rejection([&] { hk.set_fv_h_o_radial_response(0, radial_response, atom_matrices, dh, dh); });
                    auto saved = atom_matrices[0].h[0];
                    if (kp->comm().rank() == 0) {
                        atom_matrices[0].h[0] = std::numeric_limits<double>::quiet_NaN();
                    }
                    expect_rejection([&] { hk.set_fv_h_o_radial_response(0, radial_response, atom_matrices, dh, ds); });
                    atom_matrices[0].h[0] = saved;
                }
            }
            if (!mag && !second_variation && part == 0) {
                check("invalid radial matrix inputs rejected collectively", rejected_matrix_failures, 0);
            }
            auto field_response = density.lapw_radial_fields_response(ks, radial_response);
            auto combined       = density.lapw_orbital_fields_response(ks, orbital_directions, radial_response);
            double combined_error{0};
            for (int p : {0, 1}) {
                auto const& field = p == 0 ? static_cast<Field4D const&>(*combined) : combined->kinetic_density();
                for (int s = 0; s < ctx->num_spins(); s++) {
                    for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                        combined_error = std::max(combined_error, std::abs(field.component(s).rg().f_pw_local(ig)));
                    }
                }
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    for (size_t i = 0; i < field.scalar().mt()[ia].size(); i++) {
                        double up      = field_response[ia][p][0][i];
                        double dn      = mag ? field_response[ia][p][1][i] : 0;
                        combined_error = std::max(combined_error, std::abs(field.scalar().mt()[ia][i] - (up + dn)));
                        if (mag) {
                            combined_error =
                                    std::max(combined_error, std::abs(field.vector(0).mt()[ia][i] - (up - dn)));
                        }
                    }
                }
            }
            check("combined orbital/radial fields response", combined_error, 1e-13);
            auto hartree_response = radial_hartree_response(*ctx, potential, field_response);
            auto xc_response      = radial_xc_response(*ctx, potential, density, field_response);
            verify_restored();
            if (!mag && !second_variation && part == 0) {
                auto invalid_orbitals = orbital_directions;
                if (ctx->comm().rank() == 0 && !invalid_orbitals.empty()) {
                    invalid_orbitals[0] = nullptr;
                }
                reject("invalid orbital direction rejected collectively",
                       [&] { density.lapw_orbital_fields_response(ks, invalid_orbitals); });
                std::vector<radial_basis_response_t> zero;
                for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                    zero.emplace_back(ctx->unit_cell().atom_symmetry_class(ic).atom_type());
                }
                double zero_peak{0};
                for (auto const& atom : density.lapw_radial_fields_response(ks, zero)) {
                    for (auto const& channel : atom) {
                        for (auto const& spin : channel) {
                            for (size_t i = 0; i < spin.size(); i++) {
                                zero_peak = std::max(zero_peak, std::abs(spin[i]));
                            }
                        }
                    }
                }
                check("zero radial field response", zero_peak, 0);
                if (ctx->comm().rank() == 0) {
                    zero[0].surface_derivatives[0] = std::numeric_limits<double>::quiet_NaN();
                }
                reject("nonfinite field response rejected collectively",
                       [&] { density.lapw_radial_fields_response(ks, zero); });
                reject("nonfinite atom matrix response rejected collectively",
                       [&] { basis_h0.lapw_radial_basis_response(zero); });
                zero[0].surface_derivatives[0] = 0;
                if (ctx->comm().rank() == 0) {
                    zero[0].core_fields[0].pop_back();
                }
                reject("invalid core field response rejected collectively",
                       [&] { density.lapw_radial_fields_response(ks, zero); });
                auto direction = radial_direction;
                for (auto& atom : direction) {
                    std::fill(atom.begin(), atom.end(), 0);
                }
                check("zero radial potential direction", std::abs(gs.forces().lapw_radial_basis_derivative(direction)),
                      0);
                if (ctx->comm().rank() == 0) {
                    direction[0].pop_back();
                }
                reject("invalid radial direction rejected collectively",
                       [&] { gs.forces().lapw_radial_basis_derivative(direction); });
                for (int j : {0, 1}) {
                    auto invalid_xc = xc_direction;
                    if (ctx->comm().rank() == 0) {
                        invalid_xc[0][j] = {std::numeric_limits<double>::quiet_NaN()};
                    }
                    reject("invalid radial XC direction rejected collectively",
                           [&] { gs.forces().lapw_radial_basis_derivative(radial_direction, invalid_xc); });
                }
                verify_restored();
            }
            std::vector<double> sampled_fields, sampled_hartree, sampled_xc;
            auto radial_lagrangian = [&](double step) {
                auto shifted_vs = spherical;
                auto shifted_xc = spherical_xc;
                for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                    auto v  = spherical[ic];
                    auto xc = spherical_xc[ic];
                    for (size_t ir = 0; ir < v.size(); ir++) {
                        v[ir] += step * radial_direction[ic][ir];
                        for (int j : {0, 1}) {
                            if (!xc_direction[ic][j].empty()) {
                                xc[j][ir] += step * xc_direction[ic][j][ir];
                            }
                        }
                    }
                    auto& cls = ctx->unit_cell().atom_symmetry_class(ic);
                    cls.set_spherical_potential(v);
                    cls.set_spherical_xc_derivatives(xc);
                    shifted_vs[ic] = v;
                    shifted_xc[ic] = xc;
                    if (cls.generate_radial_functions(relativity_t::none, false)) {
                        RTE_THROW("radial basis derivative fixture failed");
                    }
                    // Only the basis changes, not the physical Hamiltonian.
                    cls.set_spherical_potential(spherical[ic]);
                    cls.set_spherical_xc_derivatives(spherical_xc[ic]);
                }
                ctx->unit_cell().generate_radial_integrals();
                Hamiltonian0<double> radial_h0(potential, false);
                radial_h0.generate_pw_coefs(potential);
                double value{0};
                for (auto& entries : sampled_matrices) {
                    entries.clear();
                }
                for (auto it : ks.spl_num_kpoints()) {
                    auto kp = ks.get<double>(it.i);
                    auto hk = radial_h0(*kp);
                    int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                    la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                    for (int s = 0; s < ctx->num_spins(); s++) {
                        hk.set_fv_h_o(s, h, o);
                        append_matrices(h, o, sampled_matrices);
                        value += occupied_lapw_lagrangian(*ctx, *kp, s, h, o);
                    }
                }
                ctx->comm().allreduce(&value, 1);
                sampled_fields = sample_radial_fields(*ctx, ks, shifted_vs, shifted_xc, &sampled_hartree, &sampled_xc);
                return value;
            };
            for (double step : {1e-3, 5e-4}) {
                double plus        = radial_lagrangian(step);
                auto plus_fields   = sampled_fields;
                auto plus_hartree  = sampled_hartree;
                auto plus_xc       = sampled_xc;
                auto plus_matrices = sampled_matrices;
                double fd          = (plus - radial_lagrangian(-step)) / (2 * step);
                for (int p : {0, 1}) {
                    double error{0}, peak{0};
                    for (size_t i = 0; i < analytic_matrices[p].size(); i++) {
                        auto analytic = analytic_matrices[p][i];
                        double e = std::abs(analytic - (plus_matrices[p][i] - sampled_matrices[p][i]) / (2 * step));
                        error    = std::isfinite(e) ? std::max(error, e) : std::numeric_limits<double>::infinity();
                        peak     = std::max(peak, std::abs(analytic));
                    }
                    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                    check(p == 0 ? "complete radial Hamiltonian response" : "complete radial overlap response",
                          error / std::max(1e-12, peak), 3e-5);
                }
                check("radial density/tau response",
                      radial_fields_error(field_response, plus_fields, sampled_fields, step), 3e-5);
                check("radial Hartree response",
                      potential_response_error(hartree_response, plus_hartree, sampled_hartree, step), 3e-5);
                check("radial XC covector response", potential_response_error(xc_response, plus_xc, sampled_xc, step),
                      3e-5);
                check("radial basis Lagrangian derivative", std::abs(fd - basis_derivative), 2e-8);
                check("relative radial basis Lagrangian derivative",
                      std::abs((fd - basis_derivative) / basis_derivative), 2e-5);
            }
            radial_lagrangian(0);
            check("nonzero radial basis Lagrangian derivative", std::abs(basis_derivative) > 1e-7 ? 0 : 1, 0);
            verify_restored();
        }
    }
    return failures;
}

int
test_lapw_force_response(int mag, double pw_cutoff)
{
    std::vector<std::vector<double>> frozen_potential;
    std::vector<std::vector<double>> displaced_potential;
    struct state_t
    {
        std::unique_ptr<Simulation_context> ctx;
        std::unique_ptr<K_point_set> ks;
        std::unique_ptr<DFT_ground_state> gs;
    };
    std::unique_ptr<state_t> reference;
    // The displaced SCF supplies the radial potential response. This decomposition
    // check does not replace the missing analytic response in the production force.
    auto radial_lagrangian = [&](std::vector<std::vector<double>> const& potential) {
        auto& ctx = *reference->ctx;
        auto& ks  = *reference->ks;
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_potential(potential[ic]);
        }
        ctx.unit_cell().generate_radial_functions(ctx.out());
        // Vary the basis, not the physical Hamiltonian or the reference orbitals.
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_potential(frozen_potential[ic]);
        }
        ctx.unit_cell().generate_radial_integrals();
        Hamiltonian0<double> h0(reference->gs->potential(), false);
        h0.generate_pw_coefs(reference->gs->potential());
        double value{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            auto hk = h0(*kp);
            int n = kp->gklo_basis_size(), bs = ctx.cyclic_block_size();
            la::dmatrix<std::complex<double>> h(n, n, ctx.blacs_grid(), bs, bs);
            la::dmatrix<std::complex<double>> o(n, n, ctx.blacs_grid(), bs, bs);
            for (int s = 0; s < ctx.num_spins(); s++) {
                hk.set_fv_h_o(s, h, o);
                value += occupied_lapw_lagrangian(ctx, *kp, s, h, o);
            }
        }
        ctx.comm().allreduce(&value, 1);
        return value;
    };
    auto calculate = [&](double displacement, bool forces, bool frozen) {
        auto state                      = std::make_unique<state_t>();
        state->ctx                      = context(mag, true, false, false, false, displacement, pw_cutoff);
        auto& ctx                       = state->ctx;
        ctx->cb().external_xc_geometry_ = functional;
        // This diagnostic keeps the Cartesian grid, weights and block centres fixed.
        ctx->cb().external_xc_pullback_ = [](auto const&, auto const&, auto const&, auto&) {};
        state->ks                       = std::make_unique<K_point_set>(*ctx);
        auto& ks                        = *state->ks;
        ks.add_kpoint({0.07, -0.03, 0.02}, 1);
        ks.initialize();
        state->gs = std::make_unique<DFT_ground_state>(ks);
        auto& gs  = *state->gs;
        gs.initial_state();
        bool converged{false};
        if (!frozen) {
            converged = gs.find(1e-11, 1e-12, 1e-11, 200, false).at("converged").get<bool>();
            if (forces) {
                frozen_potential = gs.potential().get_spherical_potential();
            }
        } else {
            for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                ctx->unit_cell().atom_symmetry_class(ic).set_spherical_potential(frozen_potential[ic]);
            }
            ctx->unit_cell().generate_radial_functions(ctx->out());
            gs.density().mixer_init(ctx->cfg().mixer());
            double previous{0};
            for (int iter = 0; iter < 200; iter++) {
                Hamiltonian0<double> h0(gs.potential(), true, false);
                sirius::diagonalize<double, double>(h0, ks, 1e-11, ctx->cfg().iterative_solver().num_steps());
                ks.find_band_occupancies<double>();
                gs.density().generate<double>(ks, false, true, true);
                double rms = gs.density().mix();
                gs.potential().generate(gs.density(), false, true);
                double energy = free_energy(*ctx, ks, gs.density(), gs.potential());
                if (rms < 1e-11 && std::abs(energy - previous) < 1e-12) {
                    converged = true;
                    break;
                }
                previous = energy;
            }
        }
        if (!converged) {
            RTE_THROW("LAPW force SCF did not converge");
        }
        double derivative{0};
        if (forces) {
            auto const& ibs = gs.forces().calc_forces_ibs();
            auto const& hf  = gs.forces().calc_forces_hf();
            auto const& rho = gs.forces().calc_forces_rho();
            auto xc         = gs.potential().external_xc_geometry(gs.density());
            r3::vector<double> direction{0.23, -0.37, 0.19};
            for (int x = 0; x < 3; x++) {
                derivative += direction[x] * (xc.atoms[0][x] - ibs(x, 0) - hf(x, 0) - rho(x, 0));
                if (ctx->comm().rank() == 0) {
                    std::cout << "LAPW force component " << x << " ibs=" << ibs(x, 0) << " hf=" << hf(x, 0)
                              << " rho=" << rho(x, 0) << " xc=" << -xc.atoms[0][x] << '\n';
                }
            }
        }
        double direct = gs.density().kinetic_density().scalar().integrate().total +
                        gs.potential().energy_exc(gs.density()) + 0.5 * gs.potential().energy_vha() +
                        energy_enuc(*ctx, gs.potential()) + ks.entropy_sum();
        std::array<double, 4> result{free_energy(*ctx, ks, gs.density(), gs.potential()), derivative, direct, 0};
        if (forces) {
            reference = std::move(state);
        } else if (!frozen) {
            displaced_potential = gs.potential().get_spherical_potential();
            result[3]           = radial_lagrangian(displaced_potential);
        }
        return result;
    };
    int failures{0};
    for (bool frozen : {false, true}) {
        auto centre = calculate(0, true, frozen);
        for (double h : {2e-4, 1e-4}) {
            auto plus       = calculate(h, false, frozen);
            auto direction  = displaced_potential;
            auto minus      = calculate(-h, false, frozen);
            double fd       = (plus[0] - minus[0]) / (2 * h);
            double response = (plus[3] - minus[3]) / (2 * h);
            double analytic_response{0};
            if (!frozen) {
                for (size_t ic = 0; ic < direction.size(); ic++) {
                    for (size_t ir = 0; ir < direction[ic].size(); ir++) {
                        direction[ic][ir] = (direction[ic][ir] - displaced_potential[ic][ir]) / (2 * h);
                    }
                }
                radial_lagrangian(frozen_potential);
                analytic_response = reference->gs->forces().lapw_radial_basis_derivative(direction);
            }
            double error          = std::abs(fd - centre[1] - response);
            double analytic_error = std::abs(fd - centre[1] - analytic_response);
            if (mpi::Communicator::world().rank() == 0) {
                std::cout << "LAPW force decomposition spin=" << mag << " pw cutoff=" << pw_cutoff
                          << " frozen radial basis=" << frozen << " h=" << h << " derivative=" << centre[1]
                          << " FD=" << fd << " direct FD=" << (plus[2] - minus[2]) / (2 * h)
                          << " energy identity=" << centre[0] - centre[2] << " radial response=" << response
                          << " analytic radial response=" << analytic_response << " error=" << error
                          << " analytic error=" << analytic_error << '\n';
            }
            failures += !std::isfinite(error) || error > 2e-6;
            failures += !std::isfinite(analytic_error) || analytic_error > 2e-6;
            failures += std::abs(response - analytic_response) > 2e-7;
            failures += !frozen && std::abs(response) < 1e-5;
            for (auto const& point : {centre, plus, minus}) {
                failures += !std::isfinite(point[0] - point[2]) || std::abs(point[0] - point[2]) > 2e-7;
            }
            failures += std::abs(fd - (plus[2] - minus[2]) / (2 * h)) > 2e-7;
        }
    }
    return failures;
}

int
test_lapw_linearization_response()
{
    auto ctx = context(0, mpi::Communicator::world().size() > 1, false, false, false, 0, 5.0, false, 1);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    gs.potential().update_atomic_potential();
    ctx->unit_cell().generate_radial_functions(ctx->out());
    ctx->unit_cell().generate_radial_integrals();
    auto& cls        = ctx->unit_cell().atom_symmetry_class(0);
    auto base        = cls.spherical_potential();
    auto const& grid = cls.atom_type().radial_grid();
    std::vector<double> direction(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
    }
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        std::cout << name << ": " << error << " (limit " << tolerance << ")\n";
        failures += !std::isfinite(error) || error > tolerance;
    };
    // Test moving and frozen linearisation energies against the same occupied-state contraction.
    for (bool update_enu : {false, true}) {
        double analytic = gs.forces().lapw_radial_basis_derivative({direction}, {}, update_enu);
        std::vector<radial_basis_response_t> radial_response;
        radial_response.push_back(cls.radial_basis_potential_response(direction, {}, update_enu));
        auto field_response   = gs.density().lapw_radial_fields_response(ks, radial_response);
        auto hartree_response = radial_hartree_response(*ctx, gs.potential(), field_response);
        auto xc_response      = radial_xc_response(*ctx, gs.potential(), gs.density(), field_response);
        std::vector<double> sampled_fields, sampled_hartree, sampled_xc;
        auto sample = [&](double step) {
            auto v = base;
            for (size_t ir = 0; ir < v.size(); ir++) {
                v[ir] += step * direction[ir];
            }
            cls.set_spherical_potential(v);
            if (cls.generate_radial_functions(relativity_t::none, update_enu)) {
                RTE_THROW("linearisation response fixture failed");
            }
            cls.set_spherical_potential(base);
            ctx->unit_cell().generate_radial_integrals();
            Hamiltonian0<double> h0(gs.potential(), false);
            h0.generate_pw_coefs(gs.potential());
            double value{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto kp = ks.get<double>(it.i);
                auto hk = h0(*kp);
                int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                hk.set_fv_h_o(0, h, o);
                value += occupied_lapw_lagrangian(*ctx, *kp, 0, h, o);
            }
            ctx->comm().allreduce(&value, 1);
            sampled_fields = sample_radial_fields(*ctx, ks, {v}, gs.potential().get_core_xc_derivatives(),
                                                  &sampled_hartree, &sampled_xc);
            return value;
        };
        std::cout << "LAPW Lagrangian update_enu=" << update_enu << '\n';
        for (double step : {1e-2, 5e-3}) {
            double plus       = sample(step);
            auto plus_fields  = sampled_fields;
            auto plus_hartree = sampled_hartree;
            auto plus_xc      = sampled_xc;
            double fd         = (plus - sample(-step)) / (2 * step);
            check("automatic-energy density/tau response",
                  radial_fields_error(field_response, plus_fields, sampled_fields, step), 3e-5);
            check("automatic-energy Hartree response",
                  potential_response_error(hartree_response, plus_hartree, sampled_hartree, step), 3e-5);
            check("automatic-energy XC covector response",
                  potential_response_error(xc_response, plus_xc, sampled_xc, step), 3e-5);
            check("automatic-energy Lagrangian derivative", std::abs(fd - analytic), 2e-7);
            check("relative automatic-energy Lagrangian derivative", std::abs((fd - analytic) / analytic), 2e-5);
        }
        check("nonzero automatic-energy Lagrangian derivative", std::abs(analytic) > 1e-5 ? 0 : 1, 0);
        sample(0);
    }
    return failures;
}

int
test_core_guard()
{
    auto ctx = context(0, false, true, true, true);
    int failures{0};
    for (bool density : {false, true}) {
        bool rejected{false};
        try {
            if (density) {
                Density d(*ctx);
            } else {
                Potential p(*ctx);
            }
        } catch (std::exception const& e) {
            rejected = std::string(e.what()).find("tau-dependent radial core response") != std::string::npos;
        }
        failures += !rejected;
    }
    return failures;
}

int
test_replicated_eigenvector_remap()
{
    auto ctx = context(0, true, true);
    double error{0};
    for (auto counts : {std::vector<int>{}, std::vector<int>{0, 0, 0, 2, 0, 1}}) {
        int ng  = ctx->gvec().num_gvec();
        int nlo = std::accumulate(counts.begin(), counts.end(), 0);
        la::dmatrix<std::complex<double>> matrix(ng + nlo, 3);
        auto value = [](int row, int band) { return std::complex<double>(row + 0.3 * band, 0.2 * row - band); };
        for (int b = 0; b < 3; b++) {
            for (int i = 0; i < ng + nlo; i++) {
                matrix(i, b) = ctx->comm().rank() == 0 ? value(i, b) : std::complex<double>(-9, -7);
            }
        }
        wf::Wave_functions<double> slab(ctx->gvec_sptr(), counts, wf::num_mag_dims(0), wf::num_bands(3),
                                        memory_t::host);
        remap_lapw_evec_to_slab(ng, nlo, 3, matrix, slab, ctx->comm());
        for (int b = 0; b < 3; b++) {
            for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                error = std::max(error, std::abs(slab.pw_coeffs(ig, wf::spin_index(0), wf::band_index(b)) -
                                                 value(ctx->gvec().offset() + ig, b)));
            }
            for (auto atom : slab.spl_num_atoms()) {
                int offset = ng + std::accumulate(counts.begin(), counts.begin() + atom.i, 0);
                for (int xi = 0; xi < counts[atom.i]; xi++) {
                    error = std::max(error, std::abs(slab.mt_coeffs(xi, atom.li, wf::spin_index(0), wf::band_index(b)) -
                                                     value(offset + xi, b)));
                }
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    return !std::isfinite(error) || error != 0;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result{0};
    if (argc > 1 && std::string(argv[1]) == "force-response") {
        for (int mag : {0, 1}) {
            result += call_test("external LAPW force response", test_lapw_force_response, mag,
                                argc > 2 ? std::stod(argv[2]) : 9.0);
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc > 1 && std::string(argv[1]) == "core") {
        for (int mag : {0, 1}) {
            for (bool second : {false, true}) {
                bool distributed = mpi::Communicator::world().size() > 1;
                result += call_test("external LAPW core SCF", test_lapw_external, mag, distributed, second, true);
            }
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    for (int mag : {0, 1}) {
        for (bool second : {false, true}) {
            result += call_test("external LAPW SCF", test_lapw_external, mag, false, second, false);
            if (mpi::Communicator::world().size() > 1) {
                result += call_test("external LAPW distributed FFT SCF", test_lapw_external, mag, true, second, false);
            }
        }
    }
    result += call_test("external LAPW core response guard", test_core_guard);
    result += call_test("external LAPW linearisation response", test_lapw_linearization_response);
    result += call_test("replicated LAPW eigenvector redistribution", test_replicated_eigenvector_remap);
    sirius::finalize();
    return std::min(result, 1);
}
