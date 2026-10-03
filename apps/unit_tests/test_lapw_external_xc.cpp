/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/dft_ground_state.hpp"
#include "dft/lapw_response.hpp"
#include "hamiltonian/diagonalize.hpp"
#include "hamiltonian/diagonalize_fp.hpp"
#include "dft/smearing.hpp"
#include "function3d/xc_point_fields.hpp"
#include "lapw/step_function.hpp"
#include <map>

using namespace sirius;

namespace {

// A fixed model parameter keeps geometry dependence in the explicit quadrature data.
constexpr double functional_normalization = 336.0;

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
    double nonlocal = 0.01 * moment / functional_normalization;
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
    double moment{0}, base_moment{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j = p + in.num_points * s;
            moment += in.points->descriptor_weights[p] * (direction.rho[j] + 0.3 * direction.tau[j]);
            base_moment += in.points->descriptor_weights[p] * (in.rho[j] + 0.3 * in.tau[j]);
            if (direction.geometry) {
                moment += direction.geometry->descriptor_weights[p] * (in.rho[j] + 0.3 * in.tau[j]);
            }
        }
    }
    ctx.comm().allreduce(&moment, 1);
    ctx.comm().allreduce(&base_moment, 1);
    double nonlocal = 0.01 * moment / functional_normalization, scale = functional_scale(ctx);
    double base_nonlocal = 0.01 * base_moment / functional_normalization;
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j    = p + in.num_points * s;
            double w = in.points->energy_weights[p], d = in.points->descriptor_weights[p];
            double dw = direction.geometry ? direction.geometry->energy_weights[p] : 0;
            double dd = direction.geometry ? direction.geometry->descriptor_weights[p] : 0;
            double r = in.rho[j], t = in.tau[j], dr = direction.rho[j], dt = direction.tau[j];
            out.vrho[j]  = scale * (w * (0.04 * dr + 0.005 * dt) + dw * (0.04 * r + 0.005 * t) + d * nonlocal +
                                    dd * base_nonlocal);
            out.vtau[j]  = scale * (w * (0.02 * dt + 0.005 * dr) + dw * (0.02 * t + 0.005 * r) +
                                    0.3 * (d * nonlocal + dd * base_nonlocal));
            double local = (0.04 * r + 0.005 * t) * dr + (0.02 * t + 0.005 * r) * dt;
            for (int x = 0; x < 3; x++) {
                int k            = p + in.num_points * (x + 3 * s);
                out.vgradient[k] = scale * 0.004 * (w * direction.gradient[k] + dw * in.gradient[k]);
                local += 0.004 * in.gradient[k] * direction.gradient[k];
            }
            if (out.geometry) {
                out.geometry->energy_weights[p] += scale * local;
                out.geometry->descriptor_weights[p] +=
                        scale * (nonlocal * (r + 0.3 * t) + base_nonlocal * (dr + 0.3 * dt));
            }
        }
    }
}

auto
context(int mag, bool distributed_fft, bool second_variation, bool core = false, bool relativistic_core = false,
        double displacement = 0, double pw_cutoff = 5.0, bool simple_radial_integrals = false, int auto_enu = 0,
        std::string smearing = "", bool moving_core_grid = false)
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
    config["parameters"]["num_bands"]         = core ? (moving_core_grid ? 16 : 8) : 4;
    config["parameters"]["pw_cutoff"]         = pw_cutoff;
    if (moving_core_grid) {
        config["mixer"]["type"] = "anderson_stable";
    }
    if (!smearing.empty()) {
        config["parameters"]["smearing"] = smearing;
    }
    config["settings"]["simple_lapw_ri"] = simple_radial_integrals;
    auto ctx                             = std::make_unique<Simulation_context>(config);
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
    } else if (moving_core_grid) {
        ctx->unit_cell().add_atom(label, {0.71, 0.65, 0.61}, {0, 0, mag ? 0.2 : 0.0});
    }
    auto points     = std::make_shared<xc_point_set_t>(*ctx);
    constexpr int n = 10;
    // Replicated FFTs exercise empty model ranks; distributed FFTs also partition model rows.
    if (ctx->comm().rank() == 0 || distributed_fft) {
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    std::array<double, 3> p{6 * (x + 0.37) / n, 7 * (y + 0.29) / n, 8 * (z + 0.41) / n};
                    bool in_sphere{false};
                    if (core) {
                        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                            auto centre =
                                    ctx->unit_cell().get_cartesian_coordinates(ctx->unit_cell().atom(ia).position());
                            if (ia == 0) {
                                centre = {1.38, 2.17, 3.44};
                            }
                            auto r = r3::vector<double>{p[0], p[1], p[2]} - centre;
                            in_sphere |= r.length() <= 1.8;
                        }
                    }
                    if (in_sphere) {
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
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                auto centre = ctx->unit_cell().get_cartesian_coordinates(ctx->unit_cell().atom(ia).position());
                if (ia == 0 && !moving_core_grid) {
                    centre = {1.38, 2.17, 3.44};
                }
                for (int ir = 0; ir < radial.num_points(); ir++) {
                    for (int axis = 0; axis < 3; axis++) {
                        for (int sign : {-1, 1}) {
                            std::array<double, 3> p{centre[0], centre[1], centre[2]};
                            p[axis] += sign * radial[ir];
                            points->coordinates.push_back(p);
                            double w = fourpi / 6 * radial[ir] * radial[ir] * weights[ir];
                            points->energy_weights.push_back(w);
                            points->descriptor_weights.push_back(w);
                        }
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
test_occupation_response(int mag, bool distributed, bool gaussian)
{
    auto ctx = context(mag, distributed, false, false, false, 0, 5.0, false, 0, gaussian ? "gaussian" : "fermi_dirac");
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    ks.initialize();
    int failures{0}, ns = ctx->num_spins(), nb = ctx->num_bands();
    auto check = [&](char const* name, double error, double tolerance) {
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        if (ctx->comm().rank() == 0) {
            std::cout << "occupation response spins=" << ns << " distributed FFT=" << distributed << " " << name << ": "
                      << error << '\n';
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
    std::vector<mdarray<double, 2>> directions;
    for (auto it : ks.spl_num_kpoints()) {
        directions.emplace_back(std::array<index_range, 2>{nb, ns});
        for (int s = 0; s < ns; s++) {
            for (int b = 0; b < nb; b++) {
                directions.back()(b, s) = 0.07 * std::cos(b + 2 * it.i + s) + 0.1 * s;
            }
        }
    }
    reject("missing chemical potential rejected", [&] { ks.band_occupation_response(directions); });
    auto snapshot = [&] {
        std::array<double, 2> mu{ks.energy_fermi_fixed_mag(0), ks.energy_fermi_fixed_mag(ns - 1)};
        std::vector<double> values;
        for (int ik = 0; ik < ks.num_kpoints(); ik++) {
            for (int s = 0; s < ns; s++) {
                for (int b = 0; b < nb; b++) {
                    values.push_back(ks.get<double>(ik)->band_occupancy(b, s));
                }
            }
        }
        return std::make_pair(mu, values);
    };
    auto sample = [&](double step) {
        size_t ik{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            for (int s = 0; s < ns; s++) {
                for (int b = 0; b < nb; b++) {
                    kp.band_energy(b, s, -0.03 + 0.02 * b + 0.003 * it.i + 0.005 * s + step * directions[ik](b, s));
                }
            }
            ik++;
        }
        ks.sync_band<double, sync_band_t::energy>();
        ks.find_band_occupancies<double>();
        return snapshot();
    };
    for (double moment : mag ? std::vector<double>{0.4, 0.0} : std::vector<double>{0.0}) {
        ctx->cfg().unlock();
        ctx->cfg().parameters().fixed_mag(moment);
        ctx->cfg().lock();
        auto before   = sample(0);
        auto response = ks.band_occupation_response(directions);
        check("response leaves the ground-state populations unchanged", snapshot() == before ? 0 : 1, 0);
        std::array<double, 2> count{};
        size_t ik{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto const& kp = *ks.get<double>(it.i);
            for (int s = 0; s < ns; s++) {
                for (int b = 0; b < nb; b++) {
                    count[s] += kp.weight() * response.occupations[ik](b, s) / kp.comm().size();
                }
            }
            ik++;
        }
        ctx->comm().allreduce(count.data(), count.size());
        check("total charge response", std::abs(count[0] + count[1]), 1e-11);
        if (moment != 0) {
            check("fixed spin moment response", std::abs(count[0] - count[1]), 1e-11);
        } else if (mag) {
            check("unconstrained spin response is retained", std::abs(count[0] - count[1]) > 1e-3 ? 0 : 1, 0);
            check("unconstrained spins share one chemical potential", std::abs(response.fermi[0] - response.fermi[1]),
                  0);
        }
        for (double step : {1e-4, 5e-5}) {
            auto plus = sample(step), minus = sample(-step);
            double mu_error{0}, occupation_error{0};
            for (int s = 0; s < ns; s++) {
                mu_error =
                        std::max(mu_error, std::abs(response.fermi[s] - (plus.first[s] - minus.first[s]) / (2 * step)));
            }
            ik = 0;
            for (auto it : ks.spl_num_kpoints()) {
                for (int s = 0; s < ns; s++) {
                    for (int b = 0; b < nb; b++) {
                        int i = (it.i * ns + s) * nb + b;
                        occupation_error =
                                std::max(occupation_error, std::abs(response.occupations[ik](b, s) -
                                                                    (plus.second[i] - minus.second[i]) / (2 * step)));
                    }
                }
                ik++;
            }
            check("chemical potential finite difference", mu_error, 3e-8);
            check("occupation finite difference", occupation_error, 2e-6);
        }
        sample(0);
    }
    for (auto& d : directions) {
        std::fill_n(d.at(memory_t::host), d.size(), 0.17);
    }
    auto uniform = ks.band_occupation_response(directions);
    double error{0};
    for (int s = 0; s < ns; s++) {
        error = std::max(error, std::abs(uniform.fermi[s] - 0.17));
    }
    for (auto const& d : uniform.occupations) {
        for (size_t i = 0; i < d.size(); i++) {
            error = std::max(error, std::abs(d[i]));
        }
    }
    check("uniform spectral shift leaves occupations unchanged", error, 1e-13);
    reject("missing k-point directions rejected", [&] { ks.band_occupation_response({}); });
    if (ctx->comm().rank() == 0 && !directions.empty()) {
        directions[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    reject("nonfinite directions rejected collectively", [&] { ks.band_occupation_response(directions); });
    if (ctx->comm().rank() == 0 && !directions.empty()) {
        directions[0](0, 0) = 0.17;
    }
    if (distributed && ctx->comm().size() > 1) {
        if (ctx->comm().rank() == 0 && !directions.empty()) {
            directions[0](0, 0) += 0.1;
        }
        reject("inconsistent k-point replicas rejected", [&] { ks.band_occupation_response(directions); });
        if (ctx->comm().rank() == 0 && !directions.empty()) {
            directions[0](0, 0) = 0.17;
        }
    }
    ctx->cb().band_occ_ = [] {};
    reject("custom occupation callback requires its own response", [&] { ks.band_occupation_response(directions); });
    ctx->cb().band_occ_ = nullptr;
    ctx->cfg().unlock();
    ctx->cfg().parameters().extra_charge(ctx->unit_cell().num_valence_electrons() - nb * ctx->max_occupancy());
    if (!mag) {
        ctx->cfg().parameters().smearing_width(0);
    } else {
        ctx->cfg().parameters().fixed_mag(nb * ctx->max_occupancy());
    }
    ctx->cfg().lock();
    sample(0);
    auto constant = ks.band_occupation_response(directions);
    error         = std::max(std::abs(constant.fermi[0]), std::abs(constant.fermi[1]));
    for (auto const& d : constant.occupations) {
        for (size_t i = 0; i < d.size(); i++) {
            error = std::max(error, std::abs(d[i]));
        }
    }
    check("completely filled or empty populations have zero occupation response", error, 0);
    return failures;
}

int
test_lapw_density_response(int mag, bool distributed)
{
    auto ctx = context(mag, distributed, false, false, false, 0, 5.0, false, 0, "fermi_dirac");
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    Hamiltonian0<double> h0(gs.potential(), false);
    h0.generate_pw_coefs(gs.potential());
    int failures{0}, ns = ctx->num_spins(), nb = ctx->num_bands();
    auto check = [&](char const* name, double error, double tolerance) {
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        if (ctx->comm().rank() == 0) {
            std::cout << "constrained LAPW spins=" << ns << " distributed FFT=" << distributed << " " << name << ": "
                      << error << '\n';
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
    using matrix_t = la::dmatrix<std::complex<double>>;
    struct matrices_t
    {
        std::array<std::unique_ptr<matrix_t>, 2> h, s, dh, ds;
    };
    std::vector<matrices_t> matrices;
    std::vector<LAPW_density_response::matrix_direction_t> directions;
    for (auto it : ks.spl_num_kpoints()) {
        auto& kp = *ks.get<double>(it.i);
        Hamiltonian_k<double> hk(h0, kp);
        int n = kp.gklo_basis_size(), bs = ctx->cyclic_block_size();
        matrices.emplace_back();
        directions.emplace_back();
        auto& m = matrices.back();
        for (int spin = 0; spin < ns; spin++) {
            for (auto* part : {&m.h, &m.s, &m.dh, &m.ds}) {
                (*part)[spin] = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
            }
            hk.set_fv_h_o(spin, *m.h[spin], *m.s[spin]);
            for (int j = 0; j < m.h[spin]->num_cols_local(); j++) {
                for (int i = 0; i < m.h[spin]->num_rows_local(); i++) {
                    int row = m.h[spin]->irow(i), col = m.h[spin]->icol(j);
                    auto local          = std::polar(1.0, 0.7 * (row - col)) / ((1.0 + row) * (1.0 + col));
                    (*m.dh[spin])(i, j) = (0.06 + 0.03 * it.i + 0.04 * spin) * (*m.s[spin])(i, j) + 0.012 * local;
                    (*m.ds[spin])(i, j) = 0.007 * (*m.s[spin])(i, j) + 0.001 * local;
                }
            }
            directions.back().h[spin] = m.dh[spin].get();
            directions.back().s[spin] = m.ds[spin].get();
        }
    }
    auto sample = [&](double step) {
        std::vector<std::array<std::unique_ptr<LAPW_eigensystem>, 2>> spectra(matrices.size());
        size_t ik{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            Hamiltonian_k<double> hk(h0, kp);
            auto const& m = matrices[ik];
            int n = kp.gklo_basis_size(), bs = ctx->cyclic_block_size();
            matrix_t h(n, n, ctx->blacs_grid(), bs, bs), s(n, n, ctx->blacs_grid(), bs, bs);
            for (int spin = 0; spin < ns; spin++) {
                for (int j = 0; j < h.num_cols_local(); j++) {
                    for (int i = 0; i < h.num_rows_local(); i++) {
                        h(i, j) = (*m.h[spin])(i, j) + step * (*m.dh[spin])(i, j);
                        s(i, j) = (*m.s[spin])(i, j) + step * (*m.ds[spin])(i, j);
                    }
                }
                spectra[ik][spin] = std::make_unique<LAPW_eigensystem>(hk, h, s);
                for (int b = 0; b < nb; b++) {
                    kp.band_energy(b, spin, spectra[ik][spin]->energies()[b]);
                }
            }
            ik++;
        }
        ks.sync_band<double, sync_band_t::energy>();
        ks.find_band_occupancies<double>();
        std::vector<mdarray<std::complex<double>, 3>> result;
        ik = 0;
        for (auto it : ks.spl_num_kpoints()) {
            auto const& kp = *ks.get<double>(it.i);
            int n          = kp.gklo_basis_size();
            result.emplace_back(std::array<index_range, 3>{n, n, ns});
            for (int spin = 0; spin < ns; spin++) {
                std::vector<double> f(n, 0);
                for (int b = 0; b < nb; b++) {
                    f[b] = kp.band_occupancy(b, spin);
                }
                auto p = spectra[ik][spin]->density_matrix(f);
                std::copy_n(p.at(memory_t::host), p.size(), result.back().at(memory_t::host) + spin * p.size());
            }
            ik++;
        }
        return result;
    };
    auto fields = [&](auto const& p) {
        auto mapped = gs.density().lapw_density_matrix_fields(ks, p);
        std::vector<double> result;
        for (int spin = 0; spin < ns; spin++) {
            for (auto const* field : {&mapped->component(spin), &mapped->kinetic_density().component(spin)}) {
                auto values = potential_samples(*field);
                result.insert(result.end(), values.begin(), values.end());
            }
        }
        return result;
    };
    sample(0);
    LAPW_density_response reference(h0, ks);
    double entropy_response{0};
    auto p = reference.density_matrices(), dp = reference.density_matrix_response(directions, &entropy_response);
    auto response = fields(dp);
    std::array<std::complex<double>, 2> trace{};
    size_t ik{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto const& kp = *ks.get<double>(it.i);
        for (int spin = 0; spin < ns; spin++) {
            auto s = matrices[ik].s[spin]->get_full_matrix(), ds = matrices[ik].ds[spin]->get_full_matrix();
            for (int j = 0; j < s.size(1); j++) {
                for (int i = 0; i < s.size(0); i++) {
                    trace[spin] += kp.weight() / kp.comm().size() *
                                   (dp[ik](i, j, spin) * s(j, i) + p[ik](i, j, spin) * ds(j, i));
                }
            }
        }
        ik++;
    }
    ctx->comm().allreduce(trace.data(), trace.size());
    check("charge including overlap response", std::abs(trace[0] + trace[1]), 1e-10);
    if (mag) {
        check("fixed spin including overlap response", std::abs(trace[0] - trace[1]), 1e-10);
    }
    // Resolve the field response above the production chemical-potential search tolerance.
    for (double step : {1e-3, 5e-4}) {
        auto plus            = fields(sample(step));
        double plus_entropy  = ks.entropy_sum();
        auto minus           = fields(sample(-step));
        double minus_entropy = ks.entropy_sum();
        check("rho/tau matrix and occupation response finite difference",
              potential_response_error(response, plus, minus, step), 2e-6);
        check("occupation entropy response finite difference",
              std::abs((plus_entropy - minus_entropy) / (2 * step) - entropy_response), 2e-8);
    }
    sample(0);
    reject("missing matrix directions rejected collectively", [&] { reference.density_matrix_response({}); });
    if (ctx->comm().size() > 1) {
        reject("inconsistent entropy output rejected collectively", [&] {
            reference.density_matrix_response(directions, ctx->comm().rank() == 0 ? &entropy_response : nullptr);
        });
    }
    std::complex<double> saved;
    if (ctx->comm().rank() == 0 && !matrices.empty()) {
        saved                      = (*matrices[0].dh[0])(0, 0);
        (*matrices[0].dh[0])(0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    reject("nonfinite matrix direction rejected collectively", [&] { reference.density_matrix_response(directions); });
    if (ctx->comm().rank() == 0 && !matrices.empty()) {
        (*matrices[0].dh[0])(0, 0) = saved;
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            kp.band_energy(0, 0, kp.band_energy(0, 0) + 0.1);
            break;
        }
    }
    reject("changed reference rejected collectively", [&] { reference.density_matrix_response(directions); });
    sample(0);
    for (auto& m : matrices) {
        for (int spin = 0; spin < ns; spin++) {
            m.ds[spin]->zero();
            for (int j = 0; j < m.dh[spin]->num_cols_local(); j++) {
                for (int i = 0; i < m.dh[spin]->num_rows_local(); i++) {
                    (*m.dh[spin])(i, j) = 0.17 * (*m.s[spin])(i, j);
                }
            }
        }
    }
    double invariant{0};
    for (auto const& values : fields(reference.density_matrix_response(directions))) {
        invariant =
                std::isfinite(values) ? std::max(invariant, std::abs(values)) : std::numeric_limits<double>::infinity();
    }
    check("uniform potential shift leaves physical fields unchanged", invariant, 1e-11);
    return failures;
}

int
test_lapw_coupled_response(int mag, bool distributed)
{
    auto ctx = context(mag, distributed, false, false, false, 0, 5.0, false, 0, "fermi_dirac");
    // Geometry derivatives require unchanged point-domain membership throughout the stencil.
    auto points          = std::make_shared<xc_point_set_t>(*ctx);
    auto const& original = *ctx->cb().external_xc_points_;
    for (size_t p = 0; p < original.coordinates.size(); p++) {
        auto const& xyz = original.coordinates[p];
        bool near_boundary{false};
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx->unit_cell().atom(ia);
            auto r           = r3::vector<double>{xyz[0], xyz[1], xyz[2]} -
                               ctx->unit_cell().get_cartesian_coordinates(atom.position());
            auto fractional  = dot(ctx->unit_cell().inverse_lattice_vectors(), r);
            for (int x = 0; x < 3; x++) {
                fractional[x] -= std::round(fractional[x]);
            }
            double distance = dot(ctx->unit_cell().lattice_vectors(), fractional).length();
            near_boundary |= std::abs(distance - atom.mt_radius()) < 0.04;
        }
        if (!near_boundary) {
            points->coordinates.push_back(xyz);
            points->energy_weights.push_back(original.energy_weights[p]);
            points->descriptor_weights.push_back(original.descriptor_weights[p]);
        }
    }
    if (!points->coordinates.empty()) {
        points->block_sizes   = {static_cast<int>(points->coordinates.size())};
        points->block_centres = original.block_centres;
    }
    ctx->cb().external_xc_points_            = std::move(points);
    ctx->cb().external_xc_geometry_          = functional;
    ctx->cb().external_xc_geometry_response_ = true;
    auto quadrature_tangent                  = [](auto const& c, auto const& q, auto const& d, auto& out) {
        r3::matrix<double> lattice;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                lattice(x, y) = d.lattice[x + 3 * y];
            }
        }
        auto strain  = dot(lattice, c.unit_cell().inverse_lattice_vectors());
        double trace = strain(0, 0) + strain(1, 1) + strain(2, 2);
        auto move    = [&](auto const& r) {
            auto delta = dot(strain, r3::vector<double>{r[0], r[1], r[2]});
            return std::array<double, 3>{delta[0] + d.atoms[0][0], delta[1] + d.atoms[0][1], delta[2] + d.atoms[0][2]};
        };
        for (size_t p = 0; p < q.coordinates.size(); p++) {
            out.coordinates[p]        = move(q.coordinates[p]);
            out.energy_weights[p]     = trace * q.energy_weights[p];
            out.descriptor_weights[p] = trace * q.descriptor_weights[p];
        }
        for (size_t b = 0; b < q.block_centres.size(); b++) {
            out.block_centres[b] = move(q.block_centres[b]);
        }
    };
    ctx->cb().external_xc_quadrature_direction_ = quadrature_tangent;
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    int failures{0}, ns = ctx->num_spins();
    auto check = [&](char const* name, double error, double tolerance) {
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        if (ctx->comm().rank() == 0) {
            std::cout << "coupled LAPW spins=" << ns << " distributed FFT=" << distributed << " " << name << ": "
                      << error << '\n';
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
    auto prepare = [&](Density& d) {
        for (int s = 0; s < ns; s++) {
            for (auto* f : {&d.component(s), &d.kinetic_density().component(s)}) {
                f->rg().fft_transform(-1);
                f->mt().sync(ctx->unit_cell().spl_num_atoms());
            }
        }
    };
    auto values = [&](Density const& d) {
        std::vector<double> result;
        for (int s = 0; s < ns; s++) {
            for (auto const* f : {&d.component(s), &d.kinetic_density().component(s)}) {
                auto v = potential_samples(*f);
                result.insert(result.end(), v.begin(), v.end());
            }
        }
        return result;
    };
    Density direction(*ctx), zero(*ctx);
    copy(gs.density(), direction);
    for (int s = 0; s < ns; s++) {
        direction.component(s) *= 0.3 * (s + 1);
        direction.kinetic_density().component(s) *= -0.2 * (s + 1);
    }
    prepare(direction);
    zero.zero();
    zero.kinetic_density().zero();
    // Differentiate one production SCF map, not a manufactured Hamiltonian.
    // Complete eigenvectors are used only for scratch density matrices; stored orbitals stay fixed.
    auto map = [&](Density const& input, K_point_set& kpoints) {
        auto& c = kpoints.ctx();
        Potential potential(c);
        potential.generate(input, false, true);
        Hamiltonian0<double> h0(potential, true);
        h0.generate_pw_coefs(potential);
        std::vector<std::array<std::unique_ptr<LAPW_eigensystem>, 2>> spectra;
        using matrix_t = la::dmatrix<std::complex<double>>;
        for (auto it : kpoints.spl_num_kpoints()) {
            auto& kp = *kpoints.get<double>(it.i);
            auto hk  = h0(kp);
            int n = kp.gklo_basis_size(), bs = c.cyclic_block_size();
            matrix_t h(n, n, c.blacs_grid(), bs, bs), s(n, n, c.blacs_grid(), bs, bs);
            spectra.emplace_back();
            for (int spin = 0; spin < ns; spin++) {
                hk.set_fv_h_o(spin, h, s);
                spectra.back()[spin] = std::make_unique<LAPW_eigensystem>(hk, h, s);
                for (int b = 0; b < c.num_bands(); b++) {
                    kp.band_energy(b, spin, spectra.back()[spin]->energies()[b]);
                }
            }
        }
        kpoints.sync_band<double, sync_band_t::energy>();
        kpoints.find_band_occupancies<double>();
        std::vector<mdarray<std::complex<double>, 3>> matrices;
        size_t ik{0};
        for (auto it : kpoints.spl_num_kpoints()) {
            auto const& kp = *kpoints.get<double>(it.i);
            int n          = kp.gklo_basis_size();
            matrices.emplace_back(std::array<index_range, 3>{n, n, ns});
            for (int spin = 0; spin < ns; spin++) {
                std::vector<double> f(n, 0);
                for (int b = 0; b < c.num_bands(); b++) {
                    f[b] = kp.band_occupancy(b, spin);
                }
                auto p = spectra[ik][spin]->density_matrix(f);
                std::copy_n(p.at(memory_t::host), p.size(), matrices.back().at(memory_t::host) + spin * p.size());
            }
            ik++;
        }
        return input.lapw_density_matrix_fields(kpoints, matrices);
    };
    auto sample = [&](double step) {
        Density input(*ctx);
        copy(gs.density(), input);
        for (int s = 0; s < ns; s++) {
            axpy(step, direction.component(s), input.component(s));
            axpy(step, direction.kinetic_density().component(s), input.kinetic_density().component(s));
        }
        prepare(input);
        auto result = map(input, ks);
        gs.potential().update_atomic_potential();
        ctx->unit_cell().generate_radial_functions(ctx->out());
        ctx->unit_cell().generate_radial_integrals();
        return result;
    };
    sample(0);
    auto before = values(gs.density());
    LAPW_response response(ks, gs.density(), gs.potential());
    auto jx       = response.apply(direction);
    auto analytic = values(*jx);
    double peak{0};
    for (auto v : analytic) {
        peak = std::max(peak, std::abs(v));
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
    check("nonzero coupled response", peak > 1e-8 ? 0 : 1, 0);
    for (double step : {1e-3, 5e-4}) {
        auto plus = values(*sample(step)), minus = values(*sample(-step));
        check("complete SCF map rho/tau finite difference", potential_response_error(analytic, plus, minus, step),
              3e-5);
    }
    sample(0);
    double zero_error{0};
    for (auto v : values(*response.apply(zero))) {
        zero_error = std::max(zero_error, std::abs(v));
    }
    check("zero coupled response", zero_error, 1e-13);
    xc_geometry_direction_t geometry(ctx->unit_cell().num_atoms());
    zero_error = 0;
    for (auto v : values(*response.geometry_rhs(geometry))) {
        zero_error = std::max(zero_error, std::abs(v));
    }
    check("zero geometry right-hand side", zero_error, 1e-13);
    auto probe = context(mag, distributed, false, false, false, 0, 5.0, false, 0, "fermi_dirac");
    K_point_set probe_ks(*probe);
    probe_ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    probe_ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    probe_ks.initialize();
    Density input(*probe);
    for (int s = 0; s < ns; s++) {
        copy(gs.density().component(s), input.component(s));
        copy(gs.density().kinetic_density().component(s), input.kinetic_density().component(s));
    }
    auto lattice    = probe->unit_cell().lattice_vectors();
    auto quadrature = ctx->cb().external_xc_points_;
    std::vector<r3::vector<double>> centres;
    for (int ia = 0; ia < probe->unit_cell().num_atoms(); ia++) {
        centres.push_back(probe->unit_cell().get_cartesian_coordinates(probe->unit_cell().atom(ia).position()));
    }
    for (int part = 0; part < 3; part++) {
        geometry = xc_geometry_direction_t(ctx->unit_cell().num_atoms());
        if (part != 1) {
            geometry.atoms[0] = {0.23, -0.37, 0.19};
        }
        r3::matrix<double> strain;
        if (part != 0) {
            strain(0, 0) = 0.07;
            strain(1, 1) = -0.03;
            strain(2, 2) = 0.02;
            strain(0, 1) = -0.04;
            strain(2, 0) = 0.05;
        }
        auto dlattice = dot(strain, lattice);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                geometry.lattice[x + 3 * y] = dlattice(x, y);
            }
        }
        auto analytic_geometry    = values(*response.geometry_rhs(geometry));
        auto quadrature_direction = gs.potential().external_xc_quadrature_direction(geometry);
        auto geometry_sample      = [&](double step) {
            auto shifted = lattice;
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    shifted(x, y) += step * dlattice(x, y);
                }
            }
            probe->unit_cell().set_lattice_vectors(shifted);
            for (int ia = 0; ia < probe->unit_cell().num_atoms(); ia++) {
                auto const& d = geometry.atoms[ia];
                probe->unit_cell().atom(ia).set_position(
                        dot(inverse(shifted), centres[ia] + r3::vector<double>{d[0], d[1], d[2]} * step));
            }
            probe->update();
            for (auto it : probe_ks.spl_num_kpoints()) {
                probe_ks.get<double>(it.i)->update();
            }
            auto next                = std::make_shared<xc_point_set_t>(*probe);
            next->coordinates        = quadrature->coordinates;
            next->energy_weights     = quadrature->energy_weights;
            next->descriptor_weights = quadrature->descriptor_weights;
            next->block_sizes        = quadrature->block_sizes;
            next->block_centres      = quadrature->block_centres;
            for (size_t p = 0; p < next->coordinates.size(); p++) {
                for (int x = 0; x < 3; x++) {
                    next->coordinates[p][x] += step * quadrature_direction.coordinates[p][x];
                }
                next->energy_weights[p] += step * quadrature_direction.energy_weights[p];
                next->descriptor_weights[p] += step * quadrature_direction.descriptor_weights[p];
            }
            for (size_t b = 0; b < next->block_centres.size(); b++) {
                for (int x = 0; x < 3; x++) {
                    next->block_centres[b][x] += step * quadrature_direction.block_centres[b][x];
                }
            }
            probe->cb().external_xc_points_ = next;
            return values(*map(input, probe_ks));
        };
        // Resolve near-origin radial-derivative cancellation without increasing the FD truncation error.
        for (double step : {2e-2, 1e-2}) {
            auto plus = geometry_sample(step), minus = geometry_sample(-step);
            auto half_plus = geometry_sample(step / 2), half_minus = geometry_sample(-step / 2);
            for (size_t i = 0; i < analytic_geometry.size(); i++) {
                plus[i]  = (8 * (half_plus[i] - half_minus[i]) - (plus[i] - minus[i])) / 3;
                minus[i] = 0;
            }
            check("explicit geometry SCF map rho/tau finite difference",
                  potential_response_error(analytic_geometry, plus, minus, step), 3e-5);
        }
    }
    ctx->cb().external_xc_quadrature_direction_ = nullptr;
    reject("geometry right-hand side requires a host quadrature tangent", [&] { response.geometry_rhs(geometry); });
    ctx->cb().external_xc_quadrature_direction_ = quadrature_tangent;
    auto zero_solution                          = response.solve(zero, 1e-8, 1);
    check("zero right-hand side converges without mixing", zero_solution.residual, 0);
    Density rhs(*ctx);
    copy(direction, rhs);
    for (int s = 0; s < ns; s++) {
        axpy(-1.0, jx->component(s), rhs.component(s));
        axpy(-1.0, jx->kinetic_density().component(s), rhs.kinetic_density().component(s));
    }
    prepare(rhs);
    auto frozen = [&] {
        auto result    = values(gs.density());
        auto potential = potential_samples(gs.potential().effective_potential());
        result.insert(result.end(), potential.begin(), potential.end());
        for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& cls = ctx->unit_cell().atom_symmetry_class(ic);
            for (int i = 0; i < cls.atom_type().mt_radial_basis_size(); i++) {
                for (int ir = 0; ir < cls.atom_type().num_mt_points(); ir++) {
                    result.push_back(cls.radial_function(ir, i));
                    result.push_back(cls.radial_function_derivative(ir, i));
                }
            }
        }
        return result;
    };
    auto frozen_before = frozen();
    auto solution      = response.solve(rhs, 1e-8, 100, 10, 0.4);
    check("coupled solve residual", solution.residual, 1e-8);
    auto obtained = values(*solution.fields), expected = values(direction);
    double error{0};
    for (size_t i = 0; i < obtained.size(); i++) {
        error = std::max(error, std::abs(obtained[i] - expected[i]));
    }
    check("coupled solve recovers prescribed fields", error, 2e-8);
    if (ctx->comm().rank() == 0) {
        std::cout << "coupled LAPW solve iterations: " << solution.iterations << '\n';
    }
    reject("one iteration does not claim convergence", [&] { response.solve(rhs, 1e-8, 1); });
    reject("invalid response controls rejected collectively",
           [&] { response.solve(rhs, ctx->comm().rank() == 0 ? -1 : 1e-8, 10); });
    double saved_tau = rhs.kinetic_density().scalar().rg().value(0);
    if (ctx->comm().rank() == 0) {
        rhs.kinetic_density().scalar().rg().value(0) = std::numeric_limits<double>::quiet_NaN();
    }
    reject("nonfinite kinetic-density direction rejected collectively", [&] { response.apply(rhs); });
    rhs.kinetic_density().scalar().rg().value(0) = saved_tau;
    auto after                                   = values(gs.density());
    error                                        = 0;
    for (size_t i = 0; i < before.size(); i++) {
        error = std::max(error, std::abs(before[i] - after[i]));
    }
    check("ground-state fields remain unchanged", error, 0);
    auto frozen_after = frozen();
    error             = 0;
    for (size_t i = 0; i < frozen_before.size(); i++) {
        error = std::max(error, std::abs(frozen_before[i] - frozen_after[i]));
    }
    check("response solve preserves reference potential and radial functions", error, 0);
    return failures;
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
        xc_geometry_direction_t geometry(ctx->unit_cell().num_atoms());
        peak = 0;
        for (auto x : potential_samples(*potential.hartree_geometry_response(direction, geometry))) {
            peak = std::max(peak, std::abs(x));
        }
        check("zero Hartree geometry response", peak, 0);
        // A separate context updates Poisson caches without resetting the reference radial basis.
        auto probe = context(mag, distributed_fft, second_variation, core, false, 0, 5.0, simple);
        Periodic_function<double> frozen(*probe, [&](int) { return lmax_t(probe->lmax_rho()); });
        copy(direction, frozen);
        Potential probe_potential(*probe);
        auto lattice = ctx->unit_cell().lattice_vectors();
        std::vector<r3::vector<double>> centres;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            centres.push_back(ctx->unit_cell().get_cartesian_coordinates(ctx->unit_cell().atom(ia).position()));
        }
        auto quadrature = probe->cb().external_xc_points_;
        for (int part = 0; part < 3; part++) {
            geometry = xc_geometry_direction_t(ctx->unit_cell().num_atoms());
            if (part != 1) {
                geometry.atoms[0] = {0.23, -0.37, 0.19};
            }
            r3::matrix<double> strain;
            if (part != 0) {
                strain(0, 0) = 0.07;
                strain(1, 1) = -0.03;
                strain(2, 2) = 0.02;
                strain(0, 1) = -0.04;
                strain(2, 0) = 0.05;
            }
            auto dlattice = dot(strain, lattice);
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    geometry.lattice[x + 3 * y] = dlattice(x, y);
                }
            }
            auto evaluate = [&](double step, bool nuclei) {
                auto shifted = lattice;
                for (int y = 0; y < 3; y++) {
                    for (int x = 0; x < 3; x++) {
                        shifted(x, y) += step * dlattice(x, y);
                    }
                }
                probe->unit_cell().set_lattice_vectors(shifted);
                for (int ia = 0; ia < probe->unit_cell().num_atoms(); ia++) {
                    auto const& d = geometry.atoms[ia];
                    probe->unit_cell().atom(ia).set_position(
                            dot(inverse(shifted), centres[ia] + r3::vector<double>{d[0], d[1], d[2]} * step));
                }
                probe->update();
                auto next                       = std::make_shared<xc_point_set_t>(*probe);
                next->coordinates               = quadrature->coordinates;
                next->energy_weights            = quadrature->energy_weights;
                next->descriptor_weights        = quadrature->descriptor_weights;
                next->block_sizes               = quadrature->block_sizes;
                next->block_centres             = quadrature->block_centres;
                probe->cb().external_xc_points_ = next;
                probe_potential.update();
                if (nuclei) {
                    probe_potential.poisson(frozen);
                    return potential_samples(probe_potential.hartree_potential());
                }
                return potential_samples(*probe_potential.hartree_response(frozen));
            };
            for (bool nuclei : {false, true}) {
                auto analytic = potential_samples(*potential.hartree_geometry_response(direction, geometry, nuclei));
                // Larger fourth-order steps avoid nuclear-potential cancellation and pole-angle roundoff.
                for (double step : {6e-3, 3e-3}) {
                    auto plus       = evaluate(step, nuclei);
                    auto minus      = evaluate(-step, nuclei);
                    auto half_plus  = evaluate(step / 2, nuclei);
                    auto half_minus = evaluate(-step / 2, nuclei);
                    double error{0}, peak{0};
                    for (size_t i = 0; i < analytic.size(); i++) {
                        double fd = (8 * (half_plus[i] - half_minus[i]) - (plus[i] - minus[i])) / (6 * step);
                        // Propagate four ulps of each sampled potential through the difference stencil.
                        double roundoff = 4 * std::numeric_limits<double>::epsilon() *
                                          (8 * (std::abs(half_plus[i]) + std::abs(half_minus[i])) + std::abs(plus[i]) +
                                           std::abs(minus[i])) /
                                          (6 * step);
                        if (!std::isfinite(fd) || !std::isfinite(analytic[i])) {
                            error = std::numeric_limits<double>::infinity();
                        } else {
                            error = std::max(error, std::abs(fd - analytic[i]) - roundoff);
                            peak  = std::max(peak, std::abs(analytic[i]));
                        }
                    }
                    check(nuclei ? "Hartree geometry response including nuclei"
                                 : "electronic Hartree geometry response",
                          error / std::max(1e-8, peak), 3e-7);
                }
            }
        }
        auto bad_geometry = geometry;
        if (ctx->comm().rank() == 0) {
            bad_geometry.atoms.clear();
        }
        reject("Hartree geometry rejects malformed directions collectively",
               [&] { potential.hartree_geometry_response(direction, bad_geometry); });
        bad_geometry = geometry;
        if (ctx->comm().rank() == 0) {
            bad_geometry.lattice[0] = std::numeric_limits<double>::quiet_NaN();
        }
        reject("Hartree geometry rejects nonfinite directions collectively",
               [&] { potential.hartree_geometry_response(direction, bad_geometry); });
        if (ctx->comm().size() > 1) {
            bad_geometry = geometry;
            if (ctx->comm().rank() == 0) {
                bad_geometry.atoms[0][0] += 0.01;
            }
            reject("Hartree geometry rejects inconsistent replicas",
                   [&] { potential.hartree_geometry_response(direction, bad_geometry); });
            reject("Hartree geometry rejects inconsistent nuclear flags",
                   [&] { potential.hartree_geometry_response(direction, geometry, ctx->comm().rank() == 0); });
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
        reject("nonfinite Hartree geometry density rejected collectively",
               [&] { potential.hartree_geometry_response(direction, geometry); });
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
        reject("missing MT Hartree geometry density rejected collectively",
               [&] { potential.hartree_geometry_response(no_spheres, geometry); });
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
        ctx->cb().external_xc_response_          = functional_response;
        ctx->cb().external_xc_geometry_          = functional;
        ctx->cb().external_xc_geometry_response_ = true;
        auto point_errors                        = potential.check_external_xc_point_hessian(density, 1e-3);
        check("joint field/quadrature Hessian matches raw covector finite differences",
              std::max(point_errors[0], point_errors[1]), 2e-8);
        auto original_quadrature = ctx->cb().external_xc_points_;
        xc_point_derivatives_t geometry_direction(*original_quadrature);
        r3::vector<double> translation(0.04, 0.02, -0.03);
        std::vector<r3::vector<double>> original_atoms;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            original_atoms.push_back(ctx->unit_cell().atom(ia).position());
        }
        for (size_t p = 0; p < geometry_direction.coordinates.size(); p++) {
            for (int x = 0; x < 3; x++) {
                geometry_direction.coordinates[p][x] = translation[x];
            }
            geometry_direction.energy_weights[p]     = 0.02 * original_quadrature->energy_weights[p];
            geometry_direction.descriptor_weights[p] = 0.03 * original_quadrature->descriptor_weights[p];
        }
        auto host_tangent = [](auto const&, auto const& q, auto const& d, auto& out) {
            for (size_t p = 0; p < q.coordinates.size(); p++) {
                out.coordinates[p]        = d.atoms.front();
                out.energy_weights[p]     = 0.5 * d.atoms.front()[0] * q.energy_weights[p];
                out.descriptor_weights[p] = 1.5 * d.atoms.front()[1] * q.descriptor_weights[p];
            }
        };
        ctx->cb().external_xc_quadrature_direction_ = host_tangent;
        xc_geometry_direction_t host_direction(ctx->unit_cell().num_atoms());
        for (auto& r : host_direction.atoms) {
            r = {translation[0], translation[1], translation[2]};
        }
        auto host_quadrature = potential.external_xc_quadrature_direction(host_direction);
        double host_error{0};
        for (size_t p = 0; p < geometry_direction.coordinates.size(); p++) {
            for (int x = 0; x < 3; x++) {
                host_error = std::max(host_error, std::abs(host_quadrature.coordinates[p][x] - translation[x]));
            }
            host_error = std::max(
                    {host_error, std::abs(host_quadrature.energy_weights[p] - geometry_direction.energy_weights[p]),
                     std::abs(host_quadrature.descriptor_weights[p] - geometry_direction.descriptor_weights[p])});
        }
        check("host tangent matches the prescribed quadrature direction",
              host_quadrature.block_centres != geometry_direction.block_centres ? 1 : host_error, 1e-13);
        auto geometry_response = xc_samples(potential.lapw_external_xc_geometry_response(density, host_direction));
        auto bad_host          = host_direction;
        if (ctx->comm().rank() == 0) {
            bad_host.atoms.clear();
        }
        reject("wrong host geometry shape rejected collectively",
               [&] { potential.external_xc_quadrature_direction(bad_host); });
        bad_host = host_direction;
        if (ctx->comm().rank() == 0) {
            bad_host.lattice[0] = std::numeric_limits<double>::quiet_NaN();
        }
        reject("nonfinite host geometry direction rejected collectively",
               [&] { potential.external_xc_quadrature_direction(bad_host); });
        if (ctx->comm().size() > 1) {
            bad_host = host_direction;
            if (ctx->comm().rank() == 0) {
                bad_host.atoms[0][0] += 1;
            }
            reject("inconsistent host geometry replicas rejected collectively",
                   [&] { potential.external_xc_quadrature_direction(bad_host); });
        }
        ctx->cb().external_xc_quadrature_direction_ = [](auto const& c, auto const&, auto const&, auto& out) {
            if (c.comm().rank() == 0) {
                out.energy_weights[0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        reject("nonfinite host quadrature tangent rejected collectively",
               [&] { potential.external_xc_quadrature_direction(host_direction); });
        ctx->cb().external_xc_quadrature_direction_ = nullptr;
        reject("missing host tangent is not silently treated as fixed quadrature",
               [&] { potential.external_xc_quadrature_direction(host_direction); });
        auto geometry_sample = [&](double step) {
            auto displacement = step * r3::dot(ctx->unit_cell().inverse_lattice_vectors(), translation);
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                ctx->unit_cell().atom(ia).set_position(original_atoms[ia] + displacement);
            }
            auto next                = std::make_shared<xc_point_set_t>(*ctx);
            next->coordinates        = original_quadrature->coordinates;
            next->energy_weights     = original_quadrature->energy_weights;
            next->descriptor_weights = original_quadrature->descriptor_weights;
            next->block_sizes        = original_quadrature->block_sizes;
            next->block_centres      = original_quadrature->block_centres;
            for (size_t p = 0; p < next->coordinates.size(); p++) {
                for (int x = 0; x < 3; x++) {
                    next->coordinates[p][x] += step * geometry_direction.coordinates[p][x];
                }
                next->energy_weights[p] += step * geometry_direction.energy_weights[p];
                next->descriptor_weights[p] += step * geometry_direction.descriptor_weights[p];
            }
            ctx->cb().external_xc_points_ = next;
            displaced.xc(density);
            auto values                   = xc_samples(*displaced.lapw_xc_derivatives());
            ctx->cb().external_xc_points_ = original_quadrature;
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                ctx->unit_cell().atom(ia).set_position(original_atoms[ia]);
            }
            return values;
        };
        // Moving fractional atom positions and Cartesian points needs a step above coordinate roundoff.
        for (double h : {1e-3, 5e-4}) {
            auto plus = geometry_sample(h), minus = geometry_sample(-h);
            check("joint geometry/XC production covectors match finite differences",
                  potential_response_error(geometry_response, plus, minus, h), 2e-7);
        }
        check("geometry XC response preserves reference potential and energy",
              xc_samples(*potential.lapw_xc_derivatives()) != saved || potential.energy_exc(density) != saved_energy
                      ? 1
                      : 0,
              0);
        auto const& quadrature = *ctx->cb().external_xc_points_;
        int nr = static_cast<int>(quadrature.coordinates.size()), ns = ctx->num_spins();
        xc_point_fields_t fields(nr, ns), raw_response(nr, ns);
        fields.rho.zero();
        fields.gradient.zero();
        fields.tau.zero();
        raw_response.rho.zero();
        raw_response.gradient.zero();
        raw_response.tau.zero();
        xc_point_derivatives_t geometry(quadrature), geometry_output(quadrature);
        xc_callback_input_t input{nr,
                                  ns,
                                  0,
                                  fields.rho.at(memory_t::host),
                                  fields.gradient.at(memory_t::host),
                                  fields.tau.at(memory_t::host),
                                  &quadrature};
        xc_callback_direction_t d{input.rho, input.gradient, input.tau, &geometry};
        xc_callback_response_t output{raw_response.rho.at(memory_t::host), raw_response.gradient.at(memory_t::host),
                                      raw_response.tau.at(memory_t::host), &geometry_output};
        auto apply_point = [&] { potential.external_xc_point_response(input, d, output); };
        if (ctx->comm().rank() == 0) {
            geometry.coordinates.push_back({0, 0, 0});
        }
        reject("wrong quadrature direction shape rejected collectively", apply_point);
        geometry = xc_point_derivatives_t(quadrature);
        if (ctx->comm().rank() == 0) {
            geometry.energy_weights[0] = std::numeric_limits<double>::quiet_NaN();
        }
        reject("nonfinite quadrature direction rejected collectively", apply_point);
        geometry                        = xc_point_derivatives_t(quadrature);
        ctx->cb().external_xc_response_ = [](auto const& c, auto const& in, auto const& d, auto& out) {
            functional_response(c, in, d, out);
            if (c.comm().rank() == 0) {
                out.geometry->coordinates[0][0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        reject("nonfinite quadrature response rejected collectively", apply_point);
        ctx->cb().external_xc_response_          = functional_response;
        ctx->cb().external_xc_geometry_response_ = false;
        reject("field-only callback rejects quadrature differentiation", apply_point);
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
    reject("LAPW forces require complete host geometry callbacks", [&] { gs.forces().calc_forces_total(); });
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
        auto lattice   = ctx->unit_cell().lattice_vectors();
        auto position  = ctx->unit_cell().atom(0).position();
        auto cartesian = ctx->unit_cell().get_cartesian_coordinates(position);
        for (int part = 0; part < 3; part++) {
            r3::vector<double> displacement{0, 0, 0};
            r3::matrix<double> strain;
            if (part != 1) {
                displacement = {0.23, -0.37, 0.19};
            }
            if (part != 0) {
                strain(0, 0) = 0.07;
                strain(1, 1) = -0.03;
                strain(2, 2) = 0.02;
                strain(0, 1) = -0.04;
                strain(2, 0) = 0.05;
            }
            double error{0}, peak{0}, conjugation_error{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto kp = ks.get<double>(it.i);
                int ng = kp->gkvec_row().count(), na = ctx->unit_cell().atom(0).type().mt_aw_basis_size();
                mdarray<std::complex<double>, 2> derivative({ng, na}), conjugate_derivative({ng, na});
                Matching_coefficients matching(ctx->unit_cell(), kp->gkvec_row());
                matching.generate_geometry_response<false>(ctx->unit_cell().atom(0), displacement, strain, derivative);
                matching.generate_geometry_response<true>(ctx->unit_cell().atom(0), displacement, strain,
                                                          conjugate_derivative);
                for (size_t i = 0; i < derivative.size(); i++) {
                    peak = std::max(peak, std::abs(derivative[i]));
                    conjugation_error =
                            std::max(conjugation_error, std::abs(conjugate_derivative[i] - std::conj(derivative[i])));
                }
                auto evaluate = [&](double step) {
                    auto shifted_lattice = lattice;
                    auto dlattice        = dot(strain, lattice);
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            shifted_lattice(x, y) += step * dlattice(x, y);
                        }
                    }
                    ctx->unit_cell().set_lattice_vectors(shifted_lattice);
                    ctx->unit_cell().atom(0).set_position(
                            dot(inverse(shifted_lattice), cartesian + displacement * step));
                    kp->update();
                    Matching_coefficients shifted(ctx->unit_cell(), kp->gkvec_row());
                    mdarray<std::complex<double>, 2> result({ng, na});
                    shifted.generate<false>(ctx->unit_cell().atom(0), result);
                    return result;
                };
                for (double step : {1e-4, 5e-5}) {
                    auto plus = evaluate(step), minus = evaluate(-step);
                    for (size_t i = 0; i < derivative.size(); i++) {
                        error = std::max(error, std::abs((plus[i] - minus[i]) / (2 * step) - derivative[i]));
                    }
                }
                ctx->unit_cell().set_lattice_vectors(lattice);
                ctx->unit_cell().atom(0).set_position(position);
                kp->update();
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
            ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            ctx->comm().allreduce<double, mpi::op_t::max>(&conjugation_error, 1);
            check("LAPW matching atom/cell derivative", error / std::max(1e-12, peak), 3e-8);
            check("LAPW matching derivative conjugation", conjugation_error, 1e-14);
            check("nonzero LAPW matching geometry derivative", peak > 1e-7 ? 0 : 1, 0);
        }
        auto const& forces = gs.forces().calc_forces_ibs();
        std::vector<r3::vector<double>> positions, centres;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            positions.push_back(ctx->unit_cell().atom(ia).position());
            centres.push_back(ctx->unit_cell().get_cartesian_coordinates(positions.back()));
        }
        Hamiltonian0<double> h0(potential, false);
        h0.generate_pw_coefs(potential);
        for (int part = 0; part < 3; part++) {
            xc_geometry_direction_t direction(ctx->unit_cell().num_atoms());
            if (part != 1) {
                direction.atoms[0] = {0.23, -0.37, 0.19};
            }
            r3::matrix<double> strain;
            if (part != 0) {
                strain(0, 0) = 0.07;
                strain(1, 1) = -0.03;
                strain(2, 2) = 0.02;
                strain(0, 1) = -0.04;
                strain(2, 0) = 0.05;
            }
            auto dlattice = dot(strain, lattice);
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    direction.lattice[x + 3 * y] = dlattice(x, y);
                }
            }
            int rejected_failures{0};
            // Frozen radial basis, local matrices, Fourier covectors and independent PW/LO coefficients.
            auto snapshot = [&](double step, bool response) {
                if (!response) {
                    auto shifted = lattice;
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            shifted(x, y) += step * dlattice(x, y);
                        }
                    }
                    ctx->unit_cell().set_lattice_vectors(shifted);
                    for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                        r3::vector<double> dr{direction.atoms[ia][0], direction.atoms[ia][1], direction.atoms[ia][2]};
                        ctx->unit_cell().atom(ia).set_position(dot(inverse(shifted), centres[ia] + dr * step));
                    }
                }
                std::pair<double, std::array<std::vector<std::complex<double>>, 2>> result;
                for (auto it : ks.spl_num_kpoints()) {
                    auto kp = ks.get<double>(it.i);
                    if (!response) {
                        kp->update();
                    }
                    auto hk = h0(*kp);
                    int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                    la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                    for (int s = 0; s < ctx->num_spins(); s++) {
                        if (response) {
                            hk.set_fv_h_o_geometry_response(s, direction, h, o);
                        } else {
                            hk.set_fv_h_o(s, h, o);
                            // Reconstruct the moved interstitial mask without resetting the radial basis.
                            for (int j = 0; j < kp->num_gkvec_col(); j++) {
                                auto qj = kp->gkvec_col().gkvec_cart(gvec_index_t::local(j));
                                for (int i = 0; i < kp->num_gkvec_row(); i++) {
                                    auto qi = kp->gkvec_row().gkvec_cart(gvec_index_t::local(i));
                                    auto g  = qi - qj;
                                    int ig  = ctx->gvec().index_g12(kp->gkvec_row().gvec(gvec_index_t::local(i)),
                                                                    kp->gkvec_col().gvec(gvec_index_t::local(j)));
                                    std::complex<double> theta = ig == 0 ? 1 : 0;
                                    for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                                        auto const& atom = ctx->unit_cell().atom(ia);
                                        auto r           = ctx->unit_cell().get_cartesian_coordinates(atom.position());
                                        theta -= fourpi / ctx->unit_cell().omega() *
                                                 unit_step_function_form_factors(atom.mt_radius(), g.length()) *
                                                 std::exp(std::complex<double>(0, -dot(g, r)));
                                    }
                                    auto delta = theta - ctx->theta_pw(ig);
                                    o(i, j) += delta;
                                    h(i, j) += 0.5 * dot(qi, qj) * delta;
                                }
                            }
                        }
                        for (int j = 0; j < h.num_cols_local(); j++) {
                            for (int i = 0; i < h.num_rows_local(); i++) {
                                result.second[0].push_back(h(i, j));
                                result.second[1].push_back(o(i, j));
                            }
                        }
                        result.first += occupied_lapw_lagrangian(*ctx, *kp, s, h, o);
                    }
                    if (response && part == 0 && !mag && !second_variation) {
                        auto reject_input = [&](auto&& operation) {
                            bool rejected{false};
                            try {
                                operation();
                            } catch (std::exception const&) {
                                rejected = true;
                            }
                            rejected_failures += !rejected;
                        };
                        auto bad = direction;
                        if (kp->comm().rank() == 0) {
                            bad.atoms.clear();
                        }
                        reject_input([&] { hk.set_fv_h_o_geometry_response(0, bad, h, o); });
                        bad = direction;
                        if (kp->comm().rank() == 0) {
                            bad.lattice[0] = std::numeric_limits<double>::quiet_NaN();
                        }
                        reject_input([&] { hk.set_fv_h_o_geometry_response(0, bad, h, o); });
                        reject_input([&] { hk.set_fv_h_o_geometry_response(0, direction, h, h); });
                        if (kp->comm().size() > 1) {
                            bad = direction;
                            if (kp->comm().rank() == 0) {
                                bad.atoms[0][0] += 0.01;
                            }
                            reject_input([&] { hk.set_fv_h_o_geometry_response(0, bad, h, o); });
                        }
                    }
                }
                ctx->comm().allreduce(&result.first, 1);
                return result;
            };
            auto analytic = snapshot(0, true);
            for (double step : {1e-4, 5e-5}) {
                auto plus = snapshot(step, false), minus = snapshot(-step, false);
                for (int field : {0, 1}) {
                    double error{0}, peak{0};
                    for (size_t i = 0; i < analytic.second[field].size(); i++) {
                        double residual = std::abs((plus.second[field][i] - minus.second[field][i]) / (2 * step) -
                                                   analytic.second[field][i]);
                        error           = std::isfinite(residual) ? std::max(error, residual)
                                                                  : std::numeric_limits<double>::infinity();
                        peak            = std::max(peak, std::abs(analytic.second[field][i]));
                    }
                    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                    check("LAPW atom/cell matrix response", error / std::max(1e-12, peak), 3e-7);
                    check("nonzero LAPW geometry matrix response", peak > 1e-7 ? 0 : 1, 0);
                }
                check("LAPW geometry Lagrangian derivative",
                      std::abs((plus.first - minus.first) / (2 * step) - analytic.first), 2e-8);
            }
            snapshot(0, false);
            if (part == 0) {
                double force{0};
                for (int x = 0; x < 3; x++) {
                    force -= direction.atoms[0][x] * forces(x, 0);
                }
                check("LAPW basis/boundary force", std::abs(analytic.first - force), 2e-8);
                check("nonzero basis/boundary force", std::abs(force) > 1e-7 ? 0 : 1, 0);
                check("invalid geometry matrix inputs rejected collectively", rejected_failures, 0);
            }
        }
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
        std::vector<mdarray<std::complex<double>, 3>> coefficient_matrices, frozen_matrices;
        std::vector<std::unique_ptr<wf::Wave_functions<double>>> matched_directions;
        std::vector<wf::Wave_functions<double>*> matched_ptrs;
        size_t matrix_k{0};
        std::vector<int> nlo;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            nlo.push_back(ctx->unit_cell().atom(ia).type().mt_lo_basis_size());
        }
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            int n = kp.gklo_basis_size(), npw = kp.gkvec().num_gvec();
            coefficient_matrices.emplace_back(std::array<index_range, 3>{n, n, ctx->num_spins()});
            frozen_matrices.emplace_back(std::array<index_range, 3>{n, n, ctx->num_spins()});
            frozen_matrices.back().zero();
            auto& p = coefficient_matrices.back();
            p.zero();
            wf::Wave_functions<double> evec(kp.gkvec_sptr(), nlo, wf::num_mag_dims(0), wf::num_bands(ctx->num_bands()),
                                            memory_t::host);
            matched_directions.push_back(std::make_unique<wf::Wave_functions<double>>(
                    kp.gkvec_sptr(), nmt, wf::num_mag_dims(ctx->num_mag_dims()), wf::num_bands(ctx->num_bands()),
                    memory_t::host));
            auto& delta = *matched_directions.back();
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    auto all = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(b));
                    std::vector<std::complex<double>> c(all.begin(), all.begin() + npw), dc(n);
                    int offset = npw;
                    for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                        int naw = ctx->unit_cell().atom(ia).type().mt_aw_basis_size();
                        c.insert(c.end(), all.begin() + offset + naw, all.begin() + offset + nmt[ia]);
                        offset += nmt[ia];
                    }
                    for (int i = 0; i < n; i++) {
                        dc[i] = 0.007 * std::polar(1.0, 0.31 * (i + b + s)) / (1.0 + i);
                    }
                    double f = kp.band_occupancy(b, s), df = occupation_directions[matrix_k](b, s);
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            p(i, j, s) += f * (dc[i] * std::conj(c[j]) + c[i] * std::conj(dc[j])) +
                                          df * c[i] * std::conj(c[j]);
                            frozen_matrices.back()(i, j, s) += f * c[i] * std::conj(c[j]);
                        }
                    }
                    for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                        evec.pw_coeffs(ig, wf::spin_index(0), wf::band_index(b)) = dc[kp.gkvec().offset() + ig];
                    }
                    for (auto atom : evec.spl_num_atoms()) {
                        int base = npw + std::accumulate(nlo.begin(), nlo.begin() + atom.i, 0);
                        for (int i = 0; i < nlo[atom.i]; i++) {
                            evec.mt_coeffs(i, atom.li, wf::spin_index(0), wf::band_index(b)) = dc[base + i];
                        }
                    }
                }
                kp.generate_lapw_wave_functions(evec, delta, s);
            }
            matched_ptrs.push_back(&delta);
            matrix_k++;
        }
        auto coefficient_fields = density.lapw_density_matrix_fields(ks, coefficient_matrices);
        auto matched_fields     = density.lapw_orbital_fields_response(ks, matched_ptrs, {}, occupation_directions);
        for (int part : {0, 1}) {
            double error{0}, peak{0};
            auto compare = [&](auto a, auto b) {
                double residual = std::abs(a - b);
                error = std::isfinite(residual) ? std::max(error, residual) : std::numeric_limits<double>::infinity();
                peak  = std::max(peak, std::abs(b));
            };
            for (int s = 0; s < ctx->num_spins(); s++) {
                auto const& a = part == 0 ? coefficient_fields->component(s)
                                          : coefficient_fields->kinetic_density().component(s);
                auto const& b =
                        part == 0 ? matched_fields->component(s) : matched_fields->kinetic_density().component(s);
                for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                    compare(a.rg().f_pw_local(ig), b.rg().f_pw_local(ig));
                }
                for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                    compare(a.rg().value(ir), b.rg().value(ir));
                }
                for (auto atom : ctx->unit_cell().spl_num_atoms()) {
                    for (size_t i = 0; i < a.mt()[atom.i].size(); i++) {
                        compare(a.mt()[atom.i][i], b.mt()[atom.i][i]);
                    }
                }
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
            ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            check(part == 0 ? "PW/LO density matrix reproduces physical rho response"
                            : "PW/LO density matrix reproduces physical tau response",
                  error / std::max(1e-12, peak), 1e-10);
        }
        reject("coefficient fields reject missing k-point matrices",
               [&] { density.lapw_density_matrix_fields(ks, {}); });
        std::complex<double> saved_coefficient{0};
        if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
            saved_coefficient                = coefficient_matrices[0](0, 0, 0);
            coefficient_matrices[0](0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        reject("coefficient fields reject a nonfinite matrix collectively",
               [&] { density.lapw_density_matrix_fields(ks, coefficient_matrices); });
        if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
            coefficient_matrices[0](0, 0, 0) = saved_coefficient;
        }
        if (distributed_fft && ctx->comm().size() > 1) {
            if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
                coefficient_matrices[0](0, 0, 0) += 0.1;
            }
            reject("coefficient fields reject inconsistent k-point replicas",
                   [&] { density.lapw_density_matrix_fields(ks, coefficient_matrices); });
            if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
                coefficient_matrices[0](0, 0, 0) = saved_coefficient;
            }
        }
        verify_restored();
        auto lattice             = ctx->unit_cell().lattice_vectors();
        auto original_quadrature = ctx->cb().external_xc_points_;
        std::vector<r3::vector<double>> positions, centres;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            positions.push_back(ctx->unit_cell().atom(ia).position());
            centres.push_back(ctx->unit_cell().get_cartesian_coordinates(positions.back()));
        }
        for (int part = 0; part < 3; part++) {
            xc_geometry_direction_t direction(ctx->unit_cell().num_atoms());
            if (part != 1) {
                direction.atoms[0] = {0.23, -0.37, 0.19};
            }
            r3::matrix<double> strain;
            if (part != 0) {
                strain(0, 0) = 0.07;
                strain(1, 1) = -0.03;
                strain(2, 2) = 0.02;
                strain(0, 1) = -0.04;
                strain(2, 0) = 0.05;
            }
            auto dlattice = dot(strain, lattice);
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    direction.lattice[x + 3 * y] = dlattice(x, y);
                }
            }
            auto analytic = density.lapw_density_matrix_geometry_response(ks, frozen_matrices, direction);
            auto evaluate = [&](double step) {
                auto shifted = lattice;
                for (int y = 0; y < 3; y++) {
                    for (int x = 0; x < 3; x++) {
                        shifted(x, y) += step * dlattice(x, y);
                    }
                }
                ctx->unit_cell().set_lattice_vectors(shifted);
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    auto const& d = direction.atoms[ia];
                    r3::vector<double> dr{d[0], d[1], d[2]};
                    ctx->unit_cell().atom(ia).set_position(dot(inverse(shifted), centres[ia] + dr * step));
                }
                for (auto it : ks.spl_num_kpoints()) {
                    ks.get<double>(it.i)->update();
                }
                auto next                     = std::make_shared<xc_point_set_t>(*ctx);
                next->coordinates             = original_quadrature->coordinates;
                next->energy_weights          = original_quadrature->energy_weights;
                next->descriptor_weights      = original_quadrature->descriptor_weights;
                next->block_sizes             = original_quadrature->block_sizes;
                next->block_centres           = original_quadrature->block_centres;
                ctx->cb().external_xc_points_ = next;
                return density.lapw_density_matrix_fields(ks, frozen_matrices);
            };
            for (double step : {1e-4, 5e-5}) {
                auto plus = evaluate(step), minus = evaluate(-step);
                for (int field : {0, 1}) {
                    double error{0}, peak{0};
                    auto compare = [&](auto a, auto p, auto m) {
                        double residual = std::abs(a - (p - m) / (2 * step));
                        error           = std::isfinite(residual) ? std::max(error, residual)
                                                                  : std::numeric_limits<double>::infinity();
                        peak            = std::max(peak, std::abs(a));
                    };
                    for (int s = 0; s < ctx->num_spins(); s++) {
                        auto const& a = field == 0 ? analytic->component(s) : analytic->kinetic_density().component(s);
                        auto const& p = field == 0 ? plus->component(s) : plus->kinetic_density().component(s);
                        auto const& m = field == 0 ? minus->component(s) : minus->kinetic_density().component(s);
                        for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                            compare(a.rg().f_pw_local(ig), p.rg().f_pw_local(ig), m.rg().f_pw_local(ig));
                        }
                        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                            compare(a.rg().value(ir), p.rg().value(ir), m.rg().value(ir));
                        }
                        for (auto atom : ctx->unit_cell().spl_num_atoms()) {
                            for (size_t i = 0; i < a.mt()[atom.i].size(); i++) {
                                compare(a.mt()[atom.i][i], p.mt()[atom.i][i], m.mt()[atom.i][i]);
                            }
                        }
                    }
                    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                    check(field == 0 ? "PW/LO geometry rho response" : "PW/LO geometry tau response",
                          error / std::max(1e-12, peak), 3e-7);
                    check("nonzero geometry field response", peak > 1e-8 ? 0 : 1, 0);
                }
            }
            ctx->unit_cell().set_lattice_vectors(lattice);
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                ctx->unit_cell().atom(ia).set_position(positions[ia]);
            }
            for (auto it : ks.spl_num_kpoints()) {
                ks.get<double>(it.i)->update();
            }
            ctx->cb().external_xc_points_ = original_quadrature;
            if (part == 0) {
                auto bad = direction;
                if (ctx->comm().rank() == 0) {
                    bad.atoms.clear();
                }
                reject("geometry fields reject malformed directions collectively",
                       [&] { density.lapw_density_matrix_geometry_response(ks, frozen_matrices, bad); });
                bad = direction;
                if (ctx->comm().rank() == 0) {
                    bad.lattice[0] = std::numeric_limits<double>::quiet_NaN();
                }
                reject("geometry fields reject nonfinite directions collectively",
                       [&] { density.lapw_density_matrix_geometry_response(ks, frozen_matrices, bad); });
                if (ctx->comm().size() > 1) {
                    bad = direction;
                    if (ctx->comm().rank() == 0) {
                        bad.atoms[0][0] += 0.01;
                    }
                    reject("geometry fields reject inconsistent replicas",
                           [&] { density.lapw_density_matrix_geometry_response(ks, frozen_matrices, bad); });
                }
            }
            verify_restored();
        }
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
            auto field_response     = density.lapw_radial_fields_response(ks, radial_response);
            auto combined           = density.lapw_orbital_fields_response(ks, orbital_directions, radial_response);
            auto coefficient_radial = density.lapw_density_matrix_radial_response(ks, frozen_matrices, radial_response);
            double combined_error{0};
            for (int p : {0, 1}) {
                auto const& field  = p == 0 ? static_cast<Field4D const&>(*combined) : combined->kinetic_density();
                auto const& mapped = p == 0 ? static_cast<Field4D const&>(*coefficient_radial)
                                            : coefficient_radial->kinetic_density();
                double coefficient_error{0}, coefficient_peak{0};
                auto compare = [&](double a, double b) {
                    coefficient_error = std::isfinite(a) && std::isfinite(b)
                                                ? std::max(coefficient_error, std::abs(a - b))
                                                : std::numeric_limits<double>::infinity();
                    coefficient_peak  = std::max(coefficient_peak, std::abs(b));
                };
                for (int s = 0; s < ctx->num_spins(); s++) {
                    for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                        combined_error = std::max(combined_error, std::abs(field.component(s).rg().f_pw_local(ig)));
                        compare(std::abs(mapped.component(s).rg().f_pw_local(ig)), 0);
                    }
                }
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    for (size_t i = 0; i < field.scalar().mt()[ia].size(); i++) {
                        double up      = field_response[ia][p][0][i];
                        double dn      = mag ? field_response[ia][p][1][i] : 0;
                        combined_error = std::max(combined_error, std::abs(field.scalar().mt()[ia][i] - (up + dn)));
                        compare(mapped.scalar().mt()[ia][i], field.scalar().mt()[ia][i]);
                        if (mag) {
                            combined_error =
                                    std::max(combined_error, std::abs(field.vector(0).mt()[ia][i] - (up - dn)));
                            compare(mapped.vector(0).mt()[ia][i], field.vector(0).mt()[ia][i]);
                        }
                    }
                }
                ctx->comm().allreduce<double, mpi::op_t::max>(&coefficient_error, 1);
                ctx->comm().allreduce<double, mpi::op_t::max>(&coefficient_peak, 1);
                check(p == 0 ? "PW/LO density matrix radial rho response" : "PW/LO density matrix radial tau response",
                      coefficient_error / std::max(1e-12, coefficient_peak), 1e-12);
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
test_lapw_force_response(int mag, double pw_cutoff, bool core)
{
    std::vector<std::vector<double>> frozen_potential;
    std::vector<std::vector<double>> displaced_potential;
    std::vector<std::array<std::vector<double>, 2>> frozen_xc, displaced_xc;
    struct state_t
    {
        std::unique_ptr<Simulation_context> ctx;
        std::unique_ptr<K_point_set> ks;
        std::unique_ptr<DFT_ground_state> gs;
    };
    std::unique_ptr<state_t> reference;
    // Displaced SCFs independently check the analytic self-consistent basis response.
    auto radial_lagrangian = [&](std::vector<std::vector<double>> const& potential,
                                 std::vector<std::array<std::vector<double>, 2>> const& xc) {
        auto& ctx = *reference->ctx;
        auto& ks  = *reference->ks;
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_potential(potential[ic]);
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_xc_derivatives(xc[ic]);
        }
        ctx.unit_cell().generate_radial_functions(ctx.out());
        // Vary the basis, not the physical Hamiltonian or the reference orbitals.
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_potential(frozen_potential[ic]);
            ctx.unit_cell().atom_symmetry_class(ic).set_spherical_xc_derivatives(frozen_xc[ic]);
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
        auto state = std::make_unique<state_t>();
        state->ctx = context(mag, true, false, core, false, displacement, pw_cutoff, false, 0, "", core);
        auto& ctx  = state->ctx;
        ctx->cb().external_xc_geometry_          = functional;
        ctx->cb().external_xc_geometry_response_ = true;
        // Interstitial points stay fixed; the core test moves its resolved atom quadratures.
        ctx->cb().external_xc_pullback_             = [](auto const&, auto const&, auto const&, auto&) {};
        ctx->cb().external_xc_quadrature_direction_ = [](auto const&, auto const&, auto const&, auto&) {};
        if (core) {
            std::vector<int> attached;
            for (auto const& p : ctx->cb().external_xc_points_->coordinates) {
                int owner{-1};
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    auto centre = ctx->unit_cell().get_cartesian_coordinates(ctx->unit_cell().atom(ia).position());
                    auto r      = r3::vector<double>{p[0], p[1], p[2]} - centre;
                    if (r.length() <= 1.8 + 1e-10) {
                        owner = ia;
                    }
                }
                attached.push_back(owner);
            }
            ctx->cb().external_xc_pullback_ = [attached](auto const& c, auto const&, auto const& in, auto& out) {
                for (size_t p = 0; p < attached.size(); p++) {
                    if (attached[p] >= 0) {
                        for (int x = 0; x < 3; x++) {
                            out.atoms[attached[p]][x] += in.coordinates[p][x];
                        }
                    }
                }
                for (auto& atom : out.atoms) {
                    c.comm().allreduce(atom.data(), 3);
                }
            };
            ctx->cb().external_xc_quadrature_direction_ = [attached](auto const&, auto const&, auto const& in,
                                                                     auto& out) {
                for (size_t p = 0; p < attached.size(); p++) {
                    if (attached[p] >= 0) {
                        out.coordinates[p] = in.atoms[attached[p]];
                    }
                }
            };
        }
        state->ks = std::make_unique<K_point_set>(*ctx);
        auto& ks  = *state->ks;
        ks.add_kpoint({0.07, -0.03, 0.02}, 1);
        ks.initialize();
        state->gs = std::make_unique<DFT_ground_state>(ks);
        auto& gs  = *state->gs;
        gs.initial_state();
        bool converged{false};
        if (!frozen) {
            auto found = gs.find(core ? 1e-10 : 1e-11, core ? 1e-11 : 1e-12, 1e-11, 200, false);
            converged  = found.at("converged").get<bool>();
            if (!converged && ctx->comm().rank() == 0) {
                auto energies = found.at("etot_history").get<std::vector<double>>();
                std::cout << "LAPW force SCF final energy change=" << energies.back() - energies[energies.size() - 2]
                          << std::endl;
            }
            if (forces) {
                frozen_potential = gs.potential().get_spherical_potential();
                frozen_xc        = gs.potential().get_core_xc_derivatives();
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
        std::array<double, 5> result{free_energy(*ctx, ks, gs.density(), gs.potential()), derivative, direct, 0,
                                     ks.entropy_sum()};
        if (forces) {
            reference = std::move(state);
        } else if (!frozen) {
            displaced_potential = gs.potential().get_spherical_potential();
            displaced_xc        = gs.potential().get_core_xc_derivatives();
            result[3]           = radial_lagrangian(displaced_potential, displaced_xc);
        }
        return result;
    };
    int failures{0};
    for (bool frozen : core ? std::vector<bool>{false} : std::vector<bool>{false, true}) {
        auto centre = calculate(0, true, frozen);
        std::vector<std::vector<double>> scalar;
        std::vector<std::array<std::vector<double>, 2>> core_direction;
        double coupled_response{0}, total_derivative = centre[1], entropy_response{0};
        if (!frozen) {
            auto& gs        = *reference->gs;
            auto& potential = gs.potential();
            auto snapshot   = [&] {
                std::vector<double> values{gs.total_energy(), potential.energy_exc(gs.density())};
                for (int s = 0; s < reference->ctx->num_spins(); s++) {
                    for (auto const* f : {&gs.density().component(s), &gs.density().kinetic_density().component(s)}) {
                        auto field = potential_samples(*f);
                        values.insert(values.end(), field.begin(), field.end());
                    }
                }
                for (int ic = 0; ic < reference->ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                    auto const& cls = reference->ctx->unit_cell().atom_symmetry_class(ic);
                    for (auto rf : cls.atom_type().indexr()) {
                        for (int ir = 0; ir < cls.atom_type().num_mt_points(); ir++) {
                            values.push_back(cls.radial_function(ir, rf.idxrf));
                        }
                    }
                }
                return values;
            };
            auto before = snapshot();
            xc_geometry_direction_t geometry(reference->ctx->unit_cell().num_atoms());
            geometry.atoms[0] = {0.23, -0.37, 0.19};
            LAPW_response response(*reference->ks, gs.density(), potential);
            auto radial                = response.radial_geometry_response(geometry, 1e-8, 200);
            scalar                     = radial.scalar;
            core_direction             = radial.core;
            entropy_response           = radial.entropy;
            coupled_response           = gs.forces().lapw_radial_basis_derivative(scalar, radial.core);
            double complete_derivative = response.energy_geometry_derivative(geometry, radial);
            if (reference->ctx->comm().rank() == 0) {
                std::cout << "LAPW complete free-energy geometry derivative=" << complete_derivative
                          << " entropy=" << radial.entropy << std::endl;
            }
            auto const& total = gs.forces().calc_forces_total();
            total_derivative  = 0;
            for (int x = 0; x < 3; x++) {
                total_derivative -= geometry.atoms[0][x] * total(x, 0);
            }
            failures += std::abs(total_derivative - complete_derivative) > 2e-7;
            failures += snapshot() != before;
            if (reference->ctx->comm().rank() == 0) {
                std::cout << "LAPW force coupled basis derivative=" << coupled_response
                          << " total derivative=" << total_derivative << '\n';
            }
        }
        struct sample_t
        {
            std::array<double, 5> energies;
            std::vector<std::vector<double>> radial_potential;
            std::vector<std::array<std::vector<double>, 2>> core_xc;
        };
        std::map<double, sample_t> samples;
        // Reuse the overlapping fourth-order stencils rather than repeating displaced SCFs.
        auto sample = [&](double step) {
            auto it = samples.find(step);
            if (it == samples.end()) {
                auto values = calculate(step, false, frozen);
                it          = samples.emplace(step, sample_t{values, displaced_potential, displaced_xc}).first;
            }
            displaced_potential = it->second.radial_potential;
            displaced_xc        = it->second.core_xc;
            return it->second.energies;
        };
        // Fourth-order differences resolve the small SCF response without short-step cancellation.
        for (double h : {2e-3, 1e-3}) {
            auto plus            = sample(h);
            auto direction       = displaced_potential;
            auto xc_direction    = displaced_xc;
            auto minus           = sample(-h);
            auto inner_minus     = displaced_potential;
            auto inner_xc_minus  = displaced_xc;
            auto outer_plus      = sample(2 * h);
            auto outer_direction = displaced_potential;
            auto outer_xc        = displaced_xc;
            auto outer_minus     = sample(-2 * h);
            auto differentiate   = [&](int part) {
                return (8 * (plus[part] - minus[part]) - (outer_plus[part] - outer_minus[part])) / (12 * h);
            };
            double fd = differentiate(0), response = differentiate(3);
            double analytic_response{0};
            double potential_error{0}, potential_peak{0};
            if (!frozen) {
                for (size_t ic = 0; ic < direction.size(); ic++) {
                    for (size_t ir = 0; ir < direction[ic].size(); ir++) {
                        double roundoff = 4 * std::numeric_limits<double>::epsilon() *
                                          (8 * (std::abs(direction[ic][ir]) + std::abs(inner_minus[ic][ir])) +
                                           std::abs(outer_direction[ic][ir]) + std::abs(displaced_potential[ic][ir])) /
                                          (12 * h);
                        direction[ic][ir] = (8 * (direction[ic][ir] - inner_minus[ic][ir]) -
                                             (outer_direction[ic][ir] - displaced_potential[ic][ir])) /
                                            (12 * h);
                        potential_error =
                                std::max(potential_error, std::abs(direction[ic][ir] - scalar[ic][ir]) - roundoff);
                        potential_peak = std::max(potential_peak, std::abs(scalar[ic][ir]));
                        for (int part = 0; part < 2; part++) {
                            xc_direction[ic][part][ir] =
                                    (8 * (xc_direction[ic][part][ir] - inner_xc_minus[ic][part][ir]) -
                                     (outer_xc[ic][part][ir] - displaced_xc[ic][part][ir])) /
                                    (12 * h);
                        }
                    }
                }
                radial_lagrangian(frozen_potential, frozen_xc);
                analytic_response = reference->gs->forces().lapw_radial_basis_derivative(direction, xc_direction);
            }
            double error          = std::abs(fd - centre[1] - response);
            double analytic_error = std::abs(fd - centre[1] - analytic_response);
            double coupled_error  = std::abs(fd - centre[1] - coupled_response);
            double total_error    = std::abs(fd - total_derivative);
            if (mpi::Communicator::world().rank() == 0) {
                std::cout << "LAPW force decomposition spin=" << mag << " pw cutoff=" << pw_cutoff << " core=" << core
                          << " frozen radial basis=" << frozen << " h=" << h << " derivative=" << centre[1]
                          << " FD=" << fd << " direct FD=" << differentiate(2)
                          << " energy identity=" << centre[0] - centre[2] << " radial response=" << response
                          << " analytic radial response=" << analytic_response << " error=" << error
                          << " analytic error=" << analytic_error << " coupled error=" << coupled_error
                          << " total error=" << total_error
                          << " radial potential error=" << potential_error / std::max(1e-12, potential_peak) << '\n';
            }
            failures += !core && (!std::isfinite(error) || error > 2e-6);
            failures += !core && (!std::isfinite(analytic_error) || analytic_error > 2e-6);
            failures += std::abs(response - analytic_response) > 2e-7;
            failures += !core && (!std::isfinite(coupled_error) || coupled_error > 2e-6);
            failures += std::abs(response - coupled_response) > 2e-7;
            failures += !std::isfinite(total_error) || total_error > 2e-6;
            failures += potential_error / std::max(1e-12, potential_peak) > 3e-5;
            failures += !frozen && std::abs(response) < 1e-5;
            failures += !frozen && std::abs(differentiate(4) - entropy_response) > 2e-7;
            for (auto const& point : {centre, plus, minus, outer_plus, outer_minus}) {
                failures += !std::isfinite(point[0] - point[2]) || std::abs(point[0] - point[2]) > 2e-7;
            }
            failures += std::abs(fd - differentiate(2)) > 2e-7;
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
    if (argc > 1 && std::string(argv[1]) == "coupled-response") {
        for (int mag : {0, 1}) {
            result += call_test("coupled LAPW response", test_lapw_coupled_response, mag, false);
            if (mpi::Communicator::world().size() > 1) {
                result += call_test("distributed coupled LAPW response", test_lapw_coupled_response, mag, true);
            }
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc > 1 && std::string(argv[1]) == "density-response") {
        for (int mag : {0, 1}) {
            result += call_test("constrained LAPW density response", test_lapw_density_response, mag, false);
            if (mpi::Communicator::world().size() > 1) {
                result += call_test("distributed LAPW density response", test_lapw_density_response, mag, true);
            }
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc > 1 && std::string(argv[1]) == "occupation-response") {
        for (int mag : {0, 1}) {
            result += call_test("weighted k-point occupation response", test_occupation_response, mag, false, false);
            if (mpi::Communicator::world().size() > 1) {
                result += call_test("distributed occupation response", test_occupation_response, mag, true, false);
            }
        }
        result += call_test("Gaussian occupation response", test_occupation_response, 0, false, true);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc > 1 && (std::string(argv[1]) == "force-response" || std::string(argv[1]) == "core-force")) {
        int first = argc > 3 ? std::stoi(argv[3]) : 0;
        int last  = argc > 3 ? first : 1;
        if (first < 0 || first > 1) {
            RTE_THROW("LAPW force test requires spin mode 0 or 1");
        }
        for (int mag = first; mag <= last; mag++) {
            result += call_test("external LAPW force response", test_lapw_force_response, mag,
                                argc > 2 ? std::stod(argv[2]) : 9.0, std::string(argv[1]) == "core-force");
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
    for (int mag : {0, 1}) {
        result += call_test("coupled LAPW response", test_lapw_coupled_response, mag, false);
        result += call_test("constrained LAPW density response", test_lapw_density_response, mag, false);
        if (mpi::Communicator::world().size() > 1) {
            result += call_test("distributed coupled LAPW response", test_lapw_coupled_response, mag, true);
            result += call_test("distributed LAPW density response", test_lapw_density_response, mag, true);
        }
        result += call_test("weighted k-point occupation response", test_occupation_response, mag, false, false);
        if (mpi::Communicator::world().size() > 1) {
            result += call_test("distributed occupation response", test_occupation_response, mag, true, false);
        }
    }
    result += call_test("Gaussian occupation response", test_occupation_response, 0, false, true);
    sirius::finalize();
    return std::min(result, 1);
}
