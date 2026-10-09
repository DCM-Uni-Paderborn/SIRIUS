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
#include <sstream>
#include <tuple>

using namespace sirius;

namespace {

// Independent occupied-state radial Pulay contraction used only in the checks below.
double
radial_basis_derivative(Potential& potential, K_point_set& kset, std::vector<std::vector<double>> const& directions__,
                        std::vector<std::array<std::vector<double>, 2>> const& xc_directions__ = {},
                        bool update_enu__                                                      = true)
{
    auto& ctx_  = kset.ctx();
    auto& uc    = ctx_.unit_cell();
    int invalid = !ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU ||
                  ctx_.valence_relativity() != relativity_t::none || ctx_.so_correction() || ctx_.num_mag_dims() == 3 ||
                  ctx_.hubbard_correction() ||
                  (uc.num_core_electrons() != 0 && ctx_.core_relativity() != relativity_t::none) ||
                  (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()) ||
                  directions__.size() != uc.num_atom_symmetry_classes() ||
                  (!xc_directions__.empty() && xc_directions__.size() != directions__.size());
    if (!invalid) {
        for (int ic = 0; ic < uc.num_atom_symmetry_classes(); ic++) {
            invalid |= directions__[ic].size() != uc.atom_symmetry_class(ic).atom_type().num_mt_points() ||
                       !std::all_of(directions__[ic].begin(), directions__[ic].end(),
                                    [](double x) { return std::isfinite(x); });
            if (!xc_directions__.empty()) {
                auto const& cls = uc.atom_symmetry_class(ic);
                for (auto const& field : xc_directions__[ic]) {
                    invalid |=
                            !field.empty() &&
                            (field.size() != directions__[ic].size() ||
                             !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); }) ||
                             (cls.atom_type().num_core_electrons() != 0 && cls.spherical_xc_derivatives()[0].empty()));
                }
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW radial basis derivative requires nonrelativistic scalar/collinear valence states and valid "
                  "directions");
    }

    // Differentiate the stored basis without updating its potential or energies.
    Hamiltonian0<double> h0(potential, false);
    std::vector<radial_basis_response_t> response;
    for (int ic = 0; ic < uc.num_atom_symmetry_classes(); ic++) {
        response.push_back(uc.atom_symmetry_class(ic).radial_basis_potential_response(
                directions__[ic], xc_directions__.empty() ? std::array<std::vector<double>, 2>{} : xc_directions__[ic],
                update_enu__));
    }
    auto matrices = h0.lapw_radial_basis_response(response);

    int ns = ctx_.num_spins();
    double result{0};
    for (auto it : kset.spl_num_kpoints()) {
        auto& kp = *kset.get<double>(it.i);
        Matching_coefficients matching(uc, kp.gkvec());
        std::vector<mdarray<std::complex<double>, 2>> alm, dalm;
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            auto const& atom = uc.atom(ia);
            std::array<index_range, 2> shape{kp.gkvec().count(), atom.type().mt_aw_basis_size()};
            alm.emplace_back(shape);
            dalm.emplace_back(shape);
            matching.generate<false>(atom, alm.back());
            matching.generate_potential_response<false>(atom, response[atom.symmetry_class().id()], dalm.back());
        }
        for (int s = 0; s < ns; s++) {
            for (int band = 0; band < ctx_.num_bands(); band++) {
                double weight = kp.weight() * kp.band_occupancy(band, s) / kp.comm().size();
                if (weight == 0) {
                    continue;
                }
                auto all   = kp.spinor_wave_functions().gather(wf::spin_index(s), wf::band_index(band));
                int offset = kp.gkvec().num_gvec();
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    auto const& type = uc.atom(ia).type();
                    int n = type.mt_basis_size(), naw = type.mt_aw_basis_size();
                    std::vector<std::complex<double>> c(n), dc(n);
                    for (int i = 0; i < naw; i++) {
                        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                            c[i] += alm[ia](ig, i) * all[kp.gkvec().offset() + ig];
                            dc[i] += dalm[ia](ig, i) * all[kp.gkvec().offset() + ig];
                        }
                    }
                    kp.comm().allreduce(c.data(), naw);
                    kp.comm().allreduce(dc.data(), naw);
                    for (int i = naw; i < n; i++) {
                        c[i] = all[offset + i];
                    }
                    auto const& d = matrices[ia];
                    double energy = kp.band_energy(band, s);
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            result +=
                                    weight * std::real(2.0 * std::conj(dc[i]) *
                                                               (h0.hmt(ia)(i, j, s) - energy * d.overlap(i, j)) * c[j] +
                                                       std::conj(c[i]) * (d.h(i, j, s) - energy * d.o(i, j)) * c[j]);
                        }
                    }
                    offset += n;
                }
            }
        }
    }
    ctx_.comm().allreduce(&result, 1);
    return result;
}

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
    bool spinor = in.num_spins == 4;
    for (int s = 0; s < (spinor ? 1 : in.num_spins); s++) {
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
            double local      = 0.02 * r * r + 0.01 * t * t + 0.005 * r * t;
            double descriptor = spinor && s > 0 ? 0 : nonlocal;
            out.vrho[j]       = w * (0.04 * r + 0.005 * t) + d * descriptor;
            out.vtau[j]       = w * (0.02 * t + 0.005 * r) + 0.3 * d * descriptor;
            for (int x = 0; x < 3; x++) {
                int k = p + in.num_points * (x + 3 * s);
                local += 0.002 * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = 0.004 * w * in.gradient[k];
            }
            energy += w * local;
            if (out.geometry) {
                out.geometry->energy_weights[p] += local;
                out.geometry->descriptor_weights[p] += descriptor * (r + 0.3 * t);
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
    bool spinor = in.num_spins == 4;
    for (int s = 0; s < (spinor ? 1 : in.num_spins); s++) {
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
            double descriptor      = spinor && s > 0 ? 0 : nonlocal;
            double base_descriptor = spinor && s > 0 ? 0 : base_nonlocal;
            out.vrho[j]  = scale * (w * (0.04 * dr + 0.005 * dt) + dw * (0.04 * r + 0.005 * t) + d * descriptor +
                                   dd * base_descriptor);
            out.vtau[j]  = scale * (w * (0.02 * dt + 0.005 * dr) + dw * (0.02 * t + 0.005 * r) +
                                   0.3 * (d * descriptor + dd * base_descriptor));
            double local = (0.04 * r + 0.005 * t) * dr + (0.02 * t + 0.005 * r) * dt;
            for (int x = 0; x < 3; x++) {
                int k            = p + in.num_points * (x + 3 * s);
                out.vgradient[k] = scale * 0.004 * (w * direction.gradient[k] + dw * in.gradient[k]);
                local += 0.004 * in.gradient[k] * direction.gradient[k];
            }
            if (out.geometry) {
                out.geometry->energy_weights[p] += scale * local;
                out.geometry->descriptor_weights[p] +=
                        scale * (descriptor * (r + 0.3 * t) + base_descriptor * (dr + 0.3 * dt));
            }
        }
    }
}

auto
context(int mag, bool distributed_fft, bool second_variation, bool core = false, bool relativistic_core = false,
        double displacement = 0, double pw_cutoff = 5.0, bool simple_radial_integrals = false, int auto_enu = 0,
        std::string smearing = "", bool moving_core_grid = false, std::string processing_unit = "CPU",
        bool spin_orbit = false, std::string valence_relativity = "none", bool external_scalar = false,
        std::string core_relativity = "", int radial_points = 0)
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
    config["parameters"]["fixed_mag"]         = mag == 1 && valence_relativity != "dirac" ? 0.4 : 0.0;
    config["control"]["use_second_variation"] = second_variation;
    config["control"]["processing_unit"]      = processing_unit;
    config["parameters"]["so_correction"]     = spin_orbit;
    config["control"]["mpi_grid_dims"]        = {distributed_fft ? mpi::Communicator::world().size() : 1, 1};
    config["parameters"]["core_relativity"] =
            core_relativity.empty() ? (relativistic_core ? "dirac" : "none") : core_relativity;
    config["parameters"]["valence_relativity"] = valence_relativity;
    // Spinor bands hold one electron; retain virtual states for the core response.
    config["parameters"]["num_bands"] =
            core ? (moving_core_grid ? 16 : 8) * (mag == 3 || valence_relativity == "dirac" ? 2 : 1)
                 : (valence_relativity == "dirac" ? 8 : 4);
    config["parameters"]["pw_cutoff"] = pw_cutoff;
    if (moving_core_grid) {
        config["mixer"]["type"] = "anderson_stable";
    }
    if (!smearing.empty()) {
        config["parameters"]["smearing"] = smearing;
        if (valence_relativity == "dirac") {
            config["parameters"]["smearing_width"] = 0.2;
        }
    }
    config["settings"]["simple_lapw_ri"] = simple_radial_integrals;
    auto ctx                             = std::make_unique<Simulation_context>(config);
    ctx->unit_cell().set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    std::string label = core ? "Ne" : "He";
    auto& type        = ctx->unit_cell().add_atom_type(label);
    type.zn(core ? 10 : 2);
    type.set_radial_grid(radial_grid_t::power, radial_points ? radial_points : (core ? 3201 : 801), 1e-6, 1.8, 2.0);
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
            {mag == 3 ? 0.09 : 0, mag == 3 ? 0.13 : 0, mag ? 0.2 : 0.0});
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
    // Scalar-relativistic fixtures test the bare basis and H/O response before enabling external XC.
    if (valence_relativity == "none" || external_scalar) {
        ctx->cb().external_xc_points_   = std::move(points);
        ctx->cb().external_xc_          = functional;
        ctx->cb().external_xc_response_ = functional_response;
        ctx->cb().external_xc_spinor_   = mag == 3;
    }
    ctx->initialize();
    return ctx;
}

std::vector<std::complex<double>>
lapw_coefficients(K_point<double> const& kp, int spin, int band)
{
    auto all   = kp.spinor_wave_functions().gather(wf::spin_index(spin), wf::band_index(band));
    int offset = kp.gkvec().num_gvec();
    std::vector<std::complex<double>> c(all.begin(), all.begin() + offset);
    for (int ia = 0; ia < kp.ctx().unit_cell().num_atoms(); ia++) {
        auto const& type = kp.ctx().unit_cell().atom(ia).type();
        c.insert(c.end(), all.begin() + offset + type.mt_aw_basis_size(), all.begin() + offset + type.mt_basis_size());
        offset += type.mt_basis_size();
    }
    return c;
}

double
occupied_lapw_lagrangian(Simulation_context const& ctx, K_point<double> const& kp, int spin,
                         la::dmatrix<std::complex<double>> const& h, la::dmatrix<std::complex<double>> const& o)
{
    double value{0};
    for (int b = 0; b < ctx.num_bands(); b++) {
        auto c        = lapw_coefficients(kp, spin, b);
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

std::array<std::vector<double>, 2>
lapw_field_samples(Density const& density)
{
    std::array<std::vector<double>, 2> result;
    for (int part : {0, 1}) {
        auto const& fields = part == 0 ? static_cast<Field4D const&>(density) : density.kinetic_density();
        for (int s = 0; s <= density.ctx().num_mag_dims(); s++) {
            auto values = potential_samples(fields.component(s));
            result[part].insert(result[part].end(), values.begin(), values.end());
        }
    }
    return result;
}

void
check_coefficient_geometry(Test_check& check, Density const& density, K_point_set& ks,
                           std::vector<mdarray<std::complex<double>, 3>> const& matrices)
{
    auto& ctx       = ks.ctx();
    auto lattice    = ctx.unit_cell().lattice_vectors();
    auto quadrature = ctx.cb().external_xc_points_;
    std::vector<r3::vector<double>> positions, centres;
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        positions.push_back(ctx.unit_cell().atom(ia).position());
        centres.push_back(ctx.unit_cell().get_cartesian_coordinates(positions.back()));
    }
    for (int part = 0; part < 3; part++) {
        xc_geometry_direction_t direction(ctx.unit_cell().num_atoms());
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
        auto analytic = lapw_field_samples(*density.lapw_density_matrix_geometry_response(ks, matrices, direction));
        auto evaluate = [&](double step) {
            auto shifted = lattice;
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    shifted(x, y) += step * dlattice(x, y);
                }
            }
            ctx.unit_cell().set_lattice_vectors(shifted);
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                auto const& d = direction.atoms[ia];
                ctx.unit_cell().atom(ia).set_position(
                        dot(inverse(shifted), centres[ia] + r3::vector<double>{d[0], d[1], d[2]} * step));
            }
            for (auto it : ks.spl_num_kpoints()) {
                ks.get<double>(it.i)->update();
            }
            auto next                    = std::make_shared<xc_point_set_t>(ctx);
            next->coordinates            = quadrature->coordinates;
            next->energy_weights         = quadrature->energy_weights;
            next->descriptor_weights     = quadrature->descriptor_weights;
            next->block_sizes            = quadrature->block_sizes;
            next->block_centres          = quadrature->block_centres;
            ctx.cb().external_xc_points_ = next;
            return lapw_field_samples(*density.lapw_density_matrix_fields(ks, matrices));
        };
        for (double step : {1e-4, 5e-5}) {
            auto plus = evaluate(step), minus = evaluate(-step);
            for (int field : {0, 1}) {
                double error{0}, peak{0};
                for (size_t i = 0; i < analytic[field].size(); i++) {
                    double residual = std::abs(analytic[field][i] - (plus[field][i] - minus[field][i]) / (2 * step));
                    error           = std::isfinite(residual) ? std::max(error, residual)
                                                              : std::numeric_limits<double>::infinity();
                    peak            = std::max(peak, std::abs(analytic[field][i]));
                }
                ctx.comm().allreduce<double, mpi::op_t::max>(&error, 1);
                ctx.comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                check(field == 0 ? "PW/LO geometry rho response" : "PW/LO geometry tau response",
                      error / std::max(1e-12, peak), 3e-7);
                check("nonzero geometry field response", peak > 1e-8 ? 0 : 1, 0);
            }
        }
        ctx.unit_cell().set_lattice_vectors(lattice);
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            ctx.unit_cell().atom(ia).set_position(positions[ia]);
        }
        for (auto it : ks.spl_num_kpoints()) {
            ks.get<double>(it.i)->update();
        }
        ctx.cb().external_xc_points_ = quadrature;
        if (part == 0) {
            auto bad = direction;
            if (ctx.comm().rank() == 0) {
                bad.atoms.clear();
            }
            check.reject("geometry fields reject malformed directions collectively",
                         [&] { density.lapw_density_matrix_geometry_response(ks, matrices, bad); });
            bad = direction;
            if (ctx.comm().rank() == 0) {
                bad.lattice[0] = std::numeric_limits<double>::quiet_NaN();
            }
            check.reject("geometry fields reject nonfinite directions collectively",
                         [&] { density.lapw_density_matrix_geometry_response(ks, matrices, bad); });
            if (ctx.comm().size() > 1) {
                bad = direction;
                if (ctx.comm().rank() == 0) {
                    bad.atoms[0][0] += 0.01;
                }
                check.reject("geometry fields reject inconsistent replicas",
                             [&] { density.lapw_density_matrix_geometry_response(ks, matrices, bad); });
            }
        }
    }
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
                     std::vector<double>* hartree = nullptr, std::vector<double>* xc = nullptr,
                     Density* valence = nullptr)
{
    std::vector<std::complex<double>> saved;
    for (auto it : ks.spl_num_kpoints()) {
        auto kp   = ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        Matching_coefficients matching(ctx.unit_cell(), kp->gkvec());
        std::vector<mdarray<std::complex<double>, 2>> alm;
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx.unit_cell().atom(ia);
            alm.emplace_back(std::array<index_range, 2>{kp->gkvec().count(), atom.type().mt_aw_basis_size()});
            matching.generate<false>(atom, alm.back());
        }
        // Reconstruct every physical band, including the second-variation spinors.
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int b = 0; b < ctx.num_bands(); b++) {
                auto spin = wf::spin_index(s);
                auto band = wf::band_index(b);
                auto data = psi.at(memory_t::host, 0, spin, band);
                saved.insert(saved.end(), data, data + psi.ld());
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    int naw = ctx.unit_cell().atom(ia).type().mt_aw_basis_size();
                    std::vector<std::complex<double>> c(naw);
                    for (int xi = 0; xi < naw; xi++) {
                        for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                            c[xi] += alm[ia](ig, xi) * psi.pw_coeffs(ig, spin, band);
                        }
                    }
                    kp->comm().allreduce(c.data(), naw);
                    auto location = psi.spl_num_atoms().location(atom_index_t::global(ia));
                    if (location.ib == kp->comm().rank()) {
                        auto local = location.index_local;
                        for (int xi = 0; xi < naw; xi++) {
                            psi.mt_coeffs(xi, local, spin, band) = c[xi];
                        }
                    }
                }
            }
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
        if (valence) {
            valence->generate<double>(ks, false, false, true);
        }
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
                for (int s = 0; s < (ctx.num_mag_dims() == 3 ? 4 : ctx.num_spins()); s++) {
                    if (ctx.num_mag_dims() == 3) {
                        auto const& component = fields->component(s).mt()[ia];
                        result.insert(result.end(), component.begin(), component.end());
                        continue;
                    }
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
lapw_spectrum_checks(Hamiltonian_k<double> const& hk, la::dmatrix<std::complex<double>>& dh,
                     la::dmatrix<std::complex<double>> const* overlap_direction  = nullptr,
                     la::dmatrix<std::complex<double>> const* physical_direction = nullptr)
{
    auto& ctx  = hk.H0().ctx();
    auto& kp   = hk.kp();
    bool dirac = ctx.valence_relativity() == relativity_t::dirac;
    int n      = dirac ? 2 * kp.num_gkvec() + ctx.unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
    int bs     = ctx.cyclic_block_size();
    la::dmatrix<std::complex<double>> h(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> s(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> ds(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> shifted_h(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> shifted_s(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> physical_s(n, n, ctx.blacs_grid(), bs, bs);
    la::dmatrix<std::complex<double>> shifted_physical(n, n, ctx.blacs_grid(), bs, bs);
    if (dirac) {
        hk.set_dirac_h_o(h, s);
    } else {
        hk.set_fv_h_o(0, h, s);
    }
    if (physical_direction) {
        hk.set_fv_h_o(0, shifted_h, physical_s, true);
    }
    for (int j = 0; j < ds.num_cols_local(); j++) {
        for (int i = 0; i < ds.num_rows_local(); i++) {
            int row = ds.irow(i), col = ds.icol(j);
            ds(i, j) = overlap_direction ? (*overlap_direction)(i, j)
                                         : 0.03 * s(i, j) + 0.002 * std::polar(1.0, 0.7 * (row - col)) /
                                                                    ((1.0 + row) * (1.0 + col));
        }
    }
    auto full_ds                 = ds.get_full_matrix();
    auto full_physical           = physical_direction ? physical_s.get_full_matrix() : s.get_full_matrix();
    auto full_physical_direction = physical_direction ? physical_direction->get_full_matrix() : ds.get_full_matrix();
    std::array<double, 4> errors{};
    for (bool degenerate : {false, true}) {
        if (degenerate) {
            for (int j = 0; j < h.num_cols_local(); j++) {
                for (int i = 0; i < h.num_rows_local(); i++) {
                    h(i, j) = 0.25 * s(i, j);
                }
            }
            if (physical_direction) {
                // A common degenerate physical norm is gauge independent.
                for (int j = 0; j < s.num_cols_local(); j++) {
                    for (int i = 0; i < s.num_rows_local(); i++) {
                        physical_s(i, j) = s(i, j);
                    }
                }
                full_physical           = s.get_full_matrix();
                full_physical_direction = ds.get_full_matrix();
            }
        }
        auto physical_ds = physical_direction ? (degenerate ? &ds : physical_direction) : nullptr;
        LAPW_eigensystem reference(hk, h, s, physical_direction ? &physical_s : nullptr);
        auto direction         = reference.perturbation(dh, ds, physical_ds);
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
                    trace += response(i, j) * full_physical(j, i) + p(i, j) * full_physical_direction(j, i);
                }
            }
            errors[2]   = std::max(errors[2], std::abs(trace - (energy_sum - slope_sum * dmu)));
            auto sample = [&](double step) {
                for (int j = 0; j < h.num_cols_local(); j++) {
                    for (int i = 0; i < h.num_rows_local(); i++) {
                        shifted_h(i, j) = h(i, j) + step * dh(i, j);
                        shifted_s(i, j) = s(i, j) + step * ds(i, j);
                        if (physical_direction) {
                            shifted_physical(i, j) = physical_s(i, j) + step * (*physical_ds)(i, j);
                        }
                    }
                }
                LAPW_eigensystem shifted(hk, shifted_h, shifted_s, physical_direction ? &shifted_physical : nullptr);
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
            expect_rejection([&] { reference.perturbation(dh, ds, physical_ds); });
            if (kp.comm().rank() == 0) {
                dh(0, 0) = saved;
            }
            if (physical_direction) {
                expect_rejection([&] { reference.perturbation(dh, ds); });
                if (kp.comm().size() > 1) {
                    expect_rejection(
                            [&] { LAPW_eigensystem bad(hk, h, s, kp.comm().rank() == 0 ? &physical_s : nullptr); });
                }
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
test_lapw_spinor_spectrum(bool distributed, std::string relativity)
{
    auto ctx = context(3, distributed, true, false, false, 0, 5.0, false, 0, "", false, "CPU", false, relativity, true);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    Hamiltonian0<double> h0(gs.potential(), false);
    auto& kp = *ks.get<double>(0);
    Hamiltonian_k<double> hk(h0, kp);
    int n = kp.gklo_basis_size(), nf = ctx->num_fv_states(), nb = ctx->num_bands(), bs = ctx->cyclic_block_size();
    int failures{0};
    Test_check check(failures, ctx->comm(), "two-stage LAPW spinor spectrum " + relativity);
    using matrix_t = la::dmatrix<std::complex<double>>;
    std::array<std::unique_ptr<matrix_t>, 4> h, dh, shifted_h;
    LAPW_spinor_eigensystem::matrix_set_t base{}, direction{}, shifted{};
    for (int s = 0; s < 4; s++) {
        for (auto* set : {&h, &dh, &shifted_h}) {
            (*set)[s] = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
        }
        base[s]      = h[s].get();
        direction[s] = dh[s].get();
        shifted[s]   = shifted_h[s].get();
    }
    matrix_t overlap(n, n, ctx->blacs_grid(), bs, bs), ds(n, n, ctx->blacs_grid(), bs, bs);
    matrix_t shifted_s(n, n, ctx->blacs_grid(), bs, bs);
    constexpr double width = 0.12;
    auto occupations       = [&](auto const& spectrum, double mu) {
        std::vector<double> f;
        for (double energy : spectrum.energies()) {
            f.push_back(smearing::fermi_dirac::occupancy(mu - energy, width));
        }
        return f;
    };
    for (bool degenerate : {false, true}) {
        for (int j = 0; j < overlap.num_cols_local(); j++) {
            for (int i = 0; i < overlap.num_rows_local(); i++) {
                int row = overlap.irow(i), col = overlap.icol(j);
                double diagonal = row == col ? 1.0 : 0.0, metric = 1 + 0.01 * row;
                auto wave     = std::polar(1.0, 0.4 * (row - col)) / ((1.0 + row) * (1.0 + col));
                overlap(i, j) = diagonal * metric + (degenerate ? 0.0 : 0.006) * wave;
                ds(i, j)      = 0.015 * diagonal + 0.01 * wave;
                for (int s = 0; s < 4; s++) {
                    double energy  = row < nf ? -0.3 : 0.4 + 0.07 * row;
                    (*h[s])(i, j)  = degenerate
                                             ? diagonal * metric * (s == 0 ? energy : 0.035 * (s + 1))
                                             : diagonal * (s == 0 ? -0.4 + 0.3 * row : 0.02 * (s + 1)) +
                                                      0.012 * (s + 1) * std::polar(1.0, 0.5 * (s + 1) * (row - col)) /
                                                              ((1.0 + row) * (1.0 + col));
                    (*dh[s])(i, j) = 0.013 * (s + 1) * diagonal * std::cos(row + 1.0) / (row + 1.0) +
                                     0.008 * (s + 1) * std::polar(1.0, 0.3 * (s + 1) * (row - col)) / (1.0 + row + col);
                }
            }
        }
        LAPW_spinor_eigensystem reference(hk, base, overlap);
        auto tangent = reference.perturbation(direction, ds);
        double mu    = 0.5 * (reference.energies().front() + reference.energies().back());
        auto f       = occupations(reference, mu);
        std::vector<double> slopes;
        double slope_sum{0}, energy_sum{0}, count = std::accumulate(f.begin(), f.end(), 0.0);
        for (int b = 0; b < nb; b++) {
            slopes.push_back(-smearing::fermi_dirac::delta(mu - reference.energies()[b], width));
            slope_sum += slopes.back();
            energy_sum += slopes.back() * tangent.spinor.energies[b];
        }
        auto p       = reference.density_matrix(f);
        auto full_ds = ds.get_full_matrix();
        std::complex<double> charge{0};
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                charge += p(i, j, 0) * reference.overlap()(j, i);
            }
        }
        check("spinor density preserves the metric electron count", std::abs(charge - count), 3e-10);
        if (degenerate) {
            double bnorm = 0.035 * std::sqrt(4.0 + 9.0 + 16.0), error{0};
            for (int s = 0; s < 4; s++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        double expected = i == j && i < nf
                                                  ? (s == 0 ? f.front() + f.back()
                                                            : (f.back() - f.front()) * 0.035 * (s + 1) / bnorm) /
                                                            (1 + 0.01 * i)
                                                  : 0;
                        error           = std::max(error, std::abs(p(i, j, s) - expected));
                    }
                }
            }
            check("Pauli density matches the analytic tilted-spin solution", error, 3e-12);
        }
        for (bool fixed_charge : {false, true}) {
            double dmu    = fixed_charge ? energy_sum / slope_sum : 0.07;
            auto response = reference.density_matrix_response(tangent, f, slopes, dmu);
            std::complex<double> trace{0};
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    trace += response(i, j, 0) * reference.overlap()(j, i) + p(i, j, 0) * full_ds(j, i);
                }
            }
            check("spinor response preserves the metric charge constraint",
                  std::abs(trace - (energy_sum - slope_sum * dmu)), 3e-10);
            std::vector<double> sampled_energies;
            auto sample = [&](double step) {
                for (int j = 0; j < overlap.num_cols_local(); j++) {
                    for (int i = 0; i < overlap.num_rows_local(); i++) {
                        shifted_s(i, j) = overlap(i, j) + step * ds(i, j);
                        for (int s = 0; s < 4; s++) {
                            (*shifted_h[s])(i, j) = (*h[s])(i, j) + step * (*dh[s])(i, j);
                        }
                    }
                }
                LAPW_spinor_eigensystem spectrum(hk, shifted, shifted_s);
                sampled_energies = spectrum.energies();
                return spectrum.density_matrix(occupations(spectrum, mu + step * dmu));
            };
            for (double step : {1e-4, 5e-5}) {
                auto plus          = sample(step);
                auto plus_energies = sampled_energies;
                auto minus         = sample(-step);
                double error{0}, peak{0};
                for (size_t i = 0; i < response.size(); i++) {
                    double e = std::abs(response[i] - (plus[i] - minus[i]) / (2 * step));
                    error    = std::isfinite(e) ? std::max(error, e) : std::numeric_limits<double>::infinity();
                    peak     = std::max(peak, std::abs(response[i]));
                }
                check("two-stage Pauli density response", error / std::max(1.0, peak), 3e-7);
                if (!degenerate) {
                    error = 0;
                    for (int b = 0; b < nb; b++) {
                        error = std::max(error, std::abs(tangent.spinor.energies[b] -
                                                         (plus_energies[b] - sampled_energies[b]) / (2 * step)));
                    }
                    check("two-stage spinor energy response", error, 3e-7);
                }
            }
        }
        auto saved        = tangent.reference;
        tangent.reference = nullptr;
        check.reject("spinor tangent rejects a different reference",
                     [&] { reference.density_matrix_response(tangent, f, slopes); });
        tangent.reference = saved;
        auto missing      = direction;
        missing[2]        = nullptr;
        check.reject("spinor response rejects a missing Pauli matrix", [&] { reference.perturbation(missing, ds); });
        auto saved_value = tangent.frame[0];
        if (kp.comm().rank() == 0) {
            tangent.frame[0] = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("spinor response rejects a nonfinite frame collectively",
                     [&] { reference.density_matrix_response(tangent, f, slopes); });
        tangent.frame[0] = saved_value;
    }
    for (int j = 0; j < overlap.num_cols_local(); j++) {
        for (int i = 0; i < overlap.num_rows_local(); i++) {
            (*h[0])(i, j) = -0.3 * overlap(i, j);
        }
    }
    LAPW_spinor_eigensystem cut(hk, base, overlap);
    check.reject("degenerate first-variation truncation is rejected", [&] { cut.perturbation(direction, ds); });
    return failures;
}

int
test_occupation_response(int mag, bool distributed, bool gaussian)
{
    auto ctx =
            context(mag, distributed, mag == 3, false, false, 0, 5.0, false, 0, gaussian ? "gaussian" : "fermi_dirac");
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    ks.initialize();
    int failures{0}, ns = ctx->num_spinors(), nb = ctx->num_bands();
    Test_check check(failures, ctx->comm(),
                     "occupation response spinors=" + std::to_string(ns) +
                             " distributed FFT=" + std::to_string(distributed));
    std::vector<mdarray<double, 2>> directions;
    for (auto it : ks.spl_num_kpoints()) {
        directions.emplace_back(std::array<index_range, 2>{nb, ns});
        for (int s = 0; s < ns; s++) {
            for (int b = 0; b < nb; b++) {
                directions.back()(b, s) = 0.07 * std::cos(b + 2 * it.i + s) + 0.1 * s;
            }
        }
    }
    check.reject("missing chemical potential rejected", [&] { ks.band_occupation_response(directions); });
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
    for (double moment : ns == 2 ? std::vector<double>{0.4, 0.0} : std::vector<double>{0.0}) {
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
        } else if (ns == 2) {
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
    check.reject("missing k-point directions rejected", [&] { ks.band_occupation_response({}); });
    if (ctx->comm().rank() == 0 && !directions.empty()) {
        directions[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    check.reject("nonfinite directions rejected collectively", [&] { ks.band_occupation_response(directions); });
    if (ctx->comm().rank() == 0 && !directions.empty()) {
        directions[0](0, 0) = 0.17;
    }
    if (distributed && ctx->comm().size() > 1) {
        if (ctx->comm().rank() == 0 && !directions.empty()) {
            directions[0](0, 0) += 0.1;
        }
        check.reject("inconsistent k-point replicas rejected", [&] { ks.band_occupation_response(directions); });
        if (ctx->comm().rank() == 0 && !directions.empty()) {
            directions[0](0, 0) = 0.17;
        }
    }
    ctx->cb().band_occ_ = [] {};
    check.reject("custom occupation callback requires its own response",
                 [&] { ks.band_occupation_response(directions); });
    ctx->cb().band_occ_ = nullptr;
    ctx->cfg().unlock();
    ctx->cfg().parameters().extra_charge(ctx->unit_cell().num_valence_electrons() - nb * ctx->max_occupancy());
    if (ns == 1) {
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
test_lapw_density_response(int mag, bool distributed, bool spin_orbit, std::string relativity, std::string device)
{
    bool dirac  = relativity == "dirac";
    bool spinor = mag == 3 && !dirac;
    auto ctx    = context(mag, distributed, spinor, false, false, 0, 5.0, false, 0, "fermi_dirac", false, device,
                          spin_orbit, relativity, true);
    int failures{0};
    Test_check check(failures, ctx->comm(),
                     "constrained LAPW spin mode=" + std::to_string(mag) + " SOC=" + std::to_string(spin_orbit) +
                             " distributed FFT=" + std::to_string(distributed) + " " + relativity + " " + device);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 0.3);
    ks.add_kpoint({-0.06, 0.04, 0.03}, 0.7);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    if (spinor) {
        // Exercise raw magnetic fields as well as the separately staged XC covectors.
        for (int s = 0; s < 3; s++) {
            auto& field  = gs.potential().component(s + 1);
            double value = 0.004 * (s + 1);
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                field.rg().value(ir) = value;
            }
            field.rg().fft_transform(-1);
            for (auto atom : ctx->unit_cell().spl_num_atoms()) {
                field.mt()[atom.i].zero();
                for (int ir = 0; ir < ctx->unit_cell().atom(atom.i).radial_grid().num_points(); ir++) {
                    field.mt()[atom.i](0, ir) = value / y00;
                }
            }
            field.mt().sync(ctx->unit_cell().spl_num_atoms());
        }
        gs.potential().update_atomic_potential();
        ctx->unit_cell().generate_radial_integrals();
    }
    if (spin_orbit) {
        ctx->unit_cell().generate_radial_functions(ctx->out());
        ctx->unit_cell().generate_radial_integrals();
        double peak{0};
        for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
            auto& cls        = ctx->unit_cell().atom_symmetry_class(ic);
            auto const& type = cls.atom_type();
            auto const& grid = type.radial_grid();
            auto base        = cls.spherical_potential();
            std::vector<double> direction(grid.num_points());
            for (int ir = 0; ir < grid.num_points(); ir++) {
                direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
            }
            auto tangent = cls.radial_basis_potential_response(direction, {}, false);
            auto sample  = [&](double step) {
                auto shifted = base;
                for (size_t ir = 0; ir < shifted.size(); ir++) {
                    shifted[ir] += step * direction[ir];
                }
                cls.set_spherical_potential(shifted);
                if (cls.generate_radial_functions(relativity_t::none, false)) {
                    RTE_THROW("SOC radial response fixture failed");
                }
                cls.generate_radial_integrals(relativity_t::none);
                std::vector<double> result;
                for (int l = 1; l <= type.indexr().lmax(); l++) {
                    for (int a = 0; a < type.indexr().max_order(l); a++) {
                        for (int b = 0; b < type.indexr().max_order(l); b++) {
                            result.push_back(cls.so_radial_integral(l, a, b));
                        }
                    }
                }
                return result;
            };
            std::vector<double> analytic;
            for (int l = 1; l <= type.indexr().lmax(); l++) {
                for (int a = 0; a < type.indexr().max_order(l); a++) {
                    for (int b = 0; b < type.indexr().max_order(l); b++) {
                        double value = tangent.spin_orbit_integrals(l, a, b);
                        peak         = std::max(peak, std::abs(value));
                        analytic.push_back(value);
                    }
                }
            }
            // The small SOC scale needs a wider fourth-order stencil above radial-solver roundoff.
            for (double step : {1e-2, 5e-3}) {
                auto plus = sample(step), minus = sample(-step);
                auto half_plus = sample(step / 2), half_minus = sample(-step / 2);
                for (size_t i = 0; i < plus.size(); i++) {
                    plus[i]  = (8 * (half_plus[i] - half_minus[i]) - (plus[i] - minus[i])) / 3;
                    minus[i] = 0;
                }
                check("SOC basis and potential derivative", potential_response_error(analytic, plus, minus, step),
                      1e-8);
            }
            sample(0);
        }
        check("nonzero SOC radial derivative", peak > 1e-10 ? 0 : 1, 0);
        ctx->unit_cell().generate_radial_integrals();
    }
    Hamiltonian0<double> h0(gs.potential(), false);
    if (!dirac) {
        h0.generate_pw_coefs(gs.potential());
    }
    lapw_hamiltonian_response_t magnetic;
    if (spinor) {
        magnetic = h0.lapw_spin_fields();
    }
    int ns = ctx->num_spinors(), nc = spinor ? 4 : ns, nb = ctx->num_bands();
    using matrix_t = la::dmatrix<std::complex<double>>;
    struct matrices_t
    {
        std::array<std::unique_ptr<matrix_t>, 4> h, dh;
        std::array<std::unique_ptr<matrix_t>, 2> s, ds;
    };
    std::vector<matrices_t> matrices;
    std::vector<LAPW_density_response::matrix_direction_t> directions;
    for (auto it : ks.spl_num_kpoints()) {
        auto& kp = *ks.get<double>(it.i);
        Hamiltonian_k<double> hk(h0, kp);
        int n  = dirac ? 2 * kp.num_gkvec() + ctx->unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
        int bs = ctx->cyclic_block_size();
        matrices.emplace_back();
        directions.emplace_back();
        auto& m = matrices.back();
        for (int spin = 0; spin < nc; spin++) {
            for (auto* part : {&m.h, &m.dh}) {
                (*part)[spin] = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
            }
            if (spin < ns) {
                for (auto* part : {&m.s, &m.ds}) {
                    (*part)[spin] = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
                }
                if (dirac) {
                    hk.set_dirac_h_o(*m.h[spin], *m.s[spin]);
                } else {
                    hk.set_fv_h_o(spin, *m.h[spin], *m.s[spin]);
                }
                directions.back().s[spin] = m.ds[spin].get();
            } else {
                hk.set_fv_h_field_response(spin, magnetic, *m.h[spin]);
            }
            for (int j = 0; j < m.h[spin]->num_cols_local(); j++) {
                for (int i = 0; i < m.h[spin]->num_rows_local(); i++) {
                    int row = m.h[spin]->irow(i), col = m.h[spin]->icol(j);
                    auto local          = std::polar(1.0, 0.7 * (row - col)) / ((1.0 + row) * (1.0 + col));
                    (*m.dh[spin])(i, j) = (spin < ns ? (0.06 + 0.03 * it.i + 0.04 * spin) * (*m.s[spin])(i, j) : 0.0) +
                                          0.012 * (1 + 0.2 * spin) * local;
                    if (spin < ns) {
                        (*m.ds[spin])(i, j) = 0.007 * (*m.s[spin])(i, j) + 0.001 * local;
                    }
                }
            }
            directions.back().h[spin] = m.dh[spin].get();
        }
    }
    auto sample = [&](double step) {
        std::vector<std::array<std::unique_ptr<LAPW_eigensystem>, 2>> spectra(matrices.size());
        std::vector<std::unique_ptr<LAPW_spinor_eigensystem>> spinors(spinor ? matrices.size() : 0);
        size_t ik{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            Hamiltonian_k<double> hk(h0, kp);
            auto const& m = matrices[ik];
            int n         = dirac ? 2 * kp.num_gkvec() + ctx->unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
            int bs        = ctx->cyclic_block_size();
            std::array<std::unique_ptr<matrix_t>, 4> h;
            std::array<std::unique_ptr<matrix_t>, 2> s;
            LAPW_spinor_eigensystem::matrix_set_t pauli{};
            for (int spin = 0; spin < nc; spin++) {
                h[spin]     = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
                pauli[spin] = h[spin].get();
                if (spin < ns) {
                    s[spin] = std::make_unique<matrix_t>(n, n, ctx->blacs_grid(), bs, bs);
                }
                for (int j = 0; j < h[spin]->num_cols_local(); j++) {
                    for (int i = 0; i < h[spin]->num_rows_local(); i++) {
                        (*h[spin])(i, j) = (*m.h[spin])(i, j) + step * (*m.dh[spin])(i, j);
                        if (spin < ns) {
                            (*s[spin])(i, j) = (*m.s[spin])(i, j) + step * (*m.ds[spin])(i, j);
                        }
                    }
                }
            }
            if (spinor) {
                spinors[ik] = std::make_unique<LAPW_spinor_eigensystem>(hk, pauli, *s[0]);
            }
            for (int spin = 0; spin < ns; spin++) {
                if (!spinor) {
                    spectra[ik][spin] = std::make_unique<LAPW_eigensystem>(hk, *h[spin], *s[spin]);
                }
                auto const& energies = spinor ? spinors[ik]->energies() : spectra[ik][spin]->energies();
                for (int b = 0; b < nb; b++) {
                    kp.band_energy(b, spin, energies[b]);
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
            if (spinor) {
                std::vector<double> f(nb);
                for (int b = 0; b < nb; b++) {
                    f[b] = kp.band_occupancy(b, 0);
                }
                result.push_back(spinors[ik++]->density_matrix(f));
                continue;
            }
            int n = dirac ? 2 * kp.num_gkvec() + ctx->unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
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
    auto physical_fields = [&](Density const& density) {
        std::vector<double> result;
        for (int spin = 0; spin < (dirac ? mag + 1 : nc); spin++) {
            for (auto const* field : {&density.component(spin), &density.kinetic_density().component(spin)}) {
                auto values = potential_samples(*field);
                result.insert(result.end(), values.begin(), values.end());
            }
        }
        return result;
    };
    auto fields = [&](auto const& p) { return physical_fields(*gs.density().lapw_density_matrix_fields(ks, p)); };
    std::vector<double> production;
    if (spinor || dirac) {
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            Hamiltonian_k<double> hk(h0, kp);
            diagonalize_fp(hk, kp, 1e-11);
        }
        ks.sync_band<double, sync_band_t::energy>();
        ks.find_band_occupancies<double>();
        LAPW_density_response production_reference(h0, ks);
        gs.density().generate<double>(ks, false, false, true);
        production  = physical_fields(gs.density());
        auto mapped = fields(production_reference.density_matrices());
        double error{0}, peak{0};
        for (size_t i = 0; i < mapped.size(); i++) {
            error = std::max(error, std::abs(mapped[i] - production[i]));
            peak  = std::max(peak, std::abs(production[i]));
        }
        check("coefficient matrices reproduce the production spinor rho/tau", error / std::max(1.0, peak), 3e-10);
    }
    sample(0);
    LAPW_density_response reference(h0, ks);
    double entropy_response{0}, eval_sum_response{0};
    auto p  = reference.density_matrices();
    auto dp = reference.density_matrix_response(directions, &entropy_response, &eval_sum_response);
    if (dirac) {
        check("nonzero Dirac occupation response", std::abs(entropy_response) > 1e-10 ? 0 : 1, 0);
    }
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
    if (mag == 1 && !dirac) {
        check("fixed spin including overlap response", std::abs(trace[0] - trace[1]), 1e-10);
    }
    // Resolve the field response above the production chemical-potential search tolerance.
    for (double step : {1e-3, 5e-4}) {
        auto plus             = fields(sample(step));
        double plus_entropy   = ks.entropy_sum();
        double plus_eval_sum  = ks.valence_eval_sum();
        auto minus            = fields(sample(-step));
        double minus_entropy  = ks.entropy_sum();
        double minus_eval_sum = ks.valence_eval_sum();
        check("rho/tau matrix and occupation response finite difference",
              potential_response_error(response, plus, minus, step), 2e-6);
        check("occupation entropy response finite difference",
              std::abs((plus_entropy - minus_entropy) / (2 * step) - entropy_response), 2e-8);
        check("valence eigenvalue-sum response finite difference",
              std::abs((plus_eval_sum - minus_eval_sum) / (2 * step) - eval_sum_response), 2e-8);
    }
    sample(0);
    check.reject("missing matrix directions rejected collectively", [&] { reference.density_matrix_response({}); });
    if (spinor) {
        auto incomplete = directions;
        if (!incomplete.empty()) {
            incomplete[0].h[3] = nullptr;
        }
        check.reject("missing transverse direction rejected collectively",
                     [&] { reference.density_matrix_response(incomplete); });
    }
    if (ctx->comm().size() > 1) {
        check.reject("inconsistent entropy output rejected collectively", [&] {
            reference.density_matrix_response(directions, ctx->comm().rank() == 0 ? &entropy_response : nullptr);
        });
        check.reject("inconsistent eigenvalue-sum output rejected collectively", [&] {
            reference.density_matrix_response(directions, nullptr,
                                              ctx->comm().rank() == 0 ? &eval_sum_response : nullptr);
        });
    }
    std::complex<double> saved;
    if (ctx->comm().rank() == 0 && !matrices.empty()) {
        saved                      = (*matrices[0].dh[0])(0, 0);
        (*matrices[0].dh[0])(0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    check.reject("nonfinite matrix direction rejected collectively",
                 [&] { reference.density_matrix_response(directions); });
    if (ctx->comm().rank() == 0 && !matrices.empty()) {
        (*matrices[0].dh[0])(0, 0) = saved;
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp = *ks.get<double>(it.i);
            kp.band_energy(0, 0, kp.band_energy(0, 0) + 0.1);
            break;
        }
    }
    check.reject("changed reference rejected collectively", [&] { reference.density_matrix_response(directions); });
    sample(0);
    auto offset_band = [&](double offset) {
        for (auto it : ks.spl_num_kpoints()) {
            auto& kp      = *ks.get<double>(it.i);
            double energy = kp.band_energy(0, 0);
            kp.band_energy(0, 0, energy + offset * std::max(1.0, std::abs(energy)));
            break;
        }
        ks.sync_band<double, sync_band_t::energy>();
    };
    double response_tolerance = ctx->cb().external_xc_response_tolerance_;
    offset_band(5e-7);
    check.reject("finite SCF residual exceeds default response accuracy", [&] { LAPW_density_response(h0, ks); });
    ctx->cb().external_xc_response_tolerance_ = 1e-6;
    LAPW_density_response finite_scf_reference(h0, ks);
    sample(0);
    offset_band(1e-3);
    check.reject("inconsistent spectrum still rejected at configured response accuracy",
                 [&] { LAPW_density_response(h0, ks); });
    sample(0);
    ctx->cb().external_xc_response_tolerance_ = response_tolerance;
    for (auto& m : matrices) {
        for (int spin = 0; spin < nc; spin++) {
            if (spin < ns) {
                m.ds[spin]->zero();
            }
            for (int j = 0; j < m.dh[spin]->num_cols_local(); j++) {
                for (int i = 0; i < m.dh[spin]->num_rows_local(); i++) {
                    (*m.dh[spin])(i, j) = spin < ns ? 0.17 * (*m.s[spin])(i, j) : 0.0;
                }
            }
        }
    }
    double invariant{0};
    for (auto const& values : fields(reference.density_matrix_response(directions, nullptr, &eval_sum_response))) {
        invariant =
                std::isfinite(values) ? std::max(invariant, std::abs(values)) : std::numeric_limits<double>::infinity();
    }
    check("uniform potential shift leaves physical fields unchanged", invariant, 1e-11);
    check("uniform potential shift changes the eigenvalue sum by the electron count",
          std::abs(eval_sum_response - 0.17 * ctx->unit_cell().num_valence_electrons()), 1e-10);
    if (dirac) {
        check_coefficient_geometry(check, gs.density(), ks, p);
    }
    return failures;
}

int
test_lapw_coupled_response(int mag, bool distributed, bool spin_orbit, std::string device, std::string relativity,
                           bool core)
{
    bool dirac  = relativity == "dirac";
    bool spinor = mag == 3 && !dirac;
    // Retain the complete native positive-energy subspace on the functional test grid.
    int radial_points = core ? 321 : 0;
    auto ctx = context(mag, distributed, spinor, core, false, 0, 5.0, false, 0, "fermi_dirac", core, device, spin_orbit,
                       relativity, true, core ? "dirac" : "none", radial_points);
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
            auto fractional = dot(ctx->unit_cell().inverse_lattice_vectors(), r);
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
    // Fixed points resolve the one-atom Dirac translation above near-core subtraction noise.
    bool fixed_points       = dirac && !mag && !core;
    auto quadrature_tangent = [fixed_points, core](auto const& c, auto const& q, auto const& d, auto& out) {
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
            for (int x = 0; x < 3; x++) {
                if (!fixed_points) {
                    delta[x] += d.atoms[0][x];
                }
            }
            return std::array<double, 3>{delta[0], delta[1], delta[2]};
        };
        for (size_t p = 0; p < q.coordinates.size(); p++) {
            auto const& r = q.coordinates[p];
            int attached{-1};
            for (int ia = 0; core && ia < c.unit_cell().num_atoms(); ia++) {
                auto const& atom = c.unit_cell().atom(ia);
                auto centre      = c.unit_cell().get_cartesian_coordinates(atom.position());
                if ((r3::vector<double>{r[0], r[1], r[2]} - centre).length() <= atom.mt_radius()) {
                    attached = ia;
                    break;
                }
            }
            out.coordinates[p]        = attached >= 0 ? d.atoms[attached] : move(r);
            out.energy_weights[p]     = attached >= 0 ? 0 : trace * q.energy_weights[p];
            out.descriptor_weights[p] = attached >= 0 ? 0 : trace * q.descriptor_weights[p];
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
    int failures{0}, ns = dirac ? mag + 1 : (spinor ? 4 : ctx->num_spins());
    Test_check check(failures, ctx->comm(),
                     "coupled LAPW " + device + " spins=" + std::to_string(ns) + " SOC=" + std::to_string(spin_orbit) +
                             " distributed FFT=" + std::to_string(distributed));
    if (device == "GPU") {
        check("GPU context, FFT grids and eigensolvers selected",
              ctx->processing_unit() != device_t::GPU || ctx->spfft<double>().processing_unit() != SPFFT_PU_GPU ||
                              ctx->spfft_coarse<double>().processing_unit() != SPFFT_PU_GPU ||
                              !is_device_memory(ctx->gen_evp_solver().data_memory_t()) ||
                              !is_device_memory(ctx->std_evp_solver().data_memory_t())
                      ? 1
                      : 0,
              0);
    }
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
    auto map = [&](Density const& input, K_point_set& kpoints, double* soc_energy = nullptr) {
        auto& c = kpoints.ctx();
        Potential potential(c);
        potential.generate(input, false, true);
        Hamiltonian0<double> h0(potential, true);
        if (!dirac) {
            h0.generate_pw_coefs(potential);
        }
        std::vector<std::array<std::unique_ptr<LAPW_eigensystem>, 2>> spectra;
        std::vector<std::unique_ptr<LAPW_spinor_eigensystem>> spinors;
        lapw_hamiltonian_response_t magnetic;
        lapw_hamiltonian_response_t soc;
        if (spinor) {
            magnetic = h0.lapw_spin_fields();
        }
        if (soc_energy) {
            soc = h0.lapw_spin_fields(true);
        }
        using matrix_t = la::dmatrix<std::complex<double>>;
        for (auto it : kpoints.spl_num_kpoints()) {
            auto& kp = *kpoints.get<double>(it.i);
            auto hk  = h0(kp);
            int n    = dirac ? 2 * kp.num_gkvec() + c.unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
            int bs   = c.cyclic_block_size();
            matrix_t h(n, n, c.blacs_grid(), bs, bs), s(n, n, c.blacs_grid(), bs, bs);
            spectra.emplace_back();
            spinors.emplace_back();
            if (spinor) {
                hk.set_fv_h_o(0, h, s);
                std::array<std::unique_ptr<matrix_t>, 3> fields;
                LAPW_spinor_eigensystem::matrix_set_t pauli{&h};
                for (int j = 0; j < 3; j++) {
                    fields[j] = std::make_unique<matrix_t>(n, n, c.blacs_grid(), bs, bs);
                    hk.set_fv_h_field_response(j + 1, magnetic, *fields[j]);
                    pauli[j + 1] = fields[j].get();
                }
                spinors.back() = std::make_unique<LAPW_spinor_eigensystem>(hk, pauli, s);
            }
            for (int spin = 0; spin < c.num_spinors(); spin++) {
                if (!spinor) {
                    if (dirac) {
                        hk.set_dirac_h_o(h, s);
                    } else {
                        hk.set_fv_h_o(spin, h, s);
                    }
                    spectra.back()[spin] = std::make_unique<LAPW_eigensystem>(hk, h, s);
                }
                auto const& energies = spinor ? spinors.back()->energies() : spectra.back()[spin]->energies();
                for (int b = 0; b < c.num_bands(); b++) {
                    kp.band_energy(b, spin, energies[b]);
                }
            }
        }
        kpoints.sync_band<double, sync_band_t::energy>();
        kpoints.find_band_occupancies<double>();
        std::vector<mdarray<std::complex<double>, 3>> matrices;
        size_t ik{0};
        for (auto it : kpoints.spl_num_kpoints()) {
            auto const& kp = *kpoints.get<double>(it.i);
            if (spinor) {
                std::vector<double> f(c.num_bands());
                for (int b = 0; b < c.num_bands(); b++) {
                    f[b] = kp.band_occupancy(b, 0);
                }
                matrices.push_back(spinors[ik++]->density_matrix(f));
                continue;
            }
            int n = dirac ? 2 * kp.num_gkvec() + c.unit_cell().mt_lo_basis_size() : kp.gklo_basis_size();
            matrices.emplace_back(std::array<index_range, 3>{n, n, c.num_spinors()});
            for (int spin = 0; spin < c.num_spinors(); spin++) {
                std::vector<double> f(n, 0);
                for (int b = 0; b < c.num_bands(); b++) {
                    f[b] = kp.band_occupancy(b, spin);
                }
                auto p = spectra[ik][spin]->density_matrix(f);
                std::copy_n(p.at(memory_t::host), p.size(), matrices.back().at(memory_t::host) + spin * p.size());
            }
            ik++;
        }
        if (soc_energy) {
            std::complex<double> energy{0};
            ik = 0;
            for (auto it : kpoints.spl_num_kpoints()) {
                auto& kp = *kpoints.get<double>(it.i);
                auto hk  = h0(kp);
                int n = kp.gklo_basis_size(), bs = c.cyclic_block_size();
                matrix_t h(n, n, c.blacs_grid(), bs, bs);
                for (int s = 1; s < 4; s++) {
                    hk.set_fv_h_field_response(s, soc, h);
                    auto full = h.get_full_matrix();
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            energy += kp.weight() / kp.comm().size() * matrices[ik](j, i, s) * full(i, j);
                        }
                    }
                }
                ik++;
            }
            c.comm().allreduce(&energy, 1);
            if (!std::isfinite(std::abs(energy)) || std::abs(energy.imag()) > 1e-12) {
                RTE_THROW("nonfinite or complex scratch SOC energy");
            }
            *soc_energy = energy.real();
        }
        auto output = input.lapw_density_matrix_fields(kpoints, matrices);
        if (core) {
            output->generate_core_charge_density(potential.get_spherical_potential(),
                                                 potential.get_core_xc_derivatives());
            for (auto atom : c.unit_cell().spl_num_atoms()) {
                for (int ir = 0; ir < c.unit_cell().atom(atom.i).type().num_mt_points(); ir++) {
                    output->rho().mt()[atom.i](0, ir) += output->core_charge_density(atom.i)[ir] / y00;
                    output->kinetic_density().scalar().mt()[atom.i](0, ir) +=
                            output->core_kinetic_density(atom.i)[ir] / y00;
                }
            }
        }
        return output;
    };
    auto sample = [&](double step, double* soc_energy = nullptr) {
        Density input(*ctx);
        copy(gs.density(), input);
        for (int s = 0; s < ns; s++) {
            axpy(step, direction.component(s), input.component(s));
            axpy(step, direction.kinetic_density().component(s), input.kinetic_density().component(s));
        }
        prepare(input);
        auto result = map(input, ks, soc_energy);
        gs.potential().update_atomic_potential();
        ctx->unit_cell().generate_radial_functions(ctx->out());
        if (!dirac) {
            ctx->unit_cell().generate_radial_integrals();
        }
        return result;
    };
    sample(0);
    auto before = values(gs.density());
    LAPW_response response(ks, gs.density(), gs.potential());
    double soc_response{0};
    auto jx       = response.apply(direction, nullptr, &soc_response);
    auto analytic = values(*jx);
    double peak{0};
    for (auto v : analytic) {
        peak = std::max(peak, std::abs(v));
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
    check("nonzero coupled response", peak > 1e-8 ? 0 : 1, 0);
    if (spin_orbit) {
        check("nonzero SOC energy response", std::abs(soc_response) > 1e-14 ? 0 : 1, 0);
    } else {
        check("SOC energy response vanishes without SOC", std::abs(soc_response), 0);
    }
    for (double step : {1e-3, 5e-4}) {
        double ep{0}, em{0};
        auto plus  = values(*sample(step, spin_orbit ? &ep : nullptr));
        auto minus = values(*sample(-step, spin_orbit ? &em : nullptr));
        if (core) {
            auto half_plus  = values(*sample(step / 2));
            auto half_minus = values(*sample(-step / 2));
            for (size_t i = 0; i < analytic.size(); i++) {
                plus[i]  = (8 * (half_plus[i] - half_minus[i]) - (plus[i] - minus[i])) / 3;
                minus[i] = 0;
            }
        }
        check("complete SCF map rho/tau finite difference", potential_response_error(analytic, plus, minus, step),
              3e-5);
        if (spin_orbit) {
            double fd = (ep - em) / (2 * step);
            if (ctx->comm().rank() == 0) {
                std::cout << "SOC energy potential response analytic=" << soc_response << " FD=" << fd << '\n';
            }
            check("SOC expectation potential finite difference",
                  std::abs(fd - soc_response) / std::max(1e-12, std::abs(soc_response)), 3e-5);
        }
    }
    sample(0);
    double zero_error{0};
    for (auto v : values(*response.apply(zero))) {
        zero_error = std::max(zero_error, std::abs(v));
    }
    check("zero coupled response", zero_error, 1e-13);
    xc_geometry_direction_t geometry(ctx->unit_cell().num_atoms());
    {
        zero_error = 0;
        for (auto v : values(*response.geometry_rhs(geometry))) {
            zero_error = std::max(zero_error, std::abs(v));
        }
        check("zero geometry right-hand side", zero_error, 1e-13);
        auto probe = context(mag, distributed, spinor, core, false, 0, 5.0, false, 0, "fermi_dirac", core, device,
                             spin_orbit, relativity, true, core ? "dirac" : "none", radial_points);
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
            double soc_geometry{0};
            auto analytic_geometry =
                    values(*response.geometry_rhs(geometry, nullptr, spin_orbit ? &soc_geometry : nullptr));
            auto quadrature_direction = gs.potential().external_xc_quadrature_direction(geometry);
            auto geometry_sample      = [&](double step, double* soc_energy = nullptr) {
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
                    for (int ia = 0; core && ia < probe->unit_cell().num_atoms(); ia++) {
                        auto const& atom = probe->unit_cell().atom(ia);
                        auto offset = r3::vector<double>{quadrature->coordinates[p][0], quadrature->coordinates[p][1],
                                                         quadrature->coordinates[p][2]} -
                                      centres[ia];
                        if (offset.length() <= atom.mt_radius()) {
                            // Keep attached points at the same offset after the fractional-coordinate round trip.
                            auto centre = probe->unit_cell().get_cartesian_coordinates(atom.position());
                            for (int x = 0; x < 3; x++) {
                                next->coordinates[p][x] = centre[x] + offset[x];
                            }
                            break;
                        }
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
                return values(*map(input, probe_ks, soc_energy));
            };
            // The scalar tau field near the origin needs a wider atom-translation stencil with SOC.
            // Fourth-order differences retain accuracy within the fixed point-domain margin.
            double step_scale = spin_orbit && part == 0 ? 4 : 1;
            for (double step : {step_scale * 1e-2, step_scale * 5e-3}) {
                double ep{0}, em{0}, hp{0}, hm{0};
                auto plus       = geometry_sample(step, spin_orbit ? &ep : nullptr);
                auto minus      = geometry_sample(-step, spin_orbit ? &em : nullptr);
                auto half_plus  = geometry_sample(step / 2, spin_orbit ? &hp : nullptr);
                auto half_minus = geometry_sample(-step / 2, spin_orbit ? &hm : nullptr);
                for (size_t i = 0; i < analytic_geometry.size(); i++) {
                    plus[i]  = (8 * (half_plus[i] - half_minus[i]) - (plus[i] - minus[i])) / 3;
                    minus[i] = 0;
                }
                check("explicit geometry SCF map rho/tau finite difference",
                      potential_response_error(analytic_geometry, plus, minus, step), 3e-5);
                if (spin_orbit) {
                    double fd = (8 * (hp - hm) - (ep - em)) / (6 * step);
                    if (ctx->comm().rank() == 0) {
                        std::cout << "SOC energy geometry part=" << part << " analytic=" << soc_geometry << " FD=" << fd
                                  << '\n';
                    }
                    check("SOC expectation geometry finite difference",
                          std::abs(fd - soc_geometry) / std::max(1e-12, std::abs(soc_geometry)), 3e-5);
                }
            }
        }
        ctx->cb().external_xc_quadrature_direction_ = nullptr;
        check.reject("geometry right-hand side requires a host quadrature tangent",
                     [&] { response.geometry_rhs(geometry); });
        ctx->cb().external_xc_quadrature_direction_ = quadrature_tangent;
    }
    auto zero_solution = response.solve(zero, 1e-8, 1);
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
                    for (int component = 0; component < (dirac ? 2 : 1); component++) {
                        result.push_back(cls.radial_function(ir, i, component));
                        result.push_back(cls.radial_function_derivative(ir, i, component));
                    }
                }
            }
        }
        return result;
    };
    auto frozen_before = frozen();
    auto solution      = response.solve(rhs, core ? 1e-11 : 1e-8, 100, 10, 0.4);
    check("coupled solve residual", solution.residual, 1e-8);
    auto obtained = values(*solution.fields), expected = values(direction);
    double error{0};
    for (size_t i = 0; i < obtained.size(); i++) {
        error = std::max(error, std::abs(obtained[i] - expected[i]));
    }
    if (dirac || core) {
        error = 0;
        for (int s = 0; s < ns; s++) {
            for (int part : {0, 1}) {
                auto const& a =
                        part == 0 ? solution.fields->component(s) : solution.fields->kinetic_density().component(s);
                auto const& b = part == 0 ? direction.component(s) : direction.kinetic_density().component(s);
                auto va = potential_samples(a), vb = potential_samples(b);
                double residual{0}, scale{1e-12};
                for (size_t i = 0; i < va.size(); i++) {
                    residual = std::max(residual, std::abs(va[i] - vb[i]));
                    scale    = std::max(scale, std::abs(vb[i]));
                }
                error = std::max(error, residual / scale);
            }
        }
    }
    check("coupled solve recovers prescribed fields", error, 2e-8);
    double regular_error{0};
    for (int part : {0, 1}) {
        double residual{0}, scale{1e-12};
        auto sample = [&](double a, double b, double factor) {
            residual = std::max(residual, factor * std::abs(a - b));
            scale    = std::max(scale, factor * std::abs(b));
        };
        for (int s = 0; s < ns; s++) {
            auto const& a = part == 0 ? solution.fields->component(s) : solution.fields->kinetic_density().component(s);
            auto const& b = part == 0 ? direction.component(s) : direction.kinetic_density().component(s);
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                sample(a.rg().value(ir), b.rg().value(ir), 1);
            }
            for (auto atom : ctx->unit_cell().spl_num_atoms()) {
                auto const& type = ctx->unit_cell().atom(atom.i).type();
                auto const& grid = type.radial_grid();
                bool rel         = ctx->valence_relativity() != relativity_t::none ||
                           (type.num_core_electrons() != 0 && ctx->core_relativity() != relativity_t::none);
                double gamma = std::sqrt(1 - std::pow(type.zn() / speed_of_light, 2));
                double power = rel ? (part == 0 ? 2 : 4) - 2 * gamma : 0;
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double factor = std::pow(grid[ir] / grid.last(), power);
                    for (int lm = 0; lm < a.mt()[atom.i].angular_domain_size(); lm++) {
                        sample(a.mt()[atom.i](lm, ir), b.mt()[atom.i](lm, ir), factor);
                    }
                }
            }
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&residual, 1);
        ctx->comm().allreduce<double, mpi::op_t::max>(&scale, 1);
        regular_error = std::max(regular_error, residual / scale);
    }
    check("coupled solve resolves Coulomb-regularized fields", regular_error, 2e-8);
    if (ctx->comm().rank() == 0) {
        std::cout << "coupled LAPW solve iterations: " << solution.iterations << '\n';
    }
    check.reject("one iteration does not claim convergence", [&] { response.solve(rhs, 1e-8, 1); });
    check.reject("invalid response controls rejected collectively",
                 [&] { response.solve(rhs, ctx->comm().rank() == 0 ? -1 : 1e-8, 10); });
    double saved_tau = rhs.kinetic_density().scalar().rg().value(0);
    if (ctx->comm().rank() == 0) {
        rhs.kinetic_density().scalar().rg().value(0) = std::numeric_limits<double>::quiet_NaN();
    }
    check.reject("nonfinite kinetic-density direction rejected collectively", [&] { response.apply(rhs); });
    rhs.kinetic_density().scalar().rg().value(0) = saved_tau;
    if (mag == 3) {
        double saved = rhs.kinetic_density().component(3).rg().value(0);
        if (ctx->comm().rank() == 0) {
            rhs.kinetic_density().component(3).rg().value(0) = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("nonfinite transverse direction rejected collectively", [&] { response.apply(rhs); });
        rhs.kinetic_density().component(3).rg().value(0) = saved;
    }
    auto after = values(gs.density());
    error      = 0;
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
    if (ctx->comm().size() > 1) {
        check.reject("SOC energy output is context-collective",
                     [&] { response.apply(zero, nullptr, ctx->comm().rank() == 0 ? &soc_response : nullptr); });
    }
    return failures;
}

int
test_lapw_external(int mag, bool distributed_fft, bool second_variation, bool core = false)
{
    bool simple = !mag && second_variation && !core;
    auto ctx    = context(mag, distributed_fft, second_variation, core, false, 0, 5.0, simple);
    int failures{0};
    Test_check check(failures, ctx->comm(),
                     "external LAPW core=" + std::to_string(core) + " spins=" + std::to_string(ctx->num_spins()) +
                             " atoms=" + std::to_string(ctx->unit_cell().num_atoms()) + " distributed FFT=" +
                             std::to_string(distributed_fft) + " second variation=" + std::to_string(second_variation) +
                             " simple radial integrals=" + std::to_string(simple));
    if (!mag && !second_variation && !core) {
        ctx->cfg().unlock();
        ctx->cfg().parameters().extra_charge(0.25);
        ctx->cfg().lock();
        Density charged(*ctx);
        charged.initial_density();
        check("charged LAPW initial density has the physical electron count",
              std::abs(charged.rho().integrate().total - (ctx->unit_cell().num_electrons() - 0.25)), 1e-12);
        check("charged LAPW density passes the electron-count check", charged.check_num_electrons() ? 0 : 1, 0);
        ctx->cfg().unlock();
        ctx->cfg().parameters().extra_charge(0);
        ctx->cfg().lock();
    }
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    check("joint point XC field derivative", check_external_xc_point_derivative(density), 2e-10);
    if (mag == 3) {
        auto energy_check = [&] {
            auto c = potential.lapw_xc_contractions(density);
            check("spinor callback Euler identity",
                  std::abs(c.rho + c.magnetic + c.tau - 2 * potential.energy_exc(density)), 2e-11);
        };
        energy_check();
        for (auto const& values : gs.check_external_xc_derivative(1e-4)) {
            check("spinor PW/AW/LO orbital derivative", std::abs(values[0] - values[1]), 2e-8);
            check("spinor orbital derivative at halved step", std::abs(values[0] - values[2]), 6e-9);
        }
        // This quadratic fixture has a linear response; a larger core stencil avoids cancellation.
        for (double error : gs.check_lapw_xc_response(core ? 1e-2 : 1e-3)) {
            check("four-component LAPW callback response", error, 1e-8);
        }
        ctx->unit_cell().generate_radial_functions(ctx->out());
        ctx->unit_cell().generate_radial_integrals();
        std::vector<std::vector<double>> base, direction;
        std::vector<radial_basis_response_t> radial;
        for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
            auto& cls        = ctx->unit_cell().atom_symmetry_class(ic);
            auto const& grid = cls.atom_type().radial_grid();
            base.push_back(cls.spherical_potential());
            direction.emplace_back(grid.num_points());
            for (int ir = 0; ir < grid.num_points(); ir++) {
                direction.back()[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
            }
            radial.push_back(cls.radial_basis_potential_response(direction.back(), {}, false));
        }
        auto analytic = density.lapw_radial_fields_response(ks, radial);
        std::vector<mdarray<std::complex<double>, 3>> matrices;
        for (auto it : ks.spl_num_kpoints()) {
            auto const& kp = *ks.get<double>(it.i);
            int n          = kp.gklo_basis_size();
            matrices.emplace_back(std::array<index_range, 3>{n, n, 4});
            auto& p = matrices.back();
            p.zero();
            for (int b = 0; b < ctx->num_bands(); b++) {
                auto up = lapw_coefficients(kp, 0, b), down = lapw_coefficients(kp, 1, b);
                double f = kp.band_occupancy(b, 0);
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        auto uu = f * up[i] * std::conj(up[j]), dd = f * down[i] * std::conj(down[j]);
                        auto ud = f * up[i] * std::conj(down[j]), du = f * down[i] * std::conj(up[j]);
                        p(i, j, 0) += uu + dd;
                        p(i, j, 1) += uu - dd;
                        p(i, j, 2) += ud + du;
                        p(i, j, 3) += std::complex<double>(0, 1) * (ud - du);
                    }
                }
            }
        }
        auto mapped = density.lapw_density_matrix_fields(ks, matrices);
        Density valence(*ctx);
        sample_radial_fields(*ctx, ks, base, potential.get_core_xc_derivatives(), nullptr, nullptr, &valence);
        for (int part : {0, 1}) {
            for (int s = 0; s < 4; s++) {
                auto a = potential_samples(part == 0 ? mapped->component(s) : mapped->kinetic_density().component(s));
                auto b = potential_samples(part == 0 ? valence.component(s) : valence.kinetic_density().component(s));
                double error{0}, peak{0};
                for (size_t i = 0; i < a.size(); i++) {
                    double residual = std::abs(a[i] - b[i]);
                    error           = std::isfinite(residual) ? std::max(error, residual)
                                                              : std::numeric_limits<double>::infinity();
                    peak            = std::max(peak, std::abs(b[i]));
                }
                ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                check("Cartesian PW/LO density matrix reproduces spinor fields", error / std::max(1e-12, peak), 1e-10);
                check("nonzero Cartesian spinor field", peak > 1e-8 ? 0 : 1, 0);
            }
        }
        {
            Potential covectors(*ctx);
            covectors.set_lapw_xc_derivatives(potential.lapw_external_xc_response(density, *mapped), 0);
            auto contractions = covectors.lapw_xc_contractions(*mapped);
            Periodic_function<double> hartree(*ctx, [&](int) { return lmax_t(ctx->lmax_pot()); });
            hartree.zero();
            Hamiltonian0<double> h0(potential, false);
            auto prepared = h0.lapw_field_response(hartree, *covectors.lapw_xc_derivatives());
            double value{0};
            size_t ik{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto& kp = *ks.get<double>(it.i);
                Hamiltonian_k<double> hk(h0, kp);
                int n = kp.gklo_basis_size(), bs = ctx->cyclic_block_size();
                la::dmatrix<std::complex<double>> matrix(n, n, ctx->blacs_grid(), bs, bs);
                for (int s = 0; s < 4; s++) {
                    hk.set_fv_h_field_response(s, prepared, matrix);
                    auto full = matrix.get_full_matrix();
                    double error{0}, peak{0};
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < n; i++) {
                            double residual = std::abs(full(i, j) - std::conj(full(j, i)));
                            error           = std::isfinite(residual) ? std::max(error, residual)
                                                                      : std::numeric_limits<double>::infinity();
                            peak            = std::max(peak, std::abs(full(i, j)));
                            if (kp.comm().rank() == 0) {
                                value += kp.weight() * std::real(matrices[ik](j, i, s) * full(i, j));
                            }
                        }
                    }
                    check("Hermitian Pauli field matrix", error / std::max(1e-12, peak), 1e-12);
                    check("nonzero Pauli field matrix", peak > 1e-12 ? 0 : 1, 0);
                }
                ik++;
                check.reject("Pauli field matrix rejects an invalid component",
                             [&] { hk.set_fv_h_field_response(4, prepared, matrix); });
            }
            ctx->comm().allreduce(&value, 1);
            double fields = contractions.rho + contractions.magnetic + contractions.tau;
            check("PW/LO Pauli matrices are the transpose of physical field reconstruction",
                  std::abs(value - fields) / std::max(1.0, std::abs(fields)), 1e-10);
            Density zero(*ctx);
            zero.zero();
            zero.kinetic_density().zero();
            auto scalar = h0.lapw_field_response(*potential.hartree_response(mapped->rho()),
                                                 potential.lapw_external_xc_response(density, zero));
            double transverse{0};
            for (int s = 1; s < 4; s++) {
                for (int ig = 0; ig < ctx->gvec().num_gvec(); ig++) {
                    transverse = std::max(transverse, std::abs(scalar.scalar_pw(ig, s)));
                }
                for (auto const& atom : scalar.local) {
                    int nb = atom.size(0);
                    for (int j = 0; j < nb; j++) {
                        for (int i = 0; i < nb; i++) {
                            transverse = std::max(transverse, std::abs(atom(i, j, s)));
                        }
                    }
                }
            }
            check("Hartree field matrices have no spin-vector contribution", transverse, 0);
        }
        auto mapped_radial = density.lapw_density_matrix_radial_response(ks, matrices, radial);
        std::vector<int> nmt;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            nmt.push_back(ctx->unit_cell().atom(ia).type().mt_basis_size());
        }
        std::vector<std::unique_ptr<wf::Wave_functions<double>>> zero;
        std::vector<wf::Wave_functions<double>*> orbitals;
        for (auto it : ks.spl_num_kpoints()) {
            zero.push_back(std::make_unique<wf::Wave_functions<double>>(
                    ks.get<double>(it.i)->gkvec_sptr(), nmt, wf::num_mag_dims(3), wf::num_bands(ctx->num_bands()),
                    memory_t::host));
            zero.back()->zero(memory_t::host);
            orbitals.push_back(zero.back().get());
        }
        auto combined = density.lapw_orbital_fields_response(ks, orbitals, radial);
        double combined_error{0};
        for (int part : {0, 1}) {
            auto const& fields = part == 0 ? static_cast<Field4D const&>(*combined) : combined->kinetic_density();
            for (int s = 0; s < 4; s++) {
                for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                    combined_error = std::max(combined_error, std::abs(fields.component(s).rg().f_pw_local(ig)));
                }
                for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                    auto const& field = fields.component(s).mt()[ia];
                    for (size_t i = 0; i < field.size(); i++) {
                        combined_error     = std::max(combined_error, std::abs(field[i] - analytic[ia][part][s][i]));
                        auto const& mapped = part == 0 ? mapped_radial->component(s).mt()[ia]
                                                       : mapped_radial->kinetic_density().component(s).mt()[ia];
                        combined_error     = std::max(combined_error, std::abs(mapped[i] - analytic[ia][part][s][i]));
                    }
                }
            }
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&combined_error, 1);
        check("combined spinor orbital/radial response", combined_error, 1e-13);
        auto sample = [&](double step) {
            auto v = base;
            for (int ic = 0; ic < ctx->unit_cell().num_atom_symmetry_classes(); ic++) {
                auto& cls = ctx->unit_cell().atom_symmetry_class(ic);
                for (size_t ir = 0; ir < v[ic].size(); ir++) {
                    v[ic][ir] += step * direction[ic][ir];
                }
                cls.set_spherical_potential(v[ic]);
                if (cls.generate_radial_functions(relativity_t::none, false)) {
                    RTE_THROW("spinor radial response fixture failed");
                }
                cls.set_spherical_potential(base[ic]);
            }
            ctx->unit_cell().generate_radial_integrals();
            return sample_radial_fields(*ctx, ks, v, potential.get_core_xc_derivatives());
        };
        double radial_step = core ? 1e-2 : 5e-3;
        for (double step : {radial_step, radial_step / 2}) {
            auto plus = sample(step), minus = sample(-step);
            check("four-component radial density/tau response", radial_fields_error(analytic, plus, minus, step), 3e-5);
        }
        sample(0);
        check_coefficient_geometry(check, density, ks, matrices);
        if (ctx->comm().rank() == 0 && !matrices.empty()) {
            matrices[0](0, 1, 3) += 0.1;
        }
        check.reject("Cartesian coefficient fields reject a non-Hermitian transverse block collectively",
                     [&] { density.lapw_density_matrix_fields(ks, matrices); });
        // Exercise the SCF lifecycle; physical convergence is not an operator test.
        gs.find(1e-9, 1e-10, 1e-9, 20, false);
        check("finite spinor SCF energy", std::isfinite(gs.total_energy()) ? 0 : 1, 0);
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        energy_check();
        return failures;
    }
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
        check.reject("Hartree geometry rejects malformed directions collectively",
                     [&] { potential.hartree_geometry_response(direction, bad_geometry); });
        bad_geometry = geometry;
        if (ctx->comm().rank() == 0) {
            bad_geometry.lattice[0] = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("Hartree geometry rejects nonfinite directions collectively",
                     [&] { potential.hartree_geometry_response(direction, bad_geometry); });
        if (ctx->comm().size() > 1) {
            bad_geometry = geometry;
            if (ctx->comm().rank() == 0) {
                bad_geometry.atoms[0][0] += 0.01;
            }
            check.reject("Hartree geometry rejects inconsistent replicas",
                         [&] { potential.hartree_geometry_response(direction, bad_geometry); });
            check.reject("Hartree geometry rejects inconsistent nuclear flags",
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
        check.reject("nonfinite Hartree direction rejected collectively",
                     [&] { potential.hartree_response(direction); });
        check.reject("nonfinite Hartree geometry density rejected collectively",
                     [&] { potential.hartree_geometry_response(direction, geometry); });
        direction.zero();
        for (auto it : ctx->unit_cell().spl_num_atoms()) {
            if (it.i == 0) {
                direction.mt()[it.i](0, 0) = std::numeric_limits<double>::quiet_NaN();
            }
        }
        check.reject("nonfinite MT Hartree direction rejected collectively",
                     [&] { potential.hartree_response(direction); });
        Periodic_function<double> no_spheres(*ctx);
        no_spheres.zero();
        check.reject("missing MT Hartree direction rejected collectively",
                     [&] { potential.hartree_response(no_spheres); });
        check.reject("missing MT Hartree geometry density rejected collectively",
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
        check.reject("missing XC response callback rejected collectively",
                     [&] { potential.lapw_external_xc_response(density, direction); });
        ctx->cb().external_xc_response_ = [](auto const& c, auto const& in, auto const& d, auto& out) {
            functional_response(c, in, d, out);
            if (c.comm().rank() == 0) {
                out.vrho[0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        check.reject("nonfinite XC response rejected collectively",
                     [&] { potential.lapw_external_xc_response(density, direction); });
        ctx->cb().external_xc_response_ = [](auto const& c, auto const&, auto const&, auto&) {
            if (c.comm().rank() == 0) {
                RTE_THROW("XC response fixture failure");
            }
        };
        check.reject("XC response callback failure propagated collectively",
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
        check.reject("wrong host geometry shape rejected collectively",
                     [&] { potential.external_xc_quadrature_direction(bad_host); });
        bad_host = host_direction;
        if (ctx->comm().rank() == 0) {
            bad_host.lattice[0] = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("nonfinite host geometry direction rejected collectively",
                     [&] { potential.external_xc_quadrature_direction(bad_host); });
        if (ctx->comm().size() > 1) {
            bad_host = host_direction;
            if (ctx->comm().rank() == 0) {
                bad_host.atoms[0][0] += 1;
            }
            check.reject("inconsistent host geometry replicas rejected collectively",
                         [&] { potential.external_xc_quadrature_direction(bad_host); });
        }
        ctx->cb().external_xc_quadrature_direction_ = [](auto const& c, auto const&, auto const&, auto& out) {
            if (c.comm().rank() == 0) {
                out.energy_weights[0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        check.reject("nonfinite host quadrature tangent rejected collectively",
                     [&] { potential.external_xc_quadrature_direction(host_direction); });
        ctx->cb().external_xc_quadrature_direction_ = nullptr;
        check.reject("missing host tangent is not silently treated as fixed quadrature",
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
        check.reject("wrong quadrature direction shape rejected collectively", apply_point);
        geometry = xc_point_derivatives_t(quadrature);
        if (ctx->comm().rank() == 0) {
            geometry.energy_weights[0] = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("nonfinite quadrature direction rejected collectively", apply_point);
        geometry                        = xc_point_derivatives_t(quadrature);
        ctx->cb().external_xc_response_ = [](auto const& c, auto const& in, auto const& d, auto& out) {
            functional_response(c, in, d, out);
            if (c.comm().rank() == 0) {
                out.geometry->coordinates[0][0] = std::numeric_limits<double>::quiet_NaN();
            }
        };
        check.reject("nonfinite quadrature response rejected collectively", apply_point);
        ctx->cb().external_xc_response_          = functional_response;
        ctx->cb().external_xc_geometry_response_ = false;
        check.reject("field-only callback rejects quadrature differentiation", apply_point);
        if (ctx->comm().rank() == 0) {
            direction.rho().mt()[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("nonfinite XC MT direction rejected collectively",
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
    check.reject("point potential cannot be symmetrized independently",
                 [&] { potential.generate(density, true, true); });

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
    check.reject("LAPW forces require complete host geometry callbacks", [&] { gs.forces().calc_forces_total(); });
    check.reject("LAPW stress requires complete host geometry callbacks", [&] { gs.stress().calc_stress_total(); });
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
    // The quadratic fixture permits a wider core stencil without truncation error.
    for (double error : gs.check_lapw_xc_response(core ? 1e-2 : 1e-3)) {
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
    check.reject("orbital diagnostic propagates callback failure", [&] { gs.check_external_xc_derivative(1e-4); });
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
                check.reject("invalid XC field matrix input rejected collectively",
                             [&] { h0.lapw_field_response(*hartree, xc); });
                xc.local[0][1][0][0] = saved;
                if (ctx->comm().rank() == 0) {
                    saved               = hartree->mt()[0][0];
                    hartree->mt()[0][0] = std::numeric_limits<double>::quiet_NaN();
                }
                check.reject("invalid Hartree field matrix input rejected collectively",
                             [&] { h0.lapw_field_response(*hartree, xc); });
                if (ctx->comm().rank() == 0) {
                    hartree->mt()[0][0] = saved;
                }
                auto invalid_energy_fields = potential.lapw_external_xc_response(density, direction);
                check.reject("nonfinite staged LAPW energy rejected collectively", [&] {
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
        auto radial_response = cls.radial_basis_potential_response(std::vector<double>(tau.size(), 0), d[cls.id()]);
        check("radial basis core eigenvalue-sum response", std::abs(radial_response.core_eval_sum - derivative), 2e-7);
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
                    auto c = lapw_coefficients(kp, s, b);
                    std::vector<std::complex<double>> dc(n);
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
        check.reject("coefficient fields reject missing k-point matrices",
                     [&] { density.lapw_density_matrix_fields(ks, {}); });
        std::complex<double> saved_coefficient{0};
        if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
            saved_coefficient                = coefficient_matrices[0](0, 0, 0);
            coefficient_matrices[0](0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        check.reject("coefficient fields reject a nonfinite matrix collectively",
                     [&] { density.lapw_density_matrix_fields(ks, coefficient_matrices); });
        if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
            coefficient_matrices[0](0, 0, 0) = saved_coefficient;
        }
        if (distributed_fft && ctx->comm().size() > 1) {
            if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
                coefficient_matrices[0](0, 0, 0) += 0.1;
            }
            check.reject("coefficient fields reject inconsistent k-point replicas",
                         [&] { density.lapw_density_matrix_fields(ks, coefficient_matrices); });
            if (ctx->comm().rank() == 0 && !coefficient_matrices.empty()) {
                coefficient_matrices[0](0, 0, 0) = saved_coefficient;
            }
        }
        verify_restored();
        check_coefficient_geometry(check, density, ks, frozen_matrices);
        verify_restored();
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
            double basis_derivative = radial_basis_derivative(gs.potential(), ks, radial_direction, xc_direction);
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
                check.reject("invalid orbital direction rejected collectively",
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
                check.reject("nonfinite field response rejected collectively",
                             [&] { density.lapw_radial_fields_response(ks, zero); });
                check.reject("nonfinite atom matrix response rejected collectively",
                             [&] { basis_h0.lapw_radial_basis_response(zero); });
                zero[0].surface_derivatives[0] = 0;
                if (ctx->comm().rank() == 0) {
                    zero[0].core_fields[0].pop_back();
                }
                check.reject("invalid core field response rejected collectively",
                             [&] { density.lapw_radial_fields_response(ks, zero); });
                auto direction = radial_direction;
                for (auto& atom : direction) {
                    std::fill(atom.begin(), atom.end(), 0);
                }
                check("zero radial potential direction",
                      std::abs(radial_basis_derivative(gs.potential(), ks, direction)), 0);
                if (ctx->comm().rank() == 0) {
                    direction[0].pop_back();
                }
                check.reject("invalid radial direction rejected collectively",
                             [&] { radial_basis_derivative(gs.potential(), ks, direction); });
                for (int j : {0, 1}) {
                    auto invalid_xc = xc_direction;
                    if (ctx->comm().rank() == 0) {
                        invalid_xc[0][j] = {std::numeric_limits<double>::quiet_NaN()};
                    }
                    check.reject("invalid radial XC direction rejected collectively",
                                 [&] { radial_basis_derivative(gs.potential(), ks, radial_direction, invalid_xc); });
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
    struct state_t
    {
        std::unique_ptr<Simulation_context> ctx;
        std::unique_ptr<K_point_set> ks;
        std::unique_ptr<DFT_ground_state> gs;
    };
    std::unique_ptr<state_t> reference;
    auto calculate = [&](double displacement, bool forces) {
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
        auto found = gs.find(core ? 1e-10 : 1e-11, core ? 1e-11 : 1e-12, 1e-11, 200, false);
        if (!found.at("converged").get<bool>()) {
            RTE_THROW("LAPW force SCF did not converge");
        }
        double direct = gs.density().kinetic_density().scalar().integrate().total +
                        gs.potential().energy_exc(gs.density()) + 0.5 * gs.potential().energy_vha() +
                        energy_enuc(*ctx, gs.potential()) + ks.entropy_sum();
        std::array<double, 3> result{free_energy(*ctx, ks, gs.density(), gs.potential()), direct, ks.entropy_sum()};
        if (forces) {
            reference = std::move(state);
        }
        return result;
    };
    auto centre     = calculate(0, true);
    auto& gs        = *reference->gs;
    auto& potential = gs.potential();
    auto& ctx       = *reference->ctx;
    int failures{0};
    Test_check check(failures, ctx.comm());
    auto snapshot = [&] {
        std::vector<double> values{gs.total_energy(), potential.energy_exc(gs.density())};
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (auto const* f : {&gs.density().component(s), &gs.density().kinetic_density().component(s)}) {
                auto field = potential_samples(*f);
                values.insert(values.end(), field.begin(), field.end());
            }
        }
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& cls = ctx.unit_cell().atom_symmetry_class(ic);
            for (auto rf : cls.atom_type().indexr()) {
                for (int ir = 0; ir < cls.atom_type().num_mt_points(); ir++) {
                    values.push_back(cls.radial_function(ir, rf.idxrf));
                }
            }
        }
        return values;
    };
    auto before = snapshot();
    xc_geometry_direction_t geometry(ctx.unit_cell().num_atoms());
    geometry.atoms[0] = {0.23, -0.37, 0.19};
    LAPW_response response(*reference->ks, gs.density(), potential);
    auto radial       = response.radial_geometry_response(geometry, 1e-8, 200);
    auto explicit_xc  = potential.external_xc_geometry(gs.density());
    double derivative = response.energy_geometry_derivative(geometry, radial, explicit_xc);
    auto const& total = gs.forces().calc_forces_total();
    double total_derivative{0};
    for (int x = 0; x < 3; x++) {
        total_derivative -= geometry.atoms[0][x] * total(x, 0);
    }
    check("complete force API agrees with directional response", std::abs(total_derivative - derivative), 2e-7);
    check("force response preserves the reference state", snapshot() != before ? 1 : 0, 0);

    // The two fourth-order stencils share displaced SCFs. Component derivatives are tested separately.
    std::map<double, std::array<double, 3>> samples;
    auto sample = [&](double step) {
        auto it = samples.find(step);
        if (it == samples.end()) {
            it = samples.emplace(step, calculate(step, false)).first;
        }
        return it->second;
    };
    for (double h : {2e-3, 1e-3}) {
        auto plus          = sample(h);
        auto minus         = sample(-h);
        auto outer_plus    = sample(2 * h);
        auto outer_minus   = sample(-2 * h);
        auto differentiate = [&](int part) {
            return (8 * (plus[part] - minus[part]) - (outer_plus[part] - outer_minus[part])) / (12 * h);
        };
        double fd    = differentiate(0);
        double error = std::abs(fd - total_derivative);
        if (ctx.comm().rank() == 0) {
            std::cout << "LAPW force spin=" << mag << " pw cutoff=" << pw_cutoff << " core=" << core << " h=" << h
                      << " analytic=" << total_derivative << " FD=" << fd << " error=" << error << '\n';
        }
        check("complete force finite difference", error, 2e-6);
        check("force free-energy identity derivative", std::abs(fd - differentiate(1)), 2e-7);
        check("force entropy response", std::abs(differentiate(2) - radial.entropy), 2e-7);
        for (auto const& point : {centre, plus, minus, outer_plus, outer_minus}) {
            check("force free-energy identity", std::abs(point[0] - point[1]), 2e-7);
        }
    }
    return failures;
}

int
test_lapw_cell_response(int mag, bool core, bool spin_orbit, std::string device, std::string relativity,
                        std::string core_relativity)
{
    bool dirac        = relativity == "dirac";
    bool spinor       = mag == 3 && !dirac;
    bool forces       = mag == 3 || relativity != "none";
    int radial_points = core && core_relativity != "none" ? 321 : 0;
    auto calculate    = [&](double step, int component, bool differentiate) {
        r3::matrix<double> strain;
        if (component == 0) {
            strain(0, 0) = 1;
        } else if (component == 1) {
            strain(0, 1) = strain(1, 0) = 0.5;
        }
        auto ctx = context(mag, true, spinor, core, false, 0, 9.0, false, 0, "fermi_dirac", core, device, spin_orbit,
                              relativity, true, core_relativity, radial_points);
        if (mag == 3) {
            // Fractional occupations retain a finite response to the vector field in this closed-shell fixture.
            ctx->cfg().unlock();
            ctx->cfg().parameters().smearing_width(0.2);
            ctx->cfg().lock();
        }
        if (ctx->num_bands() * ctx->max_occupancy() <= ctx->unit_cell().num_valence_electrons()) {
            RTE_THROW("LAPW cell response fixture requires unoccupied bands");
        }
        if (device == "GPU" &&
            (ctx->processing_unit() != device_t::GPU || ctx->spfft<double>().processing_unit() != SPFFT_PU_GPU ||
             ctx->spfft_coarse<double>().processing_unit() != SPFFT_PU_GPU ||
             !is_device_memory(ctx->gen_evp_solver().data_memory_t()) ||
             !is_device_memory(ctx->std_evp_solver().data_memory_t()))) {
            RTE_THROW("LAPW cell test must select GPU FFT grids and eigensolvers");
        }
        auto lattice  = ctx->unit_cell().lattice_vectors();
        auto dlattice = dot(strain, lattice);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                lattice(x, y) += step * dlattice(x, y);
            }
        }
        auto const& original = *ctx->cb().external_xc_points_;
        std::vector<int> attached;
        auto points = std::make_shared<xc_point_set_t>(*ctx);
        for (size_t p = 0; p < original.coordinates.size(); p++) {
            auto const& xyz = original.coordinates[p];
            int owner{-1};
            bool near_boundary{false};
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                auto const& atom = ctx->unit_cell().atom(ia);
                auto centre      = ctx->unit_cell().get_cartesian_coordinates(atom.position());
                auto r           = r3::vector<double>{xyz[0], xyz[1], xyz[2]} - centre;
                if (core && r.length() <= atom.mt_radius() + 1e-10) {
                    owner = ia;
                }
                auto fractional = dot(ctx->unit_cell().inverse_lattice_vectors(), r);
                for (int x = 0; x < 3; x++) {
                    fractional[x] -= std::round(fractional[x]);
                }
                double distance = dot(ctx->unit_cell().lattice_vectors(), fractional).length();
                near_boundary |= std::abs(distance - atom.mt_radius()) < 0.04;
            }
            // Keep domain membership fixed. Resolved MT quadratures retain their local radii.
            if (near_boundary && owner < 0) {
                continue;
            }
            points->coordinates.push_back(xyz);
            points->energy_weights.push_back(original.energy_weights[p]);
            points->descriptor_weights.push_back(original.descriptor_weights[p]);
            attached.push_back(owner);
        }
        int ng             = ctx->gvec().num_gvec();
        auto base_inverse  = ctx->unit_cell().inverse_lattice_vectors();
        double base_volume = ctx->unit_cell().omega();
        std::vector<r3::vector<double>> base_centres;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            base_centres.push_back(ctx->unit_cell().get_cartesian_coordinates(ctx->unit_cell().atom(ia).position()));
        }
        K_point_set ks(*ctx);
        ks.add_kpoint({0.07, -0.03, 0.02}, 1);
        ks.initialize();
        ctx->unit_cell().set_lattice_vectors(lattice);
        r3::vector<double> displacement{0.23, -0.37, 0.19};
        if (component == 2) {
            ctx->unit_cell().atom(0).set_position(dot(inverse(lattice), base_centres[0] + step * displacement));
        }
        ctx->update();
        ks.update();
        if (ng != ctx->gvec().num_gvec()) {
            RTE_THROW("LAPW cell stencil changed the density basis");
        }
        auto transform      = dot(lattice, base_inverse);
        double volume_ratio = ctx->unit_cell().omega() / base_volume;
        auto moved          = std::make_shared<xc_point_set_t>(*ctx);
        for (size_t p = 0; p < points->coordinates.size(); p++) {
            auto const& xyz = points->coordinates[p];
            auto r          = r3::vector<double>{xyz[0], xyz[1], xyz[2]};
            if (attached[p] >= 0) {
                auto const& centre = base_centres[attached[p]];
                r += dot(transform, centre) - centre;
                if (component == 2 && attached[p] == 0) {
                    r += step * displacement;
                }
                auto const& atom  = ctx->unit_cell().atom(attached[p]);
                auto moved_centre = ctx->unit_cell().get_cartesian_coordinates(atom.position());
                if (std::abs((r - moved_centre).length() -
                                (r3::vector<double>{xyz[0], xyz[1], xyz[2]} - centre).length()) > 1e-12) {
                    RTE_THROW("LAPW cell stencil must retain atom-local quadrature radii");
                }
            } else {
                r = dot(transform, r);
            }
            moved->coordinates.push_back({r[0], r[1], r[2]});
            double measure = attached[p] >= 0 ? 1 : volume_ratio;
            moved->energy_weights.push_back(measure * points->energy_weights[p]);
            moved->descriptor_weights.push_back(measure * points->descriptor_weights[p]);
        }
        if (!moved->coordinates.empty()) {
            auto centre = dot(transform, base_centres[0]);
            if (component == 2) {
                centre += step * displacement;
            }
            moved->block_sizes   = {static_cast<int>(moved->coordinates.size())};
            moved->block_centres = {{centre[0], centre[1], centre[2]}};
        }
        ctx->cb().external_xc_points_            = std::move(moved);
        ctx->cb().external_xc_geometry_          = functional;
        ctx->cb().external_xc_geometry_response_ = true;
        if (mag == 3) {
            // A fixed vector field retains finite magnetization in the convex quadratic fixture.
            std::array<double, 4> bias{0, 0.02, 0.013, -0.007};
            ctx->cb().external_xc_ = [bias](auto const& c, auto const& in, auto& out) {
                functional(c, in, out);
                double energy{0};
                for (int s = 1; s < in.num_spins; s++) {
                    for (int p = 0; p < in.num_points; p++) {
                        int j = p + in.num_points * s;
                        energy -= bias[s] * in.points->energy_weights[p] * in.rho[j];
                        out.vrho[j] -= bias[s] * in.points->energy_weights[p];
                        if (out.geometry) {
                            out.geometry->energy_weights[p] -= bias[s] * in.rho[j];
                        }
                    }
                }
                c.comm().allreduce(&energy, 1);
                out.energy += energy;
            };
            ctx->cb().external_xc_geometry_ = ctx->cb().external_xc_;
            ctx->cb().external_xc_response_ = [bias](auto const& c, auto const& in, auto const& d, auto& out) {
                functional_response(c, in, d, out);
                for (int s = 1; s < in.num_spins; s++) {
                    for (int p = 0; p < in.num_points; p++) {
                        int j = p + in.num_points * s;
                        if (d.geometry) {
                            out.vrho[j] -= bias[s] * d.geometry->energy_weights[p];
                        }
                        if (out.geometry) {
                            out.geometry->energy_weights[p] -= bias[s] * d.rho[j];
                        }
                    }
                }
            };
        }
        ctx->cb().external_xc_quadrature_direction_ = [attached](auto const& c, auto const& q, auto const& in,
                                                                 auto& out) {
            r3::matrix<double> dlattice;
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    dlattice(x, y) = in.lattice[x + 3 * y];
                }
            }
            auto deformation = dot(dlattice, c.unit_cell().inverse_lattice_vectors());
            double trace     = deformation(0, 0) + deformation(1, 1) + deformation(2, 2);
            for (size_t p = 0; p < attached.size(); p++) {
                if (attached[p] >= 0) {
                    out.coordinates[p] = in.atoms[attached[p]];
                } else {
                    auto const& xyz           = q.coordinates[p];
                    auto delta                = dot(deformation, r3::vector<double>{xyz[0], xyz[1], xyz[2]});
                    out.coordinates[p]        = {delta[0], delta[1], delta[2]};
                    out.energy_weights[p]     = trace * q.energy_weights[p];
                    out.descriptor_weights[p] = trace * q.descriptor_weights[p];
                }
            }
            for (auto& centre : out.block_centres) {
                centre = in.atoms[0];
            }
        };
        ctx->cb().external_xc_pullback_ = [attached](auto const& c, auto const& q, auto const& in, auto& out) {
            auto const& inverse = c.unit_cell().inverse_lattice_vectors();
            for (size_t p = 0; p < attached.size(); p++) {
                if (attached[p] >= 0) {
                    for (int x = 0; x < 3; x++) {
                        out.atoms[attached[p]][x] += in.coordinates[p][x];
                    }
                } else {
                    auto const& xyz = q.coordinates[p];
                    auto fractional = dot(inverse, r3::vector<double>{xyz[0], xyz[1], xyz[2]});
                    double measure  = q.energy_weights[p] * in.energy_weights[p] +
                                     q.descriptor_weights[p] * in.descriptor_weights[p];
                    for (int y = 0; y < 3; y++) {
                        for (int x = 0; x < 3; x++) {
                            out.lattice[x + 3 * y] += in.coordinates[p][x] * fractional[y] + measure * inverse(y, x);
                        }
                    }
                }
            }
            for (auto const& centre : in.block_centres) {
                for (int x = 0; x < 3; x++) {
                    out.atoms[0][x] += centre[x];
                }
            }
            for (auto& atom : out.atoms) {
                c.comm().allreduce(atom.data(), 3);
            }
            c.comm().allreduce(out.lattice.data(), 9);
        };
        DFT_ground_state gs(ks);
        gs.initial_state();
        auto found = gs.find(core ? 1e-10 : 1e-11, core ? 1e-11 : 1e-12, 1e-11, 200, false);
        if (!found.at("converged").get<bool>()) {
            if (ctx->comm().rank() == 0) {
                auto energies = found.at("etot_history").get<std::vector<double>>();
                std::cout << "LAPW cell SCF final energy change=" << energies.back() - energies[energies.size() - 2]
                          << std::endl;
            }
            RTE_THROW("LAPW cell SCF did not converge");
        }
        double energy = free_energy(*ctx, ks, gs.density(), gs.potential());
        std::array<double, 4> result{energy, 0, 0, 0};
        if (differentiate) {
            if (mag == 3) {
                for (int s = 0; s < 3; s++) {
                    double moment = gs.density().mag(s).integrate().total;
                    if (!std::isfinite(moment) || std::abs(moment) < 1e-6) {
                        std::ostringstream message;
                        message << "spinor forces and stress require finite magnetization; component=" << s
                                << " moment=" << moment;
                        RTE_THROW(message.str());
                    }
                }
            }
            if (relativity != "none" || core_relativity != "none") {
                Hamiltonian0<double> h0(gs.potential(), false);
                if (!dirac) {
                    h0.generate_pw_coefs(gs.potential());
                }
                LAPW_density_response spectrum(h0, ks);
                auto mapped = gs.density().lapw_density_matrix_fields(ks, spectrum.density_matrices());
                for (auto atom : ctx->unit_cell().spl_num_atoms()) {
                    for (int ir = 0; ir < ctx->unit_cell().atom(atom.i).type().num_mt_points(); ir++) {
                        mapped->rho().mt()[atom.i](0, ir) += gs.density().core_charge_density(atom.i)[ir] / y00;
                        mapped->kinetic_density().scalar().mt()[atom.i](0, ir) +=
                                gs.density().core_kinetic_density(atom.i)[ir] / y00;
                    }
                }
                auto reference_fields = lapw_field_samples(gs.density()), spectral_fields = lapw_field_samples(*mapped);
                double error{0};
                for (int part : {0, 1}) {
                    double part_error{0}, peak{1e-12};
                    for (size_t i = 0; i < reference_fields[part].size(); i++) {
                        double residual = std::abs(reference_fields[part][i] - spectral_fields[part][i]);
                        part_error      = std::isfinite(residual) ? std::max(part_error, residual)
                                                                     : std::numeric_limits<double>::infinity();
                        peak            = std::max(peak, std::abs(reference_fields[part][i]));
                    }
                    ctx->comm().allreduce<double, mpi::op_t::max>(&part_error, 1);
                    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                    error = std::max(error, part_error / peak);
                }
                if (!std::isfinite(error) || error > 3e-7 || !mapped->check_num_electrons()) {
                    RTE_THROW("relativistic spectral fields must reproduce the normalized SCF density and tau");
                }
            }
            auto rho_before       = potential_samples(gs.density().rho());
            auto tau_before       = potential_samples(gs.density().kinetic_density().scalar());
            auto potential_before = potential_samples(gs.potential().effective_potential());
            auto stress           = gs.stress().calc_stress_total();
            result[1]             = ctx->unit_cell().omega() * stress(0, 0);
            result[2]             = 0.5 * ctx->unit_cell().omega() * (stress(0, 1) + stress(1, 0));
            if (forces) {
                auto const& force = gs.forces().calc_forces_total();
                for (int x = 0; x < 3; x++) {
                    result[3] -= displacement[x] * force(x, 0);
                }
                std::ostringstream printed;
                gs.forces().print_info(printed, 2);
                if (printed.str().find("total Forces") == std::string::npos ||
                    printed.str().find("contribution from IBS") != std::string::npos) {
                    RTE_THROW("coupled LAPW forces must print their computed total, not the unused IBS decomposition");
                }
            }
            for (int y = 0; y < 3; y++) {
                for (int x = 0; x < 3; x++) {
                    if (!std::isfinite(stress(x, y))) {
                        RTE_THROW("nonfinite LAPW stress tensor");
                    }
                }
            }
            if (rho_before != potential_samples(gs.density().rho()) ||
                tau_before != potential_samples(gs.density().kinetic_density().scalar()) ||
                potential_before != potential_samples(gs.potential().effective_potential()) ||
                energy != free_energy(*ctx, ks, gs.density(), gs.potential())) {
                RTE_THROW("LAPW cell response changed the reference state");
            }
        }
        return result;
    };
    auto reference = calculate(0, 0, true);
    std::map<std::pair<int, double>, double> samples;
    auto sample = [&](double step, int component) {
        auto key = std::make_pair(component, step);
        auto it  = samples.find(key);
        if (it == samples.end()) {
            it = samples.emplace(key, calculate(step, component, false)[0]).first;
        }
        return it->second;
    };
    int failures{0};
    for (int component = 0; component < (forces ? 3 : 2); component++) {
        for (double h : {2e-3, 1e-3}) {
            double fd = (8 * (sample(h, component) - sample(-h, component)) -
                         (sample(2 * h, component) - sample(-2 * h, component))) /
                        (12 * h);
            double error = std::abs(fd - reference[component + 1]);
            if (mpi::Communicator::world().rank() == 0) {
                std::cout << "LAPW cell " << device << " spin=" << mag << " core=" << core << " SOC=" << spin_orbit
                          << " relativity=" << relativity << " core_relativity=" << core_relativity
                          << " component=" << component << " h=" << h << " analytic=" << reference[component + 1]
                          << " FD=" << fd << " error=" << error << std::endl;
            }
            failures += !std::isfinite(error) || error > 2e-6;
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
    Test_check check(failures, ctx->comm());
    // Test moving and frozen linearisation energies against the same occupied-state contraction.
    for (bool update_enu : {false, true}) {
        double analytic = radial_basis_derivative(gs.potential(), ks, {direction}, {}, update_enu);
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
test_lapw_relativistic_basis_response(std::string const& device)
{
    int failures{0};
    std::vector<std::pair<std::string, bool>> cases{{"iora", false}};
    if (device == "CPU") {
        cases = {{"koelling_harmon", false}, {"zora", false}, {"iora", false}, {"iora", true}};
    }
    for (auto const& [relativity, simple] : cases) {
        auto ctx = context(0, mpi::Communicator::world().size() > 1, false, false, false, 0, 5.0, simple, 1, "", false,
                           device, false, relativity);
        Test_check check(failures, ctx->comm());
        if (device == "GPU") {
            check("scalar-relativistic GPU orbitals and both GPU FFT grids selected",
                  ctx->processing_unit() != device_t::GPU || ctx->spfft<double>().processing_unit() != SPFFT_PU_GPU ||
                                  ctx->spfft_coarse<double>().processing_unit() != SPFFT_PU_GPU
                          ? 1
                          : 0,
                  0);
        }
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
        auto rel         = ctx->valence_relativity();
        std::vector<double> direction(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            direction[ir] = 0.13 * std::exp(-0.3 * grid[ir]) + 0.04 * std::sin(grid[ir]);
        }
        auto& potential = gs.potential();
        auto& effective = potential.effective_potential();
        std::vector<double> base_rg(effective.rg().values().begin(), effective.rg().values().end());
        std::vector<double> base_mt(effective.mt()[0].begin(), effective.mt()[0].end());
        Periodic_function<double> hartree(*ctx, [&](int) { return lmax_t(ctx->lmax_pot()); });
        hartree.zero();
        for (int ig = 0; ig < ctx->gvec().count(); ig++) {
            auto g = ctx->gvec().gvec_cart(gvec_index_t::local(ig));
            hartree.rg().f_pw_local(ig) =
                    std::polar(0.09 * std::exp(-0.8 * g.length() * g.length()), 0.6 * g[0] - 0.2 * g[1] + 0.3 * g[2]);
        }
        hartree.rg().fft_transform(1);
        for (auto it : ctx->unit_cell().spl_num_atoms()) {
            for (int ir = 0; ir < grid.num_points(); ir++) {
                hartree.mt()[it.i](0, ir) = direction[ir] / y00;
                hartree.mt()[it.i](2, ir) = 0.03 * grid[ir] * std::exp(-grid[ir]);
            }
        }
        hartree.mt().sync(ctx->unit_cell().spl_num_atoms());
        Spline<double> prototype(grid);
        mdarray<double, 2> integral({grid.num_points(), 4});
        integral.zero();
        for (int ir = 0; ir < grid.num_points() - 1; ir++) {
            double h = grid[ir + 1] - grid[ir];
            for (int j = 0; j < 4; j++) {
                integral(ir, j) = std::pow(h, j + 1) / (j + 1);
            }
        }
        auto weights = prototype.interpolation_adjoint(integral);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            weights[ir] *= fourpi * grid[ir] * grid[ir];
        }
        auto covectors = [&](double scale) {
            lapw_xc_field_adjoint_t xc;
            xc.rho.emplace_back(ctx->spfft<double>(), ctx->gvec_fft_sptr());
            xc.tau.emplace_back(ctx->spfft<double>(), ctx->gvec_fft_sptr());
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                xc.rho[0].value(ir) = scale * 0.03 * ctx->theta(ir);
                xc.tau[0].value(ir) = scale * 0.02 * ctx->theta(ir);
            }
            xc.rho[0].fft_transform(-1);
            xc.tau[0].fft_transform(-1);
            xc.local.resize(1);
            xc.core.resize(1);
            for (int p : {0, 1}) {
                xc.local[0][p].emplace_back(sf::lmmax(ctx->lmax_rho()), grid);
                xc.local[0][p][0].zero();
                xc.core[0][p].resize(grid.num_points());
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    xc.local[0][p][0](0, ir) = scale * (p ? 0.02 : 0.03) * weights[ir] * std::exp(-grid[ir]);
                    xc.core[0][p][ir]        = xc.local[0][p][0](0, ir) / y00;
                }
            }
            return xc;
        };
        int part{0}; // Radial basis, physical fields, or both.
        auto matrices = [&](double step, bool update_enu) {
            auto v = base;
            for (size_t ir = 0; ir < v.size(); ir++) {
                v[ir] += step * direction[ir];
            }
            cls.set_spherical_potential(v);
            if (part != 1 && cls.generate_radial_functions(rel, update_enu)) {
                RTE_THROW("scalar-relativistic basis fixture failed");
            }
            double field_step = part ? step : 0;
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                effective.rg().value(ir) = base_rg[ir] + field_step * hartree.rg().value(ir);
            }
            for (size_t i = 0; i < base_mt.size(); i++) {
                effective.mt()[0][i] = base_mt[i] + field_step * hartree.mt()[0][i];
            }
            effective.rg().fft_transform(-1);
            potential.set_lapw_xc_derivatives(covectors(field_step));
            potential.update_atomic_potential();
            ctx->unit_cell().generate_radial_integrals();
            Hamiltonian0<double> h0(gs.potential(), false);
            h0.generate_pw_coefs(gs.potential());
            std::array<std::vector<std::complex<double>>, 3> result;
            for (auto it : ks.spl_num_kpoints()) {
                auto kp = ks.get<double>(it.i);
                auto hk = h0(*kp);
                int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                hk.set_fv_h_o(0, h, o);
                result[0].insert(result[0].end(), h.at(memory_t::host), h.at(memory_t::host) + h.size_local());
                result[1].insert(result[1].end(), o.at(memory_t::host), o.at(memory_t::host) + o.size_local());
                hk.set_fv_h_o(0, h, o, true);
                result[2].insert(result[2].end(), o.at(memory_t::host), o.at(memory_t::host) + o.size_local());
            }
            return result;
        };
        for (part = 0; part < 3; part++) {
            if (simple && part != 2) {
                continue;
            }
            for (bool update_enu : {false, true}) {
                if (part && !update_enu) {
                    continue;
                }
                auto baseline = matrices(0, update_enu);
                std::vector<radial_basis_response_t> response;
                if (part != 1) {
                    response.push_back(cls.radial_basis_potential_response(direction, {}, update_enu));
                } else {
                    response.emplace_back(cls.atom_type());
                }
                Hamiltonian0<double> h0(gs.potential(), false);
                h0.generate_pw_coefs(gs.potential());
                auto atoms          = h0.lapw_radial_basis_response(response);
                auto physical_atoms = h0.lapw_radial_basis_response(response, false, true);
                std::optional<lapw_hamiltonian_response_t> field;
                if (part) {
                    field = h0.lapw_field_response(hartree, covectors(1));
                }
                if (rel == relativity_t::iora && part != 1) {
                    double peak{0};
                    for (int j = 0; j < cls.atom_type().mt_aw_basis_size(); j++) {
                        for (int i = 0; i < cls.atom_type().mt_aw_basis_size(); i++) {
                            peak = std::max(peak, std::abs(atoms[0].o(i, j)));
                        }
                    }
                    check("nonzero IORA AW overlap correction response", peak > 1e-12 ? 0 : 1, 0);
                }
                std::array<std::vector<std::complex<double>>, 3> analytic;
                for (auto it : ks.spl_num_kpoints()) {
                    auto kp = ks.get<double>(it.i);
                    auto hk = h0(*kp);
                    int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                    la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> physical_h(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> physical_o(n, n, ctx->blacs_grid(), bs, bs);
                    hk.set_fv_h_o_radial_response(0, response, atoms, h, o);
                    hk.set_fv_h_o_radial_response(0, response, physical_atoms, physical_h, physical_o);
                    if (field) {
                        la::dmatrix<std::complex<double>> fh(n, n, ctx->blacs_grid(), bs, bs);
                        la::dmatrix<std::complex<double>> fo(n, n, ctx->blacs_grid(), bs, bs);
                        hk.set_fv_h_field_response(0, *field, fh, &fo);
                        for (size_t i = 0; i < h.size_local(); i++) {
                            h.at(memory_t::host)[i] += fh.at(memory_t::host)[i];
                            o.at(memory_t::host)[i] += fo.at(memory_t::host)[i];
                        }
                        if (part == 1 && rel == relativity_t::iora) {
                            check.reject("IORA field response cannot discard overlap",
                                         [&] { hk.set_fv_h_field_response(0, *field, fh); });
                            check.reject("field matrices cannot alias",
                                         [&] { hk.set_fv_h_field_response(0, *field, fh, &fh); });
                            if (!simple) {
                                auto errors = lapw_spectrum_checks(hk, h, &o);
                                check("IORA spectral density response", errors[0], 3e-7);
                                check("IORA degenerate spectral response", errors[1], 3e-7);
                                check("IORA response preserves charge", errors[2], 3e-10);
                                check("IORA invalid spectral inputs rejected", errors[3], 0);
                            }
                        }
                    }
                    if (rel == relativity_t::iora && update_enu && !simple) {
                        auto errors = lapw_spectrum_checks(hk, h, &o, &physical_o);
                        check("IORA physically normalized density response", errors[0], 3e-7);
                        check("IORA normalized degenerate response", errors[1], 3e-7);
                        check("IORA physical response preserves electron count", errors[2], 3e-10);
                        check("IORA physical metric guards", errors[3], 0);
                    }
                    analytic[0].insert(analytic[0].end(), h.at(memory_t::host), h.at(memory_t::host) + h.size_local());
                    analytic[1].insert(analytic[1].end(), o.at(memory_t::host), o.at(memory_t::host) + o.size_local());
                    analytic[2].insert(analytic[2].end(), physical_o.at(memory_t::host),
                                       physical_o.at(memory_t::host) + physical_o.size_local());
                }
                std::cout << "Scalar-relativistic LAPW basis " << relativity << " device=" << device
                          << " simple=" << simple << " update_enu=" << update_enu << " part=" << part << '\n';
                double step = part == 1 ? 0.1 : 0.01;
                for (double h : {step, step / 2}) {
                    auto plus = matrices(h, update_enu), minus = matrices(-h, update_enu);
                    for (int p : {0, 1, 2}) {
                        double error{0}, peak{0};
                        for (size_t i = 0; i < analytic[p].size(); i++) {
                            error = std::max(error, std::abs(analytic[p][i] - (plus[p][i] - minus[p][i]) / (2 * h)));
                            peak  = std::max(peak, std::abs(analytic[p][i]));
                        }
                        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                        ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                        if (part == 1 && (p == 2 || (p == 1 && rel != relativity_t::iora))) {
                            check("zero non-IORA overlap field response", std::max(error, peak), 1e-14);
                            continue;
                        }
                        check("nonzero scalar-relativistic matrix response", peak > 1e-14 ? 0 : 1, 0);
                        check(p == 0 ? "scalar-relativistic Hamiltonian derivative"
                                     : (p == 1 ? "scalar-relativistic overlap derivative"
                                               : "physical overlap derivative"),
                              error / std::max(1e-12, peak), 3e-5);
                    }
                }
                auto restored = matrices(0, update_enu);
                double error{0};
                for (int part : {0, 1, 2}) {
                    for (size_t i = 0; i < baseline[part].size(); i++) {
                        error = std::max(error, std::abs(restored[part][i] - baseline[part][i]));
                    }
                }
                check("scalar-relativistic reference matrices restored", error, 1e-9);
            }
        }
        auto lattice  = ctx->unit_cell().lattice_vectors();
        auto position = ctx->unit_cell().atom(0).position();
        auto centre   = ctx->unit_cell().get_cartesian_coordinates(position);
        Hamiltonian0<double> h0(potential, false);
        h0.generate_pw_coefs(potential);
        for (int geometry_part = 0; geometry_part < 3; geometry_part++) {
            if (simple && geometry_part != 2) {
                continue;
            }
            xc_geometry_direction_t geometry(1);
            r3::vector<double> displacement;
            if (geometry_part != 1) {
                displacement      = {0.23, -0.37, 0.19};
                geometry.atoms[0] = {displacement[0], displacement[1], displacement[2]};
            }
            r3::matrix<double> strain;
            if (geometry_part != 0) {
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
            std::optional<lapw_interstitial_geometry_response_t> mass;
            if (rel == relativity_t::zora || rel == relativity_t::iora) {
                mass = h0.lapw_interstitial_geometry_response(geometry);
            }
            auto snapshot = [&](double step, bool response) {
                std::vector<std::complex<double>> delta_theta(ctx->gvec().num_gvec());
                std::array<std::vector<std::complex<double>>, 2> delta_mass;
                if (!response) {
                    auto shifted = lattice;
                    for (int y = 0; y < 3; y++) {
                        for (int x = 0; x < 3; x++) {
                            shifted(x, y) += step * dlattice(x, y);
                        }
                    }
                    ctx->unit_cell().set_lattice_vectors(shifted);
                    ctx->unit_cell().atom(0).set_position(dot(inverse(shifted), centre + displacement * step));
                    for (int ig = 0; ig < ctx->gvec().num_gvec(); ig++) {
                        auto index = ctx->gvec().gvec(gvec_index_t::global(ig));
                        auto g     = dot(ctx->unit_cell().reciprocal_lattice_vectors(),
                                         r3::vector<double>(index[0], index[1], index[2]));
                        auto moved = (ig == 0 ? 1.0 : 0.0) -
                                     fourpi / ctx->unit_cell().omega() *
                                             unit_step_function_form_factors(cls.atom_type().mt_radius(), g.length()) *
                                             std::exp(std::complex<double>(0, -dot(g, centre + displacement * step)));
                        delta_theta[ig] = moved - ctx->theta_pw(ig);
                    }
                    if (mass) {
                        Smooth_periodic_function<double> mask(ctx->spfft<double>(), ctx->gvec_fft_sptr());
                        for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                            mask.f_pw_local(ig) = delta_theta[ctx->gvec().offset() + ig];
                        }
                        mask.fft_transform(1);
                        Smooth_periodic_function<double> product(ctx->spfft<double>(), ctx->gvec_fft_sptr());
                        double alpha_half = 0.5 / (speed_of_light * speed_of_light);
                        for (int p = 0; p < (rel == relativity_t::iora ? 2 : 1); p++) {
                            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                                double minv       = 1 / (1 - alpha_half * effective.rg().value(ir));
                                product.value(ir) = mask.value(ir) * minv * (p ? alpha_half * minv : 1);
                            }
                            product.fft_transform(-1);
                            delta_mass[p] = product.gather_f_pw();
                        }
                    }
                }
                std::array<std::vector<std::complex<double>>, 3> result;
                for (auto it : ks.spl_num_kpoints()) {
                    auto kp = ks.get<double>(it.i);
                    if (!response) {
                        kp->update();
                    }
                    auto hk = h0(*kp);
                    int n = kp->gklo_basis_size(), bs = ctx->cyclic_block_size();
                    la::dmatrix<std::complex<double>> h(n, n, ctx->blacs_grid(), bs, bs);
                    la::dmatrix<std::complex<double>> o(n, n, ctx->blacs_grid(), bs, bs);
                    if (response) {
                        hk.set_fv_h_o_geometry_response(0, geometry, h, o, false, mass ? &*mass : nullptr);
                        if (mass && geometry_part == 0) {
                            check.reject("scalar mass geometry payload required",
                                         [&] { hk.set_fv_h_o_geometry_response(0, geometry, h, o); });
                            auto bad = *mass;
                            bad.direction[0] += 0.01;
                            check.reject("mass geometry direction must match",
                                         [&] { hk.set_fv_h_o_geometry_response(0, geometry, h, o, false, &bad); });
                            bad = *mass;
                            if (kp->comm().rank() == 0) {
                                bad.kinetic_pw[0] = std::numeric_limits<double>::quiet_NaN();
                            }
                            check.reject("nonfinite mass geometry payload rejected collectively",
                                         [&] { hk.set_fv_h_o_geometry_response(0, geometry, h, o, false, &bad); });
                        }
                    } else {
                        hk.set_fv_h_o(0, h, o);
                        for (int j = 0; j < kp->num_gkvec_col(); j++) {
                            auto qj = kp->gkvec_col().gkvec_cart(gvec_index_t::local(j));
                            for (int i = 0; i < kp->num_gkvec_row(); i++) {
                                auto qi  = kp->gkvec_row().gkvec_cart(gvec_index_t::local(i));
                                int ig   = ctx->gvec().index_g12(kp->gkvec_row().gvec(gvec_index_t::local(i)),
                                                                 kp->gkvec_col().gvec(gvec_index_t::local(j)));
                                double t = 0.5 * dot(qi, qj);
                                h(i, j) += t * (mass ? delta_mass[0][ig] : delta_theta[ig]);
                                o(i, j) += delta_theta[ig] + (rel == relativity_t::iora ? t * delta_mass[1][ig] : 0);
                            }
                        }
                    }
                    result[0].insert(result[0].end(), h.at(memory_t::host), h.at(memory_t::host) + h.size_local());
                    result[1].insert(result[1].end(), o.at(memory_t::host), o.at(memory_t::host) + o.size_local());
                    if (response) {
                        hk.set_fv_h_o_geometry_response(0, geometry, h, o, false, mass ? &*mass : nullptr, true);
                    } else {
                        hk.set_fv_h_o(0, h, o, true);
                        for (int j = 0; j < kp->num_gkvec_col(); j++) {
                            for (int i = 0; i < kp->num_gkvec_row(); i++) {
                                int ig = ctx->gvec().index_g12(kp->gkvec_row().gvec(gvec_index_t::local(i)),
                                                               kp->gkvec_col().gvec(gvec_index_t::local(j)));
                                o(i, j) += delta_theta[ig];
                            }
                        }
                    }
                    result[2].insert(result[2].end(), o.at(memory_t::host), o.at(memory_t::host) + o.size_local());
                }
                return result;
            };
            auto analytic = snapshot(0, true);
            std::cout << "Scalar-relativistic LAPW geometry " << relativity << " device=" << device
                      << " simple=" << simple << " part=" << geometry_part << '\n';
            for (double step : {1e-4, 5e-5}) {
                auto plus = snapshot(step, false), minus = snapshot(-step, false);
                for (int p : {0, 1, 2}) {
                    double error{0}, peak{0};
                    for (size_t i = 0; i < analytic[p].size(); i++) {
                        double residual = std::abs(analytic[p][i] - (plus[p][i] - minus[p][i]) / (2 * step));
                        error           = std::isfinite(residual) ? std::max(error, residual)
                                                                  : std::numeric_limits<double>::infinity();
                        peak            = std::max(peak, std::abs(analytic[p][i]));
                    }
                    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
                    ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
                    check("nonzero scalar-relativistic geometry response", peak > 1e-7 ? 0 : 1, 0);
                    check(p == 0 ? "scalar-relativistic geometry Hamiltonian derivative"
                                 : (p == 1 ? "scalar-relativistic geometry overlap derivative"
                                           : "physical geometry overlap derivative"),
                          error / std::max(1e-12, peak), 3e-7);
                }
            }
            snapshot(0, false);
        }
    }
    return failures;
}

int
test_lapw_dirac_xc(int mag, bool distributed_fft, std::string device)
{
    auto ctx = context(mag, distributed_fft, false, false, false, 0, 5.0, false, 0, "", false, device, false, "dirac",
                       true);
    int failures{0};
    Test_check check(failures, ctx->comm(), "external Dirac LAPW XC mag=" + std::to_string(mag) + " " + device);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    check("nonzero joint point XC energy", std::abs(potential.energy_exc(density)) > 1e-8 ? 0 : 1, 0);
    auto c = potential.lapw_xc_contractions(density);
    check("Dirac callback Euler identity", std::abs(c.rho + c.magnetic + c.tau - 2 * potential.energy_exc(density)),
          2e-10);
    for (auto const& values : gs.check_external_xc_derivative(1e-4)) {
        check("Dirac PW/full-j LO orbital derivative", std::abs(values[0] - values[1]), 2e-8);
        check("Dirac orbital derivative at halved step", std::abs(values[0] - values[2]), 6e-9);
    }
    for (double error : gs.check_lapw_xc_response(1e-3)) {
        check("Dirac point-field and Hamiltonian response", error, 1e-8);
    }
    Density direction(*ctx);
    direction.zero();
    direction.kinetic_density().zero();
    for (int component = 0; component <= ctx->num_mag_dims(); component++) {
        axpy(0.3 + 0.1 * component, density.component(component), direction.component(component));
        axpy(-0.2 + 0.05 * component, density.kinetic_density().component(component),
             direction.kinetic_density().component(component));
    }
    Hamiltonian0<double> h0(potential, false);
    auto response = h0.lapw_field_response(*potential.hartree_response(direction.rho()),
                                           potential.lapw_external_xc_response(density, direction));
    for (auto it : ks.spl_num_kpoints()) {
        auto& kp = *ks.get<double>(it.i);
        auto hk  = h0(kp);
        int n = 2 * kp.num_gkvec() + ctx->unit_cell().mt_lo_basis_size(), bs = ctx->cyclic_block_size();
        la::dmatrix<std::complex<double>> dh(n, n, ctx->blacs_grid(), bs, bs);
        hk.set_dirac_h_field_response(response, dh);
        auto errors = lapw_spectrum_checks(hk, dh);
        check("Dirac density-matrix spectral derivative", errors[0], 2e-7);
        check("Dirac degenerate spectral derivative", errors[1], 2e-7);
        check("Dirac spectral electron-count derivative", errors[2], 2e-10);
        check("Dirac spectral invalid input rejection", errors[3], 0);
    }
    return failures;
}

int
test_core_guard()
{
    auto ctx = context(0, false, true, true, false, 0, 5.0, false, 0, "", false, "CPU", false, "none", false, "iora");
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
test_lapw_gpu(int mag, bool distributed_fft, bool spin_orbit = false)
{
    int failures{0};
    Test_check check(failures, mpi::Communicator::world(),
                     "external LAPW GPU spin mode=" + std::to_string(mag) + " SOC=" + std::to_string(spin_orbit));
    std::vector<double> reference;
    for (std::string device : {"CPU", "GPU"}) {
        auto ctx = context(mag, distributed_fft, true, false, false, 0, 5.0, false, 0, "", false, device, spin_orbit);
        if (device == "GPU") {
            check("GPU orbitals and both GPU FFT grids selected",
                  ctx->processing_unit() != device_t::GPU || ctx->spfft<double>().processing_unit() != SPFFT_PU_GPU ||
                                  ctx->spfft_coarse<double>().processing_unit() != SPFFT_PU_GPU
                          ? 1
                          : 0,
                  0);
        }
        K_point_set ks(*ctx);
        ks.add_kpoint({0.07, -0.03, 0.02}, 1);
        ks.initialize();
        DFT_ground_state gs(ks);
        gs.initial_state();
        auto& density   = gs.density();
        auto& potential = gs.potential();
        if (spin_orbit) {
            check("second-variation SOC selected", ctx->so_correction() ? 0 : 1, 0);
            Hamiltonian0<double> h0(potential, false);
            std::vector<int> nmt(ctx->unit_cell().num_atoms());
            for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
                nmt[ia] = ctx->unit_cell().atom(ia).mt_basis_size();
            }
            double peak{0};
            for (auto it : ks.spl_num_kpoints()) {
                auto& kp = *ks.get<double>(it.i);
                std::vector<wf::Wave_functions<double>> so;
                for (int j = 0; j < 3; j++) {
                    so.emplace_back(kp.gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(ctx->num_fv_states()),
                                    memory_t::host);
                    so.back().zero(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx->num_fv_states()));
                }
                h0.apply_so_correction(kp.fv_states(), so);
                for (auto const& field : so) {
                    for (auto atom : field.spl_num_atoms()) {
                        for (int b = 0; b < ctx->num_fv_states(); b++) {
                            for (int i = 0; i < nmt[atom.i]; i++) {
                                peak = std::max(peak, std::abs(field.mt_coeffs(i, atom.li, wf::spin_index(0),
                                                                               wf::band_index(b))));
                            }
                        }
                    }
                }
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            check("nonzero SOC operator", std::isfinite(peak) && peak > 1e-12 ? 0 : 1, 0);
        }
        auto c = potential.lapw_xc_contractions(density);
        check("callback Euler identity", std::abs(c.rho + c.magnetic + c.tau - 2 * potential.energy_exc(density)),
              2e-11);
        check("joint point XC field derivative", check_external_xc_point_derivative(density), 2e-10);
        std::vector<double> sample{potential.energy_exc(density)};
        for (auto const* fields : std::array<Field4D const*, 2>{&density, &density.kinetic_density()}) {
            for (int s = 0; s <= mag; s++) {
                auto values = potential_samples(fields->component(s));
                sample.insert(sample.end(), values.begin(), values.end());
            }
        }
        if (device == "CPU") {
            reference = sample;
        } else {
            double error{0}, peak{0};
            for (size_t i = 0; i < sample.size(); i++) {
                error = std::max(error, std::abs(sample[i] - reference[i]));
                peak  = std::max(peak, std::abs(reference[i]));
            }
            check("CPU/GPU density, tau and XC energy", error / std::max(1.0, peak), 1e-10);
        }
        for (auto const& values : gs.check_external_xc_derivative(1e-4)) {
            check("PW/AW/LO orbital derivative", std::abs(values[0] - values[1]), 2e-8);
            check("orbital derivative at halved step", std::abs(values[0] - values[2]), 6e-9);
        }
        for (double error : gs.check_lapw_xc_response(1e-3)) {
            check("field and point response", error, 1e-8);
        }
        // Exercise GPU diagonalization and MT operators without imposing SCF convergence.
        gs.find(1e-9, 1e-10, 1e-9, 1, false);
        check("finite SCF lifecycle energy", std::isfinite(gs.total_energy()) ? 0 : 1, 0);
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

int
test_response_suite(std::string const& mode)
{
    bool coupled     = mode.empty() || mode == "coupled-response";
    bool density     = mode.empty() || mode == "density-response";
    bool occupations = mode.empty() || mode == "occupation-response";
    bool distributed = mpi::Communicator::world().size() > 1;
    int result{0};
    auto run = [&](int mag, bool fft, bool soc) {
        std::string prefix = soc ? "SOC " : mag == 3 ? "noncollinear " : fft ? "distributed " : "";
        if (coupled) {
            result += call_test(prefix + "coupled LAPW response", test_lapw_coupled_response, mag, fft, soc, "CPU",
                                "none", false);
        }
        if (density) {
            result += call_test((prefix.empty() ? "constrained " : prefix) + "LAPW density response",
                                test_lapw_density_response, mag, fft, soc, "none", "CPU");
        }
        if (occupations && !soc) {
            result += call_test((prefix.empty() ? "weighted k-point " : prefix) + "occupation response",
                                test_occupation_response, mag, fft, false);
        }
    };
    for (int mag : {0, 1}) {
        run(mag, distributed, false);
    }
    run(3, distributed, false);
    if (coupled || density) {
        run(3, distributed, true);
    }
    if (occupations) {
        result += call_test("Gaussian occupation response", test_occupation_response, 0, false, true);
    }
    return result;
}

int
run_tests(int argc, char** argv)
{
    std::string mode = argc > 1 ? argv[1] : "";
    bool distributed = mpi::Communicator::world().size() > 1;
    int result{0};
    if (mode == "dirac") {
        int mag            = argc > 2 ? std::stoi(argv[2]) : 3;
        std::string device = argc > 3 ? argv[3] : "CPU";
        if ((mag != 0 && mag != 1 && mag != 3) || (device != "CPU" && device != "GPU")) {
            RTE_THROW("Dirac XC fixture requires spin mode 0/1/3 and CPU/GPU");
        }
        return call_test("external Dirac LAPW XC", test_lapw_dirac_xc, mag, distributed, device);
    }
    if (mode == "dirac-density") {
        int mag            = argc > 2 ? std::stoi(argv[2]) : 3;
        std::string device = argc > 3 ? argv[3] : "CPU";
        if ((mag != 0 && mag != 1 && mag != 3) || (device != "CPU" && device != "GPU")) {
            RTE_THROW("Dirac density fixture requires spin mode 0/1/3 and CPU/GPU");
        }
        return call_test("native Dirac LAPW density response", test_lapw_density_response, mag, distributed, false,
                         "dirac", device);
    }
    if (mode == "dirac-response") {
        result = call_test("external Dirac LAPW XC", test_lapw_dirac_xc, 3, distributed, "CPU");
        result += call_test("native Dirac LAPW density response", test_lapw_density_response, 3, distributed, false,
                            "dirac", "CPU");
        result += call_test("native Dirac coupled LAPW response", test_lapw_coupled_response, 3, distributed, false,
                            "CPU", "dirac", false);
        return result;
    }
    if (mode == "core-guard") {
        return call_test("external LAPW core response guard", test_core_guard);
    }
    bool scalar_dirac = mode == "scalar-dirac-core-coupled" || mode == "scalar-dirac-core-soc-coupled";
    if (mode == "dirac-coupled" || mode == "dirac-core-coupled" || scalar_dirac) {
        int mag            = argc > 2 ? std::stoi(argv[2]) : 3;
        std::string device = argc > 3 ? argv[3] : "CPU";
        bool soc           = mode == "scalar-dirac-core-soc-coupled";
        if ((mag != 0 && mag != 1 && mag != 3) || (device != "CPU" && device != "GPU") || (soc && mag != 3)) {
            RTE_THROW("coupled Dirac-core fixture requires spin mode 0/1/3 and CPU/GPU");
        }
        return call_test("native Dirac-core coupled LAPW response", test_lapw_coupled_response, mag, distributed, soc,
                         device, scalar_dirac ? "zora" : "dirac", mode != "dirac-coupled");
    }
    if (mode == "dirac-cell" || mode == "dirac-core-cell") {
        int mag            = argc > 2 ? std::stoi(argv[2]) : 0;
        std::string device = argc > 3 ? argv[3] : "CPU";
        if (argc > 4 || (mag != 0 && mag != 1 && mag != 3) || (device != "CPU" && device != "GPU")) {
            RTE_THROW("Dirac LAPW cell test requires spin mode 0/1/3 and CPU/GPU");
        }
        bool core = mode == "dirac-core-cell";
        return call_test("native Dirac LAPW forces and stress", test_lapw_cell_response, mag, core, false, device,
                         "dirac", core ? "dirac" : "none");
    }
    if (mode == "linearization") {
        return call_test("external LAPW linearisation response", test_lapw_linearization_response);
    }
    if (mode == "relativistic-basis") {
        std::string device = argc > 2 ? argv[2] : "CPU";
        if (argc > 3 || (device != "CPU" && device != "GPU")) {
            RTE_THROW("scalar-relativistic basis test requires CPU/GPU");
        }
        return call_test("scalar-relativistic LAPW basis response", test_lapw_relativistic_basis_response, device);
    }
    if (mode == "scalar-cell") {
        std::string rel      = argc > 2 ? argv[2] : "iora";
        std::string device   = argc > 3 ? argv[3] : "CPU";
        int mag              = argc > 4 ? std::stoi(argv[4]) : 0;
        int core             = argc > 5 ? std::stoi(argv[5]) : 0;
        std::string core_rel = argc > 6 ? argv[6] : "none";
        if ((rel != "zora" && rel != "iora" && rel != "koelling_harmon") || (device != "CPU" && device != "GPU") ||
            (mag != 0 && mag != 1) || (core != 0 && core != 1) ||
            (core_rel != "none" && core_rel != "zora" && core_rel != "dirac") || (!core && core_rel != "none")) {
            RTE_THROW(
                    "scalar LAPW cell test requires ZORA/IORA/KH, CPU/GPU, spin/core mode 0/1 and NR/ZORA/Dirac core");
        }
        result = call_test("scalar-relativistic LAPW forces and stress", test_lapw_cell_response, mag, bool(core),
                           false, device, rel, core_rel);
        return result;
    }
    if (mode == "gpu") {
        for (int mag : {0, 3}) {
            result += call_test("external LAPW GPU", test_lapw_gpu, mag, distributed, false);
        }
        result += call_test("external LAPW GPU SOC", test_lapw_gpu, 3, distributed, true);
        return result;
    }
    if (mode == "gpu-coupled") {
        for (int mag : {0, 3}) {
            result += call_test("GPU coupled LAPW response", test_lapw_coupled_response, mag, distributed, mag == 3,
                                "GPU", "none", false);
        }
        return result;
    }
    if (mode == "gpu-cell") {
        int mag = argc > 2 ? std::stoi(argv[2]) : 3;
        if (mag != 0 && mag != 1 && mag != 3) {
            RTE_THROW("GPU LAPW cell test requires spin mode 0, 1 or 3");
        }
        result = call_test("GPU LAPW forces and stress", test_lapw_cell_response, mag, false, mag == 3, "GPU", "none",
                           "none");
        return result;
    }
    if (mode == "spinor" || mode == "core-spinor") {
        return call_test("external LAPW spinor SCF", test_lapw_external, 3, distributed, true, mode == "core-spinor");
    }
    if (mode == "cell-response" || mode == "core-cell") {
        int first = argc > 2 ? std::stoi(argv[2]) : 0;
        int last  = argc > 2 ? first : 1;
        if ((first != 0 && first != 1 && first != 3) || (first == 3 && mode == "core-cell")) {
            RTE_THROW("LAPW cell test requires spin mode 0, 1 or a core-free spinor mode 3");
        }
        for (int mag = first; mag <= last; mag++) {
            result += call_test("external LAPW cell response", test_lapw_cell_response, mag, mode == "core-cell", false,
                                "CPU", "none", "none");
        }
        return result;
    }
    if (mode == "coupled-response" || mode == "density-response" || mode == "occupation-response") {
        return test_response_suite(mode);
    }
    if (mode == "soc-response") {
        result =
                call_test("SOC LAPW density response", test_lapw_density_response, 3, distributed, true, "none", "CPU");
        result += call_test("SOC coupled LAPW response", test_lapw_coupled_response, 3, distributed, true, "CPU",
                            "none", false);
        return result;
    }
    if (mode == "soc-cell") {
        std::string rel      = argc > 2 ? argv[2] : "none";
        std::string device   = argc > 3 ? argv[3] : "CPU";
        int core             = argc > 4 ? std::stoi(argv[4]) : 0;
        std::string core_rel = argc > 5 ? argv[5] : "none";
        if ((rel != "none" && rel != "zora") || (device != "CPU" && device != "GPU") || (core != 0 && core != 1) ||
            (core_rel != "none" && core_rel != "zora" && core_rel != "dirac") || (!core && core_rel != "none")) {
            RTE_THROW("SOC LAPW cell test requires NR/ZORA valence, CPU/GPU, core mode 0/1 and NR/ZORA/Dirac core");
        }
        result = call_test("SOC LAPW forces and stress", test_lapw_cell_response, 3, bool(core), true, device, rel,
                           core_rel);
        return result;
    }
    if (mode == "spinor-spectrum") {
        std::string rel = argc > 2 ? argv[2] : "none";
        if (rel != "none" && rel != "zora") {
            RTE_THROW("LAPW spinor spectrum test requires NR/ZORA valence");
        }
        return call_test("two-stage LAPW spinor response", test_lapw_spinor_spectrum, distributed, rel);
    }
    if (mode == "force-response" || mode == "core-force") {
        int first = argc > 3 ? std::stoi(argv[3]) : 0;
        int last  = argc > 3 ? first : 1;
        if (first < 0 || first > 1) {
            RTE_THROW("LAPW force test requires spin mode 0 or 1");
        }
        for (int mag = first; mag <= last; mag++) {
            result += call_test("external LAPW force response", test_lapw_force_response, mag,
                                argc > 2 ? std::stod(argv[2]) : 9.0, mode == "core-force");
        }
        return result;
    }
    if (mode == "core") {
        for (int mag : {0, 1}) {
            for (bool second : {false, true}) {
                result += call_test("external LAPW core SCF", test_lapw_external, mag, distributed, second, true);
            }
        }
        return result;
    }
    for (int mag : {0, 1}) {
        result += call_test("external LAPW SCF", test_lapw_external, mag, false, bool(mag), false);
    }
    result += call_test("external LAPW spinor SCF", test_lapw_external, 3, distributed, true, false);
    result += call_test("external LAPW core response guard", test_core_guard);
    result += call_test("external LAPW linearisation response", test_lapw_linearization_response);
    result += call_test("replicated LAPW eigenvector redistribution", test_replicated_eigenvector_remap);
    result += test_response_suite("");
    result += call_test("two-stage LAPW spinor response", test_lapw_spinor_spectrum, distributed, "none");
    return result;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result = call_test("LAPW external XC tests", run_tests, argc, argv);
    sirius::finalize();
    return std::min(result, 1);
}
