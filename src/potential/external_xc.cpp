/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "density/lapw_xc_field_map.hpp"
#include "density/paw_xc_field_map.hpp"

namespace sirius {

xc_point_set_t::xc_point_set_t(Simulation_context const& ctx__)
{
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            lattice_[i + 3 * j] = ctx__.unit_cell().lattice_vectors()(i, j);
        }
    }
    for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
        auto r = ctx__.unit_cell().atom(ia).position();
        atoms_.push_back({r[0], r[1], r[2]});
    }
}

void
xc_point_set_t::validate(Simulation_context const& ctx__) const
{
    size_t nr  = coordinates.size();
    int failed = nr > static_cast<size_t>(std::numeric_limits<int>::max()) || energy_weights.size() != nr ||
                 descriptor_weights.size() != nr || block_sizes.size() != block_centres.size() ||
                 atoms_.size() != static_cast<size_t>(ctx__.unit_cell().num_atoms());
    int64_t count{0};
    for (int n : block_sizes) {
        failed |= n <= 0;
        count += n;
    }
    failed |= count != static_cast<int64_t>(nr);
    for (auto const* a : {&coordinates, &block_centres}) {
        for (auto const& r : *a) {
            for (double x : r) {
                failed |= !std::isfinite(x);
            }
        }
    }
    for (auto const* a : {&energy_weights, &descriptor_weights}) {
        for (double w : *a) {
            failed |= !std::isfinite(w);
        }
    }
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            failed |= std::abs(lattice_[i + 3 * j] - ctx__.unit_cell().lattice_vectors()(i, j)) > 1e-12;
        }
    }
    if (atoms_.size() == static_cast<size_t>(ctx__.unit_cell().num_atoms())) {
        for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
            for (int x = 0; x < 3; x++) {
                failed |= std::abs(atoms_[ia][x] - ctx__.unit_cell().atom(ia).position()[x]) > 1e-12;
            }
        }
    }
    int any_points = nr > 0;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&any_points, 1);
    failed |= !any_points;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC: invalid point quadrature or geometry changed since its construction");
    }
}

namespace {

std::array<std::vector<Smooth_periodic_function<double>>, 2>
external_smooth_fields(Simulation_context const& ctx, Density const& density)
{
    mdarray<std::complex<double>, 2> augmentation;
    if (ctx.unit_cell().num_paw_atoms()) {
        // Cancel Q in its original Fourier representation, not on a radial grid.
        augmentation = density.generate_rho_aug();
    }
    std::array<std::vector<Smooth_periodic_function<double>>, 2> fields;
    auto& rho = fields[0];
    auto& tau = fields[1];
    for (int s = 0; s < ctx.num_spins(); s++) {
        rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            auto r = density.rho().rg().f_pw_local(ig);
            auto t = density.kinetic_density().scalar().rg().f_pw_local(ig);
            if (ctx.unit_cell().num_paw_atoms()) {
                r -= augmentation(ig, 0);
            }
            if (ctx.num_spins() == 2) {
                auto m = density.mag(0).rg().f_pw_local(ig);
                if (ctx.unit_cell().num_paw_atoms()) {
                    m -= augmentation(ig, 1);
                }
                r = 0.5 * (r + (1.0 - 2 * s) * m);
                t = 0.5 * (t + (1.0 - 2 * s) * density.kinetic_density().vector(0).rg().f_pw_local(ig));
            }
            rho.back().f_pw_local(ig) = r;
            tau.back().f_pw_local(ig) = t;
        }
    }
    return fields;
}

void
evaluate_external_xc(Simulation_context const& ctx, xc_callback_input_t const& input, xc_callback_output_t& output)
{
    size_t n    = static_cast<size_t>(input.num_points) * input.num_spins;
    auto finite = [](double const* p, size_t count) {
        for (size_t i = 0; i < count; i++) {
            if (!std::isfinite(p[i])) {
                return false;
            }
        }
        return true;
    };
    int failed = !finite(input.rho, n) || !finite(input.gradient, 3 * n) || !finite(input.tau, n);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC received nonfinite reconstructed fields on a context rank");
    }
    std::string message;
    try {
        auto const& callback = output.geometry ? ctx.cb().external_xc_geometry_ : ctx.cb().external_xc_;
        callback(ctx, input, output);
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    failed |= !std::isfinite(output.energy) || !finite(output.vrho, n) || !finite(output.vgradient, 3 * n) ||
              !finite(output.vtau, n);
    if (output.geometry) {
        auto const& d = *output.geometry;
        failed |= !input.points || d.coordinates.size() != static_cast<size_t>(input.num_points) ||
                  d.energy_weights.size() != d.coordinates.size() ||
                  d.descriptor_weights.size() != d.coordinates.size();
        if (input.points) {
            failed |= d.block_centres.size() != input.points->block_centres.size();
        }
        for (auto const* v : {&d.coordinates, &d.block_centres}) {
            for (auto const& r : *v) {
                failed |= !finite(r.data(), r.size());
            }
        }
        failed |= !finite(d.energy_weights.data(), d.energy_weights.size()) ||
                  !finite(d.descriptor_weights.data(), d.descriptor_weights.size());
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC failed or returned nonfinite output on a context rank: " + message);
    }
    double emin = output.energy, emax = output.energy;
    ctx.comm().allreduce<double, mpi::op_t::min>(&emin, 1);
    ctx.comm().allreduce<double, mpi::op_t::max>(&emax, 1);
    if (emax - emin > 1e-12 * std::max(1.0, std::max(std::abs(emin), std::abs(emax)))) {
        RTE_THROW("external XC must return the same total energy on every context rank");
    }
}

} // namespace

void
validate_external_xc_context(Simulation_context const& ctx__)
{
    if (!ctx__.cb().external_xc_) {
        return;
    }
    auto const& points = ctx__.cb().external_xc_points_;
    int point_min = bool(points), point_max = point_min;
    ctx__.comm().allreduce<int, mpi::op_t::min>(&point_min, 1);
    ctx__.comm().allreduce<int, mpi::op_t::max>(&point_max, 1);
    if (point_min != point_max) {
        RTE_THROW("external XC point/FFT quadrature mode differs between context ranks");
    }
    if ((ctx__.full_potential() || ctx__.unit_cell().augment()) && !points) {
        RTE_THROW("external XC: PAW and full potential require reconstructed point quadrature");
    }
    if (ctx__.processing_unit() != device_t::CPU || ctx__.spfft<double>().processing_unit() != SPFFT_PU_HOST ||
        ctx__.spfft_coarse<double>().processing_unit() != SPFFT_PU_HOST || ctx__.num_mag_dims() == 3 ||
        ctx__.so_correction()) {
        RTE_THROW("external XC currently requires CPU FFTs and nonmagnetic or collinear scalar orbitals");
    }
    if (!ctx__.xc_functionals().empty() || ctx__.cb().veff_) {
        RTE_THROW("external XC cannot be combined with built-in XC or an effective-potential callback");
    }
    if (ctx__.cfg().parameters().extra_charge() != 0) {
        RTE_THROW("external XC: charged-cell background must be separated from the physical density first");
    }
    if (ctx__.cfg().parameters().veff_pw_cutoff() > 0) {
        RTE_THROW("external XC does not support an additional potential-only Fourier filter");
    }
    if (points) {
        points->validate(ctx__);
        if (ctx__.cfg().parameters().use_symmetry()) {
            RTE_THROW("external XC point quadrature requires symmetry disabled until its symmetry map is supplied");
        }
    }
    if (ctx__.full_potential()) {
        if (ctx__.valence_relativity() != relativity_t::none || ctx__.cfg().parameters().precision_wf() != "fp64") {
            RTE_THROW("external LAPW XC currently requires nonrelativistic fp64 valence orbitals");
        }
        if (ctx__.unit_cell().num_core_electrons() != 0 && ctx__.core_relativity() != relativity_t::none) {
            RTE_THROW("external LAPW XC: tau-dependent radial core response requires nonrelativistic core states");
        }
        return;
    }
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        auto const& type = ctx__.unit_cell().atom_type(iat);
        auto const& core = type.ps_core_charge_density();
        if ((type.augment() && !type.is_paw()) || type.spin_orbit_coupling() ||
            (!type.is_paw() && std::any_of(core.begin(), core.end(), [](double x) { return x != 0; }))) {
            RTE_THROW("external XC: this atomic setup needs core or spinor reconstruction");
        }
        if (type.is_paw() && (!points || !type.has_paw_core_kinetic_density())) {
            RTE_THROW("external XC: PAW requires a point quadrature and explicitly supplied positive core tau");
        }
    }
}

void
Potential::xc_external(Density const& density__)
{
    PROFILE("sirius::Potential::xc_external");
    validate_external_xc_context(ctx_);
    if (ctx_.cb().external_xc_points_) {
        xc_external_points(density__);
        return;
    }
    auto& fft = ctx_.spfft<double>();
    int nr    = fft.local_slice_size();
    int ns    = ctx_.num_spins();
    mdarray<double, 2> rho({nr, ns}), tau({nr, ns}), vrho({nr, ns}), vtau({nr, ns});
    mdarray<double, 3> grad({nr, 3, ns}), vgrad({nr, 3, ns});
    vrho.zero();
    vtau.zero();
    vgrad.zero();

    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> r(fft, ctx_.gvec_fft_sptr());
        for (int ir = 0; ir < nr; ir++) {
            double n = (1 + add_delta_rho_xc_) * density__.rho().rg().value(ir);
            double t = density__.kinetic_density().scalar().rg().value(ir);
            if (ns == 2) {
                n = 0.5 * (n + (1 - 2 * s) * (1 + add_delta_mag_xc_) * density__.mag(0).rg().value(ir));
                t = 0.5 * (t + (1 - 2 * s) * density__.kinetic_density().vector(0).rg().value(ir));
            }
            rho(ir, s) = r.value(ir) = n;
            tau(ir, s)               = t;
        }
        r.fft_transform(-1);
        auto g = to_rg(gradient(r));
        for (int x = 0; x < 3; x++) {
            for (int ir = 0; ir < nr; ir++) {
                grad(ir, x, s) = g[x].value(ir);
            }
        }
    }

    xc_callback_input_t input{nr,
                              ns,
                              unit_cell_.omega() / ctx_.fft_grid().num_points(),
                              rho.at(memory_t::host),
                              grad.at(memory_t::host),
                              tau.at(memory_t::host)};
    xc_callback_output_t output{0, vrho.at(memory_t::host), vgrad.at(memory_t::host), vtau.at(memory_t::host)};
    evaluate_external_xc(ctx_, input, output);
    external_xc_energy_ = output.energy;
    xc_potential_->zero();
    kinetic_potential_->zero();

    for (int s = 0; s < ns; s++) {
        Smooth_periodic_vector_function<double> g(fft, ctx_.gvec_fft_sptr());
        for (int x = 0; x < 3; x++) {
            for (int ir = 0; ir < nr; ir++) {
                g[x].value(ir) = vgrad(ir, x, s);
            }
            g[x].fft_transform(-1);
        }
        auto div = to_rg(divergence(g));
        for (int ir = 0; ir < nr; ir++) {
            double v      = vrho(ir, s) - div.value(ir);
            double factor = ns == 1 ? 1.0 : 0.5;
            xc_potential_->rg().value(ir) += factor * v;
            kinetic_potential_->scalar().rg().value(ir) += factor * vtau(ir, s);
            if (ns == 2) {
                effective_magnetic_field(0).rg().value(ir) += (1 - 2 * s) * factor * v;
                kinetic_potential_->vector(0).rg().value(ir) += (1 - 2 * s) * factor * vtau(ir, s);
            }
        }
    }
    kinetic_potential_->fft_transform(-1);
}

void
Potential::xc_external_points(Density const& density__)
{
    PROFILE("sirius::Potential::xc_external_points");
    if (add_delta_rho_xc_ != 0 || add_delta_mag_xc_ != 0) {
        RTE_THROW("external XC point reconstruction does not support scaled-density diagnostic perturbations");
    }
    auto const& quadrature = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : quadrature.coordinates) {
        points.push_back({r[0], r[1], r[2]});
    }
    int nr      = static_cast<int>(points.size());
    int ns      = ctx_.num_spins();
    auto smooth = external_smooth_fields(ctx_, density__);
    auto& rho   = smooth[0];
    auto& tau   = smooth[1];
    xc_point_fields_t derivatives(nr, ns);
    auto evaluate = [&](auto const& map) {
        auto fields = map.sample(rho, tau);
        derivatives.rho.zero();
        derivatives.gradient.zero();
        derivatives.tau.zero();
        xc_callback_input_t input{nr,
                                  ns,
                                  0,
                                  fields.rho.at(memory_t::host),
                                  fields.gradient.at(memory_t::host),
                                  fields.tau.at(memory_t::host),
                                  &quadrature};
        xc_callback_output_t output{0, derivatives.rho.at(memory_t::host), derivatives.gradient.at(memory_t::host),
                                    derivatives.tau.at(memory_t::host)};
        evaluate_external_xc(ctx_, input, output);
        return output.energy;
    };
    xc_potential_->zero();
    kinetic_potential_->zero();
    if (ctx_.full_potential()) {
        LAPW_xc_field_map map(density__, points);
        double energy = evaluate(map);
        set_lapw_xc_derivatives(map.adjoint(derivatives), energy);
        return;
    }
    PAW_xc_field_map map(density__, points);
    double energy        = evaluate(map);
    auto adjoint         = map.adjoint(derivatives);
    external_xc_energy_  = energy;
    external_paw_d_mtrx_ = std::move(adjoint.local_matrices);
    for (int s = 0; s < ns; s++) {
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            auto v = adjoint.rho[s].f_pw_local(ig) / double(ns);
            auto t = adjoint.tau[s].f_pw_local(ig) / double(ns);
            xc_potential_->rg().f_pw_local(ig) += v;
            kinetic_potential_->scalar().rg().f_pw_local(ig) += t;
            if (ns == 2) {
                effective_magnetic_field(0).rg().f_pw_local(ig) += (1.0 - 2 * s) * v;
                kinetic_potential_->vector(0).rg().f_pw_local(ig) += (1.0 - 2 * s) * t;
            }
        }
    }
    xc_potential_->rg().fft_transform(1);
    kinetic_potential_->fft_transform(1);
    if (ns == 2) {
        effective_magnetic_field(0).rg().fft_transform(1);
    }
}

void
Potential::external_xc_point_response(xc_callback_input_t const& input__, xc_callback_direction_t const& direction__,
                                      xc_callback_response_t& output__) const
{
    int failed = !ctx_.cb().external_xc_response_ || !input__.points || input__.num_points < 0 ||
                 input__.num_spins != ctx_.num_spins() ||
                 ((direction__.geometry || output__.geometry) && !ctx_.cb().external_xc_geometry_response_);
    if (input__.points) {
        failed |= size_t(input__.num_points) != input__.points->coordinates.size();
    }
    auto geometry_matches = [&](auto const* g) {
        return !g || (input__.points && g->coordinates.size() == input__.points->coordinates.size() &&
                      g->block_centres.size() == input__.points->block_centres.size() &&
                      g->energy_weights.size() == input__.points->energy_weights.size() &&
                      g->descriptor_weights.size() == input__.points->descriptor_weights.size());
    };
    failed |= !geometry_matches(direction__.geometry) || !geometry_matches(output__.geometry);
    auto finite = [&](double const* p, size_t n) {
        if (!p && n) {
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(p[i])) {
                return false;
            }
        }
        return true;
    };
    size_t n = size_t(std::max(0, input__.num_points)) * ctx_.num_spins();
    failed |= !finite(input__.rho, n) || !finite(input__.gradient, 3 * n) || !finite(input__.tau, n) ||
              !finite(direction__.rho, n) || !finite(direction__.gradient, 3 * n) || !finite(direction__.tau, n) ||
              (!output__.vrho && n) || (!output__.vgradient && n) || (!output__.vtau && n);
    auto finite_geometry = [&](auto const* g) {
        if (!g) {
            return true;
        }
        for (auto const* a : {&g->coordinates, &g->block_centres}) {
            for (auto const& r : *a) {
                if (!finite(r.data(), 3)) {
                    return false;
                }
            }
        }
        return finite(g->energy_weights.data(), g->energy_weights.size()) &&
               finite(g->descriptor_weights.data(), g->descriptor_weights.size());
    };
    failed |= !finite_geometry(direction__.geometry);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("invalid external XC point response inputs or unsupported quadrature direction");
    }
    std::string message;
    try {
        ctx_.cb().external_xc_response_(ctx_, input__, direction__, output__);
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    failed |= !finite(output__.vrho, n) || !finite(output__.vgradient, 3 * n) || !finite(output__.vtau, n) ||
              !finite_geometry(output__.geometry);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC point response failed or returned nonfinite output on a context rank: " + message);
    }
}

lapw_xc_field_adjoint_t
Potential::lapw_external_xc_response(Density const& density__, Density const& density_response__) const
{
    int failed = !ctx_.full_potential() || !ctx_.cb().external_xc_ || !ctx_.cb().external_xc_points_ ||
                 !ctx_.cb().external_xc_response_ || &density__.ctx() != &ctx_ || &density_response__.ctx() != &ctx_ ||
                 !density__.has_kinetic_density() || !density_response__.has_kinetic_density() ||
                 add_delta_rho_xc_ != 0 || add_delta_mag_xc_ != 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external LAPW XC response requires matching physical fields and a point-response callback");
    }
    validate_external_xc_context(ctx_);
    auto const& quadrature = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : quadrature.coordinates) {
        points.emplace_back(r[0], r[1], r[2]);
    }
    LAPW_xc_field_map map(density__, points), direction_map(density_response__, points);
    auto smooth           = external_smooth_fields(ctx_, density__);
    auto smooth_direction = external_smooth_fields(ctx_, density_response__);
    auto fields           = map.sample(smooth[0], smooth[1]);
    auto direction        = direction_map.sample(smooth_direction[0], smooth_direction[1]);
    xc_point_fields_t response(points.size(), ctx_.num_spins());
    response.rho.zero();
    response.gradient.zero();
    response.tau.zero();
    xc_callback_input_t input{static_cast<int>(points.size()),
                              ctx_.num_spins(),
                              0,
                              fields.rho.at(memory_t::host),
                              fields.gradient.at(memory_t::host),
                              fields.tau.at(memory_t::host),
                              &quadrature};
    xc_callback_direction_t delta{direction.rho.at(memory_t::host), direction.gradient.at(memory_t::host),
                                  direction.tau.at(memory_t::host)};
    xc_callback_response_t output{response.rho.at(memory_t::host), response.gradient.at(memory_t::host),
                                  response.tau.at(memory_t::host)};
    external_xc_point_response(input, delta, output);
    return map.adjoint(response);
}

lapw_xc_field_adjoint_t
Potential::lapw_external_xc_geometry_response(Density const& density__,
                                              xc_point_derivatives_t const& quadrature_direction__,
                                              mdarray<double, 2> const& atom_directions__,
                                              r3::matrix<double> const& lattice_direction__) const
{
    int failed = !ctx_.full_potential() || !ctx_.cb().external_xc_ || !ctx_.cb().external_xc_points_ ||
                 !ctx_.cb().external_xc_response_ || !ctx_.cb().external_xc_geometry_response_ ||
                 &density__.ctx() != &ctx_ || !density__.has_kinetic_density() || add_delta_rho_xc_ != 0 ||
                 add_delta_mag_xc_ != 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external LAPW XC geometry response requires physical fields and a joint point Hessian");
    }
    validate_external_xc_context(ctx_);
    auto const& q = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points, point_directions;
    for (auto const& r : q.coordinates) {
        points.emplace_back(r[0], r[1], r[2]);
    }
    for (auto const& r : quadrature_direction__.coordinates) {
        point_directions.emplace_back(r[0], r[1], r[2]);
    }
    LAPW_xc_field_map map(density__, points);
    auto smooth = external_smooth_fields(ctx_, density__);
    auto fields = map.sample(smooth[0], smooth[1]);
    auto direction =
            map.geometry_response(smooth[0], smooth[1], point_directions, atom_directions__, lattice_direction__);
    xc_point_fields_t response(points.size(), ctx_.num_spins()), base(points.size(), ctx_.num_spins());
    response.rho.zero();
    response.gradient.zero();
    response.tau.zero();
    base.rho.zero();
    base.gradient.zero();
    base.tau.zero();
    xc_callback_input_t input{static_cast<int>(points.size()),
                              ctx_.num_spins(),
                              0,
                              fields.rho.at(memory_t::host),
                              fields.gradient.at(memory_t::host),
                              fields.tau.at(memory_t::host),
                              &q};
    xc_callback_direction_t delta{direction.rho.at(memory_t::host), direction.gradient.at(memory_t::host),
                                  direction.tau.at(memory_t::host), &quadrature_direction__};
    xc_callback_response_t output{response.rho.at(memory_t::host), response.gradient.at(memory_t::host),
                                  response.tau.at(memory_t::host)};
    external_xc_point_response(input, delta, output);
    xc_callback_output_t v{0, base.rho.at(memory_t::host), base.gradient.at(memory_t::host),
                           base.tau.at(memory_t::host)};
    std::string message;
    try {
        ctx_.cb().external_xc_(ctx_, input, v);
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    failed |= !std::isfinite(v.energy);
    for (auto const* f : {&base.rho, &base.tau}) {
        for (size_t i = 0; i < f->size(); i++) {
            failed |= !std::isfinite((*f)[i]);
        }
    }
    for (size_t i = 0; i < base.gradient.size(); i++) {
        failed |= !std::isfinite(base.gradient[i]);
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC callback failed during the sampling-transpose geometry response: " + message);
    }
    auto result   = map.adjoint(response);
    auto sampling = map.geometry_adjoint_response(base, point_directions, atom_directions__, lattice_direction__);
    for (int s = 0; s < ctx_.num_spins(); s++) {
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            result.rho[s].f_pw_local(ig) += sampling.rho[s].f_pw_local(ig);
            result.tau[s].f_pw_local(ig) += sampling.tau[s].f_pw_local(ig);
            for (auto z : {result.rho[s].f_pw_local(ig), result.tau[s].f_pw_local(ig)}) {
                failed |= !std::isfinite(z.real()) || !std::isfinite(z.imag());
            }
        }
    }
    for (size_t ia = 0; ia < result.local.size(); ia++) {
        for (int part = 0; part < 2; part++) {
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (size_t i = 0; i < result.local[ia][part][s].size(); i++) {
                    result.local[ia][part][s][i] += sampling.local[ia][part][s][i];
                    failed |= !std::isfinite(result.local[ia][part][s][i]);
                }
            }
            for (size_t i = 0; i < result.core[ia][part].size(); i++) {
                result.core[ia][part][i] += sampling.core[ia][part][i];
                failed |= !std::isfinite(result.core[ia][part][i]);
            }
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external LAPW XC geometry response returned nonfinite combined covectors");
    }
    return result;
}

std::array<double, 2>
Potential::check_external_xc_point_hessian(Density const& density__, double step__) const
{
    int invalid = !ctx_.full_potential() || !ctx_.cb().external_xc_points_ || !ctx_.cb().external_xc_geometry_ ||
                  !ctx_.cb().external_xc_geometry_response_ || &density__.ctx() != &ctx_ ||
                  !density__.has_kinetic_density() || !std::isfinite(step__) || step__ <= 0;
    double hmin = step__, hmax = step__;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    ctx_.comm().allreduce<double, mpi::op_t::min>(&hmin, 1);
    ctx_.comm().allreduce<double, mpi::op_t::max>(&hmax, 1);
    if (invalid || hmin != hmax) {
        RTE_THROW("point Hessian check requires matching LAPW fields, joint callbacks and a common positive step");
    }
    validate_external_xc_context(ctx_);
    auto const& q = *ctx_.cb().external_xc_points_;
    int nr = static_cast<int>(q.coordinates.size()), ns = ctx_.num_spins();
    std::vector<r3::vector<double>> points;
    for (auto const& r : q.coordinates) {
        points.emplace_back(r[0], r[1], r[2]);
    }
    LAPW_xc_field_map map(density__, points);
    auto smooth = external_smooth_fields(ctx_, density__);
    auto fields = map.sample(smooth[0], smooth[1]);
    xc_point_fields_t direction(nr, ns), response(nr, ns), plus(nr, ns), minus(nr, ns), shifted(nr, ns);
    xc_point_derivatives_t geometry(q), hgeometry(q), pgeometry(q), mgeometry(q);
    for (size_t i = 0; i < fields.rho.size(); i++) {
        direction.rho[i] = 0.07 * fields.rho[i];
        direction.tau[i] = -0.11 * fields.tau[i];
    }
    for (size_t i = 0; i < fields.gradient.size(); i++) {
        direction.gradient[i] = 0.03 * fields.gradient[i];
    }
    for (int i = 0; i < nr; i++) {
        geometry.energy_weights[i]     = 0.02 * q.energy_weights[i];
        geometry.descriptor_weights[i] = 0.03 * q.descriptor_weights[i] * std::cos(0.31 * i);
        for (int x = 0; x < 3; x++) {
            geometry.coordinates[i][x] = 0.04 * std::sin(0.23 * (i + x + 1));
        }
    }
    for (size_t i = 0; i < q.block_centres.size(); i++) {
        for (int x = 0; x < 3; x++) {
            geometry.block_centres[i][x] = 0.03 * std::cos(0.17 * (i + x + 1));
        }
    }
    response.rho.zero();
    response.gradient.zero();
    response.tau.zero();
    xc_callback_input_t input{
            nr, ns, 0, fields.rho.at(memory_t::host), fields.gradient.at(memory_t::host), fields.tau.at(memory_t::host),
            &q};
    xc_callback_direction_t d{direction.rho.at(memory_t::host), direction.gradient.at(memory_t::host),
                              direction.tau.at(memory_t::host), &geometry};
    xc_callback_response_t out{response.rho.at(memory_t::host), response.gradient.at(memory_t::host),
                               response.tau.at(memory_t::host), &hgeometry};
    external_xc_point_response(input, d, out);
    std::array<double, 2> result{};
    for (int k = 0; k < 2; k++) {
        double h      = step__ / (k + 1);
        auto evaluate = [&](double sign, xc_point_fields_t& f, xc_point_derivatives_t& g) {
            auto displaced = q;
            for (int i = 0; i < nr; i++) {
                displaced.energy_weights[i] += sign * h * geometry.energy_weights[i];
                displaced.descriptor_weights[i] += sign * h * geometry.descriptor_weights[i];
                for (int x = 0; x < 3; x++) {
                    displaced.coordinates[i][x] += sign * h * geometry.coordinates[i][x];
                }
            }
            for (size_t i = 0; i < q.block_centres.size(); i++) {
                for (int x = 0; x < 3; x++) {
                    displaced.block_centres[i][x] += sign * h * geometry.block_centres[i][x];
                }
            }
            for (size_t i = 0; i < fields.rho.size(); i++) {
                shifted.rho[i] = fields.rho[i] + sign * h * direction.rho[i];
                shifted.tau[i] = fields.tau[i] + sign * h * direction.tau[i];
            }
            for (size_t i = 0; i < fields.gradient.size(); i++) {
                shifted.gradient[i] = fields.gradient[i] + sign * h * direction.gradient[i];
            }
            f.rho.zero();
            f.gradient.zero();
            f.tau.zero();
            g = xc_point_derivatives_t(q);
            xc_callback_input_t base{nr,
                                     ns,
                                     0,
                                     shifted.rho.at(memory_t::host),
                                     shifted.gradient.at(memory_t::host),
                                     shifted.tau.at(memory_t::host),
                                     &displaced};
            xc_callback_output_t v{0, f.rho.at(memory_t::host), f.gradient.at(memory_t::host), f.tau.at(memory_t::host),
                                   &g};
            int failed{0};
            try {
                ctx_.cb().external_xc_geometry_(ctx_, base, v);
            } catch (std::exception const&) {
                failed = 1;
            }
            ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
            if (failed) {
                RTE_THROW("XC geometry callback failed during point Hessian check");
            }
        };
        evaluate(1, plus, pgeometry);
        evaluate(-1, minus, mgeometry);
        auto compare = [&](auto const& r, auto const& p, auto const& m) {
            double error{0}, peak{0};
            for (size_t i = 0; i < r.size(); i++) {
                auto e = std::abs(r[i] - (p[i] - m[i]) / (2 * h));
                error  = std::isfinite(e) ? std::max(error, e) : std::numeric_limits<double>::infinity();
                peak   = std::max(peak, std::abs(r[i]));
            }
            ctx_.comm().allreduce<double, mpi::op_t::max>(&error, 1);
            ctx_.comm().allreduce<double, mpi::op_t::max>(&peak, 1);
            result[k] = std::max(result[k], error / std::max(1e-12, peak));
        };
        compare(response.rho, plus.rho, minus.rho);
        compare(response.gradient, plus.gradient, minus.gradient);
        compare(response.tau, plus.tau, minus.tau);
        compare(hgeometry.energy_weights, pgeometry.energy_weights, mgeometry.energy_weights);
        compare(hgeometry.descriptor_weights, pgeometry.descriptor_weights, mgeometry.descriptor_weights);
        auto flatten = [](auto const& v) {
            std::vector<double> result;
            for (auto const& r : v) {
                result.insert(result.end(), r.begin(), r.end());
            }
            return result;
        };
        compare(flatten(hgeometry.coordinates), flatten(pgeometry.coordinates), flatten(mgeometry.coordinates));
        compare(flatten(hgeometry.block_centres), flatten(pgeometry.block_centres), flatten(mgeometry.block_centres));
    }
    return result;
}

xc_geometry_derivatives_t
Potential::external_xc_geometry(Density const& density__) const
{
    int failed = !ctx_.cb().external_xc_ || !ctx_.cb().external_xc_points_ || !ctx_.cb().external_xc_geometry_ ||
                 !ctx_.cb().external_xc_pullback_ || !density__.has_kinetic_density() || add_delta_rho_xc_ != 0 ||
                 add_delta_mag_xc_ != 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC geometry requires point, field-geometry and quadrature-pullback callbacks");
    }
    validate_external_xc_context(ctx_);
    auto const& quadrature = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : quadrature.coordinates) {
        points.emplace_back(r[0], r[1], r[2]);
    }
    auto smooth = external_smooth_fields(ctx_, density__);
    xc_point_derivatives_t geometry(quadrature);
    auto evaluate = [&](auto const& map) {
        auto fields = map.sample(smooth[0], smooth[1]);
        xc_point_fields_t derivatives(points.size(), ctx_.num_spins());
        derivatives.rho.zero();
        derivatives.gradient.zero();
        derivatives.tau.zero();
        xc_callback_input_t input{static_cast<int>(points.size()),
                                  ctx_.num_spins(),
                                  0,
                                  fields.rho.at(memory_t::host),
                                  fields.gradient.at(memory_t::host),
                                  fields.tau.at(memory_t::host),
                                  &quadrature};
        xc_callback_output_t output{0, derivatives.rho.at(memory_t::host), derivatives.gradient.at(memory_t::host),
                                    derivatives.tau.at(memory_t::host), &geometry};
        evaluate_external_xc(ctx_, input, output);
        return map.geometry_adjoint(smooth[0], smooth[1], derivatives);
    };
    auto mapped = ctx_.full_potential() ? evaluate(LAPW_xc_field_map(density__, points))
                                        : evaluate(PAW_xc_field_map(density__, points));
    for (size_t p = 0; p < points.size(); p++) {
        for (int x = 0; x < 3; x++) {
            geometry.coordinates[p][x] += mapped.points[p][x];
        }
    }
    xc_geometry_derivatives_t result(unit_cell_.num_atoms());
    try {
        ctx_.cb().external_xc_pullback_(ctx_, quadrature, geometry, result);
    } catch (std::exception const&) {
        failed = 1;
    }
    failed |= result.atoms.size() != static_cast<size_t>(unit_cell_.num_atoms());
    std::vector<double> values(result.lattice.begin(), result.lattice.end());
    for (auto const& atom : result.atoms) {
        values.insert(values.end(), atom.begin(), atom.end());
    }
    for (double value : values) {
        failed |= !std::isfinite(value);
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC quadrature pullback failed or returned invalid derivatives on a context rank");
    }
    auto minimum = values, maximum = values;
    ctx_.comm().allreduce<double, mpi::op_t::min>(minimum.data(), minimum.size());
    ctx_.comm().allreduce<double, mpi::op_t::max>(maximum.data(), maximum.size());
    for (size_t i = 0; i < values.size(); i++) {
        if (maximum[i] - minimum[i] > 1e-12 * std::max({1.0, std::abs(minimum[i]), std::abs(maximum[i])})) {
            RTE_THROW("external XC quadrature pullback must return globally summed derivatives on every rank");
        }
    }
    for (int x = 0; x < 3; x++) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            result.atoms[ia][x] += mapped.atoms(x, ia);
            failed |= !std::isfinite(result.atoms[ia][x]);
        }
        for (int y = 0; y < 3; y++) {
            result.lattice[x + 3 * y] += mapped.lattice(x, y);
            failed |= !std::isfinite(result.lattice[x + 3 * y]);
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC geometry returned nonfinite combined derivatives");
    }
    return result;
}

} // namespace sirius
