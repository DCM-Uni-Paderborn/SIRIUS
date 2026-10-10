/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "density/lapw_xc_field_map.hpp"
#include "density/paw_xc_field_map.hpp"
#include "symmetry/xc_point_symmetry.hpp"
#include <cstdio>

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

xc_point_set_t::xc_point_set_t(xc_point_set_t const& other)
    : coordinates(other.coordinates)
    , energy_weights(other.energy_weights)
    , descriptor_weights(other.descriptor_weights)
    , block_sizes(other.block_sizes)
    , block_centres(other.block_centres)
    , lattice_(other.lattice_)
    , atoms_(other.atoms_)
{
}

xc_point_set_t&
xc_point_set_t::operator=(xc_point_set_t const& other)
{
    if (this != &other) {
        coordinates        = other.coordinates;
        energy_weights     = other.energy_weights;
        descriptor_weights = other.descriptor_weights;
        block_sizes        = other.block_sizes;
        block_centres      = other.block_centres;
        lattice_           = other.lattice_;
        atoms_             = other.atoms_;
        symmetry_.reset();
    }
    return *this;
}

XC_point_symmetry const&
xc_point_set_t::symmetry(Simulation_context const& ctx__) const
{
    if (!symmetry_) {
        symmetry_ = std::make_shared<XC_point_symmetry>(ctx__, *this);
    }
    return *symmetry_;
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

std::array<std::vector<Smooth_periodic_function<double>>, 2>
external_xc_smooth_fields(Simulation_context const& ctx, Density const& density)
{
    mdarray<std::complex<double>, 2> augmentation;
    if (ctx.unit_cell().augment()) {
        // Cancel Q in its original Fourier representation, not on a radial grid.
        augmentation = density.generate_rho_aug();
    }
    std::array<std::vector<Smooth_periodic_function<double>>, 2> fields;
    auto& rho = fields[0];
    auto& tau = fields[1];
    for (int s = 0; s < ctx.num_mag_dims() + 1; s++) {
        rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            auto r = density.rho().rg().f_pw_local(ig);
            auto t = density.kinetic_density().scalar().rg().f_pw_local(ig);
            if (!ctx.full_potential() && ctx.gvec().offset() + ig == 0) {
                // The neutralizing PP background is not an electronic XC field.
                r -= ctx.cfg().parameters().extra_charge() / ctx.unit_cell().omega();
            }
            if (ctx.unit_cell().augment()) {
                r -= augmentation(ig, 0);
            }
            if (ctx.num_mag_dims() == 3 && s > 0) {
                r = density.mag(s - 1).rg().f_pw_local(ig);
                t = density.kinetic_density().vector(s - 1).rg().f_pw_local(ig);
                if (ctx.unit_cell().augment()) {
                    r -= augmentation(ig, s);
                }
            } else if (ctx.num_mag_dims() == 1) {
                auto m = density.mag(0).rg().f_pw_local(ig);
                if (ctx.unit_cell().augment()) {
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

namespace {

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
    int spinor_min = ctx__.cb().external_xc_spinor_, spinor_max = spinor_min;
    ctx__.comm().allreduce<int, mpi::op_t::min>(&spinor_min, 1);
    ctx__.comm().allreduce<int, mpi::op_t::max>(&spinor_max, 1);
    if (spinor_min != spinor_max) {
        RTE_THROW("external XC field convention differs between context ranks");
    }
    if (spinor_min) {
        if (ctx__.num_mag_dims() != 3 || !points) {
            RTE_THROW("external spinor XC requires reconstructed total/z/x/y fields");
        }
    } else if (ctx__.num_mag_dims() == 3 || ctx__.so_correction()) {
        RTE_THROW("external XC spinors require explicit four-component callback registration");
    }
    if (!ctx__.xc_functionals().empty() || ctx__.cb().veff_) {
        RTE_THROW("external XC cannot be combined with built-in XC or an effective-potential callback");
    }
    if (ctx__.cfg().parameters().veff_pw_cutoff() > 0) {
        RTE_THROW("external XC does not support an additional potential-only Fourier filter");
    }
    if (points) {
        points->validate(ctx__);
        if (ctx__.cfg().parameters().use_symmetry()) {
            points->symmetry(ctx__);
        }
    }
    if (ctx__.full_potential()) {
        bool dirac = ctx__.valence_relativity() == relativity_t::dirac;
        if (dirac && (ctx__.cfg().control().use_second_variation() || ctx__.so_correction())) {
            RTE_THROW("external Dirac LAPW XC requires full variation without a separate SOC correction");
        }
        if (!dirac && spinor_min && !ctx__.cfg().control().use_second_variation()) {
            RTE_THROW("external LAPW spinor XC requires second variation");
        }
        if (ctx__.cfg().parameters().precision_wf() != "fp64" ||
            (!dirac && spinor_min && ctx__.valence_relativity() != relativity_t::none &&
             ctx__.valence_relativity() != relativity_t::zora)) {
            RTE_THROW("external LAPW XC requires scalar/collinear, NR/ZORA spinor or Dirac fp64 valence orbitals");
        }
        if (ctx__.unit_cell().num_core_electrons() != 0 && ctx__.core_relativity() != relativity_t::none &&
            ctx__.core_relativity() != relativity_t::zora && ctx__.core_relativity() != relativity_t::dirac) {
            RTE_THROW("external LAPW XC: tau-dependent radial core response requires nonrelativistic, ZORA or "
                      "Dirac core states");
        }
        return;
    }
    int failed{0};
    std::string message;
    try {
        for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
            auto const& type = ctx__.unit_cell().atom_type(iat);
            if (type.augment() && !type.is_paw()) {
                type.check_kinetic_augmentation();
                if (!points) {
                    RTE_THROW("external XC: ultrasoft augmentation requires reconstructed point quadrature");
                }
            }
            if (!type.is_paw()) {
                type.check_ps_core_kinetic_density();
                if (type.has_ps_core_kinetic_density() && !points) {
                    RTE_THROW("external XC: nonlinear core correction requires reconstructed point quadrature");
                }
            }
            if (type.is_paw() && (!points || !type.has_paw_core_kinetic_density())) {
                RTE_THROW("external XC: PAW requires a point quadrature and explicitly supplied positive core tau");
            }
        }
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC: invalid atomic setup on a context rank: " + message);
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
            double n = (1 + add_delta_rho_xc_) *
                       (density__.rho().rg().value(ir) - ctx_.cfg().parameters().extra_charge() / unit_cell_.omega());
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

double
check_external_xc_point_derivative(Density const& density__)
{
    auto const& ctx = density__.ctx();
    validate_external_xc_context(ctx);
    if (!ctx.cb().external_xc_points_ || !density__.has_kinetic_density()) {
        RTE_THROW("point XC derivative check requires reconstructed density and tau");
    }
    auto const& q = *ctx.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : q.coordinates) {
        points.emplace_back(r[0], r[1], r[2]);
    }
    auto smooth = external_xc_smooth_fields(ctx, density__);
    auto fields = ctx.full_potential() ? LAPW_xc_field_map(density__, points).sample(smooth[0], smooth[1])
                                       : PAW_xc_field_map(density__, points).sample(smooth[0], smooth[1]);
    if (ctx.use_symmetry()) {
        fields = q.symmetry(ctx).apply(fields);
    }
    int nr = static_cast<int>(points.size()), ns = ctx.num_mag_dims() + 1;
    xc_point_fields_t shifted(nr, ns), derivatives(nr, ns);
    auto evaluate = [&](double scale) {
        for (size_t i = 0; i < fields.rho.size(); i++) {
            shifted.rho[i] = scale * fields.rho[i];
            shifted.tau[i] = scale * fields.tau[i];
        }
        for (size_t i = 0; i < fields.gradient.size(); i++) {
            shifted.gradient[i] = scale * fields.gradient[i];
        }
        derivatives.rho.zero();
        derivatives.gradient.zero();
        derivatives.tau.zero();
        xc_callback_input_t in{nr,
                               ns,
                               0,
                               shifted.rho.at(memory_t::host),
                               shifted.gradient.at(memory_t::host),
                               shifted.tau.at(memory_t::host),
                               &q};
        xc_callback_output_t out{0, derivatives.rho.at(memory_t::host), derivatives.gradient.at(memory_t::host),
                                 derivatives.tau.at(memory_t::host)};
        evaluate_external_xc(ctx, in, out);
        return out.energy;
    };
    evaluate(1);
    // Covectors already contain quadrature weights; contract every primitive field once.
    double analytic{0};
    for (size_t i = 0; i < fields.rho.size(); i++) {
        analytic += derivatives.rho[i] * fields.rho[i] + derivatives.tau[i] * fields.tau[i];
    }
    for (size_t i = 0; i < fields.gradient.size(); i++) {
        analytic += derivatives.gradient[i] * fields.gradient[i];
    }
    ctx.comm().allreduce(&analytic, 1);
    double error{0};
    for (double h : {1e-3, 5e-4, 1e-5, 5e-6}) {
        double plus = evaluate(1 + h), minus = evaluate(1 - h);
        double finite_difference = (plus - minus) / (2 * h);
        error                    = std::abs(analytic - finite_difference) /
                std::max({1.0, std::abs(analytic), std::abs(finite_difference)});
        if (ctx.comm().rank() == 0) {
            std::printf("Point XC rho/gradient/tau derivative: %.10e, FD: %.10e, h: %.3e, relative error: %.6e\n",
                        analytic, finite_difference, h, error);
        }
    }
    return error;
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
    int ns      = ctx_.num_mag_dims() + 1;
    auto smooth = external_xc_smooth_fields(ctx_, density__);
    auto& rho   = smooth[0];
    auto& tau   = smooth[1];
    xc_point_fields_t derivatives(nr, ns);
    auto evaluate = [&](auto const& map) {
        auto fields = map.sample(rho, tau);
        if (ctx_.use_symmetry()) {
            fields = quadrature.symmetry(ctx_).apply(fields);
        }
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
        if (ctx_.use_symmetry()) {
            derivatives = quadrature.symmetry(ctx_).apply(derivatives, true);
        }
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
    external_projector_d_mtrx_ = std::move(adjoint.local_matrices);
    for (int s = 0; s < ns; s++) {
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            if (ns == 4) {
                auto& v          = s == 0 ? xc_potential_->rg() : effective_magnetic_field(s - 1).rg();
                v.f_pw_local(ig) = adjoint.rho[s].f_pw_local(ig);
                kinetic_potential_->component(s).rg().f_pw_local(ig) = adjoint.tau[s].f_pw_local(ig);
                continue;
            }
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
    for (int s = 0; s < ctx_.num_mag_dims(); s++) {
        effective_magnetic_field(s).rg().fft_transform(1);
    }
}

void
Potential::external_xc_point_response(xc_callback_input_t const& input__, xc_callback_direction_t const& direction__,
                                      xc_callback_response_t& output__) const
{
    int failed = !ctx_.cb().external_xc_response_ || !input__.points || input__.num_points < 0 ||
                 input__.num_spins != ctx_.num_mag_dims() + 1 ||
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
    size_t n = size_t(std::max(0, input__.num_points)) * (ctx_.num_mag_dims() + 1);
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
    PROFILE("sirius::Potential::lapw_external_xc_response");
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
    auto smooth           = external_xc_smooth_fields(ctx_, density__);
    auto smooth_direction = external_xc_smooth_fields(ctx_, density_response__);
    auto fields           = map.sample(smooth[0], smooth[1]);
    auto direction        = direction_map.sample(smooth_direction[0], smooth_direction[1]);
    if (ctx_.use_symmetry()) {
        fields    = quadrature.symmetry(ctx_).apply(fields);
        direction = quadrature.symmetry(ctx_).apply(direction);
    }
    int nf                = ctx_.num_mag_dims() + 1;
    xc_point_fields_t response(points.size(), nf);
    response.rho.zero();
    response.gradient.zero();
    response.tau.zero();
    xc_callback_input_t input{static_cast<int>(points.size()),
                              nf,
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
    if (ctx_.use_symmetry()) {
        response = quadrature.symmetry(ctx_).apply(response, true);
    }
    return map.adjoint(response);
}

xc_point_derivatives_t
Potential::external_xc_quadrature_direction(xc_geometry_direction_t const& direction__) const
{
    int failed = !ctx_.cb().external_xc_points_ || !ctx_.cb().external_xc_quadrature_direction_ ||
                 direction__.atoms.size() != static_cast<size_t>(ctx_.unit_cell().num_atoms());
    std::vector<double> lo(direction__.lattice.begin(), direction__.lattice.end());
    for (auto const& r : direction__.atoms) {
        lo.insert(lo.end(), r.begin(), r.end());
    }
    for (auto x : lo) {
        failed |= !std::isfinite(x);
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("quadrature direction requires a host callback and finite matching atom/cell directions");
    }
    auto hi = lo;
    ctx_.comm().allreduce<double, mpi::op_t::min>(lo.data(), lo.size());
    ctx_.comm().allreduce<double, mpi::op_t::max>(hi.data(), hi.size());
    if (lo != hi) {
        RTE_THROW("quadrature atom/cell directions differ between context ranks");
    }
    auto const& q = *ctx_.cb().external_xc_points_;
    q.validate(ctx_);
    xc_point_derivatives_t result(q);
    std::string message;
    try {
        ctx_.cb().external_xc_quadrature_direction_(ctx_, q, direction__, result);
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    failed |= result.coordinates.size() != q.coordinates.size() ||
              result.block_centres.size() != q.block_centres.size() ||
              result.energy_weights.size() != q.energy_weights.size() ||
              result.descriptor_weights.size() != q.descriptor_weights.size();
    for (auto const* a : {&result.coordinates, &result.block_centres}) {
        for (auto const& r : *a) {
            for (auto x : r) {
                failed |= !std::isfinite(x);
            }
        }
    }
    for (auto const* a : {&result.energy_weights, &result.descriptor_weights}) {
        for (auto x : *a) {
            failed |= !std::isfinite(x);
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("host quadrature direction failed or returned invalid output: " + message);
    }
    return result;
}

lapw_xc_field_adjoint_t
Potential::lapw_external_xc_geometry_response(Density const& density__,
                                              xc_geometry_direction_t const& direction__) const
{
    auto quadrature = external_xc_quadrature_direction(direction__);
    mdarray<double, 2> atoms({3, ctx_.unit_cell().num_atoms()});
    r3::matrix<double> lattice;
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        for (int x = 0; x < 3; x++) {
            atoms(x, ia) = direction__.atoms[ia][x];
        }
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            lattice(x, y) = direction__.lattice[x + 3 * y];
        }
    }
    return lapw_external_xc_geometry_response(density__, quadrature, atoms, lattice);
}

lapw_xc_field_adjoint_t
Potential::lapw_external_xc_geometry_response(Density const& density__,
                                              xc_point_derivatives_t const& quadrature_direction__,
                                              mdarray<double, 2> const& atom_directions__,
                                              r3::matrix<double> const& lattice_direction__) const
{
    PROFILE("sirius::Potential::lapw_external_xc_geometry_response");
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
    auto smooth = external_xc_smooth_fields(ctx_, density__);
    auto raw_fields = map.sample(smooth[0], smooth[1]);
    auto fields     = ctx_.use_symmetry() ? q.symmetry(ctx_).apply(raw_fields) : std::move(raw_fields);
    auto direction =
            map.geometry_response(smooth[0], smooth[1], point_directions, atom_directions__, lattice_direction__);
    if (ctx_.use_symmetry()) {
        direction     = q.symmetry(ctx_).apply(direction);
        auto rotation = q.symmetry(ctx_).apply(raw_fields, false, &lattice_direction__);
        for (size_t i = 0; i < direction.gradient.size(); i++) {
            direction.gradient[i] += rotation.gradient[i];
        }
    }
    int nf = ctx_.num_mag_dims() + 1;
    xc_point_fields_t response(points.size(), nf), base(points.size(), nf);
    response.rho.zero();
    response.gradient.zero();
    response.tau.zero();
    base.rho.zero();
    base.gradient.zero();
    base.tau.zero();
    xc_callback_input_t input{
            static_cast<int>(points.size()), nf, 0, fields.rho.at(memory_t::host), fields.gradient.at(memory_t::host),
            fields.tau.at(memory_t::host),   &q};
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
    if (ctx_.use_symmetry()) {
        response      = q.symmetry(ctx_).apply(response, true);
        auto rotation = q.symmetry(ctx_).apply(base, true, &lattice_direction__);
        for (size_t i = 0; i < response.gradient.size(); i++) {
            response.gradient[i] += rotation.gradient[i];
        }
        base = q.symmetry(ctx_).apply(base, true);
    }
    auto result   = map.adjoint(response);
    auto sampling = map.geometry_adjoint_response(base, point_directions, atom_directions__, lattice_direction__);
    for (int s = 0; s < nf; s++) {
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
            for (int s = 0; s < nf; s++) {
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
    auto smooth = external_xc_smooth_fields(ctx_, density__);
    xc_point_derivatives_t geometry(quadrature);
    r3::matrix<double> symmetry_lattice;
    auto evaluate = [&](auto const& map) {
        auto raw_fields = map.sample(smooth[0], smooth[1]);
        auto fields     = ctx_.use_symmetry() ? quadrature.symmetry(ctx_).apply(raw_fields) : std::move(raw_fields);
        xc_point_fields_t derivatives(points.size(), ctx_.num_mag_dims() + 1);
        derivatives.rho.zero();
        derivatives.gradient.zero();
        derivatives.tau.zero();
        xc_callback_input_t input{static_cast<int>(points.size()),
                                  ctx_.num_mag_dims() + 1,
                                  0,
                                  fields.rho.at(memory_t::host),
                                  fields.gradient.at(memory_t::host),
                                  fields.tau.at(memory_t::host),
                                  &quadrature};
        xc_callback_output_t output{0, derivatives.rho.at(memory_t::host), derivatives.gradient.at(memory_t::host),
                                    derivatives.tau.at(memory_t::host), &geometry};
        evaluate_external_xc(ctx_, input, output);
        if (ctx_.use_symmetry()) {
            symmetry_lattice = quadrature.symmetry(ctx_).geometry_adjoint(raw_fields, derivatives);
            derivatives      = quadrature.symmetry(ctx_).apply(derivatives, true);
        }
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
            result.lattice[x + 3 * y] += mapped.lattice(x, y) + symmetry_lattice(x, y);
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
