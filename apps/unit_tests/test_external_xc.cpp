/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <chrono>
#include <filesystem>
#include "dft/dft_ground_state.hpp"
#include "density/paw_xc_reconstruction.hpp"
#include "hamiltonian/hamiltonian.hpp"

using namespace sirius;

// A nonlinear primitive functional plus a nonlocal squared tau integral.
void
test_functional(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out)
{
    int nr = in.num_points;
    double moment{0};
    for (int j = 0; j < nr * in.num_spins; j++) {
        moment += (in.points ? in.points->descriptor_weights[j % nr] : in.point_weight) * in.tau[j];
    }
    mpi::Communicator comm(in.points ? ctx.comm().native() : ctx.spfft<double>().communicator());
    comm.allreduce(&moment, 1);
    double volume = ctx.unit_cell().omega();
    if (in.points) {
        volume = std::accumulate(in.points->descriptor_weights.begin(), in.points->descriptor_weights.end(), 0.0);
        comm.allreduce(&volume, 1);
    }
    double e{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int ir = 0; ir < nr; ir++) {
            int j    = ir + nr * s;
            double n = in.rho[j], t = in.tau[j];
            double w     = in.points ? in.points->energy_weights[ir] : in.point_weight;
            double d     = in.points ? in.points->descriptor_weights[ir] : in.point_weight;
            double scale = in.points ? 1 : 1 / in.point_weight;
            double local = 0.7 * n * n + 0.3 * t * t + 0.2 * n * t;
            out.vrho[j]  = scale * w * (1.4 * n + 0.2 * t);
            out.vtau[j]  = scale * (w * (0.6 * t + 0.2 * n) + d * 0.4 * moment / volume);
            for (int x = 0; x < 3; x++) {
                int k = ir + nr * (x + 3 * s);
                local += 0.1 * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = scale * w * 0.2 * in.gradient[k];
            }
            e += w * local;
            if (out.geometry) {
                out.geometry->energy_weights[ir] += local;
                out.geometry->descriptor_weights[ir] += 0.4 * moment * t / volume;
                if (s == 0) {
                    out.geometry->descriptor_weights[ir] -= 0.2 * moment * moment / (volume * volume);
                }
            }
        }
    }
    comm.allreduce(&e, 1);
    out.energy = e + 0.2 * moment * moment / volume;
}

auto
test_context(bool gamma, int mag, bool scf = false, bool exact = false, bool parallel_fft = false, bool external = true,
             double fixed_mag = 0, bool paw = false, bool points = false, std::string smearing = "gaussian",
             double extra_charge = 0, std::function<void(Simulation_context&)> setup = {})
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method": "pseudopotential", "use_symmetry": false,
        "pw_cutoff": 8.0, "gk_cutoff": 2.6, "num_bands": 4, "xc_functionals": [],
        "smearing_width": 0.01
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false},
    "iterative_solver": {"min_tolerance": 1e-11, "num_steps": 80}})"_json;
    conf["parameters"]["gamma_point"]  = gamma;
    conf["parameters"]["num_mag_dims"] = mag;
    conf["parameters"]["fixed_mag"]    = fixed_mag;
    conf["parameters"]["smearing"]     = smearing;
    conf["parameters"]["extra_charge"] = extra_charge;
    conf["settings"]["tol_ne"]         = 1e-11;
    conf["iterative_solver"]["type"]   = exact ? "exact" : "davidson";
    if (!external) {
        conf["parameters"]["xc_functionals"] = {"XC_LDA_X"};
    }
    if (parallel_fft) {
        conf["control"]["mpi_grid_dims"] = {mpi::Communicator::world().size(), 1};
    }
    auto ctx = std::make_unique<Simulation_context>(conf);
    ctx->unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    auto& type = ctx->unit_cell().add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::lin_exp, 400, paw ? 0.001 : 0.0, paw ? 3.0 : 20.0, 6);
    std::vector<double> v(400), rho(400), orbital(400);
    for (int ir = 0; ir < 400; ir++) {
        double r    = type.radial_grid(ir);
        v[ir]       = scf ? -2 / std::sqrt(1 + r * r) : 0;
        rho[ir]     = 4 * std::exp(-r * r) * r;
        orbital[ir] = std::exp(-r);
    }
    type.local_potential(v);
    type.ps_total_charge_density(rho);
    type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
    if (paw) {
        type.is_paw(true);
        std::vector<double> ae(400), ps(400), beta(400), q(400), core(400), core_tau(400), ps_core(400);
        for (int ir = 0; ir < 400; ir++) {
            double r     = type.radial_grid(ir);
            ae[ir]       = r * std::exp(-r * r);
            ps[ir]       = r * std::exp(-0.8 * r * r);
            beta[ir]     = 0.2 * r * std::exp(-2 * r * r);
            q[ir]        = 0.05 * r * r * std::exp(-2 * r * r);
            core[ir]     = 0.02 * std::exp(-3 * r * r);
            core_tau[ir] = 0.03 * std::exp(-3 * r * r);
            ps_core[ir]  = 0.01 * std::exp(-2 * r * r);
        }
        type.add_beta_radial_function(angular_momentum(0), beta);
        type.add_ae_paw_wf(ae);
        type.add_ps_paw_wf(ps);
        type.add_q_radial_function(0, 0, 0, q);
        type.paw_ae_core_charge_density(core);
        type.paw_ae_core_kinetic_density(core_tau);
        type.ps_core_charge_density(ps_core);
        type.paw_wf_occ({2});
        matrix<double> dion({1, 1});
        dion.zero();
        type.d_mtrx_ion(dion);
    } else {
        type.d_mtrx_ion(matrix<double>({0, 0}));
    }
    ctx->unit_cell().add_atom("He", {0.2, 0.3, 0.4});
    if (external) {
        ctx->cb().external_xc_ = test_functional;
        if (points) {
            // One complete block is deliberately owned by rank zero, not each
            // FFT replica. Other ranks exercise the empty-point collective path.
            auto quadrature = std::make_shared<xc_point_set_t>(*ctx);
            if (ctx->comm().rank() == 0) {
                for (int z = 0; z < 4; z++) {
                    for (int y = 0; y < 4; y++) {
                        for (int x = 0; x < 4; x++) {
                            quadrature->coordinates.push_back(
                                    {8 * (0.037 + x / 4.0), 9 * (0.029 + y / 4.0), 10 * (0.041 + z / 4.0)});
                            quadrature->energy_weights.push_back(720.0 / 64);
                            quadrature->descriptor_weights.push_back(720.0 / 64);
                        }
                    }
                }
                quadrature->energy_weights[0] = 0;
                if (paw) {
                    // Exercise inner continuation in the orbital Hamiltonian/energy derivative,
                    // including a descriptor-only point and a point with nonzero energy weight.
                    quadrature->coordinates[0] = {1.6002, 2.6997, 4.0001};
                    quadrature->coordinates[1] = {1.5998, 2.7001, 3.9997};
                }
                quadrature->block_sizes   = {64};
                quadrature->block_centres = {{1.6, 2.7, 4.0}};
            }
            ctx->cb().external_xc_points_ = std::move(quadrature);
        }
    }
    if (setup) {
        setup(*ctx);
    }
    ctx->initialize();
    return ctx;
}

int
test_point_geometry(Simulation_context& ctx, Density const& density, Potential& potential)
{
    auto original  = ctx.cb().external_xc_points_;
    auto evaluator = ctx.cb().external_xc_;
    auto lattice = ctx.unit_cell().lattice_vectors(), reciprocal = ctx.gvec().lattice_vectors();
    auto fractional = ctx.unit_cell().atom(0).position();
    auto atom       = r3::dot(lattice, fractional);
    std::array<double, 3> atom_scale{0.03, -0.02, 0.01};
    auto image = [](int p, int y) { return (p / (y == 0 ? 1 : y == 1 ? 3 : 9)) % 3 - 1; };
    // Independent dependence on fields, points, centres and both weight arrays.
    ctx.cb().external_xc_ = [](auto const& c, auto const& in, auto& out) {
        auto const& q = *in.points;
        int nr        = in.num_points;
        double moment{0}, energy{0};
        for (int j = 0; j < nr * in.num_spins; j++) {
            moment += q.descriptor_weights[j % nr] * in.rho[j];
        }
        c.comm().allreduce(&moment, 1);
        for (int s = 0; s < in.num_spins; s++) {
            for (int p = 0; p < nr; p++) {
                int j    = p + nr * s;
                double n = in.rho[j], t = in.tau[j], w = q.energy_weights[p];
                double local = 0.3 * n * n + 0.2 * t * t;
                out.vrho[j]  = w * 0.6 * n + q.descriptor_weights[p] * 0.04 * moment;
                out.vtau[j]  = w * 0.4 * t;
                for (int x = 0; x < 3; x++) {
                    int k = p + nr * (x + 3 * s);
                    local +=
                            0.1 * in.gradient[k] * in.gradient[k] + 0.004 * q.coordinates[p][x] * q.block_centres[0][x];
                    out.vgradient[k] = w * 0.2 * in.gradient[k];
                    if (out.geometry) {
                        out.geometry->coordinates[p][x] += w * 0.004 * q.block_centres[0][x];
                        out.geometry->block_centres[0][x] += w * 0.004 * q.coordinates[p][x];
                    }
                }
                energy += w * local;
                if (out.geometry) {
                    out.geometry->energy_weights[p] += local;
                    out.geometry->descriptor_weights[p] += 0.04 * moment * n;
                }
            }
        }
        c.comm().allreduce(&energy, 1);
        out.energy = energy + 0.02 * moment * moment;
    };
    ctx.cb().external_xc_geometry_ = ctx.cb().external_xc_;
    ctx.cb().external_xc_pullback_ = [&](auto const& c, auto const&, auto const& d, auto& out) {
        double weight{0};
        for (size_t p = 0; p < d.coordinates.size(); p++) {
            weight += original->energy_weights[p] * d.energy_weights[p] -
                      0.4 * original->descriptor_weights[p] * d.descriptor_weights[p];
            for (int x = 0; x < 3; x++) {
                out.atoms[0][x] += d.coordinates[p][x];
                for (int y = 0; y < 3; y++) {
                    out.lattice[x + 3 * y] += d.coordinates[p][x] * image(p, y);
                }
            }
        }
        for (int x = 0; x < 3; x++) {
            out.atoms[0][x] += weight * atom_scale[x];
            for (auto const& centre : d.block_centres) {
                out.atoms[0][x] += centre[x];
            }
        }
        for (int i = 0; i < 9; i++) {
            out.lattice[i] += weight * 0.002 * (i + 1);
        }
        c.comm().allreduce(out.atoms[0].data(), 3);
        c.comm().allreduce(out.lattice.data(), 9);
    };
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        if (ctx.comm().rank() == 0) {
            std::cout << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    auto rho_hash       = density.rho().rg().hash_f_pw();
    auto potential_hash = potential.effective_potential().rg().hash_f_pw();
    auto derivative     = potential.external_xc_geometry(density);
    check("geometry getter preserves density and potential",
          rho_hash == density.rho().rg().hash_f_pw() &&
                          potential_hash == potential.effective_potential().rg().hash_f_pw()
                  ? 0
                  : 1,
          0);
    auto displaced_energy = [&](int component, double step) {
        auto cell     = lattice;
        auto position = atom;
        if (component < 3) {
            position[component] += step;
        } else {
            cell((component - 3) % 3, (component - 3) / 3) += step;
        }
        ctx.unit_cell().set_lattice_vectors(cell);
        ctx.unit_cell().atom(0).set_position(r3::dot(r3::inverse(cell), position));
        ctx.gvec_sptr()->lattice_vectors(twopi * r3::transpose(r3::inverse(cell)));
        auto q                = std::make_shared<xc_point_set_t>(ctx);
        q->coordinates        = original->coordinates;
        q->block_sizes        = original->block_sizes;
        q->block_centres      = original->block_centres;
        q->energy_weights     = original->energy_weights;
        q->descriptor_weights = original->descriptor_weights;
        double scale          = step * (component < 3 ? atom_scale[component] : 0.002 * (component - 2));
        for (size_t p = 0; p < q->coordinates.size(); p++) {
            int x = component < 3 ? component : (component - 3) % 3;
            q->coordinates[p][x] += step * (component < 3 ? 1 : image(p, (component - 3) / 3));
            q->energy_weights[p] *= 1 + scale;
            q->descriptor_weights[p] *= 1 - 0.4 * scale;
        }
        if (component < 3 && !q->block_centres.empty()) {
            q->block_centres[0][component] += step;
        }
        ctx.cb().external_xc_points_ = q;
        potential.generate(density, false, true);
        return potential.energy_exc(density);
    };
    for (double step : {1e-4, 5e-5}) {
        double error{0};
        for (int i = 0; i < 12; i++) {
            double plus = displaced_energy(i, step), minus = displaced_energy(i, -step);
            double exact = i < 3 ? derivative.atoms[0][i] : derivative.lattice[i - 3];
            error = std::max(error, std::abs((plus - minus) / (2 * step) - exact) / std::max(1.0, std::abs(exact)));
        }
        check("combined point/weight/field geometry derivative", error, 2e-8);
    }
    ctx.unit_cell().set_lattice_vectors(lattice);
    ctx.unit_cell().atom(0).set_position(fractional);
    ctx.gvec_sptr()->lattice_vectors(reciprocal);
    ctx.cb().external_xc_points_ = original;
    auto reject                  = [&](char const* name) {
        bool rejected{false};
        try {
            potential.external_xc_geometry(density);
        } catch (std::exception const&) {
            rejected = true;
        }
        check(name, rejected ? 0 : 1, 0);
    };
    auto geometry_callback = ctx.cb().external_xc_geometry_;
    for (bool nonfinite : {false, true}) {
        ctx.cb().external_xc_geometry_ = [&](auto const& c, auto const& in, auto& out) {
            geometry_callback(c, in, out);
            if (c.comm().rank() == 0) {
                if (nonfinite) {
                    out.geometry->energy_weights[0] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    out.geometry->coordinates.pop_back();
                }
            }
        };
        reject("collective invalid geometry callback rejection");
    }
    ctx.cb().external_xc_geometry_ = geometry_callback;
    auto pullback                  = ctx.cb().external_xc_pullback_;
    ctx.cb().external_xc_pullback_ = [&](auto const& c, auto const& q, auto const& d, auto& out) {
        pullback(c, q, d, out);
        if (c.comm().rank() == 0) {
            out.atoms.pop_back();
        }
    };
    reject("collective invalid quadrature pullback rejection");
    if (ctx.comm().size() > 1) {
        ctx.cb().external_xc_pullback_ = [&](auto const& c, auto const& q, auto const& d, auto& out) {
            pullback(c, q, d, out);
            out.lattice[0] += c.comm().rank();
        };
        reject("collective unreduced quadrature pullback rejection");
    }
    ctx.cb().external_xc_geometry_ = nullptr;
    ctx.cb().external_xc_pullback_ = nullptr;
    ctx.cb().external_xc_          = evaluator;
    potential.generate(density, false, true);
    return failures;
}

int
test_external_xc(bool gamma, int mag, bool parallel_fft, bool points = false, bool paw = false)
{
    auto ctx = test_context(gamma, mag, false, false, parallel_fft, true, 0, paw, points);
    K_point_set ks(*ctx);
    if (gamma) {
        ks.add_kpoint({0, 0, 0}, 1);
    } else {
        ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
        ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    }
    ks.initialize();
    auto coefficient = [&](r3::vector<int> g, double angle, bool derivative) {
        std::complex<double> a{0}, b{0};
        if (g[1] == 0 && g[2] == 0 && (gamma ? std::abs(g[0]) == 2 : g[0] == 2)) {
            a = gamma ? std::sqrt(0.5) : 1;
        }
        if (g[0] == 0 && g[2] == 0 && (gamma ? std::abs(g[1]) == 1 : g[1] == 1)) {
            b = std::polar(gamma ? std::sqrt(0.5) : 1.0, gamma && g[1] < 0 ? -0.4 : 0.4);
        }
        return derivative ? -std::sin(angle) * a + std::cos(angle) * b : std::cos(angle) * a + std::sin(angle) * b;
    };
    auto set_orbitals = [&](double step) {
        for (auto it : ks.spl_num_kpoints()) {
            auto kp   = ks.get<double>(it.i);
            auto& psi = kp->spinor_wave_functions();
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int ib = 0; ib < ctx->num_bands(); ib++) {
                    kp->band_occupancy(ib, s, ib == 0 ? (mag ? (s == 0 ? 1.2 : 0.8) : 2.0) : 0);
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                        psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(ib)) =
                                ib == 0 ? coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i + step, false) : 0;
                    }
                }
            }
        }
        ks.sync_band<double, sync_band_t::occupancy>();
    };
    Density density(*ctx);
    Potential potential(*ctx);
    auto evaluate = [&](double step) {
        set_orbitals(step);
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        return density.kinetic_density().scalar().integrate().total + 0.5 * potential.energy_vha() +
               energy_vloc(density, potential) + potential.energy_exc(density) + potential.ewald_energy() +
               potential.PAW_total_energy(density);
    };
    std::vector<double> input_rho;
    if (paw) {
        ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
            input_rho.resize(in.num_points * in.num_spins);
            for (size_t i = 0; i < input_rho.size(); i++) {
                input_rho[i] = in.rho[i];
            }
            test_functional(c, in, out);
        };
    }
    double center          = evaluate(0);
    ctx->cb().external_xc_ = test_functional;
    Hamiltonian0<double> H0(potential, false);
    double derivative{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto Hk = H0(*kp);
        wf::Wave_functions<double> hphi(kp->gkvec_sptr(), wf::num_mag_dims(mag), wf::num_bands(1), memory_t::host);
        for (int s = 0; s < ctx->num_spins(); s++) {
            if (gamma) {
                Hk.apply_h_s<double>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(), &hphi,
                                     nullptr);
            } else {
                Hk.apply_h_s<std::complex<double>>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(),
                                                   &hphi, nullptr);
            }
            double expectation{0};
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto g        = kp->gkvec().gvec(gvec_index_t::local(ig));
                auto d        = coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i, true);
                auto h        = hphi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                double factor = gamma && dot(g, g) ? 2 : 1;
                derivative += 2 * kp->weight() * kp->band_occupancy(0, s) * factor * std::real(std::conj(d) * h);
                auto c = kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                expectation += factor * std::real(std::conj(c) * h);
            }
            kp->comm().allreduce(&expectation, 1);
            kp->band_energy(0, s, expectation);
        }
    }
    ctx->comm().allreduce(&derivative, 1);
    ks.sync_band<double, sync_band_t::energy>();
    int failures{0};
    auto check = [&](char const* name, double error, double tol) {
        if (ctx->comm().rank() == 0) {
            std::cout << "gamma=" << gamma << " mag=" << mag << " parallel_fft=" << parallel_fft << " points=" << points
                      << " paw=" << paw << ' ' << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tol;
    };
    {
        double saved_energy = potential.energy_vha();
        auto response       = potential.hartree_response(density.rho());
        double error{0};
        for (int ig = 0; ig < ctx->gvec().count(); ig++) {
            double g      = ctx->gvec().gvec_len(gvec_index_t::local(ig));
            auto expected = g == 0 ? std::complex<double>{0} : fourpi * density.rho().rg().f_pw_local(ig) / (g * g);
            error         = std::max(error, std::abs(response->rg().f_pw_local(ig) - expected));
            error         = std::max(
                    error, std::abs(response->rg().f_pw_local(ig) - potential.hartree_potential().rg().f_pw_local(ig)));
        }
        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(response->rg().value(ir) - potential.hartree_potential().rg().value(ir)));
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        check("smooth Hartree response and zero-G gauge", error, 2e-13);
        check("Hartree response preserves energy", std::abs(potential.energy_vha() - saved_energy), 0);
    }
    if (paw) {
        auto const& coordinates = ctx->cb().external_xc_points_->coordinates;
        int np                  = static_cast<int>(coordinates.size());
        mdarray<double, 2> rho({np, ctx->num_spins()}), tau({np, ctx->num_spins()});
        mdarray<double, 3> gradient({np, 3, ctx->num_spins()});
        rho.zero();
        tau.zero();
        gradient.zero();
        std::vector<r3::vector<double>> points;
        for (auto const& r : coordinates) {
            points.push_back({r[0], r[1], r[2]});
        }
        // Direct orbital density is independent of the host's Fourier Q augmentation.
        for (int s = 0; s < ctx->num_spins(); s++) {
            double occupation = mag ? (s == 0 ? 1.2 : 0.8) : 2;
            for (int p = 0; p < np; p++) {
                double ax = fourpi * points[p][0] / 8, by = twopi * points[p][1] / 9 + 0.4;
                auto a = gamma ? std::complex<double>(std::sqrt(2.0) * std::cos(ax)) : std::polar(1.0, ax);
                auto b = gamma ? std::complex<double>(std::sqrt(2.0) * std::cos(by)) : std::polar(1.0, by);
                for (int ik = 0; ik < (gamma ? 1 : 2); ik++) {
                    double weight = gamma ? 1 : (ik == 0 ? 0.3 : 0.7);
                    double angle  = 0.37 + 0.1 * s + 0.05 * ik;
                    rho(p, s) += occupation * weight / ctx->unit_cell().omega() *
                                 std::norm(std::cos(angle) * a + std::sin(angle) * b);
                }
            }
        }
        PAW_xc_reconstruction(density, points).add_to(rho, gradient, tau);
        double error{0};
        for (size_t i = 0; i < rho.size(); i++) {
            error = std::max(error, std::abs(rho[i] - input_rho[i]));
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        check("physical XC charge excludes compensation", error, 2e-12);
    }
    check("total-energy double counting", std::abs(ks_energy(*ctx, ks, density, potential) - center), 2e-10);
    double h  = 1e-4;
    double fd = (evaluate(h) - evaluate(-h)) / (2 * h);
    check("full orbital derivative", std::abs(fd - derivative), 2e-8);
    h *= 0.5;
    double fd2 = (evaluate(h) - evaluate(-h)) / (2 * h);
    check("halved-step orbital derivative", std::abs(fd2 - derivative), 6e-9);

    evaluate(0);
    DFT_ground_state diagnostic(ks);
    copy(density, diagnostic.density());
    diagnostic.potential().generate(diagnostic.density(), false, true);
    double original_exc = diagnostic.potential().energy_exc(diagnostic.density());
    auto rho_hash       = diagnostic.density().rho().rg().hash_f_pw();
    auto tau_hash       = diagnostic.density().kinetic_density().scalar().rg().hash_f_pw();
    auto potential_hash = diagnostic.potential().effective_potential().rg().hash_f_pw();
    std::vector<double> band_metadata;
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < ctx->num_spins(); s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                band_metadata.push_back(kp->band_occupancy(b, s));
                band_metadata.push_back(kp->band_energy(b, s));
            }
        }
    }
    auto verify_restored = [&]() {
        double delta{0};
        size_t ib{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    delta = std::max(delta, std::abs(kp->band_occupancy(b, s) - band_metadata[ib++]));
                    delta = std::max(delta, std::abs(kp->band_energy(b, s) - band_metadata[ib++]));
                }
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                    auto c = kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                    delta  = std::max(delta, std::abs(c - coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i, false)));
                }
            }
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&delta, 1);
        check("diagnostic restores orbitals and band data", delta, 0);
        check("diagnostic retains XC energy",
              std::abs(diagnostic.potential().energy_exc(diagnostic.density()) - original_exc), 0);
        check("diagnostic retains density and potential",
              rho_hash == diagnostic.density().rho().rg().hash_f_pw() &&
                              tau_hash == diagnostic.density().kinetic_density().scalar().rg().hash_f_pw() &&
                              potential_hash == diagnostic.potential().effective_potential().rg().hash_f_pw()
                      ? 0
                      : 1,
              0);
    };
    for (double bad_step : {0.0, -0.1, std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected{false};
        try {
            diagnostic.check_external_xc_derivative(bad_step);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("diagnostic rejects invalid step", rejected ? 0 : 1, 0);
    }
    if (ctx->comm().size() > 1) {
        bool rejected{false};
        try {
            diagnostic.check_external_xc_derivative(ctx->comm().rank() == 0 ? 0.001 : 0.002);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("diagnostic rejects inconsistent step", rejected ? 0 : 1, 0);
    }
    auto derivatives = diagnostic.check_external_xc_derivative(1e-4);
    for (auto const& values : derivatives) {
        check("XC diagnostic derivative", std::abs(values[0] - values[1]), 2e-8);
        check("XC diagnostic halved step", std::abs(values[0] - values[2]), 6e-9);
    }
    verify_restored();
    int evaluations{0};
    ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
        test_functional(c, in, out);
        if (++evaluations == 2 && c.comm().rank() == 0) {
            throw std::runtime_error("deliberate derivative-check callback failure");
        }
    };
    bool check_failed{false};
    try {
        diagnostic.check_external_xc_derivative(1e-4);
    } catch (std::exception const&) {
        check_failed = true;
    }
    check("diagnostic propagates callback failure", check_failed ? 0 : 1, 0);
    verify_restored();
    diagnostic.potential().generate(diagnostic.density(), false, true);
    check("diagnostic restores evaluator", evaluations == 3 ? 0 : 1, 0);
    ctx->cb().external_xc_ = test_functional;
    if (points && !paw && !gamma && mag == 1) {
        failures += test_point_geometry(*ctx, density, potential);
    }
    if (paw) {
        ctx->cb().external_xc_ = [](auto const& c, auto const& in, auto& out) {
            test_functional(c, in, out);
            for (int j = 0; j < in.num_points * in.num_spins; j++) {
                out.vtau[j] = 0;
            }
        };
        auto incomplete = diagnostic.check_external_xc_derivative(1e-4);
        double missing_tau{0};
        for (auto const& values : incomplete) {
            missing_tau = std::max(missing_tau, std::abs(values[0] - values[2]));
        }
        check("diagnostic detects missing tau", missing_tau > 1e-7 ? 0 : 1, 0);
        ctx->cb().external_xc_ = test_functional;
        verify_restored();
    }
    Density previous(*ctx);
    copy(density, previous);
    density.mixer_init(ctx->cfg().mixer());
    evaluate(0.15);
    std::vector<double> expected;
    for (int j = 0; j < mag + 1; j++) {
        auto const& now = density.kinetic_density().component(j).rg();
        auto const& old = previous.kinetic_density().component(j).rg();
        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
            expected.push_back(0.4 * now.value(ir) + 0.6 * old.value(ir));
        }
    }
    density.mix();
    double error{0};
    size_t idx{0};
    for (int j = 0; j < mag + 1; j++) {
        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) - expected[idx++]));
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    check("tau mixer and copy", error, 1e-12);

    unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    ctx->comm().bcast(&stamp, 1, 0);
    auto path = std::filesystem::temp_directory_path() / ("sirius-xc-restart-" + std::to_string(stamp) + ".h5");
    ctx->create_storage_file(path.string());
    density.save(path.string());
    ctx->comm().barrier();
    previous.load(path.string());
    error = 0;
    for (int j = 0; j < mag + 1; j++) {
        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) -
                                             previous.kinetic_density().component(j).rg().value(ir)));
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    check("tau restart", error, 1e-12);
    ctx->comm().barrier();
    if (ctx->comm().rank() == 0) {
        std::filesystem::remove(path);
    }

    if (points) {
        auto original = ctx->cb().external_xc_points_;
        auto invalid  = std::make_shared<xc_point_set_t>(*original);
        if (ctx->comm().rank() == 0) {
            invalid->energy_weights.pop_back();
        }
        ctx->cb().external_xc_points_ = invalid;
        bool rejected{false};
        try {
            potential.generate(density, false, true);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("collective invalid-quadrature rejection", rejected ? 0 : 1, 0);
        ctx->cb().external_xc_points_ = original;
        if (ctx->comm().size() > 1) {
            if (ctx->comm().rank() == 0) {
                ctx->cb().external_xc_points_.reset();
            }
            rejected = false;
            try {
                potential.generate(density, false, true);
            } catch (std::exception const&) {
                rejected = true;
            }
            check("collective quadrature-mode mismatch rejection", rejected ? 0 : 1, 0);
            ctx->cb().external_xc_points_ = original;
        }
        auto const lattice = ctx->unit_cell().lattice_vectors();
        ctx->unit_cell().set_lattice_vectors({{8.01, 0, 0}, {0, 9, 0}, {0, 0, 10}});
        rejected = false;
        try {
            original->validate(*ctx);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("stale quadrature after cell change rejection", rejected ? 0 : 1, 0);
        ctx->unit_cell().set_lattice_vectors(lattice);
    }

    ctx->cb().external_xc_ = [](auto const& c, auto const&, auto& out) {
        if (c.comm().rank() == 0) {
            out.energy = std::numeric_limits<double>::quiet_NaN();
        }
    };
    bool rejected{false};
    try {
        potential.generate(density, false, true);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("collective nonfinite rejection", rejected ? 0 : 1, 0);
    return failures;
}

int
test_external_scf(bool exact, bool external, bool paw = false)
{
    bool points = external && (exact || paw);
    bool gamma  = paw && !exact;
    int mag     = paw && !exact ? 1 : 0;
    auto ctx    = test_context(gamma, mag, true, exact, false, external, 0, paw, points);
    if (mag) {
        ctx->unit_cell().atom(0).set_vector_field({0, 0, 0.5});
    }
    bool initial_fields_valid{true};
    int calls{0};
    if (points) {
        ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
            if (calls++ == 0 && !paw) {
                for (int s = 0; s < in.num_spins; s++) {
                    for (int i = 0; i < in.num_points; i++) {
                        double n = in.rho[i + in.num_points * s], t = in.tau[i + in.num_points * s];
                        double sigma{0};
                        for (int x = 0; x < 3; x++) {
                            sigma += std::pow(in.gradient[i + in.num_points * (x + 3 * s)], 2);
                        }
                        initial_fields_valid &= std::isfinite(n) && std::isfinite(t) && std::isfinite(sigma) &&
                                                n >= -1e-12 && t >= -1e-12 && sigma <= 8 * n * t + 1e-12;
                    }
                }
            }
            test_functional(c, in, out);
        };
    }
    K_point_set ks(*ctx);
    ks.add_kpoint(gamma ? r3::vector<double>{0, 0, 0} : r3::vector<double>{0.07, 0.03, 0.02}, 1);
    ks.initialize();
    std::vector<double> initial_spin;
    if (mag) {
        Density guess(*ctx);
        guess.initial_density();
        auto const& values = guess.mag(0).rg().values();
        initial_spin.assign(values.begin(), values.end());
    }
    DFT_ground_state gs(ks);
    gs.initial_state();
    int failures{0};
    if (points) {
        if (mag) {
            // A closed shell can keep zero net spin while its local spin densities differ.
            double overlap{0};
            for (size_t ir = 0; ir < initial_spin.size(); ir++) {
                overlap += initial_spin[ir] * gs.density().mag(0).rg().value(ir);
            }
            ctx->comm_fft().allreduce(&overlap, 1);
            overlap *= ctx->unit_cell().omega() / ctx->spfft<double>().global_size();
            failures += !std::isfinite(overlap) || overlap < 1e-8;
            if (ctx->comm().rank() == 0) {
                std::cout << "Initial orbital spin-pattern overlap=" << overlap << '\n';
            }
        }
        Density initial(*ctx);
        copy(gs.density(), initial);
        double charge{0};
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    charge += kp->weight() * kp->band_occupancy(b, s);
                }
            }
        }
        ks.comm().allreduce(&charge, 1);
        gs.density().generate<double>(ks, false, true, true);
        double difference{0};
        auto compare = [&](auto const& a, auto const& b) {
            for (size_t i = 0; i < a.size(); i++) {
                difference = std::max(difference, std::abs(a[i] - b[i]));
            }
        };
        for (int s = 0; s < ctx->num_spins(); s++) {
            compare(initial.component(s).rg().values(), gs.density().component(s).rg().values());
            compare(initial.kinetic_density().component(s).rg().values(),
                    gs.density().kinetic_density().component(s).rg().values());
        }
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            compare(initial.density_matrix(ia), gs.density().density_matrix(ia));
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&difference, 1);
        double charge_error = std::abs(charge - ctx->unit_cell().num_valence_electrons());
        if (ctx->comm().rank() == 0) {
            std::cout << "Initial orbital fields PAW=" << paw << " charge error=" << charge_error
                      << " field difference=" << difference << '\n';
        }
        failures += charge_error > 1e-10 || difference > 1e-10 || !initial_fields_valid || calls != 1;
    }
    auto result     = gs.find(1e-8, 1e-9, 1e-8, 100, false);
    bool converged  = result.at("converged").get<bool>();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    density.generate<double>(ks, false, true, true);
    potential.generate(density, false, true);
    double direct = gs.energy_kin_sum_pw() + 0.5 * potential.energy_vha() + energy_vloc(density, potential) +
                    potential.energy_exc(density) + potential.ewald_energy() + potential.PAW_total_energy(density);
    double error  = std::abs(ks_energy(*ctx, ks, density, potential) - direct);
    if (ctx->comm().rank() == 0) {
        std::cout << (external ? "external XC " : "LibXC ") << (exact ? "exact" : "Davidson") << " PAW=" << paw
                  << " SCF converged=" << converged << " direct-energy difference=" << error << '\n';
    }
    return failures + (!converged || !std::isfinite(error) || error > 1e-7);
}

int
test_external_geometry(bool paw, int mag, bool strain, bool gamma)
{
    auto calculate = [&](double displacement, bool get_force) {
        std::array<double, 3> direction{0.7, -0.4, 0.2};
        r3::matrix<double> deformation{{0.3, 0.1, -0.07}, {0.1, -0.2, 0.06}, {-0.07, 0.06, 0.15}};
        auto setup = [&](Simulation_context& ctx) {
            if (gamma) {
                ctx.cfg().mixer().beta(0.2);
            }
            std::array<double, 3> lengths{8, 9, 10};
            if (strain) {
                auto lattice = ctx.unit_cell().lattice_vectors();
                lattice += dot(deformation, lattice) * displacement;
                ctx.unit_cell().set_lattice_vectors(lattice);
            } else {
                ctx.unit_cell().atom(0).set_position({0.2 + displacement * direction[0] / lengths[0],
                                                      0.3 + displacement * direction[1] / lengths[1],
                                                      0.4 + displacement * direction[2] / lengths[2]});
            }
            ctx.unit_cell().add_atom("He", {0.48, 0.51, 0.59});
            auto const& lattice = ctx.unit_cell().lattice_vectors();
            auto q              = std::make_shared<xc_point_set_t>(ctx);
            // Atom-following blocks sample the full cell. Unoccupied point ranks
            // and k-point ranks must still participate in the geometry pullback.
            if (ctx.comm().rank() == 0) {
                for (int ia = 0; ia < 2; ia++) {
                    auto r      = ctx.unit_cell().atom(ia).position();
                    auto centre = dot(lattice, r);
                    q->block_centres.push_back({centre[0], centre[1], centre[2]});
                    q->block_sizes.push_back(64);
                    for (int z = 0; z < 4; z++) {
                        for (int y = 0; y < 4; y++) {
                            for (int x = 0; x < 4; x++) {
                                auto p = dot(lattice,
                                             r + r3::vector<double>(0.037 + x / 4.0, 0.029 + y / 4.0, 0.041 + z / 4.0));
                                q->coordinates.push_back({p[0], p[1], p[2]});
                                q->energy_weights.push_back(lattice.det() / 128);
                                q->descriptor_weights.push_back(lattice.det() / 128);
                            }
                        }
                    }
                }
            }
            ctx.cb().external_xc_points_   = q;
            ctx.cb().external_xc_geometry_ = test_functional;
            ctx.cb().external_xc_pullback_ = [](auto const& c, auto const& points, auto const& d, auto& result) {
                int offset{0};
                auto inverse = r3::inverse(c.unit_cell().lattice_vectors());
                for (size_t ia = 0; ia < points.block_sizes.size(); ia++) {
                    for (int x = 0; x < 3; x++) {
                        result.atoms[ia][x] += d.block_centres[ia][x];
                        for (int i = 0; i < points.block_sizes[ia]; i++) {
                            int p = offset + i;
                            result.atoms[ia][x] += d.coordinates[p][x];
                            r3::vector<double> relative;
                            for (int j = 0; j < 3; j++) {
                                relative[j] = points.coordinates[p][j] - points.block_centres[ia][j];
                            }
                            auto fractional = dot(inverse, relative);
                            for (int j = 0; j < 3; j++) {
                                result.lattice[x + 3 * j] += d.coordinates[p][x] * fractional[j] +
                                                             (d.energy_weights[p] * points.energy_weights[p] +
                                                              d.descriptor_weights[p] * points.descriptor_weights[p]) *
                                                                     inverse(j, x);
                            }
                        }
                    }
                    offset += points.block_sizes[ia];
                }
                for (auto& r : result.atoms) {
                    c.comm().allreduce(r.data(), 3);
                }
                c.comm().allreduce(result.lattice.data(), 9);
            };
        };
        auto ctx = test_context(gamma, mag, true, !gamma, strain && !gamma, true, mag ? 0.6 : 0, paw, false, "gaussian",
                                0, setup);
        K_point_set ks(*ctx);
        if (gamma) {
            ks.add_kpoint({0, 0, 0}, 1);
        } else {
            ks.add_kpoint({0.07, 0.03, 0.02}, 0.4);
            ks.add_kpoint({-0.06, 0.02, -0.04}, 0.6);
        }
        ks.initialize();
        DFT_ground_state gs(ks);
        gs.initial_state();
        auto result = gs.find(1e-11, 1e-12, 1e-11, 150, false);
        if (!result.at("converged").get<bool>()) {
            RTE_THROW("geometry finite-difference SCF did not converge");
        }
        double derivative{0};
        if (get_force) {
            auto e = gs.total_energy();
            if (strain) {
                auto stress = gs.stress().calc_stress_total();
                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        derivative += ctx->unit_cell().omega() * stress(mu, nu) * deformation(mu, nu);
                    }
                }
            } else {
                auto const& f = gs.forces().calc_forces_total();
                for (int x = 0; x < 3; x++) {
                    derivative -= f(x, 0) * direction[x];
                    if (!std::isfinite(f(x, 0)) || std::abs(f(x, 0) + f(x, 1)) > 2e-7) {
                        RTE_THROW("external XC forces violate translation invariance");
                    }
                }
            }
            if (e != gs.total_energy()) {
                RTE_THROW("force calculation changed the ground-state energy");
            }
            ctx->cb().external_xc_geometry_ = nullptr;
            bool rejected{false};
            try {
                if (strain) {
                    gs.stress().calc_stress_total();
                } else {
                    gs.forces().calc_forces_total();
                }
            } catch (std::exception const&) {
                rejected = true;
            }
            if (!rejected) {
                RTE_THROW("geometry derivative accepted an incomplete external XC registration");
            }
        }
        // With fractional occupations the variational force differentiates E-TS.
        return std::array<double, 2>{free_energy(*ctx, ks, gs.density(), gs.potential()), derivative};
    };
    auto centre = calculate(0, true);
    int failures{0};
    for (double h : {2e-4, 1e-4}) {
        double fd    = (calculate(h, false)[0] - calculate(-h, false)[0]) / (2 * h);
        double error = std::abs(fd - centre[1]);
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "External XC " << (strain ? "strain" : "force") << " PAW=" << paw << " spin=" << mag
                      << " gamma=" << gamma << " h=" << h << " derivative=" << centre[1] << " FD=" << fd
                      << " error=" << error << '\n';
        }
        failures += !std::isfinite(error) || error > 2e-6;
    }
    return failures;
}

int
test_fixed_spin_occupancy(double moment)
{
    auto ctx = test_context(false, 1, false, false, false, false, moment);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
    ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    ks.initialize();
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < 2; s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                // The empty spin channel can have the lowest eigenvalue.
                double offset = ((moment > 0) == (s == 1)) ? -0.3 : 0.1;
                kp->band_energy(b, s, offset + 0.2 * b + 0.03 * it.i);
            }
        }
    }
    ks.sync_band<double, sync_band_t::energy>();
    ks.find_band_occupancies<double>();
    double occupation[2]{0, 0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < 2; s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                occupation[s] += kp->weight() * kp->band_occupancy(b, s);
            }
        }
    }
    ks.comm().allreduce(occupation, 2);
    double error = std::max(std::abs(occupation[0] - (2 + moment) / 2), std::abs(occupation[1] - (2 - moment) / 2));
    std::cout << "Fixed spin moment " << moment << ", occupation error: " << error << '\n';
    return !std::isfinite(error) || error > 2e-11;
}

int
test_generic_occupancy(int mag, std::string smearing, double electrons)
{
    auto ctx = test_context(false, mag, false, false, false, false, 0, false, false, smearing, 2 - electrons);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
    ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    ks.initialize();
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < ctx->num_spinors(); s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                // Closely spaced bands put the chemical potential outside the spectrum.
                kp->band_energy(b, s, -0.3 + 0.001 * (b + s + it.i));
            }
        }
    }
    ks.sync_band<double, sync_band_t::energy>();
    ks.find_band_occupancies<double>();
    double occupation{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < ctx->num_spinors(); s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                occupation += kp->weight() * kp->band_occupancy(b, s);
            }
        }
    }
    ks.comm().allreduce(&occupation, 1);
    double error = std::abs(occupation - electrons);
    std::cout << "Generic occupations " << smearing << ", mag=" << mag << ", electrons=" << electrons
              << ", error=" << error << '\n';
    return !std::isfinite(error) || error > 2e-11;
}

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result{0};
    if (argc == 2 && std::string(argv[1]) == "forces") {
        result += call_test("external PW forces", test_external_geometry, false, 0, false, false);
        result += call_test("external PAW forces", test_external_geometry, true, 1, false, false);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && std::string(argv[1]) == "stress") {
        result += call_test("external PW stress", test_external_geometry, false, 0, true, true);
        result += call_test("external PAW stress", test_external_geometry, true, 1, true, false);
        sirius::finalize();
        return std::min(result, 1);
    }
    for (int mag : {0, 1}) {
        for (auto smearing : {"gaussian", "fermi_dirac", "cold", "methfessel_paxton"}) {
            for (double electrons : {1.0, 7.0}) {
                result += call_test("generic occupations", test_generic_occupancy, mag, smearing, electrons);
            }
        }
    }
    for (bool gamma : {false, true}) {
        for (int mag : {0, 1}) {
            result += call_test("external XC energy and operator", test_external_xc, gamma, mag, false, false, false);
            result += call_test("external XC point quadrature", test_external_xc, gamma, mag, false, true, false);
            result += call_test("external PAW XC energy and operator", test_external_xc, gamma, mag, false, true, true);
            if (mpi::Communicator::world().size() > 1) {
                result += call_test("external XC distributed FFT", test_external_xc, gamma, mag, true, false, false);
                if (!gamma && mag == 1) {
                    result += call_test("point geometry with distributed FFT", test_external_xc, gamma, mag, true, true,
                                        false);
                }
                result += call_test("external PAW XC distributed FFT", test_external_xc, gamma, mag, true, true, true);
            }
        }
    }
    for (bool external : {false, true}) {
        for (bool exact : {false, true}) {
            result += call_test("SCF energy accounting", test_external_scf, exact, external, false);
        }
    }
    for (bool exact : {false, true}) {
        result += call_test("PAW SCF energy accounting", test_external_scf, exact, true, true);
    }
    result += call_test("external PW forces", test_external_geometry, false, 0, false, false);
    result += call_test("external PAW forces", test_external_geometry, true, 1, false, false);
    result += call_test("external PW stress", test_external_geometry, false, 0, true, true);
    result += call_test("external PAW stress", test_external_geometry, true, 1, true, false);
    for (double moment : {2.0, -2.0, 1.2}) {
        result += call_test("fixed-spin occupations", test_fixed_spin_occupancy, moment);
    }
    sirius::finalize();
    return std::min(result, 1);
}
