/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <chrono>
#include <filesystem>
#include "dft/dft_ground_state.hpp"
#include "dft/energy.hpp"
#include "density/paw_xc_reconstruction.hpp"
#include "function3d/smooth_periodic_function_sampler.hpp"
#include "geometry/force.hpp"
#include "geometry/stress.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "hamiltonian/initialize_subspace.hpp"

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
             double extra_charge = 0, std::function<void(Simulation_context&)> setup = {}, bool gpu = false,
             bool soc = false)
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
    conf["control"]["processing_unit"] = gpu ? "GPU" : "CPU";
    conf["control"]["verification"]    = scf && exact ? 2 : 0;
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
    type.spin_orbit_coupling(soc);
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
            ae[ir]       = std::pow(r, soc ? 2 : 1) * std::exp(-r * r);
            ps[ir]       = std::pow(r, soc ? 2 : 1) * std::exp(-0.8 * r * r);
            beta[ir]     = 0.2 * std::pow(r, soc ? 2 : 1) * std::exp(-2 * r * r);
            q[ir]        = 0.05 * r * r * std::exp(-2 * r * r);
            core[ir]     = 0.02 * std::exp(-3 * r * r);
            core_tau[ir] = 0.03 * std::exp(-3 * r * r);
            ps_core[ir]  = 0.01 * std::exp(-2 * r * r);
        }
        int nbeta = soc ? 2 : 1;
        for (int i = 0; i < nbeta; i++) {
            type.add_beta_radial_function(soc ? angular_momentum(1, i == 0 ? -1 : 1) : angular_momentum(0), beta);
            type.add_ae_paw_wf(ae);
            type.add_ps_paw_wf(ps);
        }
        for (int j = 0; j < nbeta; j++) {
            for (int i = 0; i <= j; i++) {
                for (int l = 0; l <= (soc ? 2 : 0); l += 2) {
                    auto radial = q;
                    for (auto& v : radial) {
                        v *= (1 + 0.4 * i + 0.2 * j) / (l + 1);
                    }
                    type.add_q_radial_function(i, j, l, radial);
                }
            }
        }
        type.paw_ae_core_charge_density(core);
        type.paw_ae_core_kinetic_density(core_tau);
        type.ps_core_charge_density(ps_core);
        type.paw_wf_occ(std::vector<double>(nbeta, 2.0 / nbeta));
        matrix<double> dion({nbeta, nbeta});
        dion.zero();
        if (soc) {
            dion(0, 0) = 0.7;
            dion(1, 1) = -0.4;
        }
        type.d_mtrx_ion(dion);
    } else if (soc) {
        std::vector<double> beta(400);
        for (int ir = 0; ir < 400; ir++) {
            double r = type.radial_grid(ir);
            beta[ir] = r * r * std::exp(-r * r);
        }
        type.add_beta_radial_function(angular_momentum(1, -1), beta);
        type.add_beta_radial_function(angular_momentum(1, 1), beta);
        matrix<double> dion({2, 2});
        dion.zero();
        dion(0, 0) = 0.7;
        dion(1, 1) = -0.4;
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
    Density displaced_density(ctx);
    copy(density, displaced_density);
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
        // Hold the physical smooth field fixed, not its volume-dependent background.
        if (ctx.gvec().offset() == 0) {
            displaced_density.rho().rg().f_pw_local(0) =
                    density.rho().rg().f_pw_local(0) +
                    ctx.cfg().parameters().extra_charge() * (1 / cell.det() - 1 / lattice.det());
        }
        displaced_density.rho().rg().fft_transform(1);
        potential.generate(displaced_density, false, true);
        return potential.energy_exc(displaced_density);
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

void
set_test_core(Simulation_context& ctx)
{
    auto& type = ctx.unit_cell().atom_type(0);
    std::vector<double> charge(type.num_mt_points()), kinetic(charge.size());
    for (int ir = 0; ir < type.num_mt_points(); ir++) {
        double r    = type.radial_grid(ir);
        charge[ir]  = 0.02 * std::exp(-3 * r * r);
        kinetic[ir] = 0.03 * std::exp(-3 * r * r);
    }
    type.ps_core_charge_density(charge);
    type.ps_core_kinetic_density(kinetic);
}

void
check_spinor_eigensystem(Hamiltonian_k<double> const& hk, K_point<double>& kp, Test_check& check)
{
    auto& ctx = hk.H0().ctx();
    int ng = kp.num_gkvec(), n = 2 * ng, nb = ctx.num_bands();
    wf::Wave_functions<double> basis(kp.gkvec_sptr(), wf::num_mag_dims(3), wf::num_bands(n), memory_t::host);
    wf::Wave_functions<double> hb(kp.gkvec_sptr(), wf::num_mag_dims(3), wf::num_bands(n), memory_t::host);
    wf::Wave_functions<double> sb(kp.gkvec_sptr(), wf::num_mag_dims(3), wf::num_bands(n), memory_t::host);
    for (int s = 0; s < 2; s++) {
        basis.zero(memory_t::host, wf::spin_index(s), wf::band_range(0, n));
        for (int ig = 0; ig < kp.gkvec().count(); ig++) {
            basis.pw_coeffs(ig, wf::spin_index(s), wf::band_index(s * ng + kp.gkvec().offset() + ig)) = 1;
        }
    }
    {
        auto bg = basis.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::device);
        auto hg = hb.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::host);
        auto sg = sb.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::host);
        hk.apply_h_s<std::complex<double>>(wf::spin_range(0, 2), wf::band_range(0, n), basis, &hb, &sb);
    }
    // Assemble the complete spinor problem, independently of the Davidson subspace and preconditioner.
    la::dmatrix<std::complex<double>> h(n, n), overlap(n, n), eigenvectors(n, n);
    h.zero();
    overlap.zero();
    for (int b = 0; b < n; b++) {
        for (int s = 0; s < 2; s++) {
            for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                int row         = s * ng + kp.gkvec().offset() + ig;
                h(row, b)       = hb.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b));
                overlap(row, b) = sb.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b));
            }
        }
    }
    kp.gkvec().comm().allreduce(h.at(memory_t::host), n * n);
    kp.gkvec().comm().allreduce(overlap.at(memory_t::host), n * n);
    check("full spinor Hamiltonian Hermiticity", check_hermitian(h, n), 2e-12);
    check("full spinor overlap Hermiticity", check_hermitian(overlap, n), 2e-12);
    std::vector<double> exact(nb);
    auto solver = la::Eigensolver_factory("lapack");
    if (solver->solve(n, nb, h, overlap, exact.data(), eigenvectors)) {
        RTE_THROW("dense spinor eigenproblem failed");
    }
    initialize_subspace<double, std::complex<double>>(hk, kp, ctx.unit_cell().num_ps_atomic_wf().first);
    auto const& itso = ctx.cfg().iterative_solver();
    auto result      = davidson<double, std::complex<double>, davidson_evp_t::hamiltonian>(
            hk, kp, wf::num_bands(nb), wf::num_mag_dims(3), kp.spinor_wave_functions(), [](int, int) { return 1e-12; },
            1e-8, 240, itso.locking(), itso.subspace_size(), false, itso.extra_ortho(), ctx.out(), 0);
    check("spinor Davidson convergence", result.converged ? 0 : 1, 0);
    double energy_error{0};
    for (int b = 0; b < nb; b++) {
        energy_error = std::max(energy_error, std::abs(result.eval(b, 0) - exact[b]));
    }
    check("spinor Davidson versus dense eigenvalues", energy_error, 2e-10);
    {
        auto pg = kp.spinor_wave_functions().memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::device);
        auto hg = hb.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::host);
        auto sg = sb.memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::host);
        hk.apply_h_s<std::complex<double>>(wf::spin_range(0, 2), wf::band_range(0, nb), kp.spinor_wave_functions(), &hb,
                                           &sb);
    }
    std::vector<double> residual(nb);
    la::dmatrix<std::complex<double>> gram(nb, nb);
    wf::inner(ctx.spla_context(), memory_t::host, wf::spin_range(0, 2), kp.spinor_wave_functions(),
              wf::band_range(0, nb), sb, wf::band_range(0, nb), gram, 0, 0);
    for (int b = 0; b < nb; b++) {
        for (int s = 0; s < 2; s++) {
            for (int ig = 0; ig < kp.gkvec().count(); ig++) {
                residual[b] += std::norm(hb.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)) -
                                         result.eval(b, 0) * sb.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)));
            }
        }
    }
    kp.gkvec().comm().allreduce(residual.data(), nb);
    check("spinor eigenpair residual", std::sqrt(*std::max_element(residual.begin(), residual.end())), 2e-8);
    check("spinor S-orthonormality", check_identity(gram, nb), 2e-12);
    std::vector<double> iterative(nb);
    for (int b = 0; b < nb; b++) {
        iterative[b] = result.eval(b, 0);
    }
    check_wave_functions<double, std::complex<double>>(hk, kp.spinor_wave_functions(), wf::spin_range(0, 2),
                                                       wf::band_range(0, nb), iterative.data());
}

int
test_spinor_xc(bool gpu, bool paw, bool soc, double charge)
{
    auto ctx = test_context(
            false, 3, false, false, soc, true, 0, paw, true, "gaussian", charge,
            [=](auto& c) {
                c.cb().external_xc_spinor_ = true;
                if (!paw) {
                    set_test_core(c);
                }
                auto q = std::make_shared<xc_point_set_t>(*c.cb().external_xc_points_);
                if (!q->coordinates.empty()) {
                    q->coordinates[1] = {1.6 + c.unit_cell().atom_type(0).radial_grid(100), 2.7, 4.0};
                }
                c.cb().external_xc_points_ = std::move(q);
            },
            gpu, soc);
    int failures{0};
    Test_check check(failures, ctx->comm(),
                     "spinor XC PAW=" + std::to_string(paw) + " SOC=" + std::to_string(soc) +
                             " GPU=" + std::to_string(gpu) + " charge=" + std::to_string(charge));
    K_point_set ks(*ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
    ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    ks.initialize();
    auto coefficient = [](r3::vector<int> g, double angle, int s, bool derivative) {
        auto a     = g == r3::vector<int>{2, 0, 0} ? std::complex<double>{1, 0} : std::complex<double>{0, 0};
        auto b     = g == r3::vector<int>{0, 1, 0} ? std::polar(1.0, 0.4 + 0.7 * s) : std::complex<double>{0, 0};
        auto scale = std::polar(std::sqrt(s == 0 ? 0.6 : 0.4), 0.9 * s);
        return scale *
               (derivative ? -std::sin(angle) * a + std::cos(angle) * b : std::cos(angle) * a + std::sin(angle) * b);
    };
    Density density(*ctx);
    Potential potential(*ctx);
    double core_error{0};
    bool audit_core{true};
    ctx->cb().external_xc_ = [&](Simulation_context const& c, auto const& in, auto& out) {
        int nr = in.num_points;
        if (in.num_spins != 4) {
            RTE_THROW("spinor callback requires four components");
        }
        std::array<double, 4> moment{};
        double volume{0}, energy{0};
        for (int p = 0; p < nr; p++) {
            volume += in.points->descriptor_weights[p];
            for (int s = 0; s < 4; s++) {
                moment[s] += in.points->descriptor_weights[p] * in.tau[p + nr * s];
            }
        }
        c.comm().allreduce(moment.data(), 4);
        c.comm().allreduce(&volume, 1);
        double square = std::inner_product(moment.begin(), moment.end(), moment.begin(), 0.0);
        for (int s = 0; s < 4; s++) {
            for (int p = 0; p < nr; p++) {
                int j    = p + nr * s;
                double n = in.rho[j], t = in.tau[j], w = in.points->energy_weights[p];
                double local = 0.7 * n * n + 0.3 * t * t + 0.2 * n * t;
                out.vrho[j]  = w * (1.4 * n + 0.2 * t);
                out.vtau[j]  = w * (0.6 * t + 0.2 * n) + in.points->descriptor_weights[p] * 0.4 * moment[s] / volume;
                for (int x = 0; x < 3; x++) {
                    int k = p + nr * (x + 3 * s);
                    local += 0.1 * in.gradient[k] * in.gradient[k];
                    out.vgradient[k] = w * 0.2 * in.gradient[k];
                }
                energy += w * local;
                if (out.geometry) {
                    out.geometry->energy_weights[p] += local;
                    out.geometry->descriptor_weights[p] += 0.4 * moment[s] * t / volume;
                    if (s == 0) {
                        out.geometry->descriptor_weights[p] -= 0.2 * square / (volume * volume);
                    }
                }
            }
        }
        c.comm().allreduce(&energy, 1);
        out.energy = energy + 0.2 * square / volume;
        if (!audit_core) {
            return;
        }
        std::vector<r3::vector<double>> point;
        if (nr > 1) {
            point.push_back({in.points->coordinates[1][0], in.points->coordinates[1][1], in.points->coordinates[1][2]});
        }
        Smooth_periodic_function_sampler sampler(c.gvec(), point);
        auto const& type = c.unit_cell().atom_type(0);
        mdarray<std::complex<double>, 2> augmentation;
        std::array<std::vector<Flm>, 2> local;
        if (paw) {
            augmentation = density.generate_rho_aug();
            local        = density.generate_paw_xc_correction(0);
        }
        for (int s = 0; s < 4; s++) {
            auto const& rho  = s == 0 ? density.rho() : density.mag(s - 1);
            auto const& tau  = s == 0 ? density.kinetic_density().scalar() : density.kinetic_density().vector(s - 1);
            auto sampled_rho = sampler.sample(rho.rg()), sampled_tau = sampler.sample(tau.rg());
            std::vector<std::array<double, 4>> sampled_q;
            if (paw) {
                Smooth_periodic_function<double> q(c.spfft<double>(), c.gvec_fft_sptr());
                for (int ig = 0; ig < c.gvec().count(); ig++) {
                    q.f_pw_local(ig) = augmentation(ig, s);
                }
                sampled_q = sampler.sample(q);
            }
            if (nr > 1) {
                double r = sampled_rho[0][0], t = sampled_tau[0][0];
                if (s == 0) {
                    r -= charge / c.unit_cell().omega();
                }
                if (paw) {
                    auto position = point[0] - dot(c.unit_cell().lattice_vectors(), c.unit_cell().atom(0).position());
                    r += Spheric_function_interpolator(local[0][s], spheric_inner_extension_t::charge)
                                 .value_gradient(position)[0] -
                         sampled_q[0][0];
                    t += Spheric_function_interpolator(local[1][s], spheric_inner_extension_t::kinetic)
                                 .value_gradient(position)[0];
                }
                core_error = std::max(core_error, std::abs(in.rho[1 + nr * s] - r -
                                                           (!paw && s == 0 ? type.ps_core_charge_density()[100] : 0)));
                core_error = std::max(core_error, std::abs(in.tau[1 + nr * s] - t -
                                                           (!paw && s == 0 ? type.ps_core_kinetic_density()[100] : 0)));
            }
        }
    };
    auto ionic_energy = [&] {
        double energy{0};
        if (soc) {
            auto const& type = ctx->unit_cell().atom_type(0);
            auto const& dm   = density.density_matrix(0);
            // The projector density matrix already includes the SOC angular rotation and k weights.
            for (int xi = 0; xi < type.mt_basis_size(); xi++) {
                int rf = type.indexb(xi).idxrf;
                energy += type.d_mtrx_ion()(rf, rf) * std::real(dm(xi, xi, 0) + dm(xi, xi, 1));
            }
        }
        return energy;
    };
    bool overlap_constraint{false};
    auto evaluate = [&](double step) {
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int ib = 0; ib < ctx->num_bands(); ib++) {
                kp->band_occupancy(ib, 0, ib < 2 ? 1 - charge / 2 : 0);
                for (int s = 0; s < 2; s++) {
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                        kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(ib)) =
                                ib == 0 ? coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i + step, s, false)
                                        : (ib == 1 && r3::dot(g, g) == 0
                                                   ? std::polar(std::sqrt(s == 0 ? 0.3 : 0.7), -0.6 * s)
                                                   : std::complex<double>{0, 0});
                    }
                }
            }
        }
        ks.sync_band<double, sync_band_t::occupancy>();
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        return density.kinetic_density().scalar().integrate().total + 0.5 * potential.energy_vha() +
               energy_vloc(density, potential) + potential.energy_exc(density) + potential.ewald_energy() +
               potential.PAW_total_energy(density) + ionic_energy() -
               (overlap_constraint ? 0.37 * density.rho().integrate().total : 0);
    };
    double center = evaluate(0), derivative{0};
    audit_core    = false;
    check("joint point XC field derivative", check_external_xc_point_derivative(density), 2e-10);
    auto callback          = ctx->cb().external_xc_;
    ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
        callback(c, in, out);
        for (int i = 0; i < in.num_points * in.num_spins; i++) {
            out.vtau[i] += 1;
        }
    };
    check("point XC derivative detects incorrect tau covectors",
          check_external_xc_point_derivative(density) > 1e-6 ? 0 : 1, 0);
    ctx->cb().external_xc_ = callback;
    audit_core             = true;
    if (soc) {
        check("nonzero SOC projector energy", std::abs(ionic_energy()) > 1e-8 ? 0 : 1, 0);
        D_operator<double> d(potential);
        double mixing{0};
        auto const& type = ctx->unit_cell().atom_type(0);
        for (int i = 0; i < type.mt_basis_size(); i++) {
            for (int j = 0; j < type.mt_basis_size(); j++) {
                mixing = std::max(mixing, std::abs(d.value<std::complex<double>>(i, j, 2, 0)));
            }
        }
        check("nonzero SOC spin mixing", mixing > 1e-3 ? 0 : 1, 0);
        if (paw) {
            Q_operator<double> q(*ctx);
            double mixing{0};
            for (int i = 0; i < type.mt_basis_size(); i++) {
                for (int j = 0; j < type.mt_basis_size(); j++) {
                    mixing = std::max(mixing, std::abs(q.value<std::complex<double>>(i, j, 2, 0)));
                }
            }
            check("nonzero SOC overlap spin mixing", mixing > 1e-10 ? 0 : 1, 0);
        }
    }
    Hamiltonian0<double> H0(potential, false);
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto Hk = H0(*kp);
        wf::Wave_functions<double> hphi(kp->gkvec_sptr(), wf::num_mag_dims(3), wf::num_bands(2), memory_t::host);
        {
            auto psi_guard =
                    kp->spinor_wave_functions().memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::device);
            auto hphi_guard = hphi.memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::host);
            Hk.apply_h_s<std::complex<double>>(wf::spin_range(0, 2), wf::band_range(0, 2), kp->spinor_wave_functions(),
                                               &hphi, nullptr);
        }
        std::array<double, 2> expectation{};
        for (int s = 0; s < 2; s++) {
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto d = coefficient(kp->gkvec().gvec(gvec_index_t::local(ig)), 0.37 + 0.1 * s + 0.05 * it.i, s, true);
                auto h = hphi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                auto psi = kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                derivative += 2 * kp->weight() * kp->band_occupancy(0, 0) * std::real(std::conj(d) * h);
                expectation[0] += std::real(std::conj(psi) * h);
                expectation[1] += std::real(
                        std::conj(kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(1))) *
                        hphi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(1)));
            }
        }
        kp->comm().allreduce(expectation.data(), 2);
        for (int ib = 0; ib < 2; ib++) {
            kp->band_energy(ib, 0, expectation[ib]);
        }
    }
    ctx->comm().allreduce(&derivative, 1);
    ks.sync_band<double, sync_band_t::energy>();
    check("spinor total-energy double counting", std::abs(ks_energy(*ctx, ks, density, potential) - center), 2e-10);
    for (double h : {1e-4, 5e-5}) {
        check("spinor orbital derivative", std::abs((evaluate(h) - evaluate(-h)) / (2 * h) - derivative), 2e-8);
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&core_error, 1);
    check("reconstructed spinor fields exclude Q and retain the scalar core", core_error, 2e-12);
    ctx->cb().external_xc_spinor_ = false;
    check.reject("legacy callback rejects spinor fields", [&] { validate_external_xc_context(*ctx); });
    ctx->cb().external_xc_spinor_ = true;
    if (soc && paw) {
        auto& type = ctx->unit_cell().atom_type(0);
        type.is_paw(false);
        check.reject("external ultrasoft SOC requires PAW reconstruction", [&] { validate_external_xc_context(*ctx); });
        type.is_paw(true);
    }
    Stress stress(*ctx, density, potential, ks);
    check.reject("spinor stress requires geometry callbacks", [&] { stress.calc_stress_xc(); });
    audit_core                      = false;
    ctx->cb().external_xc_geometry_ = ctx->cb().external_xc_;
    ctx->cb().external_xc_pullback_ = [](auto const& c, auto const& q, auto const& d, auto& out) {
        auto inverse = r3::inverse(c.unit_cell().lattice_vectors());
        for (size_t p = 0; p < q.coordinates.size(); p++) {
            auto fractional =
                    dot(inverse, r3::vector<double>(q.coordinates[p][0], q.coordinates[p][1], q.coordinates[p][2]));
            double weights =
                    d.energy_weights[p] * q.energy_weights[p] + d.descriptor_weights[p] * q.descriptor_weights[p];
            for (int x = 0; x < 3; x++) {
                for (int j = 0; j < 3; j++) {
                    out.lattice[x + 3 * j] += d.coordinates[p][x] * fractional[j] + weights * inverse(j, x);
                }
            }
        }
        for (auto const& centre : d.block_centres) {
            for (int x = 0; x < 3; x++) {
                out.atoms[0][x] += centre[x];
            }
        }
        c.comm().allreduce(out.atoms[0].data(), 3);
        c.comm().allreduce(out.lattice.data(), 9);
    };
    // Geometry stencils keep coefficients fixed. A common, nonzero SOC multiplier
    // tests the full spinor overlap response via the independent charge integral.
    overlap_constraint = paw && soc;
    if (paw) {
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int ib = 0; ib < ctx->num_bands(); ib++) {
                kp->band_energy(ib, 0, overlap_constraint && ib < 2 ? 0.37 : 0);
            }
        }
        ks.sync_band<double, sync_band_t::energy>();
    }
    center        = evaluate(0);
    auto exact    = stress.calc_stress_xc();
    auto total    = stress.calc_stress_total();
    auto lattice  = ctx->unit_cell().lattice_vectors();
    auto original = ctx->cb().external_xc_points_;
    double volume = ctx->unit_cell().omega(), xc = potential.energy_exc(density);
    // Keep fractional PW coefficients and occupations fixed: test every strain
    // component without SCF, including transverse tau and frozen scalar core fields.
    auto strained_energy = [&](int mu, int nu, double step) {
        auto cell = lattice;
        for (int j = 0; j < 3; j++) {
            cell(mu, j) += step * lattice(nu, j);
        }
        auto deformation = dot(cell, r3::inverse(lattice));
        ctx->unit_cell().set_lattice_vectors(cell);
        ctx->update();
        ks.update();
        density.update();
        potential.update();
        auto q                = std::make_shared<xc_point_set_t>(*ctx);
        q->coordinates        = original->coordinates;
        q->block_centres      = original->block_centres;
        q->block_sizes        = original->block_sizes;
        q->energy_weights     = original->energy_weights;
        q->descriptor_weights = original->descriptor_weights;
        for (auto* coordinates : {&q->coordinates, &q->block_centres}) {
            for (auto& point : *coordinates) {
                auto r = dot(deformation, r3::vector<double>(point[0], point[1], point[2]));
                point  = {r[0], r[1], r[2]};
            }
        }
        for (size_t p = 0; p < q->coordinates.size(); p++) {
            q->energy_weights[p] *= cell.det() / volume;
            q->descriptor_weights[p] *= cell.det() / volume;
        }
        ctx->cb().external_xc_points_ = std::move(q);
        double energy                 = evaluate(0);
        return std::array<double, 2>{potential.energy_exc(density), energy};
    };
    for (double h : {1e-4, 5e-5}) {
        std::array<double, 2> error{};
        for (int mu = 0; mu < 3; mu++) {
            for (int nu = 0; nu < 3; nu++) {
                auto plus = strained_energy(mu, nu, h), minus = strained_energy(mu, nu, -h);
                error[0] = std::max(error[0], std::abs((plus[0] - minus[0]) / (2 * h) - volume * exact(mu, nu)));
                error[1] = std::max(error[1], std::abs((plus[1] - minus[1]) / (2 * h) - volume * total(mu, nu)));
            }
        }
        // PAW projector response belongs to the nonlocal stress component;
        // only the full tensor contains every contribution to the energy derivative.
        if (!paw) {
            check("spinor XC strain derivative", error[0], 2e-8);
        }
        check("spinor total strain derivative", error[1], 2e-6);
    }
    check("spinor strain test restores energy", std::abs(strained_energy(0, 0, 0)[0] - xc), 2e-12);
    Force force(*ctx, density, potential, ks);
    auto const& f = force.calc_forces_total(false);
    std::array<double, 3> exact_force{f(0, 0), f(1, 0), f(2, 0)};
    auto position         = ctx->unit_cell().atom(0).position();
    auto displaced_energy = [&](int axis, double step) {
        r3::vector<double> shift{0, 0, 0};
        shift[axis] = step;
        ctx->unit_cell().atom(0).set_position(position + dot(r3::inverse(lattice), shift));
        ctx->update();
        ks.update();
        density.update();
        potential.update();
        auto q                = std::make_shared<xc_point_set_t>(*ctx);
        q->coordinates        = original->coordinates;
        q->block_centres      = original->block_centres;
        q->block_sizes        = original->block_sizes;
        q->energy_weights     = original->energy_weights;
        q->descriptor_weights = original->descriptor_weights;
        for (auto& centre : q->block_centres) {
            centre[axis] += step;
        }
        ctx->cb().external_xc_points_ = std::move(q);
        return evaluate(0);
    };
    for (double h : {1e-4, 5e-5}) {
        double error{0};
        for (int axis = 0; axis < 3; axis++) {
            double plus = displaced_energy(axis, h), minus = displaced_energy(axis, -h);
            error = std::max(error, std::abs((plus - minus) / (2 * h) + exact_force[axis]));
        }
        check("spinor total force derivative", error, 2e-8);
    }
    check("spinor force test restores energy", std::abs(displaced_energy(0, 0) - center), 2e-12);
    if (soc) {
        Hamiltonian0<double> h0(potential, false);
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            check_spinor_eigensystem(h0(*kp), *kp, check);
        }
    }
    return failures;
}

int
test_external_xc(bool gamma, int mag, bool parallel_fft, bool points, bool paw, bool nlcc, bool gpu, double charge)
{
    auto core = [=](Simulation_context& ctx) {
        if (nlcc) {
            set_test_core(ctx);
            auto const& type = ctx.unit_cell().atom_type(0);
            auto quadrature  = std::make_shared<xc_point_set_t>(*ctx.cb().external_xc_points_);
            if (!quadrature->coordinates.empty()) {
                quadrature->coordinates[1] = {1.6 + type.radial_grid(100), 2.7, 4.0};
            }
            ctx.cb().external_xc_points_ = std::move(quadrature);
        }
    };
    auto ctx =
            test_context(gamma, mag, false, false, parallel_fft, true, 0, paw, points, "gaussian", charge, core, gpu);
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
                    kp->band_occupancy(ib, s, ib == 0 ? (mag ? (s == 0 ? 1.2 : 0.8) : 2.0) * (1 - charge / 2) : 0);
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
    std::vector<double> input_rho, input_tau;
    if (paw || nlcc || charge != 0) {
        ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
            input_rho.resize(in.num_points * in.num_spins);
            input_tau.resize(input_rho.size());
            for (size_t i = 0; i < input_rho.size(); i++) {
                input_rho[i] = in.rho[i];
                input_tau[i] = in.tau[i];
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
            {
                auto psi_guard =
                        kp->spinor_wave_functions().memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::device);
                auto hphi_guard = hphi.memory_guard(ctx->processing_unit_memory_t(), wf::copy_to::host);
                if (gamma) {
                    Hk.apply_h_s<double>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(), &hphi,
                                         nullptr);
                } else {
                    Hk.apply_h_s<std::complex<double>>(wf::spin_range(s), wf::band_range(0, 1),
                                                       kp->spinor_wave_functions(), &hphi, nullptr);
                }
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
    Test_check check(failures, ctx->comm(),
                     "gamma=" + std::to_string(gamma) + " mag=" + std::to_string(mag) +
                             " parallel FFT=" + std::to_string(parallel_fft) + " points=" + std::to_string(points) +
                             " PAW=" + std::to_string(paw) + " NLCC=" + std::to_string(nlcc) +
                             " GPU=" + std::to_string(gpu) + " charge=" + std::to_string(charge));
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
    if (paw || nlcc || charge != 0) {
        std::vector<std::array<double, 3>> coordinates;
        if (points) {
            coordinates = ctx->cb().external_xc_points_->coordinates;
        } else {
            auto const& fft = ctx->spfft<double>();
            for (int z = 0; z < fft.local_z_length(); z++) {
                for (int y = 0; y < fft.dim_y(); y++) {
                    for (int x = 0; x < fft.dim_x(); x++) {
                        coordinates.push_back({8.0 * x / fft.dim_x(), 9.0 * y / fft.dim_y(),
                                               10.0 * (z + fft.local_z_offset()) / fft.dim_z()});
                    }
                }
            }
        }
        int np = static_cast<int>(coordinates.size());
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
            double occupation = (mag ? (s == 0 ? 1.2 : 0.8) : 2) * (1 - charge / 2);
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
        if (nlcc) {
            Smooth_periodic_function_sampler sampler(ctx->gvec(), points);
            auto scalar      = sampler.sample(density.kinetic_density().scalar().rg());
            auto vector      = sampler.sample(density.kinetic_density().vector(0).rg());
            auto const& type = ctx->unit_cell().atom_type(0);
            double error{0};
            if (np > 1) {
                for (int s = 0; s < ctx->num_spins(); s++) {
                    error          = std::max(error, std::abs(input_rho[1 + np * s] - rho(1, s) -
                                                              type.ps_core_charge_density()[100] / ctx->num_spins()));
                    double valence = (scalar[1][0] + (1 - 2 * s) * vector[1][0]) / 2;
                    error          = std::max(error, std::abs(input_tau[1 + np * s] - valence -
                                                              type.ps_core_kinetic_density()[100] / ctx->num_spins()));
                }
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
            check("independently supplied NLCC charge and tau", error, 2e-12);
        }
        if (paw || nlcc) {
            PAW_xc_reconstruction(density, points).add_to(rho, gradient, tau);
        }
        double error{0};
        for (size_t i = 0; i < rho.size(); i++) {
            error = std::max(error, std::abs(rho[i] - input_rho[i]));
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        check("physical XC charge excludes compensation", error, 2e-12);
    }
    check("total-energy double counting", std::abs(ks_energy(*ctx, ks, density, potential) - center), 2e-10);
    if (!paw) {
        check("kinetic-energy contraction",
              std::abs(energy_kin(*ctx, ks, density, potential) - density.kinetic_density().scalar().integrate().total),
              2e-10);
    }
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
        check.reject("diagnostic rejects invalid step", [&] { diagnostic.check_external_xc_derivative(bad_step); });
    }
    if (ctx->comm().size() > 1) {
        check.reject("diagnostic rejects inconsistent step",
                     [&] { diagnostic.check_external_xc_derivative(ctx->comm().rank() == 0 ? 0.001 : 0.002); });
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
    check.reject("diagnostic propagates callback failure", [&] { diagnostic.check_external_xc_derivative(1e-4); });
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
    // Copy and mixing are covered by the density and LAPW field-map tests.
    Density restored(*ctx);
    unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    ctx->comm().bcast(&stamp, 1, 0);
    auto path = std::filesystem::temp_directory_path() / ("sirius-xc-restart-" + std::to_string(stamp) + ".h5");
    ctx->create_storage_file(path.string());
    density.save(path.string());
    ctx->comm().barrier();
    restored.load(path.string());
    double error{0};
    for (int j = 0; j < mag + 1; j++) {
        for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) -
                                             restored.kinetic_density().component(j).rg().value(ir)));
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
        check.reject("collective invalid-quadrature rejection", [&] { potential.generate(density, false, true); });
        ctx->cb().external_xc_points_ = original;
        if (ctx->comm().size() > 1) {
            if (ctx->comm().rank() == 0) {
                ctx->cb().external_xc_points_.reset();
            }
            check.reject("collective quadrature-mode mismatch rejection",
                         [&] { potential.generate(density, false, true); });
            ctx->cb().external_xc_points_ = original;
        }
        auto const lattice = ctx->unit_cell().lattice_vectors();
        ctx->unit_cell().set_lattice_vectors({{8.01, 0, 0}, {0, 9, 0}, {0, 0, 10}});
        check.reject("stale quadrature after cell change rejection", [&] { original->validate(*ctx); });
        ctx->unit_cell().set_lattice_vectors(lattice);
    }

    ctx->cb().external_xc_ = [](auto const& c, auto const&, auto& out) {
        if (c.comm().rank() == 0) {
            out.energy = std::numeric_limits<double>::quiet_NaN();
        }
    };
    check.reject("collective nonfinite rejection", [&] { potential.generate(density, false, true); });
    return failures;
}

int
test_external_nlcc()
{
    int failures = test_external_xc(false, 1, mpi::Communicator::world().size() > 1, true, false, true, false, 0);
    Test_check check(failures, mpi::Communicator::world());
    check.reject("NLCC without setup kinetic density", [] {
        auto ctx = test_context(false, 1, false, false, false, true, 0, false, true);
        if (ctx->comm().rank() == 0) {
            auto& type = ctx->unit_cell().atom_type(0);
            type.ps_core_charge_density(std::vector<double>(type.num_mt_points(), 0.02));
        }
        Density density(*ctx);
    });
    return failures;
}

int
test_external_scf(bool exact, bool external, bool paw, bool gpu, double* total = nullptr)
{
    bool points = external && (exact || paw);
    bool gamma  = paw && !exact;
    int mag     = paw && !exact ? 1 : 0;
    auto ctx    = test_context(gamma, mag, true, exact, false, external, 0, paw, points, "gaussian", 0, {}, gpu);
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
    double error = std::abs(ks_energy(*ctx, ks, density, potential) - direct);
    if (total) {
        *total = direct;
    }
    if (ctx->comm().rank() == 0) {
        std::cout << (external ? "external XC " : "LibXC ") << (exact ? "exact" : "Davidson") << " PAW=" << paw
                  << " GPU=" << gpu << " SCF converged=" << converged << " direct-energy difference=" << error << '\n';
    }
    return failures + (!converged || !std::isfinite(error) || error > 1e-7);
}

int
test_external_geometry(bool paw, int mag, bool strain, bool gamma, bool nlcc, bool gpu, double charge)
{
    auto calculate = [&](double displacement, bool get_force) {
        std::array<double, 3> direction{0.7, -0.4, 0.2};
        r3::matrix<double> deformation{{0.3, 0.1, -0.07}, {0.1, -0.2, 0.06}, {-0.07, 0.06, 0.15}};
        auto setup = [&](Simulation_context& ctx) {
            if (nlcc) {
                set_test_core(ctx);
            }
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
                                charge, setup, gpu);
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
                      << " gamma=" << gamma << " NLCC=" << nlcc << " GPU=" << gpu << " charge=" << charge << " h=" << h
                      << " derivative=" << centre[1] << " FD=" << fd << " error=" << error << '\n';
        }
        failures += !std::isfinite(error) || error > 2e-6;
    }
    return failures;
}

int
test_occupancy(int mag, std::string smearing, double electrons, double moment = 0)
{
    auto ctx = test_context(false, mag, false, false, false, false, moment, false, false, smearing, 2 - electrons);
    K_point_set ks(*ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
    ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    ks.initialize();
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < ctx->num_spinors(); s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                // Saturation can empty the lowest spin channel; clustered bands
                // put the unconstrained chemical potential outside the spectrum.
                double energy = moment ? (((moment > 0) == (s == 1)) ? -0.3 : 0.1) + 0.2 * b + 0.03 * it.i
                                       : -0.3 + 0.001 * (b + s + it.i);
                kp->band_energy(b, s, energy);
            }
        }
    }
    ks.sync_band<double, sync_band_t::energy>();
    ks.find_band_occupancies<double>();
    double occupation[2]{0, 0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        for (int s = 0; s < ctx->num_spinors(); s++) {
            for (int b = 0; b < ctx->num_bands(); b++) {
                occupation[s] += kp->weight() * kp->band_occupancy(b, s);
            }
        }
    }
    ks.comm().allreduce(occupation, 2);
    double error = std::abs(occupation[0] + occupation[1] - electrons);
    if (moment) {
        error = std::max({error, std::abs(occupation[0] - (electrons + moment) / 2),
                          std::abs(occupation[1] - (electrons - moment) / 2)});
    }
    std::cout << "Occupations " << smearing << ", mag=" << mag << ", electrons=" << electrons << ", moment=" << moment
              << ", error=" << error << '\n';
    return !std::isfinite(error) || error > 2e-11;
}

int
test_dispersion_energy()
{
    std::map<std::string, double> energies{{"ekin", 11},
                                           {"exc", 13},
                                           {"vha", 17},
                                           {"enuc", 19},
                                           {"valence_eval_sum", 23},
                                           {"vxc", 29},
                                           {"bxc", 31},
                                           {"PAW_one_elec", 37},
                                           {"PAW_total_energy", 41},
                                           {"ewald", 43},
                                           {"vtau", 47},
                                           {"dftd3_energy", -0.3},
                                           {"dftd4_energy", -0.4}};
    int failures{0};
    for (bool full_potential : {false, true}) {
        for (int corrections = 0; corrections < 4; corrections++) {
            auto config = R"({"parameters":{"electronic_structure_method":"pseudopotential"},
                             "control":{"processing_unit":"CPU"}})"_json;
            config["parameters"]["electronic_structure_method"] =
                    full_potential ? "full_potential_lapwlo" : "pseudopotential";
            config["parameters"]["dftd3_correction"] = bool(corrections & 1);
            config["parameters"]["dftd4_correction"] = bool(corrections & 2);
            Simulation_context ctx(config);
            ctx.electronic_structure_method(full_potential ? "full_potential_lapwlo" : "pseudopotential");
            double expected =
                    (full_potential ? 51.5 : -32.5) - ((corrections & 1) ? 0.3 : 0) - ((corrections & 2) ? 0.4 : 0);
            double actual = ks_energy(ctx, energies);
            failures += !std::isfinite(actual) || std::abs(actual - expected) > 1e-13;
        }
    }
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result{0};
    if (argc == 2 && (std::string(argv[1]) == "spinor" || std::string(argv[1]) == "spinor-gpu" ||
                      std::string(argv[1]) == "spinor-paw" || std::string(argv[1]) == "spinor-paw-gpu" ||
                      std::string(argv[1]) == "spinor-soc" || std::string(argv[1]) == "spinor-soc-gpu" ||
                      std::string(argv[1]) == "spinor-paw-soc" || std::string(argv[1]) == "spinor-paw-soc-gpu")) {
        std::string mode(argv[1]);
        result += call_test("external spinor XC", test_spinor_xc, mode.find("gpu") != std::string::npos,
                            mode.find("paw") != std::string::npos, mode.find("soc") != std::string::npos, 0);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && std::string(argv[1]) == "gpu") {
#if defined(SIRIUS_GPU)
        bool parallel = mpi::Communicator::world().size() > 1;
        result += call_test("GPU external XC", test_external_xc, true, 0, false, false, false, false, true, 0);
        result += call_test("GPU PAW point XC", test_external_xc, false, 1, parallel, true, true, false, true, 0);
        result += call_test("GPU NLCC point XC", test_external_xc, false, 1, false, true, false, true, true, 0);
        result += call_test("GPU exact LibXC SCF", test_external_scf, true, false, false, true, nullptr);
        result += call_test("CPU/GPU SCF agreement", [] {
            int failures{0};
            Test_check check(failures, mpi::Communicator::world());
            for (bool paw : {false, true}) {
                double cpu{0}, gpu{0};
                failures += test_external_scf(false, true, paw, false, &cpu);
                failures += test_external_scf(false, true, paw, true, &gpu);
                check("CPU/GPU total energy PAW=" + std::to_string(paw), std::abs(cpu - gpu), 5e-8);
            }
            return failures;
        });
#else
        result = 1;
#endif
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && std::string(argv[1]) == "nlcc") {
        result += call_test("external nonlinear core correction", test_external_nlcc);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && (std::string(argv[1]) == "charged" || std::string(argv[1]) == "charged-gpu")) {
        bool gpu      = std::string(argv[1]) == "charged-gpu";
        bool parallel = mpi::Communicator::world().size() > 1;
        result += call_test("charged FFT XC", test_external_xc, true, 0, false, false, false, false, gpu, 0.25);
        result +=
                call_test("charged NLCC point XC", test_external_xc, false, 1, parallel, true, false, true, gpu, -0.25);
        result += call_test("charged PAW point XC", test_external_xc, false, 1, parallel, true, true, false, gpu, 0.25);
        result += call_test("charged spinor SOC XC", test_spinor_xc, gpu, false, true, 0.25);
        result += call_test("charged PAW spinor SOC XC", test_spinor_xc, gpu, true, true, 0.25);
        result += call_test("charged PW stress", test_external_geometry, false, 0, true, true, false, gpu, 0.25);
        result += call_test("charged PAW stress", test_external_geometry, true, 1, true, false, false, gpu, -0.25);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && (std::string(argv[1]) == "gpu-geometry" || std::string(argv[1]) == "core-geometry")) {
        bool gpu  = std::string(argv[1]) == "gpu-geometry";
        bool nlcc = !gpu;
        result += call_test("external NC forces", test_external_geometry, false, 0, false, false, nlcc, gpu, 0);
        result += call_test("external NC stress", test_external_geometry, false, 0, true, true, nlcc, gpu, 0);
        if (gpu) {
            result += call_test("GPU PAW forces", test_external_geometry, true, 1, false, false, false, true, 0);
            result += call_test("GPU PAW stress", test_external_geometry, true, 1, true, false, false, true, 0);
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && std::string(argv[1]) == "forces") {
        result += call_test("external PW forces", test_external_geometry, false, 0, false, false, false, false, 0);
        result += call_test("external PAW forces", test_external_geometry, true, 1, false, false, false, false, 0);
        sirius::finalize();
        return std::min(result, 1);
    }
    if (argc == 2 && std::string(argv[1]) == "stress") {
        result += call_test("external PW stress", test_external_geometry, false, 0, true, true, false, false, 0);
        result += call_test("external PAW stress", test_external_geometry, true, 1, true, false, false, false, 0);
        sirius::finalize();
        return std::min(result, 1);
    }
    result += call_test("dispersion energy accounting", test_dispersion_energy);
    for (auto smearing : {"gaussian", "fermi_dirac", "cold", "methfessel_paxton"}) {
        result += call_test("low-filling occupations", test_occupancy, 0, smearing, 1.0, 0.0);
        result += call_test("high-filling spin occupations", test_occupancy, 1, smearing, 7.0, 0.0);
    }
    for (bool gamma : {false, true}) {
        for (int mag : {0, 1}) {
            result += call_test("external XC energy and operator", test_external_xc, gamma, mag, false, false, false,
                                false, false, gamma && !mag ? 0.25 : 0);
            if (gamma != mag) {
                result += call_test("external XC point quadrature", test_external_xc, gamma, mag, false, true, false,
                                    false, false, 0);
            } else {
                result += call_test("external PAW XC energy and operator", test_external_xc, gamma, mag, false, true,
                                    true, false, false, gamma ? -0.25 : 0);
            }
            if (mpi::Communicator::world().size() > 1 && gamma != mag) {
                result += call_test("external XC distributed FFT", test_external_xc, gamma, mag, true, false, false,
                                    false, false, 0);
                if (!gamma && mag == 1) {
                    result += call_test("point geometry with distributed FFT", test_external_xc, gamma, mag, true, true,
                                        false, false, false, 0);
                }
                result += call_test("external PAW XC distributed FFT", test_external_xc, gamma, mag, true, true, true,
                                    false, false, 0);
            }
        }
    }
    result += call_test("external nonlinear core correction", test_external_nlcc);
    result += call_test("external spinor SOC", test_spinor_xc, false, false, true, 0);
    result += call_test("external PAW spinor SOC", test_spinor_xc, false, true, true, 0);
    for (bool external : {false, true}) {
        for (bool exact : {false, true}) {
            result += call_test("SCF energy accounting", test_external_scf, exact, external, false, false, nullptr);
        }
    }
    for (bool exact : {false, true}) {
        result += call_test("PAW SCF energy accounting", test_external_scf, exact, true, true, false, nullptr);
    }
    result += call_test("external PW forces", test_external_geometry, false, 0, false, false, false, false, 0);
    result += call_test("external PAW forces", test_external_geometry, true, 1, false, false, false, false, 0);
    result += call_test("external PW stress", test_external_geometry, false, 0, true, true, false, false, 0);
    result += call_test("external PAW stress", test_external_geometry, true, 1, true, false, false, false, 0);
    for (double moment : {2.0, -2.0, 1.2}) {
        result += call_test("fixed-spin occupations", test_occupancy, 1, "gaussian", 2.0, moment);
    }
    sirius::finalize();
    return std::min(result, 1);
}
