/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file poisson.hpp
 *
 *  \brief Implementation of the Poisson solver for the full-potential muffin-tin case.
 */

#include "potential.hpp"
#include "core/profiler.hpp"
#include "lapw/sum_fg_fl_yg.hpp"

namespace sirius {
namespace {

void
validate_poisson_density(Simulation_context const& ctx, Periodic_function<double> const& density)
{
    int invalid = &density.ctx() != &ctx;
    if (!invalid) {
        for (auto z : density.rg().f_pw_local()) {
            invalid |= !std::isfinite(z.real()) || !std::isfinite(z.imag());
        }
        if (ctx.full_potential()) {
            if (density.mt().atoms().size() != static_cast<size_t>(ctx.unit_cell().num_atoms())) {
                invalid = 1;
            } else {
                for (auto it : ctx.unit_cell().spl_num_atoms()) {
                    auto const& rho = density.mt()[it.i];
                    if (rho.angular_domain_size() != ctx.lmmax_rho() ||
                        rho.size() !=
                                static_cast<size_t>(ctx.lmmax_rho() * ctx.unit_cell().atom(it.i).num_mt_points())) {
                        invalid = 1;
                        continue;
                    }
                    invalid |= &rho.radial_grid() != &ctx.unit_cell().atom(it.i).radial_grid();
                    for (size_t i = 0; i < rho.size(); i++) {
                        invalid |= !std::isfinite(rho[i]);
                    }
                }
            }
        }
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Hartree response requires finite density fields with matching context and MT dimensions");
    }
}

} // namespace

void
Potential::poisson_add_pseudo_pw(mdarray<std::complex<double>, 2> const& qmt__,
                                 mdarray<std::complex<double>, 2> const& qit__, std::complex<double>* rho_pw__) const
{
    PROFILE("sirius::Potential::poisson_add_pseudo_pw");

    int lmmax = ctx_.lmmax_rho();
    int ngv   = ctx_.gvec().count();

    /* The following term is added to the plane-wave coefficients of the charge density:
     * Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] /
     *  Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
     * i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by
     * the difference bethween true and interstitial-in-the-mt multipole moments and divided by the
     * moment of the pseudodensity.
     */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        double R = unit_cell_.atom_type(iat).mt_radius();
        int na   = unit_cell_.atom_type(iat).num_atoms();

        mdarray<std::complex<double>, 2> pf;
        mdarray<std::complex<double>, 2> qa;
        mdarray<std::complex<double>, 2> qapf;

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                auto& mp = get_memory_pool(memory_t::host);
                pf       = mdarray<std::complex<double>, 2>({ngv, na}, mp);
                qa       = mdarray<std::complex<double>, 2>({lmmax, na}, mp);
                qapf     = mdarray<std::complex<double>, 2>({lmmax, ngv}, mp);
                break;
            }
            case device_t::GPU: {
                auto& mp  = get_memory_pool(memory_t::host_pinned);
                auto& mpd = get_memory_pool(memory_t::device);
                /* allocate on GPU */
                pf = mdarray<std::complex<double>, 2>({ngv, na}, nullptr);
                pf.allocate(mpd);
                /* allocate on CPU & GPU */
                qa = mdarray<std::complex<double>, 2>({lmmax, na}, mp);
                qa.allocate(mpd);
                /* allocate on CPU & GPU */
                qapf = mdarray<std::complex<double>, 2>({lmmax, ngv}, mp);
                qapf.allocate(mpd);
                break;
            }
        }

        ctx_.generate_phase_factors(iat, pf);

        for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                qa(lm, i) = qmt__(lm, ia) - qit__(lm, ia);
            }
        }

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'C', ctx_.lmmax_rho(), ctx_.gvec().count(), unit_cell_.atom_type(iat).num_atoms(),
                              &la::constant<std::complex<double>>::one(), qa.at(memory_t::host), qa.ld(),
                              pf.at(memory_t::host), pf.ld(), &la::constant<std::complex<double>>::zero(),
                              qapf.at(memory_t::host), qapf.ld());
                break;
            }
            case device_t::GPU: {
                qa.copy_to(memory_t::device);
                la::wrap(la::lib_t::gpublas)
                        .gemm('N', 'C', ctx_.lmmax_rho(), ctx_.gvec().count(), unit_cell_.atom_type(iat).num_atoms(),
                              &la::constant<std::complex<double>>::one(), qa.at(memory_t::device), qa.ld(),
                              pf.at(memory_t::device), pf.ld(), &la::constant<std::complex<double>>::zero(),
                              qapf.at(memory_t::device), qapf.ld());
                qapf.copy_to(memory_t::host);
                break;
            }
        }

        double fourpi_omega = fourpi / unit_cell_.omega();

        /* add pseudo_density to interstitial charge density so that rho(G) has the correct
         * multipole moments in the muffin-tins */
        #pragma omp parallel for schedule(static)
        for (int igloc = ctx_.gvec().skip_g0(); igloc < ctx_.gvec().count(); igloc++) {
            double gR  = ctx_.gvec().gvec_len(gvec_index_t::local(igloc)) * R;
            double gRn = std::pow(2.0 / gR, pseudo_density_order_ + 1);

            std::complex<double> rho_G(0, 0);
            /* loop over atoms of the same type */
            for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++) {
                std::complex<double> zt1(0, 0);
                for (int m = -l; m <= l; m++, lm++) {
                    zt1 += gvec_ylm_(lm, igloc) * qapf(lm, igloc);
                }
                rho_G += fourpi_omega * std::conj(zil_[l]) * zt1 * gamma_factors_R_(l, iat) *
                         sbessel_mt_(l + pseudo_density_order_ + 1, igloc, iat) * gRn;
            }
            rho_pw__[igloc] += rho_G;
        }
        /* for G=0 case */
        if (ctx_.comm().rank() == 0) {
            std::complex<double> rho_G(0, 0);
            for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                rho_G += fourpi_omega * y00 * (qmt__(0, ia) - qit__(0, ia));
            }
            rho_pw__[0] += rho_G;
        }
    }
}

void
Potential::poisson_solve(Periodic_function<double> const& rho__, Periodic_function<double>& vha__,
                         bool include_nuclei__) const
{
    PROFILE("sirius::Potential::poisson");

    /* pointer to plane-wave coefficients of density */
    std::complex<double> const* rho_pw = &rho__.rg().f_pw_local(0);

    std::vector<std::complex<double>> rho_pw_modified;

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (ctx_.full_potential()) {

        /* true multipole moments */
        auto qmt = poisson_vmt(rho__.mt(), vha__.mt(), include_nuclei__);

        if (env::print_checksum()) {
            print_checksum("qmt", qmt.checksum(), ctx_.out());
        }

        /* compute multipoles of interstitial density in MT region */
        auto qit = sum_fg_fl_yg(ctx_, ctx_.lmax_rho(), rho_pw, sbessel_mom_, gvec_ylm_);

        if (env::print_checksum()) {
            print_checksum("qit", qit.checksum(), ctx_.out());
        }

        /* copy original rho(G) into temporaty storage */
        rho_pw_modified.resize(ctx_.gvec().count());
        std::copy(rho_pw, rho_pw + ctx_.gvec().count(), rho_pw_modified.begin());

        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, rho_pw_modified.data());

        if (ctx_.cfg().control().verification() >= 2) {
            auto qit = sum_fg_fl_yg(ctx_, ctx_.lmax_rho(), rho_pw_modified.data(), sbessel_mom_, gvec_ylm_);

            double d = 0.0;
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                    d += std::abs(qmt(lm, ia) - qit(lm, ia));
                }
            }
            if (ctx_.verbosity() >= 1) {
                RTE_OUT(ctx_.out()) << "pseudocharge error: " << d << std::endl;
            }
        }
        rho_pw = rho_pw_modified.data();
    }

    /* compute pw coefficients of Hartree potential */
    if (ctx_.gvec().comm().rank() == 0) {
        vha__.rg().f_pw_local(0) = 0.0;
    }
    if (!ctx_.molecule()) {
        #pragma omp parallel for
        for (int igloc = ctx_.gvec().skip_g0(); igloc < ctx_.gvec().count(); igloc++) {
            vha__.rg().f_pw_local(igloc) =
                    fourpi * rho_pw[igloc] / std::pow(ctx_.gvec().gvec_len(gvec_index_t::local(igloc)), 2);
        }
    } else {
        /* reference paper:

           Supercell technique for total-energy calculations of finite charged and polar systems
           M. R. Jarvis, I. D. White, R. W. Godby, and M. C. Payne
           Phys. Rev. B 56, 14972 – Published 15 December 1997
           DOI:https://doi.org/10.1103/PhysRevB.56.14972
        */
        double R_cut = 0.5 * std::pow(unit_cell_.omega(), 1.0 / 3);
        #pragma omp parallel for
        for (int igloc = ctx_.gvec().skip_g0(); igloc < ctx_.gvec().count(); igloc++) {
            auto glen = ctx_.gvec().gvec_len(gvec_index_t::local(igloc));
            vha__.rg().f_pw_local(igloc) =
                    (fourpi * rho_pw[igloc] / std::pow(glen, 2)) * (1.0 - std::cos(glen * R_cut));
        }
    }

    /* boundary condition for muffin-tins */
    if (ctx_.full_potential()) {
        /* compute V_lm at the MT boundary */
        auto vmtlm = sum_fg_fl_yg(ctx_, ctx_.lmax_pot(), &vha__.rg().f_pw_local(0), sbessel_mt_, gvec_ylm_);

        /* add boundary condition and convert to Rlm */
        PROFILE("sirius::Potential::poisson|bc");
        mdarray<double, 3> rRl({unit_cell_.max_num_mt_points(), ctx_.lmax_pot() + 1, unit_cell_.num_atom_types()});
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            int nmtp = unit_cell_.atom_type(iat).num_mt_points();
            double R = unit_cell_.atom_type(iat).mt_radius();

            #pragma omp parallel for default(shared)
            for (int l = 0; l <= ctx_.lmax_pot(); l++) {
                for (int ir = 0; ir < nmtp; ir++) {
                    rRl(ir, l, iat) = std::pow(unit_cell_.atom_type(iat).radial_grid(ir) / R, l);
                }
            }
        }

        for (auto it : unit_cell_.spl_num_atoms()) {
            int nmtp = unit_cell_.atom(it.i).num_mt_points();

            std::vector<double> vlm(ctx_.lmmax_pot());
            SHT::convert(ctx_.lmax_pot(), &vmtlm(0, it.i), &vlm[0]);

            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) {
                    vha__.mt()[it.i](lm, ir) += vlm[lm] * rRl(ir, l, unit_cell_.atom(it.i).type_id());
                }
            }
        }
    }
    if (ctx_.cfg().parameters().veff_pw_cutoff() > 0) {
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            if (ctx_.gvec().gvec_len(gvec_index_t::local(ig)) > ctx_.cfg().parameters().veff_pw_cutoff()) {
                vha__.rg().f_pw_local(ig) = 0;
            }
        }
    }

    /* transform Hartree potential to real space */
    vha__.rg().fft_transform(1);

    if (env::print_checksum()) {
        auto cs  = vha__.rg().checksum_rg();
        auto cs1 = vha__.rg().checksum_pw();
        print_checksum("vha_rg", cs, ctx_.out());
        print_checksum("vha_pw", cs1, ctx_.out());
    }
}

std::unique_ptr<Periodic_function<double>>
Potential::hartree_response(Periodic_function<double> const& density_response__) const
{
    validate_poisson_density(ctx_, density_response__);
    auto result = ctx_.full_potential()
                          ? std::make_unique<Periodic_function<double>>(
                                    ctx_, [&](int) { return lmax_t(ctx_.lmax_pot()); }, &unit_cell_.spl_num_atoms())
                          : std::make_unique<Periodic_function<double>>(ctx_);
    result->zero();
    poisson_solve(density_response__, *result, false);
    return result;
}

std::unique_ptr<Periodic_function<double>>
Potential::hartree_geometry_response(Periodic_function<double> const& density__,
                                     xc_geometry_direction_t const& direction__, bool include_nuclei__) const
{
    validate_poisson_density(ctx_, density__);
    int invalid = !ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU || ctx_.gvec().reduced() ||
                  direction__.atoms.size() != static_cast<size_t>(unit_cell_.num_atoms());
    std::vector<double> values(direction__.lattice.begin(), direction__.lattice.end());
    values.push_back(include_nuclei__ ? 1 : 0);
    for (auto const& atom : direction__.atoms) {
        values.insert(values.end(), atom.begin(), atom.end());
    }
    for (auto x : values) {
        invalid |= !std::isfinite(x);
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW Hartree geometry response requires CPU fields and matching finite geometry directions");
    }
    auto root = values;
    ctx_.comm().bcast(root.data(), root.size(), 0);
    invalid = root != values;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Hartree geometry directions and nuclear flags must agree across the context communicator");
    }
    r3::matrix<double> dlattice;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            dlattice(x, y) = direction__.lattice[x + 3 * y];
        }
    }
    auto strain  = dot(dlattice, unit_cell_.inverse_lattice_vectors());
    double trace = strain(0, 0) + strain(1, 1) + strain(2, 2);
    int lmax = std::max(ctx_.lmax_rho(), ctx_.lmax_pot()), ng = ctx_.gvec().count();
    matrix<std::complex<double>> dy({sf::lmmax(lmax), ng});
    mdarray<double, 3> dmom({ctx_.lmax_rho() + 1, ng, unit_cell_.num_atom_types()});
    mdarray<double, 3> dbessel({sbessel_mt_.size(0), ng, unit_cell_.num_atom_types()});
    dy.zero();
    dmom.zero();
    dbessel.zero();
    std::vector<r3::vector<double>> dg(ng);
    std::vector<double> dlength(ng), drlm(sf::lmmax(lmax));
    std::vector<std::complex<double>> derivative(sf::lmmax(lmax));
    mdarray<double, 2> gradient({sf::lmmax(lmax), 3});
    for (int ig = ctx_.gvec().skip_g0(); ig < ng; ig++) {
        auto g        = ctx_.gvec().gvec_cart(gvec_index_t::local(ig));
        double length = g.length();
        dg[ig]        = dot(transpose(strain), g) * -1.0;
        dlength[ig]   = dot(g, dg[ig]) / length;
        sf::dRlm_dr(lmax, g, gradient);
        for (int lm = 0; lm < sf::lmmax(lmax); lm++) {
            drlm[lm] = 0;
            for (int x = 0; x < 3; x++) {
                drlm[lm] += gradient(lm, x) * dg[ig][x];
            }
        }
        SHT::convert(lmax, drlm.data(), derivative.data());
        for (int lm = 0; lm < sf::lmmax(lmax); lm++) {
            dy(lm, ig) = std::conj(derivative[lm]);
        }
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            double R = unit_cell_.atom_type(iat).mt_radius();
            for (int l = 0; l < dbessel.size(0); l++) {
                double derivative =
                        l == 0 ? -R * sbessel_mt_(1, ig, iat)
                               : R * sbessel_mt_(l - 1, ig, iat) - (l + 1) / length * sbessel_mt_(l, ig, iat);
                dbessel(l, ig, iat) = derivative * dlength[ig];
            }
            for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                dmom(l, ig, iat) = std::pow(R, l + 2) / length *
                                   (dbessel(l + 1, ig, iat) - sbessel_mt_(l + 1, ig, iat) * dlength[ig] / length);
            }
        }
    }
    // Differentiate the radial factor, spherical harmonic and atom phase together.
    auto multipole_response = [&](int cutoff, std::complex<double> const* fpw, auto const& fl, auto const& dfl) {
        mdarray<std::complex<double>, 2> result({sf::lmmax(cutoff), unit_cell_.num_atoms()});
        result.zero();
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& atom = unit_cell_.atom(ia);
            auto r           = unit_cell_.get_cartesian_coordinates(atom.position());
            auto const& d    = direction__.atoms[ia];
            r3::vector<double> dr{d[0], d[1], d[2]};
            for (int ig = 0; ig < ng; ig++) {
                auto g     = ctx_.gvec().gvec_cart(gvec_index_t::local(ig));
                auto phase = std::exp(std::complex<double>(0, dot(g, r)));
                std::complex<double> dp(0, dot(dg[ig], r) + dot(g, dr));
                for (int l = 0, lm = 0; l <= cutoff; l++) {
                    auto scale = fourpi * zil_[l] * fpw[ig] * phase;
                    for (int m = -l; m <= l; m++, lm++) {
                        result(lm, ia) += scale * (dfl(l, ig, atom.type_id()) * std::conj(gvec_ylm_(lm, ig)) +
                                                   fl(l, ig, atom.type_id()) *
                                                           (std::conj(dy(lm, ig)) + dp * std::conj(gvec_ylm_(lm, ig))));
                    }
                }
            }
        }
        ctx_.comm().allreduce(result.at(memory_t::host), result.size());
        return result;
    };
    auto make_potential = [&] {
        return std::make_unique<Periodic_function<double>>(
                ctx_, [&](int) { return lmax_t(ctx_.lmax_pot()); }, &unit_cell_.spl_num_atoms());
    };
    auto base = make_potential(), result = make_potential();
    base->zero();
    result->zero();
    // True atom-local multipoles stay fixed; the removed interstitial multipoles move.
    auto qmt  = poisson_vmt(density__.mt(), base->mt(), include_nuclei__);
    auto qit  = sum_fg_fl_yg(ctx_, ctx_.lmax_rho(), density__.rg().f_pw_local().at(memory_t::host), sbessel_mom_,
                             gvec_ylm_);
    auto dqit = multipole_response(ctx_.lmax_rho(), density__.rg().f_pw_local().at(memory_t::host), sbessel_mom_, dmom);
    auto const& rho = density__.rg().f_pw_local();
    std::vector<std::complex<double>> modified(rho.begin(), rho.end()), delta(ng);
    poisson_add_pseudo_pw(qmt, qit, modified.data());
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto const& atom = unit_cell_.atom(ia);
        int iat          = atom.type_id();
        double R = atom.mt_radius(), scale = fourpi / unit_cell_.omega();
        auto r        = unit_cell_.get_cartesian_coordinates(atom.position());
        auto const& d = direction__.atoms[ia];
        r3::vector<double> dr{d[0], d[1], d[2]};
        for (int ig = 0; ig < ng; ig++) {
            auto g        = ctx_.gvec().gvec_cart(gvec_index_t::local(ig));
            double length = g.length();
            if (length == 0) {
                delta[ig] += scale * y00 * (-dqit(0, ia) - trace * (qmt(0, ia) - qit(0, ia)));
                continue;
            }
            auto phase = std::exp(std::complex<double>(0, -dot(g, r)));
            std::complex<double> dp(0, -dot(dg[ig], r) - dot(g, dr));
            double power = std::pow(2.0 / (length * R), pseudo_density_order_ + 1);
            for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++) {
                int order = l + pseudo_density_order_ + 1;
                double b  = sbessel_mt_(order, ig, iat);
                double db = dbessel(order, ig, iat) - b * (trace + (pseudo_density_order_ + 1) * dlength[ig] / length);
                auto factor = scale * std::conj(zil_[l]) * gamma_factors_R_(l, iat) * power * phase;
                for (int m = -l; m <= l; m++, lm++) {
                    auto q = qmt(lm, ia) - qit(lm, ia), y = gvec_ylm_(lm, ig);
                    delta[ig] += factor * (db * y * q + b * (dy(lm, ig) * q + y * (-dqit(lm, ia) + dp * q)));
                }
            }
        }
    }
    double Rcut = 0.5 * std::cbrt(unit_cell_.omega());
    for (int ig = ctx_.gvec().skip_g0(); ig < ng; ig++) {
        double length = ctx_.gvec().gvec_len(gvec_index_t::local(ig));
        double kernel = ctx_.molecule() ? 1 - std::cos(length * Rcut) : 1;
        double dk     = ctx_.molecule() ? std::sin(length * Rcut) * Rcut * (dlength[ig] + length * trace / 3) : 0;
        auto v        = fourpi * modified[ig] / (length * length);
        base->rg().f_pw_local(ig) = kernel * v;
        result->rg().f_pw_local(ig) =
                fourpi * kernel * delta[ig] / (length * length) + (dk - 2 * kernel * dlength[ig] / length) * v;
    }
    auto boundary =
            sum_fg_fl_yg(ctx_, ctx_.lmax_pot(), result->rg().f_pw_local().at(memory_t::host), sbessel_mt_, gvec_ylm_);
    auto moved_boundary =
            multipole_response(ctx_.lmax_pot(), base->rg().f_pw_local().at(memory_t::host), sbessel_mt_, dbessel);
    for (auto it : unit_cell_.spl_num_atoms()) {
        auto const& atom = unit_cell_.atom(it.i);
        std::vector<double> vlm(ctx_.lmmax_pot());
        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
            boundary(lm, it.i) += moved_boundary(lm, it.i);
        }
        SHT::convert(ctx_.lmax_pot(), &boundary(0, it.i), vlm.data());
        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
            for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                result->mt()[it.i](lm, ir) = vlm[lm] * std::pow(atom.radial_grid(ir) / atom.mt_radius(), l_by_lm_[lm]);
            }
        }
    }
    if (ctx_.cfg().parameters().veff_pw_cutoff() > 0) {
        for (int ig = 0; ig < ng; ig++) {
            if (ctx_.gvec().gvec_len(gvec_index_t::local(ig)) > ctx_.cfg().parameters().veff_pw_cutoff()) {
                result->rg().f_pw_local(ig) = 0;
            }
        }
    }
    result->rg().fft_transform(1);
    return result;
}

void
Potential::poisson(Periodic_function<double> const& rho__)
{
    poisson_solve(rho__, *hartree_potential_, true);
    if (ctx_.full_potential()) {
        for (auto it : unit_cell_.spl_num_atoms()) {
            /* save electronic part of the potential at the point of origin */
#ifdef __VHA_AUX
            vh_el_(it.i) = y00 * hartree_potential_->mt()[it.i](0, 0) +
                           unit_cell_.atom(it.i).zn() / unit_cell_.atom(it.i).radial_grid(0);
#else
            vh_el_(it.i) = y00 * hartree_potential_->mt()[it.i](0, 0);
#endif
        }
        ctx_.comm().allgather(vh_el_.at(memory_t::host), unit_cell_.spl_num_atoms().local_size(),
                              unit_cell_.spl_num_atoms().global_offset());
    }

    /* compute contribution from the smooth part of Hartree potential */
    energy_vha_ = sirius::inner(rho__, hartree_potential());

#ifndef __VHA_AUX
    /* add nucleus potential and contribution to Hartree energy */
    if (ctx_.full_potential()) {
        double evha_nuc{0};
        for (auto it : unit_cell_.spl_num_atoms()) {
            auto& atom = unit_cell_.atom(it.i);
            Spline<double> srho(atom.radial_grid());
            for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                double r = atom.radial_grid(ir);
                hartree_potential_->mt()[it.i](0, ir) -= atom.zn() / r / y00;
                srho(ir) = rho__.mt()[it.i](0, ir) * r;
            }
            evha_nuc -= atom.zn() * srho.interpolate().integrate(0) / y00;
        }
        ctx_.comm().allreduce(&evha_nuc, 1);
        energy_vha_ += evha_nuc;
    }
#endif

    /* check values at MT boundary */
    // if (true) {
    //     /* compute V_lm at the MT boundary */
    //     auto vmtlm = ctx_.sum_fg_fl_yg(ctx_.lmax_pot(), &hartree_potential_->f_pw_local(0), sbessel_mt_, gvec_ylm_);

    //    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
    //        int ia = unit_cell_.spl_num_atoms(ialoc);
    //        int nmtp = unit_cell_.atom(ia).num_mt_points();

    //        std::vector<double> vlm(ctx_.lmmax_pot());
    //        SHT::convert(ctx_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);

    //        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
    //            printf("ia=%i lm=%i vlmdiff=%20.16f\n", ia, lm,
    //              std::abs(hartree_potential_->f_mt<index_domain_t::local>(lm, nmtp - 1, ialoc) - vlm[lm]));
    //        }
    //    }

    //}
}

} // namespace sirius
