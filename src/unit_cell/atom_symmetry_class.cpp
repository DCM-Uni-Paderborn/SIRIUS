/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom_symmetry_class.cpp
 *
 *  \brief Partial implementation of sirius::Atom_symmetry_class class.
 */

#include "atom_symmetry_class.hpp"
#include "core/la/eigensolver.hpp"
#include "radial/radial_core_solver.hpp"

namespace sirius {

Atom_symmetry_class::Atom_symmetry_class(int id__, Atom_type const& atom_type__)
    : id_(id__)
    , atom_type_(atom_type__)
{
    if (!atom_type_.initialized()) {
        RTE_THROW("atom type is not initialized");
    }

    int nl        = atom_type_.indexr().lmax() + 1;
    int max_order = atom_type_.indexr().max_order();
    int nrf       = atom_type_.mt_radial_basis_size();

    surface_derivatives_ = mdarray<double, 2>({3, atom_type_.mt_radial_basis_size() - atom_type_.num_lo_descriptors()});

    radial_functions_ = mdarray<double, 3>({atom_type_.num_mt_points(), nrf, 2});

    h_spherical_integrals_ = mdarray<double, 2>({nrf, nrf});
    h_spherical_integrals_.zero();

    o_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
    o_radial_integrals_.zero();

    so_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
    so_radial_integrals_.zero();

    if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
        o1_radial_integrals_ = mdarray<double, 2>({nrf, nrf});
        o1_radial_integrals_.zero();
    }

    /* copy descriptors because enu is different between atom classes */
    for (int i = 0; i < atom_type_.num_aw_descriptors(); i++) {
        aw_descriptors_.push_back(atom_type_.aw_descriptor(i));
    }

    for (int i = 0; i < atom_type_.num_lo_descriptors(); i++) {
        lo_descriptors_.push_back(atom_type_.lo_descriptor(i));
    }

    /* find which aw functions need auto enu */
    for (int l = 0; l < num_aw_descriptors(); l++) {
        for (auto const& d : aw_descriptor(l)) {
            if (d.auto_enu) {
                enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
            }
        }
    }

    /* find which lo functions need auto enu */
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
        for (auto const& d : lo_descriptor(idxlo).rsd_set) {
            if (d.auto_enu) {
                enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
            }
        }
    }
}

int
Atom_symmetry_class::generate_aw_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__,
                                                  mdarray<double, 2>& sd__, core_states_t const& core__,
                                                  std::vector<double> const* direction__,
                                                  radial_basis_response_t* response__,
                                                  core_states_t const* core_response__,
                                                  enu_response_t const* enu_response__) const
{
    int nmtp = atom_type_.num_mt_points();

    struct compute_all_orders_result
    {
        bool success{false};
        bool solver_failed{false};
        std::string error;
    };

    auto compute_all_orders = [&](int l, double enu_shift) -> compute_all_orders_result {
        Spline<double> s(atom_type_.radial_grid());

        compute_all_orders_result r;
        std::stringstream sinfo;
        sinfo << " l = " << l << ", enu_shift = " << enu_shift << std::endl;

        for (int order = 0; order < (int)aw_descriptor(l).size(); order++) {
            sinfo << "  order = " << order << std::endl;
            auto rsd = aw_descriptor(l)[order];

            auto idxrf = atom_type_.indexr().index_of(angular_momentum(l), order);

            try {
                /* integrate radial equation forward and find radial solution */
                radial_solver_result_t derivative;
                double denu = enu_response__ && rsd.auto_enu ? enu_response__->at({rsd.n, rsd.l}) : 0;
                auto result = solve_radial(rel__, rsd.dme, rsd.l, rsd.enu + enu_shift, core__, direction__, &derivative,
                                           core_response__, denu);
                for (int ir = 0; ir < nmtp; ir++) {
                    rf__(ir, idxrf, 0) = result.p[ir];
                    rf__(ir, idxrf, 1) = result.rdudr[ir];
                    if (response__) {
                        response__->radial_functions(ir, idxrf, 0) = derivative.p[ir];
                        response__->radial_functions(ir, idxrf, 1) = derivative.rdudr[ir];
                    }
                }
                for (int i : {0, 1, 2}) {
                    sd__(i, idxrf) = result.uderiv[i];
                    if (response__) {
                        response__->surface_derivatives(i, idxrf) = derivative.uderiv[i];
                    }
                }
            } catch (std::exception const& e) {
                r.success       = false;
                r.solver_failed = true;
                sinfo << e.what();
                r.error = sinfo.str();
                return r;
            }

            /* orthogonalize to previous radial functions */
            for (int order1 = 0; order1 < order; order1++) {
                auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);

                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = rf__(ir, idxrf, 0) * rf__(ir, idxrf1, 0);
                }

                /* <u_{\nu'}|u_{\nu}> */
                double ovlp = s.interpolate().integrate(0);

                if (response__) {
                    auto& drf = response__->radial_functions;
                    auto& dsd = response__->surface_derivatives;
                    for (int ir = 0; ir < nmtp; ir++) {
                        s(ir) = drf(ir, idxrf, 0) * rf__(ir, idxrf1, 0) + rf__(ir, idxrf, 0) * drf(ir, idxrf1, 0);
                    }
                    double dovlp = s.interpolate().integrate(0);
                    for (int ir = 0; ir < nmtp; ir++) {
                        for (int j : {0, 1}) {
                            drf(ir, idxrf, j) -= drf(ir, idxrf1, j) * ovlp + rf__(ir, idxrf1, j) * dovlp;
                        }
                    }
                    for (int i : {0, 1, 2}) {
                        dsd(i, idxrf) -= dsd(i, idxrf1) * ovlp + sd__(i, idxrf1) * dovlp;
                    }
                }
                for (int ir = 0; ir < nmtp; ir++) {
                    rf__(ir, idxrf, 0) -= rf__(ir, idxrf1, 0) * ovlp;
                    rf__(ir, idxrf, 1) -= rf__(ir, idxrf1, 1) * ovlp;
                }
                for (int i : {0, 1, 2}) {
                    sd__(i, idxrf) -= sd__(i, idxrf1) * ovlp;
                }
            }

            /* normalize orthogonal function */
            for (int ir = 0; ir < nmtp; ir++) {
                s(ir) = std::pow(rf__(ir, idxrf, 0), 2);
            }
            auto norm = s.interpolate().integrate(0);

            /* in case of linear dependency return failure */
            if (std::abs(norm) < 1e-8) {
                r.success = false;
                sinfo << "  linear dependent radial function" << std::endl << "  norm : " << norm << std::endl;
                r.error = sinfo.str();
                return r;
            }

            norm = 1.0 / std::sqrt(norm);

            if (response__) {
                auto& drf = response__->radial_functions;
                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = rf__(ir, idxrf, 0) * drf(ir, idxrf, 0);
                }
                double dnorm = -std::pow(norm, 3) * s.interpolate().integrate(0);
                for (int ir = 0; ir < nmtp; ir++) {
                    for (int j : {0, 1}) {
                        drf(ir, idxrf, j) = norm * drf(ir, idxrf, j) + dnorm * rf__(ir, idxrf, j);
                    }
                }
                for (int i : {0, 1, 2}) {
                    auto& dsd = response__->surface_derivatives(i, idxrf);
                    dsd       = norm * dsd + dnorm * sd__(i, idxrf);
                }
            }
            for (int ir = 0; ir < nmtp; ir++) {
                rf__(ir, idxrf, 0) *= norm;
                rf__(ir, idxrf, 1) *= norm;
            }
            /* aw radial function can't be zero at MT boundary */
            if (std::abs(rf__(nmtp - 1, idxrf, 0)) < 1e-2) {
                r.success = false;
                sinfo << "  radial function is zero at the muffin-tin boundary"
                      << "  value : " << rf__(nmtp - 1, idxrf, 0);
                r.error = sinfo.str();
                return r;
            }
            for (int i : {0, 1, 2}) {
                sd__(i, idxrf) *= norm;
            }
        } // order
        r.success = true;
        return r;
    };

    std::vector<std::string> errors(num_aw_descriptors());
    std::vector<int> status(num_aw_descriptors(), 0);

    #pragma omp parallel for schedule(dynamic, 1) default(shared)
    for (int l = 0; l < num_aw_descriptors(); l++) {
        /* take first radial function */
        auto rsd = aw_descriptor(l)[0];
        /* try to increase linearisation energy several times in order to find linearly independent
         * radial functions */
        compute_all_orders_result r;
        /* Enu for this level was searched; this should not produce degenerate radial functions */
        double e_shift{0.0};
        if (rsd.auto_enu) {
            r = compute_all_orders(l, e_shift);
        } else {
            /* for high l values, Enu is typically set in the species files and is not searched;
             * in case of trouble with them we need to increase linearisation energies */
            double de{0.1};
            for (int k = 0; k < 100; k++) {
                r = compute_all_orders(l, e_shift);
                // A degenerate boundary value permits an energy shift, but an
                // unresolved discrete XC operator must not be retried at another energy.
                if (r.success || (r.solver_failed && !spherical_xc_derivatives_[0].empty())) {
                    break;
                } else {
                    e_shift += de;
                    de *= 1.1;
                }
            }
        }

        if (r.success) {
            status[l] = 1;
            /* divide by r */
            for (int order = 0; order < (int)aw_descriptor(l).size(); order++) {
                auto idxrf = atom_type_.indexr().index_of(angular_momentum(l), order);
                for (int ir = 0; ir < nmtp; ir++) {
                    rf__(ir, idxrf, 0) *= atom_type_.radial_grid().x_inv(ir);
                    if (response__) {
                        response__->radial_functions(ir, idxrf, 0) *= atom_type_.radial_grid().x_inv(ir);
                    }
                }
            }
        } else {
            errors[l] = r.error;
        }
    } // l

    int ierr{0};
    std::stringstream s;
    s << "Atom_symmetry_class::generate_aw_radial_functions()" << std::endl
      << "  atom symmetry class id : " << id_ << std::endl
      << "  atom type label        : " << atom_type_.label() << std::endl
      << "  atom symbol            : " << atom_type_.symbol() << std::endl;
    for (int i = 0; i < num_aw_descriptors(); i++) {
        if (!status[i]) {
            ierr++;
            s << errors[i] << std::endl;
        }
    }
    if (ierr) {
        RTE_WARNING(s);
    }
    return ierr;
}

int
Atom_symmetry_class::generate_lo_radial_functions(relativity_t rel__, mdarray<double, 3>& rf__,
                                                  core_states_t const& core__, std::vector<double> const* direction__,
                                                  radial_basis_response_t* response__,
                                                  core_states_t const* core_response__,
                                                  enu_response_t const* enu_response__) const
{
    int nmtp = atom_type_.num_mt_points();

    bool found{true};
    std::vector<std::string> errors(num_lo_descriptors());

    #pragma omp parallel for schedule(dynamic, 1)
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
        Spline<double> s(atom_type_.radial_grid());
        double a[3][3];
        double rderiv[3][3];
        double da[3][3];

        /* number of radial functions */
        int num_rf = static_cast<int>(lo_descriptor(idxlo).rsd_set.size());
        RTE_ASSERT(num_rf <= 3);

        std::vector<std::vector<double>> u(num_rf);
        std::vector<std::vector<double>> rdudr(num_rf);
        std::vector<std::vector<double>> du(num_rf), drdudr(num_rf);

        for (int irf = 0; irf < num_rf; irf++) {
            auto rsd = lo_descriptor(idxlo).rsd_set[irf];

            radial_solver_result_t result;
            radial_solver_result_t derivative;
            try {
                double denu = enu_response__ && rsd.auto_enu ? enu_response__->at({rsd.n, rsd.l}) : 0;
                result = solve_radial(rel__, rsd.dme, rsd.l, rsd.enu, core__, direction__, &derivative, core_response__,
                                      denu);
            } catch (std::exception const& e) {
                errors[idxlo] = e.what();
                break;
            }

            u[irf]     = result.p;
            rdudr[irf] = result.rdudr;
            if (response__) {
                du[irf]     = derivative.p;
                drdudr[irf] = derivative.rdudr;
            }

            /* divide by r */
            for (int ir = 0; ir < nmtp; ir++) {
                /* store u(r) = p(r)/r */
                u[irf][ir] *= atom_type_.radial_grid().x_inv(ir);
                if (response__) {
                    du[irf][ir] *= atom_type_.radial_grid().x_inv(ir);
                }
            }

            for (int i = 0; i < num_rf; i++) {
                /* matrix of derivatives */
                a[irf][i] = rderiv[irf][i] = result.uderiv[i];
                if (response__) {
                    da[irf][i] = derivative.uderiv[i];
                }
            }
        }

        if (!errors[idxlo].empty()) {
            continue;
        }
        double b[]    = {0, 0, 0};
        b[num_rf - 1] = 1.0;

        int info = la::wrap(la::lib_t::lapack).gesv(num_rf, 1, &a[0][0], 3, b, 3);

        if (info) {
            std::stringstream s;
            s << "a[i][j] = ";
            for (int i = 0; i < num_rf; i++) {
                for (int j = 0; j < num_rf; j++) {
                    s << rderiv[i][j] << " ";
                }
            }
            s << std::endl;
            s << "atom: " << atom_type_.label() << std::endl
              << "zn: " << atom_type_.zn() << std::endl
              << "l: " << lo_descriptor(idxlo).am.l() << std::endl;
            s << "gesv returned " << info;
            RTE_THROW(s);
        }

        double db[] = {0, 0, 0};
        if (response__) {
            // Differentiate the boundary constraints A b = e: A db = -dA b.
            for (int i = 0; i < num_rf; i++) {
                for (int j = 0; j < num_rf; j++) {
                    db[i] -= da[j][i] * b[j];
                    a[j][i] = rderiv[j][i];
                }
            }
            if (la::wrap(la::lib_t::lapack).gesv(num_rf, 1, &a[0][0], 3, db, 3)) {
                RTE_THROW("local-orbital boundary response is singular");
            }
        }

        /* index of local orbital radial function */
        auto idxrf = atom_type_.indexr().index_of(rf_lo_index(idxlo));
        /* take linear combination of radial solutions */
        for (int order = 0; order < num_rf; order++) {
            for (int ir = 0; ir < nmtp; ir++) {
                /* u(r) function */
                rf__(ir, idxrf, 0) += b[order] * u[order][ir];
                /* r(du/dr) function */
                rf__(ir, idxrf, 1) += b[order] * rdudr[order][ir];
                if (response__) {
                    response__->radial_functions(ir, idxrf, 0) += b[order] * du[order][ir] + db[order] * u[order][ir];
                    response__->radial_functions(ir, idxrf, 1) +=
                            b[order] * drdudr[order][ir] + db[order] * rdudr[order][ir];
                }
            }
        }

        /* find norm of constructed local orbital */
        for (int ir = 0; ir < nmtp; ir++) {
            s(ir) = std::pow(rf__(ir, idxrf, 0), 2);
        }
        double norm = 1.0 / std::sqrt(s.interpolate().integrate(2));

        if (response__) {
            auto& drf = response__->radial_functions;
            for (int ir = 0; ir < nmtp; ir++) {
                s(ir) = rf__(ir, idxrf, 0) * drf(ir, idxrf, 0);
            }
            double dnorm = -std::pow(norm, 3) * s.interpolate().integrate(2);
            for (int ir = 0; ir < nmtp; ir++) {
                for (int j : {0, 1}) {
                    drf(ir, idxrf, j) = norm * drf(ir, idxrf, j) + dnorm * rf__(ir, idxrf, j);
                }
            }
        }
        /* normalize */
        for (int ir = 0; ir < nmtp; ir++) {
            rf__(ir, idxrf, 0) *= norm;
            rf__(ir, idxrf, 1) *= norm;
        }

        if (std::abs(rf__(nmtp - 1, idxrf, 0)) > 1e-10) {
            std::stringstream s;
            s << "local orbital " << idxlo << " is not zero at MT boundary" << std::endl
              << "  atom symmetry class id : " << id() << " (" << atom_type().symbol() << ")" << std::endl
              << "  value : " << rf__(nmtp - 1, idxrf, 0) << std::endl
              << "  number of MT points: " << nmtp << std::endl
              << "  MT radius: " << atom_type_.radial_grid().last() << std::endl
              << "  matrix of derivatives:" << std::endl;
            for (int i = 0; i < num_rf; i++) {
                for (int j = 0; j < num_rf; j++) {
                    s << rderiv[i][j] << " ";
                }
                s << std::endl;
            }
            s << "  b_coeffs: ";
            for (int j = 0; j < num_rf; j++) {
                s << b[j] << " ";
            }
            s << std::endl;
            s << "  norm: " << norm << std::endl;
            double d{0};
            for (int i = 0; i < num_rf; i++) {
                d += b[i] * rderiv[i][0];
            }
            s << "  expected value at MT boundary from the linear equations: " << d << std::endl;
            for (int i = 0; i < num_rf; i++) {
                s << " rderiv, u: " << rderiv[i][0] << " " << u[i][nmtp - 1] << std::endl;
            }
            RTE_WARNING(s);
        }
    }

    for (auto const& error : errors) {
        if (!error.empty()) {
            RTE_THROW(error);
        }
    }
    if (found && atom_type_.parameters().cfg().control().verification() > 0 && num_lo_descriptors() > 0) {
        check_lo_linear_independence(0.0001);
    }

    return found ? 0 : 1;
}

void
Atom_symmetry_class::orthogonalize_radial_functions(mdarray<double, 3>& rf__, mdarray<double, 3>* response__) const
{
    int nmtp = atom_type().num_mt_points();
    Spline<double> s(atom_type().radial_grid());
    /* orthogonalize local orbitals */
    for (int l = 0; l <= atom_type().indexr().lmax_lo(); l++) {
        for (int j = 0; j < atom_type().indexr().num_lo(l); j++) {
            int idxrf = atom_type().indexr().index_of(angular_momentum(l), j + atom_type_.aw_order(l));
            for (int j1 = 0; j1 < j; j1++) {
                int idxrf1 = atom_type().indexr().index_of(angular_momentum(l), j1 + atom_type().aw_order(l));

                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = rf__(ir, idxrf, 0) * rf__(ir, idxrf1, 0);
                }
                /* <u_{j}|u_{j'}> */
                double ovlp = s.interpolate().integrate(2);

                if (response__) {
                    auto& drf = *response__;
                    for (int ir = 0; ir < nmtp; ir++) {
                        s(ir) = drf(ir, idxrf, 0) * rf__(ir, idxrf1, 0) + rf__(ir, idxrf, 0) * drf(ir, idxrf1, 0);
                    }
                    double dovlp = s.interpolate().integrate(2);
                    for (int ir = 0; ir < nmtp; ir++) {
                        for (int j : {0, 1}) {
                            drf(ir, idxrf, j) -= drf(ir, idxrf1, j) * ovlp + rf__(ir, idxrf1, j) * dovlp;
                        }
                    }
                }
                for (int ir = 0; ir < nmtp; ir++) {
                    rf__(ir, idxrf, 0) -= rf__(ir, idxrf1, 0) * ovlp;
                    rf__(ir, idxrf, 1) -= rf__(ir, idxrf1, 1) * ovlp;
                }
            }
            /* normalize again */
            for (int ir = 0; ir < nmtp; ir++) {
                s(ir) = std::pow(rf__(ir, idxrf, 0), 2);
            }
            auto norm = s.interpolate().integrate(2);

            if (std::abs(norm) < 1e-10) {
                std::stringstream s;
                s << "LO radial function for atom " << atom_type_.label() << " is linearly dependent" << std::endl
                  << "  l : " << l << std::endl
                  << "  index for a given l : " << j;
            }

            norm = 1.0 / std::sqrt(norm);

            if (response__) {
                auto& drf = *response__;
                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = rf__(ir, idxrf, 0) * drf(ir, idxrf, 0);
                }
                double dnorm = -std::pow(norm, 3) * s.interpolate().integrate(2);
                for (int ir = 0; ir < nmtp; ir++) {
                    for (int j : {0, 1}) {
                        drf(ir, idxrf, j) = norm * drf(ir, idxrf, j) + dnorm * rf__(ir, idxrf, j);
                    }
                }
            }
            for (int ir = 0; ir < nmtp; ir++) {
                rf__(ir, idxrf, 0) *= norm;
                rf__(ir, idxrf, 1) *= norm;
            }
        }
    }
}

std::vector<int>
Atom_symmetry_class::check_lo_linear_independence(double tol__) const
{
    int nmtp = atom_type_.num_mt_points();

    Spline<double> s(atom_type_.radial_grid());
    la::dmatrix<double> loprod(num_lo_descriptors(), num_lo_descriptors());
    loprod.zero();
    for (int idxlo1 = 0; idxlo1 < num_lo_descriptors(); idxlo1++) {

        int idxrf1 = atom_type_.indexr().index_of(rf_lo_index(idxlo1));

        for (int idxlo2 = 0; idxlo2 < num_lo_descriptors(); idxlo2++) {

            int idxrf2 = atom_type_.indexr().index_of(rf_lo_index(idxlo2));

            if (lo_descriptor(idxlo1).am == lo_descriptor(idxlo2).am) {

                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                }
                loprod(idxlo1, idxlo2) = s.interpolate().integrate(2);
            }
        }
    }

    mdarray<double, 2> ovlp({num_lo_descriptors(), num_lo_descriptors()});
    copy(loprod, ovlp);

    auto stdevp = la::Eigensolver_factory("lapack");

    std::vector<double> loprod_eval(num_lo_descriptors());
    la::dmatrix<double> loprod_evec(num_lo_descriptors(), num_lo_descriptors());

    stdevp->solve(num_lo_descriptors(), loprod, &loprod_eval[0], loprod_evec);

    if (loprod_eval[0] < tol__) {
        std::printf("\n");
        std::printf("local orbitals for atom symmetry class %i are almost linearly dependent\n", id_);
        std::printf("local orbitals overlap matrix:\n");
        for (int i = 0; i < num_lo_descriptors(); i++) {
            for (int j = 0; j < num_lo_descriptors(); j++) {
                std::printf("%12.6f", ovlp(i, j));
            }
            std::printf("\n");
        }
        std::printf("overlap matrix eigen-values:\n");
        for (int i = 0; i < num_lo_descriptors(); i++) {
            std::printf("%12.6f", loprod_eval[i]);
        }
        std::printf("\n");
        std::printf("smallest eigenvalue: %20.16f\n", loprod_eval[0]);
    }

    std::vector<int> inc(num_lo_descriptors(), 0);

    /* try all local orbitals */
    for (int i = 0; i < num_lo_descriptors(); i++) {
        inc[i] = 1;

        std::vector<int> ilo;
        for (int j = 0; j < num_lo_descriptors(); j++) {
            if (inc[j] == 1) {
                ilo.push_back(j);
            }
        }

        std::vector<double> eval(ilo.size());
        la::dmatrix<double> evec(static_cast<int>(ilo.size()), static_cast<int>(ilo.size()));
        la::dmatrix<double> tmp(static_cast<int>(ilo.size()), static_cast<int>(ilo.size()));
        for (int j1 = 0; j1 < (int)ilo.size(); j1++) {
            for (int j2 = 0; j2 < (int)ilo.size(); j2++) {
                tmp(j1, j2) = ovlp(ilo[j1], ilo[j2]);
            }
        }

        stdevp->solve(static_cast<int>(ilo.size()), tmp, &eval[0], evec);

        if (eval[0] < tol__) {
            std::printf("local orbital %i can be removed\n", i);
            inc[i] = 0;
        }
    }
    return inc;
}

void
Atom_symmetry_class::set_spherical_potential(std::vector<double> const& vs__)
{
    if (atom_type_.num_mt_points() != (int)vs__.size()) {
        RTE_THROW("wrong size of effective potential array");
    }

    spherical_potential_ = vs__;
}

void
Atom_symmetry_class::set_spherical_xc_derivatives(std::array<std::vector<double>, 2> const& fields__)
{
    for (auto const& field : fields__) {
        if (field.size() != fields__[0].size() || (!field.empty() && field.size() != atom_type_.num_mt_points()) ||
            !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("invalid spherical XC derivatives");
        }
    }
    spherical_xc_derivatives_ = fields__;
}

radial_solver_result_t
Atom_symmetry_class::solve_radial(relativity_t rel__, int dme__, int l__, double energy__, core_states_t const& core__,
                                  std::vector<double> const* direction__, radial_solver_result_t* response__,
                                  core_states_t const* core_response__, double energy_direction__) const
{
    Radial_solver solver(atom_type_.zn(), spherical_potential_, atom_type_.radial_grid());
    radial_solver_result_t result;
    if (direction__) {
        if (rel__ != relativity_t::none || !response__) {
            RTE_THROW("radial basis response requires nonrelativistic functions");
        }
        auto direction = *direction__;
        for (auto& v : direction) {
            v -= energy_direction__;
        }
        auto radial = solver.solve_potential_response(dme__, l__, energy__, direction);
        *response__ = radial_solver_result_t{0, std::move(radial.p), std::move(radial.rdudr), radial.uderiv};
        result      = std::move(radial.value);
    } else {
        result = solver.solve(rel__, dme__, l__, energy__);
    }
    auto core = core__.find(l__);
    if (core != core__.end()) {
        if (direction__ && (!core_response__ || !core_response__->count(l__))) {
            RTE_THROW("radial basis response requires the occupied core response");
        }
        // The auxiliary local equation need not reproduce the discrete XC
        // operator. Its full matrix is assembled separately in the Hamiltonian.
        orthogonalize_radial_to_core(atom_type_.radial_grid(), core->second, result,
                                     direction__ ? &core_response__->at(l__) : nullptr,
                                     direction__ ? response__ : nullptr);
    }
    return result;
}

int
Atom_symmetry_class::find_enu(relativity_t rel__)
{
    PROFILE("sirius::Atom_symmetry_class::find_enu");

    /* unroll {n,l} -> enu map to enable omp for loop */
    std::vector<std::pair<std::pair<int, int>, enu_search_t>> nl_enu_vec(enu_search_.begin(), enu_search_.end());

    std::vector<std::string> errors(nl_enu_vec.size());
    std::vector<int> status(nl_enu_vec.size(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < nl_enu_vec.size(); i++) {
        int n = nl_enu_vec[i].first.first;
        int l = nl_enu_vec[i].first.second;
        try {
            nl_enu_vec[i].second = sirius::find_enu(rel__, atom_type_.zn(), n, l, atom_type_.radial_grid(),
                                                    spherical_potential_, nl_enu_vec[i].second);
            status[i]            = 1;
        } catch (std::exception const& e) {
            errors[i] = e.what();
        }
    }
    int ierr{0};
    std::stringstream s;
    s << "Atom_symmetry_class::find_enu()" << std::endl
      << "  atom symmetry class id : " << id_ << std::endl
      << "  atom type label        : " << atom_type_.label() << std::endl
      << "  atom symbol            : " << atom_type_.symbol() << std::endl;
    for (size_t i = 0; i < nl_enu_vec.size(); i++) {
        if (status[i]) {
            enu_search_[nl_enu_vec[i].first] = nl_enu_vec[i].second;
        } else {
            ierr++;
            s << errors[i] << std::endl;
        }
    }
    if (ierr) {
        RTE_WARNING(s);
    }

    /* update the {n,l} -> enu map */
    for (auto& e : nl_enu_vec) {
        enu_search_[e.first] = e.second;
    }

    update_enu_descriptors();
    return ierr;
}

void
Atom_symmetry_class::update_enu_descriptors()
{
    /* update AW linearization energies */
    for (int l = 0; l < num_aw_descriptors(); l++) {
        for (auto& d : aw_descriptor(l)) {
            if (d.auto_enu) {
                d.enu = enu_search_[{d.n, d.l}].enu;
            }
        }
    }
    /* update LO linearization energies */
    for (int idxlo = 0; idxlo < num_lo_descriptors(); idxlo++) {
        for (auto& d : lo_descriptor(idxlo).rsd_set) {
            if (d.auto_enu) {
                d.enu = enu_search_[{d.n, d.l}].enu;
            }
        }
    }
}

Atom_symmetry_class::core_states_t
Atom_symmetry_class::generate_core_radial_states(std::vector<double> const* direction__, core_states_t* response__,
                                                 std::array<std::vector<double>, 2> const& xc_directions__,
                                                 std::array<std::vector<double>, 2>* fields_response__) const
{
    core_states_t result;
    if (!spherical_xc_derivatives_[0].empty() && atom_type_.num_core_electrons() != 0) {
        auto [grid, potential] = atom_type_.core_radial_problem(spherical_potential_);
        std::vector<double> direction;
        if (direction__) {
            direction = *direction__;
            // The free-atom tail is shifted by the potential at the MT boundary.
            direction.resize(grid.num_points(), direction.back());
        }
        std::map<int, int> states_per_l;
        for (int i = 0; i < atom_type_.num_atomic_levels(); i++) {
            auto const& level = atom_type_.atomic_level(i);
            if (level.core && level.occupancy > 0) {
                states_per_l[level.l] = std::max(states_per_l[level.l], level.n - level.l);
            }
        }
        for (auto const& [l, count] : states_per_l) {
            std::vector<radial_core_state_t> response;
            auto states = solve_radial_core_states(
                    grid, l, atom_type_.zn(), potential, count, spherical_xc_derivatives_, atom_type_.num_mt_points(),
                    direction__ ? &direction : nullptr, direction__ ? &response : nullptr, xc_directions__);
            for (int j = 0; j < count; j++) {
                double occupancy{0};
                for (int i = 0; i < atom_type_.num_atomic_levels(); i++) {
                    auto const& level = atom_type_.atomic_level(i);
                    if (level.core && level.l == l && level.n - l - 1 == j) {
                        occupancy += level.occupancy;
                    }
                }
                if (occupancy > 0) {
                    if (states[j].energy >= potential.back()) {
                        RTE_THROW("tau-dependent core state is not bound within the extended radial domain");
                    }
                    result[l].push_back(std::move(states[j]));
                    if (fields_response__) {
                        for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
                            (*fields_response__)[0][ir] += occupancy * response[j].rho[ir];
                            (*fields_response__)[1][ir] += occupancy * response[j].tau[ir];
                        }
                    }
                    if (response__) {
                        (*response__)[l].push_back(std::move(response[j]));
                    }
                }
            }
        }
    }
    return result;
}

int
Atom_symmetry_class::generate_radial_functions(relativity_t rel__, bool update_enu__)
{
    PROFILE("sirius::Atom_symmetry_class::generate_radial_functions");

    mdarray<double, 2> sd({3, atom_type_.mt_radial_basis_size() - atom_type_.num_lo_descriptors()});
    mdarray<double, 3> rf({atom_type_.num_mt_points(), atom_type_.mt_radial_basis_size(), 2});
    sd.zero();
    rf.zero();

    if (!spherical_xc_derivatives_[0].empty() && atom_type_.num_core_electrons() != 0 && rel__ != relativity_t::none) {
        RTE_THROW("XC-dependent LAPW radial functions require nonrelativistic states");
    }
    auto core = generate_core_radial_states();

    if (update_enu__) {
        auto ierr_enu = this->find_enu(rel__);
        /* write spherical potential */
        if (ierr_enu && atom_type().parameters().cfg().control().save_rf()) {
            save_spherical_potential();
        }
    }

    auto ierr_aw = generate_aw_radial_functions(rel__, rf, sd, core);
    if (ierr_aw && !spherical_xc_derivatives_[0].empty()) {
        RTE_THROW("XC-dependent LAPW radial basis construction failed");
    }
    if (ierr_aw) {
        std::stringstream s;
        s << "generate_aw_radial_functions() failed for atom class " << id_;
        RTE_WARNING(s);
    }

    auto ierr_lo = generate_lo_radial_functions(rel__, rf, core);
    if (ierr_lo) {
        std::stringstream s;
        s << "generate_lo_radial_functions() failed for atom class " << id_;
        RTE_WARNING(s);
    }

    if (ierr_aw + ierr_lo == 0) {
        copy(rf, radial_functions_);
        copy(sd, surface_derivatives_);
        if (atom_type().parameters().cfg().control().ortho_rf()) {
            orthogonalize_radial_functions(radial_functions_);
        }
    } else {
        std::stringstream s;
        s << "radial functions for atom class " << id_ << " were not found";
        RTE_WARNING(s);
        if (atom_type().parameters().cfg().control().save_rf()) {
            save_spherical_potential();
        }
    }

    if (atom_type().parameters().cfg().control().save_rf()) {
        static int count{0};
        std::string fname =
                "radial_functions_class_" + std::to_string(id_) + "_step_" + std::to_string(count) + ".json";
        save_radial_functions(fname);
        count++;
    }
    return ierr_aw + ierr_lo;
}

radial_basis_response_t
Atom_symmetry_class::radial_basis_potential_response(std::vector<double> const& direction__,
                                                     std::array<std::vector<double>, 2> const& xc_directions__,
                                                     bool update_enu__) const
{
    if (atom_type_.parameters().valence_relativity() != relativity_t::none) {
        RTE_THROW("radial basis response requires nonrelativistic functions");
    }
    if (direction__.size() != static_cast<size_t>(atom_type_.num_mt_points()) ||
        spherical_potential_.size() != direction__.size() ||
        !std::all_of(direction__.begin(), direction__.end(), [](double x) { return std::isfinite(x); })) {
        RTE_THROW("invalid radial basis potential direction");
    }
    for (auto const& field : xc_directions__) {
        if (!field.empty() && (field.size() != direction__.size() ||
                               !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); }))) {
            RTE_THROW("invalid radial basis XC direction");
        }
        if (!field.empty() && atom_type_.num_core_electrons() != 0 && spherical_xc_derivatives_[0].empty()) {
            RTE_THROW("XC-dependent core response requires initialized core projection");
        }
    }
    mdarray<double, 3> rf({atom_type_.num_mt_points(), atom_type_.mt_radial_basis_size(), 2});
    mdarray<double, 2> sd({3, atom_type_.mt_radial_basis_size() - atom_type_.num_lo_descriptors()});
    rf.zero();
    sd.zero();
    radial_basis_response_t result(atom_type_);
    core_states_t dcore;
    auto core = generate_core_radial_states(&direction__, &dcore, xc_directions__, &result.core_fields);
    enu_response_t denu;
    if (update_enu__) {
        for (auto const& entry : enu_search_) {
            denu[entry.first] =
                    linearization_energy_response(atom_type_.zn(), entry.first.second, atom_type_.radial_grid(),
                                                  spherical_potential_, entry.second, direction__);
        }
    }
    if (generate_aw_radial_functions(relativity_t::none, rf, sd, core, &direction__, &result, &dcore,
                                     update_enu__ ? &denu : nullptr) ||
        generate_lo_radial_functions(relativity_t::none, rf, core, &direction__, &result, &dcore,
                                     update_enu__ ? &denu : nullptr)) {
        RTE_THROW("radial basis potential response construction failed");
    }
    if (atom_type_.parameters().cfg().control().ortho_rf()) {
        orthogonalize_radial_functions(rf, &result.radial_functions);
    }
    return result;
}

void
Atom_symmetry_class::sync_radial_functions(mpi::Communicator const& comm__, int const rank__)
{
    // Kinetic-density and XC matrix elements need both R and rR' on every rank.
    comm__.bcast(radial_functions_.at(memory_t::host), static_cast<int>(radial_functions_.size()), rank__);
    comm__.bcast(surface_derivatives_.at(memory_t::host), (int)surface_derivatives_.size(), rank__);
    // Response construction on non-owner ranks must use the same energies as the stored basis.
    std::vector<double> energies;
    for (auto const& entry : enu_search_) {
        auto const& e = entry.second;
        energies.insert(energies.end(), {e.enu, e.ebot, e.etop, e.etop_first_pass});
    }
    if (!energies.empty()) {
        comm__.bcast(energies.data(), static_cast<int>(energies.size()), rank__);
        size_t i{0};
        for (auto& entry : enu_search_) {
            auto& e           = entry.second;
            e.enu             = energies[i++];
            e.ebot            = energies[i++];
            e.etop            = energies[i++];
            e.etop_first_pass = energies[i++];
        }
        update_enu_descriptors();
    }
}

void
Atom_symmetry_class::sync_radial_integrals(mpi::Communicator const& comm__, int const rank__)
{
    comm__.bcast(h_spherical_integrals_.at(memory_t::host), (int)h_spherical_integrals_.size(), rank__);
    comm__.bcast(o_radial_integrals_.at(memory_t::host), (int)o_radial_integrals_.size(), rank__);
    comm__.bcast(so_radial_integrals_.at(memory_t::host), (int)so_radial_integrals_.size(), rank__);
    if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
        comm__.bcast(o1_radial_integrals_.at(memory_t::host), (int)o1_radial_integrals_.size(), rank__);
    }
}

void
Atom_symmetry_class::generate_radial_integrals(relativity_t rel__)
{
    PROFILE("sirius::Atom_symmetry_class::generate_radial_integrals");

    int nmtp = atom_type_.num_mt_points();

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);
    if (rel__ == relativity_t::none) {
        sq_alpha_half = 0;
    }

    h_spherical_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid());
        #pragma omp for
        for (int i1 = 0; i1 < atom_type_.mt_radial_basis_size(); i1++) {
            for (int i2 = 0; i2 < atom_type_.mt_radial_basis_size(); i2++) {
                /* for spherical part of potential integrals are diagonal in l */
                if (atom_type_.indexr(i1).am.l() == atom_type_.indexr(i2).am.l()) {
                    int ll = atom_type_.indexr(i1).am.l() * (atom_type_.indexr(i1).am.l() + 1);
                    for (int ir = 0; ir < nmtp; ir++) {
                        double Minv = 1.0 / (1 - spherical_potential_[ir] * sq_alpha_half);
                        /* u_1(r) * u_2(r) */
                        double t0 = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 0);
                        /* r*u'_1(r) * r*u'_2(r) */
                        double t1 = radial_functions_(ir, i1, 1) * radial_functions_(ir, i2, 1);
                        s(ir)     = 0.5 * t1 * Minv +
                                    t0 * (0.5 * ll * Minv +
                                          spherical_potential_[ir] * std::pow(atom_type_.radial_grid(ir), 2));
                    }
                    h_spherical_integrals_(i1, i2) = s.interpolate().integrate(0) / y00;
                }
            }
        }
    }

    o_radial_integrals_.zero();
    #pragma omp parallel default(shared)
    {
        Spline<double> s(atom_type_.radial_grid());
        #pragma omp for
        for (int l = 0; l <= atom_type_.indexr().lmax(); l++) {
            int nrf = atom_type_.indexr().max_order(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);
                for (int order2 = 0; order2 < nrf; order2++) {
                    auto idxrf2 = atom_type_.indexr().index_of(angular_momentum(l), order2);
                    if (order1 == order2) {
                        o_radial_integrals_(l, order1, order2) = 1.0;
                    } else {
                        if (atom_type_.parameters().cfg().settings().simple_lapw_ri()) {
                            for (int ir = 0; ir < nmtp; ir++) {
                                s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) *
                                        std::pow(atom_type_.radial_grid(ir), 2);
                            }
                            o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(0);
                        } else {
                            for (int ir = 0; ir < nmtp; ir++) {
                                s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                            }
                            o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(2);
                        }
                    }
                }
            }
        }
    }
    if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
        o1_radial_integrals_.zero();
        #pragma omp parallel for
        for (int i1 = 0; i1 < atom_type_.mt_radial_basis_size(); i1++) {
            Spline<double> s(atom_type_.radial_grid());
            for (int i2 = 0; i2 < atom_type_.mt_radial_basis_size(); i2++) {
                /* for spherical part of potential integrals are diagonal in l */
                if (atom_type_.indexr(i1).am.l() == atom_type_.indexr(i2).am.l()) {
                    int ll = atom_type_.indexr(i1).am.l() * (atom_type_.indexr(i1).am.l() + 1);
                    for (int ir = 0; ir < nmtp; ir++) {
                        double Minv2 = std::pow(1 - spherical_potential_[ir] * sq_alpha_half, -2);
                        /* u_1(r) * u_2(r) */
                        double t0 = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 0);
                        /* r*u'_1(r) * r*u'_2(r) */
                        double t1 = radial_functions_(ir, i1, 1) * radial_functions_(ir, i2, 1);
                        s(ir)     = sq_alpha_half * 0.5 * Minv2 * (t1 + t0 * ll);
                    }
                    o1_radial_integrals_(i1, i2) = s.interpolate().integrate(0);
                }
            }
        }
    }

    if (atom_type_.parameters().so_correction()) {
        double soc = std::pow(2 * speed_of_light, -2);

        Spline<double> s(atom_type_.radial_grid());
        Spline<double> s1(atom_type_.radial_grid());
        Spline<double> ve(atom_type_.radial_grid());

        for (int i = 0; i < nmtp; i++) {
            ve(i) = spherical_potential_[i] + atom_type_.zn() / atom_type_.radial_grid(i);
        }
        ve.interpolate();

        so_radial_integrals_.zero();
        for (int l = 0; l <= atom_type_.indexr().lmax(); l++) {
            int nrf = atom_type_.indexr().max_order(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);
                for (int order2 = 0; order2 < nrf; order2++) {
                    auto idxrf2 = atom_type_.indexr().index_of(angular_momentum(l), order2);

                    for (int ir = 0; ir < nmtp; ir++) {
                        double M = 1.0 - 2 * soc * spherical_potential_[ir];
                        /* first part <f| dVe / dr |f'> */
                        s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * soc *
                                ve.deriv(1, ir) / pow(M, 2);

                        /* second part <f| d(z/r) / dr |f'> */
                        s1(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * soc *
                                 atom_type_.zn() / pow(M, 2);
                    }
                    s.interpolate();
                    s1.interpolate();

                    so_radial_integrals_(l, order1, order2) = s.integrate(1) + s1.integrate(-1);
                }
            }
        }
    }
}

void
Atom_symmetry_class::save_spherical_potential() const
{
    nlohmann::json dict;
    dict["x"]                   = atom_type_.radial_grid().values();
    dict["z"]                   = atom_type_.zn();
    dict["rmt"]                 = atom_type_.mt_radius();
    dict["spherical_potential"] = spherical_potential_;
    write_json_to_file(dict, "spherical_potential_" + std::to_string(id_) + ".json");
}

void
Atom_symmetry_class::save_radial_functions(std::string const& fname__) const
{
    nlohmann::json dict;
    dict["x"]                   = atom_type_.radial_grid().values();
    dict["z"]                   = atom_type_.zn();
    dict["rmt"]                 = atom_type_.mt_radius();
    dict["spherical_potential"] = spherical_potential_;

    std::vector<double> veff(spherical_potential_.size());
    for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
        veff[ir] = spherical_potential_[ir] + atom_type_.zn() / atom_type_.radial_grid(ir);
    }
    dict["spherical_potential_el"] = veff;

    dict["radial_functions"] = nlohmann::json::array();
    for (int idxrf = 0; idxrf < atom_type_.indexr().size(); idxrf++) {
        std::vector<double> u(atom_type_.num_mt_points());
        std::vector<double> rdudr(atom_type_.num_mt_points());

        for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
            u[ir]     = radial_functions_(ir, idxrf, 0);
            rdudr[ir] = radial_functions_(ir, idxrf, 1);
        }

        auto const& rfd = atom_type_.indexr(idxrf);
        dict["radial_functions"].push_back({{"idxrf", idxrf},
                                            {"l", rfd.am.l()},
                                            {"order", rfd.order},
                                            {"idxlo", static_cast<int>(rfd.idxlo)},
                                            {"u", u},
                                            {"rdudr", rdudr}});
    }

    write_json_to_file(dict, fname__);
}

} // namespace sirius
