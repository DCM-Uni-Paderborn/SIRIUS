/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPHERIC_FUNCTION_INTERPOLATOR_HPP__
#define __SPHERIC_FUNCTION_INTERPOLATOR_HPP__

#include "function3d/spheric_function.hpp"
#include "radial/bspline.hpp"
#include <optional>

namespace sirius {

enum class spheric_inner_extension_t
{
    none,
    charge,
    kinetic
};

enum class spheric_interpolation_t
{
    spline,
    positive_spline
};

/// Sample an atom-local real-harmonic field and return derivatives to its radial samples.
/** Coordinates are Cartesian displacements from the source atom, in bohr. The
 *  caller supplies the displacements for every contributing periodic image.
 *  By default, values and gradients use the same not-a-knot spline, without independent
 *  interpolation of precomputed gradients. An optional inner continuation uses
 *  r^p times a quadratic, matching value and the first two radial derivatives
 *  at the first sample. Charge multipoles use p=l. Positive kinetic density can
 *  contain r^(l-1) terms from cusp/angular-gradient products (in particular a
 *  nonvanishing s/p dipole), so it uses p=max(l-1,0). This is a specified finite-
 *  grid continuation, not a substitute for setup/radial convergence. Exact
 *  nuclei remain excluded because cusp gradients need not have a unique limit.
 *  Optional radial powers factor Coulomb singularities out of each multipole
 *  before interpolation. Ordinary splines use a quadratic regular inner continuation.
 *  Sampling, its transpose and geometry derivatives retain the same factors.
 *  Positive B-spline sampling uses convex nodal weights and a C2 join to a
 *  constant regular inner continuation. With a common Coulomb power it
 *  preserves positivity at each angle without clipping.
 *  Outside-support truncation belongs to the reconstruction.
 */
class Spheric_function_interpolator
{
  private:
    Radial_grid<double> const& grid_;
    int lmax_;
    std::vector<Spline<double>> radial_;
    spheric_inner_extension_t inner_;
    spheric_interpolation_t interpolation_;
    std::vector<int> inner_power_;
    std::vector<double> radial_power_;
    std::optional<bspline_basis<4>> positive_basis_;
    struct control_t
    {
        int left;
        double fraction;
    };
    std::vector<control_t> positive_controls_;

    struct gradient_coupling_t
    {
        int from, to, angular;
        r3::vector<double> weight;
    };
    std::vector<gradient_coupling_t> gradient_couplings_;

    // Transform the spherical-gradient couplings used by gradient(Flm) to real
    // harmonics and Cartesian components. Retain the extra l shell for a Hessian.
    void
    build_gradient_couplings()
    {
        double isqrt2 = 1 / std::sqrt(2.0);
        for (int l = 0; l <= lmax_ + 1; l++) {
            for (int m = -l; m <= l; m++) {
                for (int lp : {l - 1, l + 1}) {
                    if (lp < 0) {
                        continue;
                    }
                    std::vector<r3::vector<double>> weights(2 * lp + 1);
                    for (int k = -std::abs(m); k <= std::abs(m); k += std::max(1, 2 * std::abs(m))) {
                        auto input = SHT::ylm_dot_rlm(l, k, m);
                        for (int mu = -1; mu <= 1; mu++) {
                            int kp = k + mu;
                            if (std::abs(kp) > lp) {
                                continue;
                            }
                            double d = lp > l ? std::sqrt(double(l + 1) / (2 * l + 3))
                                              : -std::sqrt(double(l) / (2 * l - 1));
                            d *= SHT::clebsch_gordan(l, 1, lp, k, mu, kp);
                            std::array<std::complex<double>, 3> cart{};
                            if (mu == 0) {
                                cart[2] = 1;
                            } else {
                                cart[0] = -mu * isqrt2;
                                cart[1] = std::complex<double>(0, isqrt2);
                            }
                            for (int mp = -std::abs(kp); mp <= std::abs(kp); mp += std::max(1, 2 * std::abs(kp))) {
                                auto factor = d * input * SHT::rlm_dot_ylm(lp, mp, kp);
                                for (int x = 0; x < 3; x++) {
                                    weights[mp + lp][x] += std::real(factor * cart[x]);
                                }
                            }
                        }
                    }
                    for (int mp = -lp; mp <= lp; mp++) {
                        auto w = weights[mp + lp];
                        if (w[0] != 0 || w[1] != 0 || w[2] != 0) {
                            gradient_couplings_.push_back({sf::lm(l, m), sf::lm(lp, mp), lp > l ? -l : l + 1, w});
                        }
                    }
                }
            }
        }
    }

    struct point_t
    {
        int segment;
        double dr;
        double radius;
        r3::vector<double> direction;
        std::vector<double> harmonics;
        mdarray<double, 2> gradient;
    };

    point_t
    point(r3::vector<double> position__, bool gradient__ = true) const
    {
        for (int x = 0; x < 3; x++) {
            if (!std::isfinite(position__[x])) {
                RTE_THROW("spherical interpolation: nonfinite coordinate");
            }
        }
        auto spherical   = r3::spherical_coordinates(position__);
        double r         = spherical[0];
        double tolerance = 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, grid_.last());
        if (r > grid_.last() && r - grid_.last() <= tolerance) {
            r = grid_.last();
        }
        if ((r < grid_.first() && inner_ == spheric_inner_extension_t::none) || r > grid_.last() || r < 1e-12) {
            RTE_THROW("spherical interpolation: point outside radial support or at the nucleus");
        }
        int segment = r < grid_.first() ? 0 : (r == grid_.last() ? grid_.num_points() - 2 : grid_.index_of(r));
        if (segment < 0) {
            RTE_THROW("spherical interpolation: radial segment not found");
        }
        int n = sf::lmmax(lmax_);
        point_t p{segment,
                  r - grid_[segment],
                  r,
                  (1.0 / r) * position__,
                  std::vector<double>(n),
                  mdarray<double, 2>({gradient__ ? n : 0, 3})};
        sf::spherical_harmonics(lmax_, position__, p.harmonics.data());
        if (gradient__) {
            sf::dRlm_dr(lmax_, position__, p.gradient);
        }
        return p;
    }

    // Weights on spline coefficients, shared by value/gradient evaluation and its exact transpose.
    std::array<std::array<double, 4>, 3>
    regular_radial_weights(point_t const& p__, int lm__) const
    {
        if (interpolation_ == spheric_interpolation_t::positive_spline) {
            if (p__.radius < grid_.first()) {
                return {{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
            }
            std::array<std::array<double, 4>, 3> weights{};
            for (int j = 0; j < 4; j++) {
                int i         = p__.segment + j;
                weights[0][j] = (*positive_basis_)(i, p__.radius);
                weights[1][j] = positive_basis_->deriv(i, p__.radius);
                weights[2][j] = positive_basis_->deriv2(i, p__.radius);
            }
            return weights;
        }
        double d = p__.dr;
        if (p__.radius >= grid_.first()) {
            return {{{1, d, d * d, d * d * d}, {0, 1, 2 * d, 3 * d * d}, {0, 0, 2, 6 * d}}};
        }
        double r0 = grid_.first();
        double u = p__.radius / r0, v = u - 1;
        int p     = inner_power_[lm__];
        double up = std::pow(u, p), dup = p == 0 ? 0 : p * std::pow(u, p - 1) / r0;
        double ddup = p < 2 ? 0 : p * (p - 1) * std::pow(u, p - 2) / (r0 * r0);
        std::array<double, 4> a{1 - p * v + 0.5 * p * (p + 1) * v * v, r0 * (v - p * v * v), r0 * r0 * v * v, 0};
        std::array<double, 4> da{(-p + p * (p + 1) * v) / r0, 1 - 2 * p * v, 2 * r0 * v, 0};
        std::array<double, 4> dda{p * (p + 1) / (r0 * r0), -2.0 * p / r0, 2, 0};
        std::array<std::array<double, 4>, 3> result{};
        for (int j = 0; j < 4; j++) {
            result[0][j] = up * a[j];
            result[1][j] = dup * a[j] + up * da[j];
            result[2][j] = ddup * a[j] + 2 * dup * da[j] + up * dda[j];
        }
        return result;
    }

    std::array<std::array<double, 4>, 3>
    radial_weights(point_t const& p__, int lm__) const
    {
        auto weights = regular_radial_weights(p__, lm__);
        double power = radial_power_[lm__];
        if (power == 0) {
            return weights;
        }
        double factor     = std::pow(p__.radius, power);
        double derivative = power * factor / p__.radius;
        double second     = power * (power - 1) * factor / (p__.radius * p__.radius);
        for (int j = 0; j < 4; j++) {
            double value = weights[0][j], first = weights[1][j];
            weights[0][j] = factor * value;
            weights[1][j] = factor * first + derivative * value;
            weights[2][j] = factor * weights[2][j] + 2 * derivative * first + second * value;
        }
        return weights;
    }

    Flm
    transpose_coefficients(mdarray<double, 3>& coeffs__) const
    {
        int n = static_cast<int>(radial_.size()), nr = grid_.num_points();
        Flm result(n, grid_);
        result.zero();
        for (int lm = 0; lm < n; lm++) {
            std::vector<double> values(nr, 0);
            if (interpolation_ == spheric_interpolation_t::positive_spline) {
                for (int ir = 0; ir < nr - 1; ir++) {
                    for (int j = 0; j < 4; j++) {
                        auto const& c = positive_controls_[ir + j];
                        values[c.left] += (1 - c.fraction) * coeffs__(ir, j, lm);
                        values[c.left + 1] += c.fraction * coeffs__(ir, j, lm);
                    }
                }
            } else {
                mdarray<double, 2> component({nr, 4}, &coeffs__(0, 0, lm));
                values = radial_[lm].interpolation_adjoint(component);
            }
            for (int ir = 0; ir < nr; ir++) {
                result(lm, ir) = values[ir] * std::pow(grid_[ir], -radial_power_[lm]);
            }
        }
        return result;
    }

    double
    coefficient(point_t const& p__, int lm__, int j__) const
    {
        if (interpolation_ == spheric_interpolation_t::positive_spline) {
            auto const& c = positive_controls_[p__.segment + j__];
            return (1 - c.fraction) * radial_[lm__](c.left) + c.fraction * radial_[lm__](c.left + 1);
        }
        return radial_[lm__].coeffs()(p__.segment, j__);
    }

  public:
    explicit Spheric_function_interpolator(Flm const& field__,
                                           spheric_inner_extension_t inner__       = spheric_inner_extension_t::none,
                                           std::vector<double> radial_power__      = {},
                                           spheric_interpolation_t interpolation__ = spheric_interpolation_t::spline)
        : grid_(field__.radial_grid())
        , lmax_(sf::lmax(field__.angular_domain_size()))
        , inner_(inner__)
        , interpolation_(interpolation__)
        , inner_power_(sf::l_by_lm(lmax_))
        , radial_power_(std::move(radial_power__))
    {
        if (grid_.num_points() < 4 || grid_.first() < 0 || sf::lmmax(lmax_) != field__.angular_domain_size()) {
            RTE_THROW("spherical interpolation: invalid grid or harmonic extent");
        }
        bool factored = !radial_power_.empty();
        if (factored &&
            (radial_power_.size() != static_cast<size_t>(field__.angular_domain_size()) || grid_.first() <= 0 ||
             !std::all_of(radial_power_.begin(), radial_power_.end(),
                          [](double power) { return std::isfinite(power); }))) {
            RTE_THROW("spherical interpolation: invalid radial powers");
        }
        radial_power_.resize(field__.angular_domain_size(), 0);
        if (interpolation_ == spheric_interpolation_t::positive_spline) {
            std::vector<double> knots(3, grid_.first());
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                knots.push_back(grid_[ir]);
            }
            knots.insert(knots.end(), 3, grid_.last());
            positive_basis_.emplace(std::move(knots));
            for (int i = 0; i < positive_basis_->size(); i++) {
                // Equal first controls give a C2 constant inner join, without extrapolating samples.
                if (i < 3) {
                    positive_controls_.push_back({0, 0});
                    continue;
                }
                if (i == positive_basis_->size() - 1) {
                    positive_controls_.push_back({grid_.num_points() - 2, 1});
                    continue;
                }
                // Greville controls interpolate the regular nodal samples with convex weights.
                double r =
                        (positive_basis_->knot(i + 1) + positive_basis_->knot(i + 2) + positive_basis_->knot(i + 3)) /
                        3;
                int left        = r >= grid_.last() ? grid_.num_points() - 2 : grid_.index_of(r);
                double fraction = (r - grid_[left]) / (grid_[left + 1] - grid_[left]);
                positive_controls_.push_back({left, fraction});
            }
        }
        for (int lm = 0; lm < field__.angular_domain_size(); lm++) {
            radial_.push_back(field__.component(lm));
            if (factored) {
                inner_power_[lm] = 0;
                for (int ir = 0; ir < grid_.num_points(); ir++) {
                    radial_.back()(ir) *= std::pow(grid_[ir], -radial_power_[lm]);
                }
                if (interpolation_ == spheric_interpolation_t::spline) {
                    radial_.back().interpolate();
                }
            } else if (inner_ == spheric_inner_extension_t::kinetic) {
                inner_power_[lm] = std::max(0, inner_power_[lm] - 1);
            }
        }
        build_gradient_couplings();
    }

    /// Sample the field without computing unused spatial derivatives.
    double
    value(r3::vector<double> const& position__) const
    {
        auto p = point(position__, false);
        double result{0};
        for (int lm = 0; lm < static_cast<int>(radial_.size()); lm++) {
            auto weights = radial_weights(p, lm);
            double f{0};
            for (int j = 0; j < 4; j++) {
                f += weights[0][j] * coefficient(p, lm, j);
            }
            result += f * p.harmonics[lm];
        }
        return result;
    }

    /// Return {field, d(field)/dx, d(field)/dy, d(field)/dz}.
    std::array<double, 4>
    value_gradient(r3::vector<double> const& position__) const
    {
        auto p = point(position__);
        std::array<double, 4> result{};
        for (int lm = 0; lm < static_cast<int>(radial_.size()); lm++) {
            auto weights = radial_weights(p, lm);
            double f{0}, df{0};
            for (int j = 0; j < 4; j++) {
                f += weights[0][j] * coefficient(p, lm, j);
                df += weights[1][j] * coefficient(p, lm, j);
            }
            result[0] += f * p.harmonics[lm];
            for (int x = 0; x < 3; x++) {
                result[x + 1] += df * p.direction[x] * p.harmonics[lm] + f * p.gradient(lm, x);
            }
        }
        return result;
    }

    /// Position direction of the same value/gradient map as position_adjoint.
    std::array<double, 4>
    position_response(r3::vector<double> const& position__, r3::vector<double> const& direction__) const
    {
        for (int x = 0; x < 3; x++) {
            if (!std::isfinite(direction__[x])) {
                RTE_THROW("spherical geometry response: nonfinite direction");
            }
        }
        std::array<double, 4> result{};
        for (int k = 0; k < 4; k++) {
            std::array<double, 4> unit{};
            unit[k]   = 1;
            result[k] = r3::dot(direction__, position_adjoint(position__, unit));
        }
        return result;
    }

    /// Cartesian position derivative of a raw value/gradient energy covector.
    /** Radial samples are held fixed. The gradient response uses the Hessian
     *  of the same spline and inner continuation, without interpolating a
     *  separately sampled derivative. Atom translations give the negative
     *  displacement derivative; periodic-image cell terms belong to the caller.
     */
    r3::vector<double>
    position_adjoint(r3::vector<double> const& position__, std::array<double, 4> const& covector__) const
    {
        for (double d : covector__) {
            if (!std::isfinite(d)) {
                RTE_THROW("spherical geometry adjoint: nonfinite covector");
            }
        }
        if (covector__[1] == 0 && covector__[2] == 0 && covector__[3] == 0) {
            auto v = value_gradient(position__);
            return covector__[0] * r3::vector<double>({v[1], v[2], v[3]});
        }
        auto p = point(position__);
        int n  = static_cast<int>(radial_.size());
        std::vector<std::array<double, 3>> radial(n);
        for (int lm = 0; lm < n; lm++) {
            auto weights = radial_weights(p, lm);
            for (int a = 0; a < 3; a++) {
                for (int j = 0; j < 4; j++) {
                    radial[lm][a] += weights[a][j] * coefficient(p, lm, j);
                }
            }
        }
        std::vector<r3::vector<double>> grad(sf::lmmax(lmax_ + 1)), dgrad(grad.size());
        for (auto const& c : gradient_couplings_) {
            if (c.from < n) {
                auto f = radial[c.from];
                grad[c.to] += (f[1] + c.angular * f[0] / p.radius) * c.weight;
                dgrad[c.to] += (f[2] + c.angular * (f[1] - f[0] / p.radius) / p.radius) * c.weight;
            }
        }
        std::vector<double> harmonics(sf::lmmax(lmax_ + 2));
        sf::spherical_harmonics(lmax_ + 2, position__, harmonics.data());
        r3::vector<double> result, dg({covector__[1], covector__[2], covector__[3]});
        for (size_t lm = 0; lm < grad.size(); lm++) {
            result += (covector__[0] * harmonics[lm]) * grad[lm];
        }
        for (auto const& c : gradient_couplings_) {
            double value = r3::dot(dg, dgrad[c.from] + (c.angular / p.radius) * grad[c.from]);
            result += (value * harmonics[c.to]) * c.weight;
        }
        return result;
    }

    /// Euclidean transpose of the combined value/Cartesian-gradient sampling.
    /** dvalues__ contains derivatives of the complete energy, including any
     *  integration/partition weights already applied by the evaluator. The
     *  returned entries are dE/d(field_lm(r_i)), not a radial potential to be
     *  integrated again. Duplicate points/images add, and empty point lists
     *  return zero. The original radial grid must outlive this object/results.
     */
    Flm
    adjoint(std::vector<r3::vector<double>> const& positions__,
            std::vector<std::array<double, 4>> const& dvalues__) const
    {
        if (positions__.size() != dvalues__.size()) {
            RTE_THROW("spherical interpolation adjoint: incompatible point derivatives");
        }
        int n  = static_cast<int>(radial_.size());
        int nr = grid_.num_points();
        mdarray<double, 3> coeffs({nr, 4, n});
        coeffs.zero();
        for (size_t i = 0; i < positions__.size(); i++) {
            auto p = point(positions__[i]);
            for (int lm = 0; lm < n; lm++) {
                double f = dvalues__[i][0] * p.harmonics[lm];
                double df{0};
                for (int x = 0; x < 3; x++) {
                    f += dvalues__[i][x + 1] * p.gradient(lm, x);
                    df += dvalues__[i][x + 1] * p.direction[x] * p.harmonics[lm];
                }
                auto weights = radial_weights(p, lm);
                for (int j = 0; j < 4; j++) {
                    coeffs(p.segment, j, lm) += weights[0][j] * f + weights[1][j] * df;
                }
            }
        }
        return transpose_coefficients(coeffs);
    }

    /// Position derivative of the radial-sample adjoint, at fixed raw covectors and grid.
    Flm
    geometry_adjoint_response(std::vector<r3::vector<double>> const& positions__,
                              std::vector<std::array<double, 4>> const& dvalues__,
                              std::vector<r3::vector<double>> const& directions__) const
    {
        if (positions__.size() != dvalues__.size() || positions__.size() != directions__.size()) {
            RTE_THROW("spherical adjoint response: incompatible point directions");
        }
        int n = static_cast<int>(radial_.size()), nr = grid_.num_points();
        mdarray<double, 3> coeffs({nr, 4, n});
        coeffs.zero();
        std::vector<std::pair<gradient_coupling_t const*, gradient_coupling_t const*>> couplings;
        for (auto const& a : gradient_couplings_) {
            if (a.from < n && a.angular != 0) {
                for (auto const& b : gradient_couplings_) {
                    if (b.from == a.to && b.angular != 1) {
                        couplings.emplace_back(&a, &b);
                    }
                }
            }
        }
        for (size_t i = 0; i < positions__.size(); i++) {
            auto p        = point(positions__[i]);
            auto const& v = directions__[i];
            for (double d : dvalues__[i]) {
                if (!std::isfinite(d)) {
                    RTE_THROW("spherical adjoint response: nonfinite covector");
                }
            }
            for (int x = 0; x < 3; x++) {
                if (!std::isfinite(v[x])) {
                    RTE_THROW("spherical adjoint response: nonfinite direction");
                }
            }
            double dr = r3::dot(p.direction, v);
            auto dn   = (1.0 / p.radius) * (v - dr * p.direction);
            std::vector<double> harmonics(sf::lmmax(lmax_ + 2));
            sf::spherical_harmonics(lmax_ + 2, positions__[i], harmonics.data());
            std::vector<r3::vector<double>> dgradient(n);
            // Apply the two angular-gradient couplings to each original harmonic.
            for (auto const& [a, b] : couplings) {
                double w = a->angular * (b->angular - 1.0) * harmonics[b->to] / (p.radius * p.radius);
                dgradient[a->from] += (w * r3::dot(b->weight, v)) * a->weight;
            }
            for (int lm = 0; lm < n; lm++) {
                double dy{0}, f = dvalues__[i][0] * p.harmonics[lm], df{0};
                for (int x = 0; x < 3; x++) {
                    dy += p.gradient(lm, x) * v[x];
                    f += dvalues__[i][x + 1] * p.gradient(lm, x);
                    df += dvalues__[i][x + 1] * p.direction[x] * p.harmonics[lm];
                }
                double delta_f = dvalues__[i][0] * dy, delta_df{0};
                for (int x = 0; x < 3; x++) {
                    delta_f += dvalues__[i][x + 1] * dgradient[lm][x];
                    delta_df += dvalues__[i][x + 1] * (dn[x] * p.harmonics[lm] + p.direction[x] * dy);
                }
                auto weights = radial_weights(p, lm);
                for (int j = 0; j < 4; j++) {
                    coeffs(p.segment, j, lm) += dr * (weights[1][j] * f + weights[2][j] * df) +
                                                weights[0][j] * delta_f + weights[1][j] * delta_df;
                }
            }
        }
        return transpose_coefficients(coeffs);
    }
};

} // namespace sirius

#endif
