/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __RADIAL_CORE_SOLVER_HPP__
#define __RADIAL_CORE_SOLVER_HPP__

#include "bspline.hpp"
#include "spline.hpp"
#include "radial_solver.hpp"
#include "core/constants.hpp"
#include "core/la/eigensolver.hpp"
#include "core/la/linalg.hpp"
#include <memory>
#include <mutex>

namespace sirius {

/// Scalar l channel, or Dirac kappa for k=j+1/2.
inline int
radial_core_channel(int l__, int k__, relativity_t relativity__)
{
    if (relativity__ != relativity_t::dirac) {
        return l__;
    }
    if (l__ < 0 || k__ < 1 || (k__ != l__ && k__ - 1 != l__)) {
        RTE_THROW("Dirac core: inconsistent orbital and total angular momentum");
    }
    return k__ == l__ ? k__ : -k__;
}

struct radial_core_state_t
{
    double energy;
    double kinetic_energy;
    std::vector<double> coefficients;
    /// Signed radial orbital R_l and its radial derivative.
    std::vector<double> value;
    std::vector<double> derivative;
    /// Lower Dirac component Q/r and its derivative; empty for scalar orbitals.
    std::vector<double> small_value;
    std::vector<double> small_derivative;
    /// Spherical averages for one electron, without occupation or radial Jacobians.
    std::vector<double> rho;
    std::vector<double> tau;
    /// Zero for scalar orbitals, otherwise the signed Dirac angular channel.
    int kappa{0};
};

struct radial_core_response_t
{
    radial_core_state_t value;
    radial_core_state_t derivative;
};

/// Exact reference states for repeated linear responses at one radial operator.
struct radial_core_reference_t
{
    int l{-1}, zn{0}, count{0}, nmtp{0}, basis{0};
    relativity_t relativity{relativity_t::none};
    std::vector<double> grid;
    std::vector<double> potential;
    std::array<std::vector<double>, 2> fields;
    std::vector<radial_core_state_t> states;
};

/// Nonrelativistic or scalar-ZORA radial generalized-Kohn-Sham solver with discrete density/tau derivatives.
/** The regular radial orbital is expanded as R_l(r) = r^s sum_i c_i B_i(r).
 *  For ZORA at a Coulomb nucleus, s=sqrt(1+l(l+1)-(Z/c)^2)-1; otherwise s=l.
 *  Dropping the last B-spline imposes a vanishing orbital at the outer boundary.
 *  Retaining it permits a regular boundary solution for LAPW radial functions.
 *  The overlap and bare kinetic/local-potential operators use Gauss-Legendre
 *  integration on the knot intervals. XC inputs are raw derivatives with respect
 *  to the sampled spherical-average rho and positive tau. Their interpolation,
 *  angular integration and radial weights must already have been differentiated
 *  by the functional. No additional quadrature weights are applied here.
 *
 *  The radial potential includes -Z/r. Only its regular remainder is splined.
 *  A zero basis size selects the finite radial space defined by the stored field
 *  grid, grading the unsampled-origin transition and including its extended tail.
 *  Independent refinement at fixed nodal XC
 *  covectors can create spurious point-localized kinetic directions. Positive
 *  basis sizes retain logarithmic knots for auxiliary radial boundary problems
 *  and controlled analytic tests. ZORA uses the bare local potential in its
 *  inverse mass; the XC tau covector acts on the unweighted positive orbital tau.
 *  A finite nonzero XC kinetic coefficient can regularize the ZORA origin.
 *  Such problems use the optional r^l origin rather than the bare Coulomb power.
 *  Energy-dependent masses and Dirac spinors are not covered.
 */
class Radial_core_solver
{
    friend class Radial_dirac_core_solver;

  private:
    static constexpr int order{6};
    bspline_basis<order> basis_;
    int l_;
    Radial_grid<double> const& grid_;
    double zn_;
    relativity_t relativity_;
    double radial_power_;
    Spline<double> regular_potential_;
    la::dmatrix<double> overlap_;
    la::dmatrix<double> kinetic_;
    la::dmatrix<double> h0_;

    struct sample_t
    {
        int index;
        double value;
        double derivative;
        double angular;
    };
    std::vector<std::vector<sample_t>> samples_;

    double
    inverse_mass(double potential__) const
    {
        if (relativity_ == relativity_t::none) {
            return 1;
        }
        double mass = radial_solver_local::rel_mass<relativity_t::zora>(0, potential__);
        if (!std::isfinite(mass) || mass <= 0) {
            RTE_THROW("radial core solver: nonpositive or nonfinite ZORA mass");
        }
        return 1 / mass;
    }

    static std::vector<double>
    knots(Radial_grid<double> const& grid__, int num_basis__, double mt_radius__, double zn__,
          std::vector<double> const& extra__, relativity_t relativity__)
    {
        if (!std::isfinite(zn__) || zn__ < 0 || (num_basis__ != 0 && num_basis__ <= order) || grid__.num_points() < 4 ||
            grid__.first() <= 0 || !std::isfinite(grid__.last()) || grid__.last() <= grid__.first()) {
            RTE_THROW("radial core solver: invalid basis or field grid");
        }
        std::vector<double> result;
        if (num_basis__ == 0) {
            result.assign(order, 0);
            auto points = grid__.values();
            // The C2 inner field continuation spans [0, r_first]. Abruptly smaller
            // orbital intervals amplify its nodal derivatives into localized kinetic modes.
            double last = 0, interval = points.front();
            for (auto point = points.begin(); point != points.end() - 1; ++point) {
                if (*point - last >= 0.5 * interval) {
                    interval = *point - last;
                    last     = *point;
                    result.push_back(*point);
                }
            }
            result.insert(result.end(), order, grid__.last());
        } else {
            result.assign(num_basis__ + order, 0);
            double scale = 1 / std::max(1.0, zn__);
            if (relativity__ == relativity_t::zora && zn__ > 0) {
                scale = std::min(scale, sq_alpha_half * zn__);
            }
            double extent = std::log1p(grid__.last() / scale);
            for (int i = 1; i <= num_basis__ - order; i++) {
                result[order + i - 1] = scale * std::expm1(extent * i / (num_basis__ - order + 1));
            }
            std::fill(result.begin() + num_basis__, result.end(), grid__.last());
            if (relativity__ == relativity_t::zora && zn__ > 0) {
                // Resolve the first sampled derivative independently of the Coulomb mass scale.
                double first_interval = result[order];
                for (double r = grid__.first(); r < first_interval; r *= 2) {
                    result.insert(std::lower_bound(result.begin(), result.end(), r), r);
                }
            }
        }
        for (double r : extra__) {
            if (!std::isfinite(r) || r <= 0 || r >= grid__.last()) {
                RTE_THROW("radial core solver: invalid refinement knot");
            }
            auto position = std::lower_bound(result.begin(), result.end(), r);
            if (position == result.end() || *position != r) {
                result.insert(position, r);
            }
        }
        if (mt_radius__ != 0) {
            if (!std::isfinite(mt_radius__) || mt_radius__ <= grid__.first() || mt_radius__ >= grid__.last()) {
                RTE_THROW("radial core solver: invalid response boundary");
            }
            // The orbital is continuous at the MT boundary, but its derivative
            // need not be when the tau-dependent operator stops there.
            auto first = std::lower_bound(result.begin(), result.end(), mt_radius__);
            auto last  = std::upper_bound(first, result.end(), mt_radius__);
            first      = result.erase(first, last);
            result.insert(first, order - 1, mt_radius__);
        }
        return result;
    }

    void
    check_covectors(std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        for (auto const* field : {&drho__, &dtau__}) {
            if (field->size() != samples_.size() ||
                !std::all_of(field->begin(), field->end(), [](double v) { return std::isfinite(v); })) {
                RTE_THROW("radial core solver: invalid density/tau covector");
            }
        }
    }

    void
    add_xc(la::dmatrix<double>& h__, std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        check_covectors(drho__, dtau__);
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            for (auto const& a : samples_[ir]) {
                for (auto const& b : samples_[ir]) {
                    h__(a.index, b.index) +=
                            (drho__[ir] * a.value * b.value +
                             0.5 * dtau__[ir] * (a.derivative * b.derivative + a.angular * b.angular)) /
                            fourpi;
                }
            }
        }
    }

    la::dmatrix<double>
    hamiltonian(std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        la::dmatrix<double> h(size(), size());
        copy(h0_, h);
        add_xc(h, drho__, dtau__);
        return h;
    }

    double
    contract(la::dmatrix<double> const& matrix__, std::vector<double> const& a__, std::vector<double> const& b__) const
    {
        long double result{0};
        for (int j = 0; j < size(); j++) {
            for (int i = std::max(0, j - order + 1); i <= std::min(size() - 1, j + order - 1); i++) {
                result += static_cast<long double>(a__[i]) * matrix__(i, j) * b__[j];
            }
        }
        return result;
    }

    double
    quadratic(la::dmatrix<double> const& matrix__, std::vector<double> const& c__) const
    {
        return contract(matrix__, c__, c__);
    }

    void
    check_coefficients(std::vector<double> const& coefficients__) const
    {
        if (coefficients__.size() != static_cast<size_t>(size()) ||
            !std::all_of(coefficients__.begin(), coefficients__.end(), [](double v) { return std::isfinite(v); })) {
            RTE_THROW("radial core solver: invalid orbital coefficients");
        }
    }

    /// Solve the normalized Sternheimer system by a banded minor and a 2x2 border.
    std::vector<double>
    solve_constrained(la::dmatrix<double> const& h__, double energy__, std::vector<double> const& c__,
                      std::vector<double> rhs__) const
    {
        int n = size();
        std::vector<double> scale(n + 1), sc(n);
        double border_norm{0};
        for (int i = 0; i < n; i++) {
            scale[i] = 1 / std::sqrt(std::max({std::abs(h__(i, i)), std::abs(energy__ * overlap_(i, i)), 1e-30}));
            long double product{0};
            for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                product += static_cast<long double>(overlap_(i, j)) * c__[j];
            }
            sc[i] = product;
            border_norm += std::pow(scale[i] * sc[i], 2);
        }
        if (!std::isfinite(border_norm) || border_norm <= 0) {
            RTE_THROW("radial core response: invalid normalization constraint");
        }
        scale[n] = 1 / std::sqrt(border_norm);
        rhs__.resize(n + 1, 0);
        int pivot{0};
        for (int i = 0; i < n; i++) {
            rhs__[i] *= scale[i];
            sc[i] *= scale[i] * scale[n];
            if (std::abs(sc[i]) > std::abs(sc[pivot]))
                pivot = i;
        }
        int m = n - 1, bandwidth = order - 1, leading = 3 * bandwidth + 1, nrhs = 3, info;
        std::vector<double> band(leading * m, 0), vectors(3 * m);
        std::vector<ftn_int> ipiv(m);
        auto index = [pivot](int i) { return i < pivot ? i : i + 1; };
        auto a     = [&](int i, int j) { return scale[i] * (h__(i, j) - energy__ * overlap_(i, j)) * scale[j]; };
        for (int j = 0; j < m; j++) {
            for (int i = std::max(0, j - bandwidth); i <= std::min(m - 1, j + bandwidth); i++) {
                band[2 * bandwidth + i - j + leading * j] = a(index(i), index(j));
            }
            vectors[j]         = rhs__[index(j)];
            vectors[j + m]     = a(index(j), pivot);
            vectors[j + 2 * m] = sc[index(j)];
        }
        FORTRAN(dgbsv)
        (&m, &bandwidth, &bandwidth, &nrhs, band.data(), &leading, ipiv.data(), vectors.data(), &m, &info);
        if (info) {
            RTE_THROW("radial core response: singular banded principal minor");
        }
        long double s00 = a(pivot, pivot), s01 = sc[pivot], s11{0}, f0 = rhs__[pivot], f1{0};
        for (int i = 0; i < m; i++) {
            double u = a(index(i), pivot), v = sc[index(i)];
            s00 -= static_cast<long double>(u) * vectors[i + m];
            s01 -= static_cast<long double>(u) * vectors[i + 2 * m];
            s11 -= static_cast<long double>(v) * vectors[i + 2 * m];
            f0 -= static_cast<long double>(u) * vectors[i];
            f1 -= static_cast<long double>(v) * vectors[i];
        }
        long double determinant = s00 * s11 - s01 * s01;
        double xp = (f0 * s11 - f1 * s01) / determinant, multiplier = (f1 * s00 - f0 * s01) / determinant;
        for (int i = 0; i < m; i++) {
            rhs__[index(i)] = vectors[i] - vectors[i + m] * xp - vectors[i + 2 * m] * multiplier;
        }
        rhs__[pivot] = xp;
        rhs__[n]     = multiplier;
        if (info || !std::all_of(rhs__.begin(), rhs__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("radial core response: singular or nonfinite eigenstate response");
        }
        for (int i = 0; i <= n; i++) {
            rhs__[i] *= scale[i];
        }
        return rhs__;
    }

  public:
    Radial_core_solver(Radial_grid<double> const& grid__, int l__, double zn__, std::vector<double> const& potential__,
                       int num_basis__, double mt_radius__ = 0, bool include_boundary__ = false,
                       std::vector<double> const& extra_knots__ = {}, relativity_t relativity__ = relativity_t::none,
                       bool regular_origin__ = false)
        // The fractional ZORA origin power requires finer quadrature than r^l.
        : basis_(knots(grid__, num_basis__, mt_radius__, zn__, extra_knots__,
                       regular_origin__ ? relativity_t::none : relativity__),
                 relativity__ == relativity_t::zora ? 32 : 16, include_boundary__,
                 !regular_origin__ && relativity__ == relativity_t::zora && zn__ > 0
                         ? std::max(1.0, 4 / radial_solver_local::scalar_relativistic_origin_power(l__, zn__))
                         : 1)
        , l_(l__)
        , grid_(grid__)
        , zn_(zn__)
        , relativity_(relativity__)
        , radial_power_(!regular_origin__ && relativity__ == relativity_t::zora
                                ? radial_solver_local::scalar_relativistic_origin_power(l__, zn__) - 1
                                : l__)
        , regular_potential_(grid__)
        , overlap_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , kinetic_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , h0_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , samples_(grid__.num_points())
    {
        if (l_ < 0 || !std::isfinite(zn__) || zn__ < 0 || potential__.size() != samples_.size()) {
            RTE_THROW("radial core solver: invalid angular momentum, charge or potential");
        }
        if (relativity_ != relativity_t::none && relativity_ != relativity_t::zora) {
            RTE_THROW("radial core solver: unsupported relativistic equation");
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            if (!std::isfinite(potential__[ir]) || !std::isfinite(grid__[ir]) || (ir && grid__[ir] <= grid__[ir - 1])) {
                RTE_THROW("radial core solver: nonfinite potential or invalid radial grid");
            }
            regular_potential_(ir) = potential__[ir] + zn__ / grid__[ir];
            if (!std::isfinite(regular_potential_(ir))) {
                RTE_THROW("radial core solver: nonfinite regular potential");
            }
            inverse_mass(potential__[ir]);
        }
        regular_potential_.interpolate();
        overlap_.zero();
        kinetic_.zero();
        h0_.zero();
        for (auto const& p : basis_.basis_pairs()) {
            for (size_t q = 0; q < p.r.size(); q++) {
                double r = p.r[q], rl = std::pow(r, radial_power_);
                double ri = rl * p.Bi[q], rj = rl * p.Bj[q];
                double di        = rl * (p.dBi[q] + radial_power_ * p.Bi[q] / r);
                double dj        = rl * (p.dBj[q] + radial_power_ * p.Bj[q] / r);
                double potential = regular_potential_.at_point(std::max(r, grid__.first())) - zn__ / r;
                double weight    = p.w[q] * r * r;
                double t = 0.5 * weight * inverse_mass(potential) * (di * dj + l_ * (l_ + 1.0) * ri * rj / (r * r));
                overlap_(p.i, p.j) += weight * ri * rj;
                kinetic_(p.i, p.j) += t;
                h0_(p.i, p.j) += t + weight * potential * ri * rj;
            }
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            double r = grid__[ir], rl = std::pow(r, radial_power_);
            double sample_r = r == mt_radius__ ? std::nextafter(r, 0.0) : r;
            for (int i = 0; i < size(); i++) {
                if (sample_r >= basis_.knot(i) && sample_r <= basis_.knot(i + order)) {
                    double b = basis_(i, sample_r);
                    samples_[ir].push_back({i, rl * b, rl * (basis_.deriv(i, sample_r) + radial_power_ * b / r),
                                            std::sqrt(l_ * (l_ + 1.0)) * rl * b / r});
                }
            }
        }
    }

    int
    size() const
    {
        return overlap_.num_rows();
    }

    /// Scalar product in the radial orbital space, using its Gauss-integrated overlap.
    double
    overlap(std::vector<double> const& a__, std::vector<double> const& b__) const
    {
        check_coefficients(a__);
        check_coefficients(b__);
        return contract(overlap_, a__, b__);
    }

    std::vector<double>
    breakpoints() const
    {
        std::vector<double> result;
        for (int i = 0; i < basis_.num_knots(); i++) {
            double r = basis_.knot(i);
            if (result.empty() || result.back() != r) {
                result.push_back(r);
            }
        }
        return result;
    }

    /// Regular radial solution with u(R)=1 and its analytic energy derivatives.
    radial_solver_result_t
    solve_at_energy(int dme__, double energy__, std::vector<double> const& drho__,
                    std::vector<double> const& dtau__) const
    {
        if (size() != basis_.size() || dme__ < 0 || !std::isfinite(energy__)) {
            RTE_THROW("radial boundary solve requires an open boundary and a finite energy");
        }
        auto h = hamiltonian(drho__, dtau__);
        int n = size(), interior = n - 1;
        std::vector<double> scale(n), c(n), rhs(interior);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(overlap_(i, i)) || overlap_(i, i) <= 0) {
                RTE_THROW("radial boundary solve: singular overlap");
            }
            scale[i] = 1 / std::sqrt(overlap_(i, i));
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                h(i, j) = (h(i, j) - energy__ * overlap_(i, j)) * scale[i] * scale[j];
            }
        }
        double boundary = std::pow(grid_.last(), -radial_power_) / scale.back();
        for (int order = 0; order <= dme__; order++) {
            la::dmatrix<double> a(interior, interior);
            for (int i = 0; i < interior; i++) {
                rhs[i] = order ? 0 : -h(i, interior) * boundary;
                if (order) {
                    for (int j = 0; j < n; j++) {
                        rhs[i] += order * overlap_(i, j) * scale[i] * scale[j] * c[j];
                    }
                }
                for (int j = 0; j < interior; j++) {
                    a(i, j) = h(i, j);
                }
            }
            int info =
                    la::wrap(la::lib_t::lapack).gesv(interior, 1, a.at(memory_t::host), interior, rhs.data(), interior);
            if (info) {
                RTE_THROW("radial boundary solve failed: " + std::to_string(info));
            }
            if (!std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); })) {
                RTE_THROW("radial boundary solve returned nonfinite coefficients");
            }
            std::copy(rhs.begin(), rhs.end(), c.begin());
            c.back() = order ? 0 : boundary;
        }
        radial_solver_result_t result{
                0, std::vector<double>(samples_.size()), std::vector<double>(samples_.size()), {}};
        Spline<double> derivative(grid_);
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            long double u{0}, du{0};
            for (auto const& s : samples_[ir]) {
                long double coefficient = static_cast<long double>(scale[s.index]) * c[s.index];
                u += coefficient * s.value;
                du += coefficient * s.derivative;
            }
            result.p[ir]     = grid_[ir] * u;
            result.rdudr[ir] = grid_[ir] * du;
            derivative(ir)   = du;
            if (ir && result.p[ir] * result.p[ir - 1] < 0) {
                result.num_nodes++;
            }
        }
        result.uderiv = {result.p.back() / grid_.last(), derivative(grid_.num_points() - 1),
                         derivative.interpolate().deriv(1, grid_.num_points() - 1)};
        return result;
    }

    /// Evaluate the field map without renormalization; energy here is the bare T+V contraction.
    radial_core_state_t
    state(std::vector<double> const& coefficients__) const
    {
        check_coefficients(coefficients__);
        radial_core_state_t result;
        result.coefficients   = coefficients__;
        result.energy         = quadratic(h0_, coefficients__);
        result.kinetic_energy = quadratic(kinetic_, coefficients__);
        result.rho.resize(samples_.size());
        result.tau.resize(samples_.size());
        result.value.resize(samples_.size());
        result.derivative.resize(samples_.size());
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            // Near the nucleus, large B-spline derivative contributions cancel.
            long double value{0}, derivative{0}, angular{0};
            for (auto const& s : samples_[ir]) {
                long double coefficient = coefficients__[s.index];
                value += coefficient * s.value;
                derivative += coefficient * s.derivative;
                angular += coefficient * s.angular;
            }
            result.rho[ir]        = value * value / fourpi;
            result.tau[ir]        = 0.5 * (derivative * derivative + angular * angular) / fourpi;
            result.value[ir]      = value;
            result.derivative[ir] = derivative;
            if (!std::isfinite(result.rho[ir]) || !std::isfinite(result.tau[ir]) || !std::isfinite(value) ||
                !std::isfinite(derivative)) {
                RTE_THROW("radial core solver: nonfinite orbital or density/tau field");
            }
        }
        return result;
    }

    /// Return the lowest radial eigenstates for this l, in ascending energy order.
    std::vector<radial_core_state_t>
    solve(int num_states__, std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        if (size() == basis_.size() || num_states__ <= 0 || num_states__ > size()) {
            RTE_THROW("radial core solver: invalid number of eigenstates");
        }
        int n           = size();
        auto physical_h = hamiltonian(drho__, dtau__);
        la::dmatrix<double> eigenvectors(n, num_states__);
        // Diagonal overlap scaling avoids the large dynamic range of r^l B_i near the origin.
        std::vector<double> scale(n), energies(n);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(overlap_(i, i)) || overlap_(i, i) <= 0) {
                RTE_THROW("radial core solver: singular overlap");
            }
            scale[i] = 1 / std::sqrt(overlap_(i, i));
        }
        int bandwidth = order - 1, leading = bandwidth + 1, one = 1, count = 0, info;
        std::vector<double> ab(leading * n, 0), bb(leading * n, 0), q(size_t(n) * n), work(7 * n);
        std::vector<ftn_int> iwork(5 * n), ifail(n);
        double vl = 0, vu = 0, tolerance = 2 * la::linalg_base::dlamch('S');
        for (int j = 0; j < n; j++) {
            for (int i = std::max(0, j - bandwidth); i <= j; i++) {
                auto offset = bandwidth + i - j + leading * j;
                ab[offset]  = physical_h(i, j) * scale[i] * scale[j];
                bb[offset]  = overlap_(i, j) * scale[i] * scale[j];
                if (!std::isfinite(ab[offset]) || !std::isfinite(bb[offset])) {
                    RTE_THROW("radial core solver: nonfinite radial operator");
                }
            }
        }
        FORTRAN(dsbgvx)
        ("V", "I", "U", &n, &bandwidth, &bandwidth, ab.data(), &leading, bb.data(), &leading, q.data(), &n, &vl, &vu,
         &one, &num_states__, &tolerance, &count, energies.data(), eigenvectors.at(memory_t::host), &n, work.data(),
         iwork.data(), ifail.data(), &info, 1, 1, 1);
        if (info || count != num_states__) {
            RTE_THROW("radial core eigensolver failed: " + std::to_string(info));
        }
        std::vector<radial_core_state_t> result;
        for (int j = 0; j < num_states__; j++) {
            std::vector<double> c(n);
            for (int i = 0; i < n; i++) {
                c[i] = scale[i] * eigenvectors(i, j);
            }
            // Refine low core eigenstates against the large near-nucleus kinetic spectrum,
            // also present in nonrelativistic bases with tightly spaced radial knots.
            for (int iteration = 0; iteration < 6; iteration++) {
                double energy = quadratic(physical_h, c) / quadratic(overlap_, c);
                std::vector<double> residual(n);
                for (int i = 0; i < n; i++) {
                    long double r{0};
                    for (int k = std::max(0, i - order + 1); k <= std::min(n - 1, i + order - 1); k++) {
                        r -= (static_cast<long double>(physical_h(i, k)) -
                              static_cast<long double>(energy) * overlap_(i, k)) *
                             c[k];
                    }
                    residual[i] = r;
                }
                auto correction = solve_constrained(physical_h, energy, c, std::move(residual));
                for (int i = 0; i < n; i++) {
                    c[i] += correction[i];
                }
                for (auto const& previous : result) {
                    double overlap = contract(overlap_, previous.coefficients, c);
                    for (int i = 0; i < n; i++) {
                        c[i] -= overlap * previous.coefficients[i];
                    }
                }
                double norm = std::sqrt(quadratic(overlap_, c));
                for (auto& value : c) {
                    value /= norm;
                }
            }
            energies[j] = quadratic(physical_h, c);
            result.push_back(state(c));
            result.back().energy = energies[j];
        }
        return result;
    }

    /// Differentiate core eigenstates at fixed radial grid, basis, nuclear charge and boundary.
    /** The potential direction varies both the local operator and the ZORA inverse mass.
     *  The two XC directions are
     *  variations of the already-weighted sample covectors, without further quadrature.
     *  The tangent orbitals satisfy c^T S dc = 0. Energies are eigenvalue derivatives.
     *  Optional reference states must be eigenstates of the same potential and XC operator. */
    std::vector<radial_core_response_t>
    solve_response(int num_states__, std::vector<double> const& drho__, std::vector<double> const& dtau__,
                   std::vector<double> const& potential_direction__, std::vector<double> const& rho_direction__,
                   std::vector<double> const& tau_direction__,
                   std::vector<radial_core_state_t> const* reference__ = nullptr) const
    {
        if (potential_direction__.size() != samples_.size() ||
            !std::all_of(potential_direction__.begin(), potential_direction__.end(),
                         [](double x) { return std::isfinite(x); })) {
            RTE_THROW("radial core response: invalid potential direction");
        }
        int n = size();
        la::dmatrix<double> dh(n, n), dk(n, n);
        dh.zero();
        dk.zero();
        Spline<double> direction(grid_, potential_direction__);
        for (auto const& p : basis_.basis_pairs()) {
            for (size_t q = 0; q < p.r.size(); q++) {
                double r = p.r[q], rl = std::pow(r, radial_power_);
                double dv = direction.at_point(std::max(r, grid_.first()));
                double ri = rl * p.Bi[q], rj = rl * p.Bj[q];
                double weight = p.w[q] * r * r;
                double dt{0};
                if (relativity_ == relativity_t::zora) {
                    double minv = inverse_mass(regular_potential_.at_point(std::max(r, grid_.first())) - zn_ / r);
                    double di   = rl * (p.dBi[q] + radial_power_ * p.Bi[q] / r);
                    double dj   = rl * (p.dBj[q] + radial_power_ * p.Bj[q] / r);
                    dt          = 0.5 * weight * sq_alpha_half * minv * minv * dv *
                         (di * dj + l_ * (l_ + 1.0) * ri * rj / (r * r));
                }
                dk(p.i, p.j) += dt;
                dh(p.i, p.j) += weight * ri * rj * dv + dt;
            }
        }
        add_xc(dh, rho_direction__, tau_direction__);
        auto h      = hamiltonian(drho__, dtau__);
        auto states = reference__ ? *reference__ : solve(num_states__, drho__, dtau__);
        if (num_states__ <= 0 || states.size() != static_cast<size_t>(num_states__)) {
            RTE_THROW("radial core response: inconsistent reference states");
        }
        std::vector<radial_core_response_t> result;
        for (auto& value : states) {
            auto const& c = value.coefficients;
            check_coefficients(c);
            if (!std::isfinite(value.energy)) {
                RTE_THROW("radial core response: nonfinite reference energy");
            }
            std::vector<double> rhs(n);
            for (int i = 0; i < n; i++) {
                long double r{0};
                for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                    r -= static_cast<long double>(dh(i, j)) * c[j];
                }
                rhs[i] = r;
            }
            auto solution = solve_constrained(h, value.energy, c, std::move(rhs));
            std::vector<double> dc(solution.begin(), solution.begin() + n);
            auto tangent           = state(dc);
            tangent.energy         = -solution[n];
            tangent.kinetic_energy = quadratic(dk, c) + 2 * contract(kinetic_, c, dc);
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                double product  = value.value[ir] * tangent.value[ir];
                tangent.rho[ir] = 2 * product / fourpi;
                tangent.tau[ir] = (value.derivative[ir] * tangent.derivative[ir] +
                                   l_ * (l_ + 1.0) * product / (grid_[ir] * grid_[ir])) /
                                  fourpi;
            }
            result.push_back({std::move(value), std::move(tangent)});
        }
        return result;
    }
};

/// Point-nucleus Dirac core reference, with energies relative to mc^2.
/** P_i=r^gamma B_i and Q=c/(E-V+2c^2) (d/dr+kappa/r)P give the upper-component
 *  min-max form of the bare Dirac equation. The lower component is eliminated
 *  exactly rather than expanded in a nearly dependent balance basis.
 *  Dropping the last upper B-spline imposes P(R)=0 and vanishing Dirac flux.
 *  The positive orbital-gradient tau includes both components and is not the Dirac kinetic energy.
 *  XC-dependent no-pair projection and its response must be supplied before using this reference
 *  in the external-XC core path. */
class Radial_dirac_core_solver
{
    friend class Radial_dirac_xc_solver;

  private:
    static constexpr int order{6};
    int kappa_;
    double zn_, gamma_;
    Radial_grid<double> const& grid_;
    bspline_basis<order> basis_;
    Spline<double> regular_potential_;
    la::dmatrix<double> overlap_, potential_;
    // Fixed potential and numerator of each eliminated-component quadrature term.
    std::vector<std::vector<std::array<double, 2>>> lower_products_;

    static double
    origin_power(int kappa__, double zn__)
    {
        double k = kappa__;
        if (!kappa__ || kappa__ == std::numeric_limits<int>::min() || !std::isfinite(zn__) || zn__ <= 0 ||
            zn__ / speed_of_light >= std::abs(k)) {
            RTE_THROW("Dirac core solver: invalid angular channel or supercritical nucleus");
        }
        return std::sqrt(k * k - std::pow(zn__ / speed_of_light, 2));
    }

    /// P, Q and the derivatives of P/r and Q/r, retaining the Coulomb power analytically.
    std::array<double, 4>
    sample(int i__, double r__, double energy__) const
    {
        double power = std::pow(r__, gamma_), b = basis_(i__, r__), db = basis_.deriv(i__, r__);
        int ir             = grid_.index_of(std::max(r__, grid_.first()));
        double regular     = regular_potential_(ir, std::max(r__, grid_.first()) - grid_[ir]);
        double dv          = r__ < grid_.first() ? 0 : regular_potential_.deriv(1, ir, r__ - grid_[ir]);
        double mass        = 2 * speed_of_light * speed_of_light + energy__ - regular;
        double denominator = mass * r__ + zn_, derivative = mass - r__ * dv;
        if (!std::isfinite(denominator) || denominator <= 0) {
            RTE_THROW("Dirac core solver: nonpositive eliminated-component denominator");
        }
        double d = r__ * db + (gamma_ + kappa_) * b;
        return {power * b, speed_of_light * power * d / denominator, power / r__ * (db + (gamma_ - 1) * b / r__),
                speed_of_light * power / (r__ * denominator) *
                        (r__ * basis_.deriv2(i__, r__) + (2 * gamma_ + kappa_) * db +
                         (gamma_ - 1) * (gamma_ + kappa_) * b / r__ - derivative * d / denominator)};
    }

    /// Reduced Hamiltonian and lower-component overlap at one trial energy.
    void
    assemble(double energy__, la::dmatrix<double>& h__, la::dmatrix<double>& lower__) const
    {
        copy(potential_, h__);
        lower__.zero();
        for (size_t ip = 0; ip < basis_.basis_pairs().size(); ip++) {
            auto const& pair = basis_.basis_pairs()[ip];
            for (auto const& point : lower_products_[ip]) {
                double denominator = 2 * speed_of_light * speed_of_light + energy__ - point[0];
                if (!std::isfinite(denominator) || denominator <= 0) {
                    RTE_THROW("Dirac core solver: nonpositive eliminated-component denominator");
                }
                double product = point[1] / denominator;
                h__(pair.i, pair.j) += product;
                lower__(pair.i, pair.j) += product / denominator;
            }
        }
        for (int j = 0; j < size(); j++) {
            for (int i = j + 1; i < std::min(size(), j + order); i++) {
                h__(i, j)     = h__(j, i);
                lower__(i, j) = lower__(j, i);
            }
        }
    }

    double
    contract(la::dmatrix<double> const& a__, std::vector<double> const& x__, std::vector<double> const& y__) const
    {
        long double result{0};
        for (int j = 0; j < size(); j++) {
            for (int i = std::max(0, j - order + 1); i <= std::min(size() - 1, j + order - 1); i++) {
                result += static_cast<long double>(x__[i]) * a__(i, j) * y__[j];
            }
        }
        return result;
    }

    std::array<double, 4>
    evaluate(std::vector<double> const& c__, double energy__, double r__) const
    {
        std::array<long double, 4> result{};
        double x = r__ == basis_.knot(basis_.num_knots() - 1) ? std::nextafter(r__, 0.0) : r__;
        int first{0}, last = basis_.num_knots();
        while (first < last) {
            int middle = first + (last - first) / 2;
            if (basis_.knot(middle) <= x) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        for (int i = std::max(0, first - order); i <= std::min(size() - 1, first - 1); i++) {
            auto values = sample(i, r__, energy__);
            for (int j = 0; j < 4; j++) {
                result[j] += static_cast<long double>(c__[i]) * values[j];
            }
        }
        return {double(result[0]), double(result[1]), double(result[2]), double(result[3])};
    }

    /// Recover one generalized eigenvector by banded inverse iteration, without a dense transformation.
    std::vector<double>
    inverse_iteration(la::dmatrix<double> const& h__, std::vector<double> const& scale__, double eigenvalue__) const
    {
        int n = size(), bandwidth = order - 1, leading = 3 * bandwidth + 1, nrhs = 1, info;
        std::vector<double> band(leading * n), c(n), rhs(n);
        std::vector<ftn_int> ipiv(n);
        double operator_norm{0};
        for (int i = 0; i < n; i++) {
            double row{0};
            for (int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); j++) {
                row += std::abs(h__(i, j) * scale__[i] * scale__[j]);
            }
            operator_norm = std::max(operator_norm, row);
            c[i]          = std::sin(0.5 + std::sqrt(2.0) * i);
        }
        double shift =
                16 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(eigenvalue__), operator_norm});
        for (int iteration = 0; iteration < 4; iteration++) {
            std::fill(band.begin(), band.end(), 0);
            for (int j = 0; j < n; j++) {
                long double product{0};
                for (int i = std::max(0, j - bandwidth); i <= std::min(n - 1, j + bandwidth); i++) {
                    band[2 * bandwidth + i - j + leading * j] =
                            (h__(i, j) - (eigenvalue__ + shift) * overlap_(i, j)) * scale__[i] * scale__[j];
                    product += static_cast<long double>(overlap_(j, i)) * scale__[j] * scale__[i] * c[i];
                }
                rhs[j] = product;
            }
            FORTRAN(dgbsv)
            (&n, &bandwidth, &bandwidth, &nrhs, band.data(), &leading, ipiv.data(), rhs.data(), &n, &info);
            double peak{0};
            for (double value : rhs) {
                peak = std::isfinite(value) ? std::max(peak, std::abs(value)) : std::numeric_limits<double>::infinity();
            }
            if (info || !std::isfinite(peak) || peak == 0) {
                RTE_THROW("Dirac core solver: singular or nonfinite inverse iteration");
            }
            for (int i = 0; i < n; i++) {
                c[i] = rhs[i] / peak;
            }
        }
        for (int i = 0; i < n; i++) {
            c[i] *= scale__[i];
        }
        double norm  = contract(overlap_, c, c);
        double error = std::abs(contract(h__, c, c) / norm - eigenvalue__);
        if (!std::isfinite(norm) || norm <= 0 || !std::isfinite(error) || error > 64 * shift) {
            RTE_THROW("Dirac core solver: inverse iteration did not recover the selected eigenvalue");
        }
        return c;
    }

    /// Eliminate a banded principal minor, retaining the eigenstate and normalization in a 2x2 border.
    std::vector<double>
    solve_constrained(la::dmatrix<double> const& matrix__, std::vector<double> rhs__) const
    {
        int n = size(), pivot{0};
        for (int i = 1; i < n; i++) {
            if (std::abs(matrix__(i, n)) > std::abs(matrix__(pivot, n))) {
                pivot = i;
            }
        }
        int m = n - 1, bandwidth = order - 1, leading = 3 * bandwidth + 1, nrhs = 3, info;
        std::vector<double> band(leading * m, 0), vectors(3 * m);
        std::vector<ftn_int> ipiv(m);
        auto index = [pivot](int i) { return i < pivot ? i : i + 1; };
        for (int j = 0; j < m; j++) {
            for (int i = std::max(0, j - bandwidth); i <= std::min(m - 1, j + bandwidth); i++) {
                band[2 * bandwidth + i - j + leading * j] = matrix__(index(i), index(j));
            }
            vectors[j]         = rhs__[index(j)];
            vectors[j + m]     = matrix__(index(j), pivot);
            vectors[j + 2 * m] = matrix__(index(j), n);
        }
        FORTRAN(dgbsv)
        (&m, &bandwidth, &bandwidth, &nrhs, band.data(), &leading, ipiv.data(), vectors.data(), &m, &info);
        if (info) {
            RTE_THROW("Dirac core response: singular banded principal minor");
        }
        long double s00 = matrix__(pivot, pivot), s01 = matrix__(pivot, n), s11 = matrix__(n, n), f0 = rhs__[pivot],
                    f1 = rhs__[n];
        for (int i = 0; i < m; i++) {
            double u = matrix__(index(i), pivot), v = matrix__(index(i), n);
            s00 -= static_cast<long double>(u) * vectors[i + m];
            s01 -= static_cast<long double>(u) * vectors[i + 2 * m];
            s11 -= static_cast<long double>(v) * vectors[i + 2 * m];
            f0 -= static_cast<long double>(u) * vectors[i];
            f1 -= static_cast<long double>(v) * vectors[i];
        }
        long double determinant = s00 * s11 - s01 * s01;
        if (!std::isfinite(determinant) || determinant == 0) {
            RTE_THROW("Dirac core response: singular normalization border");
        }
        double xp = (f0 * s11 - f1 * s01) / determinant, multiplier = (f1 * s00 - f0 * s01) / determinant;
        for (int i = 0; i < m; i++) {
            rhs__[index(i)] = vectors[i] - vectors[i + m] * xp - vectors[i + 2 * m] * multiplier;
        }
        rhs__[pivot] = xp;
        rhs__[n]     = multiplier;
        if (!std::all_of(rhs__.begin(), rhs__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac core response: nonfinite constrained solution");
        }
        return rhs__;
    }

    /// Refine the nonlinear root and spinor, whose second derivatives amplify eigenvector roundoff.
    double
    refine(std::vector<double>& c__, double& energy__) const
    {
        int n = size();
        double correction{0};
        for (int iteration = 0; iteration < 3; iteration++) {
            la::dmatrix<double> h(n, n), lower(n, n), matrix(n + 1, n + 1);
            assemble(energy__, h, lower);
            matrix.zero();
            std::vector<double> rhs(n + 1, 0), scale(n + 1);
            double border_norm{0};
            for (int i = 0; i < n; i++) {
                scale[i] = 1 / std::sqrt(std::max({std::abs(h(i, i)), std::abs(energy__ * overlap_(i, i)), 1e-30}));
                long double residual{0}, mc{0};
                for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                    long double a =
                            static_cast<long double>(h(i, j)) - static_cast<long double>(energy__) * overlap_(i, j);
                    matrix(i, j) = a;
                    residual -= a * c__[j];
                    mc += static_cast<long double>(overlap_(i, j) + lower(i, j)) * c__[j];
                }
                rhs[i]       = residual;
                matrix(i, n) = matrix(n, i) = -mc;
                border_norm += std::pow(scale[i] * double(mc), 2);
            }
            if (!std::isfinite(border_norm) || border_norm <= 0) {
                RTE_THROW("Dirac core refinement: invalid spinor normalization");
            }
            scale[n] = 1 / std::sqrt(border_norm);
            rhs[n]   = (contract(overlap_, c__, c__) + contract(lower, c__, c__) - 1) / 2;
            for (auto const& pair : basis_.basis_pairs()) {
                for (size_t q = 0; q < pair.r.size(); q++) {
                    double r = pair.r[q], power = std::pow(r, gamma_ - 1);
                    double v           = regular_potential_.at_point(std::max(r, grid_.first())) - zn_ / r;
                    double denominator = 2 * speed_of_light * speed_of_light + energy__ - v;
                    double a           = power * (r * pair.dBi[q] + (gamma_ + kappa_) * pair.Bi[q]);
                    double b           = power * (r * pair.dBj[q] + (gamma_ + kappa_) * pair.Bj[q]);
                    matrix(n, n) += pair.w[q] * speed_of_light * speed_of_light * a * b /
                                    (denominator * denominator * denominator) * c__[pair.i] * c__[pair.j];
                }
            }
            for (int j = 0; j <= n; j++) {
                rhs[j] *= scale[j];
                for (int i = 0; i <= n; i++) {
                    matrix(i, j) *= scale[i] * scale[j];
                }
            }
            rhs = solve_constrained(matrix, std::move(rhs));
            for (int i = 0; i < n; i++) {
                c__[i] += scale[i] * rhs[i];
            }
            correction = scale[n] * rhs[n];
            energy__ += correction;
        }
        return correction;
    }

  public:
    Radial_dirac_core_solver(Radial_grid<double> const& grid__, int kappa__, double zn__,
                             std::vector<double> const& potential__, int num_basis__)
        : kappa_(kappa__)
        , zn_(zn__)
        , gamma_(origin_power(kappa__, zn__))
        , grid_(grid__)
        , basis_(Radial_core_solver::knots(grid__, num_basis__, 0, zn__, {}, relativity_t::none), 32, false,
                 std::max(1.0, 4 / gamma_))
        , regular_potential_(grid__)
        , overlap_(basis_.size() - 1, basis_.size() - 1)
        , potential_(overlap_.num_rows(), overlap_.num_rows())
    {
        if (potential__.size() != static_cast<size_t>(grid__.num_points())) {
            RTE_THROW("Dirac core solver: inconsistent radial potential");
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            if (!std::isfinite(potential__[ir]) || !std::isfinite(grid__[ir]) || (ir && grid__[ir] <= grid__[ir - 1])) {
                RTE_THROW("Dirac core solver: nonfinite potential or invalid radial grid");
            }
            regular_potential_(ir) = potential__[ir] + zn__ / grid__[ir];
        }
        regular_potential_.interpolate();
        overlap_.zero();
        potential_.zero();
        for (auto const& pair : basis_.basis_pairs()) {
            auto& products = lower_products_.emplace_back();
            products.reserve(pair.r.size());
            for (size_t q = 0; q < pair.r.size(); q++) {
                double r = pair.r[q], w = pair.w[q];
                double v       = regular_potential_.at_point(std::max(r, grid__.first())) - zn_ / r;
                double product = w * std::pow(r, 2 * gamma_) * pair.Bi[q] * pair.Bj[q];
                overlap_(pair.i, pair.j) += product;
                potential_(pair.i, pair.j) += product * v;
                double power = std::pow(r, gamma_ - 1);
                double a     = power * (r * pair.dBi[q] + (gamma_ + kappa_) * pair.Bi[q]);
                double b     = power * (r * pair.dBj[q] + (gamma_ + kappa_) * pair.Bj[q]);
                products.push_back({v, w * speed_of_light * speed_of_light * a * b});
            }
        }
        for (int j = 0; j < size(); j++) {
            for (int i = j + 1; i < std::min(size(), j + order); i++) {
                overlap_(i, j)   = overlap_(j, i);
                potential_(i, j) = potential_(j, i);
            }
        }
    }

    int
    size() const
    {
        return overlap_.num_rows();
    }

    double
    overlap(radial_core_state_t const& a__, radial_core_state_t const& b__) const
    {
        if (a__.coefficients.size() != static_cast<size_t>(size()) ||
            b__.coefficients.size() != a__.coefficients.size()) {
            RTE_THROW("Dirac core solver: inconsistent spinor coefficients");
        }
        long double result = contract(overlap_, a__.coefficients, b__.coefficients);
        for (auto const& pair : basis_.basis_pairs()) {
            for (size_t q = 0; q < pair.r.size(); q++) {
                auto a = sample(pair.i, pair.r[q], a__.energy), b = sample(pair.j, pair.r[q], b__.energy);
                result += static_cast<long double>(a__.coefficients[pair.i]) * b__.coefficients[pair.j] * pair.w[q] *
                          a[1] * b[1];
            }
        }
        return result;
    }

    radial_core_state_t
    state(std::vector<double> const& coefficients__, double energy__) const
    {
        if (coefficients__.size() != static_cast<size_t>(size()) ||
            !std::all_of(coefficients__.begin(), coefficients__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac core solver: invalid spinor coefficients");
        }
        radial_core_state_t result;
        result.coefficients = coefficients__;
        result.energy       = energy__;
        result.kappa        = kappa_;
        la::dmatrix<double> h(size(), size()), lower(size(), size());
        assemble(energy__, h, lower);
        result.kinetic_energy = 2 * (contract(h, coefficients__, coefficients__) -
                                     contract(potential_, coefficients__, coefficients__) -
                                     speed_of_light * speed_of_light * contract(lower, coefficients__, coefficients__));
        for (auto* field : {&result.value, &result.derivative, &result.small_value, &result.small_derivative,
                            &result.rho, &result.tau}) {
            field->resize(grid_.num_points());
        }
        int l = kappa_ > 0 ? kappa_ : -kappa_ - 1, small_l = kappa_ > 0 ? kappa_ - 1 : -kappa_;
        for (int ir = 0; ir < grid_.num_points(); ir++) {
            double r                    = grid_[ir];
            auto values                 = evaluate(coefficients__, energy__, r);
            result.value[ir]            = values[0] / r;
            result.derivative[ir]       = values[2];
            result.small_value[ir]      = values[1] / r;
            result.small_derivative[ir] = values[3];
            result.rho[ir]              = (values[0] * values[0] + values[1] * values[1]) / (fourpi * r * r);
            result.tau[ir] =
                    (std::pow(result.derivative[ir], 2) + std::pow(result.small_derivative[ir], 2) +
                     (l * (l + 1.0) * values[0] * values[0] + small_l * (small_l + 1.0) * values[1] * values[1]) /
                             (r * r * r * r)) /
                    (2 * fourpi);
            if (!std::isfinite(result.rho[ir]) || !std::isfinite(result.tau[ir])) {
                RTE_THROW("Dirac core solver: nonfinite spinor density or kinetic density");
            }
        }
        return result;
    }

    /// Solve the decreasing upper-component eigenvalue branches by Newton iteration.
    std::vector<radial_core_state_t>
    solve(int num_states__) const
    {
        if (num_states__ <= 0 || num_states__ > basis_.size() - 1) {
            RTE_THROW("Dirac core solver: invalid number of core states");
        }
        int n = size();
        std::vector<double> scale(n), eigenvalues(n);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(overlap_(i, i)) || overlap_(i, i) <= 0) {
                RTE_THROW("Dirac core solver: singular spinor overlap");
            }
            scale[i] = 1 / std::sqrt(overlap_(i, i));
        }
        std::vector<radial_core_state_t> result;
        for (int j = 0; j < num_states__; j++) {
            double energy{0}, step{0};
            for (int iteration = 0; iteration < 32; iteration++) {
                la::dmatrix<double> h(n, n), lower(n, n);
                assemble(energy, h, lower);
                int bandwidth = order - 1, leading = order, index = j + 1, one = 1, count{0}, ifail{0}, info;
                std::vector<double> ab(leading * n, 0), bb(leading * n, 0), work(7 * n);
                std::vector<ftn_int> iwork(5 * n);
                double q{0}, z{0}, vl{0}, vu{0}, tolerance = 2 * la::linalg_base::dlamch('S');
                for (int b = 0; b < n; b++) {
                    for (int a = std::max(0, b - bandwidth); a <= b; a++) {
                        int offset = bandwidth + a - b + leading * b;
                        ab[offset] = h(a, b) * scale[a] * scale[b];
                        bb[offset] = overlap_(a, b) * scale[a] * scale[b];
                    }
                }
                FORTRAN(dsbgvx)
                ("N", "I", "U", &n, &bandwidth, &bandwidth, ab.data(), &leading, bb.data(), &leading, &q, &one, &vl,
                 &vu, &index, &index, &tolerance, &count, eigenvalues.data(), &z, &one, work.data(), iwork.data(),
                 &ifail, &info, 1, 1, 1);
                if (info || count != 1) {
                    RTE_THROW("Dirac core eigensolver failed: " + std::to_string(info));
                }
                auto c = inverse_iteration(h, scale, eigenvalues[0]);
                // The Rayleigh residual retains low-energy accuracy on highly graded core grids.
                double norm = contract(overlap_, c, c) + contract(lower, c, c);
                step        = (contract(h, c, c) - energy * contract(overlap_, c, c)) / norm;
                if (!std::isfinite(step) || energy + step <= -speed_of_light * speed_of_light) {
                    RTE_THROW("Dirac core solver: root outside the electronic gap");
                }
                if (std::abs(step) <=
                    std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(energy))) {
                    // Refine the individual root beyond the eigensolver's roundoff floor.
                    for (auto& value : c) {
                        value /= std::sqrt(norm);
                    }
                    if (std::abs(refine(c, energy)) <= 1e-12 * std::max(1.0, std::abs(energy))) {
                        result.push_back(state(c, energy));
                        break;
                    }
                    continue;
                }
                energy += step;
            }
            if (static_cast<int>(result.size()) != j + 1) {
                std::ostringstream message;
                message << "Dirac core solver: nonlinear eigenvalue did not converge; kappa=" << kappa_
                        << " state=" << j << " energy=" << energy << " step=" << step;
                RTE_THROW(message.str());
            }
        }
        return result;
    }

    /// Differentiate the min-max equation and full spinor normalization at fixed nuclear charge.
    std::vector<radial_core_response_t>
    solve_response(int num_states__, std::vector<double> const& direction__) const
    {
        if (direction__.size() != static_cast<size_t>(grid_.num_points()) ||
            !std::all_of(direction__.begin(), direction__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac core response: invalid regular potential direction");
        }
        Spline<double> direction(grid_, direction__);
        auto states = solve(num_states__);
        std::vector<radial_core_response_t> result;
        for (auto& value : states) {
            int n = size();
            la::dmatrix<double> h(n, n), lower(n, n), dh(n, n), lower_energy(n, n), lower_direction(n, n);
            assemble(value.energy, h, lower);
            dh.zero();
            lower_energy.zero();
            lower_direction.zero();
            for (auto const& pair : basis_.basis_pairs()) {
                for (size_t q = 0; q < pair.r.size(); q++) {
                    double r = pair.r[q], power = std::pow(r, gamma_);
                    double v           = regular_potential_.at_point(std::max(r, grid_.first())) - zn_ / r;
                    double denominator = 2 * speed_of_light * speed_of_light + value.energy - v;
                    double dv          = direction.at_point(std::max(r, grid_.first()));
                    double p = power * pair.Bi[q], t = power * pair.Bj[q];
                    double a =
                            speed_of_light * power * (pair.dBi[q] + (gamma_ + kappa_) * pair.Bi[q] / r) / denominator;
                    double b =
                            speed_of_light * power * (pair.dBj[q] + (gamma_ + kappa_) * pair.Bj[q] / r) / denominator;
                    dh(pair.i, pair.j) += pair.w[q] * dv * (p * t + a * b);
                    lower_energy(pair.i, pair.j) += pair.w[q] * a * b / denominator;
                    lower_direction(pair.i, pair.j) += pair.w[q] * a * b * dv / denominator;
                }
            }
            // The energy derivative of the eliminated component contributes to both borders.
            la::dmatrix<double> matrix(n + 1, n + 1);
            matrix.zero();
            std::vector<double> rhs(n + 1, 0), scale(n + 1);
            double border_norm{0};
            for (int i = 0; i < n; i++) {
                scale[i] = 1 / std::sqrt(std::max({std::abs(h(i, i)), std::abs(value.energy * overlap_(i, i)), 1e-30}));
                long double mc{0}, dc{0};
                for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                    matrix(i, j) = h(i, j) - value.energy * overlap_(i, j);
                    mc += static_cast<long double>(overlap_(i, j) + lower(i, j)) * value.coefficients[j];
                    dc += static_cast<long double>(dh(i, j)) * value.coefficients[j];
                }
                matrix(i, n) = matrix(n, i) = -mc;
                rhs[i]                      = -dc;
                border_norm += std::pow(scale[i] * double(mc), 2);
            }
            if (!std::isfinite(border_norm) || border_norm <= 0) {
                RTE_THROW("Dirac core response: invalid spinor normalization constraint");
            }
            scale[n]     = 1 / std::sqrt(border_norm);
            matrix(n, n) = contract(lower_energy, value.coefficients, value.coefficients);
            rhs[n]       = contract(lower_direction, value.coefficients, value.coefficients);
            for (int j = 0; j <= n; j++) {
                rhs[j] *= scale[j];
                for (int i = 0; i <= n; i++) {
                    matrix(i, j) *= scale[i] * scale[j];
                }
            }
            rhs = solve_constrained(matrix, std::move(rhs));
            for (int i = 0; i <= n; i++) {
                rhs[i] *= scale[i];
            }
            double de = rhs[n];
            rhs.resize(n);
            auto tangent   = state(rhs, value.energy);
            tangent.energy = de;
            int l = kappa_ > 0 ? kappa_ : -kappa_ - 1, small_l = kappa_ > 0 ? kappa_ - 1 : -kappa_;
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                double r = grid_[ir], regular = regular_potential_(ir);
                double denominator = (2 * speed_of_light * speed_of_light + value.energy - regular) * r + zn_;
                double dv = direction__[ir] - de, f = dv * r / denominator;
                double df = (direction.deriv(1, ir) * r +
                             dv * (zn_ + r * r * regular_potential_.deriv(1, ir)) / denominator) /
                            denominator;
                tangent.small_value[ir] += f * value.small_value[ir];
                tangent.small_derivative[ir] += f * value.small_derivative[ir] + df * value.small_value[ir];
                tangent.rho[ir] =
                        2 * (value.value[ir] * tangent.value[ir] + value.small_value[ir] * tangent.small_value[ir]) /
                        fourpi;
                tangent.tau[ir] = (value.derivative[ir] * tangent.derivative[ir] +
                                   value.small_derivative[ir] * tangent.small_derivative[ir] +
                                   (l * (l + 1.0) * value.value[ir] * tangent.value[ir] +
                                    small_l * (small_l + 1.0) * value.small_value[ir] * tangent.small_value[ir]) /
                                           (r * r)) /
                                  fourpi;
            }
            long double kinetic{0};
            for (auto const& pair : basis_.basis_pairs()) {
                for (size_t q = 0; q < pair.r.size(); q++) {
                    double r = pair.r[q];
                    auto a = sample(pair.i, r, value.energy), b = sample(pair.j, r, value.energy);
                    double v = regular_potential_.at_point(std::max(r, grid_.first())) - zn_ / r;
                    double f = (direction.at_point(std::max(r, grid_.first())) - de) /
                               (2 * speed_of_light * speed_of_light + value.energy - v);
                    kinetic -=
                            2 * pair.w[q] * v * value.coefficients[pair.i] *
                            (rhs[pair.j] * (a[0] * b[0] + a[1] * b[1]) + value.coefficients[pair.j] * f * a[1] * b[1]);
                }
            }
            tangent.kinetic_energy = kinetic;
            result.push_back({std::move(value), std::move(tangent)});
        }
        return result;
    }
};

/// Variational Dirac core operator in a fixed nuclear positive-energy subspace.
/** The nuclear Coulomb projector is independent of the electronic potential and XC covectors.
 *  Density and positive-tau operators include both spinor components, without extra sample weights.
 *  Potential and XC responses therefore differentiate one fixed no-pair variational problem.
 *  Occupied-core generation, valence orthogonalization and radial response share this operator. */
class Radial_dirac_xc_solver
{
  private:
    using sample_t = std::array<double, 4>;
    struct quadrature_t
    {
        double r, weight;
        std::vector<sample_t> values;
    };
    Radial_grid<double> grid_;
    int kappa_, num_basis_;
    double zn_;
    std::vector<std::vector<sample_t>> samples_;
    std::vector<quadrature_t> quadrature_;
    la::dmatrix<double> overlap_, kinetic_, nuclear_;
    mutable std::mutex states_mutex_;
    mutable std::unique_ptr<la::dmatrix<double>> solved_hamiltonian_;
    mutable std::vector<radial_core_state_t> solved_states_;

    static Radial_grid<double>
    copy_grid(Radial_grid<double> const& grid__)
    {
        if (grid__.num_points() < 4) {
            RTE_THROW("Dirac XC core solver: invalid radial grid size");
        }
        return grid__.segment(grid__.num_points());
    }

    static std::vector<double>
    nuclear_potential(Radial_grid<double> const& grid__, double zn__)
    {
        std::vector<double> potential(grid__.num_points());
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            potential[ir] = -zn__ / grid__[ir];
        }
        return potential;
    }

    static sample_t
    sample(Radial_dirac_core_solver const& reference__, radial_core_state_t const& state__, double r__)
    {
        auto values = reference__.evaluate(state__.coefficients, state__.energy, r__);
        values[0] /= r__;
        values[1] /= r__;
        return values;
    }

    void
    check_field(std::vector<double> const& field__) const
    {
        if (field__.size() != samples_.size() ||
            !std::all_of(field__.begin(), field__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac XC core solver: invalid radial field");
        }
    }

    void
    check_coefficients(std::vector<double> const& c__) const
    {
        if (c__.size() != static_cast<size_t>(size()) ||
            !std::all_of(c__.begin(), c__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac XC core solver: invalid electronic coefficients");
        }
    }

    double
    contract(la::dmatrix<double> const& a__, std::vector<double> const& x__, std::vector<double> const& y__) const
    {
        long double result{0};
        for (int j = 0; j < size(); j++) {
            for (int i = 0; i < size(); i++) {
                result += static_cast<long double>(x__[i]) * a__(i, j) * y__[j];
            }
        }
        return result;
    }

    double
    density(sample_t const& a__, sample_t const& b__) const
    {
        return a__[0] * b__[0] + a__[1] * b__[1];
    }

    double
    tau(sample_t const& a__, sample_t const& b__, double r__) const
    {
        int l = kappa_ > 0 ? kappa_ : -kappa_ - 1, small_l = kappa_ > 0 ? kappa_ - 1 : -kappa_;
        return (a__[2] * b__[2] + a__[3] * b__[3] +
                (l * (l + 1.0) * a__[0] * b__[0] + small_l * (small_l + 1.0) * a__[1] * b__[1]) / (r__ * r__)) /
               2;
    }

    /// Contract diagonal sample operators in blocks, retaining every radial point and basis state.
    template <typename Samples, typename Weights>
    void
    add_sampled_operator(la::dmatrix<double>& h__, int num_points__, Samples samples__, Weights weights__) const
    {
        constexpr int block{128};
        int n = size();
        mdarray<double, 2> a({n, 4 * block}), b({n, 4 * block});
        for (int first = 0; first < num_points__; first += block) {
            int columns{0};
            for (int p = first; p < std::min(num_points__, first + block); p++) {
                auto weights       = weights__(p);
                auto const& values = samples__(p);
                for (int part = 0; part < 4; part++) {
                    if (weights[part] != 0) {
                        for (int i = 0; i < n; i++) {
                            a(i, columns) = values[i][part];
                            b(i, columns) = weights[part] * values[i][part];
                        }
                        columns++;
                    }
                }
            }
            if (columns) {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'T', n, n, columns, &la::constant<double>::one(), a.at(memory_t::host), n,
                              b.at(memory_t::host), n, &la::constant<double>::one(), h__.at(memory_t::host), n);
            }
        }
    }

    void
    add_potential(la::dmatrix<double>& h__, std::vector<double> const& field__, bool nuclear__) const
    {
        check_field(field__);
        auto regular = field__;
        if (nuclear__) {
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                regular[ir] += zn_ / grid_[ir];
            }
        }
        Spline<double> potential(grid_, regular);
        add_sampled_operator(
                h__, static_cast<int>(quadrature_.size()), [&](int p) -> auto const& { return quadrature_[p].values; },
                [&](int p) {
                    auto const& point = quadrature_[p];
                    double v          = point.weight * potential.at_point(std::max(point.r, grid_.first()));
                    return sample_t{v, v, 0, 0};
                });
    }

    void
    add_xc(la::dmatrix<double>& h__, std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        check_field(drho__);
        check_field(dtau__);
        int l = kappa_ > 0 ? kappa_ : -kappa_ - 1, small_l = kappa_ > 0 ? kappa_ - 1 : -kappa_;
        add_sampled_operator(
                h__, grid_.num_points(), [&](int ir) -> auto const& { return samples_[ir]; },
                [&](int ir) {
                    double rho = drho__[ir] / fourpi, tau = dtau__[ir] / (2 * fourpi);
                    double angular = tau / (grid_[ir] * grid_[ir]);
                    return sample_t{rho + angular * l * (l + 1.0), rho + angular * small_l * (small_l + 1.0), tau, tau};
                });
        for (int j = 0; j < size(); j++) {
            for (int i = j + 1; i < size(); i++) {
                h__(i, j) = h__(j, i);
            }
        }
    }

    la::dmatrix<double>
    hamiltonian(std::vector<double> const& potential__, std::vector<double> const& drho__,
                std::vector<double> const& dtau__) const
    {
        la::dmatrix<double> h(size(), size());
        copy(nuclear_, h);
        add_potential(h, potential__, true);
        add_xc(h, drho__, dtau__);
        return h;
    }

    /// Normalized eigenpair correction and Sternheimer response in the same fixed no-pair space.
    std::vector<double>
    solve_constrained(la::dmatrix<double> const& h__, double energy__, std::vector<double> const& c__,
                      std::vector<double> rhs__) const
    {
        int n = size();
        la::dmatrix<double> matrix(n + 1, n + 1);
        matrix.zero();
        rhs__.resize(n + 1, 0);
        std::vector<double> scale(n + 1);
        double border_norm{0};
        for (int i = 0; i < n; i++) {
            scale[i] = 1 / std::sqrt(std::max({std::abs(h__(i, i)), std::abs(energy__ * overlap_(i, i)), 1e-30}));
            long double sc{0};
            for (int j = 0; j < n; j++) {
                matrix(i, j) =
                        static_cast<long double>(h__(i, j)) - static_cast<long double>(energy__) * overlap_(i, j);
                sc += static_cast<long double>(overlap_(i, j)) * c__[j];
            }
            matrix(i, n) = matrix(n, i) = sc;
            border_norm += std::pow(scale[i] * double(sc), 2);
        }
        if (!std::isfinite(border_norm) || border_norm <= 0) {
            RTE_THROW("Dirac XC core response: invalid normalization constraint");
        }
        scale[n] = 1 / std::sqrt(border_norm);
        for (int j = 0; j <= n; j++) {
            rhs__[j] *= scale[j];
            for (int i = 0; i <= n; i++) {
                matrix(i, j) *= scale[i] * scale[j];
            }
        }
        int info = la::wrap(la::lib_t::lapack).gesv(n + 1, 1, matrix.at(memory_t::host), n + 1, rhs__.data(), n + 1);
        if (info || !std::all_of(rhs__.begin(), rhs__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("Dirac XC core response: singular or nonfinite variational response");
        }
        for (int i = 0; i <= n; i++) {
            rhs__[i] *= scale[i];
        }
        return rhs__;
    }

  public:
    Radial_dirac_xc_solver(Radial_grid<double> const& grid__, int kappa__, double zn__, int num_basis__)
        : grid_(copy_grid(grid__))
        , kappa_(kappa__)
        , num_basis_(num_basis__)
        , zn_(zn__)
    {
        Radial_dirac_core_solver reference(grid__, kappa__, zn__, nuclear_potential(grid__, zn__), num_basis__);
        auto states = reference.solve(reference.size());
        int n       = states.size();
        overlap_    = la::dmatrix<double>(n, n);
        kinetic_    = la::dmatrix<double>(n, n);
        nuclear_    = la::dmatrix<double>(n, n);
        overlap_.zero();
        kinetic_.zero();
        nuclear_.zero();
        samples_.resize(grid__.num_points());
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            for (auto const& state : states) {
                samples_[ir].push_back(sample(reference, state, grid__[ir]));
            }
        }
        std::vector<double> x, w;
        gauss_legendre_rule(32, x, w);
        for (int interval = reference.order - 1; interval < reference.basis_.size(); interval++) {
            double a = reference.basis_.knot(interval), b = reference.basis_.knot(interval + 1);
            if (b <= a) {
                continue;
            }
            for (size_t iq = 0; iq < x.size(); iq++) {
                double r = 0.5 * ((b - a) * x[iq] + (b + a)), weight = 0.5 * (b - a) * w[iq];
                if (a == 0) {
                    double t = 0.5 * (x[iq] + 1), power = std::max(1.0, 4 / reference.gamma_);
                    r      = b * std::pow(t, power);
                    weight = 0.5 * b * power * std::pow(t, power - 1) * w[iq];
                }
                quadrature_t point{r, weight * r * r, {}};
                for (auto const& state : states) {
                    point.values.push_back(sample(reference, state, r));
                }
                for (int j = 0; j < n; j++) {
                    auto const& t = point.values[j];
                    double dj     = r * t[2] + (1 + kappa_) * t[0];
                    for (int i = 0; i <= j; i++) {
                        auto const& s  = point.values[i];
                        double di      = r * s[2] + (1 + kappa_) * s[0];
                        double metric  = point.weight * density(s, t);
                        double kinetic = weight * (speed_of_light * r * (di * t[1] + s[1] * dj) -
                                                   2 * speed_of_light * speed_of_light * r * r * s[1] * t[1]);
                        overlap_(i, j) += metric;
                        kinetic_(i, j) += kinetic;
                        nuclear_(i, j) += kinetic - zn_ * metric / r;
                    }
                }
                quadrature_.push_back(std::move(point));
            }
        }
        for (auto* matrix : {&overlap_, &kinetic_, &nuclear_}) {
            for (int j = 0; j < n; j++) {
                for (int i = j + 1; i < n; i++) {
                    (*matrix)(i, j) = (*matrix)(j, i);
                }
            }
        }
    }

    int
    size() const
    {
        return overlap_.num_rows();
    }

    bool
    matches(Radial_grid<double> const& grid__, int kappa__, double zn__, int num_basis__ = 0) const
    {
        if (kappa__ != kappa_ || zn__ != zn_ || num_basis__ != num_basis_ ||
            grid__.num_points() != grid_.num_points()) {
            return false;
        }
        for (int ir = 0; ir < grid_.num_points(); ir++) {
            if (grid__[ir] != grid_[ir]) {
                return false;
            }
        }
        return true;
    }

    double
    overlap(std::vector<double> const& a__, std::vector<double> const& b__) const
    {
        check_coefficients(a__);
        check_coefficients(b__);
        return contract(overlap_, a__, b__);
    }

    radial_core_state_t
    state(std::vector<double> const& c__) const
    {
        check_coefficients(c__);
        radial_core_state_t result;
        result.coefficients   = c__;
        result.energy         = contract(nuclear_, c__, c__);
        result.kinetic_energy = contract(kinetic_, c__, c__);
        result.kappa          = kappa_;
        for (auto* field : {&result.value, &result.derivative, &result.small_value, &result.small_derivative,
                            &result.rho, &result.tau}) {
            field->resize(grid_.num_points());
        }
        for (int ir = 0; ir < grid_.num_points(); ir++) {
            std::array<long double, 4> sum{};
            for (int j = 0; j < size(); j++) {
                for (int part = 0; part < 4; part++) {
                    sum[part] += static_cast<long double>(c__[j]) * samples_[ir][j][part];
                }
            }
            sample_t s{double(sum[0]), double(sum[1]), double(sum[2]), double(sum[3])};
            result.value[ir]            = s[0];
            result.small_value[ir]      = s[1];
            result.derivative[ir]       = s[2];
            result.small_derivative[ir] = s[3];
            result.rho[ir]              = density(s, s) / fourpi;
            result.tau[ir]              = tau(s, s, grid_[ir]) / fourpi;
            if (!std::isfinite(result.rho[ir]) || !std::isfinite(result.tau[ir])) {
                RTE_THROW("Dirac XC core solver: nonfinite density or kinetic density");
            }
        }
        return result;
    }

  private:
    std::vector<radial_core_state_t>
    solve(int num_states__, la::dmatrix<double> const& physical_h) const
    {
        if (num_states__ <= 0 || num_states__ > size()) {
            RTE_THROW("Dirac XC core solver: invalid number of core states");
        }
        // Response directions share the same core spectrum. Reuse only an identical
        // operator and return copies; nuclear projectors can be shared by symmetry classes.
        std::scoped_lock lock(states_mutex_);
        if (solved_hamiltonian_ && solved_states_.size() == static_cast<size_t>(num_states__) &&
            std::equal(physical_h.begin(), physical_h.end(), solved_hamiltonian_->begin())) {
            return solved_states_;
        }
        int n = size();
        la::dmatrix<double> h(n, n), s(n, n), vectors(n, num_states__);
        copy(physical_h, h);
        copy(overlap_, s);
        std::vector<double> energies(n);
        int info = la::Eigensolver_factory("lapack")->solve(n, num_states__, h, s, energies.data(), vectors);
        if (info) {
            RTE_THROW("Dirac XC core eigensolver failed: " + std::to_string(info));
        }
        std::vector<radial_core_state_t> result;
        for (int j = 0; j < num_states__; j++) {
            std::vector<double> c(n);
            for (int i = 0; i < n; i++) {
                c[i] = vectors(i, j);
            }
            // Refine low eigenpairs independently of the largest near-nucleus eigenvalues.
            for (int iteration = 0; iteration < 6; iteration++) {
                double energy = contract(physical_h, c, c) / contract(overlap_, c, c);
                std::vector<double> residual(n);
                for (int i = 0; i < n; i++) {
                    long double r{0};
                    for (int k = 0; k < n; k++) {
                        r -= (static_cast<long double>(physical_h(i, k)) -
                              static_cast<long double>(energy) * overlap_(i, k)) *
                             c[k];
                    }
                    residual[i] = r;
                }
                auto correction = solve_constrained(physical_h, energy, c, std::move(residual));
                for (int i = 0; i < n; i++) {
                    c[i] += correction[i];
                }
                for (auto const& previous : result) {
                    double projection = contract(overlap_, previous.coefficients, c);
                    for (int i = 0; i < n; i++) {
                        c[i] -= projection * previous.coefficients[i];
                    }
                }
                double norm = contract(overlap_, c, c);
                if (!std::isfinite(norm) || norm <= 0) {
                    RTE_THROW("Dirac XC core solver: invalid refined spinor normalization");
                }
                for (auto& value : c) {
                    value /= std::sqrt(norm);
                }
            }
            auto value   = state(c);
            value.energy = contract(physical_h, c, c);
            result.push_back(std::move(value));
        }
        if (!solved_hamiltonian_) {
            solved_hamiltonian_ = std::make_unique<la::dmatrix<double>>(n, n);
        }
        copy(physical_h, *solved_hamiltonian_);
        solved_states_ = result;
        return result;
    }

  public:
    std::vector<radial_core_state_t>
    solve(int num_states__, std::vector<double> const& potential__, std::vector<double> const& drho__,
          std::vector<double> const& dtau__) const
    {
        return solve(num_states__, hamiltonian(potential__, drho__, dtau__));
    }

    std::vector<radial_core_response_t>
    solve_response(int num_states__, std::vector<double> const& potential__, std::vector<double> const& drho__,
                   std::vector<double> const& dtau__, std::vector<double> const& potential_direction__,
                   std::vector<double> const& rho_direction__, std::vector<double> const& tau_direction__) const
    {
        int n  = size();
        auto h = hamiltonian(potential__, drho__, dtau__);
        la::dmatrix<double> dh(n, n);
        dh.zero();
        add_potential(dh, potential_direction__, false);
        add_xc(dh, rho_direction__, tau_direction__);
        auto states = solve(num_states__, h);
        std::vector<radial_core_response_t> result;
        for (auto& value : states) {
            std::vector<double> rhs(n, 0);
            for (int i = 0; i < n; i++) {
                long double dc{0};
                for (int j = 0; j < n; j++) {
                    dc += static_cast<long double>(dh(i, j)) * value.coefficients[j];
                }
                rhs[i] = -dc;
            }
            rhs       = solve_constrained(h, value.energy, value.coefficients, std::move(rhs));
            double de = -rhs[n];
            rhs.resize(n);
            auto tangent           = state(rhs);
            tangent.energy         = de;
            tangent.kinetic_energy = 2 * contract(kinetic_, value.coefficients, rhs);
            int l = kappa_ > 0 ? kappa_ : -kappa_ - 1, small_l = kappa_ > 0 ? kappa_ - 1 : -kappa_;
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                double product  = value.value[ir] * tangent.value[ir],
                       small    = value.small_value[ir] * tangent.small_value[ir];
                tangent.rho[ir] = 2 * (product + small) / fourpi;
                tangent.tau[ir] =
                        (value.derivative[ir] * tangent.derivative[ir] +
                         value.small_derivative[ir] * tangent.small_derivative[ir] +
                         (l * (l + 1.0) * product + small_l * (small_l + 1.0) * small) / (grid_[ir] * grid_[ir])) /
                        fourpi;
            }
            result.push_back({std::move(value), std::move(tangent)});
        }
        return result;
    }
};

/// Solve XC-dependent Dirac cores on the stored radial field space, reusing a fixed nuclear projector.
inline std::vector<radial_core_state_t>
solve_radial_dirac_core_states(Radial_grid<double> const& grid__, int kappa__, int zn__,
                               std::vector<double> const& potential__, int count__,
                               std::array<std::vector<double>, 2> fields__, int nmtp__,
                               std::vector<double> const* direction__                     = nullptr,
                               std::vector<radial_core_state_t>* response__               = nullptr,
                               std::array<std::vector<double>, 2> xc_directions__         = {},
                               std::shared_ptr<Radial_dirac_xc_solver const>* reference__ = nullptr)
{
    auto finite = [](std::vector<double> const& x, int size) {
        return x.size() == static_cast<size_t>(size) &&
               std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
    };
    if (nmtp__ < 2 || nmtp__ >= grid__.num_points() || count__ <= 0 || !finite(potential__, grid__.num_points()) ||
        bool(direction__) != bool(response__) || (direction__ && !finite(*direction__, grid__.num_points()))) {
        RTE_THROW("Dirac core solve: invalid radial domain, potential or response direction");
    }
    for (int part = 0; part < 2; part++) {
        if (!finite(fields__[part], nmtp__) ||
            (!xc_directions__[part].empty() && (!response__ || !finite(xc_directions__[part], nmtp__)))) {
            RTE_THROW("Dirac core solve: invalid muffin-tin XC covector");
        }
        fields__[part].resize(grid__.num_points(), 0);
        xc_directions__[part].resize(grid__.num_points(), 0);
    }
    auto solver = reference__ ? *reference__ : nullptr;
    if (!solver || !solver->matches(grid__, kappa__, zn__)) {
        solver = std::make_shared<Radial_dirac_xc_solver>(grid__, kappa__, zn__, 0);
    }
    std::vector<radial_core_state_t> result;
    if (response__) {
        auto states = solver->solve_response(count__, potential__, fields__[0], fields__[1], *direction__,
                                             xc_directions__[0], xc_directions__[1]);
        response__->clear();
        for (auto& state : states) {
            result.push_back(std::move(state.value));
            response__->push_back(std::move(state.derivative));
        }
    } else {
        result = solver->solve(count__, potential__, fields__[0], fields__[1]);
    }
    if (reference__) {
        *reference__ = std::move(solver);
    }
    return result;
}

/// Converge the occupied nonrelativistic core states and use the same basis for their response.
inline std::vector<radial_core_state_t>
solve_radial_core_states(Radial_grid<double> const& grid__, int l__, int zn__, std::vector<double> const& potential__,
                         int count__, std::array<std::vector<double>, 2> fields__, int nmtp__,
                         std::vector<double> const* direction__             = nullptr,
                         std::vector<radial_core_state_t>* response__       = nullptr,
                         std::array<std::vector<double>, 2> xc_directions__ = {},
                         relativity_t relativity__ = relativity_t::none, radial_core_reference_t* reference__ = nullptr)
{
    if (nmtp__ < 2 || nmtp__ >= grid__.num_points()) {
        RTE_THROW("radial core solve requires an extended muffin-tin domain");
    }
    if (bool(direction__) != bool(response__) ||
        (direction__ &&
         (direction__->size() != grid__.num_points() ||
          !std::all_of(direction__->begin(), direction__->end(), [](double x) { return std::isfinite(x); })))) {
        RTE_THROW("radial core solve: invalid potential response direction");
    }
    for (auto& field : xc_directions__) {
        if (!field.empty() && (!response__ || field.size() != nmtp__ ||
                               !std::all_of(field.begin(), field.end(), [](double x) { return std::isfinite(x); }))) {
            RTE_THROW("radial core solve: invalid XC response direction");
        }
        if (response__) {
            field.resize(grid__.num_points(), 0);
        }
    }
    for (auto& field : fields__) {
        if (field.size() != nmtp__) {
            RTE_THROW("radial core solve: inconsistent muffin-tin XC derivatives");
        }
        field.resize(grid__.num_points(), 0);
    }
    auto finish = [&](Radial_core_solver const& solver, std::vector<radial_core_state_t> states) {
        if (response__) {
            auto response = solver.solve_response(count__, fields__[0], fields__[1], *direction__, xc_directions__[0],
                                                  xc_directions__[1], &states);
            response__->clear();
            for (int j = 0; j < count__; j++) {
                states[j] = std::move(response[j].value);
                response__->push_back(std::move(response[j].derivative));
            }
        }
        return states;
    };
    bool reuse = reference__ &&
                 (relativity__ == relativity_t::none ? reference__->basis > 6 : reference__->basis == 0) &&
                 reference__->l == l__ && reference__->zn == zn__ && reference__->count == count__ &&
                 reference__->nmtp == nmtp__ && reference__->relativity == relativity__ &&
                 reference__->grid == grid__.values() && reference__->potential == potential__ &&
                 reference__->fields == fields__ && reference__->states.size() == static_cast<size_t>(count__);
    if (reuse) {
        if (!response__) {
            return reference__->states;
        }
        Radial_core_solver solver(grid__, l__, zn__, potential__, reference__->basis, grid__[nmtp__ - 1], false, {},
                                  relativity__);
        return finish(solver, reference__->states);
    }
    if (relativity__ != relativity_t::none) {
        Radial_core_solver solver(grid__, l__, zn__, potential__, 0, grid__[nmtp__ - 1], false, {}, relativity__);
        auto states = solver.solve(count__, fields__[0], fields__[1]);
        if (reference__) {
            *reference__ = {l__,         zn__,     count__, nmtp__, 0, relativity__, grid__.values(),
                            potential__, fields__, states};
        }
        return finish(solver, std::move(states));
    }
    std::vector<radial_core_state_t> previous;
    std::ostringstream convergence;
    for (int nb : {64, 128, 192, 256, 384, 512, 768}) {
        Radial_core_solver solver(grid__, l__, zn__, potential__, nb, grid__[nmtp__ - 1]);
        auto states = solver.solve(count__, fields__[0], fields__[1]);
        if (!previous.empty()) {
            double energy_error{0}, field_error{0};
            int worst_state{0}, worst_ir{0}, worst_part{0};
            for (int j = 0; j < count__; j++) {
                energy_error = std::max(energy_error, std::abs(states[j].energy - previous[j].energy) /
                                                              (1e-10 * std::max(1.0, std::abs(states[j].energy))));
                int part{0};
                for (auto pair : {std::make_pair(&states[j].rho, &previous[j].rho),
                                  std::make_pair(&states[j].tau, &previous[j].tau)}) {
                    double peak = *std::max_element(pair.first->begin(), pair.first->end());
                    for (int ir = 0; ir < nmtp__; ir++) {
                        double error =
                                std::abs((*pair.first)[ir] - (*pair.second)[ir]) / (1e-6 * std::max(peak, 1e-30));
                        if (error > field_error) {
                            field_error = error;
                            worst_state = j;
                            worst_ir    = ir;
                            worst_part  = part;
                        }
                    }
                    part++;
                }
            }
            convergence << " basis=" << nb << " energy=" << energy_error << " fields=" << field_error
                        << " state=" << worst_state << " part=" << worst_part << " r=" << grid__[worst_ir];
            if (std::max(energy_error, field_error) <= 1) {
                if (reference__) {
                    *reference__ = {l__,         zn__,     count__, nmtp__, nb, relativity__, grid__.values(),
                                    potential__, fields__, states};
                }
                return finish(solver, std::move(states));
            }
        }
        previous = std::move(states);
    }
    RTE_THROW("tau-dependent radial core solve did not converge with basis refinement. "
              "Check the radial field grid and XC quadrature resolution; scaled errors:" +
              convergence.str());
    return {};
}

enum class radial_core_overlap_t
{
    stored_p,
    radial_function,
    scalar_dirac
};

/// Remove core overlaps without changing the LAPW boundary value or its first two derivatives.
/** The compact projectors w(r) R_core(r), with w=(1-(r/R_MT)^2)^3, span
 *  the occupied core constraints and vanish through second order at R_MT.
 *  Dirac projection uses both components in one kappa channel, with the full spinor overlap.
 *  Projection is independent of the auxiliary radial energy, so it also applies
 *  to energy derivatives. The caller normalizes the resulting LAPW functions.
 *  The radial_function metric interpolates R_core*(p/r) before integration with r^2,
 *  matching the joint R/S metric used by the full-j LAPW basis.
 *  scalar_dirac constrains a scalar l basis against both occupied Dirac j channels'
 *  large components. Only the temporary projector basis is transformed; physical
 *  two-component core states and their density, tau and energy remain unchanged.
 */
inline void
orthogonalize_radial_to_core(Radial_grid<double> const& grid__, std::vector<radial_core_state_t> const& core__,
                             radial_solver_result_t& radial__,
                             std::vector<radial_core_state_t> const* core_response__ = nullptr,
                             radial_solver_result_t* response__                      = nullptr,
                             radial_core_overlap_t metric__ = radial_core_overlap_t::stored_p, int scalar_l__ = -1)
{
    if (core_response__ && (!response__ || core_response__->size() != core__.size())) {
        RTE_THROW("core projection: inconsistent core response");
    }
    if (core__.empty()) {
        return;
    }
    int n = static_cast<int>(core__.size()), nr = grid__.num_points();
    auto size = static_cast<size_t>(nr);
    if (metric__ == radial_core_overlap_t::scalar_dirac) {
        if (radial__.kappa || scalar_l__ < 0) {
            RTE_THROW("scalar Dirac-core projection requires a scalar valence angular channel");
        }
        auto validate = [&](radial_core_state_t const& state) {
            int l = state.kappa < 0 ? -state.kappa - 1 : state.kappa;
            if (!state.kappa || l != scalar_l__ || state.value.size() < size || state.derivative.size() < size ||
                state.small_value.size() < size || state.small_derivative.size() < size) {
                RTE_THROW("scalar Dirac-core projection: incompatible core angular channel or components");
            }
        };
        std::vector<radial_core_state_t> projectors(n), tangents(n);
        for (int i = 0; i < n; i++) {
            validate(core__[i]);
            if (core_response__) {
                validate((*core_response__)[i]);
                if ((*core_response__)[i].kappa != core__[i].kappa) {
                    RTE_THROW("scalar Dirac-core projection: inconsistent response angular channel");
                }
            }
            for (auto member : {&radial_core_state_t::value, &radial_core_state_t::derivative}) {
                auto const& source = core__[i].*member;
                (projectors[i].*member).assign(source.begin(), source.begin() + nr);
                (tangents[i].*member).resize(nr, 0);
                if (core_response__) {
                    std::copy_n(((*core_response__)[i].*member).begin(), nr, (tangents[i].*member).begin());
                }
            }
        }
        Spline<double> product(grid__);
        std::vector<double> taper(nr);
        for (int ir = 0; ir < nr; ir++) {
            double t  = 1 - std::pow(grid__[ir] / grid__.last(), 2);
            taper[ir] = t * t * t;
        }
        auto overlap = [&](std::vector<double> const& a, std::vector<double> const& b) {
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = a[ir] * b[ir] * taper[ir];
            }
            return product.interpolate().integrate(2);
        };
        // Reorthogonalize before forming the Gram matrix: the two j partners can
        // have almost identical large components, but neither constraint may be dropped.
        for (int i = 0; i < n; i++) {
            double source_norm = overlap(projectors[i].value, projectors[i].value);
            for (int pass = 0; pass < 2; pass++) {
                for (int j = 0; j < i; j++) {
                    double coefficient = overlap(projectors[i].value, projectors[j].value);
                    double tangent     = overlap(tangents[i].value, projectors[j].value) +
                                     overlap(projectors[i].value, tangents[j].value);
                    for (auto member : {&radial_core_state_t::value, &radial_core_state_t::derivative}) {
                        for (int ir = 0; ir < nr; ir++) {
                            (tangents[i].*member)[ir] -=
                                    tangent * (projectors[j].*member)[ir] + coefficient * (tangents[j].*member)[ir];
                            (projectors[i].*member)[ir] -= coefficient * (projectors[j].*member)[ir];
                        }
                    }
                }
            }
            double norm = overlap(projectors[i].value, projectors[i].value);
            if (!std::isfinite(source_norm) || source_norm <= 0 || !std::isfinite(norm) ||
                norm <= std::pow(std::numeric_limits<double>::epsilon(), 2) * source_norm) {
                RTE_THROW("scalar Dirac-core projection: unresolved occupied core constraints");
            }
            double scale = std::sqrt(norm), tangent = overlap(projectors[i].value, tangents[i].value) / norm;
            for (auto member : {&radial_core_state_t::value, &radial_core_state_t::derivative}) {
                for (int ir = 0; ir < nr; ir++) {
                    (tangents[i].*member)[ir] =
                            ((tangents[i].*member)[ir] - tangent * (projectors[i].*member)[ir]) / scale;
                    (projectors[i].*member)[ir] /= scale;
                }
            }
        }
        orthogonalize_radial_to_core(grid__, projectors, radial__, response__ ? &tangents : nullptr, response__,
                                     radial_core_overlap_t::radial_function);
        return;
    }
    int components = radial__.kappa ? 2 : 1;
    bool physical  = metric__ == radial_core_overlap_t::radial_function;
    int power      = physical ? 2 : 1;
    if (radial__.p.size() != size || radial__.rdudr.size() != size ||
        (components == 2 && (radial__.small_p.size() != size || radial__.small_rdudr.size() != size)) ||
        (components == 1 && (!radial__.small_p.empty() || !radial__.small_rdudr.empty())) ||
        (response__ &&
         (response__->kappa != radial__.kappa || response__->p.size() != size || response__->rdudr.size() != size ||
          (components == 2 && (response__->small_p.size() != size || response__->small_rdudr.size() != size))))) {
        RTE_THROW("core projection: inconsistent radial field size");
    }
    auto check_core = [&](std::vector<radial_core_state_t> const& states) {
        for (auto const& state : states) {
            if (state.kappa != radial__.kappa || state.value.size() < size || state.derivative.size() < size ||
                (components == 2 && (state.small_value.size() < size || state.small_derivative.size() < size)) ||
                (components == 1 && (!state.small_value.empty() || !state.small_derivative.empty()))) {
                RTE_THROW("core projection: inconsistent angular channel or missing core orbital component");
            }
        }
    };
    check_core(core__);
    if (core_response__) {
        check_core(*core_response__);
    }
    la::dmatrix<double> overlap(n, n);
    la::dmatrix<double> response_overlap(n, n);
    std::vector<double> rhs(n), weight(nr), derivative(nr);
    std::vector<double> drhs(n, 0);
    double radius = grid__.last();
    for (int ir = 0; ir < nr; ir++) {
        double x = grid__[ir] / radius, t = 1 - x * x;
        weight[ir]     = t * t * t;
        derivative[ir] = -6 * x * t * t / radius;
    }
    Spline<double> product(grid__);
    for (int i = 0; i < n; i++) {
        for (int ir = 0; ir < nr; ir++) {
            product(ir) = core__[i].value[ir] * radial__.p[ir];
            if (components == 2) {
                product(ir) += core__[i].small_value[ir] * radial__.small_p[ir];
            }
            if (physical) {
                product(ir) *= grid__.x_inv(ir);
            }
        }
        rhs[i] = product.interpolate().integrate(power);
        if (response__) {
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = core__[i].value[ir] * response__->p[ir] +
                              (core_response__ ? (*core_response__)[i].value[ir] * radial__.p[ir] : 0);
                if (components == 2) {
                    product(ir) += core__[i].small_value[ir] * response__->small_p[ir] +
                                   (core_response__ ? (*core_response__)[i].small_value[ir] * radial__.small_p[ir] : 0);
                }
                if (physical) {
                    product(ir) *= grid__.x_inv(ir);
                }
            }
            drhs[i] = product.interpolate().integrate(power);
        }
        for (int j = 0; j <= i; j++) {
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = core__[i].value[ir] * core__[j].value[ir];
                if (components == 2) {
                    product(ir) += core__[i].small_value[ir] * core__[j].small_value[ir];
                }
                // Use precisely the overlap functional applied to the stored projector p=r*w*R.
                // Multiplying by r after interpolation would define a different discrete constraint.
                product(ir) *= (physical ? 1 : grid__[ir]) * weight[ir];
            }
            overlap(i, j) = overlap(j, i) = product.interpolate().integrate(power);
        }
    }
    if (response__) {
        copy(overlap, response_overlap);
    }
    int info = la::wrap(la::lib_t::lapack).gesv(n, 1, overlap.at(memory_t::host), n, rhs.data(), n);
    if (info || !std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); })) {
        RTE_THROW("core projection: singular or nonfinite core overlap");
    }
    if (response__) {
        if (core_response__) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int ir = 0; ir < nr; ir++) {
                        product(ir) = (*core_response__)[i].value[ir] * core__[j].value[ir] +
                                      core__[i].value[ir] * (*core_response__)[j].value[ir];
                        if (components == 2) {
                            product(ir) += (*core_response__)[i].small_value[ir] * core__[j].small_value[ir] +
                                           core__[i].small_value[ir] * (*core_response__)[j].small_value[ir];
                        }
                        product(ir) *= (physical ? 1 : grid__[ir]) * weight[ir];
                    }
                    drhs[i] -= product.interpolate().integrate(power) * rhs[j];
                }
            }
        }
        info = la::wrap(la::lib_t::lapack).gesv(n, 1, response_overlap.at(memory_t::host), n, drhs.data(), n);
        if (info || !std::all_of(drhs.begin(), drhs.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("core projection: singular or nonfinite projection response");
        }
    }
    radial__.num_nodes = 0;
    for (int ir = 0; ir < nr; ir++) {
        for (int part = 0; part < components; part++) {
            auto& p  = part ? radial__.small_p : radial__.p;
            auto& dp = part ? radial__.small_rdudr : radial__.rdudr;
            for (int j = 0; j < n; j++) {
                auto const& c = part ? core__[j].small_value : core__[j].value;
                auto const& d = part ? core__[j].small_derivative : core__[j].derivative;
                if (response__) {
                    auto& response_p  = part ? response__->small_p : response__->p;
                    auto& response_dp = part ? response__->small_rdudr : response__->rdudr;
                    double dr         = core_response__ ? (part ? (*core_response__)[j].small_value[ir]
                                                                : (*core_response__)[j].value[ir])
                                                        : 0;
                    double dd         = core_response__ ? (part ? (*core_response__)[j].small_derivative[ir]
                                                                : (*core_response__)[j].derivative[ir])
                                                        : 0;
                    response_p[ir] -= grid__[ir] * weight[ir] * (drhs[j] * c[ir] + rhs[j] * dr);
                    response_dp[ir] -= grid__[ir] * (drhs[j] * (derivative[ir] * c[ir] + weight[ir] * d[ir]) +
                                                     rhs[j] * (derivative[ir] * dr + weight[ir] * dd));
                }
                p[ir] -= rhs[j] * grid__[ir] * weight[ir] * c[ir];
                dp[ir] -= rhs[j] * grid__[ir] * (derivative[ir] * c[ir] + weight[ir] * d[ir]);
            }
            if (!std::isfinite(p[ir]) || !std::isfinite(dp[ir]) ||
                (response__ && (!std::isfinite((part ? response__->small_p : response__->p)[ir]) ||
                                !std::isfinite((part ? response__->small_rdudr : response__->rdudr)[ir])))) {
                RTE_THROW("core projection: nonfinite projected radial function");
            }
        }
        if (ir && radial__.p[ir] * radial__.p[ir - 1] < 0) {
            radial__.num_nodes++;
        }
    }
}

} // namespace sirius

#endif
