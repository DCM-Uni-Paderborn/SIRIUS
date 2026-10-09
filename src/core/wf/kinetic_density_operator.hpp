/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __KINETIC_DENSITY_OPERATOR_HPP__
#define __KINETIC_DENSITY_OPERATOR_HPP__

#include <array>
#include "core/fft/fft.hpp"
#include "core/fft/gvec.hpp"

namespace sirius::wf {

/// Smooth kinetic-energy density and its variational operator on the same FFT grid.
/** For one spin component, tau = 1/2 sum_nk f_nk w_k |grad psi_nk|^2.
 *  Coefficients use the SIRIUS wave-function normalization. add_density() takes
 *  f_nk w_k / Omega, whereas add_potential() applies -1/2 D_k . v_tau D_k
 *  without occupation or cell-volume factors, with D_k = grad + i k.
 *
 *  Host buffers are used with synchronous CPU or GPU FFTs. Complex k-point and
 *  real Gamma-point wave functions are supported. All FFT ranks must call the
 *  methods for the same band. It neither reduces occupations over k-points nor
 *  supplies the atom-local PAW or muffin-tin/core contributions.
 */
template <typename T>
class Kinetic_density_operator
{
  private:
    fft::spfft_transform_type<T>& fft_;
    fft::Gvec_fft const& gkvec_;
    mdarray<std::complex<T>, 1> work_;

    void
    gradient(std::complex<T> const* phi__, int direction__)
    {
        if (fft_.execution_mode() != SPFFT_EXEC_SYNCHRONOUS) {
            RTE_THROW("kinetic-density operator requires synchronous FFTs for its host buffers");
        }
        // i(G+k) preserves Hermitian symmetry for the real Gamma-point FFT.
        #pragma omp parallel for schedule(static)
        for (int ig = 0; ig < gkvec_.count(); ig++) {
            work_[ig] = std::complex<T>(0, static_cast<T>(gkvec_.gkvec_cart(ig)[direction__])) * phi__[ig];
        }
        fft_.backward(reinterpret_cast<T const*>(work_.at(memory_t::host)), SPFFT_PU_HOST);
    }

  public:
    Kinetic_density_operator(fft::spfft_transform_type<T>& fft__, fft::Gvec_fft const& gkvec__)
        : fft_(fft__)
        , gkvec_(gkvec__)
        , work_({gkvec__.count()})
    {
        if (fft_.execution_mode() != SPFFT_EXEC_SYNCHRONOUS) {
            RTE_THROW("kinetic-density operator requires synchronous FFTs for its host buffers");
        }
        if (fft_.num_local_elements() != gkvec_.count()) {
            RTE_THROW("kinetic-density operator: inconsistent FFT and G-vector sizes");
        }
        bool real_fft = fft_.type() == SPFFT_TRANS_R2C;
        if (real_fft != gkvec_.gvec().reduced()) {
            RTE_THROW("kinetic-density operator: inconsistent real FFT and G-vector reduction");
        }
        if (real_fft && gkvec_.gvec().vk().length() != 0) {
            RTE_THROW("a real kinetic-density FFT requires the Gamma point");
        }
    }

    /// Accumulate one occupied wave-function contribution into a host real-space array.
    void
    add_density(T weight__, std::complex<T> const* phi__, T* tau__)
    {
        for (int x = 0; x < 3; x++) {
            gradient(phi__, x);
            auto ptr = fft_.space_domain_data(SPFFT_PU_HOST);
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft_.local_slice_size(); ir++) {
                T square = fft_.type() == SPFFT_TRANS_R2C
                                   ? ptr[ir] * ptr[ir]
                                   : std::norm(reinterpret_cast<std::complex<T> const*>(ptr)[ir]);
                tau__[ir] += T(0.5) * weight__ * square;
            }
        }
    }

    /// Accumulate the fixed-occupation tau variation from an orbital direction.
    void
    add_density_response(T weight__, std::complex<T> const* phi__, std::complex<T> const* direction__, T* tau__)
    {
        std::vector<std::complex<T>> base(fft_.local_slice_size());
        bool real_fft = fft_.type() == SPFFT_TRANS_R2C;
        for (int x = 0; x < 3; x++) {
            gradient(phi__, x);
            auto ptr = fft_.space_domain_data(SPFFT_PU_HOST);
            for (int ir = 0; ir < fft_.local_slice_size(); ir++) {
                base[ir] = real_fft ? std::complex<T>(ptr[ir], 0) : reinterpret_cast<std::complex<T> const*>(ptr)[ir];
            }
            gradient(direction__, x);
            ptr = fft_.space_domain_data(SPFFT_PU_HOST);
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft_.local_slice_size(); ir++) {
                auto d = real_fft ? std::complex<T>(ptr[ir], 0) : reinterpret_cast<std::complex<T> const*>(ptr)[ir];
                tau__[ir] += weight__ * std::real(std::conj(base[ir]) * d);
            }
        }
    }

    /// Accumulate a spinor's tau in Field4D order: scalar, z, x, y.
    /** Inputs and outputs are host arrays; spinors require a complex FFT, including at Gamma. */
    void
    add_spinor_density(T weight__, std::array<std::complex<T> const*, 2> phi__, std::array<T*, 4> tau__)
    {
        if (fft_.type() != SPFFT_TRANS_C2C) {
            RTE_THROW("spinor kinetic density requires a complex FFT");
        }
        std::vector<std::complex<T>> up(fft_.local_slice_size());
        for (int x = 0; x < 3; x++) {
            gradient(phi__[0], x);
            auto ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
            if (!up.empty()) {
                std::copy(ptr, ptr + up.size(), up.begin());
            }
            gradient(phi__[1], x);
            ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft_.local_slice_size(); ir++) {
                T uu = T(0.5) * weight__ * std::norm(up[ir]), dd = T(0.5) * weight__ * std::norm(ptr[ir]);
                auto cross = weight__ * std::conj(up[ir]) * ptr[ir];
                tau__[0][ir] += uu + dd;
                tau__[1][ir] += uu - dd;
                tau__[2][ir] += cross.real();
                tau__[3][ir] += cross.imag();
            }
        }
    }

    /// Accumulate a spinor orbital direction in the same scalar/z/x/y convention.
    void
    add_spinor_density_response(T weight__, std::array<std::complex<T> const*, 2> phi__,
                                std::array<std::complex<T> const*, 2> direction__, std::array<T*, 4> tau__)
    {
        if (fft_.type() != SPFFT_TRANS_C2C) {
            RTE_THROW("spinor kinetic-density response requires a complex FFT");
        }
        int nr = fft_.local_slice_size();
        std::array<std::vector<std::complex<T>>, 2> base;
        std::vector<std::complex<T>> up(nr);
        for (auto& values : base) {
            values.resize(nr);
        }
        for (int x = 0; x < 3; x++) {
            for (int s = 0; s < 2; s++) {
                gradient(phi__[s], x);
                auto ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
                if (nr) {
                    std::copy(ptr, ptr + nr, base[s].begin());
                }
            }
            gradient(direction__[0], x);
            auto ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
            if (nr) {
                std::copy(ptr, ptr + nr, up.begin());
            }
            gradient(direction__[1], x);
            ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < nr; ir++) {
                T uu       = weight__ * std::real(std::conj(base[0][ir]) * up[ir]);
                T dd       = weight__ * std::real(std::conj(base[1][ir]) * ptr[ir]);
                auto cross = weight__ * (std::conj(up[ir]) * base[1][ir] + std::conj(base[0][ir]) * ptr[ir]);
                tau__[0][ir] += uu + dd;
                tau__[1][ir] += uu - dd;
                tau__[2][ir] += cross.real();
                tau__[3][ir] += cross.imag();
            }
        }
    }

    /// Apply -1/2 D_k . (v_0 I + v . sigma) D_k in Field4D order.
    void
    add_spinor_potential(std::array<T const*, 4> vtau__, std::array<std::complex<T> const*, 2> phi__,
                         std::array<std::complex<T>*, 2> hphi__)
    {
        if (fft_.type() != SPFFT_TRANS_C2C) {
            RTE_THROW("spinor kinetic-density potential requires a complex FFT");
        }
        int overlap = gkvec_.count() && (hphi__[0] == hphi__[1] || phi__[0] == hphi__[0] || phi__[1] == hphi__[0] ||
                                         phi__[0] == hphi__[1] || phi__[1] == hphi__[1]);
        mpi::Communicator(fft_.communicator()).allreduce<int, mpi::op_t::max>(&overlap, 1);
        if (overlap) {
            RTE_THROW("spinor kinetic-density operator requires distinct input and output arrays");
        }
        int nr = fft_.local_slice_size();
        std::array<std::vector<std::complex<T>>, 2> grad;
        for (auto& values : grad) {
            values.resize(nr);
        }
        for (int x = 0; x < 3; x++) {
            for (int s = 0; s < 2; s++) {
                gradient(phi__[s], x);
                auto ptr = reinterpret_cast<std::complex<T> const*>(fft_.space_domain_data(SPFFT_PU_HOST));
                if (nr) {
                    std::copy(ptr, ptr + nr, grad[s].begin());
                }
            }
            for (int s = 0; s < 2; s++) {
                auto ptr = reinterpret_cast<std::complex<T>*>(fft_.space_domain_data(SPFFT_PU_HOST));
                #pragma omp parallel for schedule(static)
                for (int ir = 0; ir < nr; ir++) {
                    auto off_diagonal = std::complex<T>(vtau__[2][ir], (2 * s - 1) * vtau__[3][ir]);
                    ptr[ir]           = (vtau__[0][ir] + (1 - 2 * s) * vtau__[1][ir]) * grad[s][ir] +
                              off_diagonal * grad[1 - s][ir];
                }
                fft_.forward(SPFFT_PU_HOST, reinterpret_cast<T*>(work_.at(memory_t::host)), SPFFT_FULL_SCALING);
                #pragma omp parallel for schedule(static)
                for (int ig = 0; ig < gkvec_.count(); ig++) {
                    hphi__[s][ig] += std::complex<T>(0, static_cast<T>(-0.5 * gkvec_.gkvec_cart(ig)[x])) * work_[ig];
                }
            }
        }
    }

    /// Add the generalized-Kohn-Sham contribution of a real v_tau to host PW coefficients.
    /** Input and output coefficient arrays must not overlap. The FFT workspace is overwritten. */
    void
    add_potential(T const* vtau__, std::complex<T> const* phi__, std::complex<T>* hphi__,
                  std::complex<T> scale__ = std::complex<T>(1, 0))
    {
        int overlap = gkvec_.count() && phi__ == hphi__;
        mpi::Communicator(fft_.communicator()).allreduce<int, mpi::op_t::max>(&overlap, 1);
        if (overlap) {
            RTE_THROW("kinetic-density operator does not support in-place application");
        }
        for (int x = 0; x < 3; x++) {
            gradient(phi__, x);
            fft::spfft_multiply<T>(fft_, [&](int ir) { return vtau__[ir]; });
            fft_.forward(SPFFT_PU_HOST, reinterpret_cast<T*>(work_.at(memory_t::host)), SPFFT_FULL_SCALING);
            #pragma omp parallel for schedule(static)
            for (int ig = 0; ig < gkvec_.count(); ig++) {
                hphi__[ig] += scale__ * std::complex<T>(0, static_cast<T>(-0.5 * gkvec_.gkvec_cart(ig)[x])) * work_[ig];
            }
        }
    }
};

} // namespace sirius::wf

#endif
