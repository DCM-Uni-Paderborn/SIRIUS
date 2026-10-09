/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file basis_functions_index.hpp
 *
 *  \brief Contains definitiona and implementation of sirius::basis_functions_index class.
 */

#ifndef __BASIS_FUNCTIONS_INDEX_HPP__
#define __BASIS_FUNCTIONS_INDEX_HPP__

#include "core/sht/sht.hpp"
#include "unit_cell/radial_functions_index.hpp"

namespace sirius {

/// Data structure that contains complete information about index of a spheric basis function.
struct basis_function_index_descriptor
{
    /// Total angular momemtum.
    angular_momentum am;
    /// Orbital projection in the l,m expansion; unused for a full-j spinor.
    int m;
    /// Composite l,m index, or -1 for a full-j spinor.
    int lm;
    /// Twice m_j for a full-j spinor, or zero in the l,m expansion.
    int two_mj;
    /// Order of the radial function for a given l (j).
    int order;
    /// Index of local orbital.
    rf_lo_index idxlo;
    /// Index of the radial function or beta projector in the case of pseudo potential.
    rf_index idxrf;
    /// Index of the given basis function.
    /** This is needed if we iterate over basis function and need to find its index. */
    bf_index xi{-1};

    basis_function_index_descriptor(angular_momentum am, int projection, int order, rf_lo_index idxlo, rf_index idxrf,
                                    bool full_j = false)
        : am(am)
        , m(full_j ? 0 : projection)
        , lm(full_j ? -1 : sf::lm(am.l(), projection))
        , two_mj(full_j ? projection : 0)
        , order(order)
        , idxlo(idxlo)
        , idxrf(idxrf)
    {
        if (full_j) {
            if (am.s() == 0 || projection < -am.two_j() || projection > am.two_j() ||
                (projection + am.two_j()) % 2 != 0) {
                RTE_THROW("invalid full-j angular projection");
            }
        } else {
            RTE_ASSERT(m >= -am.l() && m <= am.l());
        }
        RTE_ASSERT(order >= 0);
        RTE_ASSERT(idxrf >= 0);
    }

    /// Complex-harmonic index and angular coefficient for one component of Omega_kappa or Omega_-kappa.
    /** The radial small component and its Dirac phase are not included. A vanishing component has lm=-1. */
    std::pair<int, double>
    spinor_component(int spin__, bool small__ = false) const
    {
        if (two_mj == 0 || (spin__ != 0 && spin__ != 1)) {
            RTE_THROW("spinor component requires a full-j descriptor and spin 0 or 1");
        }
        int l = small__ ? am.two_j() - am.l() : am.l();
        int m = (two_mj + 2 * spin__ - 1) / 2;
        if (m < -l || m > l) {
            return {-1, 0};
        }
        return {sf::lm(l, m), sht::ClebschGordan(l, am.j(), two_mj / 2.0, spin__)};
    }
};

/// A helper class to establish various index mappings for the atomic basis functions.
/** Atomic basis function is a radial function multiplied by a spherical harmonic:
    \f[
      \phi_{\ell m \nu}({\bf r}) = f_{\ell \nu}(r) Y_{\ell m}(\hat {\bf r})
    \f]
    A full-j expansion uses spinor harmonics instead, keeping both j branches distinct.
    Multiple radial functions for each angular channel are allowed. This is reflected by
    the \f$ \nu \f$ index and called "order".
  */
class basis_functions_index
{
  private:
    std::vector<basis_function_index_descriptor> vbd_;

    mdarray<int, 2> index_by_lm_order_;

    int offset_lo_{-1};

    std::vector<int> offset_;

    radial_functions_index indexr_;
    bool full_j_{false};

  public:
    basis_functions_index()
    {
    }

    basis_functions_index(radial_functions_index const& indexr__, bool expand_full_j__)
        : indexr_(indexr__)
        , full_j_(expand_full_j__)
    {
        if (!full_j_) {
            index_by_lm_order_ = mdarray<int, 2>({sf::lmmax(indexr_.lmax()), indexr_.max_order()});
            std::fill(index_by_lm_order_.begin(), index_by_lm_order_.end(), -1);
        }
        for (auto e : indexr_) {
            auto size = this->size();
            if (e.idxlo == 0) {
                offset_lo_ = size;
            }
            auto am = e.am;
            if (full_j_ && am.s() == 0) {
                RTE_THROW("full-j basis expansion requires spin-resolved radial channels");
            }
            offset_.push_back(size);
            int last = full_j_ ? am.two_j() : am.l(), step = full_j_ ? 2 : 1;
            for (int projection = -last; projection <= last; projection += step) {
                vbd_.emplace_back(am, projection, e.order, e.idxlo, e.idxrf, full_j_);
                vbd_.back().xi = bf_index(size);
                if (!full_j_) {
                    index_by_lm_order_(sf::lm(am.l(), projection), e.order) = size;
                }
                size++;
            }
        }
    }
    /// Return total number of MT basis functions.
    inline int
    size() const
    {
        return static_cast<int>(vbd_.size());
    }

    /// Return size of AW part of basis functions in case of LAPW.
    inline auto
    size_aw() const
    {
        if (offset_lo_ == -1) {
            return this->size();
        } else {
            return offset_lo_;
        }
    }

    /// Return size of local-orbital part of basis functions in case of LAPW.
    inline auto
    size_lo() const
    {
        if (offset_lo_ == -1) {
            return 0;
        } else {
            return this->size() - offset_lo_;
        }
    }

    inline int
    index_by_l_m_order(int l, int m, int order) const
    {
        return index_by_lm_order(sf::lm(l, m), order);
    }

    inline int
    index_by_lm_order(int lm, int order) const
    {
        if (full_j_) {
            RTE_THROW("l,m lookup is undefined for a full-j basis");
        }
        return index_by_lm_order_(lm, order);
    }

    /// Reverse mapping within a j-resolved radial channel, with m_j supplied in half-integer units.
    int
    index_by_j_mj_order(angular_momentum am__, int two_mj__, int order__) const
    {
        if (!full_j_ || am__.s() == 0 || am__.l() > indexr_.lmax() || order__ < 0 ||
            order__ >= indexr_.max_order(am__.l()) || two_mj__ < -am__.two_j() || two_mj__ > am__.two_j() ||
            (two_mj__ + am__.two_j()) % 2 != 0) {
            RTE_THROW("invalid full-j basis lookup");
        }
        auto idxrf = indexr_.index_of(am__, order__);
        if (idxrf < 0 || idxrf >= indexr_.size()) {
            RTE_THROW("missing full-j radial channel");
        }
        return offset_[idxrf] + (two_mj__ + am__.two_j()) / 2;
    }

    bool
    full_j() const
    {
        return full_j_;
    }

    inline int
    index_of(rf_index idxrf__) const
    {
        return offset_[idxrf__];
    }

    /// Return descriptor of the given basis function.
    inline auto const&
    operator[](int i) const
    {
        RTE_ASSERT(i >= 0 && i < this->size());
        return vbd_[i];
    }

    inline auto
    begin() const
    {
        return vbd_.begin();
    }

    inline auto
    end() const
    {
        return vbd_.end();
    }

    auto const&
    indexr() const
    {
        return indexr_;
    }
};

inline auto
begin(basis_functions_index const& idx__)
{
    return idx__.begin();
}

inline auto
end(basis_functions_index const& idx__)
{
    return idx__.end();
}

} // namespace sirius

#endif
