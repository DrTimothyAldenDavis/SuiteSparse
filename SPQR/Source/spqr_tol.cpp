// =============================================================================
// === spqr_tol ================================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Return the default column 2-norm tolerance


#include <limits>
#include <algorithm>
#include "spqr.hpp"

// return the default tol (-1 if error)
template <typename Entry, typename Int> double spqr_tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    double tol = (20 * ((double) A->nrow + (double) A->ncol) * DBL_EPSILON *
                  spqr_maxcolnorm <Entry, Int> (A, cc));
    // MathWorks modification: if the tolerance becomes Inf, replace it with
    // realmax; otherwise, we may end up with an all-zero matrix R
    // (see g1284493)
    tol = std::min(tol, std::numeric_limits<double>::max());
    
    return (tol) ;
}

template double spqr_tol <double, int32_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
template double spqr_tol <Complex, int32_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template double spqr_tol <double, int64_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
template double spqr_tol <Complex, int64_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
