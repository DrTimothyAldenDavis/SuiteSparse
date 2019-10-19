// =============================================================================
// === spqr_tol ================================================================
// =============================================================================

// Return the default column 2-norm tolerance

#include "spqr.hpp"

// return the default tol (-1 if error)
template <typename Entry> double spqr_tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
)
{
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    return (20 * ((double) A->nrow + (double) A->ncol) * DBL_EPSILON *
        spqr_maxcolnorm <Entry> (A, cc)) ;
}

template double spqr_tol <Complex>   // return the default tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template double spqr_tol <double>    // return the default tol
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

