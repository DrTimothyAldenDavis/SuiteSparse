// =============================================================================
// === spqr_maxcolnorm =========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Given an m-by-n sparse matrix A, compute the max 2-norm of its columns.

#include "spqr.hpp"

template <typename Int> inline double spqr_private_nrm2 (Int n, double *X, cholmod_common *cc)
{
    double norm ;
    SUITESPARSE_BLAS_dnrm2 (norm, n, X, 1, cc->blas_ok) ;
    return (norm) ;
}

template <typename Int> inline double spqr_private_nrm2 (Int n, Complex *X, cholmod_common *cc)
{
    double norm ;
    SUITESPARSE_BLAS_dznrm2 (norm, n, X, 1, cc->blas_ok) ;
    return (norm) ;
}


// =============================================================================
// === spqr_maxcolnorm =========================================================
// =============================================================================

template <typename Entry, typename Int> double spqr_maxcolnorm
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
)
{
    double norm, maxnorm ;
    Int j, p, len, n, *Ap ;
    Entry *Ax ;

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;

    cc->blas_ok = TRUE ;
    n = A->ncol ;
    Ap = (Int *) A->p ;
    Ax = (Entry *) A->x ;

    maxnorm = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        p = Ap [j] ;
        len = Ap [j+1] - p ;
        norm = spqr_private_nrm2 (len, Ax + p, cc) ;
        maxnorm = MAX (maxnorm, norm) ;
    }

    if (sizeof (SUITESPARSE_BLAS_INT) < sizeof (Int) && !cc->blas_ok)
    {
        ERROR (CHOLMOD_INVALID, "problem too large for the BLAS") ;
        return (EMPTY) ;
    }

    return (maxnorm) ;
}

template double spqr_maxcolnorm <double, int32_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
template double spqr_maxcolnorm <Complex, int32_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
template double spqr_maxcolnorm <double, int64_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
template double spqr_maxcolnorm <Complex, int64_t>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
