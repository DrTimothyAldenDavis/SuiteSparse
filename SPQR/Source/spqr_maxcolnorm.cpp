// =============================================================================
// === spqr_maxcolnorm =========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Given an m-by-n sparse matrix A, compute the max 2-norm of its columns.

#include "spqr.hpp"

inline double spqr_private_nrm2 (int64_t n, double *X, cholmod_common *cc)
{
    double norm ;
    SUITESPARSE_BLAS_dnrm2 (norm, n, X, 1, cc->blas_ok) ;
    return (norm) ;
}

inline double spqr_private_nrm2 (int64_t n, Complex *X, cholmod_common *cc)
{
    double norm ;
    SUITESPARSE_BLAS_dznrm2 (norm, n, X, 1, cc->blas_ok) ;
    return (norm) ;
}


// =============================================================================
// === spqr_maxcolnorm =========================================================
// =============================================================================

template <typename Entry> double spqr_maxcolnorm
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
)
{
    double norm, maxnorm ;
    int64_t j, p, len, n, *Ap ;
    Entry *Ax ;

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;

    cc->blas_ok = TRUE ;
    n = A->ncol ;
    Ap = (int64_t *) A->p ;
    Ax = (Entry *) A->x ;

    maxnorm = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        p = Ap [j] ;
        len = Ap [j+1] - p ;
        norm = spqr_private_nrm2 (len, Ax + p, cc) ;
        maxnorm = MAX (maxnorm, norm) ;
    }

    if (sizeof (SUITESPARSE_BLAS_INT) < sizeof (int64_t) && !cc->blas_ok)
    {
        ERROR (CHOLMOD_INVALID, "problem too large for the BLAS") ;
        return (EMPTY) ;
    }

    return (maxnorm) ;
}

// =============================================================================

template double spqr_maxcolnorm <double>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;

template double spqr_maxcolnorm <Complex>
(
    // inputs, not modified
    cholmod_sparse *A,

    // workspace and parameters
    cholmod_common *cc
) ;
