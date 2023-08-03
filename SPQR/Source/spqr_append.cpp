// =============================================================================
// === spqr_append =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

// Appends a dense column X onto a sparse matrix A, increasing nnzmax(A) as
// needed.  The column pointer array is not modified; it must be large enough
// to accomodate the new column.

template <typename Entry, typename Int> int spqr_append       // TRUE/FALSE if OK or not
(
    // inputs, not modified
    Entry *X,           // size m-by-1
    Int *P,            // size m, or NULL; permutation to apply to X.
                        // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,  // size m-by-(A->ncol) where A->ncol > n must hold
    Int *p_n,          // n = # of columns of A so far; increased one

    // workspace and parameters
    cholmod_common *cc
)
{
    Entry *Ax ;
    Int *Ai, *Ap ;
    Int nzmax, nz, i, k, nznew, n, m, nz2 ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    m = A->nrow ;
    n = *p_n ;
    Ap = (Int *) A->p ;

    if (m == 0)
    {
        // quick return
        n++ ;
        *p_n = n ;
        Ap [n] = 0 ;
        return (TRUE) ;
    }

    Ai = (Int *) A->i ;
    Ax = (Entry *) A->x ;
    nzmax = A->nzmax ;      // current nzmax(A)
    nz = Ap [n] ;           // current nnz(A)
    PR (("nz %ld nzmax %ld\n", nz, nzmax)) ;
    ASSERT (nz <= nzmax) ;

    // -------------------------------------------------------------------------
    // append X onto A
    // -------------------------------------------------------------------------

    nz2 = spqr_add (nz, m, &ok) ;

    if (ok && nz2 <= nzmax)
    {

        // ---------------------------------------------------------------------
        // A is large enough to hold all of X without reallocating
        // ---------------------------------------------------------------------

        for (k = 0 ; k < m ; k++)
        {
            i = P ? P [k] : k ;
            if (X [i] != (Entry) 0)
            {
                Ai [nz] = k ;
                Ax [nz] = X [i] ;
                nz++ ;
            }
        }

    }
    else
    {

        // ---------------------------------------------------------------------
        // A might need to be increased in size
        // ---------------------------------------------------------------------

        for (k = 0 ; k < m ; k++)
        {
            i = P ? P [k] : k ;
            if (X [i] != (Entry) 0)
            {
                if (nz >= nzmax)
                {
                    // Ai and Ax are not big enough; increase their size.
                    // nznew = 2*nzmax + m ;
                    nznew = spqr_mult <Int> (2, nzmax, &ok) ;
                    nznew = spqr_add (nznew, m, &ok) ;
                    if (!ok || !spqr_reallocate_sparse <Int> (nznew, A, cc))
                    {
                        // out of memory
                        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
                        return (FALSE) ;
                    }
                    // Ai and Ax have moved, reaquire the pointers
                    Ai = (Int *) A->i ;
                    Ax = (Entry *) A->x ;
                    PR (("reallocated from %ld to %ld\n", nzmax, nznew)) ;
                    nzmax = nznew ;
                }
                Ai [nz] = k ;
                Ax [nz] = X [i] ;
                nz++ ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // finalize column pointers
    // -------------------------------------------------------------------------

    PR (("new nz %ld\n", nz)) ;
    n++ ;
    *p_n = n ;

    A->nzmax = nzmax ;
    A->i = Ai ;
    A->x = Ax ;
    Ap [n] = nz ;
    return (TRUE) ;
}


// explicit instantiations

template int spqr_append <double, int32_t>
(
    double *X, int32_t *P, cholmod_sparse *A, int32_t *p_n, cholmod_common *cc
) ;

template int spqr_append <Complex, int32_t>
(
    Complex *X, int32_t *P, cholmod_sparse *A, int32_t *p_n, cholmod_common *cc
) ;

template int spqr_append <double, int64_t>
(
    double *X, int64_t *P, cholmod_sparse *A, int64_t *p_n, cholmod_common *cc
) ;

template int spqr_append <Complex, int64_t>
(
    Complex *X, int64_t *P, cholmod_sparse *A, int64_t *p_n, cholmod_common *cc
) ;
