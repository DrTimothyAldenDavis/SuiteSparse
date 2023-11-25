//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_triplet_to_sparse: convert triplet to sparse
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// t_cholmod_triplet_to_sparse_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_triplet_to_sparse_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_triplet_to_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_triplet_to_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_triplet_to_sparse_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_triplet_to_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_triplet_to_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_triplet_to_sparse_worker.c"

//------------------------------------------------------------------------------
// cholmod_triplet_to_sparse: convert triplet matrix to sparse matrix
//------------------------------------------------------------------------------

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_sparse) (&A, Common) ;     \
        CHOLMOD(free_sparse) (&R, Common) ;     \
        return (NULL) ;                         \
    }

// Converts a triplet matrix T into a sparse matrix A.  The nzmax parameter
// can be used to add additional space in A for future entries.  The # of
// entries that can be held in A is max (nnz (A), nzmax), so pass in nzmax
// as zero if you do not need any additional space for future growth.

// workspace: Iwork (max (nrow,ncol))

cholmod_sparse *CHOLMOD(triplet_to_sparse)      // return sparse matrix A
(
    // input:
    cholmod_triplet *T,     // input triplet matrix
    size_t nzmax,           // allocate space for max(nzmax,nnz(A)) entries
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_TRIPLET_MATRIX_INVALID (T, NULL) ;
    Common->status = CHOLMOD_OK ;
    cholmod_sparse *A = NULL ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    size_t nrow = T->nrow ;
    size_t ncol = T->ncol ;
    size_t nz = T->nnz ;
    Int *Ti = (Int *) T->i ;
    Int *Tj = (Int *) T->j ;
    int stype = T->stype ;

    //--------------------------------------------------------------------------
    // allocate temporary matrix R to hold the transpose of A
    //--------------------------------------------------------------------------

    // R is unpacked so that duplicates can be easily assembled in place.
    // Note that the stype of R is negated, to hold the transpose of A.
    cholmod_sparse *R = CHOLMOD(allocate_sparse) (ncol, nrow, nz,
        /* R is unsorted: */ FALSE, /* R is unpacked: */ FALSE,
        /* stype is flipped: */ -stype, T->xtype + T->dtype, Common) ;
    RETURN_IF_ERROR ;

    Int *Rp  = (Int *) R->p ;
    Int *Rnz = (Int *) R->nz ;

    //--------------------------------------------------------------------------
    // count entries in each column of R, imcluding duplicates
    //--------------------------------------------------------------------------

    // Rnz [0..nrow-1] = 0
    memset (Rnz, 0, nrow * sizeof (Int)) ;

    for (Int k = 0 ; k < nz ; k++)
    {
        // get the entry T(i,j), which becomes R(j,i)
        Int i = Ti [k] ;
        Int j = Tj [k] ;
        if (i < 0 || j < 0 || i >= nrow || j >= ncol)
        {
            ERROR (CHOLMOD_INVALID, "index out of range") ;
            break ;
        }
        if (stype > 0)
        {
            // A will be symmetric, and only its upper triangular part is
            // stored, so R must be lower triangular.  Ensure that entries
            // in the upper triangular part of R are transposed to the lower
            // triangular part, by placing the entry T(i,j) in column
            // MIN(i,j) of R.
            Rnz [MIN (i,j)]++ ;
        }
        else if (stype < 0)
        {
            // See comment above; A is lower triangular so R must be upper.
            Rnz [MAX (i,j)]++ ;
        }
        else
        {
            // T and A are unsymmetric
            Rnz [i]++ ;
        }
    }

    RETURN_IF_ERROR ;   // return if index out of range

    //--------------------------------------------------------------------------
    // Rp = cumulative sum of the row counts, Rnz
    //--------------------------------------------------------------------------

    CHOLMOD(cumsum) (Rp, Rnz, nrow) ;

    //--------------------------------------------------------------------------
    // allocate Iwork workspace for the template work, of size MAX (nrow,ncol)
    //--------------------------------------------------------------------------

    CHOLMOD(alloc_work) (0, MAX (nrow, ncol), 0, 0, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // R = T' using template worker
    //--------------------------------------------------------------------------

    size_t anz = 0 ;

    switch ((T->xtype + T->dtype) % 8)
    {
        default:
            anz = p_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            anz = rs_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            anz = cs_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            anz = zs_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            anz = rd_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            anz = cd_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            anz = zd_cholmod_triplet_to_sparse_worker (T, R, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // allocate the final output matrix A
    //--------------------------------------------------------------------------

    anz = MAX (anz, nzmax) ;
    A = CHOLMOD(allocate_sparse) (nrow, ncol, anz, TRUE, TRUE, stype,
        T->xtype + T->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // A = R'
    //--------------------------------------------------------------------------

    // uses Iwork [0..ncol-1] workspace
    if (stype == 0)
    {
        // unsymmetric transpose
        CHOLMOD(transpose_unsym) (R, 1, NULL, NULL, 0, A, Common) ;
    }
    else
    {
        // symmetric array transpose (not conjugate tranpose if complex)
        CHOLMOD(transpose_sym) (R, 1, NULL, A, Common) ;
    }
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&R, Common) ;
    ASSERT (CHOLMOD(dump_sparse) (A, "triplet_to_sparse:A", Common) >= 0) ;
    return (A) ;
}

