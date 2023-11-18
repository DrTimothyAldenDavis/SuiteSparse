//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_sym: symmetric permuted transpose
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// C = A' or A(p,p)' if A is symmetric.  The output matrix C must already be
// allocated, with enough space, and with the correct xtype (the same as A,
// or CHOLMOD_PATTERN if only the pattern of C is to be computed.  The dtype
// of C and A must also match.  A must be symmetric; and C is constructed as
// symmetric with the C->stype = -(A->stype). C->packed must be true.
//
// For a method that creates C itself, see cholmod_ptranspose instead.
//
// workspace:  at most 2*nrow

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        return (FALSE) ;                            \
    }

//------------------------------------------------------------------------------
// t_cholmod_transpose_sym_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_transpose_sym_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_transpose_sym_worker.c"

#define COMPLEX
#include "t_cholmod_transpose_sym_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_sym_worker.c"

#define ZOMPLEX
#include "t_cholmod_transpose_sym_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_sym_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_transpose_sym_worker.c"

#define COMPLEX
#include "t_cholmod_transpose_sym_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_sym_worker.c"

#define ZOMPLEX
#include "t_cholmod_transpose_sym_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_sym_worker.c"

//------------------------------------------------------------------------------

int CHOLMOD(transpose_sym)
(
    // input:
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj)
                        // 1: numerical (non-conj.)
                        // <= 0: pattern (with diag)
    Int *Perm,          // permutation for C=A(p,p)', or NULL
    // input/output:
    cholmod_sparse *C,  // output matrix, must be allocated on input
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, FALSE) ;
    RETURN_IF_NULL (C, FALSE) ;
    Common->status = CHOLMOD_OK ;

    mode = RANGE (mode, 0, 2) ;

    if (A->xtype == CHOLMOD_PATTERN || C->xtype == CHOLMOD_PATTERN)
    {
        // A or C is pattern: C must be pattern, so mode can only be zero
        mode = 0 ;
    }

    Int n = A->ncol ;
    if (A->stype == 0 || n != A->nrow)
    {
        ERROR (CHOLMOD_INVALID, "A must be symmetric") ;
        return (FALSE) ;
    }

    if ((C->xtype != ((mode == 0) ? CHOLMOD_PATTERN : A->xtype)) ||
        (C->dtype != A->dtype) || (n != C->ncol) || (n != C->nrow) ||
        (!C->packed))
    {
        ERROR (CHOLMOD_INVALID, "C is invalid") ;
        return (FALSE) ;
    }

    ASSERT (CHOLMOD(dump_sparse) (A, "transpose_sym:A", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    CHOLMOD(allocate_work) (0, ((Perm == NULL) ? 1: 2) * A->ncol, 0, Common) ;
    RETURN_IF_ERROR ;

    Int *Wi = (Int *) Common->Iwork ;   // size n integers
    Int *Pinv = NULL ;
    memset (Wi, 0, n * sizeof (Int)) ;

    //--------------------------------------------------------------------------
    // compute Pinv and make sure Perm is valid
    //--------------------------------------------------------------------------

    if (Perm != NULL)
    {
        Pinv = Wi + n ;                 // size n
        CHOLMOD(set_empty) (Pinv, n) ;
        for (Int k = 0 ; k < n ; k++)
        {
            Int i = Perm [k] ;
            if ((i < 0 || i > n) || (Pinv [i] >= 0))
            {
                ERROR (CHOLMOD_INVALID, "invalid permutation") ;
                return (FALSE) ;
            }
            Pinv [i] = k ;
        }
    }

    //--------------------------------------------------------------------------
    // count the # of entries in each column of C
    //--------------------------------------------------------------------------

    Int *Ap  = (Int *) A->p ;
    Int *Ai  = (Int *) A->i ;
    Int *Anz = (Int *) A->nz ;

    #include "t_cholmod_transpose_sym_template.c"

    //--------------------------------------------------------------------------
    // compute the column pointers of C
    //--------------------------------------------------------------------------

    if (CHOLMOD(cumsum) (C->p, Wi, n) > C->nzmax)
    {
        ERROR (CHOLMOD_INVALID, "C->nzmax is too small") ;
        return (FALSE) ;
    }
    memcpy (Wi, C->p, n * sizeof (Int)) ;

    //--------------------------------------------------------------------------
    // compute the pattern and values of C
    //--------------------------------------------------------------------------

    bool conj = (mode == 2) ;

    switch ((C->xtype + C->dtype) % 8)
    {
        default:
            p_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            if (conj)
            {
                cs_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            else
            {
                cs_t_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            if (conj)
            {
                zs_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            else
            {
                zs_t_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            if (conj)
            {
                cd_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            else
            {
                cd_t_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            if (conj)
            {
                zd_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            else
            {
                zd_t_cholmod_transpose_sym_worker (C, A, Pinv, Wi) ;
            }
            break ;
    }

    //--------------------------------------------------------------------------
    // finalize C and return result
    //--------------------------------------------------------------------------

    C->sorted = (Perm == NULL) ;
    C->stype = - SIGN (A->stype) ;
    ASSERT (CHOLMOD(dump_sparse) (C, "transpose_sym:C", Common) >= 0) ;
    return (TRUE) ;
}

