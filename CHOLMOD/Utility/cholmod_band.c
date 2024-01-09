//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_band: extract the band of a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Construct a sparse matrix C that contains entries in bands k1:k2 of an input
// sparse matrix A.  A can have any xtype, stype, or dtype.  C is sorted if and
// only if A is sorted.

// A can be constructed in place, but only if it is in packed form.
// The diagonal can be ignored, if the ignore_diag flag is true.
// C can optionally be constructed as a pattern matrix.

// The stype is not changed, and no transpose takes place, so a mode of 1 and 2
// have the same effect (unlike cholmod_tranpose, cholmod_copy, cholmod_aat,
// cholmod_vertcat, cholmod_horzcat, ...

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_sparse) (&C2, Common) ;        \
        return (NULL) ;                             \
    }

//------------------------------------------------------------------------------
// t_cholmod_band_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_band_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_band_worker.c"
#define COMPLEX
#include "t_cholmod_band_worker.c"
#define ZOMPLEX
#include "t_cholmod_band_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_band_worker.c"
#define COMPLEX
#include "t_cholmod_band_worker.c"
#define ZOMPLEX
#include "t_cholmod_band_worker.c"

//------------------------------------------------------------------------------
// band_helper
//------------------------------------------------------------------------------

static cholmod_sparse *band_helper
(
    cholmod_sparse *A,
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    bool values,            // if true and A numerical, C is numerical
    bool inplace,           // if true, convert A in place (A cannot be packed)
    bool ignore_diag,       // if true, ignore diagonal
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;

    if (A->stype > 0 && k1 < 0)
    {
        // A is symmetric with the strictly lower triangular part ignored
        k1 = 0 ;
    }
    else if (A->stype < 0 && k2 > 0)
    {
        // A is symmetric with the strictly upper triangular part ignored
        k2 = 0 ;
    }

    // ensure k1 and k2 are in range -nrow:ncol
    k1 = RANGE (k1, -nrow, ncol) ;
    k2 = RANGE (k2, -nrow, ncol) ;

    values = values && (A->xtype != CHOLMOD_PATTERN) ;

    //--------------------------------------------------------------------------
    // allocate new matrix C, or operate on A in place
    //--------------------------------------------------------------------------

    cholmod_sparse *C, *C2 = NULL ;
    if (inplace)
    {
        // convert A in place
        if (!values)
        {
            // change A to pattern
            CHOLMOD(sparse_xtype) (CHOLMOD_PATTERN + A->dtype, A, Common) ;
        }
        C = A ;
    }
    else
    {
        // count # of entries in C and allocate it
        int64_t cnz = CHOLMOD(band_nnz) (A, k1, k2, ignore_diag, Common) ;
        int cxtype = values ? A->xtype : CHOLMOD_PATTERN ;
        C2 = CHOLMOD(allocate_sparse) (nrow, ncol, cnz, A->sorted,
            /* C is packed: */ TRUE, A->stype, cxtype + A->dtype, Common) ;
        C = C2 ;
    }
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // C = band (A)
    //--------------------------------------------------------------------------

    switch ((C->xtype + C->dtype) % 8)
    {
        default:
            p_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_band_worker (C, A, k1, k2, ignore_diag) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // shrink A if computed in-place
    //--------------------------------------------------------------------------

    if (inplace)
    {
        int64_t anz = CHOLMOD(nnz) (A, Common) ;
        CHOLMOD(reallocate_sparse) (anz, A, Common) ;
        RETURN_IF_ERROR ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (int64_t nzdiag = CHOLMOD(dump_sparse) (C, "band:C", Common)) ;
    ASSERT (nzdiag >= 0) ;
    ASSERT (IMPLIES (ignore_diag, nzdiag == 0)) ;
    return (C) ;
}

//------------------------------------------------------------------------------
// cholmod_band
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(band)   // return a new matrix C
(
    // input:
    cholmod_sparse *A,      // input matrix
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    int mode,               // >0: numerical, 0: pattern, <0: pattern (no diag)
    cholmod_common *Common
)
{
    mode = RANGE (mode, -1, 1) ;
    bool values = (mode > 0) ;
    bool inplace = FALSE ;
    bool ignore_diag = (mode < 0) ;
    return (band_helper (A, k1, k2, values, inplace, ignore_diag, Common)) ;
}

//------------------------------------------------------------------------------
// cholmod_band_inplace
//------------------------------------------------------------------------------

int CHOLMOD(band_inplace)
(
    // input:
    int64_t k1,             // count entries in k1:k2 diagonals
    int64_t k2,
    int mode,               // >0: numerical, 0: pattern, <0: pattern (no diag)
    // input/output:
    cholmod_sparse *A,      // input/output matrix
    cholmod_common *Common
)
{
    mode = RANGE (mode, -1, 1) ;
    bool values = (mode > 0) ;
    bool inplace = TRUE ;
    bool ignore_diag = (mode < 0) ;
    if (A != NULL && !(A->packed))
    {
        /* cannot operate on an unpacked matrix in place */
        ERROR (CHOLMOD_INVALID, "cannot operate on unpacked matrix in-place") ;
        return (FALSE) ;
    }
    return (band_helper (A, k1, k2, values, inplace, ignore_diag, Common)
        != NULL) ;
}

