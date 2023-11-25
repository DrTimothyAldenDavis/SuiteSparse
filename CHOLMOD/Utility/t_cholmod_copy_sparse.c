//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy_sparse: copy a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Creates an exact copy of a sparse matrix.  For making a copy with a change
// of stype and/or copying the pattern of a numerical matrix, see cholmod_copy.
// For changing the xtype and/or dtype, see cholmod_sparse_xtype.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_sparse) (&C, Common) ;     \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_copy_sparse_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_copy_sparse_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_copy_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_copy_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_sparse_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_copy_sparse_worker.c"
#define COMPLEX
#include "t_cholmod_copy_sparse_worker.c"
#define ZOMPLEX
#include "t_cholmod_copy_sparse_worker.c"

cholmod_sparse *CHOLMOD(copy_sparse)  // return new sparse matrix
(
    // input:
    cholmod_sparse *A,     // sparse matrix to copy
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_sparse) (A, "copy_sparse:A", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate the copy C with the same characteristcs as A
    //--------------------------------------------------------------------------

    cholmod_sparse *C = CHOLMOD(allocate_sparse) (A->nrow, A->ncol,
        A->nzmax, A->sorted, A->packed, A->stype, A->xtype + A->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (A->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ex = e * ((A->xtype == CHOLMOD_PATTERN) ? 0 :
                    ((A->xtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ez = e * ((A->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // copy the contents from A to C
    //--------------------------------------------------------------------------

    size_t ncol = A->ncol ;

    // copy A->p (both packed and unpacked cases)
    memcpy (C->p, A->p, (ncol+1) * ei) ;

    if (A->packed)
    {
        // use memcpy when A is packed
        int64_t anz = CHOLMOD(nnz) (A, Common) ;
        if (A->i != NULL) memcpy (C->i, A->i, anz * ei) ;
        if (A->x != NULL) memcpy (C->x, A->x, anz * ex) ;
        if (A->z != NULL) memcpy (C->z, A->z, anz * ez) ;
    }
    else
    {
        // copy A->nz (for any xtype and dtype)
        if (A->nz != NULL) memcpy (C->nz, A->nz, ncol * ei) ;

        // use a template worker when A is unpacked to copy A->i, A->x, and A->z
        switch ((A->xtype + A->dtype) % 8)
        {
            default:
                p_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                rs_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                cs_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                zs_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                rd_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                cd_cholmod_copy_sparse_worker (C, A) ;
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                zd_cholmod_copy_sparse_worker (C, A) ;
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (C, "copy_sparse:C", Common) >= 0) ;
    return (C) ;
}

