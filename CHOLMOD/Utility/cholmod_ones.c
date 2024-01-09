//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_ones: dense matrix of all ones
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Create a dense matrix with all entries equal to one, of any xtype or dtype.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (&X, Common) ;      \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_ones_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_ones_worker.c"
#define COMPLEX
#include "t_cholmod_ones_worker.c"
#define ZOMPLEX
#include "t_cholmod_ones_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_ones_worker.c"
#define COMPLEX
#include "t_cholmod_ones_worker.c"
#define ZOMPLEX
#include "t_cholmod_ones_worker.c"

//------------------------------------------------------------------------------
// cholmod_ones: create a dense matrix all equal to 1
//------------------------------------------------------------------------------

cholmod_dense *CHOLMOD(ones)
(
    // input:
    size_t nrow,    // # of rows
    size_t ncol,    // # of columns
    int xdtype,     // xtype + dtype of the matrix:
                    // (CHOLMOD_DOUBLE, _SINGLE) +
                    // (_REAL, _COMPLEX, or _ZOMPLEX)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // allocate the matrix
    //--------------------------------------------------------------------------

    cholmod_dense *X = CHOLMOD(allocate_dense) (nrow, ncol, nrow, xdtype,
        Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's
    //--------------------------------------------------------------------------

    switch (xdtype % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_ones_worker (X) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_ones_worker (X) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_ones_worker (X) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_ones_worker (X) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_ones_worker (X) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_ones_worker (X) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_dense) (X, "ones:X", Common) >= 0) ;
    return (X) ;
}

