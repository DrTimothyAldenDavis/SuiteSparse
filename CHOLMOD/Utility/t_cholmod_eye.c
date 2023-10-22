//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_eye: dense identity matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Create a dense identity matrix, possibly rectangular, of any xtype or
// dtype.

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                         \
    if (Common->status < CHOLMOD_OK)            \
    {                                           \
        CHOLMOD(free_dense) (&X, Common) ;      \
        return (NULL) ;                         \
    }

//------------------------------------------------------------------------------
// t_cholmod_eye_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_eye_worker.c"
#define COMPLEX
#include "t_cholmod_eye_worker.c"
#define ZOMPLEX
#include "t_cholmod_eye_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_eye_worker.c"
#define COMPLEX
#include "t_cholmod_eye_worker.c"
#define ZOMPLEX
#include "t_cholmod_eye_worker.c"

//------------------------------------------------------------------------------
// cholmod_eye: create a dense identity matrix
//------------------------------------------------------------------------------

cholmod_dense *CHOLMOD(eye)     // return a dense identity matrix
(
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
    // allocate the matrix and set it to all zero
    //--------------------------------------------------------------------------

    cholmod_dense *X = CHOLMOD(zeros) (nrow, ncol, xdtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // fill the matrix with all 1's on the diagonal
    //--------------------------------------------------------------------------

    switch (xdtype % 8)
    {
        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_eye_worker (X) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_eye_worker (X) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_eye_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_eye_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_eye_worker (X) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_eye_worker (X) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_dense) (X, "eye:X", Common) >= 0) ;
    return (X) ;
}

