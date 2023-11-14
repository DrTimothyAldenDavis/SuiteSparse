//------------------------------------------------------------------------------
// CHOLMOD/Cholesky/cholmod_rcond: estimate the reciprocal of condition number
//------------------------------------------------------------------------------

// CHOLMOD/Cholesky Module.  Copyright (C) 2005-2023, Timothy A. Davis
// All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Return a rough estimate of the reciprocal of the condition number:
// the minimum entry on the diagonal of L (or absolute entry of D for an LDL'
// factorization) divided by the maximum entry (squared for LL').  L can be
// real, complex, or zomplex.  Returns -1 on error, 0 if the matrix is singular
// or has a zero entry on the diagonal of L, 1 if the matrix is 0-by-0, or
// min(diag(L))/max(diag(L)) otherwise.  Never returns NaN; if L has a NaN on
// the diagonal it returns zero instead.
//
// For an LL' factorization,  (min(diag(L))/max(diag(L)))^2 is returned.
// For an LDL' factorization, (min(diag(D))/max(diag(D))) is returned.
//
// The real and zomplex cases are the same, since this method only accesses the
// the diagonal of L for an LL' factorization, or D for LDL' factorization,
// and these entries are always real.

#include "cholmod_internal.h"

#ifndef NCHOLESKY

//------------------------------------------------------------------------------
// t_cholmod_rcond_worker template
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_rcond_worker.c"
#define COMPLEX
#include "t_cholmod_rcond_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_rcond_worker.c"
#define COMPLEX
#include "t_cholmod_rcond_worker.c"

//------------------------------------------------------------------------------
// cholmod_rcond
//------------------------------------------------------------------------------

double CHOLMOD(rcond)       // return rcond estimate
(
    // input:
    cholmod_factor *L,      // factorization to query; not modified
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (L, EMPTY) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, EMPTY) ;
    Common->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // handle special cases
    //--------------------------------------------------------------------------

    if (L->n == 0)
    {
        return (1) ;
    }
    if (L->minor < L->n)
    {
        return (0) ;
    }

    //--------------------------------------------------------------------------
    // compute rcond
    //--------------------------------------------------------------------------

    double rcond = 0 ;

    switch ((L->xtype + L->dtype) % 8)
    {

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            rcond = rs_cholmod_rcond_worker (L) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            rcond = cs_cholmod_rcond_worker (L) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            rcond = rd_cholmod_rcond_worker (L) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            rcond = cd_cholmod_rcond_worker (L) ;
            break ;
    }

    return (rcond) ;
}
#endif

