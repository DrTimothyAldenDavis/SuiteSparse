//------------------------------------------------------------------------------
// CHOLMOD/Supernodal/cholmod_super_solve: solve using supernodal factorization
//------------------------------------------------------------------------------

// CHOLMOD/Supernodal Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Solve Lx=b or L'x=b for a supernodal factorization.  These routines do not
// apply the permutation L->Perm.  See cholmod_solve for a more general
// interface that performs that operation.
//
// L is supernodal, and real or complex (not pattern, nor zomplex).  The xtype
// and dtype of L, X, and E must match.

#include "cholmod_internal.h"

#ifndef NGPL
#ifndef NSUPERNODAL

//------------------------------------------------------------------------------
// t_cholmod_super_solve
//------------------------------------------------------------------------------

#define DOUBLE
#define REAL
#include "t_cholmod_super_solve_worker.c"
#define COMPLEX
#include "t_cholmod_super_solve_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_super_solve_worker.c"
#define COMPLEX
#include "t_cholmod_super_solve_worker.c"

//------------------------------------------------------------------------------
// cholmod_super_lsolve: solve x=L\b
//------------------------------------------------------------------------------

// Solve Lx=b where x and b are of size n-by-nrhs.  b is overwritten by the
// solution x.  On input, b is stored in col-major order with leading dimension
// of d, and on output x is stored in the same manner.
//
// The contents of the workspace E are undefined on both input and output.
//
// workspace: none

int CHOLMOD(super_lsolve)   // TRUE if OK, FALSE if BLAS overflow occured
(
    // input:
    cholmod_factor *L,  // factor to use for the forward solve
    // input/output:
    cholmod_dense *X,   // b on input, solution to Lx=b on output
    // workspace:
    cholmod_dense *E,   // workspace of size nrhs*(L->maxesize)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_NULL (X, FALSE) ;
    RETURN_IF_NULL (E, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (E, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;

    if (L->xtype != X->xtype || L->dtype != X->dtype)
    {
        ERROR (CHOLMOD_INVALID, "L and X must have the same xtype and dtype") ;
        return (FALSE) ;
    }
    if (L->xtype != E->xtype || L->dtype != E->dtype)
    {
        ERROR (CHOLMOD_INVALID, "L and E must have the same xtype and dtype") ;
        return (FALSE) ;
    }

    if (X->d < X->nrow || L->n != X->nrow)
    {
        ERROR (CHOLMOD_INVALID, "X and L dimensions must match") ;
        return (FALSE) ;
    }
    if (E->nzmax < X->ncol * (L->maxesize))
    {
        ERROR (CHOLMOD_INVALID, "workspace E not large enough") ;
        return (FALSE) ;
    }
    if (!(L->is_ll) || !(L->is_super))
    {
        ERROR (CHOLMOD_INVALID, "L not supernodal") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;
    ASSERT (IMPLIES (L->n == 0, L->nsuper == 0)) ;
    if (L->n == 0 || X->ncol == 0)
    {
        // nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // solve Lx=b using template routine
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_super_lsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_super_lsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_super_lsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_super_lsolve_worker (L, X, E, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHECK_FOR_BLAS_INTEGER_OVERFLOW ;
    return (Common->blas_ok) ;
}

//------------------------------------------------------------------------------
// cholmod_super_ltsolve: solve x=L'\b
//------------------------------------------------------------------------------

// Solve L'x=b where x and b are of size n-by-nrhs.  b is overwritten by the
// solution x.  On input, b is stored in col-major order with leading dimension
// of d, and on output x is stored in the same manner.
//
// The contents of the workspace E are undefined on both input and output.
//
// workspace: none

int CHOLMOD(super_ltsolve)  // TRUE if OK, FALSE if BLAS overflow occured
(
    // input:
    cholmod_factor *L,  // factor to use for the backsolve
    // input/output:
    cholmod_dense *X,   // b on input, solution to L'x=b on output
    // workspace:
    cholmod_dense *E,   // workspace of size nrhs*(L->maxesize)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (L, FALSE) ;
    RETURN_IF_NULL (X, FALSE) ;
    RETURN_IF_NULL (E, FALSE) ;
    RETURN_IF_XTYPE_INVALID (L, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (X, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;
    RETURN_IF_XTYPE_INVALID (E, CHOLMOD_REAL, CHOLMOD_COMPLEX, FALSE) ;

    if (L->xtype != X->xtype || L->dtype != X->dtype)
    {
        ERROR (CHOLMOD_INVALID, "L and X must have the same xtype and dtype") ;
        return (FALSE) ;
    }
    if (L->xtype != E->xtype || L->dtype != E->dtype)
    {
        ERROR (CHOLMOD_INVALID, "L and E must have the same xtype and dtype") ;
        return (FALSE) ;
    }

    if (X->d < X->nrow || L->n != X->nrow)
    {
        ERROR (CHOLMOD_INVALID, "X and L dimensions must match") ;
        return (FALSE) ;
    }
    if (E->nzmax < X->ncol * (L->maxesize))
    {
        ERROR (CHOLMOD_INVALID, "workspace E not large enough") ;
        return (FALSE) ;
    }
    if (!(L->is_ll) || !(L->is_super))
    {
        ERROR (CHOLMOD_INVALID, "L not supernodal") ;
        return (FALSE) ;
    }
    Common->status = CHOLMOD_OK ;
    ASSERT (IMPLIES (L->n == 0, L->nsuper == 0)) ;
    if (L->n == 0 || X->ncol == 0)
    {
        // nothing to do
        return (TRUE) ;
    }

    //--------------------------------------------------------------------------
    // solve Lx=b using template routine
    //--------------------------------------------------------------------------

    switch ((L->xtype + L->dtype) % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_super_ltsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_super_ltsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_super_ltsolve_worker (L, X, E, Common) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_super_ltsolve_worker (L, X, E, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHECK_FOR_BLAS_INTEGER_OVERFLOW ;
    return (Common->blas_ok) ;
}

#endif
#endif

