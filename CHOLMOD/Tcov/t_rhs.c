//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_rhs: create a right-hand side
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Create a right-hand-side, b = A*x, where x is a known solution

cholmod_dense *rhs (cholmod_sparse *A, Int nrhs, Int d, int tweak)
{
    Int n ;
    cholmod_dense *W, *Z, *B ;

    if (A == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute rhs") ;
        return (NULL) ;
    }

    n = A->nrow ;

    // B = zeros (n,rhs) but with leading dimension d
    B = zeros (n, nrhs, d, A->xtype + DTYPE) ;

    //--------------------------------------------------------------------------
    // create a known solution
    //--------------------------------------------------------------------------

    Z = xtrue (n, nrhs, d, A->xtype + DTYPE, tweak) ;

    //--------------------------------------------------------------------------
    // compute B = A*Z or A*A'*Z
    //--------------------------------------------------------------------------

    if (A->stype == 0)
    {
        // W = A'*Z
        W  = CHOLMOD(zeros) (A->ncol, nrhs, A->xtype + DTYPE, cm) ;
        CHOLMOD(sdmult) (A, TRUE, one, zero, Z, W, cm) ;
        // B = A*W
        CHOLMOD(sdmult) (A, FALSE, one, zero, W, B, cm) ;
        CHOLMOD(free_dense) (&W, cm) ;
    }
    else
    {
        // B = A*Z
        CHOLMOD(sdmult) (A, FALSE, one, zero, Z, B, cm) ;
    }
    CHOLMOD(free_dense) (&Z, cm) ;
    return (B) ;
}

