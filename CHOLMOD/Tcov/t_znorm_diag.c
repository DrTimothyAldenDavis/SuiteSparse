//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_znorm_diag: compute norm (imag (diag (A)))
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// r = norm (imag (diag (A)))

double znorm_diag (cholmod_sparse *A, cholmod_common *cm) ;

double znorm_diag (cholmod_sparse *A, cholmod_common *cm)
{
    double one [2] = {1,0} ;
    double minusone [2] = {-1,0} ;

    // D1 = diag (A), converted to zomplex
    cholmod_sparse *D1 = CHOLMOD(band) (A, 0, 0, true, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_ZOMPLEX + DTYPE, D1, cm) ;

    // D2 = zomplex (real (D1))
    cholmod_sparse *D2 = CHOLMOD(copy_sparse) (D1, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, D2, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_ZOMPLEX + DTYPE, D2, cm) ;

    // G = D1-D2 = imaginary part of the diagonal of A
    cholmod_sparse *G = CHOLMOD(add) (D1, D2, one, minusone, true, false, cm) ;

    // r is zero if the diagonal of A is all real
    double r = CHOLMOD(norm_sparse) (G, 0, cm) ;
    printf ("norm(G) (G = imag(diag(A))) : %g\n", r) ;

    CHOLMOD(free_sparse) (&G, cm) ;
    CHOLMOD(free_sparse) (&D2, cm) ;
    CHOLMOD(free_sparse) (&D1, cm) ;
    return (r) ;
}

