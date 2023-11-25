//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_dtype_tests: tests with changing dtype
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

double dtype_tests (cholmod_sparse *A_input, cholmod_common *cm)
{

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    int xtype = A_input->xtype ;
    double maxerr = 0 ;

    // A = double(single(A_input)) or A = single(double(A_input))
    int other = (DTYPE == CHOLMOD_SINGLE) ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE ;
    cholmod_sparse *A = CHOLMOD(copy_sparse) (A_input, cm) ;
    CHOLMOD(sparse_xtype) (xtype + other, A, cm) ;
    CHOLMOD(sparse_xtype) (xtype + DTYPE, A, cm) ;

    // E = A_input - A
    cholmod_sparse *E = CHOLMOD(add) (A_input, A, one, minusone,
        true, true, cm) ;
    double anorm = CHOLMOD(norm_sparse) (A_input, 0, cm) ;
    double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;

    printf ("dtype maxerr %g\n", maxerr) ;
    OK (maxerr < 1e-5) ;

    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    return (maxerr) ;
}

