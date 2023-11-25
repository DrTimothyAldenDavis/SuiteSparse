//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_dense_tests: dense matrix tests
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

double dense_tests (cholmod_sparse *A_input, cholmod_common *cm)
{

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    int xtype = A_input->xtype ;
    double maxerr = 0 ;

    if (nrow > 100000)
    {
        // test skipped
        return (-1) ;
    }

    cholmod_dense *X = rand_dense (nrow, 4, xtype + DTYPE, cm) ;
    cholmod_dense *Y = CHOLMOD(allocate_dense) (nrow, 4, nrow + 7,
        xtype + DTYPE, cm) ;
    cholmod_dense *Z = CHOLMOD(allocate_dense) (nrow, 4, nrow + 4,
        xtype + DTYPE, cm) ;

    // Y = X (with change of leading dimension)
    CHOLMOD(copy_dense2) (X, Y, cm) ;

    // Z = Y (with another change of leading dimension)
    CHOLMOD(copy_dense2) (Y, Z, cm) ;

    // SX = sparse (X)
    cholmod_sparse *SX = CHOLMOD(dense_to_sparse) (X, true, cm) ;

    // SZ = sparse (Z)
    cholmod_sparse *SZ = CHOLMOD(dense_to_sparse) (Z, true, cm) ;

    // E = SX-SZ
    cholmod_sparse *E = CHOLMOD(add) (SX, SZ, one, minusone, true, true, cm) ;
    double cnorm = CHOLMOD(norm_sparse) (SX, 0, cm) ;
    double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, cnorm) ;

    // free matrices and return results
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&SX, cm) ;
    CHOLMOD(free_sparse) (&SZ, cm) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_dense) (&Y, cm) ;
    CHOLMOD(free_dense) (&Z, cm) ;

    printf ("dense maxerr %g\n", maxerr) ;
    return (maxerr) ;
}

