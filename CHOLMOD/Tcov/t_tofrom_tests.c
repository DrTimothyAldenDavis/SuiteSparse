//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_tofrom_tests: convert to/from sparse/dense/triplet
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

double tofrom_tests (cholmod_sparse *A_input, cholmod_common *cm)
{

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    int xtype = A_input->xtype ;
    double maxerr = 0 ;
    int64_t anz = CHOLMOD(nnz) (A_input, cm) ;

// FIXME
// int ss = cm->print ;
// cm->print = 5 ;
// CHOLMOD(print_sparse) (A_input, "A tofrom tests", cm) ;
// cm->print = ss ;

    if (nrow > 1000 || ncol > 1000)
    {
        // test skipped
        return (-1) ;
    }

    //--------------------------------------------------------------------------
    // test many conversions
    //--------------------------------------------------------------------------

    for (int to_xtype = 0 ; to_xtype <= 3 ; to_xtype++)
    {

        cholmod_sparse *A = CHOLMOD(copy_sparse) (A_input, cm) ;
        CHOLMOD(sparse_xtype) (to_xtype + DTYPE, A, cm) ;
        cholmod_dense *Y = CHOLMOD(sparse_to_dense) (A, cm) ;

        cholmod_triplet *T = CHOLMOD(sparse_to_triplet) (A, cm) ;
        cholmod_sparse *B = CHOLMOD(triplet_to_sparse) (T, anz, cm) ;
        cholmod_dense *X = CHOLMOD(sparse_to_dense) (B, cm) ;
        cholmod_sparse *C = CHOLMOD(dense_to_sparse) (X, true, cm) ;
        cholmod_sparse *G = CHOLMOD(dense_to_sparse) (Y, true, cm) ;

        if (to_xtype == CHOLMOD_PATTERN)
        {
            CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, A, cm) ;
            CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, C, cm) ;
        }

        // E = A-C
        cholmod_sparse *E = CHOLMOD(add) (A, C, one, minusone, true, true, cm) ;
        double anorm = CHOLMOD(norm_sparse) (A, 0, cm) ;
        double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        MAXERR (maxerr, enorm, anorm) ;
        CHOLMOD(free_sparse) (&E, cm) ;

        // E = A-G
        E = CHOLMOD(add) (A, G, one, minusone, true, true, cm) ;
        enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        MAXERR (maxerr, enorm, anorm) ;
        CHOLMOD(free_sparse) (&E, cm) ;

        CHOLMOD(free_sparse) (&A, cm) ;
        CHOLMOD(free_sparse) (&B, cm) ;
        CHOLMOD(free_sparse) (&C, cm) ;
        CHOLMOD(free_sparse) (&G, cm) ;
        CHOLMOD(free_triplet) (&T, cm) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Y, cm) ;
    }

    return (maxerr) ;
}

