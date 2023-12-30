//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_test_ops2: test more ops
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

double test_ops2 (cholmod_sparse *A_input)
{

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    int xtype = A_input->xtype ;
    double maxerr = 0 ;

    //--------------------------------------------------------------------------
    // create test matrices
    //--------------------------------------------------------------------------

    // X = random ncol-by-6 matrix
    cholmod_dense *X = rand_dense (ncol, 6, xtype + DTYPE, cm) ;

    // B = sparse (X)
    cholmod_sparse *B = CHOLMOD(dense_to_sparse) (X, true, cm) ;

    // A = sparse (A_input), change stype to 0
    cholmod_sparse *A = CHOLMOD(copy) (A_input, 0, 2, cm) ;

    // P = random nrow-by-nrow permutation
    Int *P_perm = prand (nrow) ;              // RAND
    cholmod_sparse *P = perm_matrix (P_perm, nrow, xtype + DTYPE, cm) ;

    // Q = random ncol-by-ncol permutation
    Int *Q_perm = prand (ncol) ;              // RAND
    cholmod_sparse *Q = perm_matrix (Q_perm, ncol, xtype + DTYPE, cm) ;

    //--------------------------------------------------------------------------
    // C1 = (P*A)*(Q*B)
    //--------------------------------------------------------------------------

    cholmod_sparse *PA = CHOLMOD(ssmult) (P, A, 0, true, true, cm) ;
    cholmod_sparse *QB = CHOLMOD(ssmult) (Q, B, 0, true, false, cm) ;
    cholmod_sparse *C1 = CHOLMOD(ssmult) (PA, QB, 0, true, true, cm) ;
    CHOLMOD(free_sparse) (&PA, cm) ;
    CHOLMOD(free_sparse) (&QB, cm) ;

    //--------------------------------------------------------------------------
    // C2 = P*((A*Q)*B)
    //--------------------------------------------------------------------------

    cholmod_sparse *AQ = CHOLMOD(ssmult) (A, Q, 0, true, true, cm) ;
    cholmod_sparse *AQB = CHOLMOD(ssmult) (AQ, B, 0, true, true, cm) ;
    cholmod_sparse *C2 = CHOLMOD(ssmult) (P, AQB, 0, true, true, cm) ;
    CHOLMOD(free_sparse) (&AQB, cm) ;

    //--------------------------------------------------------------------------
    // E = C1-C2
    //--------------------------------------------------------------------------

    cholmod_sparse *E = CHOLMOD(add) (C1, C2, one, minusone, true, false, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;

    double anorm = CHOLMOD(norm_sparse) (A, 0, cm) ;
    double bnorm = CHOLMOD(norm_sparse) (B, 0, cm) ;
    double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm + bnorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    //--------------------------------------------------------------------------
    // C3 = (P*A*Q)*X
    //--------------------------------------------------------------------------

    // PAQ = P*AQ = P*(A*Q)
    cholmod_sparse *PAQ = CHOLMOD(ssmult) (P, AQ, 0, true, true, cm) ;
    CHOLMOD(free_sparse) (&AQ, cm) ;

    // Y = PAQ*X
    cholmod_dense *Y = CHOLMOD(zeros) (nrow, 6, xtype + DTYPE, cm) ;
    CHOLMOD(sdmult) (PAQ, false, one, zero, X, Y, cm) ;
    CHOLMOD(free_sparse) (&PAQ, cm) ;

    // C3 = sparse (Y)
    cholmod_sparse *C3 = CHOLMOD(dense_to_sparse) (Y, true, cm) ;
    CHOLMOD(free_dense) (&Y, cm) ;

    //--------------------------------------------------------------------------
    // E = C1-C3
    //--------------------------------------------------------------------------

    E = CHOLMOD(add) (C1, C3, one, minusone, true, false, cm) ;
    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;

    double fnorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, fnorm, anorm + bnorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    //--------------------------------------------------------------------------
    // free matrices and return result
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nrow, sizeof (Int), P_perm, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), Q_perm, cm) ;
    CHOLMOD(free_sparse) (&P, cm) ;
    CHOLMOD(free_sparse) (&Q, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_sparse) (&B, cm) ;
    CHOLMOD(free_dense) (&X, cm) ;

    printf ("test_ops2 maxerr %g\n", maxerr) ;
    return (maxerr) ;
}

