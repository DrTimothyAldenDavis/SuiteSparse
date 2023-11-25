//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_cat_tests: horzcat and vertcat tests
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

double cat_tests (cholmod_sparse *A_input, cholmod_common *cm)
{

    if (!A_input) return (-1) ;

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    int xtype = A_input->xtype ;
    double maxerr = 0 ;
    double alpha [2] = {3.14158, 2.12345} ;
    cholmod_sparse *C1, *C2, *C3, *C4, *T1, *T2, *E, *G = NULL ;

    //--------------------------------------------------------------------------
    // create test matrices
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(copy_sparse) (A_input, cm) ;
    cholmod_sparse *S = CHOLMOD(copy) (A, 0, 0, cm) ;
    double anorm = CHOLMOD(norm_sparse) (A, 0, cm) ;
    cholmod_sparse *T = CHOLMOD(copy_sparse) (S, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, T, cm) ;

    Int *rset = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    Int *cset = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;
    if (rset != NULL && cset != NULL)
    {
        // rset = 0:nrow-1
        for (Int k = 0 ; k < nrow ; k++)
        {
            rset [k] = k ;
        }
        // cset = 0:ncol-1
        for (Int k = 0 ; k < ncol ; k++)
        {
            cset [k] = k ;
        }
    }

    //--------------------------------------------------------------------------
    // basic horzcat tests: no scaling, no change of stype
    //--------------------------------------------------------------------------

    // C1 = [A A]
    C1 = CHOLMOD(horzcat) (A, A, 2, cm) ;

    // C3 = [A' ; A']'
    T1 = CHOLMOD(transpose) (A, 2, cm) ;
    C2 = CHOLMOD(vertcat) (T1, T1, 2, cm) ;
    C3 = CHOLMOD(transpose) (C2, 2, cm) ;

    // E = C1-C3
    E = CHOLMOD(add) (C1, C3, one, minusone, 2, true, cm) ;
    double cnorm = CHOLMOD(norm_sparse) (C1, 0, cm) ;
    double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, cnorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    // G = C1 (0:nrow-1, 0:ncol-1)
    G = CHOLMOD(submatrix) (C1, rset, nrow, cset, ncol, 2, true, cm) ;

    // E = A-G
    E = CHOLMOD(add) (A, G, one, minusone, 2, true, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;

    // G = C1 (:, 0:ncol-1)
    G = CHOLMOD(submatrix) (C1, NULL, -1, cset, ncol, 2, true, cm) ;

    // E = A-G
    E = CHOLMOD(add) (A, G, one, minusone, 2, true, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;

    //--------------------------------------------------------------------------
    // submatrix: pattern
    //--------------------------------------------------------------------------

    // S1 = [S A]
    cholmod_sparse *S1 = CHOLMOD(horzcat) (S, A, 2, cm) ;
    if (S1 != NULL) { OK (S1->stype == CHOLMOD_PATTERN) ; } ;

    // G = S1 (0:nrow-1, 0:ncol-1)
    G = CHOLMOD(submatrix) (S1, rset, nrow, cset, ncol, 2, true, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, G, cm) ;

    // E = T-G
    E = CHOLMOD(add) (T, G, one, minusone, 1, true, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;

    // G = S1 (:, 0:ncol-1)
    G = CHOLMOD(submatrix) (S1, NULL, -1, cset, ncol, 2, true, cm) ;
    CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, G, cm) ;

    // E = T-G
    E = CHOLMOD(add) (T, G, one, minusone, 1, true, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;

    CHOLMOD(free_sparse) (&S1, cm) ;

    // free matrices

    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;
    CHOLMOD(free_sparse) (&T1, cm) ;

    //--------------------------------------------------------------------------
    // basic vertcat tests: no scaling, no change of stype
    //--------------------------------------------------------------------------

    // C1 = [A ; A]
    C1 = CHOLMOD(vertcat) (A, A, 2, cm) ;

    // C3 = [A' A']'
    T1 = CHOLMOD(transpose) (A, 2, cm) ;
    C2 = CHOLMOD(horzcat) (T1, T1, 2, cm) ;
    C3 = CHOLMOD(transpose) (C2, 2, cm) ;

    // E = C1-C3
    E = CHOLMOD(add) (C1, C3, one, minusone, 2, true, cm) ;
    cnorm = CHOLMOD(norm_sparse) (C1, 0, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, cnorm) ;

    // free matrices
    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;
    CHOLMOD(free_sparse) (&T1, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    //--------------------------------------------------------------------------
    // scaled tests
    //--------------------------------------------------------------------------

    for (int values = 0 ; values <= 1 ; values++)
    {

        // A1 = unsymmetric form of A
        cholmod_sparse *A1 = CHOLMOD(copy) (A, 0, values, cm) ;

        // B = sparse (nrow,ncol)
        cholmod_sparse *B = CHOLMOD(spzeros) (nrow, ncol, 1, xtype + DTYPE,
            cm) ;

        // A2 = alpha*A1 + B
        cholmod_sparse *A2 = CHOLMOD(add) (A1, B, alpha, zero, values, true,
            cm) ;

        // C1 = [A1 A2]
        C1 = CHOLMOD(horzcat) (A1, A2, values, cm) ;

        int mode = values ? 2 : 0 ;

        // C3 = [A1' ; A2']'
        T1 = CHOLMOD(transpose) (A1, mode, cm) ;
        T2 = CHOLMOD(transpose) (A2, mode, cm) ;
        C2 = CHOLMOD(vertcat) (T1, T2, values, cm) ;
        C3 = CHOLMOD(transpose) (C2, mode, cm) ;

        if (!values)
        {
            // convert C1 and C3 to binary
            CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, C1, cm) ;
            CHOLMOD(sparse_xtype) (CHOLMOD_REAL + DTYPE, C3, cm) ;
        }

        // E = C1-C3
        E = CHOLMOD(add) (C1, C3, one, minusone, true, true, cm) ;
        cnorm = CHOLMOD(norm_sparse) (C1, 0, cm) ;
        enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        MAXERR (maxerr, enorm, cnorm) ;

        // free matrices
        CHOLMOD(free_sparse) (&A1, cm) ;
        CHOLMOD(free_sparse) (&A2, cm) ;
        CHOLMOD(free_sparse) (&B, cm) ;
        CHOLMOD(free_sparse) (&C1, cm) ;
        CHOLMOD(free_sparse) (&C2, cm) ;
        CHOLMOD(free_sparse) (&C3, cm) ;
        CHOLMOD(free_sparse) (&T1, cm) ;
        CHOLMOD(free_sparse) (&T2, cm) ;
        CHOLMOD(free_sparse) (&E, cm) ;
    }

    //--------------------------------------------------------------------------
    // submatrix: symmetric upper
    //--------------------------------------------------------------------------

    // C1 = [I1 A]
    cholmod_sparse *I1 = CHOLMOD(speye) (nrow, nrow, xtype + DTYPE, cm) ;
    C1 = CHOLMOD(horzcat) (I1, A, 2, cm) ;
    CHOLMOD(free_sparse) (&I1, cm) ;

    // C2 = [0 I2]
    cholmod_sparse *Z  = CHOLMOD(spzeros) (ncol, nrow, 0, xtype + DTYPE, cm) ;
    cholmod_sparse *I2 = CHOLMOD(speye) (ncol, ncol, xtype + DTYPE, cm) ;
    C2 = CHOLMOD(horzcat) (Z, I2, 2, cm) ;
    CHOLMOD(free_sparse) (&Z, cm) ;
    CHOLMOD(free_sparse) (&I2, cm) ;

    // C3 = [C1 ; C2]
    C3 = CHOLMOD(vertcat) (C1, C2, 2, cm) ;
    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;

    // C4 = symmetric upper copy of C3
    C4 = CHOLMOD(copy) (C3, 1, 2, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;
    if (C4 != NULL) { OK (C4->stype == 1) ; } ;

    if (cset != NULL)
    {
        // cset = nrow:(nrow+ncol-1)
        for (Int k = 0 ; k < ncol ; k++)
        {
            cset [k] = k + nrow ;
        }
    }

    // G = C4 (0:nrow-1, nrow:(nrow+ncol-1))
    G = CHOLMOD(submatrix) (C4, rset, nrow, cset, ncol, 2, true, cm) ;
    CHOLMOD(free_sparse) (&C4, cm) ;

    // E = A-G
    E = CHOLMOD(add) (A, G, one, minusone, 2, true, cm) ;
    enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, anorm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;

    //--------------------------------------------------------------------------
    // free matrices and return result
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nrow, sizeof (Int), rset, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), cset, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&T, cm) ;
    return (maxerr) ;
}

