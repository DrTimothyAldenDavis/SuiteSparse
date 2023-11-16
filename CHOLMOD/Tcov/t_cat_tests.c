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
    cholmod_sparse *C1, *C2, *C3, *T1, *T2, *E ;

    cholmod_sparse *A = CHOLMOD(copy_sparse) (A_input, cm) ;

    //--------------------------------------------------------------------------
    // basic horzcat tests: no scaling, no change of stype
    //--------------------------------------------------------------------------

    // C1 = [A A]
    C1 = CHOLMOD(horzcat) (A, A, true, cm) ;

    // C3 = [A' ; A']'
    T1 = CHOLMOD(transpose) (A, 2, cm) ;
    C2 = CHOLMOD(vertcat) (T1, T1, 2, cm) ;
    C3 = CHOLMOD(transpose) (C2, 2, cm) ;

    // E = C1-C3
    E = CHOLMOD(add) (C1, C3, one, minusone, true, true, cm) ;
    double cnorm = CHOLMOD(norm_sparse) (C1, 0, cm) ;
    double enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
    MAXERR (maxerr, enorm, cnorm) ;

    // free matrices
    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;
    CHOLMOD(free_sparse) (&T1, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    //--------------------------------------------------------------------------
    // basic vertcat tests: no scaling, no change of stype
    //--------------------------------------------------------------------------

    // C1 = [A ; A]
    C1 = CHOLMOD(vertcat) (A, A, true, cm) ;

    // C3 = [A' A']'
    T1 = CHOLMOD(transpose) (A, 2, cm) ;
    C2 = CHOLMOD(horzcat) (T1, T1, 2, cm) ;
    C3 = CHOLMOD(transpose) (C2, 2, cm) ;

    // E = C1-C3
    E = CHOLMOD(add) (C1, C3, one, minusone, true, true, cm) ;
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

    CHOLMOD(free_sparse) (&A, cm) ;

    // return results
    printf ("cat maxerr %g\n", maxerr) ;
    return (maxerr) ;
}

