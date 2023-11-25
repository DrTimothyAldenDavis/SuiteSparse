//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_error_tests: misc error tests
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

void error_tests (cholmod_sparse *A_input, cholmod_common *cm)
{

    Int nrow = A_input->nrow ;
    Int ncol = A_input->ncol ;
    OK (DTYPE == A_input->dtype) ;
    int xtype = A_input->xtype ;
    int stype = A_input->stype ;
    double maxerr = 0 ;
    int save ;
    int cm_print_save = cm->print ;
    cm->print = 4 ;
    cm->error_handler = NULL ;
    int ok ;

    //--------------------------------------------------------------------------
    // make a copy of the input matrix
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(copy_sparse) (A_input, cm) ;

    int other_xtype ;
    switch (xtype)
    {
        case CHOLMOD_REAL:    other_xtype = CHOLMOD_COMPLEX ; break ;
        case CHOLMOD_COMPLEX: other_xtype = CHOLMOD_REAL ;    break ;
        case CHOLMOD_ZOMPLEX: other_xtype = CHOLMOD_REAL ;    break ;
        default:              other_xtype = xtype ;           break ;
    }

    int other_dtype ;
    switch (DTYPE)
    {
        case CHOLMOD_DOUBLE:  other_dtype = CHOLMOD_SINGLE;   break ;
        case CHOLMOD_SINGLE:  other_dtype = CHOLMOD_DOUBLE;   break ;
    }

    //--------------------------------------------------------------------------
    // matrices with invalid dtypes
    //--------------------------------------------------------------------------

    {

        //----------------------------------------------------------------------
        // cholmod_sparse
        //----------------------------------------------------------------------

        cholmod_sparse *S = CHOLMOD(speye) (2, 2, xtype + DTYPE, cm) ;
        if (S != NULL)
        {
            save = S->dtype ;
            S->dtype = 99 ;
            ok = CHOLMOD(print_sparse) (S, "S:dtype mangled", cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;
            S->dtype = save ;
        }
        CHOLMOD(free_sparse) (&S, cm) ;

        //----------------------------------------------------------------------
        // cholmod_triplet
        //----------------------------------------------------------------------

        cholmod_triplet *T = CHOLMOD(allocate_triplet) (nrow, ncol, 10, stype,
            xtype + DTYPE, cm) ;
        if (T != NULL)
        {
            save = T->dtype ;
            T->dtype = 911 ;
            ok = CHOLMOD(print_triplet) (T, "T:dtype mangled", cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;
            T->dtype = save ;
        }
        CHOLMOD(free_triplet) (&T, cm) ;

        //----------------------------------------------------------------------
        // cholmod_dense
        //----------------------------------------------------------------------

        cholmod_dense *X = CHOLMOD(allocate_dense) (5, 5, 5, xtype + DTYPE,
            cm) ;
        if (X != NULL)
        {
            save = X->dtype ;
            X->dtype = 911 ;
            ok = CHOLMOD(print_dense) (X, "X:dtype mangled", cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;
            X->dtype = save ;
        }
        CHOLMOD(free_dense) (&X, cm) ;

        //----------------------------------------------------------------------
        // cholmod_ensure_dense
        //----------------------------------------------------------------------

        CHOLMOD(ensure_dense) (&X, 5, 5, 5, CHOLMOD_PATTERN + DTYPE, cm) ;
        OK (X == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

    }

    //--------------------------------------------------------------------------
    // matrices with different xtypes
    //--------------------------------------------------------------------------

    if (other_xtype != xtype)
    {

        cholmod_sparse *B = CHOLMOD(copy_sparse) (A, cm) ;
        CHOLMOD(sparse_xtype) (other_xtype + DTYPE, B, cm) ;

        //----------------------------------------------------------------------
        // horzcat
        //----------------------------------------------------------------------

        cholmod_sparse *C = CHOLMOD(horzcat) (A, B, true, cm) ;
        OK (C == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // vertcat
        //----------------------------------------------------------------------

        C = CHOLMOD(vertcat) (A, B, true, cm) ;
        OK (C == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // scale
        //----------------------------------------------------------------------

        cholmod_dense *X = CHOLMOD(ones) (1, 1, other_xtype + DTYPE, cm) ;
        ok = CHOLMOD(scale) (X, CHOLMOD_SCALAR, A, cm) ;
        NOT (ok) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;
        CHOLMOD(free_dense) (&X, cm) ;

        CHOLMOD(free_sparse) (&B, cm) ;
    }

    //--------------------------------------------------------------------------
    // matrices with different dtypes
    //--------------------------------------------------------------------------

    if (xtype == CHOLMOD_REAL)
    {
        cholmod_sparse *R = CHOLMOD(speye) (nrow, 1, xtype + other_dtype, cm) ;
        cholmod_factor *L = CHOLMOD(analyze) (A, cm) ;
        CHOLMOD(factorize) (A, L, cm) ;

        cholmod_sparse *B = CHOLMOD(copy_sparse) (A, cm) ;
        CHOLMOD(sparse_xtype) (xtype + other_dtype, B, cm) ;

        if (cm->status == CHOLMOD_OK)
        {

            //------------------------------------------------------------------
            // super_numeric
            //------------------------------------------------------------------

            int is_super = L->is_super ;
            L->is_super = true ;
            CHOLMOD(super_numeric) (B, B, one, L, cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;
            L->is_super = is_super ;

            //------------------------------------------------------------------
            // rowadd
            //------------------------------------------------------------------

            ok = CHOLMOD(rowadd) (0, R, L, cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;

            //------------------------------------------------------------------
            // updown
            //------------------------------------------------------------------

            ok = CHOLMOD(updown) (1, R, L, cm) ;
            NOT (ok) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;

            //------------------------------------------------------------------
            // spsolve
            //------------------------------------------------------------------

            cholmod_sparse *C = CHOLMOD(spsolve) (CHOLMOD_A, L, R, cm) ;
            OK (C == NULL) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;

            //------------------------------------------------------------------
            // add
            //------------------------------------------------------------------

            C = CHOLMOD(add) (A, B, one, one, true, true, cm) ;
            OK (C == NULL) ;
            OK (cm->status == CHOLMOD_INVALID) ;
            cm->status = CHOLMOD_OK ;

        }

        CHOLMOD(free_sparse) (&B, cm) ;
        CHOLMOD(free_factor) (&L, cm) ;
        CHOLMOD(free_sparse) (&R, cm) ;
    }

    //--------------------------------------------------------------------------
    // matrices with different dimensions
    //--------------------------------------------------------------------------

    cholmod_sparse *S = CHOLMOD(speye) (nrow+1, ncol+1, xtype + DTYPE, cm) ;

    if (S != NULL)
    {

        //---------------------------------------------------------------------
        // add
        //---------------------------------------------------------------------

        cholmod_sparse *C = CHOLMOD(add) (A, S, one, one, true, true, cm) ;
        OK (C == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

    }

    CHOLMOD(free_sparse) (&S, cm) ;

    //--------------------------------------------------------------------------
    // invalid stype
    //--------------------------------------------------------------------------

    if (nrow != ncol)
    {

        //----------------------------------------------------------------------
        // allocate_triplet
        //----------------------------------------------------------------------

        cholmod_triplet *T = CHOLMOD(allocate_triplet) (nrow, ncol, 10, 1,
            xtype + DTYPE, cm) ;
        OK (T == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // copy
        //----------------------------------------------------------------------

        cholmod_sparse *S = CHOLMOD(copy) (A, 1, 0, cm) ;
        OK (S == NULL) ;
        OK (cm->status == CHOLMOD_INVALID) ;
        cm->status = CHOLMOD_OK ;

    }

    //--------------------------------------------------------------------------
    // realloc_multiple
    //--------------------------------------------------------------------------

    size_t siz = 0 ;
    ok = CHOLMOD(realloc_multiple) (2, 2, CHOLMOD_PATTERN + DTYPE,
        NULL, NULL, NULL, NULL, &siz, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_INVALID) ;
    cm->status = CHOLMOD_OK ;

    //--------------------------------------------------------------------------
    // solve2
    //--------------------------------------------------------------------------

    cholmod_factor *L = CHOLMOD(analyze) (A, cm) ;
    ok = CHOLMOD(factorize) (A, L, cm) ;
    OK (ok) ;
    cholmod_dense *B = CHOLMOD(ones) (nrow, 1, xtype + other_dtype, cm) ;
    cholmod_dense *X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
    NOP (X) ;
    OK (cm->status == CHOLMOD_INVALID) ;
    cm->status = CHOLMOD_OK ;
    CHOLMOD(free_dense) (&B, cm) ;

    cholmod_sparse *Xset = NULL ;
    cholmod_dense *Y = NULL, *E = NULL ;
    cholmod_sparse *Bset = CHOLMOD(speye) (nrow, 1, xtype + DTYPE, cm) ;
    B = CHOLMOD(ones) (nrow, 2, xtype + DTYPE, cm) ;
    X = CHOLMOD(ones) (nrow, 1, xtype + DTYPE, cm) ;
    ok = CHOLMOD(solve2) (CHOLMOD_A, L, B, Bset, &X, &Xset, &Y, &E, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_INVALID) ;
    cm->status = CHOLMOD_OK ;
    CHOLMOD(free_dense) (&B, cm) ;

    B = CHOLMOD(ones) (nrow, 1, other_xtype + DTYPE, cm) ;
    ok = CHOLMOD(solve2) (CHOLMOD_A, L, B, Bset, &X, &Xset, &Y, &E, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_INVALID) ;
    cm->status = CHOLMOD_OK ;
    CHOLMOD(free_dense) (&B, cm) ;

    CHOLMOD(free_sparse) (&Xset, cm) ;
    CHOLMOD(free_sparse) (&Bset, cm) ;
    CHOLMOD(free_dense) (&E, cm) ;
    CHOLMOD(free_dense) (&Y, cm) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // free matrices and restore error handling
    //--------------------------------------------------------------------------

    CHOLMOD(free_sparse) (&A, cm) ;
    cm->print = cm_print_save ;
    cm->error_handler = my_handler ;
}

