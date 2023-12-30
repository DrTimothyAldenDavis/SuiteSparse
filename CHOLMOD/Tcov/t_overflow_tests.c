//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_overflow_tests: integer overflow tests
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

void overflow_tests (cholmod_common *cm)
{

    //--------------------------------------------------------------------------
    // cholmod_read_triplet
    //--------------------------------------------------------------------------

    FILE *f = fopen ("Matrix/int_overflow.tri", "r") ;
    if (f != NULL)
    {
        cholmod_triplet *T = CHOLMOD(read_triplet) (f, cm) ;
        OK (cm->status = CHOLMOD_TOO_LARGE) ;
        OK (T == NULL) ;
    }

    //--------------------------------------------------------------------------
    // AMD and CAMD
    //--------------------------------------------------------------------------

    #ifdef CHOLMOD_INT64
    {
        int64_t n = 1 ;
        int64_t Ap [2] = { 0, INT64_MAX } ;
        int64_t Ai [1] = { 0 } ;
        int64_t P [1] = { 0 } ;
        int result = amd_l_order (n, Ap, Ai, P, NULL, NULL) ;
        OK (result == AMD_OUT_OF_MEMORY) ;
        result = camd_l_order (n, Ap, Ai, P, NULL, NULL, NULL) ;
        OK (result == AMD_OUT_OF_MEMORY) ;
    }
    #else
    {
        int32_t n = 1 ;
        int32_t Ap [2] = { 0, INT32_MAX/2 } ;
        int32_t Ai [1] = { 0 } ;
        int32_t P [1] = { 0 } ;
        int result = amd_order (n, Ap, Ai, P, NULL, NULL) ;
        OK (result == AMD_OUT_OF_MEMORY) ;
        result = camd_order (n, Ap, Ai, P, NULL, NULL, NULL) ;
        OK (result == AMD_OUT_OF_MEMORY) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // cholmod_amd, cholmod_camd, ...
    //--------------------------------------------------------------------------

    cholmod_sparse *A = CHOLMOD(spzeros) (1, 1, 1, CHOLMOD_REAL + DTYPE, cm) ;
    cholmod_sparse *R = CHOLMOD(spzeros) (1, 1, 1, CHOLMOD_REAL + DTYPE, cm);
    cholmod_sparse *C = CHOLMOD(spzeros) (1, 8, 1, CHOLMOD_REAL + DTYPE, cm);
    cholmod_factor *L = CHOLMOD(alloc_factor) (1, DTYPE, cm) ;
    cm->print = 5 ;
    CHOLMOD(print_sparse) (C, "C", cm) ;

    if (A != NULL && R != NULL && L != NULL)
    {
        Int P [1] ;

        size_t n_save = L->n ;
        bool is_super_save = L->is_super ;
        size_t ncol_save = A->ncol ;
        size_t nrow_save = A->nrow ;
        size_t R_nrow_save = R->nrow ;
        size_t C_nrow_save = C->nrow ;
        int stype_save = A->stype ;
        int L_xtype_save = L->xtype ;
        int A_xtype_save = A->xtype ;

        A->nrow = INT64_MAX ;

        //----------------------------------------------------------------------
        // cholmod_amd
        //----------------------------------------------------------------------

        int ok = CHOLMOD(amd) (A, NULL, 0, P, cm) ;
        printf ("cholmod_amd result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_camd
        //----------------------------------------------------------------------

        ok = CHOLMOD(camd) (A, NULL, 0, NULL, P, cm) ;
        printf ("cholmod_camd result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_metis
        //----------------------------------------------------------------------

        ok = CHOLMOD(metis) (A, NULL, 0, false, P, cm) ;
        printf ("cholmod_metis result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_bisect
        //----------------------------------------------------------------------

        A->ncol = SIZE_MAX ;

        int64_t nsep = CHOLMOD(bisect) (A, NULL, 0, false, P, cm) ;
        printf ("cholmod_bisect result: %ld\n", nsep) ;
        OK (nsep == EMPTY) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->ncol = ncol_save ;

        //----------------------------------------------------------------------
        // cholmod_nested_dissection
        //----------------------------------------------------------------------

        nsep = CHOLMOD(nested_dissection) (A, NULL, 0, P, P, P, cm) ;
        printf ("cholmod_nested_dissection result: %ld\n", nsep) ;
        OK (nsep == EMPTY) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_analyze
        //----------------------------------------------------------------------

        cholmod_factor *Lbad = CHOLMOD(analyze) (A, cm) ;
        printf ("cholmod_analyze result: %d\n", ok) ;
        OK (Lbad == NULL) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_postorder
        //----------------------------------------------------------------------

        Int npost = CHOLMOD(postorder) (P, SIZE_MAX, P, P, cm) ;
        printf ("cholmod_postorder result: "ID"\n", npost) ;
        OK (npost == EMPTY) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_super_symbolic
        //----------------------------------------------------------------------

        A->ncol = SIZE_MAX ;
        A->nrow = SIZE_MAX ;
        A->stype = 1 ;
        L->n = SIZE_MAX ;

        ok = CHOLMOD(super_symbolic) (A, NULL, P, L, cm) ;
        NOT (ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        L->n = n_save ;
        A->ncol = ncol_save ;
        A->nrow = nrow_save ;
        A->stype = stype_save ;

        //----------------------------------------------------------------------
        // cholmod_cumsum
        //----------------------------------------------------------------------

        Int Result [5] ;
        Int Set [4] = { 0, Int_max/2, Int_max/2, Int_max } ;
        int64_t sum = CHOLMOD(cumsum) (Result, Set, 5) ;
        OK (sum == EMPTY) ;

        //----------------------------------------------------------------------
        // cholmod_ensure_dense
        //----------------------------------------------------------------------

        cholmod_dense *Y = NULL ;
        cholmod_dense *Z = CHOLMOD(ensure_dense) (&Y, SIZE_MAX, SIZE_MAX,
            SIZE_MAX, CHOLMOD_REAL + DTYPE, cm) ;
        printf ("status %d\n", cm->status) ;
        OK (Z == NULL) ;
        OK (Y == NULL) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_alloc_factor
        //----------------------------------------------------------------------

        Lbad = CHOLMOD(alloc_factor) (Int_max, DTYPE, cm) ;
        OK (Lbad == NULL) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        //----------------------------------------------------------------------
        // cholmod_factorize
        //----------------------------------------------------------------------

        Real X [2] ;

        A->ncol = SIZE_MAX ;
        A->nrow = SIZE_MAX ;
        A->stype = 0 ;

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;

        ok = CHOLMOD(factorize) (A, L, cm) ;
        printf ("cholmod_factorize result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->nrow = nrow_save ;
        A->ncol = ncol_save ;
        A->stype = stype_save ;
        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;

        //----------------------------------------------------------------------
        // cholmod_resymbol
        //----------------------------------------------------------------------

        A->nrow = SIZE_MAX ;
        A->ncol = SIZE_MAX ;
        A->stype = 0 ;

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;

        ok = CHOLMOD(resymbol) (A, NULL, 0, true, L, cm) ;
        printf ("cholmod_resymbol result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->nrow = nrow_save ;
        A->ncol = ncol_save ;
        A->stype = stype_save ;

        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;

        //----------------------------------------------------------------------
        // cholmod_resymbol_noperm
        //----------------------------------------------------------------------

        A->ncol = SIZE_MAX ;
        A->nrow = SIZE_MAX ;
        A->stype = -1 ;

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;

        ok = CHOLMOD(resymbol_noperm) (A, NULL, 0, true, L, cm) ;
        printf ("cholmod_resymbol_noperm result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->ncol = ncol_save ;
        A->nrow = nrow_save ;
        A->stype = stype_save ;

        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;

        //----------------------------------------------------------------------
        // cholmod_etree
        //----------------------------------------------------------------------

        A->stype = 0 ;
        A->ncol = SIZE_MAX ;

        ok = CHOLMOD(etree) (A, P, cm) ;
        printf ("cholmod_etree result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->ncol = ncol_save ;
        A->stype = stype_save ;

        //----------------------------------------------------------------------
        // cholmod_rowadd
        //----------------------------------------------------------------------

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;
        R->nrow = SIZE_MAX ;

        ok = CHOLMOD(rowadd) (0, R, L, cm) ;
        printf ("cholmod_rowadd result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;
        R->nrow = R_nrow_save ;

        //----------------------------------------------------------------------
        // cholmod_rowdel
        //----------------------------------------------------------------------

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;

        ok = CHOLMOD(rowdel) (0, NULL, L, cm) ;
        printf ("cholmod_rowdel result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;

        //----------------------------------------------------------------------
        // cholmod_rowcolcounts
        //----------------------------------------------------------------------

        A->stype = 0 ;
        A->nrow = SIZE_MAX ;
        A->ncol = SIZE_MAX ;

        ok = CHOLMOD(rowcolcounts) (A, NULL, 0, P, P, P, P, P, P, cm) ;
        printf ("cholmod_rowadd result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->stype = stype_save ;
        A->nrow = nrow_save ;
        A->ncol = ncol_save ;

        //----------------------------------------------------------------------
        // cholmod_rowfac
        //----------------------------------------------------------------------

        A->stype = 0 ;
        A->xtype = CHOLMOD_COMPLEX ;
        A->nrow = SIZE_MAX ;
        A->ncol = SIZE_MAX ;

        L->n = SIZE_MAX ;

        ok = CHOLMOD(rowfac) (A, A, one, 0, 0, L, cm) ;
        printf ("cholmod_rowfac result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->stype = stype_save ;
        A->xtype = A_xtype_save ;
        A->nrow = nrow_save ;
        A->ncol = ncol_save ;

        L->n = n_save ;

        //----------------------------------------------------------------------
        // cholmod_submatrix
        //----------------------------------------------------------------------

        A->stype = 0 ;
        A->nrow = SIZE_MAX ;
        A->ncol = SIZE_MAX ;

        cholmod_sparse *S = CHOLMOD(submatrix) (A, P, 1, P, 1, false, true, cm);
        printf ("cholmod_submatrix result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        A->stype = stype_save ;
        A->nrow = nrow_save ;
        A->ncol = ncol_save ;

        //----------------------------------------------------------------------
        // cholmod_updown
        //----------------------------------------------------------------------

        L->n = SIZE_MAX ;
        L->xtype = CHOLMOD_REAL ;
        L->x = X ;
        C->nrow = SIZE_MAX ;

        ok = CHOLMOD(updown) (1, C, L, cm) ;
        printf ("cholmod_updown result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        L->n = n_save ;
        L->xtype = L_xtype_save ;
        L->x = NULL ;
        C->nrow = C_nrow_save ;

        //----------------------------------------------------------------------
        // cholmod_super_numeric
        //----------------------------------------------------------------------

        L->n = SIZE_MAX ;
        L->is_super = true ;
        A->stype = -1 ;
        A->nrow = SIZE_MAX ;
        A->ncol = SIZE_MAX ;

        ok = CHOLMOD(super_numeric) (A, NULL, one, L, cm) ;
        printf ("cholmod_super_numeric result: %d\n", ok) ;
        OK (!ok) ;
        OK (cm->status == CHOLMOD_TOO_LARGE) ;
        cm->status = CHOLMOD_OK ;

        L->is_super = is_super_save ;
        L->n = n_save ;
        A->stype = stype_save ;
        A->nrow = nrow_save ;
        A->ncol = ncol_save ;
    }

    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_sparse) (&R, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;
}

