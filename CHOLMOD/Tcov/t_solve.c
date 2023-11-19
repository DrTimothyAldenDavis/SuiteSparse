//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_solve: test CHOLMOD solvers
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Test CHOLMOD for solving various systems of linear equations.

#include "cm.h"

#define NFTYPES 17
Int ll_types [NFTYPES] = { 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0 } ;
Int pk_types [NFTYPES] = { 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0 } ;
Int mn_types [NFTYPES] = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0 } ;
Int co_types [NFTYPES] = { 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1 } ;

#define NRHS 9

#ifndef NDEBUG
#ifndef EXTERN
#define EXTERN extern
#endif
EXTERN int cholmod_dump, cholmod_l_dump ;
#endif

//------------------------------------------------------------------------------
// test_solver
//------------------------------------------------------------------------------

// Test solve(A) with various control parameters

double test_solver (cholmod_sparse *A)
{
    double err, maxerr = 0 ;
    int save ;

    for (cm->postorder = 0 ; cm->postorder <= 1 ; cm->postorder++)
    {

        my_srand (42) ;                  // RAND reset

        // simplicial, no extra memory
        printf ("test_solver: simplicial, no extra memory\n") ;
        cm->supernodal = CHOLMOD_SIMPLICIAL ;
        cm->grow2 = 0 ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // simplicial, extra space in columns of L
        printf ("test_solver: simplicial, extra space in columns of L\n") ;
        cm->grow2 = 5 ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // supernodal
        printf ("test_solver: supernodal\n") ;
        cm->supernodal = CHOLMOD_SUPERNODAL ;
        save = cm->nthreads_max ;
        cm->nthreads_max = -1 ;
        err = solve (A) ;
        cm->nthreads_max = save ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // supernodal, without final resymbol
        printf ("test_solver: supernodal, without final resymbol\n") ;
        cm->final_resymbol = FALSE ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // supernodal, with resymbol, final_super false
        printf ("test_solver: supernodal, with resymbol\n") ;
        cm->supernodal = CHOLMOD_SUPERNODAL ;
        cm->final_asis = FALSE ;
        cm->final_resymbol = TRUE ;
        cm->final_super = FALSE ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // supernodal, with resymbol, final_super tree
        printf ("test_solver: supernodal, with resymbol\n") ;
        cm->supernodal = CHOLMOD_SUPERNODAL ;
        cm->final_asis = FALSE ;
        cm->final_resymbol = TRUE ;
        cm->final_super = TRUE ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        // simplicial LL'
        printf ("test_solver: simplicial LL', try NESDIS instead of METIS\n") ;
        cm->supernodal = CHOLMOD_SIMPLICIAL ;
        cm->final_ll = TRUE ;
        cm->default_nesdis = TRUE ;
        err = solve (A) ;
        cm->default_nesdis = FALSE ;
        MAXERR (maxerr, err, 1) ;
        cm->final_ll = FALSE ;
        printf ("test_solver err: %6.2e\n", err) ;

        // supernodal, without final resymbol, and no relaxed supernodes
        printf ( "test_solver: supernodal, without final resymbol, "
            "and no relaxed supernodes\n") ;
        cm->supernodal = CHOLMOD_SUPERNODAL ;
        cm->final_asis = TRUE ;
        cm->nrelax [0] = 0 ;
        cm->nrelax [1] = 0 ;
        cm->nrelax [2] = 0 ;
        cm->zrelax [0] = 0 ;
        cm->zrelax [1] = 0 ;
        cm->zrelax [2] = 0 ;
        cm->grow0 = 1 ;
        cm->grow1 = 1 ;
        cm->grow2 = 0 ;
        err = solve (A) ;
        MAXERR (maxerr, err, 1) ;
        printf ("test_solver err: %6.2e\n", err) ;

        //----------------------------------------------------------------------
        // restore defaults
        //----------------------------------------------------------------------

        cm->dbound = 0.0 ;
        cm->sbound = 0.0 ;

        cm->grow0 = 1.2 ;
        cm->grow1 = 1.2 ;
        cm->grow2 = 5 ;

        cm->final_asis = TRUE ;
        cm->final_super = TRUE ;
        cm->final_ll = FALSE ;
        cm->final_pack = TRUE ;
        cm->final_monotonic = TRUE ;
        cm->final_resymbol = FALSE ;

        cm->supernodal = CHOLMOD_AUTO ;
        cm->nrelax [0] = 4 ;
        cm->nrelax [1] = 16 ;
        cm->nrelax [2] = 48 ;
        cm->zrelax [0] = 0.8 ;
        cm->zrelax [1] = 0.1 ;
        cm->zrelax [2] = 0.05 ;

        // do not restore these defaults:
        // cm->maxrank = ...
        // cm->metis_memory = 2.0
        // cm->metis_nswitch = 3000
        // cm->metis_dswitch = 0.66
        // cm->print = 3
        // cm->precise = FALSE
    }

    progress (1, '.') ;
    return (maxerr) ;
}

//------------------------------------------------------------------------------
// solve
//------------------------------------------------------------------------------

// solve Ax=b or AA'x=b, systems involving just L, D, or L', and update/downdate
// the system.  Returns the worst-case residual.  This routine keeps going if
// it runs out of memory (unless the error handler terminates it), because
// it is used both normally and in the memory tests.

double solve (cholmod_sparse *A)
{
    double r, enorm, snorm, maxerr = 0, gnorm, anorm, bnorm, xnorm, norm, rcond;
    cholmod_factor *L, *Lcopy, *L2 ;
    cholmod_sparse *Lmat, *Lo, *D, *Up, *S, *LD, *LDL, *E, *I, *C, *CC, *Ct,
        *Ssym, *Cperm, *C2, *S2, *H, *F, *Lo1, *Aboth, *Bset ;
    cholmod_dense *X, *Cdense, *B, *Bcomplex, *Bzomplex, *Breal, *W, *R,
        *A3, *C3, *E3 ;
    cholmod_dense *B2, *B2complex, *B2zomplex, *B2real, *Ywork, *Ework ;
    cholmod_sparse *AFt, *AF, *G, *RowK, *Bsparse, *Xsparse ;
    Real *Cx ;
    Int *P, *cset, *fset, *Parent, *Post, *RowCount, *ColCount,
             *First, *Level, *rcount, *ccount, *Lp, *Li ;
    Int p, i, j, k, n, nrhs, save, save2, csize, rank, nrow, ncol, is_ll,
        xtype, isreal, prefer_zomplex, xtype2, save3 ;
    cm->blas_ok = TRUE ;
    int Lxtype ;

    Ywork = NULL ;
    Ework = NULL ;

    if (cm->print > 1)
    {
        printf ("============================================== in solve:\n") ;
    }

    if (A == NULL)
    {
        ERROR (CHOLMOD_INVALID, "nothing to solve") ;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // construct right-hand-side (Ax=b if symmetric, AA'x=b otherwise)
    //--------------------------------------------------------------------------

    n = A->nrow ;
    nrow = A->nrow ;
    ncol = A->ncol ;
    xtype = A->xtype ;
    isreal = (xtype == CHOLMOD_REAL) ;
    anorm = CHOLMOD(norm_sparse) (A, 1, cm) ;
    save = cm->final_asis ;
    cm->final_asis = TRUE ;

    B = rhs (A, NRHS, n, 0) ;

    Bcomplex = CHOLMOD(copy_dense) (B, cm) ;
    CHOLMOD(dense_xtype) (CHOLMOD_COMPLEX + DTYPE, Bcomplex, cm) ;

    B2 = rhs (A, NRHS, n, 1) ;

    Breal = CHOLMOD(copy_dense) (B, cm) ;
    CHOLMOD(dense_xtype) (CHOLMOD_REAL + DTYPE, Breal, cm) ;

    Bzomplex = rhs (A, NRHS, n, 2) ;
    CHOLMOD(dense_xtype) (CHOLMOD_ZOMPLEX + DTYPE, Bzomplex, cm) ;

    B2complex = rhs (A, NRHS, n, 3) ;

    B2real = rhs (A, NRHS, n, 4) ;
    CHOLMOD(dense_xtype) (CHOLMOD_REAL + DTYPE, B2real, cm) ;

    B2zomplex = rhs (A, NRHS, n, 5) ;
    CHOLMOD(dense_xtype) (CHOLMOD_ZOMPLEX + DTYPE, B2zomplex, cm) ;

    //--------------------------------------------------------------------------
    // analyze
    //--------------------------------------------------------------------------

    if (n < 100 && cm->nmethods == 1 && cm->method[0].ordering == CHOLMOD_GIVEN)
    {
        Int *UserPerm = prand (nrow) ;                          // RAND
        CHOLMOD(print_perm) (UserPerm, nrow, nrow, "UserPerm", cm) ;
        L = CHOLMOD(analyze_p) (A, UserPerm, NULL, 0, cm) ;
        OK (CHOLMOD(print_common) ("36:cm with UserPerm", cm)) ;
        CHOLMOD(free) (nrow, sizeof (Int), UserPerm, cm) ;
    }
    else
    {
        L = CHOLMOD(analyze) (A, cm) ;
    }

    //--------------------------------------------------------------------------
    // test rowadd on a symbolic factor (real matrices only)
    //--------------------------------------------------------------------------

    if (isreal)
    {
        RowK = CHOLMOD(spzeros) (n, 1, 0, CHOLMOD_REAL, cm) ;
        Lcopy = CHOLMOD(copy_factor) (L, cm) ;
        if (n > 0)
        {
            // rowadd does not work on complex/zomplex matrices
            CHOLMOD(rowadd) (0, RowK, Lcopy, cm) ;
            CHOLMOD(check_factor) (Lcopy, cm) ;
        }
        CHOLMOD(free_sparse) (&RowK, cm) ;
        CHOLMOD(free_factor) (&Lcopy, cm) ;
    }

    //--------------------------------------------------------------------------
    // factorize
    //--------------------------------------------------------------------------

    CHOLMOD(factorize) (A, L, cm) ;

//  CHOLMOD(print_sparse) (A, "A here::", cm) ;
//  CHOLMOD(print_factor) (L, "L here::", cm) ;

    // make a of copy of L->Perm, just for testing
    Int *PP = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    if (PP != NULL && L != NULL && L->Perm != NULL)
    {
        memcpy (PP, L->Perm, nrow * sizeof (Int)) ;
    }

    //--------------------------------------------------------------------------
    // various solves
    //--------------------------------------------------------------------------

    if (B != NULL)
    {
        // B->ncol = 1 ;
        prefer_zomplex = (B->xtype == CHOLMOD_ZOMPLEX) ;
    }
    else
    {
        prefer_zomplex = FALSE ;
    }

    for (k = 0 ; k <= 1 ; k++)
    {

        cm->prefer_zomplex = k ;

        // compute the residual, X complex if L or B not real
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        rcond = CHOLMOD(rcond) (L, cm) ;
        if (cm->print > 1)
        {
            printf ("rcond for solv2: %g k %d\n", rcond, (int) k) ;
        }

        // now test the same thing, but with cholmod_solve2

        CHOLMOD(solve2) (CHOLMOD_A, L, B2, NULL, &X, NULL, &Ywork, &Ework, cm) ;
        r = resid (A, X, B2) ;
        MAXERR (maxerr, r, 1) ;

        CHOLMOD(solve2) (CHOLMOD_A, L, B, NULL, &X, NULL, &Ywork, &Ework, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;

        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

        // see what happens if X has the wrong size or type on input.  This
        // will free X on input, and reallocate it the right size and type
        X = CHOLMOD(ones) (1, 1, CHOLMOD_REAL + DTYPE, cm) ;
        CHOLMOD(solve2) (CHOLMOD_A, L, B, NULL, &X, NULL, &Ywork, &Ework, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;

        // now X is OK, and so is E, but Y is null
        CHOLMOD(solve2) (CHOLMOD_A, L, B, NULL, &X, NULL, &Ywork, &Ework, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

        X = CHOLMOD(solve) (CHOLMOD_A, L, Bzomplex, cm) ;
        r = resid (A, X, Bzomplex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // now test the same thing, but with cholmod_solve2
        CHOLMOD(solve2) (CHOLMOD_A, L, Bzomplex, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, Bzomplex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(solve2) (CHOLMOD_A, L, B2zomplex, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, B2zomplex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

        // complex right-hand-side
        X = CHOLMOD(solve) (CHOLMOD_A, L, Bcomplex, cm) ;
        r = resid (A, X, Bcomplex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // now test the same thing, but with cholmod_solve2
        CHOLMOD(solve2) (CHOLMOD_A, L, Bcomplex, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, Bcomplex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(solve2) (CHOLMOD_A, L, B2complex, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, B2complex) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

        // real right-hand-side

        X = CHOLMOD(solve) (CHOLMOD_A, L, Breal, cm) ;
        r = resid (A, X, Breal) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // now test the same thing, but with cholmod_solve2
        CHOLMOD(solve2) (CHOLMOD_A, L, Breal, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, Breal) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(solve2) (CHOLMOD_A, L, B2real, NULL, &X, NULL,
            &Ywork, &Ework, cm) ;
        r = resid (A, X, B2real) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

        // sparse solve of Ax=b, b real
        Bsparse = CHOLMOD(dense_to_sparse) (Breal, TRUE, cm) ;
        Xsparse = CHOLMOD(spsolve) (CHOLMOD_A, L, Bsparse, cm) ;
        r = resid_sparse (A, Xsparse, Bsparse) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_sparse) (&Bsparse, cm) ;
        CHOLMOD(free_sparse) (&Xsparse, cm) ;

        // sparse solve of Ax=b, b complex
        Bsparse = CHOLMOD(dense_to_sparse) (Bcomplex, TRUE, cm) ;
        Xsparse = CHOLMOD(spsolve) (CHOLMOD_A, L, Bsparse, cm) ;
        r = resid_sparse (A, Xsparse, Bsparse) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_sparse) (&Bsparse, cm) ;
        CHOLMOD(free_sparse) (&Xsparse, cm) ;

        // sparse solve of Ax=b, b zomplex
        Bsparse = CHOLMOD(dense_to_sparse) (Bzomplex, TRUE, cm) ;
        Xsparse = CHOLMOD(spsolve) (CHOLMOD_A, L, Bsparse, cm) ;
        r = resid_sparse (A, Xsparse, Bsparse) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_sparse) (&Bsparse, cm) ;
        CHOLMOD(free_sparse) (&Xsparse, cm) ;

    }

    cm->prefer_zomplex = FALSE ;

    //--------------------------------------------------------------------------
    // sparse solve to compute inv(A)
    //--------------------------------------------------------------------------

    rcond = CHOLMOD(rcond) (L, cm) ;
    if (cm->print > 1)
    {
        printf ("rcond: %g\n", rcond) ;
    }

    #ifdef DOUBLE
    bool rcond_ok = rcond > 1e-6 ;
    #else
    bool rcond_ok = rcond > 1e-3 ;
    #endif

    if (n < 100 && A->stype != 0 && rcond_ok)
    {
        // solve A*C=I, so C should equal A inverse
        I = CHOLMOD(speye) (n, n, A->xtype + DTYPE, cm) ;
        C = CHOLMOD(spsolve) (CHOLMOD_A, L, I, cm) ;
        CHOLMOD(sparse_xtype) (A->xtype + DTYPE, C, cm) ;

        // compute norm of A*C-I

        // E = A*C
        E = CHOLMOD(ssmult) (A, C, 0, 2, false, cm) ;

        // r = norm (E-I)
        F = CHOLMOD(add) (E, I, minusone, one, 2, false, cm) ;
        r = CHOLMOD(norm_sparse) (F, 1, cm) ;
        CHOLMOD(free_sparse) (&F, cm) ;

        // also use ptest to compute norm (E-I)
        double r2 = ptest (E, I, PP, n) ;

        CHOLMOD(free_sparse) (&E, cm) ;
        if (cm->print > 1)
        {
            printf ("norm (A*C-I): %g %g\n", r, r2) ;
        }
        MAXERR (maxerr, r, 1) ;
        MAXERR (maxerr, r2, 1) ;
        CHOLMOD(free_sparse) (&I, cm) ;
        CHOLMOD(free_sparse) (&C, cm) ;
    }

    //--------------------------------------------------------------------------
    // change complexity of L and solve again; test copy/change routines
    //--------------------------------------------------------------------------

    // change to complex, otherwise leave as
    Lcopy = CHOLMOD(copy_factor) (L, cm) ;
    CHOLMOD(factor_xtype) (CHOLMOD_COMPLEX + DTYPE, Lcopy, cm) ;
    X = CHOLMOD(solve) (CHOLMOD_A, Lcopy, B, cm) ;
    r = resid (A, X, B) ;
    MAXERR (maxerr, r, 1) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_factor) (&Lcopy, cm) ;

    // change to zomplex LDL'
    Lxtype = (L == NULL) ? CHOLMOD_REAL : (L->xtype) ;
    Lcopy = CHOLMOD(copy_factor) (L, cm) ;
    CHOLMOD(check_factor) (L, cm) ;
    CHOLMOD(change_factor) (Lxtype, FALSE, FALSE, TRUE, TRUE, Lcopy, cm) ;
    CHOLMOD(factor_xtype) (CHOLMOD_ZOMPLEX + DTYPE, Lcopy, cm) ;
    X = CHOLMOD(solve) (CHOLMOD_A, Lcopy, B, cm) ;
    r = resid (A, X, B) ;
    MAXERR (maxerr, r, 1) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_factor) (&Lcopy, cm) ;

    Lcopy = CHOLMOD(copy_factor) (L, cm) ;
    CHOLMOD(change_factor) (Lxtype, TRUE, FALSE, FALSE, FALSE, Lcopy, cm) ;
    CHOLMOD(check_factor) (L, cm) ;
    CHOLMOD(free_factor) (&Lcopy, cm) ;

    CHOLMOD(free_factor) (&L, cm) ;
    cm->final_asis = save ;

    //--------------------------------------------------------------------------
    // solve again, but use cm->final_asis as given
    //--------------------------------------------------------------------------

    if (n < 100 && cm->nmethods == 1 && cm->method[0].ordering == CHOLMOD_GIVEN)
    {
        Int *UserPerm = prand (nrow) ;                          // RAND
        CHOLMOD(print_perm) (UserPerm, nrow, nrow, "UserPerm", cm) ;
        L = CHOLMOD(analyze_p) (A, UserPerm, NULL, 0, cm) ;
        CHOLMOD(free) (nrow, sizeof (Int), UserPerm, cm) ;
    }
    else
    {
        L = CHOLMOD(analyze) (A, cm) ;
    }

    CHOLMOD(factorize) (A, L, cm) ;

    // turn off memory tests [
    save3 = my_tries ;
    my_tries = -1 ;

    if (B != NULL && Breal != NULL && Bcomplex != NULL && Bzomplex != NULL)
    {
        for (nrhs = 1 ; nrhs <= NRHS ; nrhs++)
        {
            for (cm->prefer_zomplex = 0 ; cm->prefer_zomplex <= 1 ;
                    cm->prefer_zomplex++)
            {
                B->ncol = nrhs ;
                Breal->ncol = nrhs ;
                Bcomplex->ncol = nrhs ;
                Bzomplex->ncol = nrhs ;

                X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
                r = resid (A, X, B) ;
                CHOLMOD(free_dense) (&X, cm) ;
                MAXERR (maxerr, r, 1) ;

                X = CHOLMOD(solve) (CHOLMOD_A, L, Breal, cm) ;
                r = resid (A, X, Breal) ;
                CHOLMOD(free_dense) (&X, cm) ;
                MAXERR (maxerr, r, 1) ;

                X = CHOLMOD(solve) (CHOLMOD_A, L, Bcomplex, cm) ;
                r = resid (A, X, Bcomplex) ;
                CHOLMOD(free_dense) (&X, cm) ;
                MAXERR (maxerr, r, 1) ;

                X = CHOLMOD(solve) (CHOLMOD_A, L, Bzomplex, cm) ;
                r = resid (A, X, Bzomplex) ;
                CHOLMOD(free_dense) (&X, cm) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
    }
    cm->prefer_zomplex = FALSE ;

    my_tries = save3 ;
    // turn memory tests back on, where we left off ]

    //--------------------------------------------------------------------------
    // convert L to LDL' packed or LL packed
    //--------------------------------------------------------------------------

    if (cm->print > 1)
    {
        printf ("before change factor : %d\n", L ? L->is_super : -1) ;
    }
    is_ll = (L == NULL) ? FALSE : (L->is_ll) ;
    Lxtype = (L == NULL) ? CHOLMOD_REAL : (L->xtype) ;
    Lcopy = CHOLMOD(copy_factor) (L, cm) ;
    CHOLMOD(change_factor) (Lxtype, is_ll, FALSE, TRUE, TRUE, Lcopy, cm) ;
    if (cm->print > 1)
    {
        printf ("after change factor : %d\n", L ? L->is_super : -1) ;
    }

    //--------------------------------------------------------------------------
    // extract L, D, and L' as matrices from Lcopy
    //--------------------------------------------------------------------------

    CHOLMOD(resymbol) (A, NULL, 0, TRUE, Lcopy, cm) ;

    Lmat = CHOLMOD(factor_to_sparse) (Lcopy, cm) ;
    CHOLMOD(free_factor) (&Lcopy, cm) ;
    if (Lmat != NULL && A != NULL && Lmat->xtype != A->xtype)
    {
        if (cm->print > 1)
        {
            printf ("change Lmat from xtype %d to %d\n",
                Lmat->xtype, A->xtype) ;
        }
        CHOLMOD(sparse_xtype) (A->xtype + DTYPE, Lmat, cm) ;
    }
    CHOLMOD(check_sparse) (Lmat, cm) ;

    Lo = NULL ;
    D = NULL ;

    Lxtype = (Lmat == NULL) ? CHOLMOD_REAL : (Lmat->xtype) ;
    I = CHOLMOD(speye) (n, n, Lxtype + DTYPE, cm) ;

    if (!is_ll)
    {
        // factorization is LDL' = Lo*D*Up
        Lo1 = CHOLMOD(band) (Lmat, -n, -1, 1, cm) ;
        Lo = CHOLMOD(add) (Lo1, I, one, one, TRUE, TRUE, cm) ;
        CHOLMOD(free_sparse) (&Lo1, cm) ;
        D = CHOLMOD(band) (Lmat, 0, 0, 1, cm) ;
    }
    else
    {
        // factorization is LL' = Lo*D*Up
        Lo = CHOLMOD(band) (Lmat, -n, 0, 1, cm) ;
        D = CHOLMOD(speye) (n, n, Lxtype + DTYPE, cm) ;
    }

    Up = CHOLMOD(transpose) (Lo, 2, cm) ;

    //--------------------------------------------------------------------------
    // compute 1-norm of (Lo*D*Up - PAP') or (Lo*D*Up - PAA'P')
    //--------------------------------------------------------------------------

    P = (L != NULL) ? (L->Perm) : NULL ;
    S = NULL ;
    G = NULL ;

    if (A->stype == 0)
    {
        // G = A*A', try with fset = prand (ncol)
        fset = prand (ncol) ;                               // RAND
        CHOLMOD(print_perm) (fset, ncol, ncol, "fset", cm) ;
        AFt = CHOLMOD(ptranspose) (A, 2, NULL, fset, ncol, cm) ;
        AF  = CHOLMOD(transpose) (AFt, 2, cm) ;

        CHOLMOD(free) (ncol, sizeof (Int), fset, cm) ;
        G = CHOLMOD(ssmult) (AF, AFt, 0, TRUE, TRUE, cm) ;

        // H = A*A' using aat
        H = CHOLMOD(aat) (AF, NULL, 0, 2, cm) ;

        // enorm = norm (G-H)
        E = CHOLMOD(add) (G, H, one, minusone, TRUE, FALSE, cm) ;
        enorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
        CHOLMOD(free_sparse) (&E, cm) ;

        // also use ptest to compute enorm (G-H)
        double enorm2 = ptest (G, H, PP, nrow) ;

        gnorm = CHOLMOD(norm_sparse) (G, 0, cm) ;
        MAXERR (maxerr, enorm, gnorm) ;
        MAXERR (maxerr, enorm2, gnorm) ;
        if (cm->print > 1)
        {
            printf ("enorm %g %g gnorm %g hnorm %g\n", enorm, enorm2, gnorm,
                CHOLMOD(norm_sparse) (H, 0, cm)) ;
        }
        if (gnorm > 0)
        {
            enorm /= gnorm ;
        }

        #ifdef DOUBLE
        OK (enorm < 1e-8) ;
        #else
        OK (enorm < 1e-4) ;
        #endif
        CHOLMOD(free_sparse) (&AFt, cm) ;
        CHOLMOD(free_sparse) (&AF, cm) ;
        CHOLMOD(free_sparse) (&H, cm) ;
    }
    else
    {
        // G = A
        G = CHOLMOD(copy) (A, 0, 2, cm) ;
    }

    if (A->stype == 0)
    {
        // S = PAA'P'
        S = CHOLMOD(submatrix) (G, P, n, P, n, TRUE, FALSE, cm) ;
    }
    else
    {
        // S = PAP'
        Aboth = CHOLMOD(copy) (A, 0, 2, cm) ;
        S = CHOLMOD(submatrix) (Aboth, P, n, P, n, TRUE, FALSE, cm) ;
        CHOLMOD(free_sparse) (&Aboth, cm) ;
    }

    if (n < NSMALL)
    {
        // only do this for small test matrices, since L*D*L' can have many
        // nonzero entries
        // E = L*D*L' - S

        LD = CHOLMOD(ssmult) (Lo, D, 0, TRUE, FALSE, cm) ;
        LDL = CHOLMOD(ssmult) (LD, Up, 0, TRUE, FALSE, cm) ;
        CHOLMOD(free_sparse) (&LD, cm) ;
        E = CHOLMOD(add) (LDL, S, one, minusone, TRUE, FALSE, cm) ;
        CHOLMOD(free_sparse) (&LDL, cm) ;

        // e = norm (E) / norm (S)
        enorm = CHOLMOD(norm_sparse) (E, 1, cm) ;
        snorm = CHOLMOD(norm_sparse) (S, 0, cm) ;
        MAXERR (maxerr, enorm, snorm) ;
        CHOLMOD(free_sparse) (&E, cm) ;
    }

    // check the row/col counts
    RowCount = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    ColCount = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Parent   = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Post     = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    First    = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Level    = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    rcount   = CHOLMOD(calloc) (n, sizeof (Int), cm) ;
    ccount   = CHOLMOD(calloc) (n, sizeof (Int), cm) ;

    if (S != NULL && Lmat != NULL && RowCount != NULL && ColCount != NULL &&
        Parent != NULL && Post != NULL && First != NULL && Level != NULL &&
        rcount != NULL && ccount != NULL)
    {
        S->stype = 1 ;

        CHOLMOD(etree) (S, Parent, cm) ;
        CHOLMOD(postorder) (Parent, n, NULL, Post, cm) ;
        S->stype = -1 ;
        CHOLMOD(rowcolcounts) (S, NULL, 0, Parent, Post, RowCount, ColCount,
                First, Level, cm) ;

        Lp = Lmat->p ;
        Li = Lmat->i ;
        OK (Lmat->packed) ;
        for (j = 0 ; j < n ; j++)
        {
            for (p = Lp [j] ; p < Lp [j+1] ; p++)
            {
                i = Li [p] ;
                rcount [i]++ ;
                ccount [j]++ ;
            }
        }
        // a singular matrix will only be partially factorized
        if (L->minor == (size_t) n)
        {
            for (j = 0 ; j < n ; j++)
            {
                OK (ccount [j] == ColCount [j]) ;
            }
        }
        for (i = 0 ; i < (Int) (L->minor) ; i++)
        {
            OK (rcount [i] == RowCount [i]) ;
        }
    }

    CHOLMOD(free) (n, sizeof (Int), RowCount, cm) ;
    CHOLMOD(free) (n, sizeof (Int), ColCount, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Parent, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Post, cm) ;
    CHOLMOD(free) (n, sizeof (Int), First, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Level, cm) ;
    CHOLMOD(free) (n, sizeof (Int), rcount, cm) ;
    CHOLMOD(free) (n, sizeof (Int), ccount, cm) ;

    CHOLMOD(free_sparse) (&S, cm) ;

    //--------------------------------------------------------------------------
    // solve other systems
    //--------------------------------------------------------------------------

    // turn off memory tests [
    save3 = my_tries ;
    my_tries = -1 ;
    rcond = CHOLMOD(rcond) (L, cm) ;

    #ifdef DOUBLE
    rcond_ok = rcond > 1e-6 ;
    #else
    rcond_ok = rcond > 1e-3 ;
    #endif

    for (nrhs = 1 ; nrhs <= 4 ; nrhs++)     // reduced here (6 to 4)
    {
        if (B == NULL)
        {
            break ;
        }

        B->ncol = nrhs ;

        // solve Ax=b and compare with the sparse solve
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // solve LDL'x=b and compare with sparse solve
        if (rcond_ok)
        {
            X = CHOLMOD(solve) (CHOLMOD_LDLt, L, B, cm) ;
            // printf ("LDL'x=b %p %p %p\n", Lo, X, B) ;
            r = resid3 (Lo, D, Up, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            // solve DL'x=b and compare with sparse solve
            X = CHOLMOD(solve) (CHOLMOD_DLt, L, B, cm) ;
            // printf ("DL'x=b %p %p %p\n", D, X, B) ;
            r = resid3 (D, Up, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            // solve LDx=b and compare with sparse solve
            X = CHOLMOD(solve) (CHOLMOD_LD, L, B, cm) ;
            // printf ("LDx=b %p %p %p\n", Lo, X, B) ;
            r = resid3 (Lo, D, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            // solve Lx=b and compare with sparse solve
            X = CHOLMOD(solve) (CHOLMOD_L, L, B, cm) ;
            // printf ("Lx=b %p %p %p\n", Lo, X, B) ;
            r = resid3 (Lo, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            // solve L'x=b and compare with sparse solve
            X = CHOLMOD(solve) (CHOLMOD_Lt, L, B, cm) ;
            // printf ("L'x=b %p %p %p\n", Up, X, B) ;
            r = resid3 (Up, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            // solve Dx=b and compare with sparse solve
            X = CHOLMOD(solve) (CHOLMOD_D, L, B, cm) ;
            // printf ("Dx=b %p %p %p\n", D, X, B) ;
            r = resid3 (D, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;

            save2 = cm->prefer_zomplex ;
            for (k = 0 ; k <= 1 ; k++)
            {
                cm->prefer_zomplex = k ;

                // x=Pb and compare with sparse solve
                X = CHOLMOD(solve) (CHOLMOD_P, L, B, cm) ;
                r = pnorm (X, P, B, FALSE) ;
                MAXERR (maxerr, r, 1) ;
                CHOLMOD(free_dense) (&X, cm) ;

                // x=P'b and compare with sparse solve
                X = CHOLMOD(solve) (CHOLMOD_Pt, L, B, cm) ;
                r = pnorm (X, P, B, TRUE) ;
                MAXERR (maxerr, r, 1) ;
                CHOLMOD(free_dense) (&X, cm) ;
            }
            cm->prefer_zomplex = save2 ;
        }
    }

    // turn memory tests back on, where we left off ]
    my_tries = save3 ;

    Real bx = 0 ;
    Real bz = 0 ;

    if (n > 0 && B != NULL)
    {
        Real *Bx = B->x ;
        Real *Bz = B->z ;
        if (B->xtype == CHOLMOD_REAL)
        {
            bx = Bx [0] ;
        }
        else if (B->xtype == CHOLMOD_COMPLEX)
        {
            bx = Bx [0] ;
            bz = Bx [1] ;
        }
        else
        {
            bx = Bx [0] ;
            bz = Bz [0] ;
        }
    }

    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free_dense) (&B2, cm) ;

    //--------------------------------------------------------------------------
    // test the sparse solve
    //--------------------------------------------------------------------------

    // turn off memory tests for large matrices [
    if (n > 10)
    {
        save3 = my_tries ;
        my_tries = -1 ;
    }

    // get ready for sparse solve
    Bset = CHOLMOD(allocate_sparse) (n, 1, 1, FALSE, TRUE, 0,
        CHOLMOD_PATTERN + DTYPE, cm) ;
    Lxtype = (L == NULL) ? CHOLMOD_REAL : L->xtype ;
    B = CHOLMOD(zeros) (n, 1, Lxtype + DTYPE, cm) ;

    if (Bset != NULL && B != NULL && n > 0 && rcond_ok)
    {
        cholmod_sparse *Xset ;
        cholmod_dense *X2 ;
        Int *Bseti, *Bsetp, *Xseti ;
        Real *X2x, *X1x ;

        Xset = NULL ;
        X2 = NULL ;

        Bseti = Bset->i ;
        Bsetp = Bset->p ;
        Bsetp [0] = 0 ;
        Bsetp [1] = 1 ;
        Bseti [0] = 0 ;   // first entry in b is nonzero, all others zero
        Real *Bx = B->x ;
        Real *Bz = B->z ;

        if (B->xtype == CHOLMOD_REAL)
        {
            Bx [0] = bx ;
        }
        else if (B->xtype == CHOLMOD_COMPLEX)
        {
            Bx [0] = bx ;
            Bx [1] = bz ;
        }
        else
        {
            Bx [0] = bx ;
            Bz [0] = bz ;
        }

        // solve Ax=b and compare with the sparse solve
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        if (X != NULL)
        {
            r = resid (A, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_work) (cm) ;
            CHOLMOD(solve2) (CHOLMOD_A, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 Ax=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve LDL'x=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_LDLt, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (Lo, D, Up, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_LDLt, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 LDL'x=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve LDx=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_LD, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (Lo, D, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_LD, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 LDx=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve DL'x=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_DLt, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (D, Up, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_DLt, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 DL'x=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve Lx=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_L, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (Lo, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_L, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 Lx=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve L'x=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_Lt, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (Up, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_Lt, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 Ltx=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

        // solve Dx=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_D, L, B, cm) ;
        if (X != NULL)
        {
            r = resid3 (D, NULL, NULL, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(solve2) (CHOLMOD_D, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 Dx=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;

#if 0
int sav = cm->print ;
cm->print = 5 ;
CHOLMOD(print_factor) (L, "L for with Px=b", cm) ;
CHOLMOD(print_sparse) (Bset, "Bset for with Px=b", cm) ;
CHOLMOD(print_dense) (B, "B for with Px=b", cm) ;
        // solve Px=b and compare with sparse solve
        X = CHOLMOD(solve) (CHOLMOD_P, L, B, cm) ;
        if (X != NULL)
        {
            CHOLMOD(solve2) (CHOLMOD_P, L, B, Bset, &X2, &Xset,
                &Ywork, &Ework, cm) ;
            if (X2 != NULL)
            {
CHOLMOD(print_dense) (X2, "X2 with Px=b", cm) ;
CHOLMOD(print_dense) (X,  "X with Px=b", cm) ;
                X2x = X2->x ;
                X1x = X->x ;
                r = fabs (X2x [0] - X1x [0]) ;
                if (cm->print > 1) printf ("solve2 Px=b err %g\n", r) ;
                MAXERR (maxerr, r, 1) ;
                OK (r < 0.1) ;
            }
        }
        CHOLMOD(free_dense) (&X, cm) ;
cm->print = sav ;
#endif

        CHOLMOD(free_sparse) (&Xset, cm) ;
        CHOLMOD(free_dense) (&X2, cm) ;
        CHOLMOD(free_dense) (&Ywork, cm) ;
        CHOLMOD(free_dense) (&Ework, cm) ;

    }

    CHOLMOD(free_sparse) (&Bset, cm) ;

    // turn memory tests back on, where we left off ]
    if (n > 10)
    {
        my_tries = save3 ;
    }

    CHOLMOD(free_sparse) (&I, cm) ;
    CHOLMOD(free_sparse) (&D, cm) ;
    CHOLMOD(free_sparse) (&Lo, cm) ;
    CHOLMOD(free_sparse) (&Up, cm) ;
    CHOLMOD(free_sparse) (&Lmat, cm) ;

    //--------------------------------------------------------------------------
    // update the factorization (real matrices only)
    //--------------------------------------------------------------------------

    // turn off memory tests [
    save3 = my_tries ;
    my_tries = -1 ;

    CHOLMOD(free_dense) (&B, cm) ;
    B = rhs (A, 1, n, 0) ;

    Int rank_max = ((n < 100) ? 33 : 2) ;

    for (rank = 1 ; rcond_ok && isreal && rank <= rank_max ; rank++)
    {

        // pick a random C
        Cdense = CHOLMOD(zeros) (n, rank, CHOLMOD_REAL + DTYPE, cm) ;

        if (Cdense != NULL)
        {
            Cx = Cdense->x ;
            for (k = 0 ; k < 10*rank ; k++)
            {
                i = nrand (n) ;                                 // RAND
                j = nrand (rank) ;                              // RAND
                Cx [i+j*n] += xrand (1.) ;                      // RAND
            }
        }

        C = CHOLMOD(dense_to_sparse) (Cdense, TRUE, cm) ;
        CHOLMOD(free_dense) (&Cdense, cm) ;

        // permute the rows according to L->Perm
        Cperm = CHOLMOD(submatrix) (C, P, n, NULL, -1, TRUE, TRUE, cm) ;

        // update
        CHOLMOD(updown) (TRUE, Cperm, L, cm) ;
        CHOLMOD(free_sparse) (&Cperm, cm) ;

        // solve (G+C*C')x=b
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

        // an alternative method would be to compute (G*x + C*(C'*x) - b)

        // compute S = G+C*C', with no sort
        Ct = CHOLMOD(transpose) (C, 1, cm) ;
        CC = CHOLMOD(ssmult) (C, Ct, 0, TRUE, FALSE, cm) ;
        S = CHOLMOD(add) (G, CC, one, one, TRUE, TRUE, cm) ;
        Ssym = CHOLMOD(copy) (S, 1, 1, cm) ;
        CHOLMOD(free_sparse) (&CC, cm) ;
        CHOLMOD(free_sparse) (&Ct, cm) ;

        // compute norm (S*x-b)
        r = resid (Ssym, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        //----------------------------------------------------------------------
        // factor A+CC' from scratch, using same permutation
        //----------------------------------------------------------------------

        if (rank == 1)
        {
            save = cm->nmethods ;
            save2 = cm->method [0].ordering ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_GIVEN ;
            L2 = CHOLMOD(analyze_p) (Ssym, P, NULL, 0, cm) ;
            cm->nmethods = save ;
            cm->method [0].ordering = save2 ;
            CHOLMOD(factorize) (Ssym, L2, cm) ;
            X = CHOLMOD(solve) (CHOLMOD_A, L2, B, cm) ;
            r = resid (Ssym, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;
            CHOLMOD(free_factor) (&L2, cm) ;
        }

        CHOLMOD(free_sparse) (&Ssym, cm) ;

        //----------------------------------------------------------------------
        // downdate, with just the first half of C
        //----------------------------------------------------------------------

        csize = MAX (1, rank / 2) ;
        cset = CHOLMOD(malloc) (csize, sizeof (Int), cm) ;
        if (cset != NULL)
        {
            for (i = 0 ; i < csize ; i++)
            {
                cset [i] = i ;
            }
        }
        C2 = CHOLMOD(submatrix) (C, NULL, -1, cset, csize, TRUE, TRUE, cm) ;
        Cperm = CHOLMOD(submatrix) (C2, P, n, NULL, -1, TRUE, TRUE, cm) ;
        CHOLMOD(free) (csize, sizeof (Int), cset, cm) ;

        CHOLMOD(updown) (FALSE, Cperm, L, cm) ;
        CHOLMOD(free_sparse) (&Cperm, cm) ;

        // solve (G+C*C'-C2*C2')x=b
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

        // This is an expensive way to compute the residual.  A better method
        // would be to compute (G*x + C*(C'*x) - C2*(C2'*x) - b), using sdmult.
        // It is done just to test the ssmult routine.

        // compute S2 = G+C*C'-C2*C2', with no sort
        Ct = CHOLMOD(transpose) (C2, 1, cm) ;
        CC = CHOLMOD(ssmult) (C2, Ct, 0, TRUE, FALSE, cm) ;
        S2 = CHOLMOD(add) (S, CC, one, minusone, TRUE, FALSE, cm) ;
        CHOLMOD(free_sparse) (&CC, cm) ;
        CHOLMOD(free_sparse) (&Ct, cm) ;
        CHOLMOD(free_sparse) (&C2, cm) ;

        // Ssym is a symmetric/upper copy of S2
        Ssym = CHOLMOD(copy) (S2, 1, 1, cm) ;
        CHOLMOD(free_sparse) (&S2, cm) ;

        // compute norm (S2*x-b)
        r = resid (Ssym, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        //----------------------------------------------------------------------
        // factor S2 scratch, using same permutation
        //----------------------------------------------------------------------

        if (rank == 1)
        {
            save = cm->nmethods ;
            save2 = cm->method [0].ordering ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_GIVEN ;
            L2 = CHOLMOD(analyze_p) (Ssym, P, NULL, 0, cm) ;
            cm->nmethods = save ;
            cm->method [0].ordering = save2 ;
            CHOLMOD(factorize) (Ssym, L2, cm) ;
            X = CHOLMOD(solve) (CHOLMOD_A, L2, B, cm) ;
            r = resid (Ssym, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;
            CHOLMOD(free_factor) (&L2, cm) ;
        }

        //----------------------------------------------------------------------
        // re-do the symbolic factorization on L
        //----------------------------------------------------------------------

        CHOLMOD(resymbol) (Ssym, NULL, 0, TRUE, L, cm) ;

        // solve (G+C*C'-C2*C2')x=b again
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

        // compute norm (S2*x-b)
        r = resid (Ssym, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        CHOLMOD(free_sparse) (&Ssym, cm) ;

        //----------------------------------------------------------------------
        // downdate, with the remaining part of C, to get original L
        //----------------------------------------------------------------------

        if (rank > csize)
        {
            cset = CHOLMOD(malloc) (rank-csize, sizeof (Int), cm) ;
            if (cset != NULL)
            {
                for (i = csize ; i < rank ; i++)
                {
                    cset [i-csize] = i ;
                }
            }
            if (rank - csize == 4)
            {
                // test the subset print/check routine
                CHOLMOD(print_subset) (cset, rank-csize, rank, "cset", cm) ;
            }
            C2 = CHOLMOD(submatrix) (C, NULL, -1, cset, rank-csize, TRUE, TRUE,
                    cm) ;
            Cperm = CHOLMOD(submatrix) (C2, P, n, NULL, -1, TRUE, TRUE, cm) ;

            CHOLMOD(free) (rank-csize, sizeof (Int), cset, cm) ;
            CHOLMOD(updown) (FALSE, Cperm, L, cm) ;
            CHOLMOD(free_sparse) (&Cperm, cm) ;
            CHOLMOD(free_sparse) (&C2, cm) ;

            // solve the original system
            X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

            // compute the residual
            r = resid (A, X, B) ;
            MAXERR (maxerr, r, 1) ;
            CHOLMOD(free_dense) (&X, cm) ;
        }

        CHOLMOD(free_sparse) (&C, cm) ;
        CHOLMOD(free_sparse) (&S, cm) ;
    }

    // turn memory tests back on, where we left off ]
    my_tries = save3 ;

    CHOLMOD(free_sparse) (&G, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // factor again, then change the factor type many times
    //--------------------------------------------------------------------------

    C = CHOLMOD(copy_sparse) (A, cm) ;
    if (C != NULL)
    {
        C->sorted = FALSE ;
    }
    L = CHOLMOD(analyze) (C, cm) ;
    CHOLMOD(factorize) (C, L, cm) ;

    if (L != NULL && !(L->is_super))
    {
        CHOLMOD(resymbol) (C, NULL, 0, TRUE, L, cm) ;
    }

    CHOLMOD(free_dense) (&B, cm) ;
    B = rhs (C, 1, n, 0) ;

    cm->prefer_zomplex =  prefer_zomplex ;
    X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
    cm->prefer_zomplex = FALSE ;
    r = resid (C, X, B) ;
    MAXERR (maxerr, r, 1) ;
    CHOLMOD(free_dense) (&X, cm) ;

    for (k = 0 ; k < NFTYPES ; k++)
    {

        if (co_types [k] && n > 1)
        {
            // reallocate column zero of L, to make it non-monotonic
            CHOLMOD(reallocate_column) (0, n, L, cm) ;
        }
        Lxtype = (L == NULL) ? CHOLMOD_REAL : (L->xtype) ;

        CHOLMOD(change_factor) (Lxtype, ll_types [k], FALSE, pk_types [k],
                mn_types [k], L, cm) ;

        cm->prefer_zomplex = prefer_zomplex ;
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        cm->prefer_zomplex = FALSE ;
        r = resid (C, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
    }

    // reallocate a column and solve again
    if (n > 3)
    {
        CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, TRUE, TRUE, L, cm) ;
        CHOLMOD(reallocate_column) (0, n, L, cm) ;
        cm->prefer_zomplex =  prefer_zomplex ;
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        cm->prefer_zomplex = FALSE ;
        r = resid (C, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
    }

    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;

    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free_dense) (&Breal, cm) ;
    CHOLMOD(free_dense) (&Bcomplex, cm) ;
    CHOLMOD(free_dense) (&Bzomplex, cm) ;

    CHOLMOD(free_dense) (&B2zomplex, cm) ;
    CHOLMOD(free_dense) (&B2complex, cm) ;
    CHOLMOD(free_dense) (&B2real, cm) ;

    //--------------------------------------------------------------------------
    // factorize and solve (A*A'+beta*I)x=b or A'x=b
    //--------------------------------------------------------------------------

    if (A->stype == 0)
    {
        Real *Rx, *Rz, *Xx, *Xz ;
        double beta [2] ;
        beta [0] = 3.14159 ;
        beta [1] = 0 ;
        L = CHOLMOD(analyze) (A, cm) ;
        CHOLMOD(factorize_p) (A, beta, NULL, 0, L, cm) ;

        CHOLMOD(free_dense) (&B, cm) ;
        B = rhs (A, 1, n, 0) ;

        cm->prefer_zomplex = prefer_zomplex ;
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        cm->prefer_zomplex = FALSE ;

        // compute the residual

        // W = A'*X
        W = CHOLMOD(zeros) (ncol, 1, xtype + DTYPE, cm) ;
        CHOLMOD(sdmult) (A, 2, one, zero, X, W, cm) ;

        // R = B
        R = CHOLMOD(copy_dense) (B, cm) ;

        // R = A*W - R
        CHOLMOD(sdmult) (A, 0, one, minusone, W, R, cm) ;

        // R = R + beta*X
        if (R != NULL && X != NULL)
        {
            Rx = R->x ;
            Rz = R->z ;
            Xx = X->x ;
            Xz = X->z ;

            for (i = 0 ; i < nrow ; i++)
            {
                switch (xtype)
                {
                    case CHOLMOD_REAL:
                        Rx [i] += beta [0] * Xx [i] ;
                        break ;

                    case CHOLMOD_COMPLEX:
                        Rx [2*i  ] += beta [0] * Xx [2*i  ] ;
                        Rx [2*i+1] += beta [0] * Xx [2*i+1] ;
                        break ;

                    case CHOLMOD_ZOMPLEX:
                        Rx [i] += beta [0] * Xx [i] ;
                        Rz [i] += beta [0] * Xz [i] ;
                        break ;
                }
            }
        }

        r = CHOLMOD(norm_dense) (R, 2, cm) ;
        bnorm = CHOLMOD(norm_dense) (B, 2, cm) ;
        xnorm = CHOLMOD(norm_dense) (X, 2, cm) ;
        norm = MAX (r, xnorm) ;
        if (norm > 0)
        {
            r /= norm ;
        }
        MAXERR (maxerr, r, 1) ;

        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_dense) (&R, cm) ;
        CHOLMOD(free_dense) (&W, cm) ;
        CHOLMOD(free_factor) (&L, cm) ;
        CHOLMOD(free_dense) (&B, cm) ;
    }

    //--------------------------------------------------------------------------
    // test rowdel and updown (real matrices only)
    //--------------------------------------------------------------------------

    if (isreal && A->stype == 1 && n > 0 && n < NLARGE)
    {
        Int save4, save5, save6 ;
        save4 = cm->nmethods ;
        save5 = cm->method [0].ordering ;
        save6 = cm->supernodal ;

        cm->nmethods = 1 ;
        cm->method [0].ordering = CHOLMOD_NATURAL ;
        cm->supernodal = CHOLMOD_SUPERNODAL ;

        CHOLMOD(free_dense) (&B, cm) ;
        B = rhs (A, 1, n, 0) ;

        L = CHOLMOD(analyze) (A, cm) ;
        CHOLMOD(factorize) (A, L, cm) ;

        // solve Ax=b
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        r = resid (A, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // determine the new system with row/column k missing
        k = n/2 ;
        S = CHOLMOD(copy) (A, 0, 1, cm) ;
        RowK = CHOLMOD(submatrix) (S, NULL, -1, &k, 1, TRUE, TRUE, cm) ;
//      CHOLMOD(print_sparse) (S, "S", cm) ;
//      CHOLMOD(print_sparse) (RowK, "RowK of S", cm) ;
        CHOLMOD(free_sparse) (&RowK, cm) ;
        prune_row (S, k) ;
        if (S != NULL)
        {
            S->stype = 1 ;
        }

        // delete row k of L (converts to LDL')
        CHOLMOD(rowdel) (k, NULL, L, cm) ;
        CHOLMOD(resymbol) (S, NULL, 0, TRUE, L, cm) ;

        // solve with row k missing
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        r = resid (S, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;
        CHOLMOD(free_sparse) (&S, cm) ;

        // factorize again
        CHOLMOD(free_factor) (&L, cm) ;
        L = CHOLMOD(analyze) (A, cm) ;
        CHOLMOD(factorize) (A, L, cm) ;

        // rank-3 update (converts to LDL') and solve
        C = CHOLMOD(speye) (n, 3, CHOLMOD_REAL + DTYPE, cm) ;
        CC = CHOLMOD(aat) (C, NULL, 0, 1, cm) ;
        S = CHOLMOD(add) (A, CC, one, one, TRUE, TRUE, cm) ;
        if (S != NULL)
        {
            S->stype = 1 ;
        }
        CHOLMOD(updown) (TRUE, C, L, cm) ;
        X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
        r = resid (S, X, B) ;
        MAXERR (maxerr, r, 1) ;
        CHOLMOD(free_dense) (&X, cm) ;

        // free everything
        CHOLMOD(free_sparse) (&S, cm) ;
        CHOLMOD(free_sparse) (&CC, cm) ;
        CHOLMOD(free_sparse) (&C, cm) ;
        CHOLMOD(free_sparse) (&S, cm) ;
        CHOLMOD(free_factor) (&L, cm) ;
        CHOLMOD(free_dense) (&B, cm) ;

        cm->nmethods = save4 ;
        cm->method [0].ordering = save5 ;
        cm->supernodal = save6 ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    CHOLMOD(free) (nrow, sizeof (Int), PP, cm) ;
    progress (0, '.') ;
    return (maxerr) ;
}

