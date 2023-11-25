//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_huge: test program for CHOLMOD on huge matrices
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Tests on huge matrices

#include "cm.h"
#include "amd.h"
#ifndef NCAMD
#include "camd.h"
#endif

#undef  ERROR
#define ERROR(status,message) \
    CHOLMOD(error) (status, __FILE__, __LINE__, message, cm)

//------------------------------------------------------------------------------
// huge
//------------------------------------------------------------------------------

void huge ( )
{

    printf ("=================================\ntests with huge matrices:\n") ;
    cholmod_sparse *A, *C ;
    cholmod_triplet *T ;
    cholmod_factor *L ;
    cholmod_dense *X ;
    size_t n ;
    int ok = TRUE, save ;
    Int junk = 0 ;
    double beta [2] ;

    //--------------------------------------------------------------------------
    // allocate_work
    //--------------------------------------------------------------------------

    n = SIZE_MAX ;
    CHOLMOD (free_work) (cm) ;
    CHOLMOD (allocate_work) (n, 0, 0, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // add_size_t
    //--------------------------------------------------------------------------

    n = CHOLMOD(add_size_t) (n, 1, &ok) ; NOT (ok) ;

    //--------------------------------------------------------------------------
    // create a fake zero sparse matrix, with huge dimensions
    //--------------------------------------------------------------------------

    A = CHOLMOD (spzeros) (1, 1, 0, CHOLMOD_REAL + DTYPE, cm) ;
    A->nrow = SIZE_MAX ;
    A->ncol = SIZE_MAX ;
    A->stype = 0 ;

    //--------------------------------------------------------------------------
    // factorize
    //--------------------------------------------------------------------------

//  L = CHOLMOD (allocate_factor) (1, cm) ;
    L = CHOLMOD (alloc_factor) (1, DTYPE, cm) ;
    OKP (L) ;
    L->n = SIZE_MAX ;
    CHOLMOD (factorize) (A, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    // free the fake factor
    L->n = 1 ;
    CHOLMOD (free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // resymbol, rowfac, rowadd, rowdel, and updown
    //--------------------------------------------------------------------------

    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL + DTYPE, cm) ;
    OKP (C) ;
    C->stype = 1 ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    OKP (L) ;
    CHOLMOD (factorize) (C, L, cm) ;
    ok = CHOLMOD (resymbol) (C, NULL, 0, 0, L, cm) ;
    OK (ok) ;
    C->nrow = SIZE_MAX ;
    C->ncol = SIZE_MAX ;
    L->n = SIZE_MAX ;

    ok = CHOLMOD (resymbol) (C, NULL, 0, 0, L, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    printf ("rowfac:\n") ;
    beta [0] = 1 ;
    beta [1] = 0 ;
    C->xtype = CHOLMOD_COMPLEX ;
    L->xtype = CHOLMOD_COMPLEX ;
    ok = CHOLMOD (rowfac) (C, NULL, beta, 0, 0, L, cm) ;
    printf ("rowfac %d\n", cm->status) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    C->xtype = CHOLMOD_REAL ;
    L->xtype = CHOLMOD_REAL ;
    printf ("rowfac done:\n") ;

    C->stype = -1 ;
    ok = CHOLMOD (resymbol_noperm) (C, NULL, 0, 0, L, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    C->ncol = 1 ;
    CHOLMOD (rowadd) (0, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (rowdel) (0, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    C->ncol = 4 ;
    CHOLMOD (updown) (1, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // allocate_sparse
    //--------------------------------------------------------------------------

    C = CHOLMOD (allocate_sparse) (SIZE_MAX, SIZE_MAX, SIZE_MAX,
        0, 0, 0, CHOLMOD_PATTERN + DTYPE, cm) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // rowcolcounts
    //--------------------------------------------------------------------------

    CHOLMOD (rowcolcounts) (A, NULL, 0,
        &junk, &junk, &junk, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // submatrix
    //--------------------------------------------------------------------------

    C = CHOLMOD (submatrix) (A, &junk, SIZE_MAX/4, &junk, SIZE_MAX/4,
        0, 0, cm) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // transpose and variants
    //--------------------------------------------------------------------------

    ok = CHOLMOD (transpose_unsym) (A, 0, &junk, &junk, SIZE_MAX, A, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE ||
        cm->status == CHOLMOD_OUT_OF_MEMORY ||
        cm->status == CHOLMOD_INVALID);

    A->stype = 1 ;
    ok = CHOLMOD (transpose_sym) (A, 0, &junk, A, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE ||
        cm->status == CHOLMOD_OUT_OF_MEMORY ||
        cm->status == CHOLMOD_INVALID);

    C = CHOLMOD (ptranspose) (A, 0, &junk, NULL, 0, cm) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    A->stype = 0 ;

    //--------------------------------------------------------------------------
    // amd
    //--------------------------------------------------------------------------

    CHOLMOD (amd) (A, NULL, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // analyze
    //--------------------------------------------------------------------------

    L = CHOLMOD (analyze) (A, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (L) ;

    //--------------------------------------------------------------------------
    // camd
    //--------------------------------------------------------------------------

    #ifndef NCAMD
    CHOLMOD (camd) (A, NULL, 0, &junk, NULL, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    #endif

    //--------------------------------------------------------------------------
    // colamd
    //--------------------------------------------------------------------------

    printf ("calling colamd\n") ;
    CHOLMOD (colamd) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // ccolamd
    //--------------------------------------------------------------------------

    #ifndef NCAMD
    printf ("calling ccolamd\n") ;
    CHOLMOD (ccolamd) (A, NULL, 0, NULL, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    #endif

    //--------------------------------------------------------------------------
    // etree
    //--------------------------------------------------------------------------

    CHOLMOD (etree) (A, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // allocate factor
    //--------------------------------------------------------------------------

    L = CHOLMOD (allocate_factor) (SIZE_MAX, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (L) ;

    //--------------------------------------------------------------------------
    // alloc factor
    //--------------------------------------------------------------------------

    L = CHOLMOD (alloc_factor) (SIZE_MAX, DTYPE, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (L) ;

    //--------------------------------------------------------------------------
    // metis, bisect, and nested dissection
    //--------------------------------------------------------------------------

    #ifndef NPARTITION
    CHOLMOD (metis) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (bisect) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (nested_dissection) (A, NULL, 0, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    #endif

    //--------------------------------------------------------------------------
    // postorder
    //--------------------------------------------------------------------------

    CHOLMOD (postorder) (&junk, SIZE_MAX, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    //--------------------------------------------------------------------------
    // read_triplet
    //--------------------------------------------------------------------------

    // causes overflow in 32-bit version, but not 64-bit
    FILE *f = fopen ("Matrix/mega.tri", "r") ;
    OK (f != NULL) ;
    T = CHOLMOD (read_triplet) (f, cm) ;
    if (sizeof (Int) == sizeof (int))
    {
        NOP (T) ;
        OK (cm->status != CHOLMOD_OK) ;
    }
    CHOLMOD (free_triplet) (&T, cm) ;
    fclose (f) ;

    //--------------------------------------------------------------------------
    // allocate_dense
    //--------------------------------------------------------------------------

    n = SIZE_MAX ;
    X = CHOLMOD (allocate_dense) (n, 1, n, CHOLMOD_REAL + DTYPE, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (X) ;

    //--------------------------------------------------------------------------
    // supernodal symbolic
    //--------------------------------------------------------------------------

    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL + DTYPE, cm) ;
    OKP (C) ;
    C->stype = 1 ;
    save = cm->supernodal ;
    cm->supernodal = CHOLMOD_SIMPLICIAL ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    junk = 0 ;
    C->nrow = SIZE_MAX ;
    C->ncol = SIZE_MAX ;
    L->n = SIZE_MAX ;
    CHOLMOD (super_symbolic) (C, C, &junk, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    cm->supernodal = save ;
    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // supernodal numeric
    //--------------------------------------------------------------------------

    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL + DTYPE, cm) ;
    OKP (C) ;
    C->stype = -1 ;
    save = cm->supernodal ;
    cm->supernodal = CHOLMOD_SUPERNODAL ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    OK (cm->status == CHOLMOD_OK) ;
    C->nrow = SIZE_MAX ;
    C->ncol = SIZE_MAX ;
    L->n = SIZE_MAX ;
    CHOLMOD (super_numeric) (C, C, beta, L, cm) ;
    cm->supernodal = save ;
    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    //--------------------------------------------------------------------------
    // sort
    //--------------------------------------------------------------------------

    n = 100000 ;
    X = CHOLMOD(ones) (n, 1, CHOLMOD_REAL + DTYPE, cm) ;
    OKP (X) ;
    C = CHOLMOD(dense_to_sparse) (X, true, cm) ;
    OKP (C) ;
    Int *P = prand ((Int) n) ;                                  // RAND
    OKP (P) ;

    Int *Ci = C->i ;
    Real *Cx = C->x ;
    for (Int k = 0 ; k < n ; k++)
    {
        Ci [k] = P [k] ;
        Cx [k] = P [k] ;
    }

    C->sorted = false ;
    CHOLMOD(sort) (C, cm) ;
    OK (C->sorted) ;

    for (Int k = 0 ; k < n ; k++)
    {
        OK (Ci [k] == k) ;
        OK (Cx [k] == k) ;
    }

    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_dense) (&X, cm) ;
    CHOLMOD (free) (n, sizeof (Int), P, cm) ;

    //--------------------------------------------------------------------------
    // free the fake matrix
    //--------------------------------------------------------------------------

    A->nrow = 1 ;
    A->ncol = 1 ;
    CHOLMOD (free_sparse) (&A, cm) ;
}

