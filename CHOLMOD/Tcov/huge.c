//------------------------------------------------------------------------------
// CHOLMOD/Tcov/huge: test program for CHOLMOD on huge matrices
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Tests on huge matrices */

#include "cm.h"
#include "amd.h"
#ifndef NCAMD
#include "camd.h"
#endif

#undef  ERROR
#include "cholmod_internal.h"

#undef  ERROR
#define ERROR(status,message) \
    CHOLMOD(error) (status, __FILE__, __LINE__, message, cm)

/* ========================================================================== */
/* === huge ================================================================= */
/* ========================================================================== */

void huge ( )
{
    cholmod_sparse *A, *C ;
    cholmod_triplet *T ;
    cholmod_factor *L ;
    cholmod_dense *X ;
    size_t n, nbig ;
    int ok = TRUE, save ;
    Int junk = 0 ;
    FILE *f ;
    double beta [2] ;

    n = SIZE_MAX ;
    CHOLMOD (free_work) (cm) ;
    CHOLMOD (allocate_work) (n, 0, 0, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    n = CHOLMOD(add_size_t) (n, 1, &ok) ; NOT (ok) ;

    /* create a fake zero sparse matrix, with huge dimensions */
    A = CHOLMOD (spzeros) (1, 1, 0, CHOLMOD_REAL, cm) ;
    A->nrow = SIZE_MAX ;
    A->ncol = SIZE_MAX ;
    A->stype = 0 ;

    /* create a fake factor, with huge dimensions.  */
    L = CHOLMOD (allocate_factor)  (1, cm) ;
    OKP (L) ;
    L->n = SIZE_MAX ;
    CHOLMOD (factorize) (A, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    /* free the fake factor */
    L->n = 1 ;
    CHOLMOD (free_factor) (&L, cm) ;

    /* create a valid factor to test resymbol */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
    C->stype = 1 ;
    L = CHOLMOD (analyze) (C, cm) ;
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

    C = CHOLMOD (allocate_sparse) (SIZE_MAX, SIZE_MAX, SIZE_MAX,
        0, 0, 0, 0, cm) ;
    printf ("cm->status %d\n", cm->status) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (rowcolcounts) (A, NULL, 0,
	&junk, &junk, &junk, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    C = CHOLMOD (submatrix) (A, &junk, SIZE_MAX/4, &junk, SIZE_MAX/4,
        0, 0, cm) ;

    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

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

    CHOLMOD (amd) (A, NULL, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    L = CHOLMOD (analyze) (A, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (L) ;

#ifndef NCAMD
    CHOLMOD (camd) (A, NULL, 0, &junk, NULL, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
#endif

    printf ("calling colamd\n") ;
    CHOLMOD (colamd) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

#ifndef NCAMD
    printf ("calling ccolamd\n") ;
    CHOLMOD (ccolamd) (A, NULL, 0, NULL, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
#endif

    CHOLMOD (etree) (A, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    L = CHOLMOD (allocate_factor) (SIZE_MAX, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (L) ;

#ifndef NPARTITION
    CHOLMOD (metis) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (bisect) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    CHOLMOD (nested_dissection) (A, NULL, 0, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
#endif

    CHOLMOD (postorder) (&junk, SIZE_MAX, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);

    /* causes overflow in 32-bit version, but not 64-bit */
    f = fopen ("../Tcov/Matrix/mega.tri", "r") ;
    T = CHOLMOD (read_triplet) (f, cm) ;
    if (sizeof (Int) == sizeof (int))
    {
	NOP (T) ;
	OK (cm->status != CHOLMOD_OK) ;
    }
    CHOLMOD (free_triplet) (&T, cm) ;
    fclose (f) ;

    n = SIZE_MAX ;
    X = CHOLMOD (allocate_dense) (n, 1, n, CHOLMOD_REAL, cm) ;
    printf ("status %d\n", cm->status) ;
    OK (cm->status == CHOLMOD_TOO_LARGE || cm->status == CHOLMOD_OUT_OF_MEMORY);
    NOP (X) ;

    /* supernodal symbolic test */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
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

    /* supernodal numeric test */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
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

    /* free the fake matrix */
    A->nrow = 1 ;
    A->ncol = 1 ;
    CHOLMOD (free_sparse) (&A, cm) ;

    fprintf (stderr, "\n") ;
}
