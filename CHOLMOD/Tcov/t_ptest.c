//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_ptest: test ssmult, sort, using permutations
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// returns r = norm (P*A - P*B), where P is a permutation matrix created from
// perm, of size n (which must equal A->nrow and B->nrow).  If perm is NULL on
// input, P = I is used.

double ptest (cholmod_sparse *A, cholmod_sparse *B, Int *perm, Int n)
{

    // P = permutation matrix from perm
    int xtype = (A == NULL) ? CHOLMOD_PATTERN : A->xtype ;
    cholmod_sparse *P = perm_matrix (perm, n, xtype + DTYPE, cm) ;

    // PA = P*A
    cholmod_sparse *PA = CHOLMOD(ssmult) (P, A, 0, true, false, cm) ;
    CHOLMOD(sort) (PA, cm) ;

    // PB = P*B
    cholmod_sparse *PB = CHOLMOD(ssmult) (P, B, 0, true, false, cm) ;
    CHOLMOD(sort) (PB, cm) ;

    // E = PA-PB
    cholmod_sparse *E = CHOLMOD(add) (PA, PB, one, minusone, true, false, cm) ;

    // rnorm = norm (A)
    double rnorm = CHOLMOD(norm_sparse) (E, 0, cm) ;

    CHOLMOD(free_sparse) (&P, cm) ;
    CHOLMOD(free_sparse) (&PA, cm) ;
    CHOLMOD(free_sparse) (&PB, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    return (rnorm) ;
}

