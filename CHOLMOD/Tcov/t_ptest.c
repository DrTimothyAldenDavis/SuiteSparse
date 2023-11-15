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
// int psave = cm->print ;
// cm->print = 5 ;
// CHOLMOD(print_sparse) (A, "A for ptest", cm) ;
// CHOLMOD(print_sparse) (B, "B for ptest", cm) ;
// CHOLMOD(print_perm) (perm, n, n, "perm for ptest", cm) ;

    // P = permutation matrix from perm
    int xtype = (A == NULL) ? CHOLMOD_PATTERN : A->xtype ;
    cholmod_sparse *P = perm_matrix (perm, n, xtype + DTYPE, cm) ;
// CHOLMOD(print_sparse) (P, "P for ptest", cm) ;

    // PA = P*A
    cholmod_sparse *PA = CHOLMOD(ssmult) (P, A, 0, true, false, cm) ;
    CHOLMOD(sort) (PA, cm) ;
// CHOLMOD(print_sparse) (PA, "PA for ptest", cm) ;

    // PB = P*B
    cholmod_sparse *PB = CHOLMOD(ssmult) (P, B, 0, true, false, cm) ;
    CHOLMOD(sort) (PB, cm) ;
// CHOLMOD(print_sparse) (PB, "PB for ptest", cm) ;

    // E = PA-PB
    cholmod_sparse *E = CHOLMOD(add) (PA, PB, one, minusone, true, false, cm) ;
// CHOLMOD(print_sparse) (E, "E for ptest", cm) ;
// CHOLMOD(print_sparse) (PA, "PA now for ptest", cm) ;

    // rnorm = norm (A)
    double rnorm = CHOLMOD(norm_sparse) (E, 0, cm) ;
// printf ("rnorm %g\n", rnorm) ;

    CHOLMOD(free_sparse) (&P, cm) ;
    CHOLMOD(free_sparse) (&PA, cm) ;
    CHOLMOD(free_sparse) (&PB, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;

// cm->print = psave ;

    return (rnorm) ;
}

