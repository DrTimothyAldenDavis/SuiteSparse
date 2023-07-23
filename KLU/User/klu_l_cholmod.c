//------------------------------------------------------------------------------
// SuiteSparse/KLU/User/klu_l_cholmod.c: KLU SuiteSparse_long interface to CHOLMOD
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* klu_l_cholmod: user-defined ordering function to interface KLU to CHOLMOD.
 *
 * This routine is an example of a user-provided ordering function for KLU.
 * Its return value is klu_l_cholmod's estimate of max (nnz(L),nnz(U)):
 *      0 if error,
 *      -1 if OK, but estimate of max (nnz(L),nnz(U)) not computed
 *      > 0 if OK and estimate computed.
 *
 * This function can be assigned to KLU's Common->user_order function pointer.
 */

#include "klu_cholmod.h"
#include "cholmod.h"
#define TRUE 1
#define FALSE 0

SuiteSparse_long klu_l_cholmod
(
    /* inputs */
    SuiteSparse_long n,              /* A is n-by-n */
    SuiteSparse_long Ap [ ],         /* column pointers */
    SuiteSparse_long Ai [ ],         /* row indices */
    /* outputs */
    SuiteSparse_long Perm [ ],       /* fill-reducing permutation */
    /* user-defined */
    klu_l_common *Common    /* user-defined data is in Common->user_data */
)
{
    double one [2] = {1,0}, zero [2] = {0,0}, lnz = 0 ;
    cholmod_sparse Amatrix, *A, *AT, *S ;
    cholmod_factor *L ;
    cholmod_common cm ;
    SuiteSparse_long *P ;
    SuiteSparse_long k ;
    int symmetric ;
    printf ("------------------- KLU User\n") ;
    klu_l_common km ;
    klu_l_defaults (&km) ;

    if (Ap == NULL || Ai == NULL || Perm == NULL || n < 0)
    {
        /* invalid inputs */
        return (0) ;
    }

    /* start CHOLMOD */
    cholmod_l_start (&cm) ;
    cm.supernodal = CHOLMOD_SIMPLICIAL ;
    cm.print = 0 ;

    /* construct a CHOLMOD version of the input matrix A */
    A = &Amatrix ;
    A->nrow = n ;                   /* A is n-by-n */
    A->ncol = n ;
    A->nzmax = Ap [n] ;             /* with nzmax entries */
    A->packed = TRUE ;              /* there is no A->nz array */
    A->stype = 0 ;                  /* A is unsymmetric */
    A->itype = CHOLMOD_INT ;
    A->xtype = CHOLMOD_PATTERN ;
    A->dtype = CHOLMOD_DOUBLE ;
    A->nz = NULL ;
    A->p = Ap ;                     /* column pointers */
    A->i = Ai ;                     /* row indices */
    A->x = NULL ;                   /* no numerical values */
    A->z = NULL ;
    A->sorted = FALSE ;             /* columns of A are not sorted */

    /* get the user_data; default is symmetric if user_data is NULL */
    symmetric = true ;
    cm.nmethods = 1 ;
    cm.method [0].ordering = CHOLMOD_AMD ;
    SuiteSparse_long *user_data = Common->user_data ;
    if (user_data != NULL)
    {
        symmetric = (user_data [0] != 0) ;
        cm.method [0].ordering = user_data [1] ;
    }

    /* AT = pattern of A' */
    AT = cholmod_l_transpose (A, 0, &cm) ;
    if (symmetric)
    {
        /* S = the symmetric pattern of A+A' */
        S = cholmod_l_add (A, AT, one, zero, FALSE, FALSE, &cm) ;
        cholmod_l_free_sparse (&AT, &cm) ;
        if (S != NULL)
        {
            S->stype = 1 ;
        }
    }
    else
    {
        /* S = A'.  CHOLMOD will order S*S', which is A'*A */
        S = AT ;
    }

    /* order and analyze S or S*S' */
    L = cholmod_l_analyze (S, &cm) ;

    /* copy the permutation from L to the output */
    if (L != NULL)
    {
        P = L->Perm ;
        for (k = 0 ; k < n ; k++)
        {
            Perm [k] = P [k] ;
        }
        lnz = cm.lnz ;
    }

    cholmod_l_free_sparse (&S, &cm) ;
    cholmod_l_free_factor (&L, &cm) ;
    cholmod_l_finish (&cm) ;
    return (lnz) ;
}
