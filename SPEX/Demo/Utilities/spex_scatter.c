//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_transpose: Transpose a CSC matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------
 
#include "spex_util_internal.h"
#include "spex_demos.h"
/* x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse */
SPEX_info spex_scatter 
(
    const SPEX_matrix A, 
    int64_t j, 
    mpz_t beta, 
    int64_t *w, 
    SPEX_matrix x, 
    int64_t mark,
    SPEX_matrix C, 
    int64_t *nz
)
{
    SPEX_info info;
     
    int64_t i, p, *Ap, *Ai, *Ci ;
    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ);
    SPEX_REQUIRE (A, SPEX_CSC,   SPEX_MPZ);
    if (!w ) return (SPEX_INCORRECT_INPUT) ;     /* check inputs */
        
    Ap = A->p ; Ai = A->i ; ; Ci = C->i ;

    for (p = Ap [j] ; p < Ap [j+1] ; p++)
    {
        i = Ai [p] ;                            /* A(i,j) is nonzero */
        if (w [i] < mark)
        {
            w [i] = mark ;                      /* i is new entry in column j */
            Ci [(*nz)++] = i ;                     /* add i to pattern of C(:,j) */
            if (x) SPEX_MPZ_MUL(x->x.mpz [i], beta, A->x.mpz [p]) ;      /* x(i) = beta*A(i,j) */
        }
        else if (x) SPEX_MPZ_ADDMUL(x->x.mpz [i], beta , A->x.mpz [p]) ;    /* i exists in C(:,j) already x [i] += beta * Ax [p] */
    }
    return (SPEX_OK);
}
