//------------------------------------------------------------------------------
// SPEX_Utilities/spex_matrix_multiply: Multiply two matrices
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2023, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_util_internal.h"
#include "spex_demos.h"

SPEX_info spex_sparse_matrix_multiply 
(
    //Output
    SPEX_matrix *C_handle,
    //Input
    const SPEX_matrix A, 
    const SPEX_matrix B
)
{
    SPEX_info info;
    //Declare variables
    int64_t p, j, nz = 0, anz, *Cp, *Ci, *Bp, m, n, bnz, *w, values, *Bi ;
    SPEX_matrix x, Bx, Cx, C=NULL;
    mpz_t one;


    //Check inputs
    //if (!CS_CSC (A) || !CS_CSC (B)) return (NULL) ;    
    if (A->n != B->m) return SPEX_INCORRECT_INPUT ;

    SPEX_mpz_init(one);
    SPEX_MPZ_SET_UI(one,1);

    m = A->m ; anz = A->p [A->n] ;
    n = B->n ; bnz = B->p [n] ;
    w = (int64_t*) SPEX_calloc (m, sizeof (int64_t)) ;                    /* get workspace */
    values = (A->x.mpz != NULL) && (B->x.mpz != NULL) ;
    //x = values ? cs_malloc (m, sizeof (double)) : NULL ; /* get workspace */
    if(values)
    {
        SPEX_CHECK(SPEX_matrix_allocate(&x, SPEX_DENSE, SPEX_MPZ, m, 1, m,
        false, false, NULL));
    }
    else
    {
        return SPEX_OUT_OF_MEMORY;
    }
    
    SPEX_CHECK(SPEX_matrix_allocate(&C, SPEX_CSC, SPEX_MPZ, m, n, anz+bnz,
        false, false, NULL));
    if (!C || !w || (values && !x)) return (SPEX_OUT_OF_MEMORY) ;

    for (j = 0 ; j < n ; j++)
    {
        if (nz + m > C->nzmax && spex_sparse_realloc(C)!=SPEX_OK)
        {
            return (SPEX_OUT_OF_MEMORY) ;             /* out of memory */
        } 
        C->p [j] = nz ;                   /* column j of C starts here */
        for (p = B->p [j] ; p < B->p [j+1] ; p++)
        {
            spex_scatter (A, B->i [p], B->x.mpz ? B->x.mpz [p] : one, w, x, j+1, C, &nz) ;//returns nz
        }
        if (values) for (p = C->p [j] ; p < nz ; p++) SPEX_MPZ_SET( C->x.mpz[p] , x->x.mpz[C->i[p]] );
    }
    C->p [n] = nz ;                       /* finalize the last column of C */

    spex_sparse_collapse(C); /*remove extra space from C*/
    //SPEX_FREE_WORKSPACE;
    (*C_handle)=C;
    return SPEX_OK;
}
