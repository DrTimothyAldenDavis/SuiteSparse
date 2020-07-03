//------------------------------------------------------------------------------
// SLIP_LU/slip_sparse_realloc: double the space for a sparse mpz matrix
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function expands a CSC SLIP_matrix by doubling its size. This
 * version merely expands x and i and does not initialize/allocate the values!
 * The only purpose of this function is for the factorization, it does not work
 * for general sparse matrices
 */

#include "slip_internal.h"

SLIP_info slip_sparse_realloc
(
    SLIP_matrix* A // the matrix to be expanded
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SLIP_REQUIRE (A, SLIP_CSC, SLIP_MPZ) ;

    //--------------------------------------------------------------------------
    // double the size of A->x and A->i
    //--------------------------------------------------------------------------

    int64_t nzmax = A->nzmax ;

    bool okx, oki ;
    A->x.mpz = (mpz_t *)
        SLIP_realloc (2*nzmax, nzmax, sizeof (mpz_t), A->x.mpz, &okx) ;
    A->i = (int64_t *)
        SLIP_realloc (2*nzmax, nzmax, sizeof (int64_t), A->i, &oki) ;
    if (!oki || !okx)
    {
        return (SLIP_OUT_OF_MEMORY) ;
    }

    A->nzmax = 2*nzmax ;

    //--------------------------------------------------------------------------
    // set newly allocated mpz entries to NULL
    //--------------------------------------------------------------------------

    for (int64_t p = nzmax ; p < 2*nzmax ; p++)
    {
        SLIP_MPZ_SET_NULL (A->x.mpz [p]) ;
    }

    return (SLIP_OK) ;
}

