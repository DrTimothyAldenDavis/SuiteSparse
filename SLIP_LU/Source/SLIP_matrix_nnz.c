//------------------------------------------------------------------------------
// SLIP_LU/SLIP_matrix_nnz: find # of entries in a matrix
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#pragma GCC diagnostic ignored "-Wunused-variable"
#include "slip_internal.h"

int64_t SLIP_matrix_nnz     // return # of entries in A, or -1 on error
(
    const SLIP_matrix *A,      // matrix to query
    const SLIP_options *option // command options, currently unused
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!slip_initialized ( )) return (-1) ;

    if (A == NULL)
    {
        return (-1) ;
    }

    //--------------------------------------------------------------------------
    // find nnz (A)
    //--------------------------------------------------------------------------

    // In all three cases, SLIP_matrix_nnz(A,option) is <= A->nzmax.

    switch (A->kind)
    {
        case SLIP_CSC:
            // CSC matrices:  nnz(A) is given by Ap[n].  A->nz is ignored.
            return ((A->p == NULL || A->n < 0) ? (-1) : A->p [A->n]) ;
        case SLIP_TRIPLET:
            // triplet matrices:  nnz(A) is given by A->nz.
            return (A->nz) ;
        case SLIP_DENSE:
            // dense matrices: nnz(A) is always m*n.  A->nz is ignored.
            return ((A->m < 0 || A->n < 0)? (-1) : (A->m * A->n)) ;
        default:
            return (-1) ;
    }
}

