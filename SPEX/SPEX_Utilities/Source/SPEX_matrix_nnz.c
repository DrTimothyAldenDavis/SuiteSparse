//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_matrix_nnz: find # of entries in a matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#if defined (__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "spex_util_internal.h"

SPEX_info SPEX_matrix_nnz       // find the # of entries in A
(
    int64_t *nnz,               // # of entries in A, -1 if A is NULL
    const SPEX_matrix A,        // matrix to query
    const SPEX_options option   // command options, currently unused
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized ( )) return (SPEX_PANIC);

    if (nnz == NULL)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    (*nnz) = -1 ;

    if (A == NULL)
    {
        return (SPEX_INCORRECT_INPUT);
    }

    //--------------------------------------------------------------------------
    // find nnz (A)
    //--------------------------------------------------------------------------

    // If kind != SPEX_DYNAMIC_CSC, SPEX_matrix_nnz(&nnz, A, option) returns
    // with nnz <= A->nzmax.

    switch (A->kind)
    {
        case SPEX_CSC:
        {
            // CSC matrices:  nnz(A) is given by Ap[n].  A->nz is ignored.
            *nnz = (A->p == NULL || A->n < 0) ? (-1) : A->p [A->n] ;
        }
        break;

        case SPEX_TRIPLET:
        {
            // triplet matrices:  nnz(A) is given by A->nz.
            *nnz = A->nz ;
        }
        break;

        case SPEX_DENSE:
        {
            // dense matrices: nnz(A) is always m*n.  A->nz is ignored.
            *nnz = (A->m < 0 || A->n < 0)? (-1) : (A->m * A->n);
        }
        break;


        default:
            return (SPEX_INCORRECT_INPUT);
    }

    return ((*nnz < 0) ? SPEX_INCORRECT_INPUT : SPEX_OK);
}

