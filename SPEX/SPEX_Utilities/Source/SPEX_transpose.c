//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_transpose: Transpose a CSC matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_WORK       \
    SPEX_FREE(w);

#define SPEX_FREE_ALL        \
    SPEX_FREE_WORK;          \
    SPEX_matrix_free(&C, option);

#include "spex_util_internal.h"

/* Purpose: This function sets C = A', where A must be a SPEX_CSC matrix
 * C_handle is NULL on input. On output, C_handle contains a pointer to A'
 */

SPEX_info SPEX_transpose
(
    SPEX_matrix *C_handle,      // C = A'
    SPEX_matrix A,              // Matrix to be transposed
    const SPEX_options option
)
{

    SPEX_info info;
    if (!spex_initialized ( )) return (SPEX_PANIC);
    // Check input
    SPEX_REQUIRE_KIND (A, SPEX_CSC);
    if (!C_handle)       { return SPEX_INCORRECT_INPUT;}

    // Declare workspace and C
    int64_t *w = NULL;
    SPEX_matrix C = NULL;
    int64_t nz;                            // Number of nonzeros in A
    int64_t p, q, j, n, m;
    info = SPEX_matrix_nnz(&nz, A, option);
    if (info != SPEX_OK) {return info;}
    m = A->m ; n = A->n ;
    ASSERT( m >= 0);
    ASSERT( n >= 0);

    // C is also CSC and its type is the same as A
    SPEX_CHECK(SPEX_matrix_allocate(&C, SPEX_CSC, A->type, n, m, nz,
        false, true, option));

    // Declare workspace
    w = (int64_t*) SPEX_calloc(m, sizeof(int64_t));
    if (!w)
    {
        SPEX_FREE_ALL;
        return SPEX_OUT_OF_MEMORY;
    }
    // Compute row counts
    for (p = 0 ; p < nz ; p++)
    {
        w [A->i [p]]++ ;
    }

    // Compute row pointers
    spex_cumsum (C->p, w, m);
    // Populate C
    for (j = 0 ; j < n ; j++)
    {
        for (p = A->p [j] ; p < A->p [j+1] ; p++)
        {
            q = w [A->i [p]]++;
            C->i [q] = j ;                 // place A(i,j) as entry C(j,i)

            // assign C->x[q] = A->x[p]
            if (A->type == SPEX_MPZ)
            {
                SPEX_MPZ_SET(C->x.mpz[q], A->x.mpz[p]);
            }
            else if (A->type == SPEX_MPQ)
            {
                SPEX_MPQ_SET(C->x.mpq[q], A->x.mpq[p]);
            }
            else if (A->type == SPEX_MPFR)
            {
                SPEX_MPFR_SET(C->x.mpfr[q], A->x.mpfr[p],
                    SPEX_OPTION_ROUND(option));
            }
            else if (A->type == SPEX_INT64)
            {
                C->x.int64[q] = A->x.int64[p];
            }
            else
            {
                C->x.fp64[q] = A->x.fp64[p];
            }
        }
    }
    SPEX_MPQ_SET(C->scale, A->scale);

    (*C_handle) = C;
    SPEX_FREE_WORK;
    return SPEX_OK;
}
