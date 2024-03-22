//------------------------------------------------------------------------------
// Demo/spex_read_dense: reads an integral dense matrix
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: Read a dense matrix for RHS vectors.
 * the values in the file must be integers
 */

#include "spex_demos.h"


SPEX_info spex_demo_read_dense
(
    SPEX_matrix *b_handle,  // Matrix to be constructed
    FILE *file,             // file to read from (must already be open)
    SPEX_options option
)
{

    if (file == NULL)
    {
        printf ("invalid inputs\n");
        return SPEX_INCORRECT_INPUT;
    }
    int64_t nrows, ncols;
    SPEX_info info ;

    // First, we obtain the dimension of the matrix
    int s = fscanf(file, "%"PRId64" %"PRId64, &nrows, &ncols);
    if (feof(file) || s < 2)
    {
        printf ("premature end-of-file\n");
        return SPEX_INCORRECT_INPUT;
    }

    // Now, we create our dense mpz_t matrix
    SPEX_matrix A = NULL;
    info = SPEX_matrix_allocate(&A, SPEX_DENSE, SPEX_MPZ, nrows, ncols,
        nrows*ncols, false, true, option);
    if (info != SPEX_OK)
    {
        return (info);
    }

    // We now populate the matrix b.
    for (int64_t i = 0; i < nrows; i++)
    {
        for (int64_t j = 0; j < ncols; j++)
        {
            info = SPEX_gmp_fscanf(file, "%Zd", &(A->x.mpz[i+j*nrows]));
            if (info != SPEX_OK)
            {
                printf("\n\nhere at i = %"PRId64" and j = %"PRId64"", i, j);
                return SPEX_INCORRECT_INPUT;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Success, set b_handle = A
    //--------------------------------------------------------------------------

    (*b_handle) = A;
    return (info);
}
