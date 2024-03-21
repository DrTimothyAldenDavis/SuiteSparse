//------------------------------------------------------------------------------
// Demo/spex_demo_tripread: reads a matrix stored in triplet format of a given type
//------------------------------------------------------------------------------

// SPEX: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function reads a matrix stored in triplet format of a given type
 * This format used is illustrated in the example mat files.
 *
 * The first line of the file contains three integers: m, n, nnz,
 * where the matrix is m-by-n with nnz entries.
 *
 * Each of the following nnz lines contains a single triplet: i, j, aij,
 * which defines the row index (i), column index (j), and value (aij) of
 * the entry A(i,j).
 */


#include "spex_demos.h"

SPEX_info spex_demo_tripread
(
    SPEX_matrix *A_handle,      // Matrix to be populated
    FILE *file,                 // file to read from (must already be open)
    SPEX_type C_type,           // C->type: mpz_t or double
    SPEX_options option
)
{
    SPEX_info info ;

    if (A_handle == NULL || file == NULL)
    {
        return (SPEX_INCORRECT_INPUT) ;
    }

    (*A_handle) = NULL ;

    int64_t m, n, nz;

    if (C_type != SPEX_FP64 && C_type != SPEX_MPZ )
    {
        printf("%d\n",C_type);
        printf ("this function only supports double or mpz matrices\n");
    }

    // Read in size of matrix & number of nonzeros
    int s = fscanf(file, "%"PRId64" %"PRId64" %"PRId64"\n", &m, &n, &nz);
    if (feof(file) || s < 3)
    {
        printf ("premature end-of-file 1\n");
        return SPEX_INCORRECT_INPUT;
    }

    // Allocate memory for A
    // A is a triplet mpz_t or  double matrix
    SPEX_matrix A = NULL;
    info = SPEX_matrix_allocate(&A, SPEX_TRIPLET, C_type, m, n, nz,
        false, true, option);

    if (info != SPEX_OK)
    {
        printf ("unable to allocate matrix\n");
        return (info);
    }

    // Read in the values from file

    switch(C_type)
    {
        case SPEX_FP64:
            for (int64_t k = 0; k < nz; k++)
            {
                s = fscanf(file, "%"PRId64" %"PRId64" %lf\n",
                    &(A->i[k]), &(A->j[k]), &(A->x.fp64[k]));
                if ((feof(file) && k != nz-1) || s < 3)
                {
                    printf ("premature end-of-file\n");
                    SPEX_matrix_free(&A, option);
                    return SPEX_INCORRECT_INPUT;
                }
                //Conversion from 1 based to 0 based
                A->i[k] -= 1;
                A->j[k] -= 1;
            }
            break;

        case SPEX_MPZ:
            for (int64_t k = 0; k < nz; k++)
            {
                info = SPEX_gmp_fscanf(file, "%"PRId64" %"PRId64" %Zd\n",
                        &A->i[k], &A->j[k], &A->x.mpz[k]);
                if ((feof(file) && k != nz-1) || info != SPEX_OK)
                {
                    printf ("premature end-of-file 2\n");
                    SPEX_matrix_free(&A, option);
                    return SPEX_INCORRECT_INPUT;
                }
                A->i[k] -= 1;
                A->j[k] -= 1;
            }
            break;
        default:
            printf ("type not supported\n");
            return SPEX_INCORRECT_INPUT;
    }

    // the triplet matrix now has nz entries
    A->nz = nz;

    info = SPEX_matrix_check (A, option);
    if (info != SPEX_OK)
    {
        printf ("invalid matrix\n");
        return (info);
    }

    // A now contains our input matrix in triplet format. We now
    // do a matrix copy to get it into CSC form
    // C is a copy of A which is CSC and mpz_t
    SPEX_matrix C = NULL;
    SPEX_matrix_copy(&C, SPEX_CSC, SPEX_MPZ, A, option);

    // Free A, set A_handle
    SPEX_matrix_free(&A, option);
    (*A_handle) = C;

    return SPEX_OK;
}
