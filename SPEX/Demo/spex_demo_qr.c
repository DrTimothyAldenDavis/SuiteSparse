//------------------------------------------------------------------------------
// SPEX_QR/spex_demo_qr.c: Demo of SPEX QR
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2021-2023, Chris Lourenco, Lorena Mejia Domenzain,
// Timothy A. Davis, and Erick Moreno-Centeno. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "SPEX.h"
#include "spex_util_internal.h"
#include "spex_qr_internal.h"
#include "spex_cholesky_internal.h"
#include "spex_demos.h"

#define FREE_WORKSPACE                \
    SPEX_matrix_free(&A1, NULL);      \
    SPEX_matrix_free(&A1_init, NULL); \
    SPEX_matrix_free(&A2, NULL);      \
    SPEX_matrix_free(&A2_init, NULL); \
    SPEX_matrix_free(&b1, NULL);      \
    SPEX_matrix_free(&b1_init, NULL); \
    SPEX_matrix_free(&b2, NULL);      \
    SPEX_matrix_free(&b2_init, NULL); \
    SPEX_matrix_free(&x1, NULL) ;     \
    SPEX_matrix_free(&x2, NULL) ;     \
    SPEX_FREE(option);                \
    SPEX_finalize();                  \

#ifndef ASSERT
#define ASSERT assert
#endif

int main(int argc, char *argv[])
{

    //--------------------------------------------------------------------------
    // Prior to using SPEX QR, its environment must be initialized. This is done
    // by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_initialize();

    // SPEX Options
    SPEX_options option = NULL;
    SPEX_create_default_options(&option);

    // We will generate two types of matrices to examine the behavior of SPEX QR
    int m1 = 50;
    int n1 = 30;
    int m2 = 30;
    int n2 = 50;

    // Rectangular random matrix with m > n
    SPEX_matrix A1 = NULL;      // mpz version
    SPEX_matrix A1_init = NULL; // int64_t version
    SPEX_matrix b1 = NULL;      // mpz version
    SPEX_matrix b1_init = NULL; // int64_t version

    // Rectangular random matrix with n > m
    SPEX_matrix A2 = NULL;      // mpz version
    SPEX_matrix A2_init = NULL; // int64_t version
    SPEX_matrix b2 = NULL;      // mpz version
    SPEX_matrix b2_init = NULL; // int64_t version

    // Solution vectors
    SPEX_matrix x1 = NULL;
    SPEX_matrix x2 = NULL;

    // Allocate memory for initial matrices

    SPEX_matrix_allocate(&A1_init, SPEX_TRIPLET, SPEX_INT64, m1, n1, m1*n1, false, false, option);
    SPEX_matrix_allocate(&b1_init, SPEX_DENSE, SPEX_INT64, m1, 1, m1, false, false, option);

    SPEX_matrix_allocate(&A2_init, SPEX_TRIPLET, SPEX_INT64, m2, n2, m2*n2, false, false, option);
    SPEX_matrix_allocate(&b2_init, SPEX_DENSE, SPEX_INT64, m2, 1, m2, false, false, option);

    // Populate the matrices
    int count = 0;
    for (int i = 0; i < m1; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            A1_init->i[count] = i;
            A1_init->j[count] = j;
            A1_init->x.int64[count] = (rand() % 1000) + 1;
            count++;
        }
        b1_init->x.int64[i] = (rand() % 1000) + 1;
    }

    A1_init->nz = m1*n1;

    count = 0;
    for (int i = 0; i < m2; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            A2_init->i[count] = i;
            A2_init->j[count] = j;
            A2_init->x.int64[count] = (rand() % 1000) + 1;
            count++;
        }
        b2_init->x.int64[i] = (rand() % 1000) + 1;
    }

    A2_init->nz = m2*n2;

    // Copy initial matrices into A1, A2, b1, b2
    SPEX_matrix_copy(&A1, SPEX_CSC, SPEX_MPZ, A1_init, option);
    SPEX_matrix_copy(&A2, SPEX_CSC, SPEX_MPZ, A2_init, option);
    SPEX_matrix_copy(&b1, SPEX_DENSE, SPEX_MPZ, b1_init, option);
    SPEX_matrix_copy(&b2, SPEX_DENSE, SPEX_MPZ, b2_init, option);

    // Solve Ax = b
    SPEX_qr_backslash(&x1, SPEX_FP64, A1, b1, option);
    SPEX_qr_backslash(&x2, SPEX_FP64, A2, b2, option);

    //--------------------------------------------------------------------------
    // Free Memory
    //--------------------------------------------------------------------------
    FREE_WORKSPACE;
}
