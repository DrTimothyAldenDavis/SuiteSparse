//------------------------------------------------------------------------------
// Demo/spex_demo_qr_simple.c: example of simple call of SPEX_QR
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* This example shows how to use SPEX QR with a given input matrix and a
 * double output. The input is read from a file */

// usage:
// example > out
// out is file for output calculated result

#include "spex_demos.h"

#define FREE_WORKSPACE                 \
    {                                  \
        SPEX_matrix_free(&A, option);  \
        SPEX_matrix_free(&b, option);  \
        SPEX_matrix_free(&x, option);  \
        SPEX_matrix_free(&x2, option); \
        SPEX_FREE(option);             \
        SPEX_finalize();               \
    }

int main(int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // Prior to using SPEX Chol, its environment must be initialized. This is
    // done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------
    SPEX_initialize();

    //--------------------------------------------------------------------------
    // Get matrix file name
    //--------------------------------------------------------------------------

    char *mat_name;
    char *rhs_name; // this is actually ignored and we're using a rhs of 1s
    int64_t rat = 1;
    if (argc > 2)
    {
        mat_name = argv[1];
    }

    //--------------------------------------------------------------------------
    // Declare our data structures
    //--------------------------------------------------------------------------
    SPEX_info ok;
    SPEX_matrix A = NULL;  // input matrix with mpz values
    SPEX_matrix b = NULL;  // Right hand side vector
    SPEX_matrix x = NULL;  // Solution vectors
    SPEX_matrix x2 = NULL; // copy of solution vectors
    SPEX_options option = NULL;
    SPEX_TRY(SPEX_create_default_options(&option));
    if (option == NULL)
    {
        fprintf(stderr, "Error! OUT of MEMORY!\n");
        FREE_WORKSPACE;
        return 0;
    }

    // Process the command line
    SPEX_TRY(spex_demo_process_command_line(argc, argv, option,
                                            &mat_name, &rhs_name, &rat));

    //--------------------------------------------------------------------------
    // Allocate memory, read in A and b
    //--------------------------------------------------------------------------

    // Read in A. The output of this demo function is A in CSC format with
    // double entries.
    FILE *mat_file = fopen(mat_name, "r");
    if (mat_file == NULL)
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }

    SPEX_TRY(spex_demo_tripread(&A, mat_file, SPEX_FP64, option));
    fclose(mat_file);

    int64_t n = A->n;
    SPEX_matrix_allocate(&b, SPEX_DENSE, SPEX_MPZ, n, 1, n, false, true,
                         option);

    // Read in b. The output of this demo function is b in dense format with
    // mpz_t entries
    FILE *rhs_file = fopen(rhs_name, "r");
    if (rhs_file == NULL)
    {
        perror("Error while opening the file");
        FREE_WORKSPACE;
        return 0;
    }
    SPEX_TRY(spex_demo_read_dense(&b, rhs_file, option));
    fclose(rhs_file);

    // Check if the size of A matches b
    if (A->n != b->m)
    {
        printf("%" PRId64 " %" PRId64 " \n", A->m, b->m);
        fprintf(stderr, "Error! Size of A and b do not match!\n");
        FREE_WORKSPACE;
        return 0;
    }
    //--------------------------------------------------------------------------
    // solve
    //--------------------------------------------------------------------------
    double start_s = SUITESPARSE_TIME;
    option->algo = SPEX_QR_GS;

    SPEX_TRY(SPEX_qr_backslash(&x, SPEX_MPQ, A, b, option));

    double end_s = SUITESPARSE_TIME;

    double t_s = (end_s - start_s);

    printf("\nSPEX QR backslash time: %lf\n", t_s);

    // check solution
    option->print_level = 1;
    SPEX_TRY(spex_demo_check_solution(A, x, b, option));

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------
    FREE_WORKSPACE;

    printf("\n%s: all tests passed\n\n", __FILE__);

    return 0;
}
