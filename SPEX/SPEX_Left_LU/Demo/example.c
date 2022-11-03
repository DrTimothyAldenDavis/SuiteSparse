//------------------------------------------------------------------------------
// SPEX_Left_LU/Demo/example.c: example main program for SPEX_Left_LU
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


#include "demos.h"


/* This example shows how to use SPEX Left LU with a given input matrix and a double
   output. The input is a randomly generate dense matrix */

// usage:
// example > out
// out is file for output calculated result

#define FREE_WORKSPACE                       \
    SPEX_matrix_free(&A,option);             \
    SPEX_matrix_free(&x,option);             \
    SPEX_matrix_free(&b,option);             \
    SPEX_matrix_free(&Rb,option);            \
    SPEX_matrix_free(&R,option);             \
    SPEX_FREE(option);                       \
    SPEX_finalize() ;                        \

   
    
int main (void)
{
    
    //--------------------------------------------------------------------------
    // Prior to using SPEX Left LU, its environment must be initialized. This is done
    // by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------

    SPEX_initialize();

    //--------------------------------------------------------------------------
    // Declare and initialize essential variables
    //--------------------------------------------------------------------------

    SPEX_info ok;
    int64_t n = 50, nz = 2500, num=0;
    SPEX_matrix *A = NULL ;                     // input matrix
    SPEX_matrix *R = NULL ;                     // Random matrix to create A
    SPEX_matrix *Rb = NULL;                     // Random matrix to create b
    SPEX_matrix *b = NULL ;                     // Right hand side vector
    SPEX_matrix *x = NULL ;                     // Solution vectors
    SPEX_options *option = NULL;
    OK(SPEX_create_default_options(&option));

    //--------------------------------------------------------------------------
    // Generate a random dense 50*50 matrix
    //--------------------------------------------------------------------------

    // R is a n*n triplet matrix whose entries are FP64 Note that the first
    // boolean parameter says that the matrix is not shallow, so that A->i,
    // A->j, and A->x are calloc'd. The second boolean parameter is meaningless
    // for FP64 matrices, but it tells SPEX Left LU to allocate the values of A->x
    // for the mpz_t, mpq_t, and mpfr_t entries
    SPEX_matrix_allocate(&R, SPEX_TRIPLET, SPEX_FP64, n, n, nz,
        false, true, option);
    
    // Rb is a n*1 dense matrix whose entries are FP64
    SPEX_matrix_allocate(&Rb, SPEX_DENSE, SPEX_FP64, n, 1, n,
        false, true, option);

    // Randomly generate the input
    unsigned int seed = 10;
    srand(seed);
    for (int64_t k = 0; k < n; k++)
    {
        Rb->x.fp64[k] = rand();
        for (int64_t p = 0; p < n; p++)
        {
            R->i[num] = k;
            R->j[num] = p;
            R->x.fp64[num] = rand();
            num+=1;
        }
    }

    R->nz = n*n;

    //--------------------------------------------------------------------------
    // Build A and b
    //--------------------------------------------------------------------------

    // A is a copy of the R matrix. A is a CSC matrix with mpz_t entries
    OK ( SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, R, option));
    // b is a copy of the Rb matrix. b is dense with mpz_t entries. 
    OK ( SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, Rb, option));

    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------

    clock_t start_s = clock();
   
    // SPEX Left LU has an optional check, to enable it, one can set the following
    // parameter to be true.
    option->check = true;
    // Solve the system and give double solution
    OK(SPEX_Left_LU_backslash( &x, SPEX_FP64, A, b, option));
         
    clock_t end_s = clock();

    double t_s = (double) (end_s - start_s) / CLOCKS_PER_SEC;

    printf("\nSPEX Left LU Factor & Solve time: %lf\n", t_s);

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__) ;
    return 0;
}

