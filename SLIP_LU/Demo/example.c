//------------------------------------------------------------------------------
// SLIP_LU/Demo/example.c: example main program for SLIP_LU
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------


#include "demos.h"


/* This example shows how to use SLIP LU with a given input matrix and a double
   output. The input is a randomly generate dense matrix */

// usage:
// example > out
// out is file for output calculated result

#define FREE_WORKSPACE                       \
    SLIP_matrix_free(&A,option);             \
    SLIP_matrix_free(&x,option);             \
    SLIP_matrix_free(&b,option);             \
    SLIP_matrix_free(&Rb,option);            \
    SLIP_matrix_free(&R,option);             \
    SLIP_FREE(option);                       \
    SLIP_finalize() ;                        \

   
    
int main (void)
{

    //--------------------------------------------------------------------------
    // Prior to using SLIP LU, its environment must be initialized. This is done
    // by calling the SLIP_initialize() function.
    //--------------------------------------------------------------------------

    SLIP_initialize();

    //--------------------------------------------------------------------------
    // Declare and initialize essential variables
    //--------------------------------------------------------------------------

    SLIP_info ok;
    int64_t n = 50, nz = 2500, num=0;
    SLIP_matrix *A = NULL ;                     // input matrix
    SLIP_matrix *R = NULL ;                     // Random matrix to create A
    SLIP_matrix *Rb = NULL;                     // Random matrix to create b
    SLIP_matrix *b = NULL ;                     // Right hand side vector
    SLIP_matrix *x = NULL ;                     // Solution vectors
    SLIP_options *option = SLIP_create_default_options();
    if (!option)
    {
        fprintf (stderr, "Error! OUT of MEMORY!\n");
        FREE_WORKSPACE;
        return 0;
    }

    //--------------------------------------------------------------------------
    // Generate a random dense 50*50 matrix
    //--------------------------------------------------------------------------

    // R is a n*n triplet matrix whose entries are FP64 Note that the first
    // boolean parameter says that the matrix is not shallow, so that A->i,
    // A->j, and A->x are calloc'd. The second boolean parameter is meaningless
    // for FP64 matrices, but it tells SLIP LU to allocate the values of A->x
    // for the mpz_t, mpq_t, and mpfr_t entries
    SLIP_matrix_allocate(&R, SLIP_TRIPLET, SLIP_FP64, n, n, nz,
        false, true, option);
    
    // Rb is a n*1 dense matrix whose entries are FP64
    SLIP_matrix_allocate(&Rb, SLIP_DENSE, SLIP_FP64, n, 1, n,
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
    OK ( SLIP_matrix_copy(&A, SLIP_CSC, SLIP_MPZ, R, option));
    // b is a copy of the Rb matrix. b is dense with mpz_t entries. 
    OK ( SLIP_matrix_copy(&b, SLIP_DENSE, SLIP_MPZ, Rb, option));

    //--------------------------------------------------------------------------
    // Solve
    //--------------------------------------------------------------------------

    clock_t start_s = clock();
   
    // SLIP LU has an optional check, to enable it, one can set the following
    // parameter to be true.
    option->check = true;
    // Solve the system and give double solution
    OK(SLIP_backslash( &x, SLIP_FP64, A, b, option));
         
    clock_t end_s = clock();

    double t_s = (double) (end_s - start_s) / CLOCKS_PER_SEC;

    printf("\nSLIP LU Factor & Solve time: %lf\n", t_s);

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    FREE_WORKSPACE;
    printf ("\n%s: all tests passed\n\n", __FILE__) ;
    return 0;
}

