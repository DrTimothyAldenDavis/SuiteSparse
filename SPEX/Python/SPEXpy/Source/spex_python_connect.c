//------------------------------------------------------------------------------
// SPEX/Python/spex_connect.c: use SPEX in Python
//------------------------------------------------------------------------------

// SPEX: (c) 2022-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_python_connect.h"
#include "spex_cholesky_internal.h"

#define FREE_WORKSPACE                  \
{                                       \
    SPEX_matrix_free(&A, option);       \
    SPEX_matrix_free(&b, option);       \
    SPEX_matrix_free(&x, option);       \
    SPEX_matrix_free(&A_in, option);    \
    SPEX_matrix_free(&b_in, option);    \
    SPEX_FREE(option);                  \
    SPEX_finalize();                    \
}


SPEX_info spex_python
(
     //output
     void **sol_void, // solution
     //input
     int64_t *Ap,     // column pointers of A, an array size is n+1
     int64_t *Ai,     // row indices of A, of size nzmax.
     double *Ax,      // values of A
     double *bx,      // values of b
     int m,           // Number of rows of A
     int n,           // Number of columns of A
     int nz,          // Number of nonzeros in A
     int ordering,    // type of ordering: 0-none, 1-colamd, 2-amd,
     int algorithm,   // 1-backslash, 2-left lu, 3-cholesky
     bool charOut     // True if char ** output, false if double
)
{
    // this function will make the SPEX_matrix A, b and x and call
    // spex_cholesky_backslash
    SPEX_info info;

    //--------------------------------------------------------------------------
    // Prior to using SPEX Chol, its environment must be initialized. This is
    // done by calling the SPEX_initialize() function.
    //--------------------------------------------------------------------------
    SPEX_initialize();

    //--------------------------------------------------------------------------
    // Validate Inputs
    //--------------------------------------------------------------------------

    if (!Ap || !Ai || !Ax || !bx)
    {
        return SPEX_INCORRECT_INPUT;
    }

    // ordering must be acceptable
    if (ordering != 0 && ordering != 1 && ordering != 2)
    {
        return SPEX_INCORRECT_INPUT;
    }

    if (n == 0 || m == 0 || n != m)
    {
        return SPEX_INCORRECT_INPUT;
    }

    //--------------------------------------------------------------------------
    // Declare our data structures
    //--------------------------------------------------------------------------
    SPEX_matrix A_in = NULL;       //input matrix
    SPEX_matrix b_in = NULL;       //input rhs
    SPEX_matrix A = NULL;          //copy of input matrix in CSC MPZ
    SPEX_matrix b = NULL;          //copy of input rhs in CSC MPZ
    SPEX_matrix x = NULL;          //solution

    SPEX_options option = NULL;
    SPEX_create_default_options(&option);
    SPEX_preorder order_in = ordering;
    option->order = ordering;

    //--------------------------------------------------------------------------
    // Allocate memory, populate in A and b
    //--------------------------------------------------------------------------

    SPEX_matrix_allocate(&A_in, SPEX_CSC, SPEX_FP64, n, n, nz, true, true,
        option);
    SPEX_matrix_allocate(&b_in, SPEX_DENSE, SPEX_FP64, n, 1, n, true, true,
        option);

    A_in->p=Ap;
    A_in->i=Ai;
    A_in->x.fp64=Ax;

    b_in->x.fp64=bx;

    // At this point, A_in is a shallow double CSC matrix. We make a copy of it
    // with A.  A is a CSC matrix with mpz entries

    SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, A_in, option);
    // b is a dense matrix with mpz entries
    SPEX_matrix_copy(&b, SPEX_DENSE, SPEX_MPZ, b_in, option);

    //--------------------------------------------------------------------------
    // solve Ax=b
    //-------------------------------------------------------------------------
    switch(algorithm)
    {
        case 1:
            SPEX_CHECK( SPEX_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 2:
            SPEX_CHECK( SPEX_lu_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 3:
            SPEX_CHECK( SPEX_cholesky_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        case 4:
            SPEX_CHECK( SPEX_ldl_backslash(&x, SPEX_MPQ, A, b, option));
            break;
        default:
            return SPEX_INCORRECT_INPUT;
    }

    //--------------------------------------------------------------------------
    // Return output as desired type
    //-----------------------------------------------------------------------
    if(charOut)
    {
        //solution as string
        for (int i = 0; i < n; ++i)
        {
            char *s ;
            int status = SPEX_mpfr_asprintf (&s, "%Qd", x->x.mpq [i]);
            if (status < 0)
            {
                printf("error converting x to string");
            }
            //check string size
            int sizeStr;
            sizeStr=strlen(s);
            //allocate sol_char[i]
             sol_void[i] = malloc (sizeStr + 1);  // +1 for NULL terminator
            //copy s into sol_char[i]
            strcpy(sol_void[i],s);
        }
    }
    else
    {
        //solution as double
        SPEX_matrix x2 = NULL ;
        SPEX_matrix_copy(&x2, SPEX_DENSE, SPEX_FP64, x, option);
        /*for (int i = 0; i < n; ++i)
        {
            sol_doub[i]=x2->x.fp64[i];
        }*/
        for (int i = 0; i < n; ++i)
        {
            (sol_void[i])=&(x2->x.fp64[i]);
        }
    }

    FREE_WORKSPACE;
    return SPEX_OK;
}
