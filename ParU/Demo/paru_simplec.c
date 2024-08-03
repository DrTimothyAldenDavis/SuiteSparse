//  =========================================================================  /
// =======================  paru_simplec.c  =================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief   a simple test to show how to use ParU with C interface
 * @author Aznaveh
 * */

#include "ParU.h"

#define FREE_ALL_AND_RETURN(info)               \
{                                               \
    if (b != NULL) free(b);                     \
    if (x != NULL) free(x);                     \
    ParU_C_FreeNumeric(&Num, Control);          \
    ParU_C_FreeSymbolic(&Sym, Control);         \
    ParU_C_FreeControl(&Control);               \
    cholmod_l_free_sparse(&A, cc);              \
    cholmod_l_finish(cc);                       \
    return (info) ;                             \
}

#define OK(method,what)                         \
{                                               \
    ParU_Info info = (method) ;                 \
    if (info != PARU_SUCCESS)                   \
    {                                           \
        printf ("ParU: %s failed\n", what) ;    \
        FREE_ALL_AND_RETURN (info) ;            \
    }                                           \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc = NULL ;
    cholmod_sparse *A = NULL ;
    ParU_C_Symbolic Sym = NULL ;
    ParU_C_Numeric Num = NULL ;
    ParU_C_Control Control = NULL ;
    double *b = NULL, *x = NULL ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);

    // read in the sparse matrix A from stdin
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        printf ("unable to read matrix\n") ;
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    printf("================= ParU, a simple demo, using C interface : ====\n");
    OK (ParU_C_Analyze(A, &Sym, Control), "analysis") ;
    int64_t n, anz ;
    OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_N, &n, Control), "n") ;
    OK (ParU_C_Get_INT64 (Sym, Num, PARU_GET_ANZ, &anz, Control), "anz") ;
    printf("Input matrix is %" PRId64 "x%" PRId64 " nnz = %" PRId64 " \n",
        n, n, anz);
    OK (ParU_C_Factorize(A, Sym, &Num, Control), "factorization") ;
    printf("ParU: factorization was successful.\n");

    //~~~~~~~~~~~~~~~~~~~ Computing the residual, norm(b-Ax) ~~~~~~~~~~~~~~~~~~~
    b = (double *)malloc(n * sizeof(double));
    x = (double *)malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
    OK (ParU_C_Solve_Axb(Sym, Num, b, x, Control), "solve") ;

    double resid, anorm, xnorm;
    OK (ParU_C_Residual_bAx(A, x, b, &resid, &anorm, &xnorm, Control),
        "resid") ;
    double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    double rcond ;
    OK (ParU_C_Get_FP64 (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond, Control),
        "rcond") ;
    printf("Relative residual is |%.2e|, anorm is %.2e, xnorm is %.2e, "
        " and rcond is %.2e.\n", rresid, anorm, xnorm, rcond);

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FREE_ALL_AND_RETURN (PARU_SUCCESS) ;
}
