//  =========================================================================  /
// =======================  paru_simple.cpp  ================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * @brief   a simple test to show how to use ParU in C++
 * @author Aznaveh
 * */
#include <iostream>
#include <iomanip>
#include <ios>
#include <cmath>

#include "ParU.h"

#define FREE_ALL_AND_RETURN(info)               \
{                                               \
    if (b != NULL) free(b);                     \
    if (x != NULL) free(x);                     \
    ParU_FreeNumeric(&Num, Control);            \
    ParU_FreeSymbolic(&Sym, Control);           \
    ParU_FreeControl(&Control);                 \
    cholmod_l_free_sparse(&A, cc);              \
    cholmod_l_finish(cc);                       \
    return (info) ;                             \
}

#define OK(method,what)                         \
{                                               \
    info = (method) ;                           \
    if (info != PARU_SUCCESS)                   \
    {                                           \
        std::cout << what << " failed\n" ;      \
        FREE_ALL_AND_RETURN (info) ;            \
    }                                           \
}

int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A = NULL ;
    ParU_Symbolic Sym = NULL ;
    ParU_Numeric Num = NULL ;
    ParU_Control Control = NULL ;
    double *b = NULL, *x = NULL ;

    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    if (A == NULL)
    {
        std::cout << "ParU: invalid input matrix\n" ;
        FREE_ALL_AND_RETURN (PARU_INVALID) ;
    }

    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    std::cout << "================= ParU, a simple demo: ===================\n";
    ParU_Info info;
    OK (ParU_Analyze(A, &Sym, Control), "symbolic analysis");
    int64_t n, anz ;
    OK (ParU_Get (Sym, Num, PARU_GET_N, &n, Control), "n") ;
    OK (ParU_Get (Sym, Num, PARU_GET_ANZ, &anz, Control), "anz") ;
    std::cout << "Input matrix is " << n << "x" << n <<
        " nnz = " << anz << std::endl;
    OK (ParU_Factorize(A, Sym, &Num, Control), "numeric factorization") ;
    int64_t unz, lnz ;
    OK (ParU_Get (Sym, Num, PARU_GET_LNZ_BOUND, &lnz, Control), "lnz") ;
    OK (ParU_Get (Sym, Num, PARU_GET_UNZ_BOUND, &unz, Control), "unz") ;
    std::cout << "ParU: factorization was successful." << std::endl;
    std::cout << "nnz(L) = " << lnz << ", nnz(U) = " << unz << std::endl;

    //~~~~~~~~~~~~~~~~~~~ Computing the residual, norm(b-Ax) ~~~~~~~~~~~~~~~~~~~
    b = (double *)malloc(n * sizeof(double));
    x = (double *)malloc(n * sizeof(double));
    for (int64_t i = 0; i < n; ++i) b[i] = i + 1;
    OK (ParU_Solve(Sym, Num, b, x, Control), "solve") ;
    double resid, anorm, xnorm, rcond ;
    OK (ParU_Residual(A, x, b, resid, anorm, xnorm, Control), "residual");
    OK (ParU_Get (Sym, Num, PARU_GET_RCOND_ESTIMATE, &rcond, Control), 
        "rcond") ;
    double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
    std::cout << std::scientific << std::setprecision(2)
        << "Relative residual is |" << rresid << "| anorm is " << anorm
        << ", xnorm is " << xnorm << " and rcond is " << rcond << "."
        << std::endl;

    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FREE_ALL_AND_RETURN (PARU_SUCCESS) ;
}
