//  =========================================================================  /
// =======================  paru_simple.cpp  ================================  /
// ==========================================================================  /

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*
 * @brief   a simple test to show how to use ParU in C++
 * @author Aznaveh
 * */
#include <iostream>
#include <iomanip>
#include <ios>
#include <cmath>

#include "ParU.hpp"
int main(int argc, char **argv)
{
    cholmod_common Common, *cc;
    cholmod_sparse *A;
    ParU_Symbolic *Sym;
    //~~~~~~~~~Reading the input matrix and test if the format is OK~~~~~~~~~~~~
    // start CHOLMOD
    cc = &Common;
    int mtype;
    cholmod_l_start(cc);
    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *)cholmod_l_read_matrix(stdin, 1, &mtype, cc);
    //~~~~~~~~~~~~~~~~~~~Starting computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    std::cout << "================= ParU, a simple demo: ========================\n";
    ParU_Control Control;
    ParU_Ret info;
    info = ParU_Analyze(A, &Sym, &Control);
    std::cout << "Input matrix is " << Sym->m << "x" << Sym->n
        << " nnz = " << Sym->anz << std::endl;
    ParU_Numeric *Num;
    info = ParU_Factorize(A, Sym, &Num, &Control);

    if (info != PARU_SUCCESS)
    {
        std::cout << "ParU: factorization was NOT successful.\n";
        if (info == PARU_OUT_OF_MEMORY)
            std::cout << "Out of memory\n";
        if (info == PARU_INVALID)
            std::cout << "Invalid!\n";
        if (info == PARU_SINGULAR)
            std::cout << "Singular!\n";
    }
    else
    {
        std::cout << "ParU: factorization was successful." << std::endl;
    }

    //~~~~~~~~~~~~~~~~~~~ Computing Ax = b ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if 1
    if (info == PARU_SUCCESS)
    {
        int64_t m = Sym->m;
        double *b = (double *)malloc(m * sizeof(double));
        double *xx = (double *)malloc(m * sizeof(double));
        for (int64_t i = 0; i < m; ++i) b[i] = i + 1;
        info = ParU_Solve(Sym, Num, b, xx, &Control);
        double resid, anorm, xnorm;
        info = ParU_Residual(A, xx, b, m, resid, anorm, xnorm, &Control);
        double rresid = (anorm == 0 || xnorm == 0 ) ? 0 : (resid/(anorm*xnorm));
        std::cout << std::scientific << std::setprecision(2)
            << "Relative residual is |" << rresid << "| anorm is " << anorm
            << ", xnorm is " << xnorm << " and rcond is " << Num->rcond << "."
            << std::endl;
        free(b);
        free(xx);
    }
#endif  // testing the results
    //~~~~~~~~~~~~~~~~~~~End computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ParU_Freenum(&Num, &Control);
    ParU_Freesym(&Sym, &Control);

    cholmod_l_free_sparse(&A, cc);
    cholmod_l_finish(cc);
}
