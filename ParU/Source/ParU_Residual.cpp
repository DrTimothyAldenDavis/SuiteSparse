////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Residual //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief    get a factorized matrix A and a vector b
 *
 *          compute ||Ax -b||
 *
 *          resid = norm1(b-A*x)
 *          anorm = norm1(A)
 *          xnorm = norm1(x)
 *
 * @author Aznaveh
 * */

#include <algorithm>

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Residual: compute residual b-A*x for vectors b and x
//------------------------------------------------------------------------------

ParU_Info ParU_Residual
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *x,          // vector of size n
    double *b,          // vector of size n
    // output:
    double &resid,      // residual: norm1(b-A*x) / (norm1(A) * norm1 (x))
    double &anorm,      // 1-norm of A
    double &xnorm,      // 1-norm of x
    // control:
    ParU_Control Control
)
{
    if (!A || !x || !b ||
        A->xtype != CHOLMOD_REAL || A->dtype != CHOLMOD_DOUBLE)
    {
        return (PARU_INVALID) ;
    }

    // get Control
    int32_t nthreads = paru_nthreads (Control) ;
    size_t mem_chunk = PARU_DEFAULT_MEM_CHUNK ;
    if (Control != NULL)
    {
        mem_chunk = Control->mem_chunk ;
    }

    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% inside residual\n"));
    int64_t m = A->nrow ;

#ifndef NDEBUG
    int64_t PR = 1;
    PRLEVEL(PR, ("%% before everything b is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, (" %.2lf, ", b[k]));
    }
    PRLEVEL(PR, (" \n"));
#endif
    double *ax_b = PARU_MALLOC (m, double);
    if (ax_b == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside residual\n"));
        return PARU_OUT_OF_MEMORY;
    }
    paru_memcpy(ax_b, b, m * sizeof(double), mem_chunk, nthreads) ;

#ifndef NDEBUG
    PRLEVEL(1, ("%% after copying x is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", ax_b[k]));
    }
    PRLEVEL(1, (" \n"));
#endif


    PRLEVEL(1, ("%% gaxpy\n"));
    paru_gaxpy(A, x, ax_b, -1);
    anorm = paru_spm_1norm(A) ;
    PRLEVEL(1, ("%% resid=%lf\n", resid));
    //resid =  paru_vec_1norm(ax_b, m)/ (anorm* paru_vec_1norm(b, m));
    resid = paru_vec_1norm(ax_b, m) ;
    xnorm = paru_vec_1norm (x, m) ;
    PARU_FREE(m, double, ax_b);
    return PARU_SUCCESS;
}

//------------------------------------------------------------------------------
// ParU_Residual: compute residual B-A*X for matrices B and X
//------------------------------------------------------------------------------

/*! @brief  get a factorized matrix A and a  multiple right hand side matrix B
 *
 *          compute ||Ax -B||
 * resid = norm1(b-A*x) / norm1(A)
 *
 * */

ParU_Info ParU_Residual
(
    // inputs:
    cholmod_sparse *A,  // an n-by-n sparse matrix
    double *X,          // array of size n-by-nrhs
    double *B,          // array of size n-by-nrhs
    int64_t nrhs,
    // output:
    double &resid,      // residual: norm1(B-A*X) / (norm1(A) * norm1 (X))
    double &anorm,      // 1-norm of A
    double &xnorm,      // 1-norm of X
    // control:
    ParU_Control Control
)
{
    if (!A || !X || !B ||
        A->xtype != CHOLMOD_REAL || A->dtype != CHOLMOD_DOUBLE)
    {
        return (PARU_INVALID) ;
    }

    // get Control
    int32_t nthreads = paru_nthreads (Control) ;
    size_t mem_chunk = PARU_DEFAULT_MEM_CHUNK ;
    if (Control != NULL)
    {
        mem_chunk = Control->mem_chunk ;
    }

    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% mRHS inside residual\n"));
    int64_t m = A->nrow ;

#ifndef NDEBUG
    int64_t PR = 1;
    PRLEVEL(PR, ("%% mRHS before everything B is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < nrhs; l++)
        {
            PRLEVEL(1, (" %.2lf, ", B[l * m + k]));
            // PRLEVEL(1, (" %.2lf, ", B[k*n+l])); B row-major
        }
        PRLEVEL(1, (" \n"));
    }

#endif
    double *AX_B = PARU_MALLOC (m * nrhs, double);
    if (AX_B == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside mRHS residual\n"));
        return PARU_OUT_OF_MEMORY;
    }
    paru_memcpy(AX_B, B, m * nrhs * sizeof(double), mem_chunk, nthreads) ;

#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS after copying X is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < nrhs; l++)
        {
            PRLEVEL(1, (" %.2lf, ", X[l * m + k]));
            // PRLEVEL(1, (" %.2lf, ", X[k*n+l])); X row major
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));
#endif
    anorm = paru_spm_1norm(A) ;
    xnorm = paru_matrix_1norm (X, m, nrhs) ;
    resid = 0;
    for (int64_t l = 0; l < nrhs; l++)
    {
        PRLEVEL(1, ("%% gaxpy\n"));
        paru_gaxpy(A, X + m * l, AX_B + m * l, -1);
        double res = paru_vec_1norm(AX_B + m * l, m);
        PRLEVEL(1, ("%% res=%lf\n", res));
        resid = std::max(resid, res);
    }
    PARU_FREE(m*nrhs, double, AX_B);
    return PARU_SUCCESS;
}

