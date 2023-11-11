////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_norms /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  computing norms: 1-norm for vectors and sparse matrix
 *  and matrix
 *  @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"
double paru_spm_1norm(cholmod_sparse *A)
{
    // 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum
    // CSparse
    DEBUGLEVEL(0);
    if (!(A) || !A->x) return (-1);
    int64_t n = A->ncol;
    int64_t *Ap = static_cast<int64_t*>(A->p);
    double *Ax = static_cast<double*>(A->x);

    double norm = 0;
    for (int64_t j = 0; j < n; j++)
    {
        double s = 0;
        for (int64_t p = Ap[j]; p < Ap[j + 1]; p++)
        {
            PRLEVEL(3, ("Ax[" LD "] = %.2lf\n", p, Ax[p]));
            s += fabs(Ax[p]);
        }
        PRLEVEL(2, ("s = %le\n", s));
        norm = std::max(norm, s);
    }
    PRLEVEL(1, ("norm = %.8lf\n", norm));
    return (norm);
}

double paru_vec_1norm(const double *x, int64_t n)
{
    DEBUGLEVEL(0);
    double norm = 0.0;
    for (int64_t i = 0; i < n; i++)
    {
        PRLEVEL(1, ("so far norm = %lf + %lf\n", norm, fabs(x[i])));
        norm += fabs(x[i]);
    }
    PRLEVEL(1, ("vec 1norm = %.8lf\n", norm));
    return (norm);
}

// 1-norm of an m-by-n dense matrix
double paru_matrix_1norm(const double *x, int64_t m, int64_t n)
{
    DEBUGLEVEL(0);
    double norm = 0.0;
    for (int64_t j = 0 ; j < n ; j++)
    {
        double colnorm = paru_vec_1norm (x + j*m, m) ;
        norm = std::max(norm, colnorm);
    }
    PRLEVEL(1, ("matrix 1norm = %.8lf\n", norm));
    return (norm);
}
