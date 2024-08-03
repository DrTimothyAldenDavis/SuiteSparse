////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_gaxpy /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  computing y = alpha * A*x+y
 *
 *  @author Aznaveh
 */

#include "paru_internal.hpp"

void paru_gaxpy
(
    cholmod_sparse *A,
    const double *x,
    double *y,
    double alpha
)
{
    DEBUGLEVEL(0);

    int64_t *Ap = static_cast<int64_t*>(A->p);
    int64_t *Ai = static_cast<int64_t*>(A->i);
    double *Ax = static_cast<double*>(A->x);
    int64_t n = A->ncol;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t p = Ap[j]; p < Ap[j + 1]; p++)
        {
            y[Ai[p]] += alpha * Ax[p] * x[j];
        }
    }

#ifndef NDEBUG
    int64_t m = A->nrow;
    PRLEVEL(1, ("%% after gaxpy y is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.8lf, ", y[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
}

