////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_backward //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief     compute the backward error
 *
 *          get a factorized matrix A and a vector x1
 *          compute Ax1=b then solve for Ax2=b
 *          return ||x2-x1||
 *
 * @author Aznaveh
 * */

#include <algorithm>

#include "paru_cov.hpp"

ParU_Info paru_backward
(
    double *x1,
    double &resid,
    double &anorm,
    double &xnorm,
    cholmod_sparse *A,
    ParU_Symbolic Sym,
    ParU_Numeric Num,
    ParU_Control Control
)
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% inside backward\n"));
    if(!Sym || !Num || !x1 || !A)
    {
        return (PARU_INVALID) ;
    }

    int64_t m = Sym->m;
#ifndef NDEBUG
    int64_t PR = 1;
    PRLEVEL(PR, ("%% before everything x1 is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, (" %.2lf, ", x1[k]));
    }
    PRLEVEL(PR, (" \n"));
#endif
    double *b = PARU_CALLOC (m, double);
    if (b == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside backward\n"));
        return PARU_OUT_OF_MEMORY;
    }
    paru_gaxpy(A, x1, b, 1);
#ifndef NDEBUG
    PRLEVEL(1, ("%% after gaxpy b is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", b[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

    ParU_Info info;
    info = ParU_Solve(Sym, Num, b, Control);
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% A problem happend during factorization\n"));
        PARU_FREE(m, double, b);
        return info;
    }

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("x2 = [ "));
    int64_t mm = std::min (m, (int64_t) 10) ;
    for (int64_t i = 0; i < mm ; ++i)
    {
        PRLEVEL(PR, ("%lf ", b[i]));
    }
    PRLEVEL(PR, (" ...]\n"));
#endif

    for (int64_t k = 0; k < m; k++) b[k] -= x1[k];

    resid = paru_vec_1norm(b, m);
    PRLEVEL(1, ("%% resid =%lf\n", resid));
    anorm = paru_spm_1norm(A) ;
    xnorm = paru_vec_1norm (x1, m) ;
    PRLEVEL(1, ("backward error is |%.2lf| and weigheted backward error is"
       "|%.2f|.\n",resid == 0 ? 0 : log10(resid), resid == 0 ? 0 :log10(xnorm)));
    PARU_FREE(m, double, b);
    return (PARU_SUCCESS) ;
}
