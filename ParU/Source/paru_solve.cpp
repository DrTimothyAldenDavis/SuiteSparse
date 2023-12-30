////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Solve /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*!  @brief  solve Ax = b
 *      get a factorized matrix and a right hand side
 *      returns x; overwrites it on b
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Solve: b = A\b
//------------------------------------------------------------------------------

ParU_Ret ParU_Solve(ParU_Symbolic *Sym, ParU_Numeric *Num, double *b,
                    ParU_Control *Control)
{
    return (ParU_Solve (Sym, Num, b, b, Control)) ;
}

//------------------------------------------------------------------------------
// ParU_Solve: x = A\b
//------------------------------------------------------------------------------

ParU_Ret ParU_Solve(ParU_Symbolic *Sym, ParU_Numeric *Num, double *b, double *x,
                    ParU_Control *Control)
{

    // note that the x and b parameters can be aliased

    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% inside solve\n"));
    if (Sym == NULL || Num == NULL)
    {
        return PARU_INVALID;
    }

    int64_t m = Sym->m;
    // if (Num->res == PARU_SINGULAR)  //THIS Won't happen because Num is freed
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    double *t = static_cast<double*>(paru_alloc(m, sizeof(double)));
    if (t == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside solve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    // t = scaled and permuted version of b
    // FIXME: make this user-callable
    paru_apply_perm_scale(Num->Pfin, Num->Rs, b, t, m);

    ParU_Ret info;
    PRLEVEL(1, ("%% lsolve\n"));
    info = ParU_Lsolve(Sym, Num, t, Control);  // t = L\t
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in lsolve\n"));
        paru_free(m, sizeof(int64_t), t);
        return info;
    }
    PRLEVEL(1, ("%% usolve\n"));
    info = ParU_Usolve(Sym, Num, t, Control);  // t = U\t
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in usolve\n"));
        paru_free(m, sizeof(int64_t), t);
        return info;
    }

    // FIXME: make this user-callable
    paru_apply_inv_perm(Sym->Qfill, NULL, t, x, m);  // x(q) = t

    paru_free(m, sizeof(int64_t), t);
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%%solve has been finished in %lf seconds\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%%after solve x is:\n%% ["));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, ("]; \n"));
#endif
    return PARU_SUCCESS;
}

//////////////////////////  ParU_Solve ////////////// mRHS /////////////////////
/*!  @brief  solve AX = B
 *      get a factorized matrix and several right hand sides
 *      returns X; overwrites it on B
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

ParU_Ret ParU_Solve(ParU_Symbolic *Sym, ParU_Numeric *Num, int64_t nrhs,
    double *B, ParU_Control *Control)
{
    return (ParU_Solve (Sym, Num, nrhs, B, B, Control)) ;
}

//------------------------------------------------------------------------------
// ParU_Solve: X = A\B
//------------------------------------------------------------------------------

ParU_Ret ParU_Solve(ParU_Symbolic *Sym, ParU_Numeric *Num, int64_t nrhs,
    double *B, double *X, ParU_Control *Control)
{
    // Note: B and X can be aliased
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% mRHS inside Solve\n"));
    if (Sym == NULL || Num == NULL)
    {
        return PARU_INVALID;
    }
    int64_t m = Sym->m;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    double *T = static_cast<double*>(paru_alloc(m * nrhs, sizeof(double)));
    if (T == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside Solve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    // T = permuted and scaled version of B
    // FIXME: make this user-callable
    paru_apply_perm_scale(Num->Pfin, Num->Rs, B, T, m, nrhs);

    // T = L\T
    ParU_Ret info;
    PRLEVEL(1, ("%%mRHS lsolve\n"));
    info = ParU_Lsolve(Sym, Num, T, nrhs, Control);
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in mRHS lsolve\n"));
        paru_free(m * nrhs, sizeof(int64_t), T);
        return info;
    }

    // T = U\T
    PRLEVEL(1, ("%%mRHS usolve\n"));
    info = ParU_Usolve(Sym, Num, T, nrhs, Control);
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in mRHS usolve\n"));
        paru_free(m * nrhs, sizeof(int64_t), T);
        return info;
    }

    // FIXME: make this user-callable
    paru_apply_inv_perm(Sym->Qfill, NULL, T, X, m, nrhs);  // X(q) = T

    // to solve A'x=b instead
    // permute t = b (p)
    // paru_apply_perm_scale(Sym->Qfill, NULL, B, T, m, nrhs);
    // T = U'\T
    // T = L'\T
    // x (q) = t and then x = x/s
    // paru_apply_inv_perm(Num->Pfin, Num->Rs, T, X, m, nrhs);

    paru_free(m * nrhs, sizeof(int64_t), T);

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% mRHS solve has been finished in %lf seconds\n", time));
#endif
    return PARU_SUCCESS;
}

