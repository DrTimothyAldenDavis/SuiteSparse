////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Solve /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*!  @brief  solve Ax = b
 *      get a factorized matrix and a right hand side
 *      returns x; overwrites it on b
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Solve: x = A\x
//------------------------------------------------------------------------------

ParU_Info ParU_Solve
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    ParU_Numeric Num,       // numeric factorization from ParU_Factorize
    // input/output:
    double *x,              // vector of size n-by-1; right-hand on input,
                            // solution on output
    // control:
    ParU_Control Control
)
{
    return (ParU_Solve (Sym, Num, x, x, Control)) ;
}

//------------------------------------------------------------------------------
// ParU_Solve: x = A\b
//------------------------------------------------------------------------------

ParU_Info ParU_Solve
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    ParU_Numeric Num,       // numeric factorization from ParU_Factorize
    double *b,              // vector of size n-by-1
    // output
    double *x,              // vector of size n-by-1
    // control:
    ParU_Control Control
)
{

    if (!Sym || !Num || !b || !x)
    {
        return PARU_INVALID;
    }

    // note that the x and b parameters can be aliased

    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% inside solve\n"));

    int64_t m = Sym->m;
    // if (Num->res == PARU_SINGULAR)  //THIS Won't happen because Num is freed
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    double *t = PARU_MALLOC (m, double);
    if (t == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside solve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    // t = scaled and permuted version of b
    const int64_t *P = Num->Pfin ;
    const double *R = Num->Rs ;
    ParU_Perm (P, R, b, m, t, Control);

    ParU_Info info;
    PRLEVEL(1, ("%% lsolve\n"));
    info = ParU_LSolve(Sym, Num, t, Control);  // t = L\t
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in lsolve\n"));
        PARU_FREE(m, int64_t, t);
        return info;
    }
    PRLEVEL(1, ("%% usolve\n"));
    info = ParU_USolve(Sym, Num, t, Control);  // t = U\t
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in usolve\n"));
        PARU_FREE(m, int64_t, t);
        return info;
    }

    const int64_t *Q = Sym->Qfill ;
    ParU_InvPerm (Q, NULL, t, m, x, Control);  // x(q) = t

    PARU_FREE(m, int64_t, t);
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

//------------------------------------------------------------------------------
// ParU_Solve: X = A\X
//------------------------------------------------------------------------------

/*!  @brief  solve AX = B
 *      get a factorized matrix and several right hand sides
 *      X holds right-hand-side on input, and the solution on output.
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

ParU_Info ParU_Solve
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    ParU_Numeric Num,       // numeric factorization from ParU_Factorize
    int64_t nrhs,           // # of right-hand sides
    // input/output:
    double *X,              // X is n-by-nrhs, where A is n-by-n;
                            // holds B on input, solution X on input
    // control:
    ParU_Control Control
)
{
    return (ParU_Solve (Sym, Num, nrhs, X, X, Control)) ;
}

//------------------------------------------------------------------------------
// ParU_Solve: X = A\B
//------------------------------------------------------------------------------

ParU_Info ParU_Solve
(
    // input
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    ParU_Numeric Num,       // numeric factorization from ParU_Factorize
    int64_t nrhs,           // # of right-hand sides
    double *B,              // n-by-nrhs, in column-major storage
    // output:
    double *X,              // n-by-nrhs, in column-major storage
    // control:
    ParU_Control Control
)
{

    if (!Sym || !Num || !B || !X)
    {
        return PARU_INVALID;
    }

    // Note: B and X can be aliased
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% mRHS inside Solve\n"));
    int64_t m = Sym->m;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    double *T = PARU_MALLOC (m * nrhs, double);
    if (T == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside Solve\n"));
        return PARU_OUT_OF_MEMORY;
    }

    // T = permuted and scaled version of B
    const int64_t *P = Num->Pfin ;
    const double *R = Num->Rs ;
    ParU_Perm (P, R, B, m, nrhs, T, Control);

    // T = L\T
    ParU_Info info;
    PRLEVEL(1, ("%%mRHS lsolve\n"));
    info = ParU_LSolve(Sym, Num, nrhs, T, Control);
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in mRHS lsolve\n"));
        PARU_FREE(m * nrhs, int64_t, T);
        return info;
    }

    // T = U\T
    PRLEVEL(1, ("%%mRHS usolve\n"));
    info = ParU_USolve(Sym, Num, nrhs, T, Control);
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% Problems in mRHS usolve\n"));
        PARU_FREE(m * nrhs, int64_t, T);
        return info;
    }

    const int64_t *Q = Sym->Qfill ;
    ParU_InvPerm (Q, NULL, T, m, nrhs, X, Control);  // X(q) = T

    // to solve A'x=b instead (future work):
    // permute t = b (p):
    // ParU_Perm (Sym->Qfill, NULL, B, m, nrhs, T, Control);
    // solve T = U'\T
    // solve T = L'\T
    // x (q) = t and then x = x/s:
    // ParU_InvPerm (Num->Pfin, Num->Rs, T, m, nrhs, X, Control);

    PARU_FREE(m * nrhs, int64_t, T);

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(-1, ("%% mRHS solve has been finished in %lf seconds\n", time));
#endif
    return PARU_SUCCESS;
}

