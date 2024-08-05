////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ParU_Perm.cpp ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief Computing and saving row permutation. This must be done after
 * factorization.
 *
 *   I have this transition   A ---> S ---> LU
 *   There are both col and row permutation form A to S.
 *   However there is no column permuation from S to LU. Therefore the overall
 *   column permutaion is the same with S. (Qfill)
 *   Row permutation happens from S to LU.
 *   Row permutation and inverse permutation is computed here
 *
 *                    ------P--->
 *                    A         LU
 *                **********
 *                **********     The rest is identity
 *                ***$$$$$$$    $$$$$$$
 *                ***$$$$$$$    $$$$$$$
 *                    <----q----
 *
 *                     Pfin (computed here)
 *               ------------------>
 *                 Pinit     Ps = (compute here) newRofS (paru_write)
 *               --------> -------->
 *              A         S           LU
 *               <-------   <-------
 *          (paru_analyze)Pinv     oldRofS (paru_write)
 *
 *   We need these permutations for compuing Ax = b:
 *        x = b (P)     via ParU_Perm
 *        x = L\x
 *        x = U\x
 *        b(q) = x      via ParU_InvPerm
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Perm: vector variant
//------------------------------------------------------------------------------

// apply perm and scale x = b(P) / s

ParU_Info ParU_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size n
    const double *s,    // vector of size n (optional)
    const double *b,    // vector of size n
    int64_t n,          // length of P, s, B, and X
    // output
    double *x,          // vector of size n
    // control
    ParU_Control Control
)
{
    if (!x || !b || !P)
    {
        return (PARU_INVALID) ;
    }

    DEBUGLEVEL(0);
#ifndef NDEBUG
    PRLEVEL(1, ("%% Inside apply permutaion and scale P is:\n%%"));
    for (int64_t k = 0; k < n; k++)
    {
        PRLEVEL(1, (" " LD ", ", P[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and b is:\n%%"));
    for (int64_t k = 0; k < n; k++)
    {
        PRLEVEL(1, (" %.2lf, ", b[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and s is\n%%"));

    for (int64_t k = 0; k < n; k++)
    {
        PRLEVEL(1, (" %lf, ", s[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

    if (s == NULL)
    {
        // no scaling
        for (int64_t k = 0; k < n; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            #ifndef NDEBUG
            PRLEVEL(1, ("b[" LD "]= %lf ", j, b[j]));
            #endif
            x[k] = b[j] ;
        }
    }
    else
    {
        // with scaling
        for (int64_t k = 0; k < n; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            #ifndef NDEBUG
            PRLEVEL(1, ("b[" LD "]= %lf ", j, b[j]));
            PRLEVEL(1, ("s[" LD "]=%lf, ", j, s[j]));
            #endif
            x[k] = b[j] / s[j] ;
        }
    }

#ifndef NDEBUG
    PRLEVEL(1, ("%% after applying permutaion x is:\n%%"));
    for (int64_t k = 0; k < n; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_Perm: matrix variant
//------------------------------------------------------------------------------

// apply perm and scale X = B(P,:) / s

ParU_Info ParU_Perm
(
    // inputs
    const int64_t *P,   // permutation vector of size nrows
    const double *s,    // vector of size nrows (optional)
    const double *B,    // array of size nrows-by-ncols
    int64_t nrows,      // # of rows of X and B
    int64_t ncols,      // # of columns of X and B
    // output
    double *X,          // array of size nrows-by-ncols
    // control
    ParU_Control Control
)
{

    if (!X || !B || !P)
    {
        return (PARU_INVALID) ;
    }
    DEBUGLEVEL(0);
#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS Inside apply Permutaion and scale P is:\n%%"));
    for (int64_t k = 0; k < nrows; k++)
    {
        PRLEVEL(1, (" " LD ", ", P[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and B is:\n"));
    for (int64_t k = 0; k < nrows; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < ncols; l++)
        {
            PRLEVEL(1, (" %.2lf, ", B[l * nrows + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and s is\n%%"));

    for (int64_t k = 0; k < nrows; k++)
    {
        PRLEVEL(1, (" %lf, ", s[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    if (s == NULL)
    {
        // no scaling
        for (int64_t k = 0; k < nrows; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            for (int64_t l = 0; l < ncols; l++)
            {
                // X(k,l) = B(j,l)
                X[k + l*nrows] = B[j + l*nrows] ;
            }
        }
    }
    else
    {
        // with scaling
        for (int64_t k = 0; k < nrows; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            for (int64_t l = 0; l < ncols; l++)
            {
                // X(k,l) = B(j,l) / s(j)
                X[k + l*nrows] = B[j + l*nrows] / s[j];
            }
        }
    }

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("%% mRHS ParU_Perm %lf seconds\n", time));
#endif

#ifndef NDEBUG
    PRLEVEL(1, ("\n%% after applying permutation X is:\n"));
    for (int64_t k = 0; k < nrows; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < ncols; l++)
        {
            PRLEVEL(1, (" %.2lf, ", X[l * nrows + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (PARU_SUCCESS) ;
}

