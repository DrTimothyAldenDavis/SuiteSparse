////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// paru_perm ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief Computing and saving row permutation. This must be doen after
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
 *                     Pfin (COMPUTED HERE)
 *               ------------------>
 *                 Pinit     Ps = (compute here) newRofS (paru_write)
 *               --------> -------->
 *              A         S           LU
 *               <-------   <-------
 *          (paru_analyze)Pinv     oldRofS (paru_write)
 *
 *
 *   We need these permuatations for compuing Ax = b
 *        x = b (P)
 *        x = L\x
 *        x = U\x
 *        b(q) = x
 *

 * @author Aznaveh
 * */
#include "paru_internal.hpp"

///////////////apply inverse perm x(p) = b, or with scaling: x(p)=b ; x=x./s
int64_t paru_apply_inv_perm(const int64_t *P, const double *s, const double *b, double *x, int64_t m)
{
    DEBUGLEVEL(0);
    if (!x || !b) return (0);
#ifndef NDEBUG
    PRLEVEL(1, ("%% Inside apply inv permutaion P is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" " LD ", ", P[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% before applying inverse permutaion b is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", b[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

    // x(p) = b ;
    for (int64_t k = 0; k < m; k++)
    {
        int64_t j = P[k];  // k-new and j-old; P(new) = old
        x[j] = b[k] ;
    }

    if (s != NULL)
    {
        // x = x ./ s
        for (int64_t j = 0; j < m; j++)
        {
            x[j] /= s [j] ;
        }
    }

#ifndef NDEBUG
    PRLEVEL(1, ("%% after applying inverse permutaion x is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.8lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (1);
}

///////////////apply inverse perm X(p,:) = B or with scaling: X(p,:)=B ; X = X./s
int64_t paru_apply_inv_perm(const int64_t *P, const double *s, const double *B, double *X, int64_t m, int64_t n)
{
    DEBUGLEVEL(0);
    if (!X || !B) return (0);
    PARU_DEFINE_PRLEVEL;
#ifndef NDEBUG
    PRLEVEL(PR, ("%% mRHS Inside apply inv permutaion P is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, (" " LD ", ", P[k]));
    }
    PRLEVEL(PR, (" \n"));

    PR = 1;
    PRLEVEL(PR, ("%% mRHS before applying inverse permutaion B is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, ("%%"));
        for (int64_t l = 0; l < n; l++)
        {
            PRLEVEL(PR, (" %.2lf, ", B[l * m + k]));
        }
        PRLEVEL(PR, (" \n"));
    }
    PRLEVEL(PR, (" \n"));
#endif

#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    // X(p,:) = B
    for (int64_t k = 0; k < m; k++)
    {
        int64_t j = P[k];  // k-new and j-old; P(new) = old
        for (int64_t l = 0; l < n; l++)
        {
            X[l * m + j] = B[l * m + k];  // Pinv(old) = new
        }
    }

    if (s != NULL)
    {
        // X = X ./ s
        for (int64_t j = 0; j < m; j++)
        {
            for (int64_t l = 0; l < n; l++)
            {
                X[l * m + j] /= s [j] ;
            }
        }
    }

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("%% mRHS paru_apply_inv_perm %lf seconds\n", time));
#endif
#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS after applying inverse permutaion X is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < n; l++)
        {
            PRLEVEL(1, (" %.2lf, ", X[l * m + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (1);
}

///////////////apply perm and scale x = sb(P) //////////////////////////////////
int64_t paru_apply_perm_scale(const int64_t *P, const double *s, const double *b,
                          double *x, int64_t m)
{
    DEBUGLEVEL(0);
    if (!x || !b) return (0);

#ifndef NDEBUG
    PRLEVEL(1, ("%% Inside apply permutaion and scale P is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" " LD ", ", P[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and b is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", b[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and s is\n%%"));

    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %lf, ", s[k]));
    }
    PRLEVEL(1, (" \n"));
#endif

    if (s == NULL)
    {
        // no scaling
        for (int64_t k = 0; k < m; k++)
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
        for (int64_t k = 0; k < m; k++)
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
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" %.2lf, ", x[k]));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (1);
}

///////////////apply perm and scale X = sB(P) /////////several mRHS ///////////
int64_t paru_apply_perm_scale(const int64_t *P, const double *s, const double *B,
                          double *X, int64_t m, int64_t n)
{
    DEBUGLEVEL(0);
    if (!X || !B) return (0);

#ifndef NDEBUG
    PRLEVEL(1, ("%% mRHS Inside apply Permutaion and scale P is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, (" " LD ", ", P[k]));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and B is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < n; l++)
        {
            PRLEVEL(1, (" %.2lf, ", B[l * m + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));

    PRLEVEL(1, ("%% and s is\n%%"));

    for (int64_t k = 0; k < m; k++)
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
        for (int64_t k = 0; k < m; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            for (int64_t l = 0; l < n; l++)
            {
                X[l * m + k] = B[l * m + j] ;
            }
        }
    }
    else
    {
        // with scaling
        for (int64_t k = 0; k < m; k++)
        {
            int64_t j = P[k];  // k-new and j-old; P(new) = old
            for (int64_t l = 0; l < n; l++)
            {
                X[l * m + k] = B[l * m + j] / s[j];
            }
        }
    }

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("%% mRHS paru_apply_perm_scale %lf seconds\n", time));
#endif

#ifndef NDEBUG
    PRLEVEL(1, ("\n%% after applying permutaion X is:\n"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(1, ("%%"));
        for (int64_t l = 0; l < n; l++)
        {
            PRLEVEL(1, (" %.2lf, ", X[l * m + k]));
        }
        PRLEVEL(1, (" \n"));
    }
    PRLEVEL(1, (" \n"));
#endif
    return (1);
}

