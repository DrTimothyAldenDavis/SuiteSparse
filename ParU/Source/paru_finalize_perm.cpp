////////////////////////////////////////////////////////////////////////////////
//////////////////////// paru_finalize_perm ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

#include <algorithm>

#include "paru_internal.hpp"

ParU_Ret paru_finalize_perm(ParU_Symbolic *Sym, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    int64_t nf = Sym->nf;
    int64_t m = Sym->m;

    int64_t *Super = Sym->Super;

    // some working memory that is freed in this function
    int64_t *Pfin = NULL;
    int64_t *Ps = NULL;
    int64_t *Pinit = Sym->Pinit;

    Num->Pfin = Pfin = (int64_t *)paru_alloc(m, sizeof(int64_t));
    Num->Ps = Ps = (int64_t *)paru_alloc(m, sizeof(int64_t));

    PRLEVEL(1, ("%% Inside Perm\n"));
    if (Pfin == NULL || Ps == NULL)
    {
        PRLEVEL(1, ("ParU: memory problem inside perm\n"));
        return PARU_OUT_OF_MEMORY;
    }

#ifndef NDEBUG
    PRLEVEL(PR, ("%% Initial row permutaion is:\n%%"));
    for (int64_t k = 0; k < m; k++)
    {
        PRLEVEL(PR, (" " LD ", ", Pinit[k]));
    }
    PRLEVEL(PR, (" \n"));
#endif

    int64_t n1 = Sym->n1;  // row+col singletons
    int64_t ip = 0;        // number of rows seen so far
    PRLEVEL(PR, ("%% singlton part"));
    for (int64_t k = 0; k < n1; k++)
    {  // first singletons
        Pfin[ip++] = Pinit[k];
        PRLEVEL(PR, ("(" LD ")" LD " ", ip - 1, Pfin[ip - 1]));
    }
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("%% the rest\n"));
    for (int64_t f = 0; f < nf; f++)
    {  // rows for each front
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        int64_t *frowList = Num->frowList[f];

        // for each pivotal column:
        for (int64_t k = 0; k < fp; k++)
        {
            // P[k] = i
            Ps[frowList[k]] = ip - n1;
            Pfin[ip++] = Pinit[frowList[k] + n1];
            PRLEVEL(PR, ("(" LD ")" LD "\n ", ip - 1, Pfin[ip - 1]));
        }
    }
    PRLEVEL(PR, ("\n"));

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% Final Ps:\n%%"));
    for (int64_t k = 0; k < m - n1; k++)
    {
        PRLEVEL(PR, (" " LD ", ", Ps[k]));
    }
    PRLEVEL(PR, (" \n"));
    PR = 1;
    PRLEVEL(PR, ("%% n1=" LD " Final row permutaion is:\n%%", n1));
    for (int64_t k = 0; k < std::min(77, m); k++) PRLEVEL(PR, ("" LD " ", Pfin[k]));
    PRLEVEL(PR, (" \n"));
#endif
    return PARU_SUCCESS;
}

