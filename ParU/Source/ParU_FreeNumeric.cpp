////////////////////////////////////////////////////////////////////////////////
////////////////////////// ParU_FreeNumeric.cpp ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Free the Numeric factorization object.
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_FreeNumeric
//------------------------------------------------------------------------------

ParU_Info ParU_FreeNumeric
(
    // input/output:
    ParU_Numeric **Num_handle,  // numeric object to free
    // control:
    ParU_Control *Control
)
{
    if (Num_handle == NULL || *Num_handle == NULL)
    {
        // nothing to do
        return PARU_SUCCESS;
    }
    if (!Control)
    {
        return (PARU_INVALID) ;
    }

    DEBUGLEVEL(0);
    ParU_Numeric *Num;
    Num = *Num_handle;

    int64_t nf = Num->nf;

    // freeing the numerical input
    paru_free(Num->snz, sizeof(double), Num->Sx);
    if (Num->sunz > 0)
    {
        paru_free(Num->sunz, sizeof(double), Num->Sux);
    }
    if (Num->slnz > 0)
    {
        paru_free(Num->slnz, sizeof(double), Num->Slx);
    }

    paru_free(Num->sym_m, sizeof(int64_t), Num->Rs);
    paru_free(Num->sym_m, sizeof(int64_t), Num->Pfin);
    paru_free(Num->sym_m, sizeof(int64_t), Num->Ps);

    // free the factors
    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;

    for (int64_t i = 0; i < nf; i++)
    {
        if (Num->frowList)
            paru_free(Num->frowCount[i], sizeof(int64_t), Num->frowList[i]);
        if (Num->fcolList)
            paru_free(Num->fcolCount[i], sizeof(int64_t), Num->fcolList[i]);

        if (Us)
        {
            if (Us[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing Us=%p\n", Us[i].p));
                int64_t mm = Us[i].m;
                int64_t nn = Us[i].n;
                paru_free(mm * nn, sizeof(double), Us[i].p);
            }
        }

        if (LUs)
        {
            if (LUs[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing LUs=%p\n", LUs[i].p));
                int64_t mm = LUs[i].m;
                int64_t nn = LUs[i].n;
                paru_free(mm * nn, sizeof(double), LUs[i].p);
            }
        }
    }

    PRLEVEL(1, ("%% Done LUs\n"));
    paru_free(1, nf * sizeof(int64_t), Num->frowCount);
    paru_free(1, nf * sizeof(int64_t), Num->fcolCount);

    paru_free(1, nf * sizeof(int64_t *), Num->frowList);
    paru_free(1, nf * sizeof(int64_t *), Num->fcolList);

    paru_free(1, nf * sizeof(ParU_Factors), LUs);
    paru_free(1, nf * sizeof(ParU_Factors), Us);

    paru_free(1, sizeof(ParU_Numeric), Num);
    *Num_handle = NULL;
    return PARU_SUCCESS;
}
