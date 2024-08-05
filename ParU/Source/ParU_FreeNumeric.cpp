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
    ParU_Numeric *Num_handle,   // numeric object to free
    // control:
    ParU_Control Control
)
{
    if (Num_handle == NULL || *Num_handle == NULL)
    {
        // nothing to do
        return PARU_SUCCESS;
    }

    DEBUGLEVEL(0);
    ParU_Numeric Num = *Num_handle ;

    int64_t nf = Num->nf;

    // freeing the numerical input
    PARU_FREE(Num->snz, double, Num->Sx);
    if (Num->sunz > 0)
    {
        PARU_FREE(Num->sunz, double, Num->Sux);
    }
    if (Num->slnz > 0)
    {
        PARU_FREE(Num->slnz, double, Num->Slx);
    }

    PARU_FREE(Num->sym_m, int64_t, Num->Rs);
    PARU_FREE(Num->sym_m, int64_t, Num->Pfin);
    PARU_FREE(Num->sym_m, int64_t, Num->Ps);

    // free the factors
    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;

    for (int64_t i = 0; i < nf; i++)
    {
        if (Num->frowList)
        {
            PARU_FREE(Num->frowCount[i], int64_t, Num->frowList[i]);
        }
        if (Num->fcolList)
        {
            PARU_FREE(Num->fcolCount[i], int64_t, Num->fcolList[i]);
        }

        if (Us)
        {
            if (Us[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing Us=%p\n", Us[i].p));
                int64_t mm = Us[i].m;
                int64_t nn = Us[i].n;
                PARU_FREE(mm * nn, double, Us[i].p);
            }
        }

        if (LUs)
        {
            if (LUs[i].p != NULL)
            {
                PRLEVEL(1, ("%% Freeing LUs=%p\n", LUs[i].p));
                int64_t mm = LUs[i].m;
                int64_t nn = LUs[i].n;
                PARU_FREE(mm * nn, double, LUs[i].p);
            }
        }
    }

    PRLEVEL(1, ("%% Done LUs\n"));
    PARU_FREE(nf, int64_t, Num->frowCount);
    PARU_FREE(nf, int64_t, Num->fcolCount);
    PARU_FREE(nf, int64_t *, Num->frowList);
    PARU_FREE(nf, int64_t *, Num->fcolList);
    PARU_FREE(nf, ParU_Factors, Num->partial_LUs) ;
    PARU_FREE(nf, ParU_Factors, Num->partial_Us) ;
    PARU_FREE(1, ParU_Numeric_struct, Num);
    (*Num_handle) = NULL ;
    return (PARU_SUCCESS) ;
}
