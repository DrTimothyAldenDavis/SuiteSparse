////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_assemble_row2U.cpp ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  assemble numbers in U part of the matrix.
 *          It is per row, and the matrices are stored in column,
 *          therefore it can reduce the performance in some cases.
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

void paru_assemble_row_2U(int64_t e, int64_t f, int64_t sR, int64_t dR,
                          std::vector<int64_t> &colHash, 
                          paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);

    paru_element **elementList = Work->elementList;
    paru_element *el = elementList[e];

    if (el->cValid != Work->time_stamp[f])
        // if not updatated
        paru_update_rel_ind_col(e, f, colHash, Work, Num);

    ParU_Factors *Us = Num->partial_Us;
    double *uPart = Us[f].p;  // uPart

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    ParU_Factors *LUs = Num->partial_LUs;
    int64_t fp = LUs[f].n;

    // int64_t *el_colIndex = colIndex_pointer (curEl);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *colRelIndex = relColInd (paru_element *el);
    int64_t *colRelIndex = (int64_t *)(el + 1) + mEl + nEl;

    // double *el_Num = numeric_pointer (el);
    double *sM = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);

    int64_t ncolsSeen = el->ncolsleft;
    for (int64_t j = el->lac; j < nEl; j++)
    {
        int64_t rj = colRelIndex[j];
        if (el_colIndex[j] >= 0)
        {  // If still valid
            ncolsSeen--;
            PRLEVEL(1,
                    ("%% sM [" LD "] =%2.5lf \n", mEl * j + sR, sM[mEl * j + sR]));
            PRLEVEL(1, ("%% uPart [" LD "] =%2.5lf \n", rj * fp + dR,
                        uPart[rj * fp + dR]));
            //**//#pragma omp atomic
            uPart[rj * fp + dR] += sM[mEl * j + sR];
            PRLEVEL(1, ("%% uPart [" LD "] =%2.5lf \n", rj * fp + dR,
                        uPart[rj * fp + dR]));
            if (ncolsSeen == 0) break;
        }
    }
}
