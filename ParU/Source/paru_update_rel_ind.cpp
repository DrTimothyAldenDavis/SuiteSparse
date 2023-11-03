////////////////////////////////////////////////////////////////////////////////
/////////////////////////  paru_update_rel_ind ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief updating element's relative indices in regard to another element
 *      using my hash to find the columns and update relative indices
 *
 * @author Aznaveh
 * */

#include "paru_internal.hpp"
void paru_update_rel_ind_col(int64_t e, int64_t f,
    std::vector<int64_t> &colHash, paru_work *Work, ParU_Numeric *Num)
{
    // updating relative column index
    // it might be for curent element or for the Upart therefore we might even
    // dont have the curEl
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%%update relative in " LD "\n", f));

    paru_element **elementList = Work->elementList;
    paru_element *el = elementList[e];

    // int64_t *el_Index = colIndex_pointer (el); //col global index of destination
    int64_t *el_Index = (int64_t *)(el + 1);  // col global index of destination

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    // int64_t *colRelIndex = relColInd (paru_element *el);
    int64_t *colRelIndex = (int64_t *)(el + 1) + mEl + nEl;

    int64_t *fcolList = Num->fcolList[f];

    for (int64_t i = el->lac; i < nEl; i++)
    {
        int64_t colInd = el_Index[i];
        if (colInd < 0)
        {
            colRelIndex[i] = -1;
            continue;
        }
        PRLEVEL(1, ("%% searching for: cb_index[" LD "]=" LD "\n", i, colInd));
        int64_t found = paru_find_hash(colInd, colHash, fcolList);
        colRelIndex[i] = found;
        ASSERT(found != -1);
    }

    PRLEVEL(1, ("%%update relative in " LD " finished\n", f));

    // update the cVal of el
    el->cValid = Work->time_stamp[f];
}
