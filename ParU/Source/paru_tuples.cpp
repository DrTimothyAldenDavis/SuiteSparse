////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_tuples ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief adding tuples to row and column list
 *
 * Row and column tuple list add is basically the same in sequential case
 * I have two functions for my parallel algorithm
 *
 * @param  a list of tuples and the the tuple we want to add
 * @return 0 on sucess
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"
ParU_Ret paru_add_rowTuple(paru_tupleList *RowList, int64_t row, paru_tuple T)
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("row =" LD ", (" LD "," LD ")\n", row, T.e, T.f));

    paru_tupleList *cur = &RowList[row];

    if (cur->len > cur->numTuple)
        cur->list[cur->numTuple++] = T;

    else
    {
        PRLEVEL(1, ("%%Row paru_tuple reallocating space\n"));
        int64_t newLen = cur->len * 2 + 1;
        paru_tuple *newList =
            static_cast<paru_tuple*>(paru_alloc(newLen, sizeof(paru_tuple)));
        if (newList == NULL)  // Error in allocating memory
            return PARU_OUT_OF_MEMORY;
        for (int64_t i = 0; i < cur->numTuple; ++i)  // copy old to new
            newList[i] = cur->list[i];
        paru_free(cur->len, sizeof(paru_tuple), cur->list);
        cur->len = newLen;
        cur->list = newList;
        cur->list[cur->numTuple++] = T;
    }
    return PARU_SUCCESS;
}
