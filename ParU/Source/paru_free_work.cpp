////////////////////////////////////////////////////////////////////////////////
////////////////////////// paru_free_work.cpp //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Free workspace for numeric factorization.
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// paru_free_work: free all workspace in Numeric Work object
//------------------------------------------------------------------------------

ParU_Info paru_free_work
(
    const ParU_Symbolic Sym,
    paru_work *Work
)
{
    int64_t m = Sym->m - Sym->n1;
    int64_t nf = Sym->nf;
    int64_t n = Sym->n - Sym->n1;
    int64_t ntasks = Sym->ntasks ;
    PARU_FREE(m, int64_t, Work->rowSize);
    PARU_FREE(m + nf + 1, int64_t, Work->rowMark);
    PARU_FREE(m + nf, int64_t, Work->elRow);
    PARU_FREE(m + nf, int64_t, Work->elCol);
    PARU_FREE(ntasks, int64_t, Work->task_num_child);

    PARU_FREE(nf, int64_t, Work->time_stamp);

    paru_tupleList *RowList = Work->RowList;
    PRLEVEL(1, ("%% RowList =%p\n", RowList));

    if (RowList)
    {
        for (int64_t row = 0; row < m; row++)
        {
            int64_t len = RowList[row].len;
            PARU_FREE(len, paru_tuple, RowList[row].list);
        }
    }
    PARU_FREE(m, paru_tupleList, Work->RowList);

    if (Work->Diag_map)
    {
        PARU_FREE(n, int64_t, Work->Diag_map);
        PARU_FREE(n, int64_t, Work->inv_Diag_map);
    }

    paru_element **elementList;
    elementList = Work->elementList;

    PRLEVEL(1, ("%% Sym = %p\n", Sym));
    PRLEVEL(1, ("%% freeing initialized elements:\n"));
    if (elementList)
    {
        for (int64_t i = 0; i < m; i++)
        {
            // freeing all row elements
            int64_t e = Sym->row2atree[i];  // element number in augmented tree
            PRLEVEL(1, ("%% e =" LD "\t", e));
            paru_free_el(e, elementList);
        }

        PRLEVEL(1, ("\n%% freeing CB elements:\n"));
        for (int64_t i = 0; i < nf; i++)
        {
            // freeing all other elements
            int64_t e = Sym->super2atree[i];  //element number in augmented tree
            paru_free_el(e, elementList);
        }
    }

    PARU_FREE(m + nf + 1, paru_element, Work->elementList);
    PARU_FREE(m + nf, int64_t, Work->lacList);

    // in practice each parent should deal with the memory for the children
    std::vector<int64_t> **heapList = Work->heapList;
    // freeing memory of heaps.
    if (heapList != NULL)
    {
        for (int64_t eli = 0; eli < m + nf + 1; eli++)
        {
            if (heapList[eli] != NULL)
            {
                PRLEVEL(1,
                        ("%% " LD " has not been freed %p\n", eli, heapList[eli]));
                delete heapList[eli];
                heapList[eli] = NULL;
            }
            ASSERT (heapList[eli] == NULL) ;
        }
    }
    PARU_FREE(m + nf + 1, std::vector<int64_t> **, Work->heapList);
    PARU_FREE(m, int64_t, Work->row_degree_bound);

    return PARU_SUCCESS;
}

