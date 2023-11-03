////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_prior_assemble ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief numerical assemble of prior fronts
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

ParU_Ret paru_prior_assemble(int64_t f, int64_t start_fac,
                                    std::vector<int64_t> &pivotal_elements,
                                    std::vector<int64_t> &colHash, heaps_info &hi,
                                    paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    int64_t *elCol = Work->elCol;

    paru_element **elementList = Work->elementList;
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;

    int64_t pMark = start_fac;

#ifndef NDEBUG
    int64_t *elRow = Work->elRow;
    int64_t el_ind = snM[f];
    PRLEVEL(PR, ("%%Inside prior eli=" LD " f=" LD "\n", el_ind, f));
    PRLEVEL(PR, ("%% pivotal size is " LD " ", pivotal_elements.size()));

#endif
    int64_t ii = 0;

    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
    {
        int64_t e = pivotal_elements[i];
        paru_element *el = elementList[e];
        PRLEVEL(PR, ("%% element= " LD "  \n", e));
        if (el == NULL)
        {
            PRLEVEL(PR, ("%% element= " LD " is NULL ii=" LD " \n", e, ii));
            continue;
        }
#ifndef NDEBUG
        PRLEVEL(PR, ("%%elRow[" LD "]=" LD " \n", e, elRow[e]));
        // if (elRow[e] != 0) PRLEVEL(-1, ("%%elRow[" LD "]=" LD " \n", e, elRow[e]));
        // ASSERT (elRow[e] == 0);
#endif

        if (el->nzr_pc == 0)  // if all the rows are available in current front
        {
            if (el->rValid == pMark || elCol[e] == 0)
            // it can be fully assembled
            // both a pivotal column and pivotal row
            {
                #ifndef NDEBUG
                PRLEVEL(PR, ("%%assembling " LD " in " LD "\n", e, el_ind));
                PRLEVEL(PR, ("%% size " LD " x " LD "\n", el->nrows, el->ncols));
                #endif
                paru_assemble_all(e, f, colHash, Work, Num);
                #ifndef NDEBUG
                PRLEVEL(PR, ("%%assembling " LD " in " LD " done\n", e, el_ind));
                #endif
                continue;
            }

            #ifndef NDEBUG
            PRLEVEL(PR, ("%%assembling " LD " in " LD "\n", e, el_ind));
            #endif
            paru_assemble_cols(e, f, colHash, Work, Num);
            #ifndef NDEBUG
            PRLEVEL(PR, ("%%partial col assembly" LD " in " LD " done\n", e, el_ind));
            #endif
            if (elementList[e] == NULL) continue;
        }
        else
        {
            if (el->rValid == pMark || elCol[e] == 0)
            // This element contributes to both pivotal rows and pivotal columns
            //  However it has zero rows in current pivotal columns therefore
            //  not all rows are there
            // it can be assembled partially
            //       ________________________________
            //       |      |                         |
            //       |      |                         |
            //       ___xxxxxxxxxxx____________________
            //       |  xxxxxxxxxxx                   |
            //       |  oxxo|oxoxox                   | <- assemble rows
            //       |  ooxx|oxoxox                   |
            //       |  oooo|oxoxox                   |
            //       ---------------------------------
            //          ooooooxxxxx  --> outsidie the front
            //          ooooooxxxxx
            //
            {
                paru_assemble_el_with0rows(e, f, colHash, Work, Num);
                if (elementList[e] == NULL) continue;
                #ifndef NDEBUG
                PRLEVEL(PR, ("%%assembling " LD " in " LD " done\n", e, el_ind));
                #endif
            }
            // keeping current element
        }

        pivotal_elements[ii++] = pivotal_elements[i];
    }

    if (ii < (int64_t)pivotal_elements.size())
    {
        PRLEVEL(PR, ("%% Prior: size was " LD " ", pivotal_elements.size()));
        PRLEVEL(PR, (" and now is " LD "\n ", ii));
        pivotal_elements.resize(ii);
    }

    /************ Making the heap from list of the immediate children
     * ******/
    PRLEVEL(1, ("%% Next: work on the heap \n"));
    ParU_Ret res_make_heap;
    res_make_heap = paru_make_heap(f, start_fac, pivotal_elements, hi, colHash,
                                   Work, Num);
    if (res_make_heap != PARU_SUCCESS) return res_make_heap;
    PRLEVEL(1, ("%% Done: work on the heap \n"));

    int64_t eli = snM[f];
    std::vector<int64_t> **heapList = Work->heapList;
    std::vector<int64_t> *curHeap = heapList[eli];

    if (curHeap->empty()) return PARU_SUCCESS;

#ifndef NDEBUG
    PR = 1;
#endif

#ifndef NDEBUG
    int64_t *lacList = Work->lacList;
    PRLEVEL(PR, ("%% current heap:\n %%"));
    for (int64_t k = 0; k < (int64_t)curHeap->size(); k++)
    {
        int64_t ee = (*curHeap)[k];
        paru_element *ell = elementList[ee];
        PRLEVEL(PR, ("" LD "-" LD "", k, ee));
        if (ell != NULL)
        {
            PRLEVEL(PR, ("(" LD ") ", lacList[ee]));
        }
        else
        {
            PRLEVEL(PR, ("(*" LD ") ", lacList[ee]));
        }
    }
    PRLEVEL(PR, ("\n"));
#endif

#ifndef NDEBUG
    // chekcing the heap
    for (int64_t i = curHeap->size() - 1; i > 0; i--)
    {
        int64_t elid = (*curHeap)[i];
        int64_t pelid = (*curHeap)[(i - 1) / 2];  // parent id
        if (lacList[pelid] > lacList[elid])
        {
            PRLEVEL(PR, ("" LD "-" LD "(" LD ") <", (i - 1) / 2, pelid, lacList[pelid]));
            PRLEVEL(PR, ("" LD "-" LD "(" LD ") \n", i, elid, lacList[elid]));
        }
        ASSERT(lacList[pelid] <= lacList[elid]);
    }

#endif
    return PARU_SUCCESS;
}
