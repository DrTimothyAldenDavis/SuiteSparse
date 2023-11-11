////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_heap.cpp //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  Wrappers for handling heap
 *
 * @author Aznaveh
 *
 */
#include "paru_internal.hpp"

void paru_check_prior_element(int64_t e, int64_t f, int64_t start_fac,
                              std::vector<int64_t> &colHash, paru_work *Work,
                              ParU_Numeric *Num)
// check if e can be assembeld into f
{
    int64_t *elRow = Work->elRow;

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];
    if (elRow[e] == 0 && el->rValid > start_fac)
    {  // all the rows are inside he current front; maybe assemble some cols
        paru_assemble_cols(e, f, colHash, Work, Num);
        return;
    }

    if (el->rValid == start_fac || el->cValid == Work->time_stamp[f])
    {  // all the cols are inside he current front; maybe assemble some rows
        paru_assemble_rows(e, f, colHash, Work, Num);
    }
}

ParU_Ret paru_make_heap(int64_t f, int64_t start_fac,
                        std::vector<int64_t> &pivotal_elements, heaps_info &hi,
                        std::vector<int64_t> &colHash, paru_work *Work,
                        ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *aChild = Sym->aChild;
    int64_t *aChildp = Sym->aChildp;
    int64_t *snM = Sym->super2atree;
    paru_element **elementList = Work->elementList;
    // int64_t m = Num-> m;

    std::vector<int64_t> **heapList = Work->heapList;

    int64_t eli = snM[f];

    // element id of the biggest child
    int64_t biggest_Child_id = -1;
    int64_t biggest_Child_size = -1;
    int64_t tot_size = 0;

    biggest_Child_id = hi.biggest_Child_id;
    biggest_Child_size = hi.biggest_Child_size;
    tot_size = hi.sum_size;

    int64_t *lacList = Work->lacList;
    auto greater = [&lacList](int64_t a, int64_t b) { return lacList[a] > lacList[b]; };

    PRLEVEL(PR, ("%% tot_size =  " LD "\n", tot_size));
    PRLEVEL(PR, ("%% biggest_Child_id = " LD " ", biggest_Child_id));
    PRLEVEL(PR, ("%% biggest_Child_size = " LD "\n", biggest_Child_size));
    int64_t size_of_rest = tot_size - biggest_Child_size + pivotal_elements.size();
    PRLEVEL(PR, ("%% the rest size = " LD "\n", size_of_rest));

    if (biggest_Child_id != -1)
    // There are still elements remained in the heaps
    {
        // shallow copy of the biggest child
        std::vector<int64_t> *curHeap = heapList[eli] = heapList[biggest_Child_id];
        heapList[biggest_Child_id] = NULL;

        // O(n) heapify of all children or O(klgn) add to the biggest child
        if (log2(biggest_Child_size) >
            (biggest_Child_size / (size_of_rest + 1)) + 1)
        {  // klogn
            PRLEVEL(PR, ("%% klogn algorhtm\n"));
            for (int64_t i = aChildp[eli]; i <= aChildp[eli + 1] - 1; i++)
            {
                int64_t chelid = aChild[i];  // element id of the child
                std::vector<int64_t> *chHeap = heapList[chelid];
                if (chHeap == NULL) continue;
                // concatening the child and freeing the memory
                for (int64_t e : *chHeap)
                {
                    if (elementList[e] != NULL)
                    {
                        paru_check_prior_element(e, f, start_fac, colHash, Work,
                                                 Num);
                        if (elementList[e] != NULL)
                        {
                            curHeap->push_back(e);
                            std::push_heap(curHeap->begin(), curHeap->end(),
                                           greater);
                        }
                    }
                }
                delete heapList[chelid];
                heapList[chelid] = NULL;
            }

            for (int64_t e : pivotal_elements)
            {
#ifndef NDEBUG
                ASSERT(elementList[e] != NULL);
#endif
                if (elementList[e] != NULL)
                {
                    PRLEVEL(PR, ("" LD "  ", e));
                    curHeap->push_back(e);
                    std::push_heap(curHeap->begin(), curHeap->end(), greater);
                }
            }
            curHeap->push_back(eli);
            std::push_heap(curHeap->begin(), curHeap->end(), greater);
            PRLEVEL(PR, ("%% " LD " pushed ", eli));
        }
        else
        {  // heapify
            PRLEVEL(PR, ("%%heapify with the size " LD "\n", tot_size));
            for (int64_t i = aChildp[eli]; i <= aChildp[eli + 1] - 1; i++)
            {
                int64_t chelid = aChild[i];  // element id of the child
                std::vector<int64_t> *chHeap = heapList[chelid];
                if (chHeap == NULL) continue;
                // concatening the child and freeing the memory

                // curHeap->insert(curHeap->end(),
                //      chHeap->begin(), chHeap->end());
                for (int64_t e : *chHeap)
                {
                    if (elementList[e] != NULL)
                    {
                        paru_check_prior_element(e, f, start_fac, colHash, Work,
                                                 Num);
                        if (elementList[e] != NULL) curHeap->push_back(e);
                    }
                }
                PRLEVEL(1,
                        ("%%Heap free %p id=" LD "\n", heapList[chelid], chelid));
                delete heapList[chelid];
                heapList[chelid] = NULL;
            }
            // adding pivotal elements and the current element
            curHeap->insert(curHeap->end(), pivotal_elements.begin(),
                            pivotal_elements.end());
            curHeap->push_back(eli);
            // heapifying
            std::make_heap(curHeap->begin(), curHeap->end(), greater);
        }
    }
    else
    {
        PRLEVEL(PR, ("Nothing in the heap. size of pivotal " LD " \n",
                     pivotal_elements.size()));
        std::vector<int64_t> *curHeap;
        try
        {
            curHeap = heapList[eli] = new std::vector<int64_t>;
        }
        catch (std::bad_alloc const &)
        {  // out of memory
            return PARU_OUT_OF_MEMORY;
        }
        // deep copy
        //*curHeap = pivotal_elements;
        // swap provides a shallow copy
        std::swap(*curHeap, pivotal_elements);
        curHeap->push_back(eli);
        std::make_heap(curHeap->begin(), curHeap->end(), greater);
    }

#ifndef NDEBUG
    std::vector<int64_t> *curHeap = heapList[eli];
    PRLEVEL(PR, ("After everything eli " LD " has " LD " elements\n", eli,
                 curHeap->size()));
    PRLEVEL(PR, ("%%Heap after making it(size = " LD ") \n", curHeap->size()));
    for (int64_t i = 0; i < (int64_t)curHeap->size(); i++)
    {
        int64_t elid = (*curHeap)[i];
        PRLEVEL(PR, (" " LD "(" LD ") ", elid, lacList[elid]));
    }
    PRLEVEL(PR, ("\n"));
    for (int64_t i = curHeap->size() - 1; i > 0; i--)
    {
        int64_t elid = (*curHeap)[i];
        int64_t pelid = (*curHeap)[(i - 1) / 2];  // parent id
        if (lacList[pelid] > lacList[elid])
            PRLEVEL(PR, ("ATT " LD "(" LD ")\n\n ", elid, lacList[elid]));
        ASSERT(lacList[pelid] <= lacList[elid]);
    }
#endif
    return PARU_SUCCESS;
}

ParU_Ret paru_make_heap_empty_el(int64_t f, std::vector<int64_t> &pivotal_elements,
                                 heaps_info &hi, paru_work *Work,
                                 ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *aChild = Sym->aChild;
    int64_t *aChildp = Sym->aChildp;
    int64_t *snM = Sym->super2atree;
    paru_element **elementList = Work->elementList;
    // int64_t m = Num-> m;

    std::vector<int64_t> **heapList = Work->heapList;

    int64_t eli = snM[f];

    // element id of the biggest child
    int64_t biggest_Child_id = -1;
    int64_t biggest_Child_size = -1;
    int64_t tot_size = 0;

    biggest_Child_id = hi.biggest_Child_id;
    biggest_Child_size = hi.biggest_Child_size;
    tot_size = hi.sum_size;

    int64_t *lacList = Work->lacList;
    auto greater = [&lacList](int64_t a, int64_t b) { return lacList[a] > lacList[b]; };

    PRLEVEL(PR, ("%% tot_size =  " LD "\n", tot_size));
    PRLEVEL(PR, ("%% biggest_Child_id = " LD " ", biggest_Child_id));
    PRLEVEL(PR, ("%% biggest_Child_size = " LD "\n", biggest_Child_size));
    int64_t size_of_rest = tot_size - biggest_Child_size + pivotal_elements.size();
    PRLEVEL(PR, ("%% the rest size = " LD "\n", size_of_rest));

    if (biggest_Child_id != -1)
    // There are still elements remained in the heaps
    {
        // shallow copy of the biggest child
        std::vector<int64_t> *curHeap = heapList[eli] = heapList[biggest_Child_id];
        heapList[biggest_Child_id] = NULL;

        // O(n) heapify of all children or O(klgn) add to the biggest child
        if (log2(biggest_Child_size) >
            (biggest_Child_size / (size_of_rest + 1)) + 1)
        {  // klogn
            PRLEVEL(PR, ("%% klogn algorhtm\n"));
            for (int64_t i = aChildp[eli]; i <= aChildp[eli + 1] - 1; i++)
            {
                int64_t chelid = aChild[i];  // element id of the child
                std::vector<int64_t> *chHeap = heapList[chelid];
                if (chHeap == NULL) continue;
                // concatening the child and freeing the memory
                for (int64_t e : *chHeap)
                {
                    if (elementList[e] != NULL)
                    {
                        curHeap->push_back(e);
                        std::push_heap(curHeap->begin(), curHeap->end(),
                                       greater);
                    }
                }
                delete heapList[chelid];
                heapList[chelid] = NULL;
            }

            for (int64_t e : pivotal_elements)
            {
                if (elementList[e] != NULL)
                {
                    PRLEVEL(PR, ("" LD "  ", e));
                    curHeap->push_back(e);
                    std::push_heap(curHeap->begin(), curHeap->end(), greater);
                }
            }
            std::push_heap(curHeap->begin(), curHeap->end(), greater);
            PRLEVEL(PR, ("%% " LD " pushed ", eli));
        }
        else
        {  // heapify
            PRLEVEL(PR, ("%%heapify with the size " LD "\n", tot_size));
            for (int64_t i = aChildp[eli]; i <= aChildp[eli + 1] - 1; i++)
            {
                int64_t chelid = aChild[i];  // element id of the child
                std::vector<int64_t> *chHeap = heapList[chelid];
                if (chHeap == NULL) continue;
                // concatening the child and freeing the memory

                // curHeap->insert(curHeap->end(),
                //      chHeap->begin(), chHeap->end());
                for (int64_t e : *chHeap)
                {
                    if (elementList[e] != NULL)
                    {
                        curHeap->push_back(e);
                    }
                }
                PRLEVEL(1,
                        ("%%Heap free %p id=" LD "\n", heapList[chelid], chelid));
                delete heapList[chelid];
                heapList[chelid] = NULL;
            }
            // adding pivotal elements and the current element
            curHeap->insert(curHeap->end(), pivotal_elements.begin(),
                            pivotal_elements.end());
            // heapifying
            std::make_heap(curHeap->begin(), curHeap->end(), greater);
        }
    }
    else
    {
        PRLEVEL(PR, ("Nothing in the heap. size of pivotal " LD " \n",
                     pivotal_elements.size()));
        std::vector<int64_t> *curHeap;
        try
        {
            curHeap = heapList[eli] = new std::vector<int64_t>;
        }
        catch (std::bad_alloc const &)
        {  // out of memory
            return PARU_OUT_OF_MEMORY;
        }
        // deep copy
        //*curHeap = pivotal_elements;
        // swap provides a shallow copy
        std::swap(*curHeap, pivotal_elements);
        std::make_heap(curHeap->begin(), curHeap->end(), greater);
    }

#ifndef NDEBUG
    std::vector<int64_t> *curHeap = heapList[eli];
    PRLEVEL(PR, ("After everything eli " LD " has " LD " elements\n", eli,
                 curHeap->size()));
    PRLEVEL(PR, ("%%Heap after making it(size = " LD ") \n", curHeap->size()));
    for (int64_t i = 0; i < (int64_t)curHeap->size(); i++)
    {
        int64_t elid = (*curHeap)[i];
        PRLEVEL(PR, (" " LD "(" LD ") ", elid, lacList[elid]));
    }
    PRLEVEL(PR, ("\n"));
    for (int64_t i = curHeap->size() - 1; i > 0; i--)
    {
        int64_t elid = (*curHeap)[i];
        int64_t pelid = (*curHeap)[(i - 1) / 2];  // parent id
        if (lacList[pelid] > lacList[elid])
            PRLEVEL(PR, ("ATT " LD "(" LD ")\n\n ", elid, lacList[elid]));
        ASSERT(lacList[pelid] <= lacList[elid]);
    }
#endif
    return PARU_SUCCESS;
}
