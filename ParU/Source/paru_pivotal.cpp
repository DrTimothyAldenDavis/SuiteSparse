////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_pivotal ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief
 *  adding the list of pivotal elements from the heap, computing the list of
 *  rows and assembling pivotal columns
 *
 * @param pivotal_elements list
 *          f and Num
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"

ParU_Ret paru_pivotal(std::vector<int64_t> &pivotal_elements,
                      std::vector<int64_t> &panel_row, int64_t &zero_piv_rows, int64_t f,
                      heaps_info &hi, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;
    std::vector<int64_t> **heapList = Work->heapList;
    int64_t eli = snM[f];

    int64_t *Super = Sym->Super;
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t col2 = Super[f + 1];
    int64_t *aChild = Sym->aChild;
    int64_t *aChildp = Sym->aChildp;

#ifndef NDEBUG
    int64_t m = Num->m;
    PRLEVEL(PR, ("%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"));
    PRLEVEL(PR, ("%% Pivotal assembly of front %ld (eli %ld) cols %ld-%ld\n", f,
                 eli, col1, col2));
#endif

    paru_element **elementList = Work->elementList;

    int64_t *lacList = Work->lacList;

    int64_t *rowMarkp = Work->rowMark;
    int64_t rowMark = 0;

    /*****  making the list of elements that contribute to pivotal columns ****/
    int64_t biggest_Child_id = -1;
    int64_t biggest_Child_size = -1;
    int64_t tot_size = 0;

    for (int64_t i = aChildp[eli]; i <= aChildp[eli + 1] - 1; i++)
    {
        int64_t chelid = aChild[i];  // element id of the child
        // max(rowMark , child->rowMark)
        int64_t f_rmark = rowMarkp[chelid];
        rowMark = rowMark > f_rmark ? rowMark : f_rmark;

        PRLEVEL(PR, ("%% chelid = %ld\n", chelid));
        std::vector<int64_t> *curHeap = heapList[chelid];

        if (curHeap == NULL) continue;

        while (curHeap->size() > 0)
        // pop from the heap and put it in pivotal_elements
        {
            int64_t frontEl = curHeap->front();
            int64_t lacFel = lacList[frontEl];
            PRLEVEL(PR, ("%% element = %ld col1=%ld", frontEl, col1));
            PRLEVEL(PR, (" lac_el = %ld \n", lacFel));
            // ASSERT(lacFel >= col1);
            PRLEVEL(PR, ("%% curHeap->size= %ld \n", curHeap->size()));

            if (lacFel >= col2) break;

            if (elementList[frontEl] != NULL)
                pivotal_elements.push_back(frontEl);
            std::pop_heap(
                curHeap->begin(), curHeap->end(),
                [&lacList](int64_t a, int64_t b) { return lacList[a] > lacList[b]; });
            curHeap->pop_back();
        }

        if (curHeap->empty())
        {
            delete heapList[chelid];
            heapList[chelid] = NULL;
        }
        else
        {
            // IMPORTANT: type conversion is necessary
            int64_t cur_size = curHeap->size();
            PRLEVEL(PR, ("%% curHeap->size= *%ld \n", curHeap->size()));
            PRLEVEL(PR, ("%% biggest_Child_size = %ld \n", biggest_Child_size));
            tot_size += curHeap->size();
            if (cur_size > biggest_Child_size)
            {
                PRLEVEL(PR, ("%% biggest_Child_id = %ld \n", biggest_Child_id));
                biggest_Child_id = chelid;
                biggest_Child_size = cur_size;
            }
        }
    }

    hi.sum_size = tot_size;
    hi.biggest_Child_id = biggest_Child_id;
    hi.biggest_Child_size = biggest_Child_size;

    PRLEVEL(PR, ("%%Inside pivot tot_size= %ld \n", hi.sum_size));
    PRLEVEL(PR, ("%% biggest_Child_id = %ld \n", hi.biggest_Child_id));
    PRLEVEL(PR, ("%% hi.biggest_Child_size = %ld \n", hi.biggest_Child_size));

    rowMarkp[eli] = rowMark;

    int64_t *isRowInFront = Work->rowSize;
    ++rowMark; 

    #ifndef PARU_COVERAGE  //overflow is very hard to test in coverage 
    if (rowMark < 0)
    // just look at the children
    {  // in rare case of overflow
        int64_t *Sleft = Sym->Sleft;
        // first column of current front until first column of next front
        for (int64_t i = Sleft[col1]; i < Sleft[Super[f + 1]]; i++)
            isRowInFront[i] = -1;
        rowMark = rowMarkp[eli] = 1;
    }
    #endif
    rowMarkp[eli] = rowMark;
    PRLEVEL(1, ("%% rowMark=%ld;\n", rowMark));

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% pivotal columns eli(%ld): ", eli));
    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
        PRLEVEL(PR, ("%ld ", pivotal_elements[i]));
    PRLEVEL(PR, ("\n"));
    std::set<int64_t> stl_rowSet;
    std::set<int64_t>::iterator it;
#endif
    ParU_Control *Control = Num->Control;
    int64_t panel_width = Control->panel_width;
    int64_t fp = col2 - col1; /* first fp columns are pivotal */
    int64_t num_panels = (int64_t)ceil((double)fp / panel_width);

    int64_t *frowList = Num->frowList[f];
    int64_t rowCount = 0;

    /*************** finding set of rows in current front *********************/
    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
    {
        int64_t e = pivotal_elements[i];
        paru_element *el = elementList[e];
        ASSERT(el != NULL);

#ifndef NDEBUG
        // int64_t *el_colIndex = colIndex_pointer (curEl);
        int64_t *el_colIndex = (int64_t *)(el + 1);
        PRLEVEL(PR, ("current element(%ld) %ld-%ld\n", e, col1, col2));
        PRLEVEL(PR, ("lac = %ld ", el->lac));
        PRLEVEL(PR, ("lac_col = %ld\n ", lacList[e]));
        ASSERT(el_colIndex[el->lac] >= col1);
        if (PR <= 0) paru_print_element(e, Work, Num);
#endif

        int64_t mEl = el->nrows;
        int64_t nEl = el->ncols;

        // int64_t *el_rowIndex = rowIndex_pointer (el);
        int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

        // int64_t *rowRelIndex = relRowInd (el);
        int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

        PRLEVEL(1, ("%% rowMark=%ld;\n", rowMark));

        el->nzr_pc = 0;  // initializing ; number of zero rows
        int64_t nrows2bSeen = el->nrowsleft;

        for (int64_t rEl = 0; rEl < mEl; rEl++)
        {
            int64_t curRow = el_rowIndex[rEl];
            PRLEVEL(1, ("%% curRow =%ld rEl=%ld\n", curRow, rEl));
            if (nrows2bSeen == 0) break;
            if (curRow < 0) continue;  // that row has already deleted
            nrows2bSeen--;

#ifndef NDEBUG
            //            stl_rowSet.insert(curRow);
            PRLEVEL(1, ("%% %p ---> isRowInFront [%ld]=%ld\n",
                        isRowInFront + curRow, curRow, isRowInFront[curRow]));
#endif

            if (isRowInFront[curRow] < rowMark)
            {  // first time seeing curRow
                // int64_t *el_colIndex = colIndex_pointer (curEl);
                int64_t *el_colIndex = (int64_t *)(el + 1);

                //#if 1
                // checkikng if the numerical values are hard zero
                // look at the pivotal columns and check if there is any
                // nonzeros if there is none I can skip adding this row
                bool nz_found = false;
                for (int64_t cEl = el->lac; cEl < nEl; cEl++)
                {
                    if (el_colIndex[cEl] < 0)  // already assembled somewhere
                        continue;
                    if (el_colIndex[cEl] >= col2) break;

                    // double *el_Num = numeric_pointer (el);
                    double *el_Num =
                        (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);
                    if (el_Num[cEl * mEl + rEl] != 0.0)
                    {
                        nz_found = true;
                        break;
                    }
                }
                if (!nz_found)
                {
                    el->nzr_pc++;
                    PRLEVEL(1, ("%% Found a row with all zeroes!! "
                                "curRow =%ld el=%ld\n",
                                curRow, e));

                    zero_piv_rows++;
                    rowRelIndex[rEl] = -1;
#ifndef NDEBUG
                    if (PR <= 0) paru_print_element(e, Work, Num);
#endif
                    continue;  // Not adding the row
                }
                //#endif
                // Adding curRow to the set
#ifndef NDEBUG
                stl_rowSet.insert(curRow);
#endif
                PRLEVEL(1, ("%%curRow =%ld rowCount=%ld\n", curRow, rowCount));
                frowList[rowCount] = curRow;
                rowRelIndex[rEl] = rowCount;
                PRLEVEL(1, ("%%1st: rowRelIndex[%ld] = %ld\n", rEl, rowCount));
                isRowInFront[curRow] = rowMark + rowCount++;
            }
            else
            {  // already seen curRow
                PRLEVEL(1, ("%%curRow =%ld rowCount=%ld\n", curRow, rowCount));
                PRLEVEL(1, ("%%before updating rowRelIndex[%ld] = %ld\n", rEl,
                            rowRelIndex[rEl]));
                PRLEVEL(1, ("%% rowMark =%ld\n", rowMark));
                rowRelIndex[rEl] = isRowInFront[curRow] - rowMark;
                PRLEVEL(1, ("%%N1st: rowRelIndex[%ld] = %ld\n", rEl,
                            rowRelIndex[rEl]));
            }

            ASSERT(rowCount <= m);
#ifndef NDEBUG
            if (rowCount != (int64_t)stl_rowSet.size())
            {
                PRLEVEL(1, ("%%curRow =%ld rowCount=%ld\n", curRow, rowCount));
                PRLEVEL(1, ("%%stl_rowSet.size()=%ld \n", stl_rowSet.size()));
            }
#endif
            ASSERT(rowCount == (int64_t)stl_rowSet.size());
        }
        panel_row[(lacList[e] - col1) / panel_width] = rowCount;
#ifndef NDEBUG
        PRLEVEL(PR, ("%%rowCount=%ld", rowCount));
        PRLEVEL(PR, (" lac=%ld", lacList[e]));
        ASSERT((lacList[e] - col1) / panel_width < num_panels);
        ASSERT((lacList[e] - col1) / panel_width >= 0);
        PRLEVEL(PR, (" ind.=%ld\n", (lacList[e] - col1) / panel_width));
#endif
    }

    if (rowCount < fp)
    {
#ifndef NDEBUG
        // there is a structural problem
        PRLEVEL(PR,
                ("%%STRUCTURAL PROBLEM! rowCount=%ld, fp =%ld", rowCount, fp));
#endif
        if (rowCount + zero_piv_rows > fp)
        {
            PRLEVEL(PR,
                    (" it can be solved by adding %ld zeros", zero_piv_rows));
        }
        else
        {
            PRLEVEL(PR, (" it wil FAIL"));
        }
#ifndef NDEBUG
        PRLEVEL(PR, ("\n"));
#endif
    }

    // make sure that all panel_row is correctly initialized
    PRLEVEL(PR, ("%% num_panels: %ld \n ", num_panels));
    PRLEVEL(PR, ("%% panel_row: \n %%"));
    int64_t pprow = panel_row[0];
    PRLEVEL(PR, ("%% %ld ", pprow));
    ASSERT(pprow != 0);
    for (int64_t i = 1; i < num_panels; i++)
    {
        if (pprow > panel_row[i])
        {
            panel_row[i] = pprow;
        }
        else
        {
            pprow = panel_row[i];
        }
        PRLEVEL(1, ("%ld ", panel_row[i]));
        ASSERT(panel_row[i] > 0);
        ASSERT(panel_row[i] <= m);
    }

    Num->frowCount[f] = rowCount;
    // No support for max and min in OpenMP C++
    //pragma omp atomic capture 
    //{
    //    Num->max_row_count = MAX(Num->max_row_count, rowCount);
    //}

#ifndef NDEBUG /* Checking if pivotal rows are correct */
    PRLEVEL(PR, ("%% panel_row: \n %%"));
    for (int64_t i = 0; i < num_panels; i++) PRLEVEL(PR, ("%ld ", panel_row[i]));
    PRLEVEL(PR, ("\n"));
    PRLEVEL(PR, ("%%There are %ld rows x %ld columns %ld - %ld "
                 "in front %ld with %ld zero rows: \n%%",
                 rowCount, fp, col1, col2, f, zero_piv_rows));
    for (int64_t i = 0; i < rowCount; i++) PRLEVEL(PR, (" %ld", frowList[i]));
    PRLEVEL(PR, ("\n"));
    int64_t stl_rowSize = stl_rowSet.size();
    if (rowCount != stl_rowSize)
    {
        PRLEVEL(PR, ("%% STL %ld:\n", stl_rowSize));
        for (it = stl_rowSet.begin(); it != stl_rowSet.end(); it++)
            PRLEVEL(PR, ("%% %ld", *it));
        PRLEVEL(PR, ("\n%%My Set %ld:\n", rowCount));
        for (int64_t i = 0; i < rowCount; i++) PRLEVEL(PR, ("%% %ld", frowList[i]));
        PRLEVEL(PR, ("\n"));
    }
    ASSERT(rowCount == stl_rowSize);
#endif

    int64_t fm = Sym->Fm[f]; /* Upper bound number of rows of F */
    ASSERT(fm >= rowCount);

    // freeing extra space for rows
    size_t sz = (size_t) fm;
    if (rowCount != fm)
    {
        frowList = 
            (int64_t *)paru_realloc(rowCount, sizeof(int64_t), frowList, &sz);
    }
    if (sz != (size_t) rowCount)
    {
        PRLEVEL(1, ("ParU: 0ut of memory when tried to reallocate for frowList"
                    "part %ld\n", f));
        return PARU_OUT_OF_MEMORY;
    }

    Num->frowList[f] = frowList ;
    double *pivotalFront = (double *)paru_calloc(rowCount * fp, sizeof(double));

    if (pivotalFront == NULL)
    {
        PRLEVEL(1, ("ParU: 0ut of memory when tried to allocate for pivotal "
                    "part %ld\n", f));
        return PARU_OUT_OF_MEMORY;
    }
#ifndef NDEBUG
    Work->actual_alloc_LUs += rowCount * fp;
    Work->actual_alloc_row_int += rowCount;
    if (fm != rowCount) PRLEVEL(PR, ("%% fm=%ld rowCount=%ld ", fm, rowCount));
    PRLEVEL(PR, ("%% LUs=%ld ", Work->actual_alloc_LUs));
    PRLEVEL(PR, ("%% pivotalFront = %p size=%ld", pivotalFront, rowCount * fp));
    int64_t act = Work->actual_alloc_LUs + Work->actual_alloc_Us +
        Work->actual_alloc_row_int;
    int64_t upp = Sym->Us_bound_size + Sym->LUs_bound_size + Sym->row_Int_bound +
        Sym->col_Int_bound;
    PRLEVEL(PR, ("%% MEM=%ld percent=%lf%%", act, 100.0 * act / upp));
    PRLEVEL(PR, ("%% MEM=%ld percent=%lf%%\n", act, 100.0 * act / upp));
#endif
    ParU_Factors *LUs = Num->partial_LUs;
    Num->frowCount[f] = rowCount;

    LUs[f].m = rowCount;
    LUs[f].n = fp;
    ASSERT(LUs[f].p == NULL);
    LUs[f].p = pivotalFront;

    /***************  assembling the pivotal part of the front ****************/
    /*
     *
     *  el           nEl
     *              6, 7, 11, 12
     *             _____________
     *          23 | X  Y  .  .     stored in memory like this:
     *      mEl 17 | X  Y  .  .     ..6, 7,11, 12, 23, 17, 2, X, X, X, Y, Y, Y,
     *           2 | X  Y  .  .
     *
     *    It must be assembled in current pivotal fron like this:
     *                                     fp
     *                                 col1, ... , col
     *
     *                                  6, 7, 8, 9, 10
     *                                  ______________
     *                          0   23 | X  Y  .  .  .
     *               rowCount   1    2 | X  Y  .  .  .
     *                          2    4 | *  *  .  .  .  isRowInFront[4] == 2
     *                          3   17 | X  Y  .  .  .
     * */

    int64_t ii = 0;  // using for resizing pivotal_elements
    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
    {
        int64_t e = pivotal_elements[i];
        paru_full_summed(e, f, Work, Num);
        if (elementList[e] != NULL)
        {  // keeping the element
            pivotal_elements[ii++] = pivotal_elements[i];
        }
    }

    if (ii < (int64_t)pivotal_elements.size())
    {
        PRLEVEL(PR, ("%% pivotal size was %ld ", pivotal_elements.size()));
        PRLEVEL(PR, ("%% and now is %ld\n ", ii));
        pivotal_elements.resize(ii);
    }

    // second pass through elements with zero rows and
    // growing them to better fit in current front
    // This can possibly help in more assembly

    int64_t num_children_with0 = 0;
    int64_t num_children_with0_which_fit = 0;

    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
    {
        int64_t e = pivotal_elements[i];
        paru_element *el = elementList[e];
        if (el->nzr_pc > 0)  // an elemen that has at least one zero row
        {
            num_children_with0++;
            int64_t mEl = el->nrows;
            int64_t nEl = el->ncols;

            // int64_t *el_rowIndex = rowIndex_pointer (el);
            int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

            // int64_t *rowRelIndex = relRowInd (el);
            int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;
            int64_t nrows2bSeen = el->nrowsleft;

            for (int64_t rEl = 0; rEl < mEl; rEl++)
            {
                int64_t curRow = el_rowIndex[rEl];
                if (nrows2bSeen == 0) break;
                if (curRow < 0) continue;  // that row has already deleted
                nrows2bSeen--;
                if (rowRelIndex[rEl] == -1)  // the zero row
                {
                    if (isRowInFront[curRow] >= rowMark)
                    {
                        el->nzr_pc--;
                        rowRelIndex[rEl] = isRowInFront[curRow] - rowMark;
                    }
                }
            }
#ifndef NDEBUG
            if (el->nzr_pc == 0)
            {  // all the zero rows fit in the front
                PRLEVEL(1, ("%%element %ld totally fit in current front %ld\n",
                            e, f));
                num_children_with0_which_fit++;
            }
            ASSERT(el->nzr_pc >= 0);
#endif
        }
    }

    if (num_children_with0 == num_children_with0_which_fit)
    {  // all the children fit within current front
        zero_piv_rows = 0;
    }

#ifndef NDEBUG
    PR = 2;
    PRLEVEL(PR, ("%% pivotal columns eli(%ld) after resizing: ", eli));
    for (int64_t i = 0; i < (int64_t)pivotal_elements.size(); i++)
        PRLEVEL(PR, ("%ld ", pivotal_elements[i]));
    PRLEVEL(PR, ("\n"));

    PR = 2;
    PRLEVEL(PR, ("%% After all the assemble %ld, z=%ld\n", f, zero_piv_rows));
    PRLEVEL(PR, ("%% x =  \t"));
    for (int64_t c = col1; c < col2; c++) PRLEVEL(PR, ("%ld\t\t", c));
    PRLEVEL(PR, (" ;\n"));
    for (int64_t r = 0; r < rowCount; r++)
    {
        PRLEVEL(PR, ("%% %ld\t", frowList[r]));
        for (int64_t c = col1; c < col2; c++)
            PRLEVEL(PR, (" %2.5lf\t", pivotalFront[(c - col1) * rowCount + r]));
        PRLEVEL(PR, ("\n"));
    }
    PRLEVEL(PR, (" %% %ld*%ld\n", rowCount, fp));
    PR = 2;
    PRLEVEL(PR, ("x%ld = [ \t", f));
    for (int64_t r = 0; r < rowCount; r++)
    {
        for (int64_t c = col1; c < col2; c++)
            PRLEVEL(PR, (" %2.5lf\t", pivotalFront[(c - col1) * rowCount + r]));
        PRLEVEL(PR, ("\n"));
    }
    PRLEVEL(PR, (" ]; %% %ld*%ld\n", rowCount, fp));
    PR = 1;
#endif

    rowMarkp[eli] += rowCount;
    PRLEVEL(1, ("%% rowMarkp[%ld] =%ld\n", eli, rowMarkp[eli]));
    return PARU_SUCCESS;
}
