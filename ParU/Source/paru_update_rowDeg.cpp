////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_update_rowDeg /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  growing current front if necessary and update the row degree of
 *   current front for current panel.
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"

void paru_update_rowDeg(int64_t panel_num, int64_t row_end, int64_t f,
    int64_t start_fac, std::set<int64_t> &stl_colSet,
    std::vector<int64_t> &pivotal_elements, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
#ifndef NDEBUG
    int64_t n = Num->n;
    static int64_t r1 = 0, r2 = 0, r3 = 0;
#endif
    PRLEVEL(1, ("%%-------ROW degree update of panel " LD " of front " LD " \n",
                panel_num, f));
    ParU_Control *Control = Num->Control;
    int64_t panel_width = Control->panel_width;
    paru_element **elementList = Work->elementList;

    int64_t *elRow = Work->elRow;
    int64_t *elCol = Work->elCol;

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *Super = Sym->Super;
    int64_t col1 = Super[f];  // fornt F has columns col1:col2-1
    int64_t col2 = Super[f + 1];
    int64_t fp = col2 - col1;  // first fp columns are pivotal

    int64_t pMark = start_fac;              // Mark for pivotal rows
    int64_t npMark = ++Work->time_stamp[f];  // making all the markings invalid

    int64_t colCount = stl_colSet.size();

    int64_t j1 = panel_num * panel_width;  // panel starting column
    int64_t j2 = (j1 + panel_width < fp) ? j1 + panel_width : fp;

    int64_t rowCount = Num->frowCount[f];
    int64_t *row_degree_bound = Work->row_degree_bound;

    std::set<int64_t> stl_newColSet;  // the list of new columns

    /*************** finding set of non pivotal cols in current front *********/
    /*
     *    Mark seen elements with pMark
     *        if Marked already added to the list
     *        else all columns are added to the current front
     *
     *                <----------fp--------->
     *                        j1     j2              Update here
     *                         ^     ^                stl_newColSet colCount
     *                         |     | stl_colSet     ^ . . .   ^
     *                         |     |        \       |         |
     *             F           |     |         [QQQQQ|OOOOOOOOO|....
     *              \  ____..._|_  ____...__ _____________________________...
     * ^              |\      |     |       #  ^     |         |
     * |              | \     |     |       #  | old | added   |
     * |              |  \    |     |       #  | list|  columns|
     * |              |___\...|_____|__...__#  |     |         |
     * |   ^    j1--> |       |\* **|       #  fp  oooooooooooo|
     * |   |     H    |       |**\**|       #  |   ooooEloooooo|
     * | panel   E    |       |***\*|       #  |   oooooooooooo|
     * | width   R    |       |***\*|       #  |               |
     * |   |     E    |       |***\*|       #  |    00000000   |
     * |   v          |____...|________..._ #  |    00El0000   |
     * |        j2--> |       |     |       #  v    00000000   |           ...
     * rowCount       |==================================================
     * |              |       |     |       |
     * |              |       |     |       |
     * |              |       |row_end      |
     * |              .       .      .      .
     * |              .       .      .      .
     * |              .       .      .      .
     * v              |___....______________|
     *
     */
    int64_t *frowList = Num->frowList[f];

    paru_tupleList *RowList = Work->RowList;
    for (int64_t i = j1; i < j2; i++)
    {
        int64_t curFsRow = frowList[i];
#ifndef NDEBUG
        int64_t curFsRowIndex = i;  // current fully summed row index
        PRLEVEL(1, ("%% 4: curFsRowIndex = " LD "\n", curFsRowIndex));
        PRLEVEL(1, ("%% curFsRow =" LD "\n", curFsRow));
#endif
        paru_tupleList *curRowTupleList = &RowList[curFsRow];
        int64_t numTuple = curRowTupleList->numTuple;
        ASSERT(numTuple >= 0);
        paru_tuple *listRowTuples = curRowTupleList->list;
        PRLEVEL(1, ("%% 4: numTuple = " LD "\n", numTuple));

        int64_t pdst = 0, psrc;
        for (psrc = 0; psrc < numTuple; psrc++)
        {
            paru_tuple curTpl = listRowTuples[psrc];

            int64_t e = curTpl.e;
            int64_t curRowIndex = curTpl.f;
#ifndef NDEBUG
            r1++;
#endif
            if (e < 0 || curRowIndex < 0) continue;

            paru_element *el = elementList[e];

            if (el == NULL) continue;

            int64_t nEl = el->ncols;
            // int64_t *el_rowIndex = rowIndex_pointer (el);
            int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;
            if (el_rowIndex[curRowIndex] < 0) continue;

            int64_t mEl = el->nrows;
            // int64_t *rowRelIndex = relRowInd (el);
            int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;
            rowRelIndex[curTpl.f] = curFsRow;

            listRowTuples[pdst++] = curTpl;  // keeping the tuple

            ASSERT(el_rowIndex[curRowIndex] == curFsRow);

            if (el->rValid != pMark)
            {  // an element never seen before

                PRLEVEL(
                    1, ("%%P: first time seen elRow[" LD "]=" LD " \n", e, elRow[e]));
                PRLEVEL(1, ("%%pMark=" LD "  npMark= " LD "\n", pMark, npMark));

                // if (el->rValid < pMark)
                if (npMark == pMark + 1)
                    elRow[e] = el->nrowsleft - 1;  // initiaze
                el->rValid = pMark;

                PRLEVEL(1, ("%%changed to elRow[" LD "]=" LD " \n", e, elRow[e]));
#ifndef NDEBUG
                if (el->rValid > pMark)
                    PRLEVEL(1, ("%%pMark=" LD "  rVal= " LD "\n", pMark, el->rValid));
#endif
            }
            else  // el->rValid == pMark
            {     // already added to pivotal rows
                if (npMark == pMark + 1) elRow[e]--;
                PRLEVEL(1, ("%%already seen elRow[" LD "]=" LD " \n", e, elRow[e]));
                continue;
            }

            // int64_t *el_colIndex = colIndex_pointer (el);
            int64_t *el_colIndex = (int64_t *)(el + 1);

            PRLEVEL(1, ("%% element= " LD "  nEl =" LD " \n", e, nEl));
            for (int64_t cEl = 0; cEl < nEl; cEl++)
            {
                int64_t curCol = el_colIndex[cEl];
                PRLEVEL(1, ("%% curCol =" LD "\n", curCol));
                ASSERT(curCol < n);

                if (curCol < 0)  // already deleted
                    continue;

                
                //Found this in coverage test
                //It also makes sense while the pivotal columns have been 
                //already deleted
                ////is a pivotal col 
                //if (curCol < col2 && curCol >= col1) continue;
                ASSERT(curCol >= col2 || curCol < col1); 

                if (stl_colSet.find(curCol) == stl_colSet.end())
                {
                    stl_colSet.insert(curCol);
                    stl_newColSet.insert(curCol);
                    colCount++;
                }
#ifndef NDEBUG
                PR = 1;
                // stl_colSet.insert (curCol);
                for (std::set<int64_t>::iterator it = stl_colSet.begin(); it != stl_colSet.end(); it++)
                    PRLEVEL(PR, ("%%@  " LD "", *it));

#endif

                ASSERT(colCount <= n);
            }
        }

        curRowTupleList->numTuple = pdst;
    }

    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];
    if (colCount == 0)
    {  // there is no CB, Nothing to be done
        Work->rowMark[eli] += rowCount;
        return;
    }

#ifndef NDEBUG /* Checking if columns are correct */
    PR = 1;
    PRLEVEL(PR, ("%% There are " LD " columns in this contribution block: \n",
                 colCount));
    PRLEVEL(PR, ("\n"));
    int64_t stl_colSize = stl_colSet.size();

    if (colCount != stl_colSize)
    {
        PRLEVEL(PR, ("%% STL " LD ":\n", stl_colSize));
        for (std::set<int64_t>::iterator it = stl_colSet.begin(); it != stl_colSet.end(); it++)
            PRLEVEL(PR, ("%%  " LD "", *it));
        PRLEVEL(PR, ("\n%% My Set " LD ":\n", colCount));
        PRLEVEL(PR, ("\n"));
    }
    ASSERT(colCount == stl_colSize);
#endif

    // if the front did not grow, there is nothing else to do
    if (stl_newColSet.size() == 0) return;

    Num->fcolCount[f] = colCount;

    /**** only travers over elements that contribute to pivotal columns *******/
    /*         to find their intersection
     *
     *         This can be fully skipped. I also can look other children in the
     *         heap
     *
     *            <----------fp--------->
     *                                          stl_newColSet
     *                             stl_colSet     ^        ^
     *                                    \       |        |
     *         F                           [QQQQQ|OOOOOOOOO|....
     *          \  ____..._________...__ _____________________________...
     * ^          |\      |     |       #  ^     |         |
     * |          | \     |     |       #  | old | added   |
     * |          |  \    |     |       #  | list|  columns|
     * |          |___\...|_____|__...__#  |     |         |
     * |   ^      |       |\* **|       #  fp  oooooooooooo|
     * |   |      |       |**\**|       #  |   ooo El ooooo|
     * | panel    |       |***\*|       #  |   oooooooooooo|
     * | width    |       |***\*|       #  |               |
     * |   |      |       |***\*|       #  |    00000000   |
     * |   v      |____...|________..._ #  |    000 El 0   |
     * |          |       |     |       #  v    00000000   |           ...
     * rowCount   |==================================================
     * |          |       |     |     ooooooooooooooooooooo
     * |          |       |     |     ooo HERE oooooooooooo
     * |          |       |row_end    oooooooo EL ooooooooo
     * |          .       .      .    ooooooooooooooooooooo
     * |          .       .      .      .
     * |          .       .      .      .      xxxxxxxxxx
     * v          |___....______________|      xxx EL xxx
     *                                         xxxxxxxxxx
     *                                         xxxxxxxxxx
     *
     *                                     oooooooooooo
     *                                     ooo El ooooo
     *                                     oooooooooooo
     *
     */
#ifndef NDEBUG
    PR = 1;
#endif
    PRLEVEL(1, ("%%Inside pivotal_elements\n"));
    for (int64_t e : pivotal_elements)
    {
        paru_element *el = elementList[e];
        //Found this in coverage test
        //It seems that I keep pivotal_elements really clean before this
        //if (el == NULL)
        //{  // removing the  element from the list
        //    PRLEVEL(1, ("%% eli = " LD ", element= " LD "  \n", eli, e));
        //    continue;
        //}
        //This next lines are also extra; I didn't have resize after them
        //There is no NULL inside pivotal_elements here.
        // keeping other elements inside the list
        //pivotal_elements[ii++] = pivotal_elements[i];
        
        ASSERT(el != NULL);

#ifndef NDEBUG
        PRLEVEL(PR, ("%% pivotal element= " LD " lac=" LD " colsleft=" LD " \n", e,
                     el->lac, el->ncolsleft));
        if (PR <= 0) paru_print_element(e, Work, Num);
#endif
        int64_t intsct = paru_intersection(e, elementList, stl_newColSet);
        if (el->cValid < pMark)
        {  // first time seeing this element in this front
            elCol[e] = el->ncolsleft - intsct;  // initiaze
        }
        else if (el->cValid != npMark)
        {  // it has been seen
            elCol[e] -= intsct;
        }

        el->cValid = npMark;
    }

#ifndef NDEBUG
    PR = 1;
#endif

    /****** travers over new non pivotal rows of current front *****************
     *          and update the number of rows can be seen in each element.
     *
     *                <----------fp--------->
     *                        j1     j2
     *                         ^     ^
     *                         |     | stl_colSet
     *                         |     |        \       stl_newColSet
     *             F           |     |         [QQQQQ|OOOOOOOOO|....
     *              \  ____..._|_  ____...__ ____________________________...
     * ^              |\      |     |       #  ^     |         |
     * |              | \     |     |       #  | old | added   |
     * |              |  \    |     |       #  | list|  columns|
     * |              |___\...|_____|__...__#  |     |         |
     * |   ^          |       |\* **|       #  fp              |
     * |   |          |       |**\**|       #  |               |
     * | panel        |       |***\*|       #  |               |
     * | width        |       |***\*|       #  |               |
     * |   |          |       |***\*|       #  |               |
     * |   v          |____...|________..._ #  |      vvvvvv   |
     * |        j2--> |       |     |       #  v    00000000   |         ...
     * rowCount       |=============================00 El 00=============
     * |              |       |     |       |       00000000
     * |              |       |     |       |          vvvvvvvvv
     * |          H   |       |     |       |          xxxxxxxxxxxxxxxxxxxx
     * |          E   |       |row_end      |          xxxxxx El xxxxxxxxxx
     * |          R   .       .      .      .          xxxxxxxxxxxxxxxxxxxx
     * |          E   .       .      .      .
     * |              .       .      .      .
     * v  rowCount--->|___....______________|
     *
     */

    if (npMark == pMark + 1)  // just once for each front
    {                         // in the first time calling this function
        PRLEVEL(1, ("UPDATING elRow\n"));
        for (int64_t k = j2; k < rowCount; k++)
        {
            int64_t r = frowList[k];
            paru_tupleList *curRowTupleList = &RowList[r];
            int64_t numTuple = curRowTupleList->numTuple;
            ASSERT(numTuple >= 0);
            paru_tuple *listRowTuples = curRowTupleList->list;
#ifndef NDEBUG
            int64_t PR = 1;
            PRLEVEL(PR, ("\n %%----r =" LD "  numTuple = " LD "\n", r, numTuple));
            if (PR <= 0) paru_print_paru_tupleList(RowList, r);
#endif
            int64_t pdst = 0, psrc;
            for (psrc = 0; psrc < numTuple; psrc++)
            {
                paru_tuple curTpl = listRowTuples[psrc];
                int64_t e = curTpl.e;

#ifndef NDEBUG
                if (PR <= 0) paru_print_element(e, Work, Num);
#endif
                int64_t curRowIndex = curTpl.f;

                if (e < 0 || curRowIndex < 0) continue;

                paru_element *el = elementList[e];
                if (el == NULL) continue;

                // int64_t *el_rowIndex = rowIndex_pointer (el);
                int64_t *el_rowIndex = (int64_t *)(el + 1) + el->ncols;

                if (el_rowIndex[curRowIndex] < 0) continue;

                listRowTuples[pdst++] = curTpl;  // keeping the tuple

                if (el->rValid == pMark)
                {  // already a pivot and wont change the row degree
                    elRow[e]--;
                    PRLEVEL(1, ("%% Pivotal elRow[" LD "]=" LD " \n", e, elRow[e]));
                }
                else if (el->rValid != npMark)
                {
                    el->rValid = npMark;
                    elRow[e] = el->nrowsleft - 1;  // initiaze
                    PRLEVEL(1, ("%%rValid=" LD " \n", el->rValid));
                    PRLEVEL(1, ("%%NP: first time seen elRow[" LD "]=" LD " \n", e,
                                elRow[e]));
                }
                else
                {  // el->rValid == npMark //it has been seen in this stage
                    elRow[e]--;
                    PRLEVEL(1, ("%%seen before: elRow[e]=" LD " \n", elRow[e]));
                }
            }

            curRowTupleList->numTuple = pdst;
        }
    }

    /**** Travers over new non pivotal rows of current panel
     *    and update the row degree. Once at the begining it updates elRow and
     *    then if elRow == 0 it compute the intersection. It might have been
     *    computed or we might need to compute it now. However if it is
     *    computed now it would be mark not to recompute
     *
     *                <----------fp--------->
     *                        j1     j2
     *                         ^     ^
     *                         |     | stl_colSet
     *                         |     |        \       stl_newColSet
     *             F           |     |         [QQQQQ|OOOOOOOOO|....
     *              \  ____..._|_  ____...__ ____________________________...
     * ^              |\      |     |       #  ^     |         |
     * |              | \     |     |       #  | old | added   |
     * |              |  \    |     |       #  | list|  columns|
     * |              |___\...|_____|__...__#  |     |         |
     * |   ^          |       |\* **|       #  fp              |
     * |   |          |       |**\**|       #  |               |
     * | panel        |       |***\*|       #  |               |
     * | width        |       |***\*|       #  |               |
     * |   |          |       |***\*|       #  |               |
     * |   v          |____...|________..._ #  |      vvvvvv   |
     * |        j2--> |       |     |       #  v    00000000   |         ...
     * rowCount   H   |=============================00 El 00=============
     * |          E   |       |     |       |       00000000
     * |          R   |       |     |       |          vvvvvvvvv
     * |          E   |       |     |       |          xxxxxxxxxxxxxxxxxxxx
     * |  row_end --->|       |row_end      |          xxxxxx El xxxxxxxxxx
     * |              .       .      .      .          xxxxxxxxxxxxxxxxxxxx
     * |              .       .      .      .
     * |              .       .      .      .
     * v  rowCount--->|___....______________|
     *
     */

    int64_t new_row_degree_bound_for_r;

    for (int64_t k = j2; k < row_end; k++)
    {
        int64_t r = frowList[k];

        new_row_degree_bound_for_r = colCount;

        paru_tupleList *curRowTupleList = &RowList[r];
        int64_t numTuple = curRowTupleList->numTuple;
        ASSERT(numTuple >= 0);
        paru_tuple *listRowTuples = curRowTupleList->list;
#ifndef NDEBUG
        int64_t PR = 1;
        PRLEVEL(PR,
                ("\n %%--------> 2nd r =" LD "  numTuple = " LD "\n", r, numTuple));
        if (PR <= 0) paru_print_paru_tupleList(RowList, r);
#endif

        int64_t pdst = 0, psrc;
        for (psrc = 0; psrc < numTuple; psrc++)
        {
            paru_tuple curTpl = listRowTuples[psrc];
            int64_t e = curTpl.e;

#ifndef NDEBUG
            if (PR <= 0) paru_print_element(e, Work, Num);
#endif
            int64_t curRowIndex = curTpl.f;

            paru_element *el = elementList[e];
            // ASSERT (el != NULL);
            if (el == NULL) continue;

            // int64_t *el_rowIndex = rowIndex_pointer (el);
            int64_t *el_rowIndex = (int64_t *)(el + 1) + el->ncols;

            if (el_rowIndex[curRowIndex] < 0) continue;

            listRowTuples[pdst++] = curTpl;  // keeping the tuple

            if (el->rValid == pMark)
            {  // already a pivot and wont change the row degree
                PRLEVEL(1, ("%% Pivotal elRow[" LD "]=" LD " \n", e, elRow[e]));
                continue;
            }

            if (elRow[e] != 0)
            {                            // use the upperbound
                if (el->cValid < pMark)  // never seen
                    new_row_degree_bound_for_r += el->ncolsleft;
                else  // tighter upperbound
                    new_row_degree_bound_for_r += elCol[e];
                continue;
            }

            if (el->cValid < pMark)
            {  // first time seeing this element in this front
                el->cValid = npMark;
                int64_t intsct = paru_intersection(e, elementList, stl_newColSet);
                elCol[e] = el->ncolsleft - intsct;  // initiaze
            }
            else if (el->cValid != npMark)
            {  // it has been seen
                el->cValid = npMark;
                if (elCol[e] != 0)
                {
                    int64_t intsct = paru_intersection(e, elementList, stl_newColSet);
                    elCol[e] -= intsct;
                }
            }
            new_row_degree_bound_for_r += elCol[e];

            PRLEVEL(1, ("%% pMark=" LD " npMark=" LD " \n", pMark, npMark));
        }
        curRowTupleList->numTuple = pdst;

        int64_t old_bound_updated = row_degree_bound[r] + colCount - 1;

#ifndef NDEBUG
        PR = 1;
        PRLEVEL(PR, ("%%old_bound_updated =" LD " \n", old_bound_updated));
        PRLEVEL(PR, ("%%new_row_degree_bound_for_r=" LD " \n",
                     new_row_degree_bound_for_r));
        PRLEVEL(PR, ("%%row_degroo_bound[" LD "]=" LD " \n", r, row_degree_bound[r]));
#endif

        row_degree_bound[r] =  // min
            old_bound_updated < new_row_degree_bound_for_r
                ? old_bound_updated
                : new_row_degree_bound_for_r;
    }

    Work->time_stamp[f] += 2;  // making all the markings invalid again
#ifndef NDEBUG
    PRLEVEL(1, ("%% Finalized counters r1=" LD " r2=" LD " r3=" LD " sum=" LD "\n", r1, r2,
                r3, r1 + r2 + r3));
#endif
}
