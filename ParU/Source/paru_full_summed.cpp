////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_full_summed ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  fully sum the pivotal column from a prior contribution block e
 *  assembling starts with el->lac and continues until no assembly is possible
 *  in curent front. If the prior contribution block is empty free it here.
 *
 *
 ***************  assembling the pivotal part of the front
 *
 *      el           nEl
 *                  6, 7, 11, 12
 *                 _____________
 *              23 | X  Y  .  .     stored in memory like this:
 *          mEl 17 | X  Y  .  .     ..6, 7,11, 12, 23, 17, 2, X, X, X, Y, Y, Y,
 *               2 | X  Y  .  .
 *
 *        It must be assembled in current pivotal fron like this:
 *                                         fp
 *                                     col1, ... , col
 *
 *                                      6, 7, 8, 9, 10
 *                                      ______________
 *                              0   23 | X  Y  .  .  .
 *                   rowCount   1    2 | X  Y  .  .  .
 *                              2    4 | *  *  .  .  .  isRowInFront[4] == 2
 *                              3   17 | X  Y  .  .  .
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"

void paru_full_summed(int64_t e, int64_t f, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
    ParU_Symbolic *Sym = Work->Sym;
#ifndef NDEBUG
    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];
    PRLEVEL(PR, ("%% Fully summing " LD " in " LD "(" LD ")\n", e, f, eli));
#endif

    int64_t *Super = Sym->Super;
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t col2 = Super[f + 1];
    PRLEVEL(PR, ("%% col1=" LD ", col2=" LD "\n", col1, col2));

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    // int64_t *el_colIndex = colIndex_pointer (el);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *rowRelIndex = relRowInd (el);
    int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

    // int64_t *el_rowIndex = rowIndex_pointer (el);
    int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

    ParU_Factors *LUs = Num->partial_LUs;
    int64_t rowCount = LUs[f].m;
    double *pivotalFront = LUs[f].p;

    // double *el_Num = numeric_pointer (el);
    double *el_Num = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);

#ifndef NDEBUG  // print the element which is going to be assembled from
    PR = 2;
    PRLEVEL(PR, ("%% ASSEMBL element= " LD "  mEl =" LD " ", e, mEl));
    if (PR <= 0) paru_print_element(e, Work, Num);
#endif

    int64_t j = el->lac;  // keep record of latest lac
    if (el->ncolsleft == 1)
    // I could add the following condition too:
    //|| next active column is out of current pivotal columns
    // However it requires searching through columns; I will just waste a
    // temp space
    {  // No need for a temp space for active rows; no reuse
        PRLEVEL(PR, ("%% 1 col left\n %%"));
        double *sC = el_Num + mEl * el->lac;  // source column pointer
        int64_t fcolInd = el_colIndex[el->lac] - col1;
#ifndef NDEBUG
        int64_t colInd = el_colIndex[el->lac];
        PRLEVEL(1, ("%% colInd =" LD " \n", fcolInd));
        ASSERT(colInd >= 0);
#endif
        double *dC = pivotalFront + fcolInd * rowCount;
        int64_t nrows2bSeen = el->nrowsleft;
        for (int64_t i = 0; i < mEl; i++)
        {
            int64_t rowInd = el_rowIndex[i];
            PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
            if (rowInd >= 0 && rowRelIndex[i] != -1)
            {  // active and do not contain zero in pivot
                int64_t ri = rowRelIndex[i];
                PRLEVEL(1, ("%% ri = " LD " \n", ri));
                PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                dC[ri] += sC[i];
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
                el_colIndex[el->lac] = flip(el_colIndex[el->lac]);
                if (--nrows2bSeen == 0) break;
            }
        }
        el->ncolsleft--;
    }
    else
    {
        PRLEVEL(PR, ("%% more than 1 col left\n %%"));

        // save the structure of the rows once at first
        int64_t nrows2assembl = el->nrowsleft - el->nzr_pc;
        // int64_t tempRow[nrows2assembl];  // C99
        std::vector<int64_t> tempRow(nrows2assembl);
        int64_t ii = 0;
        for (int64_t i = 0; i < mEl; i++)
        {
            int64_t rowInd = el_rowIndex[i];
            PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
            if (rowInd >= 0 && rowRelIndex[i] != -1)
            {
                tempRow[ii++] = i;
                if (ii == nrows2assembl) break;
            }
        }

#ifndef NDEBUG
        PR = 1;
        PRLEVEL(PR, ("%% list of the rows to be assembled:\n%%"));
        for (int64_t i = 0; i < nrows2assembl; i++)
            PRLEVEL(PR, ("" LD " ", el_rowIndex[tempRow[i]]));
        PRLEVEL(PR, ("%% \n"));
#endif
        // note: parallelism slows this down
        // int64_t *Depth = Sym->Depth;
        //#pragma omp parallel
        //#pragma omp single
        //#pragma omp taskgroup
        for (; j < nEl; j++)
        {  // j already defined out of this scope while it is needed
            PRLEVEL(1, ("%% j =" LD " \n", j));
            double *sC = el_Num + mEl * j;  // source column pointer
            int64_t colInd = el_colIndex[j];
            PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
            if (colInd >= col2) break;
            if (colInd < 0) continue;

            int64_t fcolInd = colInd - col1;

            double *dC = pivotalFront + fcolInd * rowCount;

            //#pragma omp task priority(Depth[f]) if(nrows2assembl > 1024)
            for (int64_t iii = 0; iii < nrows2assembl; iii++)
            {
                int64_t i = tempRow[iii];
                int64_t ri = rowRelIndex[i];

                ASSERT(rowRelIndex[i] != -1);  // I already picked the rows
                // that are not in zero pivots
                ASSERT(el_rowIndex[i] >= 0);  // and also still alive

                PRLEVEL(1, ("%% ri = " LD " \n", ri));
                PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                dC[ri] += sC[i];
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
            }

            el_colIndex[j] = flip(el_colIndex[j]);
            if (--el->ncolsleft == 0) break;
            PRLEVEL(1, ("\n"));
        }
    }

    if (el->ncolsleft == 0)
    {  // free el
        PRLEVEL(PR, ("%% element " LD " is freed after pivotal assembly\n", e));
        paru_free_el(e, elementList);
    }

    if (elementList[e] != NULL)
    {
        el->lac = j;
        int64_t *lacList = Work->lacList;
        lacList[e] = lac_el(elementList, e);
        PRLEVEL(1, ("%%e = " LD ", el->lac= " LD " ", e, el->lac));
        PRLEVEL(1, ("el_colIndex[" LD "]=" LD " :\n", el->lac, el_colIndex[el->lac]));
        ASSERT(j < nEl);
    }
#ifndef NDEBUG  // print the element which has been assembled from
    PR = 1;
    PRLEVEL(PR, ("%% ASSEMBLED element= " LD "  mEl =" LD " ", e, mEl));
    if (PR <= 0) paru_print_element(e, Work, Num);

    // Printing the pivotal front
    PR = 2;
    PRLEVEL(PR, ("%% After Assemble element " LD "\n", e));
    PRLEVEL(PR, ("%% x =  \t"));
    for (int64_t c = col1; c < col2; c++) PRLEVEL(PR, ("" LD "\t\t", c));
    PRLEVEL(PR, (" ;\n"));

    int64_t *frowList = Num->frowList[f];
    for (int64_t r = 0; r < rowCount; r++)
    {
        PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
        for (int64_t c = col1; c < col2; c++)
            PRLEVEL(PR, (" %2.5lf\t", pivotalFront[(c - col1) * rowCount + r]));
        PRLEVEL(PR, ("\n"));
    }
#endif
}
