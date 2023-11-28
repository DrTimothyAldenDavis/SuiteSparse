////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_print   ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief printing datas are implemented in this file
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"
void paru_print_element(int64_t e, paru_work *Work, ParU_Numeric *Num)
{
    // print out contribution blocks
    paru_element **elementList;
    elementList = Work->elementList;
    paru_element *curEl = elementList[e];

    int64_t morign = Num->m;
    int64_t nf = Work->Sym->nf;

    if (e > morign + nf + 1)
    {
        printf("%% paru_element " LD " is out of range; just " LD " elements \n", e,
               morign + nf + 1);
        return;
    }

    if (curEl == NULL)
    {
        printf("%% paru_element " LD " is empty\n", e);
        return;
    }

    int64_t m, n;
    m = curEl->nrows;
    n = curEl->ncols;

    int64_t *el_colIndex = colIndex_pointer(curEl);
    int64_t *el_rowIndex = rowIndex_pointer(curEl);

    // int64_t *rel_col = relColInd (curEl);
    // int64_t *rel_row = relRowInd (curEl);

    double *el_colrowNum = numeric_pointer(curEl);

    printf("\n");
    printf("%% paru_element " LD " is " LD " x " LD ":\n", e, m, n);

    printf("\t");
    //    for (int j = 0; j < n; j++)
    //        printf("%% " LD "\t", rel_col[j] );
    //    printf("\n\t");
    for (int j = 0; j < n; j++) printf("%% " LD "\t", el_colIndex[j]);

    printf("\n");
    for (int i = 0; i < m; i++)
    {
        //     printf("%% " LD "\t " LD "\t",rel_row[i], el_rowIndex [i] );
        printf("%% " LD "\t", el_rowIndex[i]);
        for (int j = 0; j < n; j++)
        {
            double value = el_colrowNum[j * m + i];
            printf("%2.4lf\t", value);
        }
        printf("\n");
    }
}

void paru_print_paru_tupleList(paru_tupleList *listSet, int64_t index)
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% listSet =%p\n", listSet));

    if (listSet == NULL)
    {
        printf("%% Empty tuple\n");
        return;
    }

    paru_tupleList cur = listSet[index];
    int64_t numTuple = cur.numTuple;
    paru_tuple *l = cur.list;

    printf("%% There are " LD " tuples in this list:\n %%", numTuple);
    for (int64_t i = 0; i < numTuple; i++)
    {
        paru_tuple curTpl = l[i];
        printf(" (" LD "," LD ")", curTpl.e, curTpl.f);
    }
    printf("\n");
}
