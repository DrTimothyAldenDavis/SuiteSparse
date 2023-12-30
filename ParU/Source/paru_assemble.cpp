////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_assemble //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief  finding the  columns or rows of prior element and fully or partially
 * assemble it  and eliminate it if needed
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"

void paru_assemble_all(int64_t e, int64_t f, std::vector<int64_t> &colHash,
        paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
#ifndef NTIME
    static double tot_assem_time = 0;
    double start_time = PARU_OPENMP_GET_WTIME;
#endif

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];
    PRLEVEL(PR, ("%% Eliminate all of " LD " in " LD "(f=" LD ") (tid=%d)\n", e, eli, f,
                 PARU_OPENMP_GET_THREAD_ID));

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);
    PR = 1;
#endif

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];
    paru_element *curEl = elementList[eli];

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    // int64_t *el_colIndex = colIndex_pointer (el);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *rowRelIndex = relRowInd (el);
    int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

    if (el->cValid != Work->time_stamp[f])
        paru_update_rel_ind_col(e, f, colHash, Work, Num);

    // int64_t *colRelIndex = relColInd (paru_element *el);
    int64_t *colRelIndex = (int64_t *)(el + 1) + mEl + nEl;

    // int64_t *el_rowIndex = rowIndex_pointer (el);
    int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

    // double *el_Num = numeric_pointer (el);
    double *el_Num = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);
    // current elemnt numerical pointer
    // double *el_Num = numeric_pointer (curEl);
    double *curEl_Num =
        (double *)((int64_t *)(curEl + 1) + 2 * curEl->nrows + 2 * curEl->ncols);

    int64_t *isRowInFront = Work->rowSize;

#ifndef NDEBUG
    ParU_Factors *Us = Num->partial_Us;
    int64_t *fcolList = Num->fcolList[f];
    int64_t colCount = Us[f].n;
    ASSERT(el_colIndex[el->lac] <= fcolList[colCount - 1]);
    ASSERT(el_colIndex[nEl - 1] <= 0 || fcolList[0] <= el_colIndex[nEl - 1]);
    PRLEVEL(PR, ("%% newColSet.size = " LD "\n", colCount));
    PRLEVEL(PR, ("%% nEl = " LD "\n", nEl));
#endif

    if (el->ncolsleft == 1)
    {
        PRLEVEL(PR, ("%% 1 col left\n %%"));
        double *sC = el_Num + mEl * el->lac;  // source column pointer
#ifndef NDEBUG
        int64_t colInd = el_colIndex[el->lac];
        PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
        ASSERT(colInd >= 0);
#endif
        int64_t fcolind = colRelIndex[el->lac];
        double *dC = curEl_Num + fcolind * curEl->nrows;
        int64_t nrowsSeen = el->nrowsleft;
        for (int64_t i = 0; i < mEl; i++)
        {
            int64_t rowInd = el_rowIndex[i];
            PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
            if (rowInd >= 0)
            {
                int64_t ri = isRowInFront[rowInd];
                PRLEVEL(1, ("%% ri = " LD " \n", ri));
                PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                dC[ri] += sC[i];
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
                if (--nrowsSeen == 0) break;
            }
        }
    }
    else
    {
        PRLEVEL(PR, ("%% more than 1 col left (" LD "->" LD ")\n %%", e, eli));

        // save the structure of the rows once at first
        // int64_t tempRow[el->nrowsleft];  // C99
        std::vector<int64_t> tempRow(el->nrowsleft);
        int64_t ii = 0;
        for (int64_t i = 0; i < mEl; i++)
        {
            int64_t rowInd = el_rowIndex[i];
            PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
            if (rowInd >= 0)
            {
                tempRow[ii++] = i;
                int64_t ri = isRowInFront[rowInd];
                rowRelIndex[i] = ri;
                if (ii == el->nrowsleft) break;
            }
        }

        int64_t naft;  // number of active frontal tasks
        #pragma omp atomic read
        naft = Work->naft;
        ParU_Control *Control = Num->Control;
        const int32_t max_threads = Control->paru_max_threads;

        if (el->nrowsleft * el->ncolsleft < 4096 || el->nrowsleft < 1024
            #ifndef PARU_COVERAGE
            // In production, do sequential assembly if the number
            // of active fronts is large.  For test coverage, don't
            // check this condition, to exercise the parallel assembly.
            || naft > max_threads / 2
            #endif
            )
        {  // not enoght resources or very small assembly
            // sequential
            PRLEVEL(1,
                    ("Seqntial Assembly naft=" LD " colsleft=" LD " rowsleft=" LD " \n",
                     naft, el->ncolsleft, el->nrowsleft));

            for (int64_t j = el->lac; j < nEl; j++)
            {
                PRLEVEL(1, ("%% j =" LD " \n", j));
                double *sC = el_Num + mEl * j;  // source column pointer
                int64_t colInd = el_colIndex[j];
                PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
                if (colInd < 0) continue;
                int64_t fcolind = colRelIndex[j];

                double *dC = curEl_Num + fcolind * curEl->nrows;
                for (int64_t iii = 0; iii < el->nrowsleft; iii++)
                {
                    int64_t i = tempRow[iii];
                    int64_t ri = rowRelIndex[i];

                    PRLEVEL(1, ("%% ri = " LD " \n", ri));
                    PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                    PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                    dC[ri] += sC[i];
                    PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
                }

                if (--el->ncolsleft == 0) break;
                PRLEVEL(1, ("\n"));
            }
        }
        else
        {
            // enoght threads and big assembly
            // go parallel
            PRLEVEL(1, ("Parallel Assembly naft=" LD " colsleft=" LD " rowsleft=" LD " "
                        "el->lac = " LD " nEl=" LD " rem =" LD " (" LD "->" LD ")\n",
                        naft, el->ncolsleft, el->nrowsleft, el->lac, nEl,
                        nEl - el->lac, e, eli));

            // // each column a tsk
            //#..pragma omp parallel proc_bind(close)
            // num_threads(max_threads / naft)
            //#..pragma omp single nowait
            //#..pragma omp task untied
            // for (int64_t j = el->lac; j < nEl; j++)
            //{
            //    PRLEVEL(1, ("%% j =" LD " \n", j));
            //    double *sC = el_Num + mEl * j;  // source column pointer
            //    int64_t colInd = el_colIndex[j];
            //    PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
            //    if (colInd < 0) continue;

            //    int64_t fcolind = colRelIndex[j];

            //    double *dC = curEl_Num + fcolind * curEl->nrows;

            //    #pragma omp task
            //    for (int64_t iii = 0; iii < el->nrowsleft; iii++)
            //    {
            //        int64_t i = tempRow[iii];
            //        int64_t ri = rowRelIndex[i];

            //        PRLEVEL(1, ("%% ri = " LD " \n", ri));
            //        PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
            //        PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
            //        dC[ri] += sC[i];
            //        PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
            //    }
            //    if (--el->ncolsleft == 0) break;
            //    PRLEVEL(1, ("\n"));
            //}

            ///////////////////////////////////////////////////////////////////
            /////////////// making tasks and such /////////////////////////////
            ///////////////////////////////////////////////////////////////////

            int64_t ntasks = (max_threads - naft + 1) * 2;
            ntasks = (ntasks <= 0) ? 1 : ntasks;
            int64_t task_size = (nEl - el->lac) / ntasks;
            PRLEVEL(1, ("BBB el->lac=" LD " nEl=" LD " ntasks=" LD " task_size=" LD "\n",
                        el->lac, nEl, ntasks, task_size));
            if (task_size == 0 || task_size == 1)
            {
                task_size = 1;
                ntasks = nEl - el->lac;
            }
            PRLEVEL(1, ("el->lac=" LD " nEl=" LD " ntasks=" LD " task_size=" LD "\n",
                        el->lac, nEl, ntasks, task_size));
            #pragma omp parallel proc_bind(close) num_threads(ntasks)
            #pragma omp single
            #pragma omp task
            for (int64_t t = 0; t < ntasks; t++)
            {
                int64_t c1 = el->lac + t * task_size;
                int64_t c2 = el->lac + (t + 1) * task_size;
                c2 = t == ntasks - 1 ? nEl : c2;
                PRLEVEL(1, ("t=" LD " c1=" LD " c2=" LD "\n", t, c1, c2));
                #pragma omp task mergeable
                for (int64_t j = c1; j < c2; j++)
                {
                    PRLEVEL(1, ("%% j =" LD " t=" LD "\n", j, t));
                    double *sC = el_Num + mEl * j;  // source column pointer
                    int64_t colInd = el_colIndex[j];
                    PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
                    if (colInd < 0) continue;
                    PRLEVEL(1, ("inside paralle region %d j=" LD " (tid=%d)\n",
                                PARU_OPENMP_GET_ACTIVE_LEVEL, j,
                                PARU_OPENMP_GET_THREAD_NUM));
                    int64_t fcolind = colRelIndex[j];

                    double *dC = curEl_Num + fcolind * curEl->nrows;

                    for (int64_t iii = 0; iii < el->nrowsleft; iii++)
                    {
                        int64_t i = tempRow[iii];
                        int64_t ri = rowRelIndex[i];

                        PRLEVEL(1, ("%% ri = " LD " \n", ri));
                        PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                        PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                        dC[ri] += sC[i];
                        PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
                    }
                    PRLEVEL(1, ("\n"));
                }
            }
        }
    }
    paru_free_el(e, elementList);
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% after assembly " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);
    PR = 1;
#endif

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    #pragma omp atomic update
    tot_assem_time += time;
    if (f > Sym->nf - 5)
        PRLEVEL(-1, ("%% assemble all " LD "\t->" LD "\t took %lf seconds tot=%lf\n",
                     e, eli, time, tot_assem_time));
#endif
}

// try to find columns and assemble them to current front. After the first
// column that is not in current front it gets a toll for each column doesn't
// fit

void paru_assemble_cols(int64_t e, int64_t f, std::vector<int64_t> &colHash,
        paru_work *Work, ParU_Numeric *Num)

{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
#ifndef NDEBUG
    int64_t c = 0;  // number of columns assembled
#endif
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];

    PRLEVEL(PR, ("%% Eliminat some cols of " LD " in " LD "\n", e, eli));
#ifndef NDEBUG
    PR = 1;

    PRLEVEL(PR, ("%% " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);
#endif

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];
    paru_element *curEl = elementList[eli];

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    // int64_t *el_colIndex = colIndex_pointer (el);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *rowRelIndex = relRowInd (el);
    int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

    // int64_t *el_rowIndex = rowIndex_pointer (el);
    int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

    // double *el_Num = numeric_pointer (el);
    double *el_Num = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);
    // current elemnt numerical pointer
    // double *el_Num = numeric_pointer (curEl);
    double *curEl_Num =
        (double *)((int64_t *)(curEl + 1) + 2 * curEl->nrows + 2 * curEl->ncols);

    int64_t *isRowInFront = Work->rowSize;

    int64_t *fcolList = Num->fcolList[f];

    // int64_t tempRow[el->nrowsleft];  // C99
    std::vector<int64_t> tempRow(el->nrowsleft);
    int64_t tempRow_ready = 0;
    int64_t toll = 8;  // number of times it continue when do not find anything

    // int64_t naft; //number of active frontal tasks
    // pragma omp atomic read
    // naft = Num->naft;
    // const int32_t max_threads = Num->paru_max_threads;
    ////int64_t *Depth = Sym->Depth;
    // pragma omp parallel proc_bind(close) num_threads(max_threads/naft)
    // if (naft < max_threads/2 &&
    //        el->nrowsleft*el->ncolsleft < 4096 && el->nrowsleft < 1024 )
    // pragma omp single nowait
    // pragma omp task untied

    // TOLL FREE zone
    while (paru_find_hash(el_colIndex[el->lac], colHash, fcolList) != -1)
    {
        PRLEVEL(PR, ("%% Toll free\n"));
        if (tempRow_ready == 0)
        {
            // save the structure of the rows once at first
            int64_t ii = 0;
            for (int64_t i = 0; i < mEl; i++)
            {
                int64_t rowInd = el_rowIndex[i];
                PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
                if (rowInd >= 0)
                {
                    tempRow[ii++] = i;
                    int64_t ri = isRowInFront[rowInd];
                    rowRelIndex[i] = ri;
                    if (ii == el->nrowsleft) break;
                }
            }
            tempRow_ready = 1;
        }

        int64_t colInd = el_colIndex[el->lac];
        int64_t fcolind = paru_find_hash(colInd, colHash, fcolList);

        PRLEVEL(1, ("%% el->lac =" LD " \n", el->lac));
        double *sC = el_Num + mEl * el->lac;  // source column pointer
        PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
        ASSERT(colInd >= 0);

        double *dC = curEl_Num + fcolind * curEl->nrows;

        // pragma omp task
        for (int64_t ii = 0; ii < el->nrowsleft; ii++)
        {
            int64_t i = tempRow[ii];
            int64_t ri = rowRelIndex[i];

            PRLEVEL(1, ("%% ri = " LD " \n", ri));
            PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
            dC[ri] += sC[i];
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
        }
#ifndef NDEBUG
        c++;
#endif
        el_colIndex[el->lac] = flip(el_colIndex[el->lac]);
        if (--el->ncolsleft == 0) break;
        while (el_colIndex[++el->lac] < 0 && el->lac < el->ncols)
            ;
    }
    // el->lac won't get updated after this
    int64_t *lacList = Work->lacList;
    lacList[e] = el_colIndex[el->lac];

    // TOLL Zone
    //**//pragma omp parallel
    //**//pragma omp single nowait
    //**//pragma omp taskgroup
    for (int64_t j = el->lac + 1; j < nEl && el->ncolsleft > 0 && toll > 0; j++)
    {
        PRLEVEL(1, ("%% Toll zone\n"));
        toll--;
        if (tempRow_ready == 0)
        {
            // save the structure of the rows once at first
            int64_t ii = 0;
            for (int64_t i = 0; i < mEl; i++)
            {
                int64_t rowInd = el_rowIndex[i];
                PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
                if (rowInd >= 0)
                {
                    tempRow[ii++] = i;
                    int64_t ri = isRowInFront[rowInd];
                    rowRelIndex[i] = ri;
                    if (ii == el->nrowsleft) break;
                }
            }
            tempRow_ready = 1;
        }

        PRLEVEL(1, ("%% j =" LD " \n", j));
        double *sC = el_Num + mEl * j;  // source column pointer
        int64_t colInd = el_colIndex[j];
        PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
        if (colInd < 0) continue;
        int64_t fcolind = paru_find_hash(colInd, colHash, fcolList);
        if (fcolind == -1) continue;
        toll++;  // if found
        double *dC = curEl_Num + fcolind * curEl->nrows;

        //**//pragma omp task priority(Depth[f]) if(el->nrowsleft > 1024)
        for (int64_t ii = 0; ii < el->nrowsleft; ii++)
        {
            int64_t i = tempRow[ii];
            int64_t ri = rowRelIndex[i];

            PRLEVEL(1, ("%% ri = " LD " \n", ri));
            PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
            dC[ri] += sC[i];
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
        }
#ifndef NDEBUG
        c++;
#endif
        el_colIndex[j] = flip(el_colIndex[j]);
        if (--el->ncolsleft == 0) break;
    }

#ifndef NDEBUG
    PRLEVEL(1, ("%%  " LD " has found and assembled, ncolsleft " LD "\n", c,
                el->ncolsleft));
#endif

    if (el->ncolsleft == 0)
    {
        paru_free_el(e, elementList);
    }
}

void paru_assemble_rows(int64_t e, int64_t f, std::vector<int64_t> &colHash,
        paru_work *Work, ParU_Numeric *Num)

{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];

    PRLEVEL(PR, ("%% Eliminat some rows of " LD " in " LD "\n", e, eli));

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];
    paru_element *curEl = elementList[eli];

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    // int64_t *el_colIndex = colIndex_pointer (el);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *rowRelIndex = relRowInd (el);
    int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

    // int64_t *colRelIndex = relColInd (paru_element *el);
    int64_t *colRelIndex = (int64_t *)(el + 1) + mEl + nEl;

    // int64_t *el_rowIndex = rowIndex_pointer (el);
    int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

    // int64_t *el_rowIndex = rowIndex_pointer (curEl);
    int64_t *curEl_rowIndex = (int64_t *)(curEl + 1) + curEl->ncols;

    // double *el_Num = numeric_pointer (el);
    double *el_Num = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);
    // current elemnt numerical pointer
    // double *el_Num = numeric_pointer (curEl);
    double *curEl_Num =
        (double *)((int64_t *)(curEl + 1) + 2 * curEl->nrows + 2 * curEl->ncols);

    int64_t *isRowInFront = Work->rowSize;

    std::vector<int64_t> tempRow;

    // searching for rows
    int64_t i = 0;
    int64_t nrowsSeen = el->nrowsleft;
    // Toll free zone
    PRLEVEL(1, ("%% Toll free\n"));
    while (i < mEl && nrowsSeen > 0)
    {
        for (; el_rowIndex[i] < 0; i++)
            ;
        nrowsSeen--;

        int64_t rowInd = isRowInFront[i];
        if (rowInd > 0 && rowInd < curEl->nrows)
        {
            // coompare their global indices
            if (curEl_rowIndex[rowInd] == el_rowIndex[i])
            {
                PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
                PRLEVEL(1, ("%% curEl_rowIndex[rowInd] =" LD " \n",
                            curEl_rowIndex[rowInd]));
                PRLEVEL(1, ("%% i =" LD " \n", i));
                PRLEVEL(1, ("%% el_rowIndex[i] =" LD " \n", el_rowIndex[i]));
                tempRow.push_back(i);
            }
            else
                break;
        }
        i++;
    }

#ifndef NDEBUG
    if (tempRow.size() > 0)
        PRLEVEL(PR, ("%% Toll free zone: " LD " rows has been found: \n%%",
                     tempRow.size()));
#endif

    PRLEVEL(1, ("%% TollED \n"));
    int64_t toll = 8;  // number of times it continue when do not find anything
    // Toll zone
    while (i < mEl && nrowsSeen > 0 && toll > 0)
    // while (i < mEl  && nrowsSeen >0 )
    {
        for (; el_rowIndex[i] < 0; i++)
            ;
        nrowsSeen--;

        int64_t rowInd = isRowInFront[i];
        if (rowInd > 0 && rowInd < curEl->nrows)
        {
            // coompare their global indices
            if (curEl_rowIndex[rowInd] == el_rowIndex[i])
            {
                PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
                PRLEVEL(1, ("%% curEl_rowIndex[rowInd] =" LD " \n",
                            curEl_rowIndex[rowInd]));
                PRLEVEL(1, ("%% i =" LD " \n", i));
                PRLEVEL(1, ("%% el_rowIndex[i] =" LD " \n", el_rowIndex[i]));

                tempRow.push_back(i);
                toll++;
            }
            else
                toll--;
        }
        i++;
    }

    if (tempRow.empty()) return;

    PRLEVEL(PR,
            ("%% " LD " rows has been found, toll " LD "\n%%", tempRow.size(), toll));
#ifndef NDEBUG
    for (int64_t ii = 0; ii < (int64_t)tempRow.size(); ii++)
        PRLEVEL(PR, ("" LD " ", tempRow[ii]));
    PRLEVEL(PR, ("\n "));
#endif
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% Before eliminiatine some rows " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);
#endif

    //This never happpens I found it in test coverage
    //It is obviouse when I look at the caller
    //if (el->cValid != Work->time_stamp[f])
    //    paru_update_rel_ind_col(e, f, colHash, Work, Num);
    ASSERT(el->cValid == Work->time_stamp[f]);

    int64_t ncolsSeen = nEl;

    for (int64_t j = el->lac; j < nEl; j++)
    {
        PRLEVEL(1, ("%% j =" LD " \n", j));
        double *sC = el_Num + mEl * j;  // source column pointer
        int64_t colInd = el_colIndex[j];
        PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
        if (colInd < 0) continue;
        ncolsSeen--;
        int64_t fcolind = colRelIndex[j];

        PRLEVEL(1, ("%% fcolind=" LD " \n", fcolind));
        double *dC = curEl_Num + fcolind * curEl->nrows;

        for (int64_t i1 : tempRow)
        {
            int64_t rowInd = el_rowIndex[i1];
            int64_t ri = isRowInFront[rowInd];

            PRLEVEL(1, ("%% ri = " LD " \n", ri));
            PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
            dC[ri] += sC[i1];
            PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
        }

        if (ncolsSeen == 0) break;
        PRLEVEL(1, ("\n"));
    }

    // invalidating assembled rows
    for (int64_t i2 : tempRow)
    {
        el_rowIndex[i2] = -1;
        rowRelIndex[i2] = -1;
    }

    el->nrowsleft -= tempRow.size();
    if (el->nrowsleft == 0)
    {
        paru_free_el(e, elementList);
    }
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% After Eliminate some rows " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);
#endif
}

void paru_assemble_el_with0rows(int64_t e, int64_t f, std::vector<int64_t> &colHash,
        paru_work *Work, ParU_Numeric *Num)

{
    // This element contributes to both pivotal rows and pivotal columns
    //  However it has zero rows in current pivotal columns therefore
    //  not all rows are there
    // it can be eliminated partially
    //       ________________________________
    //       |      |                         |
    //       |      |                         |
    //       ___xxxxxxxxxxx____________________
    //       |  xxxxxxxxxxx                   |
    //       |  oxxo|oxoxox                   | <- assemble these rows
    //       |  ooxx|oxoxox                   |  and mark them assembled
    //       |  oooo|oxoxox                   |
    //       ---------------------------------
    //          ooooooxxxxx  --> outsidie the front
    //          ooooooxxxxx
    //
    //
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    ParU_Symbolic *Sym = Work->Sym;
    int64_t *snM = Sym->super2atree;
    int64_t eli = snM[f];
    PRLEVEL(PR, ("%% \n+++++++++++++++++++++++++++++++++++++++\n"));
    PRLEVEL(PR, ("%% Eliminat elment " LD "  with0rows in " LD "\n", e, eli));

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%% " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);

#endif

    paru_element **elementList = Work->elementList;

    paru_element *el = elementList[e];
    paru_element *curEl = elementList[eli];

    int64_t nEl = el->ncols;
    int64_t mEl = el->nrows;

    ASSERT(el->nzr_pc > 0);

    // int64_t *el_colIndex = colIndex_pointer (el);
    int64_t *el_colIndex = (int64_t *)(el + 1);

    // int64_t *rowRelIndex = relRowInd (el);
    int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

    if (el->cValid != Work->time_stamp[f])
        paru_update_rel_ind_col(e, f, colHash, Work, Num);

    // int64_t *colRelIndex = relColInd (paru_element *el);
    int64_t *colRelIndex = (int64_t *)(el + 1) + mEl + nEl;

    // int64_t *el_rowIndex = rowIndex_pointer (el);
    int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

    // double *el_Num = numeric_pointer (el);
    double *el_Num = (double *)((int64_t *)(el + 1) + 2 * nEl + 2 * mEl);
    // current elemnt numerical pointer
    // double *el_Num = numeric_pointer (curEl);
    double *curEl_Num =
        (double *)((int64_t *)(curEl + 1) + 2 * curEl->nrows + 2 * curEl->ncols);

    int64_t *isRowInFront = Work->rowSize;

#ifndef NDEBUG
    ParU_Factors *Us = Num->partial_Us;
    int64_t *fcolList = Num->fcolList[f];
    int64_t colCount = Us[f].n;
    ASSERT(el_colIndex[el->lac] <= fcolList[colCount - 1]);
    ASSERT(el_colIndex[nEl - 1] <= 0 || fcolList[0] <= el_colIndex[nEl - 1]);
    PRLEVEL(PR, ("%% newColSet.size = " LD "\n", colCount));
    PRLEVEL(PR, ("%% nEl = " LD "\n", nEl));
#endif

    if (el->ncolsleft == 1)
    {
        PRLEVEL(PR, ("%% 1 col left\n %%"));
        double *sC = el_Num + mEl * el->lac;  // source column pointer
#ifndef NDEBUG
        int64_t colInd = el_colIndex[el->lac];
        PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
        ASSERT(colInd >= 0);
#endif
        int64_t fcolind = colRelIndex[el->lac];
        double *dC = curEl_Num + fcolind * curEl->nrows;
        int64_t nrows2bSeen = el->nrowsleft;
        for (int64_t i = 0; i < mEl; i++)
        {
            int64_t rowInd = el_rowIndex[i];
            PRLEVEL(1, ("%% rowInd =" LD " \n", rowInd));
            if (rowInd >= 0)
            {
                if (rowRelIndex[i] != -1)  // row with at least one nz
                {
                    int64_t ri = isRowInFront[rowInd];
                    PRLEVEL(1, ("%% ri = " LD " \n", ri));
                    PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                    PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                    dC[ri] += sC[i];
                    PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
                }
                if (--nrows2bSeen == 0) break;
            }
        }
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
            PRLEVEL(1, ("%% rowInd =" LD " ", rowInd));
#ifndef NDEBUG
            if (rowRelIndex[i] == -1) PRLEVEL(1, ("%% row_with0 "));
#endif

            PRLEVEL(1, ("%% \n"));
            if (rowInd >= 0 && rowRelIndex[i] != -1)
            {
                tempRow[ii++] = i;
                int64_t ri = isRowInFront[rowInd];
                rowRelIndex[i] = ri;
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
        int64_t ncols2bSeen = el->ncolsleft;
        // int64_t *Depth = Sym->Depth;
        //**//pragma omp parallel
        //**//pragma omp single nowait
        //**//pragma omp taskgroup
        for (int64_t j = el->lac; j < nEl; j++)
        {
            PRLEVEL(1, ("%% j =" LD " \n", j));
            double *sC = el_Num + mEl * j;  // source column pointer
            int64_t colInd = el_colIndex[j];
            PRLEVEL(1, ("%% colInd =" LD " \n", colInd));
            if (colInd < 0) continue;
            int64_t fcolind = colRelIndex[j];

            double *dC = curEl_Num + fcolind * curEl->nrows;

            //**//pragma omp task priority(Depth[f]) if(nrows2assembl > 1024)
            for (int64_t iii = 0; iii < nrows2assembl; iii++)
            {
                int64_t i = tempRow[iii];
                int64_t ri = rowRelIndex[i];
                ASSERT(rowRelIndex[i] != -1);  // I already picked the rows
                // that are not in zero pivots
                PRLEVEL(1, ("%% ri = " LD " \n", ri));
                PRLEVEL(1, ("%% sC [" LD "] =%2.5lf \n", i, sC[i]));
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", ri, dC[ri]));
                dC[ri] += sC[i];
                PRLEVEL(1, ("%% dC [" LD "] =%2.5lf \n", i, dC[ri]));
            }

            if (--ncols2bSeen == 0) break;
            PRLEVEL(1, ("\n"));
        }
    }

    // Mark rows as assembled and updat lac

    int64_t nrows2bSeen = el->nrowsleft;
    int64_t new_lac = nEl;
    for (int64_t ii = 0; ii < mEl; ii++)
    {
        int64_t rowInd = el_rowIndex[ii];
        if (rowInd < 0) continue;  // already gone

        if (rowRelIndex[ii] == -1)  // row with all zeros in piv
        {                           // update lac
            PRLEVEL(1, ("%%Searching for lac in " LD "\n%%", rowInd));
            PRLEVEL(1, ("%%col=" LD "\n%%", el->lac));
            for (int64_t jj = el->lac; jj < new_lac; jj++)
            // searching for the first nz
            {
                if (el_colIndex[jj] < 0) continue;
                // el [rowInd, jj]
                PRLEVEL(1, ("%% el[" LD "," LD "]=%2.5lf\n%%", rowInd, jj,
                            el_Num[mEl * jj + ii]));
                if (el_Num[mEl * jj + ii] != 0)
                {
                    new_lac = jj;
                    PRLEVEL(1, ("%%Found new-lac in " LD "\n%%", jj));
                    break;
                }
            }
        }
        else  // It was assembled here; mark row as assembled
        {
            el_rowIndex[ii] = -1;
        }
        if (--nrows2bSeen == 0) break;
    }
    // updating lac can have effect on number of columns left
    // I should update number of columns left too

    if (new_lac != el->lac)
    {
        int64_t ncolsleft = 0;
        for (int64_t j = new_lac; j < nEl; j++)
        {
            if (el_colIndex[j] > 0) ncolsleft++;
        }
        PRLEVEL(1, ("%%colsleft was " LD " and now is " LD "\n%%", el->ncolsleft,
                    ncolsleft));
        el->ncolsleft = ncolsleft;
        for (int64_t j = el->lac; j < new_lac; j++)
        {
            if (el_colIndex[j] >= 0) el_colIndex[j] = flip(el_colIndex[j]);
        }
    }

    el->nrowsleft = el->nzr_pc;
    el->lac = new_lac;
    int64_t *lacList = Work->lacList;
    lacList[e] = el_colIndex[el->lac];
#ifndef NDEBUG
    int64_t *Super = Sym->Super;
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t col2 = Super[f + 1];
    PR = 1;
    PRLEVEL(PR, ("%% " LD "(" LD ") " LD "-" LD " :\n", f, eli, col1, col2));
    PRLEVEL(PR, ("%%Finally new-lac is " LD " ", el->lac));
    PRLEVEL(PR, ("nEl=" LD "\n lacList[" LD "]=" LD " nrowsleft=" LD "\n", nEl, e,
                 lacList[e], el->nrowsleft));

    PR = 1;
    if (nEl != new_lac && el_colIndex[new_lac] < col2) PR = -2;

    PRLEVEL(PR, ("%% " LD " :\n", eli));
    if (PR <= 0) paru_print_element(eli, Work, Num);

    PRLEVEL(PR, ("%% " LD " :\n", e));
    if (PR <= 0) paru_print_element(e, Work, Num);
    PR = 1;
    ASSERT(nEl == new_lac || col2 <= el_colIndex[new_lac]);

#endif
    if (new_lac == nEl)
    {
#ifndef NDEBUG
        PRLEVEL(PR, ("%% " LD " is freed inside with0\n", eli));
#endif
        paru_free_el(e, elementList);
    }
}
