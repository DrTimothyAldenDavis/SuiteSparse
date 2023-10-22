////////////////////////////////////////////////////////////////////////////////
/////////////////////////// paru_front  ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief Computing factorization of current front and doing the numerical
 * assembly that ancestors will assemble. Degree update will be used in this
 * version. Just like ./paru_assemble.cpp
 *
 *
 * @param  the front that is going to be computed
 * @return  ParU_Ret
 *
 *  @author Aznaveh
 */
#include "paru_internal.hpp"
ParU_Ret paru_front(int64_t f,  // front need to be assembled
                    paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(-3);
    PARU_DEFINE_PRLEVEL;
    /*
     * -2 Print Nothing
     * -1 Just Matlab
     *  0 Detailed
     *  > 0 Everything
     */
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *Super = Sym->Super;
    /* ---------------------------------------------------------------------- */
    /* get the front F  */
    /* ---------------------------------------------------------------------- */

    PRLEVEL(-2, ("%%~~~~~~~  Assemble Front %ld start ~~%.0lf~~~~~~~(%d)\n", f,
                 Sym->stree_flop_bound[f], omp_get_thread_num()));

    /* pivotal columns Super [f] ... Super [f+1]-1 */
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t col2 = Super[f + 1];
    int64_t fp = col2 - col1; /* first fp columns are pivotal */

    paru_element **elementList = Work->elementList;

    PRLEVEL(-1, ("%% fp=%ld pivotal columns:clo1=%ld...col2=%ld\n", fp, col1,
                 col2 - 1));
    ASSERT(fp > 0);

    /* computing number of rows, set union */

    ParU_Control *Control = Num->Control;
    int64_t panel_width = Control->panel_width;

    try
    {
        int64_t num_panels = (int64_t)ceil((double)fp / panel_width);

        // panel_row shows number of rows in each panel.
        // Needs to be initialized in my new algorithm

        std::vector<int64_t> panel_row(num_panels, 0);
        int64_t *snM = Sym->super2atree;
        int64_t eli = snM[f];

        int64_t *isRowInFront = Work->rowSize;

        int64_t fm = Sym->Fm[f]; /* Upper bound number of rows of F */
        PRLEVEL(1, ("%% the size of fm is %ld\n", fm));
        int64_t *frowList = (int64_t *)paru_alloc(fm, sizeof(int64_t));
        if (frowList == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory when tried to allocate"
                        " for frowList %ld\n",
                        f));
            return PARU_OUT_OF_MEMORY;
        }
        Num->frowList[f] = frowList;

        std::set<int64_t>::iterator it;
#ifndef NDEBUG
        std::set<int64_t> stl_rowSet;
#endif

        // Initializing relative index validation flag of current front
        paru_init_rel(f, Work);

#ifndef NDEBUG
        int64_t time_f = Work->time_stamp[f];
        PRLEVEL(0, ("%% Begin of Front %ld time_f = %ld\n", f, time_f));
#endif

        // int64_t panel_num = 0;
        Num->frowCount[f] = 0;
        // int64_t rowCount = 0;

        /*********** Making the heap from list of the immediate children ******/

        /****************** pivotal column assembly  **************************/
        /***********  assembling the pivotal part of the front ****************/
        /*
         *
         *  el           nEl
         *            6, 7, 3, 12
         *           ____________
         *        23 | X  Y  .  .  stored in memory like this:
         *    mEl 17 | X  Y  .  .  ...6, 7, 3, 10, 23, 17, 2, X, X, X, Y, Y, Y,
         *         2 | X  Y  .  .
         *
         *  It must be assembled in current pivotal fron like this:
         *                                   fp
         *                               col1, ... , col
         *
         *                                6, 7, 8, 9, 10
         *                                ______________
         *                        0   23 | X  Y  .  .  .
         *             rowCount   1    2 | X  Y  .  .  .
         *                        2    4 | *  *  .  .  .  isRowInFront[4] == 2
         *                        3   17 | X  Y  .  .  .
         * */

        std::vector<int64_t> pivotal_elements;
        heaps_info hi;
        int64_t zero_piv_rows = 0;  // If there are zero rows is
        // importiant for Exit point
        PRLEVEL(1, ("%% Next: work on pivotal column assembly\n"));
        ParU_Ret res_pivotal;
        res_pivotal = paru_pivotal(pivotal_elements, panel_row, zero_piv_rows,
                                   f, hi, Work, Num);
        if (res_pivotal == PARU_OUT_OF_MEMORY)
        {
            PRLEVEL(1,
                    ("ParU: out of memory making pivotal of front %ld\n", f));
            return PARU_OUT_OF_MEMORY;
        }
        PRLEVEL(1, ("%% Done: work on pivotal column assembly\n"));

        int64_t rowCount = Num->frowCount[f];
        frowList = Num->frowList[f];

#ifndef NDEBUG /* chekcing first part of Work to be zero */
        int64_t *rowMarkp = Work->rowMark;
        int64_t rowMark = rowMarkp[eli];
        int64_t m = Num->m;

        PRLEVEL(1, ("%% rowMark=%ld;\n", rowMark));
        for (int64_t i = 0; i < m; i++)
        {
            if (isRowInFront[i] >= rowMark)
                PRLEVEL(1, ("%%rowMark = %ld, isRowInFront[%ld] = %ld\n",
                            rowMark, i, isRowInFront[i]));
        }
#endif

        ParU_Factors *LUs = Num->partial_LUs;
        double *pivotalFront = LUs[f].p;
        LUs[f].m = rowCount;
        LUs[f].n = fp;

        /***********  factorizing the fully summed part of the matrix        ***
         ** a set of pivot is found in this part that is crucial to assemble **/
        PRLEVEL(1, ("%% rowCount =%ld\n", rowCount));

#ifndef NDEBUG  // Printing the list of rows
        PRLEVEL(PR, ("%% Befor factorization (inside assemble): \n"));
        for (int64_t i = 0; i < rowCount; i++)
            PRLEVEL(PR, ("%% frowList [%ld] =%ld\n", i, frowList[i]));
        PRLEVEL(PR, ("\n"));

#endif

        // provide paru_alloc as the allocator
        int64_t fn = Sym->Cm[f];      /* Upper bound number of cols of F */
        std::set<int64_t> stl_colSet; /* used in this scope */

        if (rowCount < fp)
        {
            PRLEVEL(-1, ("ParU: singular, structural problem on %ld: %ldx%ld\n",
                         f, rowCount, fp));
#pragma omp atomic write
            Num->res = PARU_SINGULAR;
            return PARU_SINGULAR;
        }

        int64_t start_fac = Work->time_stamp[f];
        PRLEVEL(1, ("%% start_fac= %ld\n", start_fac));

        ParU_Ret ffs_blas_ok = paru_factorize_full_summed(
            f, start_fac, panel_row, stl_colSet, pivotal_elements, Work, Num);
        if (ffs_blas_ok != PARU_SUCCESS) return ffs_blas_ok; //failed blas
        ++Work->time_stamp[f];

#ifndef NDEBUG
        time_f = Work->time_stamp[f];
        PRLEVEL(1, ("%%After factorization time_f = %ld\n", time_f));
#endif

        /*To this point fully summed part of the front is computed and L and U /
         * The next part is to find columns of nonfully summed then rows
         * the rest of the matrix and doing TRSM and GEMM,                    */

        PRLEVEL(1, ("%% num_panels = %ld\n", num_panels));
        PRLEVEL(1, ("%% After free num_panels = %ld\n", num_panels));

#ifndef NDEBUG  // Printing the list of rows
        PRLEVEL(PR, ("%% After factorization (inside assemble): \n"));
        for (int64_t i = 0; i < rowCount; i++)
            PRLEVEL(PR, ("%% frowList [%ld] =%ld\n", i, frowList[i]));
        PRLEVEL(PR, ("\n"));
#endif

#ifndef NDEBUG  // Printing the permutation
        PRLEVEL(PR, ("%% pivotal rows:\n"));
        for (int64_t i = 0; i < fp; i++)
            PRLEVEL(PR, ("%% frowList[%ld] =%ld\n", i, frowList[i]));
        PRLEVEL(PR, ("%% =======\n"));
        for (int64_t i = fp; i < rowCount; i++)
            PRLEVEL(PR, ("%% frowList[%ld] =%ld\n", i, frowList[i]));
        PRLEVEL(PR, ("\n"));
#endif

#ifndef NDEBUG  // Printing the pivotal front
        PR = -1;
        PRLEVEL(PR, ("%%L part:\n"));

        // col permutatin
        PRLEVEL(PR, ("cols{%ld} = [", f + 1));
        for (int64_t c = col1; c < col2; c++) PRLEVEL(PR, ("%ld ", c + 1));
        PRLEVEL(PR, ("];\n"));

        // row permutatin
        PRLEVEL(PR, ("rows{%ld} = [", f + 1));
        for (int64_t r = 0; r < rowCount; r++)
            PRLEVEL(PR, ("%ld ", frowList[r] + 1));  // Matlab is base 1
        PRLEVEL(PR, ("];\n"));

        // inv row permutatin

        PRLEVEL(PR, ("Luf{%ld}= [", f + 1));
        for (int64_t r = 0; r < rowCount; r++)
        {
            PRLEVEL(PR, (" "));
            for (int64_t c = col1; c < col2; c++)
                PRLEVEL(PR,
                        (" %.1g ", pivotalFront[(c - col1) * rowCount + r]));
            PRLEVEL(PR, (";\n   "));
        }
        PRLEVEL(PR, ("];\n"));
        PR = 1;
        // just in cases that there is no U for MATLAB
        PRLEVEL(PR, ("Us{%ld} =[];\n", f + 1));
        PRLEVEL(PR, ("Ucols{%ld}=[];\n", f + 1));
        PRLEVEL(PR, ("Urows{%ld}=[];\n", f + 1));
#endif

        int64_t colCount = stl_colSet.size();
        ASSERT(fn >= colCount);

        int64_t *fcolList = NULL;

        if (fn != 0)
        {
            PRLEVEL(1, ("%% fp=%ld fn=%ld \n", fp, fn));
            fcolList = (int64_t *)paru_calloc(stl_colSet.size(), sizeof(int64_t));

            if (fcolList == NULL)
            {
                PRLEVEL(1, ("ParU: out of memory when tried to allocate for"
                            " fcolList=%ld with the size %ld\n",
                            f, fn));
                return PARU_OUT_OF_MEMORY;
            }
        }

        Num->fcolList[f] = fcolList;

        std::vector<int64_t> **heapList = Work->heapList;
        std::vector<int64_t> *curHeap = heapList[eli];

        // EXIT point HERE
        if (colCount == 0)
        {  // there is no CB, Nothing to be done
            Num->fcolCount[f] = 0;
            // if (zero_piv_rows > 0 )
            if (zero_piv_rows > 0 || rowCount > fp)
            {
                // make the heap and return
                PRLEVEL(-2, ("%%~~~~~~~Assemble Front %ld finished~~~1\n", f));
                return paru_make_heap_empty_el(f, pivotal_elements, hi, Work,
                                               Num);
            }
            else
            {
                PRLEVEL(
                    1, ("%%Heap freed inside front %p id=%ld\n", curHeap, eli));
                delete curHeap;
                Work->heapList[eli] = NULL;
                PRLEVEL(1, ("%% pivotalFront =%p\n", pivotalFront));
                PRLEVEL(-2, ("%%~~~~~~~Assemble Front %ld finished~~~2\n", f));
                return PARU_SUCCESS;
            }
        }

        // fcolList copy from the stl_colSet
        // hasing from fcolList indices to column index
        // the last elment of the hash shows if it is a lookup table
        // int64_t hash_size = (colCount*2 > Sym->n )? Sym->n : colCount;
        int64_t hash_size = ((int64_t)2) << ((int64_t)floor(log2((double)colCount)) + 1);
        PRLEVEL(1, ("%% 1Front hash_size=%ld\n", hash_size));
        hash_size = (hash_size > Sym->n) ? Sym->n : hash_size;
        PRLEVEL(1, ("%% 2Front hash_size=%ld\n", hash_size));
        std::vector<int64_t> colHash(hash_size + 1, -1);
        int64_t ii = 0;
        if (hash_size == Sym->n)
        {
            PRLEVEL(PR, ("%% colHash LOOKUP size = %ld LU %ld\n", hash_size,
                         Sym->n));
            for (it = stl_colSet.begin(); it != stl_colSet.end(); ++it)
            {
                colHash[*it] = ii;
                fcolList[ii++] = *it;
            }
        }
        else
        {
            // hash_bits is a bit mask to compute the result modulo
            // the hash table size, which is always a power of 2.

            PRLEVEL(PR, ("%% colHash HASH hash_size=%ld\n", hash_size));
            PRLEVEL(PR, ("%% colCount=%ld\n", colCount));
            for (it = stl_colSet.begin(); it != stl_colSet.end(); ++it)
            {
                paru_insert_hash(*it, ii, colHash);
                fcolList[ii++] = *it;
            }
            colHash[hash_size] = colCount;
        }
#ifndef NDEBUG
        PRLEVEL(PR, ("%% colHash %%"));
        for (auto i : colHash) PRLEVEL(PR, (" %ld ", i));
        PRLEVEL(PR, ("\n"));
#endif

        /**** 5 ** assemble U part         Row by Row                      ****/

        // consider a parallel calloc
        double *uPart = (double *)paru_calloc(fp * colCount, sizeof(double));
        if (uPart == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory when tried to"
                        " allocate for U part %ld\n",
                        f));
            return PARU_OUT_OF_MEMORY;
        }

#ifndef NDEBUG
        if (f == Sym->nf - 1)
        {
            PR = 3;
        }
        if (fn != colCount)
            PRLEVEL(PR, ("%% fn=%ld colCount=%ld ", fn, colCount));
        PRLEVEL(PR, ("%% uPart = %p size=%ld", uPart, colCount * fp));
#endif

        ParU_Factors *Us = Num->partial_Us;
        Us[f].m = fp;
        Us[f].n = colCount;
        Num->fcolCount[f] = colCount;
        ASSERT(Us[f].p == NULL);
        Us[f].p = uPart;

        paru_tupleList *RowList = Work->RowList;

        // int64_t *Depth = Sym->Depth;
        //**//pragma omp parallel
        //**//pragma omp single nowait
        //**//pragma omp taskgroup
        for (int64_t i = 0; i < fp; i++)
        {
            int64_t curFsRowIndex = i;  // current fully summed row index
            int64_t curFsRow = frowList[curFsRowIndex];
            PRLEVEL(1, ("%% curFsRow =%ld\n", curFsRow));
            paru_tupleList *curRowTupleList = &RowList[curFsRow];
            int64_t numTuple = curRowTupleList->numTuple;
            ASSERT(numTuple >= 0);
            ASSERT(numTuple <= m);
            paru_tuple *listRowTuples = curRowTupleList->list;
            PRLEVEL(1, ("%% numTuple = %ld\n", numTuple));
            for (int64_t k = 0; k < numTuple; k++)
            {
                paru_tuple curTpl = listRowTuples[k];
                int64_t e = curTpl.e;
                paru_element *el = elementList[e];
                if (el == NULL) continue;

                int64_t curRowIndex = curTpl.f;
                int64_t nEl = el->ncols;
                // int64_t *el_rowIndex = rowIndex_pointer (el);
                int64_t *el_rowIndex = (int64_t *)(el + 1) + nEl;

                if (el_rowIndex[curRowIndex] < 0) continue;

                int64_t mEl = el->nrows;
                // int64_t *rowRelIndex = relRowInd (el);
                int64_t *rowRelIndex = (int64_t *)(el + 1) + 2 * nEl + mEl;

                PRLEVEL(1, ("%% curFsRowIndex =%ld\n", curFsRowIndex));
                ASSERT(el_rowIndex[curRowIndex] == curFsRow);
                ASSERT(curRowIndex < mEl);
                PRLEVEL(1, ("%% curColIndex =%ld\n", curRowIndex));

                PRLEVEL(1, ("%% element= %ld  nEl =%ld \n", e, nEl));

                //**//pragma omp task priority(Depth[f]) if(mEl > 1024)
                paru_assemble_row_2U(e, f, curRowIndex, curFsRowIndex, colHash,
                                     Work, Num);
                //**//pragma omp taskwait

                // FLIP(el_rowIndex[curRowIndex]); //marking row assembled
                el_rowIndex[curRowIndex] = -1;
                rowRelIndex[curRowIndex] = -1;
                el->nrowsleft--;
                if (el->nrowsleft == 0)
                {
                    paru_free_el(e, elementList);
                }
            }
        }

#ifndef NDEBUG  // Printing the  U part
        PR = -1;
        PRLEVEL(PR, ("%% U part Before TRSM: %ld x %ld\n", fp, colCount));
        PRLEVEL(PR, ("%% U\t"));
        for (int64_t i = 0; i < colCount; i++)
            PRLEVEL(PR, ("%ld\t\t", fcolList[i]));
        PRLEVEL(PR, ("\n"));
        for (int64_t i = 0; i < fp; i++)
        {
            PRLEVEL(PR, ("%% %ld\t", frowList[i]));
            for (int64_t j = 0; j < colCount; j++)
                PRLEVEL(PR, (" %2.5lf\t", uPart[j * fp + i]));
            PRLEVEL(PR, ("\n"));
        }

#endif

        /**** 6 ****                 TRSM and DGEMM                         ***/

        int64_t trsm_blas_ok = paru_trsm(f, pivotalFront, uPart, fp, rowCount,
                                     colCount, Work, Num);
        if (!trsm_blas_ok) return PARU_TOO_LARGE;

#ifndef NDEBUG  // Printing the  U part
        PR = -1;
        PRLEVEL(PR, ("%% rowCount=%ld;\n", rowCount));
        PRLEVEL(PR, ("%% U part After TRSM: %ld x %ld\n", fp, colCount));

        PRLEVEL(PR, ("Ucols{%ld} = [", f + 1));
        for (int64_t i = 0; i < colCount; i++)
            PRLEVEL(PR, ("%ld ", fcolList[i] + 1));
        PRLEVEL(PR, ("];\n"));

        PRLEVEL(PR, ("Urows{%ld} = [", f + 1));
        for (int64_t i = 0; i < fp; i++) PRLEVEL(PR, ("%ld ", frowList[i] + 1));
        PRLEVEL(PR, ("];\n"));

        PRLEVEL(PR, ("Us{%ld} = [", f + 1));

        for (int64_t i = 0; i < fp; i++)
        {
            for (int64_t j = 0; j < colCount; j++)
                PRLEVEL(PR, (" %.16g ", uPart[j * fp + i]));
            PRLEVEL(PR, (";\n    "));
        }
        PRLEVEL(PR, ("];\n"));
        PR = 1;
#endif

        paru_element *curEl;
        PRLEVEL(1, ("%% rowCount=%ld, colCount=%ld, fp=%ld\n", rowCount,
                    colCount, fp));
        PRLEVEL(1, ("%% curEl is %ld by %ld\n", rowCount - fp, colCount));
        if (fp < rowCount)
        {
            // allocating an un-initialized part of memory
            curEl = elementList[eli] =
                paru_create_element(rowCount - fp, colCount);

            // While insided the DGEMM BETA == 0
            if (curEl == NULL)
            {
                PRLEVEL(1, ("ParU: out of memory when tried to allocate"
                            " current CB %ld\n",
                            eli));
                return PARU_OUT_OF_MEMORY;
            }
            PRLEVEL(1, ("%% Created ele %ld in curEl =%p\n", eli, curEl));
        }
        else  // EXIT point
        {     // NO rows for current contribution block
            if (zero_piv_rows > 0)
            {
                // keep the heap and do it for the parent.
                PRLEVEL(-2, ("%%~~~~~~~Assemble Front %ld finished~~~3\n", f));
                return paru_make_heap_empty_el(f, pivotal_elements, hi, Work,
                                               Num);
                // There are stuff left from in zero
                // then return
            }
            else
            {
                return paru_make_heap_empty_el(f, pivotal_elements, hi, Work,
                                               Num);
                // delete curHeap;
                // Work->heapList[eli] = NULL;
                // PRLEVEL(1, ("%%(2)Heap freed inside front %p id=%ld\n"
                //            , curHeap, eli));
                // PRLEVEL(1, ("%% pivotalFront =%p\n", pivotalFront));
                // PRLEVEL(-2, ("%%~~~~~~~Assemble Front %ld finished~~~4\n",
                // f)); return PARU_SUCCESS;
            }
        }

        // Initializing curEl global indices
        // int64_t *el_colIndex = colIndex_pointer (curEl);
        int64_t *el_colIndex = (int64_t *)(curEl + 1);
        curEl->lac = 0;
        int64_t *lacList = Work->lacList;
        lacList[eli] = fcolList[0];
        for (int64_t i = 0; i < colCount; ++i) el_colIndex[i] = fcolList[i];
        int64_t *el_rowIndex = rowIndex_pointer(curEl);
        for (int64_t i = fp; i < rowCount; ++i)
        {
            int64_t locIndx = i - fp;
            int64_t curRow = frowList[i];
            el_rowIndex[locIndx] = curRow;
            // Updating isRowInFront after the pivoting
            // attention it is the old rowMark not the updated rowMarkp + eli
            // If I decide to add rowMark I should change paru_pivotal
            isRowInFront[curRow] = locIndx;
            PRLEVEL(1, ("%% el_rowIndex [%ld] =%ld\n", locIndx,
                        el_rowIndex[locIndx]));
        }

        // double *el_numbers = numeric_pointer (curEl);
        double *el_numbers =
            (double *)((int64_t *)(curEl + 1) + 2 * colCount + 2 * (rowCount - fp));

        int64_t dgemm_blas_ok = paru_dgemm(f, pivotalFront, uPart, el_numbers, fp,
                                       rowCount, colCount, Work, Num);
        if (!dgemm_blas_ok) return PARU_TOO_LARGE;

#ifdef COUNT_FLOPS
#if 0
        for (int64_t ii = 0; ii < fp; ii++)
            for (int64_t jj = 0; jj < colCount; jj++)
                for (int64_t kk = fp; kk < rowCount; kk++)
                {
                    if (uPart[fp * jj + ii] != 0 &&
                            pivotalFront[rowCount * ii + kk] != 0)
                        Work->flp_cnt_real_dgemm += 2.0;
                }
#endif

        PRLEVEL(PR, ("\n%% FlopCount Dgemm front %ld %ld %ld \n", rowCount - fp,
                     fp, colCount));
        PRLEVEL(PR, ("%ld %ld %ld \n", rowCount - fp, fp, colCount));
#endif

#ifndef NDEBUG
        // Printing the contribution block after dgemm
        PR = 1;
        PRLEVEL(PR, ("\n%%After DGEMM:"));
        if (PR <= 0) paru_print_element(eli, Work, Num);
#endif

        /*** 7 * Count number of rows and columsn of prior CBs to asslemble ***/

        PRLEVEL(-1, ("\n%%||||  Start Finalize %ld ||||\n", f));
        ParU_Ret res_prior;
        res_prior = paru_prior_assemble(f, start_fac, pivotal_elements, colHash,
                                        hi, Work, Num);
        if (res_prior != PARU_SUCCESS) return res_prior;
        PRLEVEL(-1, ("\n%%||||  Finish Finalize %ld ||||\n", f));

        ////////////////////////////////////////////////////////////////////////

        for (int64_t i = fp; i < rowCount; ++i)
        {
            int64_t locIndx = i - fp;
            paru_tuple rowTuple;
            rowTuple.e = eli;
            rowTuple.f = locIndx;
            if (paru_add_rowTuple(RowList, frowList[i], rowTuple) ==
                PARU_OUT_OF_MEMORY)
            {
                PRLEVEL(1, ("ParU: out of memory: add_rowTuple \n"));
                return PARU_OUT_OF_MEMORY;
            }
        }

#ifndef NDEBUG /* chekcing if isRowInFront is correct */
        rowMark = rowMarkp[eli];
        // int64_t *Sleft = Sym->Sleft;
        //    for (int64_t i = Sleft[col1]; i < Sleft[Super[f+1]]; i++)
        //        ASSERT ( isRowInFront [i] < rowMark);
#endif

        PRLEVEL(1, ("%%rowCount =%ld  ", rowCount));
        PRLEVEL(1, ("colCount =%ld", colCount));
        PRLEVEL(1, ("fp =%ld;\n", fp));
        PRLEVEL(-2, ("%%~~~~~~~Assemble Front %ld finished~~~5\n", f));
    }

    catch (std::bad_alloc const &)
    {  // out of memory
        PRLEVEL(1, ("ParU: Out of memory, bad_alloc in front\n"));
        return PARU_OUT_OF_MEMORY;
    }
    return PARU_SUCCESS;
}
