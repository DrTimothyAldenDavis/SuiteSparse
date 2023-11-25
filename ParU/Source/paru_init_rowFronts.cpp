////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_init_rowFronts  ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*!  @brief  Initializing row fronts; fronts will be assembled later.
 *         Initializing Row and column tuple lists:
 *            allocating memory and updating lists and initializing matrix
 *            structre
 *        Assemble each front:
 *            Adding numerical values, allocating data
 *            updating the list
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

ParU_Ret paru_init_rowFronts(paru_work *Work,
                             ParU_Numeric **Num_handle,  // in/out
                                                         // inputs, not modified
                             cholmod_sparse *A,
                             // symbolic analysis
                             ParU_Symbolic *Sym, ParU_Control *Control)
{
    // mallopt(M_TRIM_THRESHOLD, -1);         // disable sbrk trimming
    // mallopt(M_TOP_PAD, 16 * 1024 * 1024);  // increase padding to speedup
    // malloc

    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    // initializing Work
    Work->Sym = Sym;
    int64_t *rowMark = Work->rowMark = NULL;
    int64_t *elRow = Work->elRow = NULL;
    int64_t *elCol = Work->elCol = NULL;
    int64_t *rowSize = Work->rowSize = NULL;
    Work->time_stamp = NULL;
    paru_tupleList *RowList = Work->RowList = NULL;
    int64_t *Diag_map = Work->Diag_map = NULL;
    int64_t *inv_Diag_map = Work->inv_Diag_map = NULL;
    paru_element **elementList = Work->elementList = NULL;
    Work->lacList = NULL;
    Work->task_num_child = NULL;
    std::vector<int64_t> **heapList = Work->heapList = NULL;
    int64_t *row_degree_bound = Work->row_degree_bound = NULL;
   
    // initializing Numeric
    ParU_Numeric *Num = NULL;
    Num = static_cast<ParU_Numeric*>(paru_alloc(1, sizeof(ParU_Numeric)));
    if (Num == NULL)
    {  // out of memory
        PRLEVEL(1, ("ParU: out of memory, Num\n"));
        // Nothing to be freed
        *Num_handle = NULL;
        return PARU_OUT_OF_MEMORY;
    }
    *Num_handle = Num;

    int64_t m, nf;
    Num->sym_m = Sym->m;
    m = Num->m = Sym->m - Sym->n1;
    nf = Num->nf = Sym->nf;
    Num->res = PARU_SUCCESS;
    Num->Control = Control;

    Num->frowCount = NULL;
    Num->fcolCount = NULL;
    Num->frowList = NULL;
    Num->fcolList = NULL;
    Num->partial_Us = NULL;
    Num->partial_LUs = NULL;
    Num->Sx = NULL;
    Num->Sux = NULL;
    Num->Slx = NULL;
    Num->Rs = NULL;
    Num->Ps = NULL;
    Num->Pfin = NULL;

    if (nf != 0)
    {
        // Memory allocations for Num
        rowMark = Work->rowMark = static_cast<int64_t*>(paru_alloc(m + nf + 1, sizeof(int64_t)));
        elRow = Work->elRow = static_cast<int64_t*>(paru_alloc(m + nf, sizeof(int64_t)));
        elCol = Work->elCol = static_cast<int64_t*>(paru_alloc(m + nf, sizeof(int64_t)));
        rowSize = Work->rowSize = static_cast<int64_t*>(paru_alloc(m, sizeof(int64_t)));
        row_degree_bound = Work->row_degree_bound =
            static_cast<int64_t*>(paru_alloc(m, sizeof(int64_t)));
        RowList = Work->RowList =
            static_cast<paru_tupleList*>(paru_calloc(1, m * sizeof(paru_tupleList)));
        Work->lacList = static_cast<int64_t*>(paru_alloc(m + nf, sizeof(int64_t)));
        Num->frowCount = static_cast<int64_t*>(paru_alloc(1, nf * sizeof(int64_t)));
        Num->fcolCount = static_cast<int64_t*>(paru_alloc(1, nf * sizeof(int64_t)));
        Num->frowList = static_cast<int64_t**>(paru_calloc(1, nf * sizeof(int64_t *)));
        Num->fcolList = static_cast<int64_t**>(paru_calloc(1, nf * sizeof(int64_t *)));
        Num->partial_Us =  // Initialize with NULL
            static_cast<ParU_Factors*>(paru_calloc(1, nf * sizeof(ParU_Factors)));
        Num->partial_LUs =  // Initialize with NULL
            static_cast<ParU_Factors*>(paru_calloc(1, nf * sizeof(ParU_Factors)));

        Work->time_stamp = static_cast<int64_t*>(paru_alloc(1, nf * sizeof(int64_t)));
        Work->task_num_child = static_cast<int64_t*>(paru_alloc(Sym->ntasks, sizeof(int64_t)));
        heapList = Work->heapList = static_cast<std::vector<int64_t>**>(paru_calloc(
            1, (m + nf + 1) * sizeof(std::vector<int64_t> *)));
        elementList = Work->elementList =  // Initialize with NULL
            static_cast<paru_element**>(paru_calloc(1,
                                         (m + nf + 1) * sizeof(paru_element)));
        if (Sym->strategy == PARU_STRATEGY_SYMMETRIC)
        {
            Diag_map = Work->Diag_map = static_cast<int64_t*>(paru_alloc(Sym->n, sizeof(int64_t)));
            inv_Diag_map = Work->inv_Diag_map =
                static_cast<int64_t*>(paru_alloc(Sym->n, sizeof(int64_t)));
#ifndef NDEBUG
            paru_memset(Diag_map, 0, Sym->n * sizeof(int64_t), Control);
            paru_memset(inv_Diag_map, 0, Sym->n * sizeof(int64_t), Control);
            PR = 2;
#endif
        }
    }

    int64_t snz = Num->snz = Sym->snz;
    double *Sx = NULL;
    Sx = Num->Sx = static_cast<double*>(paru_alloc(snz, sizeof(double)));
    int64_t *cSp = NULL;  // copy of Sp, temporary for making Sx
    cSp = static_cast<int64_t*>(paru_alloc(m + 1, sizeof(int64_t)));
    double *Sux = NULL;
    int64_t *cSup = NULL;  // copy of Sup temporary for making Sux
    int64_t cs1 = Sym->cs1;
    int64_t rs1 = Sym->rs1;
    int64_t sunz = 0;
    if (cs1 > 0)
    {
        sunz = Sym->ustons.nnz;
        Sux = static_cast<double*>(paru_alloc(sunz, sizeof(double)));
        cSup = static_cast<int64_t*>(paru_alloc(cs1 + 1, sizeof(int64_t)));
    }
    Num->Sux = Sux;
    double *Slx = NULL;
    int64_t *cSlp = NULL;  // copyf of Slp temporary, for making Slx
    int64_t slnz = 0;
    if (rs1 > 0)
    {
        slnz = Sym->lstons.nnz;
        Slx = static_cast<double*>(paru_alloc(slnz, sizeof(double)));
        cSlp = static_cast<int64_t*>(paru_alloc(rs1 + 1, sizeof(int64_t)));
    }
    Num->sunz = sunz;
    Num->slnz = slnz;
    Num->Slx = Slx;
    double *Rs = NULL;
    int64_t scale = Control->scale;  // if 1 the S will be scaled by max_row
    if (scale == 1) Rs = static_cast<double*>(paru_calloc(Sym->m, sizeof(double)));
    Num->Rs = Rs;
    if ((nf != 0 &&
         (rowMark == NULL || elRow == NULL || elCol == NULL ||
          rowSize == NULL || Work->lacList == NULL || RowList == NULL ||
          row_degree_bound == NULL || elementList == NULL ||
          Num->frowCount == NULL || Num->fcolCount == NULL ||
          Num->frowList == NULL || Num->fcolList == NULL || heapList == NULL ||
          Num->partial_Us == NULL || Num->partial_LUs == NULL ||
          Work->time_stamp == NULL || Work->task_num_child == NULL ||
          (Sym->strategy == PARU_STRATEGY_SYMMETRIC &&
           (Diag_map == NULL || inv_Diag_map == NULL)))) ||

        // stuff that can be allocated even when nf==0
        Sx == NULL || (scale == 1 && Rs == NULL) ||
        (cs1 > 0 && (Sux == NULL || cSup == NULL)) ||
        (rs1 > 0 && (Slx == NULL || cSlp == NULL)) || cSp == NULL)
    {
        paru_free(m + 1, sizeof(int64_t), cSp);
        if (cs1 > 0) paru_free((cs1 + 1), sizeof(int64_t), cSup);
        if (rs1 > 0) paru_free((rs1 + 1), sizeof(int64_t), cSlp);
        return PARU_OUT_OF_MEMORY;
    }

    // Initializations
    if (nf != 0)
    {
        PRLEVEL(PR, ("%% $RowList =%p\n", RowList));
        paru_memset(rowSize, -1, m * sizeof(int64_t), Control);
        PRLEVEL(PR, ("%% rowSize pointer=%p size=" LD " \n", rowSize,
                     m * sizeof(int64_t)));

        PRLEVEL(PR, ("%% rowMark pointer=%p size=" LD " \n", rowMark,
                     (m + nf) * sizeof(int64_t)));

        paru_memset(elRow, -1, (m + nf) * sizeof(int64_t), Control);
        PRLEVEL(PR, ("%% elRow=%p\n", elRow));

        paru_memset(elCol, -1, (m + nf) * sizeof(int64_t), Control);
        PRLEVEL(PR, ("%% elCol=%p\n", elCol));

        PRLEVEL(PR, ("%% Work =%p\n ", Work));
    }

    //////////////////Initializing numerics Sx, Sux and Slx //////////////////{
    int64_t *Ap = static_cast<int64_t*>(A->p);
    int64_t *Ai = static_cast<int64_t*>(A->i);
    double *Ax = static_cast<double*>(A->x);
    int64_t *Sp = Sym->Sp;
    int64_t *Slp = NULL;
    int64_t *Sup = NULL;
    paru_memcpy(cSp, Sp, (m + 1) * sizeof(int64_t), Control);
    if (cs1 > 0)
    {
        Sup = Sym->ustons.Sup;
        paru_memcpy(cSup, Sup, (cs1 + 1) * sizeof(int64_t), Control);
    }
    if (rs1 > 0)
    {
        Slp = Sym->lstons.Slp;
        paru_memcpy(cSlp, Slp, (rs1 + 1) * sizeof(int64_t), Control);
    }
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Init Sup and Slp in the middle\n"));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Sup =", sunz));
        for (int64_t k = 0; k <= cs1; k++)
        {
            PRLEVEL(PR, ("" LD " ", Sup[k]));
            PRLEVEL(PR + 2, ("c" LD " ", cSup[k]));
            if (Sup[k] != cSup[k])
                PRLEVEL(PR, ("Sup[" LD "] =" LD ", cSup=" LD "", k, Sup[k], cSup[k]));
            ASSERT(Sup[k] == cSup[k]);
        }
        PRLEVEL(PR, ("\n"));
    }
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Slp =", slnz));
        for (int64_t k = 0; k <= rs1; k++)
        {
            PRLEVEL(PR, ("" LD " ", Slp[k]));
            PRLEVEL(PR + 2, ("o" LD " ", cSlp[k]));
            if (Slp[k] != cSlp[k])
                PRLEVEL(PR,
                        ("\nSup[" LD "] =" LD ", cSup=" LD "\n", k, Slp[k], cSlp[k]));
            ASSERT(Slp[k] == cSlp[k]);
        }
        PRLEVEL(PR, ("\n"));
    }
#endif

    int64_t n1 = Sym->n1;

    int64_t *Qinit = Sym->Qfill;
    int64_t *Pinv = Sym->Pinv;
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Iniit Pinv =\n"));
    for (int64_t i = 0; i < m; i++) PRLEVEL(PR, ("" LD " ", Pinv[i]));
    PRLEVEL(PR, ("\n"));
#endif

    if (Rs)
    {
        for (int64_t newcol = 0; newcol < Sym->n; newcol++)
        {
            int64_t oldcol = Qinit[newcol];
            for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
            {
                int64_t oldrow = Ai[p];
                Rs[oldrow] = std::max(Rs[oldrow], fabs(Ax[p]));
            }
        }
    }

    PRLEVEL(PR, ("%% Rs:\n["));
    if (Rs)
    {  // making sure that every row has at most one element more than zero
        for (int64_t k = 0; k < m; k++)
        {
            PRLEVEL(PR, ("%lf ", Rs[k]));
            if (Rs[k] <= 0)
            {
                PRLEVEL(1, ("ParU: Matrix is singular, row " LD " is zero\n", k));
                Num->res = PARU_SINGULAR;
                return PARU_SINGULAR;
            }
        }
    }
    PRLEVEL(PR, ("]\n"));

    for (int64_t newcol = 0; newcol < Sym->n; newcol++)
    {
        int64_t oldcol = Qinit[newcol];
        for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
        {
            int64_t oldrow = Ai[p];
            int64_t newrow = Pinv[oldrow];
            int64_t srow = newrow - n1;
            int64_t scol = newcol - n1;
            if (srow >= 0 && scol >= 0)
            {  // it is inside S otherwise it is part of singleton
                Sx[cSp[srow]++] = (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
            }
            else if (srow < 0 && scol >= 0)
            {  // inside the U singletons
                PRLEVEL(PR, ("Usingleton newcol = " LD " newrow=" LD "\n", newcol,
                             newrow));
                // let the diagonal entries be first
                Sux[++cSup[newrow]] = (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
            }
            else
            {
                if (newrow < cs1)
                {  // inside U singletons CSR
                    // PRLEVEL(PR, ("Inside U singletons\n"));
                    if (newcol == newrow)
                    {  // diagonal entry
                        Sux[Sup[newrow]] =
                            (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
                    }
                    else
                    {
                        Sux[++cSup[newrow]] =
                            (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
                    }
                }
                else
                {  // inside L singletons CSC
                    // PRLEVEL(PR, ("Inside L singletons\n"));
                    if (newcol == newrow)
                    {  // diagonal entry
                        Slx[Slp[newcol - cs1]] =
                            (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
                    }
                    else
                    {
                        Slx[++cSlp[newcol - cs1]] =
                            (Rs == NULL) ? Ax[p] : Ax[p] / Rs[oldrow];
                    }
                }
            }
        }
    }
    paru_free(m + 1, sizeof(int64_t), cSp);
    if (Sym->cs1 > 0) paru_free((cs1 + 1), sizeof(int64_t), cSup);
    if (Sym->rs1 > 0) paru_free((rs1 + 1), sizeof(int64_t), cSlp);
        //////////////////Initializing numerics Sx, Sux and Slx
        /////////////////////}
#ifdef COUNT_FLOPS
    // flop count info init
    Work->flp_cnt_dgemm = 0.0;
    Work->flp_cnt_trsm = 0.0;
    Work->flp_cnt_dger = 0.0;
    Work->flp_cnt_real_dgemm = 0.0;
#endif

    // RowList, ColList and elementList are place holders
    // pointers to pointers that are allocated

    int64_t *Sj = Sym->Sj;

    /// ------------------------------------------------------------------------
    // create S = A (p,q)', or S=A(p,q), S is considered to be in row-form
    // -------------------------------------------------------------------------
#ifndef NDEBUG
    int64_t n = Num->n = Sym->n - Sym->n1;
    PRLEVEL(1, ("%% m=" LD ", n=" LD "\n", m, n));
    PR = 1;
    PRLEVEL(PR, ("\n%% Inside init row fronts\n"));
    PRLEVEL(PR, ("%% Sp =\n%%"));
    for (int64_t i = 0; i <= m; i++) PRLEVEL(PR, ("" LD " ", Sp[i]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("Sj =\n"));
    for (int64_t k = 0; k < snz; k++) PRLEVEL(PR, ("" LD " ", Sj[k]));
    PRLEVEL(PR, ("\n"));
#endif

    PRLEVEL(1, ("InMatrix=[\n"));  // MATLAB matrix,

    // copying Diag_map
    if (Diag_map)
    {
        #pragma omp taskloop default(none) shared(Sym, Diag_map, inv_Diag_map) \
        grainsize(512) 
        for (int64_t i = 0; i < Sym->n; i++)
        {
            Diag_map[i] = Sym->Diag_map[i];
            inv_Diag_map[Diag_map[i]] = i;
        }
#ifndef NDEBUG
        PR = 1;
        PRLEVEL(PR, ("init_row Diag_map (" LD ") =\n", Sym->n));
        for (int64_t i = 0; i < std::min(64, Sym->n); i++)
            PRLEVEL(PR, ("" LD " ", Diag_map[i]));
        PRLEVEL(PR, ("\n"));
        PRLEVEL(PR, ("inv_Diag_map =\n"));
        for (int64_t i = 0; i < std::min(64, Sym->n); i++)
            PRLEVEL(PR, ("" LD " ", inv_Diag_map[i]));
        PRLEVEL(PR, ("\n"));
        for (int64_t i = 0; i < Sym->n; i++)
        {
            if (Diag_map[i] == -1)
                PRLEVEL(PR,
                        ("Diag_map[" LD "] is not correctly initialized\n", i));

            if (inv_Diag_map[i] == -1)
                PRLEVEL(PR, ("inv_Diag_map[" LD "] is not correctly initialized\n",
                             i));

            ASSERT(Diag_map[i] != -1);
            // ASSERT(inv_Diag_map[i] != -1);
        }
        PR = 1;
#endif
    }

    // Activating comments after this parts will break the matlab input matrix
    // allocating row tuples, elements and updating column tuples

    ParU_Ret info;
    int64_t out_of_memory = 0;

    //XXX weird situation here:
    //   with omp parallel for I can see correct out_of_memory but not with 
    //   taskloop; However I don't see any leaks with either

    //pragma omp taskloop grainsize(512)
    #pragma omp parallel for num_threads(Control->paru_max_threads)
    for (int64_t row = 0; row < m; row++)
    {
        int64_t e = Sym->row2atree[row];
        int64_t nrows = 1,
            ncols =
                Sp[row + 1] - Sp[row];  // nrows and ncols of current front/row
        row_degree_bound[row] = ncols;  // Initialzing row degree

        paru_element *curEl = elementList[e] =
            paru_create_element(nrows, ncols);
        if (curEl == NULL)
        {  // out of memory
            PRLEVEL(1, ("ParU: Out of memory: curEl\n"));
            #pragma omp atomic update
            out_of_memory += 1;
        }
        else
            try
            {
                rowMark[e] = 0;

                // My new is calling paru_alloc
                std::vector<int64_t> *curHeap = Work->heapList[e] =
                    new std::vector<int64_t>;

                curHeap->push_back(e);

                // constants for initialzing lists
                int64_t slackRow = 2;

                // Allocating Rowlist and updating its tuples
                RowList[row].list = static_cast<paru_tuple*>(paru_alloc(
                    slackRow * nrows, sizeof(paru_tuple)));
                if (RowList[row].list == NULL)
                {  // out of memory
                    PRLEVEL(1, ("ParU: out of memory, RowList[row].list \n"));
                    #pragma omp atomic update
                    out_of_memory += 1;
                }
                else
                {
                    RowList[row].numTuple = 0;
                    RowList[row].len = slackRow;

                    paru_tuple rowTuple;
                    rowTuple.e = e;
                    rowTuple.f = 0;
                    if (paru_add_rowTuple(RowList, row, rowTuple) ==
                        PARU_OUT_OF_MEMORY)
                    {
                        PRLEVEL(1, ("ParU: out of memory, add_rowTuple \n"));
                        #pragma omp atomic update
                        out_of_memory += 1;
                    }
                    else
                    {
                        // Allocating elements
                        int64_t *el_colrowIndex = colIndex_pointer(curEl);
                        double *el_colrowNum = numeric_pointer(curEl);

                        int64_t j = 0;  // Index inside an element
                        for (int64_t p = Sp[row]; p < Sp[row + 1]; p++)
                        {
                            el_colrowIndex[j] = Sj[p];
                            el_colrowNum[j++] = Sx[p];
                        }
                        el_colrowIndex[j++] =
                            row;  // initializing element row index
                        Work->lacList[e] = lac_el(elementList, e);
                    }
                }
            }
            catch (std::bad_alloc const &)
            {  // out of memory
                PRLEVEL(1, ("ParU: Out of memory: curHeap\n"));
                #pragma omp atomic update
                out_of_memory += 1;
            }
    }
    if (out_of_memory > 0)
    {
        info = PARU_OUT_OF_MEMORY;
    }
    else
        info = PARU_SUCCESS;

    PRLEVEL(1, ("];\n"));
    PRLEVEL(1, ("I = InMatrix(:,1);\n"));
    PRLEVEL(1, ("J = InMatrix(:,2);\n"));
    PRLEVEL(1, ("X = InMatrix(:,3);\n"));
    PRLEVEL(1, ("S = sparse(I,J,X);\n"));

    return info;
}
