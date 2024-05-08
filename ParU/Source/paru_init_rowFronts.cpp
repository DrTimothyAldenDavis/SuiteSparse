////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_init_rowFronts  ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

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

#define FREE_WORK                           \
{                                           \
    PARU_FREE (m + 1, int64_t, cSp);        \
    PARU_FREE (cs1 + 1, int64_t, cSup);     \
    PARU_FREE (rs1 + 1, int64_t, cSlp);     \
}

ParU_Info paru_init_rowFronts
(
    // input/output:
    paru_work *Work,
    ParU_Numeric *Num_handle,
    // inputs, not modified:
    cholmod_sparse *A,
    const ParU_Symbolic Sym     // symbolic analysis
)
{

    // get Control
    int32_t nthreads = Work->nthreads ;
    size_t mem_chunk = Work->mem_chunk ;
    int32_t prescale = Work->prescale ;

    // workspace:
    int64_t *cSp = NULL;    // copy of Sp, temporary for making Sx
    int64_t *cSup = NULL;   // copy of Sup temporary for making Sux
    int64_t *cSlp = NULL;   // copyf of Slp temporary, for making Slx
    int64_t cs1 = 0 ;
    int64_t rs1 = 0 ;

    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    // initializing Work
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
    ParU_Numeric Num = NULL;
    Num = PARU_CALLOC (1, ParU_Numeric_struct);
    if (Num == NULL)
    {
        // out of memory
        PRLEVEL(1, ("ParU: out of memory, Num\n"));
        // Nothing to be freed
        (*Num_handle) = NULL;
        return PARU_OUT_OF_MEMORY;
    }
    (*Num_handle) = Num;

    int64_t m, nf;
    Num->sym_m = Sym->m;
    m = Num->m = Sym->m - Sym->n1;
    nf = Num->nf = Sym->nf;
    Num->res = PARU_SUCCESS;

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
        rowMark = Work->rowMark = PARU_MALLOC (m + nf + 1, int64_t);
        elRow = Work->elRow = PARU_MALLOC (m + nf, int64_t);
        elCol = Work->elCol = PARU_MALLOC (m + nf, int64_t);
        rowSize = Work->rowSize = PARU_MALLOC (m, int64_t);
        row_degree_bound = Work->row_degree_bound = PARU_MALLOC (m, int64_t);
        RowList = Work->RowList = PARU_CALLOC (m, paru_tupleList);
        Work->lacList = PARU_MALLOC (m + nf, int64_t);
        Num->frowCount = PARU_MALLOC (nf, int64_t);
        Num->fcolCount = PARU_MALLOC (nf, int64_t);
        Num->frowList = PARU_CALLOC (nf, int64_t *);
        Num->fcolList = PARU_CALLOC (nf, int64_t *);
        Num->partial_Us = PARU_CALLOC (nf, ParU_Factors);
        Num->partial_LUs = PARU_CALLOC (nf, ParU_Factors);

        Work->time_stamp = PARU_MALLOC (nf, int64_t);
        Work->task_num_child = PARU_MALLOC (Sym->ntasks, int64_t);
        heapList = Work->heapList =
            PARU_CALLOC (m+nf+1, std::vector<int64_t> *);
        elementList = Work->elementList = PARU_CALLOC (m+nf+1, paru_element *);

        if (rowMark == NULL || elRow == NULL || elCol == NULL ||
            rowSize == NULL || row_degree_bound == NULL || RowList == NULL ||
            Work->lacList == NULL ||
            Num->frowCount == NULL  || Num->fcolCount == NULL   ||
            Num->frowList == NULL   || Num->fcolList == NULL    ||
            Num->partial_Us == NULL || Num->partial_LUs == NULL ||
            Work->time_stamp == NULL || Work->task_num_child == NULL ||
            heapList == NULL ||
            elementList == NULL)
        {
            // out of memory
            FREE_WORK ;
            return (PARU_OUT_OF_MEMORY) ;
        }

        if (Sym->strategy_used == PARU_STRATEGY_SYMMETRIC)
        {
            Diag_map = Work->Diag_map = PARU_MALLOC (Sym->n, int64_t);
            inv_Diag_map = Work->inv_Diag_map = PARU_MALLOC (Sym->n, int64_t);
            if (Diag_map == NULL || inv_Diag_map == NULL)
            {
                // out of memory
                FREE_WORK ;
                return (PARU_OUT_OF_MEMORY) ;
            }
#ifndef NDEBUG
            paru_memset(Diag_map, 0, Sym->n * sizeof(int64_t),
                mem_chunk, nthreads) ;
            paru_memset(inv_Diag_map, 0, Sym->n * sizeof(int64_t),
                mem_chunk, nthreads) ;
            PR = 2;
#endif
        }
    }

    int64_t snz = Num->snz = Sym->snz;
    double *Sx = NULL;
    Sx = Num->Sx = PARU_MALLOC (snz, double);
    cSp = PARU_MALLOC (m + 1, int64_t);
    if (Num->Sx == NULL)
    {
        FREE_WORK ;
        return (PARU_OUT_OF_MEMORY) ;
    }

    cs1 = Sym->cs1;
    rs1 = Sym->rs1;
    if (cs1 > 0)
    {
        Num->sunz = Sym->ustons.nnz;
        Num->Sux = PARU_MALLOC (Num->sunz, double);
        cSup = PARU_MALLOC (cs1 + 1, int64_t);
        if (cSup == NULL || Num->Sux == NULL)
        {
            FREE_WORK ;
            return (PARU_OUT_OF_MEMORY) ;
        }
    }
    double *Sux = Num->Sux ;

    if (rs1 > 0)
    {
        Num->slnz = Sym->lstons.nnz;
        Num->Slx = PARU_MALLOC (Num->slnz, double);
        cSlp = PARU_MALLOC (rs1 + 1, int64_t);
        if (cSlp == NULL || Num->Slx == NULL)
        {
            FREE_WORK ;
            return (PARU_OUT_OF_MEMORY) ;
        }
    }
    double *Slx = Num->Slx ;

    bool prescaling = (prescale != PARU_PRESCALE_NONE) ;
    if (prescaling)
    {
        // S will be scaled by the maximum absolute value in each row
        Num->Rs = PARU_CALLOC (Sym->m, double);
        if (Num->Rs == NULL)
        {
            FREE_WORK ;
            return (PARU_OUT_OF_MEMORY) ;
        }
    }
    double *Rs = Num->Rs ;

    // Initializations
    if (nf != 0)
    {
        PRLEVEL(PR, ("%% $RowList =%p\n", RowList));
        paru_memset(rowSize, -1, m * sizeof(int64_t), mem_chunk, nthreads) ;
        PRLEVEL(PR, ("%% rowSize pointer=%p size=" LD " \n", rowSize,
                     m * sizeof(int64_t)));

        PRLEVEL(PR, ("%% rowMark pointer=%p size=" LD " \n", rowMark,
                     (m + nf) * sizeof(int64_t)));

        paru_memset(elRow, -1, (m + nf) * sizeof(int64_t),
                mem_chunk, nthreads) ;
        PRLEVEL(PR, ("%% elRow=%p\n", elRow));

        paru_memset(elCol, -1, (m + nf) * sizeof(int64_t),
                mem_chunk, nthreads) ;
        PRLEVEL(PR, ("%% elCol=%p\n", elCol));

        PRLEVEL(PR, ("%% Work =%p\n ", Work));
    }

    //////////////////Initializing numerics Sx, Sux and Slx //////////////////{
    int64_t *Ap = static_cast<int64_t*>(A->p);
    int64_t *Ai = static_cast<int64_t*>(A->i);
    double *Ax = static_cast<double*>(A->x);
    const int64_t *Sp = Sym->Sp;
    const int64_t *Slp = (rs1 > 0) ? Sym->lstons.Slp : NULL ;
    const int64_t *Sup = (cs1 > 0) ? Sym->ustons.Sup : NULL ;
    paru_memcpy(cSp, Sp, (m + 1) * sizeof(int64_t), mem_chunk, nthreads) ;
    if (cs1 > 0)
    {
        paru_memcpy(cSup, Sup, (cs1 + 1) * sizeof(int64_t),
                mem_chunk, nthreads) ;
    }
    if (rs1 > 0)
    {
        paru_memcpy(cSlp, Slp, (rs1 + 1) * sizeof(int64_t),
                mem_chunk, nthreads) ;
    }
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Init Sup and Slp in the middle\n"));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Sup =", Num->sunz));
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
        PRLEVEL(PR, ("(" LD ") Slp =", Num->slnz));
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

    const int64_t *Qinit = Sym->Qfill;
    const int64_t *Pinv = Sym->Pinv;
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Init Pinv =\n"));
    for (int64_t i = 0; i < m; i++) PRLEVEL(PR, ("" LD " ", Pinv[i]));
    PRLEVEL(PR, ("\n"));
#endif

    // compute the scale factors
    if (prescale == PARU_PRESCALE_MAX)
    {
        // this is the ParU default
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
    else if (prescale == PARU_PRESCALE_SUM)
    {
        // this is the UMFPACK default
        for (int64_t newcol = 0; newcol < Sym->n; newcol++)
        {
            int64_t oldcol = Qinit[newcol];
            for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
            {
                int64_t oldrow = Ai[p];
                Rs[oldrow] += fabs(Ax[p]);
            }
        }
    }

    PRLEVEL(PR, ("%% Rs:\n["));
    if (prescaling)
    {
        // making sure that every row has at most one element more than zero
        for (int64_t k = 0; k < m; k++)
        {
            PRLEVEL(PR, ("%lf ", Rs[k]));
            if (Rs[k] <= 0)
            {
                PRLEVEL(1, ("ParU: Matrix is singular, row " LD
                    " is zero\n", k));
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
            {
                // it is inside S otherwise it is part of singleton
                Sx[cSp[srow]++] =
                    (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
            }
            else if (srow < 0 && scol >= 0)
            {
                // inside the U singletons
                PRLEVEL(PR, ("Usingleton newcol = " LD " newrow=" LD "\n",
                    newcol, newrow));
                // let the diagonal entries be first
                Sux[++cSup[newrow]] =
                    (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
            }
            else
            {
                if (newrow < cs1)
                {
                    // inside U singletons CSR
                    // PRLEVEL(PR, ("Inside U singletons\n"));
                    if (newcol == newrow)
                    {
                        // diagonal entry
                        Sux[Sup[newrow]] =
                            (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
                    }
                    else
                    {
                        Sux[++cSup[newrow]] =
                            (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
                    }
                }
                else
                {
                    // inside L singletons CSC
                    // PRLEVEL(PR, ("Inside L singletons\n"));
                    if (newcol == newrow)
                    {
                        // diagonal entry
                        Slx[Slp[newcol - cs1]] =
                            (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
                    }
                    else
                    {
                        Slx[++cSlp[newcol - cs1]] =
                            (prescaling) ? (Ax[p] / Rs[oldrow]) : Ax[p];
                    }
                }
            }
        }
    }

    FREE_WORK;

        //////////////////Initializing numerics Sx, Sux and Slx
        /////////////////////}
#ifdef COUNT_FLOPS
    // flop count init
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
        int64_t nn = std::min (Sym->n, (int64_t) 64) ;
        for (int64_t i = 0; i < nn ; i++)
        {
            PRLEVEL(PR, ("" LD " ", Diag_map[i]));
        }
        PRLEVEL(PR, ("\n"));
        PRLEVEL(PR, ("inv_Diag_map =\n"));
        for (int64_t i = 0; i < nn ; i++)
        {
            PRLEVEL(PR, ("" LD " ", inv_Diag_map[i]));
        }
        PRLEVEL(PR, ("\n"));
        for (int64_t i = 0; i < Sym->n; i++)
        {
            if (Diag_map[i] == -1)
            {
                PRLEVEL(PR, ("Diag_map[" LD
                    "] is not correctly initialized\n", i));
            }
            if (inv_Diag_map[i] == -1)
            {
                PRLEVEL(PR, ("inv_Diag_map[" LD
                    "] is not correctly initialized\n", i));
            }
            ASSERT(Diag_map[i] != -1);
            // ASSERT(inv_Diag_map[i] != -1);
        }
        PR = 1;
#endif
    }

    // allocating row tuples, elements and updating column tuples

    int64_t out_of_memory = 0;

    #pragma omp parallel for num_threads(nthreads)
    for (int64_t row = 0; row < m; row++)
    {
        int64_t e = Sym->row2atree[row];
        int64_t nrows = 1 ;                     // # rows in front e
        int64_t ncols = Sp[row + 1] - Sp[row];  // # cols in front e

        row_degree_bound[row] = ncols;  // Initialzing row degree

        paru_element *curEl = elementList[e] =
            paru_create_element(nrows, ncols);
        if (curEl == NULL)
        {
            // out of memory
            PRLEVEL(1, ("ParU: Out of memory: curEl\n"));
            #pragma omp atomic update
            out_of_memory++ ;
        }
        else
        {
            try
            {
                rowMark[e] = 0;

                // new is calling paru_malloc
                std::vector<int64_t> *curHeap = Work->heapList[e] =
                    new std::vector<int64_t>;

                curHeap->push_back(e);

                // constants for initialzing lists
                int64_t slackRow = 2;

                // Allocating Rowlist and updating its tuples
                RowList[row].list = PARU_MALLOC (slackRow * nrows, paru_tuple);
                if (RowList[row].list == NULL)
                {
                    // out of memory
                    PRLEVEL(1, ("ParU: out of memory, RowList[row].list \n"));
                    #pragma omp atomic update
                    out_of_memory++ ;
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
                        out_of_memory++ ;
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
            {
                // out of memory
                PRLEVEL(1, ("ParU: Out of memory: curHeap\n"));
                #pragma omp atomic update
                out_of_memory++ ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ParU_Info info ;
    if (out_of_memory > 0)
    {
        info = PARU_OUT_OF_MEMORY;
    }
    else
    {
        info = PARU_SUCCESS;
    }

    PRLEVEL(1, ("];\n"));
    PRLEVEL(1, ("I = InMatrix(:,1);\n"));
    PRLEVEL(1, ("J = InMatrix(:,2);\n"));
    PRLEVEL(1, ("X = InMatrix(:,3);\n"));
    PRLEVEL(1, ("S = sparse(I,J,X);\n"));

    return (info) ;
}
