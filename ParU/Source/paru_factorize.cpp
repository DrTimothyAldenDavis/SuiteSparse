////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Factorize /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief    get a matrix and factorize it
 *      specify the order of eliminating fronts
 *      Allocate space for Num
 *      the user should free the space
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// paru_exec_tasks_seq: execute all tasks on a single thread
//------------------------------------------------------------------------------

ParU_Ret paru_exec_tasks_seq(int64_t t, int64_t *task_num_child, paru_work *Work,
                             ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *task_parent = Sym->task_parent;
    int64_t daddy = task_parent[t];
    int64_t *task_map = Sym->task_map;

    int64_t num_original_children = 0;
    if (daddy != -1) num_original_children = Sym->task_num_child[daddy];
    PRLEVEL(1, ("Seq: executing task " LD " fronts " LD "-" LD " (" LD " children)\n", t,
                task_map[t] + 1, task_map[t + 1], num_original_children));
    ParU_Ret myInfo;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    for (int64_t f = task_map[t] + 1; f <= task_map[t + 1]; f++)
    {
        PRLEVEL(2, ("Seq: calling " LD "\n", f));
        myInfo = paru_front(f, Work, Num);
        if (myInfo != PARU_SUCCESS)
        {
            return myInfo;
        }
    }
    int64_t num_rem_children;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("task time task " LD " is %lf\n", t, time));
#endif

#ifndef NDEBUG
    if (daddy == -1) PRLEVEL(1, ("%% finished task root(" LD ")\n", t));
#endif

    if (daddy != -1)  // if it is not a root
    {
        if (num_original_children != 1)
        {
            task_num_child[daddy]--;
            num_rem_children = task_num_child[daddy];

            PRLEVEL(1,
                    ("%%Seq finished task " LD "(" LD "," LD ")Parent has " LD " left\n", t,
                     task_map[t] + 1, task_map[t + 1], task_num_child[daddy]));
            if (num_rem_children == 0)
            {
                PRLEVEL(
                    1, ("%%Seq task " LD " executing its parent " LD "\n", t, daddy));
                return myInfo = paru_exec_tasks_seq(daddy, task_num_child, Work,
                                                    Num);
            }
        }
        else  // I was the only spoiled kid in the family;
        {
            PRLEVEL(1, ("%% Seq task " LD " only child executing its parent " LD "\n",
                        t, daddy));
            return myInfo =
                       paru_exec_tasks_seq(daddy, task_num_child, Work, Num);
        }
    }
    return myInfo;
}

//------------------------------------------------------------------------------
// paru_exec_tasks: execute all tasks in parallel
//------------------------------------------------------------------------------

ParU_Ret paru_exec_tasks(int64_t t, int64_t *task_num_child, int64_t &chain_task,
                         paru_work *Work, ParU_Numeric *Num)
{
    ParU_Symbolic *Sym = Work->Sym;
    int64_t *task_parent = Sym->task_parent;
    int64_t daddy = task_parent[t];
    int64_t *task_map = Sym->task_map;

    int64_t num_original_children = 0;
    if (daddy != -1) num_original_children = Sym->task_num_child[daddy];
    PRLEVEL(1, ("executing task " LD " fronts " LD "-" LD " (" LD " children)\n", t,
                task_map[t] + 1, task_map[t + 1], num_original_children));
    ParU_Ret myInfo;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    for (int64_t f = task_map[t] + 1; f <= task_map[t + 1]; f++)
    {
        myInfo = paru_front(f, Work, Num);
        if (myInfo != PARU_SUCCESS)
        {
            return myInfo;
        }
    }
    int64_t num_rem_children;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("task time task " LD " is %lf\n", t, time));
#endif

#ifndef NDEBUG
    if (daddy == -1) PRLEVEL(1, ("%% finished task root(" LD ")\n", t));
#endif

    if (daddy != -1)  // if it is not a root
    {
        if (num_original_children != 1)
        {
            #pragma omp atomic capture
            {
                task_num_child[daddy]--;
                num_rem_children = task_num_child[daddy];
            }

            PRLEVEL(1,
                    ("%% finished task " LD "(" LD "," LD ")  Parent has " LD " left\n", t,
                     task_map[t] + 1, task_map[t + 1], task_num_child[daddy]));
            if (num_rem_children == 0)
            {
                PRLEVEL(1,
                        ("%% task " LD " executing its parent " LD "\n", t, daddy));
                int64_t resq;
                #pragma omp atomic read
                resq = Work->resq;
                if (resq == 1)
                {
                    chain_task = daddy;
                    PRLEVEL(2, ("%% CHAIN ALERT1: task " LD " calling " LD ""
                                " resq = " LD "\n",
                                t, daddy, resq));
                }
                else
                {
                    return myInfo = paru_exec_tasks(daddy, task_num_child,
                                                    chain_task, Work, Num);
                }
            }
        }
        else  // I was the only spoiled kid in the family;
        {
            PRLEVEL(1, ("%% task " LD " only child executing its parent " LD "\n", t,
                        daddy));
            int64_t resq;
            #pragma omp atomic read
            resq = Work->resq;

            if (resq == 1)
            {
                chain_task = daddy;
                PRLEVEL(2, ("%% CHAIN ALERT1: task " LD " calling " LD ""
                            " resq = " LD "\n",
                            t, daddy, resq));
            }
            else
            {
                return myInfo = paru_exec_tasks(daddy, task_num_child,
                                                chain_task, Work, Num);
            }
        }
    }
    return myInfo;
}

//------------------------------------------------------------------------------
// ParU_Factorize: factorize a sparse matrix A
//------------------------------------------------------------------------------

ParU_Ret ParU_Factorize(cholmod_sparse *A, ParU_Symbolic *Sym,
                        ParU_Numeric **Num_handle, ParU_Control *user_Control)
{
    PARU_DEFINE_PRLEVEL;
#ifndef NTIME
    double my_start_time = PARU_OPENMP_GET_WTIME;
#endif
    if (A == NULL)
    {
        PRLEVEL(1, ("ParU: input matrix is invalid\n"));
        return PARU_INVALID;
    }

    if (A->xtype != CHOLMOD_REAL)
    {
        PRLEVEL(1, ("ParU: input matrix must be real\n"));
        return PARU_INVALID;
    }

    if (Sym == NULL)
    {
        return PARU_INVALID;
    }

    ParU_Ret info;
    // populate my_Control with tested values of Control
    ParU_Control my_Control = *user_Control;
    {
        int64_t panel_width = my_Control.panel_width;
        if (panel_width < 0 || panel_width > Sym->m)
            my_Control.panel_width = 32;
        int64_t paru_strategy = my_Control.paru_strategy;
        // at this point the strategy should be known
        // if the user didnot decide I
        if (paru_strategy == PARU_STRATEGY_AUTO)  // user didn't specify
            // so I use the same strategy as umfpack
            my_Control.paru_strategy = Sym->strategy;
        else if (paru_strategy != PARU_STRATEGY_SYMMETRIC &&
                 paru_strategy != PARU_STRATEGY_UNSYMMETRIC)
            // user input is not correct so I go to default
            my_Control.paru_strategy = Sym->strategy;
        // else user already picked symmetric or unsymmetric
        // and it has been copied over

        double piv_toler = my_Control.piv_toler;
        if (piv_toler > 1 || piv_toler < 0) my_Control.piv_toler = .1;
        double diag_toler = my_Control.diag_toler;
        if (diag_toler > 1 || diag_toler < 0) my_Control.diag_toler = .001;
        int64_t trivial = my_Control.trivial;
        if (trivial < 0) my_Control.trivial = 4;
        int64_t worthwhile_dgemm = my_Control.worthwhile_dgemm;
        if (worthwhile_dgemm < 0) my_Control.worthwhile_dgemm = 512;
        int64_t worthwhile_trsm = my_Control.worthwhile_trsm;
        if (worthwhile_trsm < 0) my_Control.worthwhile_trsm = 4096;
        int32_t max_threads = PARU_OPENMP_MAX_THREADS;
        if (my_Control.paru_max_threads > 0)
            my_Control.paru_max_threads =
                std::min(max_threads, my_Control.paru_max_threads);
        else
            my_Control.paru_max_threads = max_threads;

        int64_t scale = my_Control.scale;
        if (scale != 0 && scale != 1) my_Control.scale = 1;
    }
    ParU_Control *Control = &my_Control;

    paru_work myWork;
    paru_work *Work;
    Work = &myWork;

    #pragma omp atomic write
    Work->naft = 0;
    ParU_Numeric *Num;
    Num = *Num_handle;

    info = paru_init_rowFronts(Work, &Num, A, Sym, Control);
    *Num_handle = Num;

    PRLEVEL(1, ("%% init_row is done\n"));
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% init_row has a problem\n"));
        paru_free_work(Sym, Work);   // free the work DS
        ParU_Freenum(Num_handle, Control);
        return info;
    }
    int64_t nf = Sym->nf;
    //////////////// Using task tree //////////////////////////////////////////
    int64_t ntasks = Sym->ntasks;
    int64_t *task_depth = Sym->task_depth;
    std::vector<int64_t> task_Q;

    int64_t *task_num_child ;
    #pragma omp atomic write
    task_num_child = Work->task_num_child;
    paru_memcpy(task_num_child, Sym->task_num_child, ntasks * sizeof(int64_t),
                Control);
    try
    {
        for (int64_t t = 0; t < ntasks; t++)
        {
            if (task_num_child[t] == 0) task_Q.push_back(t);
        }
    }
    catch (std::bad_alloc const &)
    {  // out of memory
        PRLEVEL(1, ("ParU: Out of memory: task_Q\n"));
        paru_free_work(Sym, Work);   // free the work DS
        ParU_Freenum(Num_handle, Control);
        return PARU_OUT_OF_MEMORY;
    }
    std::sort(task_Q.begin(), task_Q.end(),
            [&task_depth](const int64_t &t1, const int64_t &t2) -> bool {
            return task_depth[t1] > task_depth[t2];
            });
    #pragma omp atomic write
    Work->resq = task_Q.size();

#ifndef NDEBUG
    double chainess = 2;
    PRLEVEL(1, ("ntasks=" LD " task_Q.size=" LD "\n", ntasks, task_Q.size()));
    if (ntasks > 0)
    {
        // chainess = (task_depth[task_Q[0]] + 1) / (double)nf;
        chainess = 1 - (task_Q.size() / (double)ntasks);
        PRLEVEL(1, ("nf = " LD ", deepest = " LD ", chainess = %lf \n", nf,
                    task_depth[task_Q[0]], chainess));
    }
    #pragma omp atomic write
    Work->actual_alloc_LUs = 0 ;
    #pragma omp atomic write
    Work->actual_alloc_Us = 0;
    #pragma omp atomic write
    Work->actual_alloc_row_int = 0 ;
    #pragma omp atomic write
    Work->actual_alloc_col_int = 0;
    PR = 1;
    int64_t *task_map = Sym->task_map;
    PRLEVEL(PR, ("\n%% task_Q:\n"));
    for (int64_t i = 0; i < (int64_t)task_Q.size(); i++)
    {
        int64_t t = task_Q[i];
        PRLEVEL(PR, ("" LD "[" LD "-" LD "](" LD ") ", t, task_map[t] + 1, task_map[t + 1],
                    task_depth[t]));
    }
    PRLEVEL(PR, ("\n"));
#endif

    //--------------------------------------------------------------------------
    // execute the task tree
    //--------------------------------------------------------------------------

#if ! defined ( PARU_GCC_WINDOWS )
    // The parallel factorization gets stuck intermittently on Windows with GCC.
    // Use the sequential factorization unconditionally in that case.
    if (task_Q.size() * 2 > Control->paru_max_threads)
    {
        PRLEVEL(1, ("Parallel\n"));
        // chekcing user input
        PRLEVEL(1, ("Control: max_th=" LD " scale=" LD " piv_toler=%lf "
                    "diag_toler=%lf trivial =" LD " worthwhile_dgemm=" LD " "
                    "worthwhile_trsm=" LD "\n",
                    Control->paru_max_threads, Control->scale,
                    Control->piv_toler, Control->diag_toler, Control->trivial,
                    Control->worthwhile_dgemm, Control->worthwhile_trsm));

#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )
        PARU_OPENMP_SET_DYNAMIC(0);
        mkl_set_dynamic(0);
        // mkl_set_threading_layer(MKL_THREADING_INTEL);
        // mkl_set_interface_layer(MKL_INTERFACE_ILP64);
#endif
        BLAS_set_num_threads(1);
        PARU_OPENMP_SET_MAX_ACTIVE_LEVELS(4);
        const int64_t size = (int64_t)task_Q.size();
        const int64_t steps = size == 0 ? 1 : size;
        const int64_t stages = size / steps + 1;
        int64_t chain_task = -1;
        int64_t start = 0;
        PRLEVEL(
            1, ("%% size=" LD ", steps =" LD ", stages =" LD "\n", size, steps, stages));

        for (int64_t ii = 0; ii < stages; ii++)
        {
            if (start >= size) break;
            int64_t end = start + steps > size ? size : start + steps;
            PRLEVEL(1, ("%% doing Queue tasks <" LD "," LD ">\n", start, end));
            #pragma omp parallel proc_bind(spread)                             \
            num_threads(Control->paru_max_threads)
            #pragma omp single nowait
            #pragma omp task untied  // clang might seg fault on untied
            for (int64_t i = start; i < end; i++)
            // for (int64_t i = 0; i < (int64_t)task_Q.size(); i++)
            {
                int64_t t = task_Q[i];
                int64_t d = task_depth[t];
                #pragma omp task mergeable priority(d)
                {
                    #pragma omp atomic update
                    Work->naft++;

                    ParU_Ret myInfo =
                        paru_exec_tasks(t, task_num_child, chain_task, Work,
                            Num);
                    if (myInfo != PARU_SUCCESS)
                    {
                        #pragma omp atomic write
                        info = myInfo;
                    }
                    #pragma omp atomic update
                    Work->naft--;

                    #pragma omp atomic update
                    Work->resq--;
                }
            }
            start += steps;
        }
        // chain break
        if (chain_task != -1 && info == PARU_SUCCESS)
        {
            #pragma omp atomic write
            Work->naft = 1;
            PRLEVEL(1, ("Chain_taskd " LD " has remained\n", chain_task));
            info = paru_exec_tasks_seq(chain_task, task_num_child, Work, Num);
        }
        if (info != PARU_SUCCESS)
        {
            PRLEVEL(1, ("%% factorization has some problem\n"));
            if (info == PARU_OUT_OF_MEMORY)
            {
                PRLEVEL(1, ("ParU: out of memory during factorization\n"));
            }
            else if (info == PARU_SINGULAR)
            {
                PRLEVEL(1, ("ParU: Input matrix is singular\n"));
            }
            paru_free_work(Sym, Work);   // free the work DS
            ParU_Freenum(Num_handle, Control);
            return info;
        }
    }
    else
#endif
    {
        PRLEVEL(1, ("Sequential\n"));
        Work->naft = 1;
        for (int64_t i = 0; i < nf; i++)
        {
            // if (i %1000 == 0) PRLEVEL(1, ("%% Wroking on front " LD "\n", i));

            info = paru_front(i, Work, Num);
            if (info != PARU_SUCCESS)
            {
                PRLEVEL(1, ("%% A problem happend in " LD "\n", i));
                paru_free_work(Sym, Work);   // free the work DS
                ParU_Freenum(Num_handle, Control);
                return info;
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the permutation
    //--------------------------------------------------------------------------

    PRLEVEL(1, ("finalize permutation\n"));
    info = paru_finalize_perm(Sym, Num);  // to form the final permutation
    paru_free_work(Sym, Work);   // free the work DS
    Num->Control = NULL;

    if (info == PARU_OUT_OF_MEMORY)
    {
        PRLEVEL(
            1, ("ParU: memory problem after factorizaiton, in perumutaion.\n"));
        ParU_Freenum(Num_handle, Control);
        return info;
    }

#ifdef COUNT_FLOPS
    double flop_count =
        Num->flp_cnt_dgemm + Num->flp_cnt_dger + Num->flp_cnt_trsm;
    PRLEVEL(-1, ("Flop count = %.17g\n", flop_count));
#endif
    int64_t max_rc = 0, max_cc = 0;
    double min_udiag = 1, max_udiag = -1;  // not to fail for nf ==0

    // using the first value of the first front just to initialize
    if (nf > 0)
    {
        ParU_Factors *LUs = Num->partial_LUs;
        max_udiag = min_udiag = fabs(*(LUs[0].p));
        #ifdef PARU_COVERAGE
        #define M1 1000
        #else
        #define M1 65536 
        #endif
        if (Num-> m < M1)
        { //Serial
            for (int64_t f = 0; f < nf; f++)
            {
                int64_t rowCount = Num->frowCount[f];
                int64_t colCount = Num->fcolCount[f];
                int64_t *Super = Sym->Super;
                int64_t col1 = Super[f];
                int64_t col2 = Super[f + 1];
                int64_t fp = col2 - col1;
                max_rc = std::max(max_rc, rowCount);
                max_cc = std::max(max_cc, colCount + fp);
                double *X = LUs[f].p;
                for (int64_t i = 0; i < fp; i++)
                {
                    double udiag = fabs(X[rowCount * i + i]);
                    min_udiag = std::min(min_udiag, udiag);
                    max_udiag = std::max(max_udiag, udiag);
                }
            }
        }
        else 
        { //Parallel
            int64_t *Super = Sym->Super;
            #pragma omp parallel for reduction(max:max_rc)    \
            reduction(max: max_cc) if (nf > 65536)            \
            num_threads(Control->paru_max_threads)
            for (int64_t f = 0; f < nf; f++)
            {
                int64_t rowCount = Num->frowCount[f];
                int64_t colCount = Num->fcolCount[f];
                int64_t col1 = Super[f];
                int64_t col2 = Super[f + 1];
                int64_t fp = col2 - col1;
                max_rc = std::max(max_rc, rowCount);
                max_cc = std::max(max_cc, colCount + fp);
            }

            for (int64_t f = 0; f < nf; f++)
            {
                int64_t rowCount = Num->frowCount[f];
                int64_t col1 = Super[f];
                int64_t col2 = Super[f + 1];
                int64_t fp = col2 - col1;
                double *X = LUs[f].p;
                #pragma omp parallel for reduction(min:min_udiag) \
                reduction(max: max_udiag)                         \
                num_threads(Control->paru_max_threads)
                for (int64_t i = 0; i < fp; i++)
                {
                    double udiag = fabs(X[rowCount * i + i]);
                    min_udiag = std::min(min_udiag, udiag);
                    max_udiag = std::max(max_udiag, udiag);
                }
            }
        }
    }

    PRLEVEL(1, ("max_rc=" LD " max_cc=" LD "\n", max_rc, max_cc));
    PRLEVEL(1, ("max_udiag=%e min_udiag=%e rcond=%e\n", max_udiag, min_udiag,
                min_udiag / max_udiag));
    Num->max_row_count = max_rc;
    Num->max_col_count = max_cc;
    Num->min_udiag = min_udiag;
    Num->max_udiag = max_udiag;
    Num->rcond = min_udiag / max_udiag;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= my_start_time;
    PRLEVEL(1, ("factorization time took is %lf\n", time));
#endif
    return Num->res;
}

