////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Factorize /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

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
// paru_frontal_flops
//------------------------------------------------------------------------------

// flop count for a single frontal matrix, size (m+k)-by-(n+k) with k pivots:
//
//       F = F1 F2
//           F3 F4
//
//   F1 is k-by-k and becomes [L1, U1] with unit diagonal of L1
//   F2 is k-by-n and because U2
//   F3 is m-by-k and becomes L2
//   F4 is m-by-n and becomes the Schur complement S, to be
//       assembled into the next frontal matrix
//
//   Factorization is:
//
//       [L1,U1] U2
//       L2      S

static void paru_frontal_flops
(
    // frontal matrix is (m+k)-by-(n+k) with k pivots
    int64_t k,                  // number of pivots
    int64_t m,                  // # of rows in Schur complement
    int64_t n,                  // # of cols in Schur complement
    // input/output:
    double *fl,                 // add the flops for this frontal matrix
    int64_t *lnz,               // add the # of entries in L for this front
    int64_t *unz                // add the # of entries in U for this front
)
{
    double K = (double) k ;
    double M = (double) m ;
    double N = (double) n ;
    double K_squared = K*K ;
    double K_cubed = K_squared*K ;

    // (1) LU factorization of the k-by-k leading submatrix:
    double f = (4*K_cubed - 3*K_squared - K) / 6 ;
    // (2) triangular solve to compute L2 the block lower part of L,
    // using a triangular solve with U1 (so count the diagonal):
    f += K_squared * M ;            // (K*(K-1) + K) * M ;
    // (3) triangular solve with L1 to compute U2 (do not count diagonal
    // terms since L1 has unit diagonal):
    f += (K_squared - K) * N ;      // (K*(K-1)    ) * N ;
    // (4) apply the k pivots to S (S -= L2*U2 as a single dgemm)
    f += 2 * K * M * N ;

    // # of entries in L for this front, including diagonal
    int64_t tril = (k*(k-1))/2 + k ;
    int64_t nzl = tril + k*m ;
    // # of entries in U for this front, including diagonal
    int64_t nzu = tril + k*n ;

    (*fl) += f ;
    (*lnz) += nzl ;
    (*unz) += nzu ;
}

//------------------------------------------------------------------------------
// ParU_Factorize: factorize a sparse matrix A
//------------------------------------------------------------------------------

ParU_Info ParU_Factorize
(
    // input:
    cholmod_sparse *A,  // input matrix to factorize
    const ParU_Symbolic Sym,  // symbolic analsys from ParU_Analyze
    // output:
    ParU_Numeric *Num_handle,
    // control:
    ParU_Control Control
)
{
    if (!A || !Sym || !Num_handle ||
        A->xtype != CHOLMOD_REAL || A->dtype != CHOLMOD_DOUBLE)
    {
        return (PARU_INVALID) ;
    }

    ParU_Info info ;
    PARU_DEFINE_PRLEVEL;
#ifndef NTIME
    double my_start_time = PARU_OPENMP_GET_WTIME;
#endif

    // get Control
    paru_work myWork ;
    paru_work *Work = &myWork ;
    Work->nthreads         = paru_nthreads (Control) ;
    Work->mem_chunk        = PARU_DEFAULT_MEM_CHUNK ;
    Work->worthwhile_dgemm = PARU_DEFAULT_DGEMM_TASKED ;
    Work->worthwhile_dtrsm = PARU_DEFAULT_DTRSM_TASKED ;
    Work->trivial          = PARU_DEFAULT_DGEMM_TINY ;
    Work->panel_width      = PARU_DEFAULT_PANEL_WIDTH ;
    Work->piv_toler        = PARU_DEFAULT_PIVOT_TOLERANCE ;
    Work->diag_toler       = PARU_DEFAULT_DIAG_PIVOT_TOLERANCE ;
    Work->prescale         = PARU_DEFAULT_PRESCALE ;
    if (Control != NULL)
    {
        Work->mem_chunk        = Control->mem_chunk ;
        Work->trivial          = Control->trivial ;
        Work->worthwhile_dgemm = Control->worthwhile_dgemm ;
        Work->worthwhile_dtrsm = Control->worthwhile_dtrsm ;
        Work->panel_width      = Control->panel_width ;
        Work->piv_toler        = Control->piv_toler ;
        Work->diag_toler       = Control->diag_toler ;
        Work->prescale         = Control->prescale ;
    }

    int32_t nthreads = Work->nthreads ;
    size_t mem_chunk = Work->mem_chunk ;

    #pragma omp atomic write
    Work->naft = 0;
    ParU_Numeric Num = *Num_handle ;

    info = paru_init_rowFronts(Work, &Num, A, Sym);
    *Num_handle = Num;

    PRLEVEL(1, ("%% init_row is done\n"));
    if (info != PARU_SUCCESS)
    {
        PRLEVEL(1, ("%% init_row has a problem\n"));
        paru_free_work(Sym, Work);   // free the work DS
        ParU_FreeNumeric(Num_handle, Control);
        return info;
    }
    int64_t nf = Sym->nf;
    //////////////// Using task tree //////////////////////////////////////////
    int64_t ntasks = Sym->ntasks;
    const int64_t *task_depth = Sym->task_depth;
    std::vector<int64_t> task_Q;

    int64_t *task_num_child ;
    #pragma omp atomic write
    task_num_child = Work->task_num_child;
    paru_memcpy(task_num_child, Sym->task_num_child, ntasks * sizeof(int64_t),
                mem_chunk, nthreads) ;

    try
    {
        for (int64_t t = 0; t < ntasks; t++)
        {
            if (task_num_child[t] == 0) task_Q.push_back(t);
        }
    }
    catch (std::bad_alloc const &)
    {
        // out of memory
        PRLEVEL(1, ("ParU: Out of memory: task_Q\n"));
        paru_free_work(Sym, Work);   // free the work DS
        ParU_FreeNumeric(Num_handle, Control);
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
    const int64_t *task_map = Sym->task_map;
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

#if ! defined ( PARU_1TASK )
    // The parallel factorization gets stuck intermittently on Windows or Mac
    // with gcc, so always use the sequential factorization in that case.
    // This case is handled by cmake.
    if (task_Q.size() * 2 > ((long unsigned int) nthreads))
    {
        PRLEVEL(1, ("Parallel\n"));
        // checking user input

#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )
        PARU_OPENMP_SET_DYNAMIC(0);
        mkl_set_dynamic((int)0);
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
        PRLEVEL( 1, ("%% size=" LD ", steps =" LD ", stages =" LD "\n",
            size, steps, stages));

        for (int64_t ii = 0; ii < stages; ii++)
        {
            if (start >= size) break;
            int64_t end = start + steps > size ? size : start + steps;
            PRLEVEL(1, ("%% doing Queue tasks <" LD "," LD ">\n", start, end));
            #pragma omp parallel proc_bind(spread) num_threads(nthreads)
            #pragma omp single nowait
            #pragma omp task untied
            for (int64_t i = start; i < end; i++)
            {
                int64_t t = task_Q[i];
                int64_t d = task_depth[t];
                #pragma omp task mergeable priority(d)
                {
                    #pragma omp atomic update
                    Work->naft++;

                    ParU_Info myInfo =
                        paru_exec_tasks(t, task_num_child, chain_task, Work,
                            Sym, Num);
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
            info = paru_exec_tasks_seq(chain_task, task_num_child, Work,
                Sym, Num);
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
            ParU_FreeNumeric(Num_handle, Control);
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
            info = paru_front(i, Work, Sym, Num);
            if (info != PARU_SUCCESS)
            {
                PRLEVEL(1, ("%% A problem happend in " LD "\n", i));
                paru_free_work(Sym, Work);   // free the work DS
                ParU_FreeNumeric(Num_handle, Control);
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

    if (info == PARU_OUT_OF_MEMORY)
    {
        PRLEVEL(
            1, ("ParU: memory problem after factorizaiton, in perumutaion.\n"));
        ParU_FreeNumeric(Num_handle, Control);
        return info;
    }

    // FUTURE: add a routine that returns L and U as plain CSC or CSR
    // matrices.

#ifdef COUNT_FLOPS
    double flop_count =
        Num->flp_cnt_dgemm + Num->flp_cnt_dger + Num->flp_cnt_trsm;
    PRLEVEL(-1, ("Flop count = %.17g\n", flop_count));
#endif
    int64_t max_rc = 0, max_cc = 0;
    double min_udiag = 1, max_udiag = -1;  // not to fail for nf ==0
    int64_t nnzL = Sym->lstons.nnz + Sym->cs1 ; // includes diagonal
    int64_t nnzU = Sym->ustons.nnz + Sym->rs1 ; // includes diagonal
    double sfc = 0.0; //simple flop count

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
        if (Num->m < M1)
        {
            //Serial
            for (int64_t f = 0; f < nf; f++)
            {
                int64_t rowCount = Num->frowCount[f];
                int64_t colCount = Num->fcolCount[f];
                const int64_t *Super = Sym->Super;
                int64_t col1 = Super[f];
                int64_t col2 = Super[f + 1];
                int64_t fp = col2 - col1;
                max_rc = std::max(max_rc, rowCount);
                max_cc = std::max(max_cc, colCount + fp);
                double *X = LUs[f].p;

                paru_frontal_flops (fp, rowCount - fp, colCount,
                    &sfc, &nnzL, &nnzU) ;

                for (int64_t i = 0; i < fp; i++)
                {
                    double udiag = fabs(X[rowCount * i + i]);
                    min_udiag = std::min(min_udiag, udiag);
                    max_udiag = std::max(max_udiag, udiag);
                }
            }
        }
        else
        {
            //Parallel
            const int64_t *Super = Sym->Super;
            #pragma omp parallel for reduction(max:max_rc)      \
                reduction(max: max_cc) if (nf > 65536)          \
                num_threads(nthreads)
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
                int64_t colCount = Num->fcolCount[f];
                int64_t col1 = Super[f];
                int64_t col2 = Super[f + 1];
                int64_t fp = col2 - col1;

                paru_frontal_flops (fp, rowCount - fp, colCount,
                    &sfc, &nnzL, &nnzU) ;

                double *X = LUs[f].p;
                #pragma omp parallel for reduction(min:min_udiag)   \
                    reduction(max: max_udiag)                       \
                    num_threads(nthreads)
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
    Num->nnzL = nnzL;
    Num->nnzU = nnzU;
    Num->sfc= sfc;
#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= my_start_time;
    PRLEVEL(1, ("factorization time took is %lf\n", time));
#endif
    return (Num->res) ;
}

