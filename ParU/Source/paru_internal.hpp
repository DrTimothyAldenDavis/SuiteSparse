////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_internal.hpp //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

#ifndef PARU_INTERNAL_H
#define PARU_INTERNAL_H
//!
//  internal libraries that are not visible to the user
//  @author Aznaveh
//

#define SUITESPARSE_BLAS_DEFINITIONS
#include "ParU.hpp"
#include "paru_omp.hpp"

// -----------------------------------------------------------------------------
// debugging and printing macros
// -----------------------------------------------------------------------------
// force debugging off
#ifndef NDEBUG
    #define NDEBUG
#endif

#ifndef NPR
    #define NPR
#endif

#ifndef NTIME
    #define NTIME
#endif

#define DLONG

extern "C"
{
#include "umf_internal.h"
#undef Int
}

// for printing information uncomment this; to activate assertions uncomment
// #undef NPR
// uncomment the following line to turn on debugging mode
// #undef NDEBUG
// uncomment the following line to turn on OpenMP timing
//#undef NTIME

#ifndef NDEBUG
    #undef NPR
#endif

// for printing int64_t types:
#define LD "%" PRId64

// uncomment if you want to count hardware flops
//#define COUNT_FLOPS

// defined somewhere else
#ifdef ASSERT
    #undef ASSERT
#endif
#ifndef NDEBUG
    #include <assert.h>
    #define ASSERT(e) assert(e)
#else
    #define ASSERT(e)
#endif

#ifndef NPR
static int print_level = -1;
    #define PRLEVEL(level, param)                   \
        {                                           \
            if (print_level >= level) printf param; \
        }
    #define DEBUGLEVEL(level)    \
        {                        \
            print_level = level; \
        }
    #define PARU_DEFINE_PRLEVEL int PR = 1
#else
    #define PRLEVEL(level, param)
    #define DEBUGLEVEL(level)
    #define PARU_DEFINE_PRLEVEL
#endif

#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )

    extern "C"
    {
        void MKL_Set_Num_Threads (int n) ;
        void MKL_Set_Num_Threads_Local (int n) ;
        void MKL_Set_Dynamic (int flag);
    }
    #define mkl_set_num_threads         MKL_Set_Num_Threads
    #define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local
    #define mkl_set_dynamic             MKL_Set_Dynamic
    #define BLAS_set_num_threads(n) mkl_set_num_threads(n)

#elif ( defined ( BLAS_OpenBLAS ) )

    // OpenBLAS
    extern "C"
    {
        void openblas_set_num_threads (int n) ;
    }
    #define BLAS_set_num_threads(n) openblas_set_num_threads(n)

#else

    // Generic BLAS
    #define BLAS_set_num_threads(n)

#endif

// These libraries are included in Suitesparse_config
//#include <stdlib.h>
//#include <math.h>
//#include <float.h>
//#include <stdio.h>
//#include <cstring>
//#include <malloc.h> // mallopt used in paru_init_rowFronts.cpp

// To be able to use set
#include <algorithm>
#include <set>
#include <vector>

// -----------------------------------------------------------------------------
// basic macros
// -----------------------------------------------------------------------------

#define EMPTY (-1)
// already defined in amd
// define TRUE 1
// define FALSE 0
#define IMPLIES(p, q) (!(p) || (q))

#define Size_max ((size_t)(-1))  // the largest value of size_t

// internal data structures
struct heaps_info
{
    int64_t sum_size, biggest_Child_size, biggest_Child_id;
};

// =============================================================================
//                   ParU_Tuple, Row data structure
// =============================================================================
struct paru_tuple
{
    // The (e,f) tuples for element lists
    int64_t e,  //  element number
        f;      //   offest
};

struct paru_tupleList
{                      // List of tuples
    int64_t numTuple,  //  number of Tuples in this element
        len;           //  length of allocated space for current list
    paru_tuple *list;  // list of tuples regarding to this element
};

// =============================================================================
//                  An element, contribution block
// =============================================================================
struct paru_element
{
    int64_t

        nrowsleft,  // number of rows remaining
        ncolsleft,  // number of columns remaining
        nrows,      // number of rows
        ncols,      // number of columns
        rValid,     // validity of relative row index
        cValid;     // validity of relative column index

    int64_t lac;  // least active column which is active
    // 0 <= lac <= ncols

    int64_t nzr_pc;  // number of zero rows in pivotal column of current front

    size_t size_allocated;
    // followed in memory by:
    //   int64_t
    //   col [0..ncols-1],  column indices of this element
    //   row [0..nrows-1] ; row indices of this element
    //
    //   relColInd [0..ncols-1];    relative indices of this element for
    //   current front
    //   relRowInd [0..nrows-1],    relative indices of this element for
    //   current front
    //   double ncols*nrows; numeric values
};

struct paru_work
{
    // gather scatter space for rows
    int64_t *rowSize;  // Initalized data structure, size of rows
    // int64_t rowMark;      // Work->rowSize[x] < rowMark[eli] for each front
    int64_t *rowMark;  // size = m+nf

    // gather scatter space for elements
    int64_t *elRow;  // Initalized data structure, size m+nf
    int64_t *elCol;  // Initalized data structure, size m+nf

    // only used for statistics when debugging is enabled:
    int64_t actual_alloc_LUs;      // actual memory allocated for LUs
    int64_t actual_alloc_Us;       // actual memory allocated for Us
    int64_t actual_alloc_row_int;  // actual memory allocated for rows
    int64_t actual_alloc_col_int;  // actual memory allocated for cols

#ifdef COUNT_FLOPS
    double flp_cnt_dgemm;
    double flp_cnt_trsm;
    double flp_cnt_dger;
    double flp_cnt_real_dgemm;
#endif

    paru_tupleList *RowList;  // size n of dynamic list
    int64_t *time_stamp;      // for relative index update; not initialized

    int64_t *Diag_map;  // size n,
    // Both of these are NULL if the stratey is not symmetric
    // copy of Diag_map from Sym;
    // this copy can be updated during the factorization
    int64_t *inv_Diag_map;  // size n,
    // inverse of Diag_map from Sym;
    // It helps editing the Diag_map

    int64_t *row_degree_bound;  // row degree size number of rows

    ParU_Symbolic *Sym;  // point to the symbolic that user sends

    paru_element **elementList;  // pointers to all elements, size = m+nf+1

    int64_t *lacList;  // size m+nf least active column of each element
                       //    el_colIndex[el->lac]  == lacList [e]
                       //    number of element
                       //
    // each active front owns and manage a heap list. The heap is based on the
    // least numbered column. The active front Takes the pointer of the biggest
    // child and release its other children after concatenating their list to
    // its own. The list of heaps are initialized by nullptr
    std::vector<int64_t> **heapList;  // size m+nf+1, initialized with nullptr

    int64_t *task_num_child;  // a copy of task_num_child of sym
    // this copy changes during factorization

    int64_t naft;  // number of actvie frontal tasks
    int64_t resq;  // number of remainig ready tasks in the queue
};

//------------------------------------------------------------------------------
// inline internal functions

inline int64_t *colIndex_pointer(paru_element *curEl)
{
    return (int64_t *)(curEl + 1);
}
// Never ever use these functions prior to initializing ncols and nrows
inline int64_t *rowIndex_pointer(paru_element *curEl)
{
    return (int64_t *)(curEl + 1) + curEl->ncols;
}

inline int64_t *relColInd(paru_element *curEl)
//{    return (int64_t*)(curEl+1) + curEl->ncols + curEl->nrows + 1;}
{
    return (int64_t *)(curEl + 1) + curEl->ncols + curEl->nrows;
}

inline int64_t *relRowInd(paru_element *curEl)
//{    return (int64_t*)(curEl+1) + 2*curEl->ncols + curEl->nrows + 2;}
{
    return (int64_t *)(curEl + 1) + 2 * curEl->ncols + curEl->nrows;
}

inline double *numeric_pointer(paru_element *curEl)
// sizeof int64_t and double are same, but I keep it like this for clarity
//{ return (double*)((int64_t*)(curEl+1) + 2*curEl->ncols + 2*curEl->nrows +
//2);}
{
    return (double *)((int64_t *)(curEl + 1) + 2 * curEl->ncols +
                      2 * curEl->nrows);
}

inline int64_t flip(int64_t colInd) { return -colInd - 2; }

inline int64_t lac_el(paru_element **elementList, int64_t eli)
{  // return least numbered column of the element i (eli)
    if (elementList[eli] == NULL)
        return LONG_MAX;
    else
    {
        int64_t *el_colIndex = (int64_t *)(elementList[eli] + 1);
        int64_t lac_ind = elementList[eli]->lac;
        return el_colIndex[lac_ind];
    }
}

inline int32_t control_nthreads (ParU_Control *Control)
{
    int32_t max_threads = PARU_OPENMP_MAX_THREADS ;
    if (Control == NULL || Control->paru_max_threads <= 0)
    {
        // default # of threads
        return (max_threads) ;
    }
    else
    {
        return std::min(max_threads, Control->paru_max_threads);
    }
}

inline int64_t control_mem_chunk (ParU_Control *Control)
{
    if (Control == NULL || Control->mem_chunk <= 0)
    {
        // default chunk size
        return (PARU_MEM_CHUNK) ;
    }
    else
    {
        return (Control->mem_chunk) ;
    }
}

//------------------------------------------------------------------------------
// internal routines
//
/* Wrappers for managing memory */
void *paru_alloc(size_t n, size_t size);
void *paru_calloc(size_t n, size_t size);
void *paru_realloc(size_t newsize, size_t size_Entry, void *oldP, size_t *size);

void paru_free(size_t n, size_t size, void *p);
void paru_free_el(int64_t e, paru_element **elementList);

void *operator new(std::size_t sz);
void operator delete(void *ptr) noexcept;

void paru_memset(void *ptr, int64_t value, size_t num, ParU_Control *Control);
void paru_memcpy(void *destination, const void *source, size_t num,
                 ParU_Control *Control);

#ifdef PARU_ALLOC_TESTING
bool paru_get_malloc_tracking(void);
void paru_set_malloc_tracking(bool track);
void paru_set_nmalloc(int64_t nmalloc);
int64_t paru_decr_nmalloc(void);
int64_t paru_get_nmalloc(void);
#endif

#ifndef NDEBUG
void paru_write(int scale, char *id, paru_work *Work, ParU_Numeric *Num);
void paru_print_element(int64_t e, paru_work *Work, ParU_Numeric *Num);
void paru_print_paru_tupleList(paru_tupleList *listSet, int64_t index);
#endif

/* add tuple functions defintions */
ParU_Ret paru_add_rowTuple(paru_tupleList *RowList, int64_t row, paru_tuple T);

ParU_Ret paru_factorize_full_summed(int64_t f, int64_t start_fac,
                                    std::vector<int64_t> &panel_row,
                                    std::set<int64_t> &stl_colSet,
                                    std::vector<int64_t> &pivotal_elements,
                                    paru_work *Work, ParU_Numeric *Num);

paru_element *paru_create_element(int64_t nrows, int64_t ncols);

void paru_assemble_row_2U(int64_t e, int64_t f, int64_t sR, int64_t dR,
                          std::vector<int64_t> &colHash, paru_work *Work,
                          ParU_Numeric *Num);

int64_t paru_trsm(int64_t f, double *pF, double *uPart, int64_t fp,
                  int64_t rowCount, int64_t colCount, paru_work *Work,
                  ParU_Numeric *Num);
int64_t paru_dgemm(int64_t f, double *pF, double *uPart, double *el, int64_t fp,
                   int64_t rowCount, int64_t colCount, paru_work *Work,
                   ParU_Numeric *Num);

void paru_init_rel(int64_t f, paru_work *Work);

void paru_update_rel_ind_col(int64_t e, int64_t f,
                             std::vector<int64_t> &colHash, paru_work *Work,
                             ParU_Numeric *Num);

void paru_update_rowDeg(int64_t panel_num, int64_t row_end, int64_t f,
                        int64_t start_fac, std::set<int64_t> &stl_colSet,
                        std::vector<int64_t> &pivotal_elements, paru_work *Work,
                        ParU_Numeric *Num);

int64_t paru_cumsum(int64_t n, int64_t *X, ParU_Control *Control);

int64_t paru_bin_srch_col(int64_t *srt_lst, int64_t l, int64_t r, int64_t num);
int64_t paru_bin_srch(int64_t *srt_lst, int64_t l, int64_t r, int64_t num);

ParU_Ret paru_init_rowFronts(paru_work *Work, ParU_Numeric **Num_handle,
                             cholmod_sparse *A, ParU_Symbolic *Sym,
                             ParU_Control *Control);
ParU_Ret paru_front(int64_t f, paru_work *Work, ParU_Numeric *Num);

ParU_Ret paru_pivotal(std::vector<int64_t> &pivotal_elements,
                      std::vector<int64_t> &panel_row, int64_t &zero_piv_rows,
                      int64_t f, heaps_info &hi, paru_work *Work,
                      ParU_Numeric *Num);

int paru_intersection(int64_t e, paru_element **elementList,
                      std::set<int64_t> &stl_colSet);

ParU_Ret paru_prior_assemble(int64_t f, int64_t start_fac,
                             std::vector<int64_t> &pivotal_elements,
                             std::vector<int64_t> &colHash, heaps_info &hi,
                             paru_work *Work, ParU_Numeric *Num);

void paru_assemble_all(int64_t e, int64_t f, std::vector<int64_t> &colHash,
                       paru_work *Work, ParU_Numeric *Num);

void paru_assemble_cols(int64_t e, int64_t f, std::vector<int64_t> &colHash,
                        paru_work *Work, ParU_Numeric *Num);

void paru_assemble_rows(int64_t e, int64_t f, std::vector<int64_t> &colHash,
                        paru_work *Work, ParU_Numeric *Num);

void paru_assemble_el_with0rows(int64_t e, int64_t f,
                                std::vector<int64_t> &colHash, paru_work *Work,
                                ParU_Numeric *Num);

void paru_full_summed(int64_t e, int64_t f, paru_work *Work, ParU_Numeric *Num);

// heap related
ParU_Ret paru_make_heap(int64_t f, int64_t start_fac,
                        std::vector<int64_t> &pivotal_elements, heaps_info &hi,
                        std::vector<int64_t> &colHash, paru_work *Work,
                        ParU_Numeric *Num);

ParU_Ret paru_make_heap_empty_el(int64_t f,
                                 std::vector<int64_t> &pivotal_elements,
                                 heaps_info &hi, paru_work *Work,
                                 ParU_Numeric *Num);
// hash related
void paru_insert_hash(int64_t key, int64_t value,
                      std::vector<int64_t> &colHash);
int64_t paru_find_hash(int64_t key, std::vector<int64_t> &colHash,
                       int64_t *fcolList);

ParU_Ret paru_finalize_perm(ParU_Symbolic *Sym, ParU_Numeric *Num) ;

// permutation stuff for the solver
int64_t paru_apply_inv_perm(const int64_t *P, const double *s, const double *b, double *x, int64_t m) ;
int64_t paru_apply_inv_perm(const int64_t *P, const double *s, const double *B, double *X, int64_t m, int64_t n) ;

int64_t paru_apply_perm_scale(const int64_t *P, const double *s, const double *b, double *x, int64_t m);
int64_t paru_apply_perm_scale(const int64_t *P, const double *s, const double *b, double *x, int64_t m, int64_t n);

int64_t paru_gaxpy(cholmod_sparse *A, const double *x, double *y, double alpha);
double paru_spm_1norm(cholmod_sparse *A);
double paru_vec_1norm(const double *x, int64_t n);
double paru_matrix_1norm(const double *x, int64_t m, int64_t n);

void paru_Diag_update(int64_t pivcol, int64_t pivrow, paru_work *Work);
int64_t paru_tasked_dgemm(int64_t f, int64_t m, int64_t n, int64_t k, double *A,
                          int64_t lda, double *B, int64_t ldb, double beta,
                          double *C, int64_t ldc, paru_work *Work,
                          ParU_Numeric *Num);
int64_t paru_tasked_trsm(int64_t f, int64_t m, int64_t n, double alpha,
                         double *a, int64_t lda, double *b, int64_t ldb,
                         paru_work *Work, ParU_Numeric *Num);
ParU_Ret paru_free_work(ParU_Symbolic *Sym, paru_work *Work);

// not user-callable: for testing only
ParU_Ret paru_backward(double *x1, double &resid, double &anorm, double &xnorm,
                       cholmod_sparse *A, ParU_Symbolic *Sym, ParU_Numeric *Num,
                       ParU_Control *Control);
#endif
