////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_internal.hpp //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PARU_INTERNAL_H
#define PARU_INTERNAL_H
//!
//  internal libraries that are not visible to the user
//  @author Aznaveh
//

#include <cinttypes>
#define SUITESPARSE_BLAS_DEFINITIONS
#include "ParU.h"
#include "paru_omp.hpp"

// =============================================================================
// =========================== ParU_Symbolic ===================================
// =============================================================================
//
// The contents of this object do not change during numeric factorization.  The
// ParU_U_singleton and ParU_L_singleton are datas tructures for singletons that
// has been borrowed from UMFPACK, but it is saved differently
//
//              ParU_L_singleton is CSC
//                                     |
//                                     v
//    ParU_U_singleton is CSR -> U U U U U U U U U
//                               . U U U U U U U U
//                               . . U U U U U U U
//                               . . . L . . . . .
//                               . . . L L . . . .
//                               . . . L L S S S S
//                               . . . L L S S S S
//                               . . . L L S S S S
//                               . . . L L S S S S

struct ParU_U_singleton
{
    // CSR format for U singletons
    int64_t nnz;   // nnz in submatrix
    int64_t *Sup;  // size cs1
    int64_t *Suj;  // size is computed
};

struct ParU_L_singleton
{
    // CSC format for L singletons
    int64_t nnz;   // nnz in submatrix
    int64_t *Slp;  // size rs1
    int64_t *Sli;  // size is computed
};

struct ParU_Symbolic_struct
{
    // -------------------------------------------------------------------------
    // row-form of the input matrix and its permutations
    // -------------------------------------------------------------------------

    // During symbolic analysis, the nonzero pattern of S = A(P,Q) is
    // constructed, where A is the user's input matrix.  Its numerical values
    // are not constructed.

    // The matrix S is stored in row-oriented form.  The rows of S are
    // sorted according to their leftmost column index (via Pinv).  Column
    // indices in each row of S are in strictly ascending order, even though
    // the input matrix A need not be sorted.

    int64_t m, n, anz;  // A is m-by-n with anz entries

    int64_t snz;  // nnz in submatrix S
    int64_t *Sp;  // size m+1-n1, row pointers of S
    int64_t *Sj;  // size snz = Sp [n], column indices of S

    // Usingletons and Lsingltons
    ParU_U_singleton ustons;
    ParU_L_singleton lstons;

    int64_t *Qfill;  // size n, fill-reducing column permutation.
    // Qfill [k] = j if column k of A is column j of S.

    int64_t *Pinit;  // size m, row permutation.
    // UMFPACK computes it and Pinv  is its inverse
    int64_t *Pinv;  // Inverse of Pinit; it is used to make S

    int64_t *Diag_map;  // size n,
    // UMFPACK computes it and I use it to find original diags out of it

    int64_t *Sleft;  // size n-n1+2.  The list of rows of S whose
    // leftmost column index is j is given by
    // Sleft [j] ... Sleft [j+1]-1.  This can be empty (that is, Sleft
    // [j] can equal Sleft [j+1]).  Sleft [n] is the number of
    // non-empty rows of S, and Sleft [n+1] == m.  That is, Sleft [n]
    // ... Sleft [n+1]-1 gives the empty rows of S, if any.

    int32_t strategy_used ;     // ParU strategy used (symmetric or unsymmetric)
    int32_t umfpack_strategy ;  // UMFPACK strategy used (sym. or unsym.)
    int32_t ordering_used ;     // UMFPACK ordering used
    int32_t unused ;            // future expansion

    // -------------------------------------------------------------------------
    // frontal matrices: pattern and tree
    // -------------------------------------------------------------------------

    // Each frontal matrix is fm-by-fn, with fnpiv pivot columns.  The fn
    // column indices are given by a set of size fnpiv pivot columns

    int64_t nf;  // number of frontal matrices; nf <= MIN (m,n)
    int64_t n1;  // number of singletons in the matrix
    // the matrix S is the one without any singletons
    int64_t rs1, cs1;  // number of row and column singletons, n1 = rs1+cs1;

    // parent, child, and childp define the row merge tree or etree (A'A)
    int64_t *Parent;  // size nf+1  Add another node just to make the forest a
    int64_t *Child;   // size nf+1      tree
    int64_t *Childp;  // size nf+2

    int64_t *Depth;  // size nf distance of each node from the root

    // The parent of a front f is Parent [f], or EMPTY if f=nf.
    // A list of children of f can be obtained in the list
    // Child [Childp [f] ... Childp [f+1]-1].

    // Node nf in the tree is a placeholder; it does not represent a frontal
    // matrix.  All roots of the frontal "tree" (may be a forest) have the
    // placeholder node nf as their parent.  Thus, the tree of nodes 0:nf is
    // truly a tree, with just one parent (node nf).

    int64_t *aParent;  // size m+nf
    int64_t *aChild;   // size m+nf+1
    int64_t *aChildp;  // size m+nf+2
    int64_t *first;    // size nf+1 first successor of front in the tree;
    // all successors are between first[f]...f-1

    // pivot column in the front F.  This refers to a column of S.  The
    // number of expected pivot columns in F is thus
    // Super [f+1] - Super [f].

    // Upper bound number of rows for each front
    int64_t *Fm;  // size nf+1

    // Upper bound  number of rows in the contribution block of each front
    int64_t *Cm;  // size nf+1

    int64_t *Super;  // size nf+1.  Super [f] gives the first
    // pivot column in the front F.  This refers to a column of S.  The
    // number of expected pivot columns in F is thus
    // Super [f+1] - Super [f].

    int64_t *row2atree;    // Mapping from rows to augmented tree size m
    int64_t *super2atree;  // Mapping from super nodes to augmented tree size nf

    int64_t *Chain_start;  // size = n_col +1;  actual size = nfr+1
    // The kth frontal matrix chain consists of frontal
    // matrices Chain_start [k] through Chain_start [k+1]-1.
    // Thus, Chain_start [0] is always 0 and
    // Chain_start[nchains] is the total number of frontal
    // matrices, nfr. For two adjacent fornts f and f+1
    // within a single chian, f+1 is always the parent of f
    // (that is, Front_parent [f] = f+1).

    int64_t *Chain_maxrows;  // size = n_col +1;  actual size = nfr+1
    int64_t *Chain_maxcols;  // The kth frontal matrix chain requires a single
    // working array of dimension Chain_maxrows [k] by
    // Chain_maxcols [k], for the unifrontal technique that
    // factorizes the frontal matrix chain. Since the
    // symbolic factorization only provides

    // only used for statistics when debugging is enabled:
    int64_t Us_bound_size;   // Upper bound on size of all Us, sum all fp*fn
    int64_t LUs_bound_size;  // Upper bound on size of all LUs, sum all fp*fm
    int64_t row_Int_bound;   // Upper bound on size of all ints for rows
    int64_t col_Int_bound;   // Upper bound on size of all ints for cols

    double *front_flop_bound;  // bound on m*n*k for each front size nf+1
    double *stree_flop_bound;  // flop bound for front and descendents size nf+1

    // data structure related to ParU tasks
    int64_t ntasks;        // number of tasks; at most nf
    int64_t *task_map;     // each task does the fronts
                       // from task_map[i]+1 to task_map[i+1]; task_map[0] is -1
    int64_t *task_parent;  // tree data structure for tasks
    int64_t *task_num_child;  // number of children of each task
    int64_t *task_depth;      // max depth of each task
} ;

// =============================================================================
// =========================== ParU_Numeric ====================================
// =============================================================================
// ParU_Numeric contains all the numeric information that user needs for solving
// a system. The factors are saved as a seried of dense matrices. User can check
// the ParU_Info to see if the factorization is successful. sizes of
// ParU_Numeric is size of S matrix in Symbolic analysis.

struct ParU_Factors
{
    // dense factorized part pointer
    int64_t m, n;   //  mxn dense matrix
    double *p;  //  point to factorized parts
};

struct ParU_Numeric_struct
{
    int64_t m, n;   // size of the sumbatrix(S) that is factorized
    int64_t sym_m;  // number of rows of original matrix; a copy of Sym->m

    int64_t nf;      // number of fronts copy of Sym->nf
    double *Rs;  // the array for row scaling based on original matrix
                 // size = m

    // Permutations are computed after all the factorization
    int64_t *Ps;  // size m, row permutation.
    // Permutation from S to LU. needed for lsolve and usolve
    int64_t *Pfin;  // size m, row permutation.
    // ParU final permutation.

    int64_t snz;     // nnz in S; copy of Sym->snz
    double *Sx;  // size snz = Sp [n], numeric values of (scaled) S;
    // Sp and Sj must be initialized in Symbolic phase
    int64_t sunz;
    double *Sux;  // Numeric values u singletons, Sup Suj are in symbolic
    int64_t slnz;
    double *Slx;  // Numeric values l singletons, Slp Sli are in symbolic

    // Computed parts of each front
    int64_t *frowCount;  // size nf   size(CB) = rowCount[f]x
    int64_t *fcolCount;  // size nf                        colCount[f]
    int64_t **frowList;  // size nf   frowList[f] is rows of the matrix S
    int64_t **fcolList;  // size nf   colList[f] is non pivotal cols of the
    //   matrix S
    ParU_Factors *partial_Us;   // size nf   size(Us)= fp*colCount[f]
    ParU_Factors *partial_LUs;  // size nf   size(LUs)= rowCount[f]*fp

    int64_t max_row_count;  // maximum number of rows/cols for all the fronts
    int64_t max_col_count;  // it is initalized after factorization

    double rcond;       // rough estimate of the reciprocal condition number
    double min_udiag;   // min (abs (diag (U)))
    double max_udiag;   // max (abs (diag (U)))
    int64_t nnzL;       //nnz of L
    int64_t nnzU;       //nnz of U
    double sfc; //simple flop count
    ParU_Info res;  // returning value of numeric phase
} ;

// =============================================================================
// =========================== ParU_Control ====================================
// =============================================================================
// The default value of some control options can be found here. All user
// callable functions use ParU_Control; some controls are used only in symbolic
// phase and some controls are only used in numeric phase.

struct ParU_Control_struct
{
    // For all phases of ParU:
    size_t mem_chunk ;         // chunk size for memset and memcpy

    // Numeric factorization parameters:
    double piv_toler ;          // tolerance for accepting sparse pivots
    double diag_toler ;         // tolerance for accepting symmetric pivots
    int64_t panel_width ;       // width of panel for dense factorizaiton
    int64_t trivial ;           // dgemms smaller than this do not call BLAS
    int64_t worthwhile_dgemm ;  // dgemms bigger than this are tasked
    int64_t worthwhile_dtrsm ;  // dtrsm bigger than this are tasked
    int64_t prescale ;          // 0: none, 1: scale by sum, 2: scale by max

    // Symbolic analysis parameters:
    int64_t strategy ;          // ParU strategy to use
    int64_t ordering ;          // ParU ordering to use
    int64_t umfpack_strategy ;  // UMFPACK strategy to use
    int64_t relaxed_amalgamation ;  // symbolic analysis tries to ensure
        // that each front have more pivot columns than this threshold
    int64_t filter_singletons ; // filter singletons if nonzero

    // For all phases of ParU:
    int64_t paru_max_threads ;  // initialized with omp_max_threads
} ;

extern "C"
{
    // These C objects are just carriers for the C++ data structures
    struct ParU_C_Symbolic_struct
    {
        void *sym_handle ;          // the C++ Symbolic struct
    } ;
    struct ParU_C_Numeric_struct
    {
        void *num_handle ;          // the C++ Numeric struct
    } ;
    struct ParU_C_Control_struct
    {
        void *control_handle ;      // the C++ Numeric struct
    } ;
}

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

// silence these diagnostics:
#ifdef __clang__
#pragma clang diagnostic ignored "-Wc++11-extensions"
#endif

//==============================================================================
// UMFPACK internal definitions
//==============================================================================

// These definitions are meant only for internal use in UMFPACK and ParU.
// The two typedefs below must exactly match their definitions in
// UMFPACK/Source/umf_internal.h.

extern "C"
{

//------------------------------------------------------------------------------
// Symbolic: symbolic factorization
//------------------------------------------------------------------------------

// This is is constructed by UMFPACK_symbolic, and is needed by UMFPACK_numeric
// to factor the matrix.

typedef struct  // SymbolicType
{

    double
        num_mem_usage_est,      /* estimated max Numeric->Memory size */
        num_mem_size_est,       /* estimated final Numeric->Memory size */
        peak_sym_usage,         /* peak Symbolic and SymbolicWork usage */
        sym,                    /* symmetry of pattern */
        dnum_mem_init_usage,    /* min Numeric->Memory for UMF_kernel_init */
        amd_lunz,               /* nz in LU for AMD, with symmetric pivoting */
        lunz_bound ;            /* max nx in LU, for arbitrary row pivoting */

    int64_t
        valid,                  /* set to SYMBOLIC_VALID, for validity check */
        max_nchains,
        nchains,
        *Chain_start,
        *Chain_maxrows,
        *Chain_maxcols,
        maxnrows,               /* largest number of rows in any front */
        maxncols,               /* largest number of columns in any front */
        *Front_npivcol,         /* Front_npivcol [j] = size of jth supercolumn*/
        *Front_1strow,          /* first row in front j */
        *Front_leftmostdesc,    /* leftmost desc of front j */
        *Front_parent,          /* super-column elimination tree */
        *Cperm_init,            /* initial column ordering */
        *Rperm_init,            /* initial row ordering */
        *Cdeg, *Rdeg,
        *Esize,
        dense_row_threshold,
        n1,                     /* number of singletons */
        n1r,                    /* number of row singletons */
        n1c,                    /* number of column singletons */
        nempty,                 /* MIN (nempty_row, nempty_col) */
        *Diagonal_map,          /* initial "diagonal" */
        esize,                  /* size of Esize array */
        nfr,
        n_row, n_col,           /* matrix A is n_row-by-n_col */
        nz,                     /* nz of original matrix */
        nb,                     /* block size for BLAS 3 */
        num_mem_init_usage,     /* min Numeric->Memory for UMF_kernel_init */
        nempty_row, nempty_col,

        strategy,
        ordering,
        fixQ,
        prefer_diagonal,
        nzaat,
        nzdiag,
        amd_dmax ;

} SymbolicType ;

//------------------------------------------------------------------------------
// SW Type: used internally in umfpack_qsymbolic
//------------------------------------------------------------------------------

typedef struct  /* SWType */
{
    int64_t
        *Front_npivcol,     /* size n_col + 1 */
        *Front_nrows,       /* size n_col */
        *Front_ncols,       /* size n_col */
        *Front_parent,      /* size n_col */
        *Front_cols,        /* size n_col */
        *InFront,           /* size n_row */
        *Ci,                /* size Clen */
        *Cperm1,            /* size n_col */
        *Rperm1,            /* size n_row */
        *InvRperm1,         /* size n_row */
        *Si,                /* size nz */
        *Sp ;               /* size n_col + 1 */
    double *Rs ;            /* size n_row */

} SWType ;

}

#undef  TRUE
#define TRUE (1)
#undef  FALSE
#define FALSE (0)

//------------------------------------------------------------------------------

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

// To look for untested lines of code (development only):
#ifdef MATLAB_MEX_FILE
    #define GOTCHA                                                  \
    {                                                               \
        /* this will cause a segfault if done inside a parallel region: */  \
        mexErrMsgIdAndTxt ("ParU:gotcha",                           \
            "gotcha: %s %d\n", __FILE__, __LINE__) ;                \
    }
#else
    #define GOTCHA                                                  \
    {                                                               \
        fprintf (stderr, "gotcha: %s %d\n", __FILE__, __LINE__) ;   \
        fflush (stderr) ;                                           \
        abort ( ) ;                                                 \
    }
#endif

#if ( defined ( BLAS_Intel10_64ilp ) || defined ( BLAS_Intel10_64lp ) )

    #ifdef MATLAB_MEX_FILE

        extern "C"
        {
            void mkl_serv_set_num_threads (int n) ;
            void mkl_serv_set_num_threads_local (int n) ;
            void mkl_serv_set_dynamic (int flag);
        }
        #define mkl_set_num_threads         mkl_serv_set_num_threads
        #define mkl_set_num_threads_local   mkl_serv_set_num_threads
        #define mkl_set_dynamic             mkl_serv_set_dynamic

    #else

        extern "C"
        {
            void MKL_Set_Num_Threads (int n) ;
            void MKL_Set_Num_Threads_Local (int n) ;
            void MKL_Set_Dynamic (int flag);
        }
        #define mkl_set_num_threads         MKL_Set_Num_Threads
        #define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local
        #define mkl_set_dynamic             MKL_Set_Dynamic

    #endif

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
{
    // List of tuples
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

    // Control parameters:
    int64_t worthwhile_dgemm ;
    int64_t worthwhile_dtrsm ;
    int64_t trivial ;
    int64_t panel_width ;
    double piv_toler ;
    double diag_toler ;
    size_t mem_chunk ;
    int32_t nthreads ;
    int32_t prescale ;

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
{
    return (int64_t *)(curEl + 1) + curEl->ncols + curEl->nrows;
}

inline int64_t *relRowInd(paru_element *curEl)
{
    return (int64_t *)(curEl + 1) + 2 * curEl->ncols + curEl->nrows;
}

inline double *numeric_pointer(paru_element *curEl)
{
    return (double *)((int64_t *)(curEl + 1) + 2 * curEl->ncols +
                      2 * curEl->nrows);
}

inline int64_t flip(int64_t colInd)
{
    return -colInd - 2;
}

inline int64_t lac_el(paru_element **elementList, int64_t eli)
{
    // return least numbered column of the element i (eli)
    int64_t result = INT64_MAX ;
    if (elementList[eli] != NULL)
    {
        int64_t *el_colIndex = (int64_t *)(elementList[eli] + 1);
        int64_t lac_ind = elementList[eli]->lac;
        result = el_colIndex[lac_ind];
    }
    return (result) ;
}

int32_t paru_nthreads (ParU_Control Control) ;

//------------------------------------------------------------------------------
// internal routines
//------------------------------------------------------------------------------

// #ifndef PARU_ALLOC_TESTING
// #define PARU_ALLOC_TESTING
// #endif

// #ifndef PARU_MEMTABLE_TESTING
// #define PARU_MEMTABLE_TESTING
// #endif

// #ifndef PARU_MEMDUMP
// #define PARU_MEMDUMP
// #endif

// #ifndef PARU_MALLOC_DEBUG
// #define PARU_MALLOC_DEBUG
// #endif

/* Wrappers for managing memory */
extern "C" {
void *paru_malloc(size_t n, size_t size);
void *paru_calloc(size_t n, size_t size);
void *paru_realloc(size_t newsize, size_t size_Entry, void *oldP, size_t *size);
void paru_free(size_t n, size_t size, void *p);

void *paru_malloc_debug(size_t n, size_t size, const char *filename, int line);
void *paru_calloc_debug(size_t n, size_t size, const char *filename, int line);
void *paru_realloc_debug(size_t newsize, size_t size_Entry, void *oldP, size_t *size, const char *filename, int line);
void paru_free_debug(size_t n, size_t size, void *p, const char *filename, int line);
}

#if defined ( PARU_MALLOC_DEBUG )

    #define PARU_MALLOC(n,T) static_cast<T *>(paru_malloc_debug (n, sizeof (T), __FILE__, __LINE__))
    #define PARU_CALLOC(n,T) static_cast<T *>(paru_calloc_debug (n, sizeof (T), __FILE__, __LINE__))

    #define PARU_REALLOC(newsize,T,oldP,size)                       \
        static_cast<T *>(paru_realloc_debug (newsize, sizeof(T),    \
            oldP, size, __FILE__, __LINE__))

    #define PARU_FREE(n,T,p)                                        \
    {                                                               \
        paru_free_debug (n, sizeof (T), p, __FILE__, __LINE__) ;    \
        (p) = NULL ;                                                \
    }

#else

    #define PARU_MALLOC(n,T) static_cast<T*>(paru_malloc (n, sizeof (T)))
    #define PARU_CALLOC(n,T) static_cast<T*>(paru_calloc (n, sizeof (T)))

    #define PARU_REALLOC(newsize,T,oldP,size)                       \
        static_cast<T *>(paru_realloc (newsize, sizeof(T), oldP, size))

    #define PARU_FREE(n,T,p)                                        \
    {                                                               \
        paru_free (n, sizeof (T), p) ;                              \
        (p) = NULL ;                                                \
    }

#endif

void paru_free_el(int64_t e, paru_element **elementList);

void *operator new(std::size_t sz);
void operator delete(void *ptr) noexcept;

void paru_memset
(
    void *ptr,
    int64_t value,
    size_t nbytes,
    size_t mem_chunk,
    int32_t nthreads
) ;

void paru_memcpy
(
    void *destination,      // output array of size nbytes
    const void *source,     // input array of size nbytes
    size_t nbytes,          // # of bytes to copy
    size_t mem_chunk,
    int32_t nthreads
) ;

#ifdef PARU_ALLOC_TESTING
bool paru_get_malloc_tracking(void);
void paru_set_malloc_tracking(bool track);
void paru_set_nmalloc(int64_t nmalloc);
int64_t paru_decr_nmalloc(void);
int64_t paru_get_nmalloc(void);
#endif

#ifndef NDEBUG
void paru_write
(
    int scale,
    char *id,
    paru_work *Work,
    ParU_Numeric Num
) ;

void paru_print_element
(
    int64_t e,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_print_paru_tupleList(paru_tupleList *listSet, int64_t index);
#endif

ParU_Info paru_umfpack_info (int status) ;

/* add tuple functions defintions */
ParU_Info paru_add_rowTuple(paru_tupleList *RowList, int64_t row, paru_tuple T);

ParU_Info paru_factorize_full_summed
(
    int64_t f,
    int64_t start_fac,
    std::vector<int64_t> &panel_row,
    std::set<int64_t> &stl_colSet,
    std::vector<int64_t> &pivotal_elements,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

ParU_Info paru_exec_tasks
(
    int64_t t,
    int64_t *task_num_child,
    int64_t &chain_task,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

ParU_Info paru_exec_tasks_seq
(
    int64_t t,
    int64_t *task_num_child,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

paru_element *paru_create_element(int64_t nrows, int64_t ncols);

void paru_assemble_row_2U
(
    int64_t e,
    int64_t f,
    int64_t sR,
    int64_t dR,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    ParU_Numeric Num
) ;

bool paru_dtrsm
(
    int64_t f,
    double *pF,
    double *uPart,
    int64_t fp,
    int64_t rowCount,
    int64_t colCount,
    paru_work *Work,
    ParU_Numeric Num
) ;

bool paru_dgemm
(
    int64_t f,
    double *pF,
    double *uPart,
    double *el,
    int64_t fp,
    int64_t rowCount,
    int64_t colCount,
    paru_work *Work,
    ParU_Numeric Num
) ;

void paru_init_rel
(
    int64_t f,
    paru_work *Work,
    const ParU_Symbolic Sym
) ;

void paru_update_rel_ind_col
(
    int64_t e,
    int64_t f,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    ParU_Numeric Num
) ;

void paru_update_rowDeg
(
    int64_t panel_num,
    int64_t row_end,
    int64_t f,
    int64_t start_fac,
    std::set<int64_t> &stl_colSet,
    std::vector<int64_t> &pivotal_elements,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

int64_t paru_cumsum
(
    int64_t n,
    int64_t *X,
    size_t mem_chunk,
    int32_t nthreads
) ;

int64_t paru_bin_srch_col(int64_t *srt_lst, int64_t l, int64_t r, int64_t num);
int64_t paru_bin_srch(int64_t *srt_lst, int64_t l, int64_t r, int64_t num);

ParU_Info paru_init_rowFronts
(
    // input/output:
    paru_work *Work,
    ParU_Numeric *Num_handle,
    // inputs, not modified:
    cholmod_sparse *A,
    ParU_Symbolic Sym       // symbolic analysis
) ;

ParU_Info paru_front
(
    int64_t f,  // front need to be assembled
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

ParU_Info paru_pivotal
(
    std::vector<int64_t> &pivotal_elements,
    std::vector<int64_t> &panel_row,
    int64_t &zero_piv_rows,
    int64_t f,
    heaps_info &hi,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

int paru_intersection(int64_t e, paru_element **elementList,
                      std::set<int64_t> &stl_colSet);

ParU_Info paru_prior_assemble
(
    int64_t f,
    int64_t start_fac,
    std::vector<int64_t> &pivotal_elements,
    std::vector<int64_t> &colHash,
    heaps_info &hi,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_assemble_all
(
    int64_t e,
    int64_t f,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_assemble_cols
(
    int64_t e,
    int64_t f,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_assemble_rows
(
    int64_t e,
    int64_t f,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_assemble_el_with0rows
(
    int64_t e,
    int64_t f,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_full_summed
(
    int64_t e,
    int64_t f,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

// heap related
ParU_Info paru_make_heap
(
    int64_t f,
    int64_t start_fac,
    std::vector<int64_t> &pivotal_elements,
    heaps_info &hi,
    std::vector<int64_t> &colHash,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

ParU_Info paru_make_heap_empty_el
(
    int64_t f,
    std::vector<int64_t> &pivotal_elements,
    heaps_info &hi,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

// hash related
void paru_insert_hash(int64_t key, int64_t value,
                      std::vector<int64_t> &colHash);
int64_t paru_find_hash(int64_t key, std::vector<int64_t> &colHash,
                       int64_t *fcolList);

ParU_Info paru_finalize_perm
(
    const ParU_Symbolic Sym,
    ParU_Numeric Num
) ;

void paru_gaxpy
(
    cholmod_sparse *A,
    const double *x,
    double *y,
    double alpha
) ;

double paru_spm_1norm(cholmod_sparse *A);
double paru_vec_1norm(const double *x, int64_t n);
double paru_matrix_1norm(const double *x, int64_t m, int64_t n);

void paru_diag_update(int64_t pivcol, int64_t pivrow, paru_work *Work);

bool paru_tasked_dgemm
(
    int64_t f,
    int64_t M,
    int64_t N,
    int64_t K,
    double *A,
    int64_t lda,
    double *B,
    int64_t ldb,
    double beta,
    double *C,
    int64_t ldc,
    paru_work *Work,
    ParU_Numeric Num
) ;

bool paru_tasked_dtrsm
(
    int64_t f,
    int64_t m,
    int64_t n,
    double alpha,
    double *a,
    int64_t lda,
    double *b,
    int64_t ldb,
    paru_work *Work,
    ParU_Numeric Num
) ;

ParU_Info paru_free_work
(
    const ParU_Symbolic Sym,
    paru_work *Work
) ;

#if defined ( PARU_ALLOC_TESTING ) && defined ( PARU_MEMTABLE_TESTING )
extern "C" {
    void paru_memtable_dump (void) ;
    int paru_memtable_n (void) ;
    void paru_memtable_add (void *p, size_t size) ;
    size_t paru_memtable_size (void *p) ;
    bool paru_memtable_find (void *p) ;
    void paru_memtable_remove (void *p) ;
}
#endif

#endif

