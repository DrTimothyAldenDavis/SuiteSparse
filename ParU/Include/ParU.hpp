// ============================================================================/
// ======================= ParU.hpp ===========================================/
// ============================================================================/

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

//------------------------------------------------------------------------------

// This is the ParU.hpp file. All user callable routines are in this file and
// all of them start with ParU_*. This file must be included in all user code
// that use ParU.
//
// ParU is a parallel sparse direct solver. This package uses OpenMP tasking
// for parallelism. ParU calls UMFPACK for symbolic analysis phase, after that
// some symbolic analysis is done by ParU itself and then numeric phase
// starts. The numeric computation is a task parallel phase using OpenMP and
// each task calls parallel BLAS; i.e. nested parallism.
//
// The performance of BLAS has a heavy impact on the performance of ParU.
// However, depending on the input problem performance of parallelsim in BLAS
// sometimes does not have effects in ParU.
//
// General Usage for solving Ax = b A is a sparse matrix in matrix market
// format with double entries and b is a dense vector of double (or a dense
// matrix B for multiple rhs):
//
//      info = ParU_Analyze(A, &Sym, &Control);
//      info = ParU_Factorize(A, Sym, &Num, &Control);
//      info = ParU_Solve(Sym, Num, b, x, &Control);
//
// See paru_demo for more examples

#ifndef PARU_H
#define PARU_H

// ============================================================================/
// ======================= ParU version =======================================/
// ============================================================================/

#include "ParU_definitions.h"

// =============================================================================
// =========================== ParU_Symbolic ===================================
// =============================================================================
//
// The contents of this object do not change during numeric factorization.  The
// ParU_U_singleton and ParU_L_singleton are datastructures for singletons that
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

struct ParU_Symbolic
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
    // the input matrix A need not be sorted. User can look inside S matrix.

    int64_t m, n, anz;  // S is m-by-n with anz entries

    int64_t snz;  // nnz in submatrix
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

    int64_t strategy;  // the strategy that is actually used by umfpack
    // symmetric or unsymmetric

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
};

// =============================================================================
// =========================== ParU_Control ====================================
// =============================================================================
// The default value of some control options can be found here. All user
// callable functions use ParU_Control; some controls are used only in symbolic
// phase and some controls are only used in numeric phase.
struct ParU_Control
{
    int64_t mem_chunk = PARU_MEM_CHUNK ;  // chunk size for memset and memcpy

    // Symbolic controls
    int64_t umfpack_ordering = UMFPACK_ORDERING_METIS;
    int64_t umfpack_strategy = UMFPACK_STRATEGY_AUTO; //symmetric or unsymmetric
    int64_t umfpack_default_singleton = 1; //filter singletons if true

    int64_t relaxed_amalgamation_threshold =
        32;  // symbolic analysis tries that each front have more pivot columns
             // than this threshold

    // Numeric controls
    int64_t scale = 1;         // if 1 matrix will be scaled using max_row
    int64_t panel_width = 32;  // width of panel for dense factorizaiton
    int64_t paru_strategy = PARU_STRATEGY_AUTO;//the same strategy umfpack used

    double piv_toler = 0.1;     // tolerance for accepting sparse pivots
    double diag_toler = 0.001;  // tolerance for accepting symmetric pivots
    int64_t trivial = 4; //dgemms with sizes less than trivial doesn't call BLAS
    int64_t worthwhile_dgemm = 512;  // dgemms bigger than worthwhile are tasked
    int64_t worthwhile_trsm = 4096;  // trsm bigger than worthwhile are tasked
    int32_t paru_max_threads = 0;  //It will be initialized with omp_max_threads
                                 // if the user do not provide a smaller number
};
// =============================================================================
// =========================== ParU_Numeric ====================================
// =============================================================================
// ParU_Numeric contains all the numeric information that user needs for solving
// a system. The factors are saved as a seried of dense matrices. User can check
// the ParU_Ret to see if the factorization is successfull. sizes of
// ParU_Numeric is size of S matrix in Symbolic analysis.
struct ParU_Factors
{               // dense factorized part pointer
    int64_t m, n;   //  mxn dense matrix
    double *p;  //  point to factorized parts
};

struct ParU_Numeric
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

    ParU_Control *Control;  // a copy of controls for internal use
    // it is freed after factorize

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

    double rcond;
    double min_udiag;
    double max_udiag;
    ParU_Ret res;  // returning value of numeric phase
};
//------------------------------------------------------------------------------
// ParU_Version:
//------------------------------------------------------------------------------

// return the version

ParU_Ret ParU_Version (int ver [3], char date [128]);

//------------------------------------------------------------------------------
// ParU_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more speciallized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_Factorize calls.
//------------------------------------------------------------------------------

ParU_Ret ParU_Analyze(
        // input:
        cholmod_sparse *A,  // input matrix to analyze ...
        // output:
        ParU_Symbolic **Sym_handle,  // output, symbolic analysis
        // control:
        ParU_Control *Control);

//------------------------------------------------------------------------------
// ParU_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here. ParU_Symbolic
// structure is computed ParU_Analyze and is an input in this routine.
//------------------------------------------------------------------------------

ParU_Ret ParU_Factorize(
    // input:
    cholmod_sparse *A, ParU_Symbolic *Sym,
    // output:
    ParU_Numeric **Num_handle,
    // control:
    ParU_Control *Control);

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// In all the solve routines Num structure must come with the same Sym struct
// that comes from ParU_Factorize

//-------- Ax = b (x is overwritten on b)---------------------------------------
ParU_Ret ParU_Solve(
    // input:
    ParU_Symbolic *Sym, ParU_Numeric *Num,
    // input/output:
    double *b,              // vector of size m-by-1
    // control:
    ParU_Control *Control);

//-------- Ax = b --------------------------------------------------------------
ParU_Ret ParU_Solve(
    // input:
    ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *b,              // vector of size m-by-1
    // output
    double *x,              // vector of size m-by-1
    // control:
    ParU_Control *Control);

//-------- AX = B  (X is overwritten on B, multiple rhs)------------------------
ParU_Ret ParU_Solve(
    // input
    ParU_Symbolic *Sym, ParU_Numeric *Num, int64_t nrhs,
    // input/output:
    double *B,  // m(num_rows of A) x nrhs
    // control:
    ParU_Control *Control);

//-------- AX = B  (multiple rhs)-----------------------------------------------
ParU_Ret ParU_Solve(
    // input
    ParU_Symbolic *Sym, ParU_Numeric *Num, int64_t nrhs,
    double *B,  // m(num_rows of A) x nrhs
    // output:
    double *X,  // m(num_rows of A) x nrhs
    // control:
    ParU_Control *Control);

// FIXME: add comments, and add to user guide
ParU_Ret ParU_Lsolve(ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *x, ParU_Control *Control) ;

ParU_Ret ParU_Lsolve( ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *X, int64_t nrhs,   // X is m-by-nrhs, where A is m-by-m
    ParU_Control *Control) ;

ParU_Ret ParU_Usolve(ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *x, ParU_Control *Control) ;

ParU_Ret ParU_Usolve(ParU_Symbolic *Sym, ParU_Numeric *Num,
    double *X, int64_t nrhs,
    ParU_Control *Control) ;

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// The user provide both x and b
// resid = norm1(b-A*x) / (norm1(A) * norm1 (x))
ParU_Ret ParU_Residual(
    // inputs:
    cholmod_sparse *A, double *x, double *b, int64_t m,
    // output:
    double &resid, double &anorm, double &xnorm,
    // control:
    ParU_Control *Control);

// resid = norm1(B-A*X) / (norm1(A) * norm1 (X))
// (multiple rhs)
ParU_Ret ParU_Residual(
    // inputs:
    cholmod_sparse *A, double *X, double *B, int64_t m, int64_t nrhs,
    // output:
    double &resid, double &anorm, double &xnorm,
    // control:
    ParU_Control *Control);

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Ret ParU_Freenum(ParU_Numeric **Num_handle, ParU_Control *Control);
ParU_Ret ParU_Freesym(ParU_Symbolic **Sym_handle, ParU_Control *Control);

#endif //PARU_H
