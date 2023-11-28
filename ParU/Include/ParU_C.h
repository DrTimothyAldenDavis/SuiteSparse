// ============================================================================/
// ======================= ParU_C.h ===========================================/
// ============================================================================/

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

//------------------------------------------------------------------------------

#ifndef PARU_C_H
#define PARU_C_H

#include <stdint.h>
#include "ParU_definitions.h"

// =============================================================================
// ========================= ParU_C_Control ====================================
// =============================================================================

// Just like ParU_Control in the C++ interface.  The only difference is the
// initialization which is handled in the C interface, ParU_C_Init_Control.

typedef struct ParU_C_Control_struct
{
    int64_t mem_chunk;  // chunk size for memset and memcpy

    // Symbolic controls
    int64_t umfpack_ordering;
    int64_t umfpack_strategy;  // symmetric or unsymmetric
    int64_t umfpack_default_singleton; //filter singletons if true

    int64_t relaxed_amalgamation_threshold;
    // symbolic analysis tries that each front have more pivot columns
    // than this threshold

    // Numeric controls
    int64_t scale;         // if 1 matrix will be scaled using max_row
    int64_t panel_width;  // width of panel for dense factorizaiton
    int64_t paru_strategy;  // the same strategy umfpack used

    double piv_toler;     // tolerance for accepting sparse pivots
    double diag_toler;  // tolerance for accepting symmetric pivots
    int64_t trivial;  // dgemms with sizes less than trivial doesn't call BLAS
    int64_t worthwhile_dgemm;  // dgemms bigger than worthwhile are tasked
    int64_t worthwhile_trsm;  // trsm bigger than worthwhile are tasked
    int32_t paru_max_threads;    // It will be initialized with omp_max_threads
    // if the user do not provide a smaller number
} ParU_C_Control;

// =========================================================================
// ========================= ParU_C_Symbolic ===============================
// =========================================================================

// just a carrier for the C++ data structure

typedef struct ParU_C_Symbolic_struct
{
    int64_t m, n, anz;
    int64_t *Qfill ;        // or Q?
    void *sym_handle;
} ParU_C_Symbolic;

// =========================================================================
// ========================= ParU_C_Numeric ================================
// =========================================================================

// just a carrier for the C++ data structure

typedef struct ParU_C_Numeric_struct
{
    double rcond;
    int64_t *Pfin ;         // or P?
    double *Rs ;
    void *num_handle;
} ParU_C_Numeric;

#ifdef __cplusplus
extern "C" {
#endif 

//------------------------------------------------------------------------------
// ParU_Version: 
//------------------------------------------------------------------------------

// return the version
ParU_Ret ParU_C_Version (int ver [3], char date [128]);

//------------------------------------------------------------------------------
// ParU_C_Init_Control
//------------------------------------------------------------------------------

// Initialize C data structure

ParU_Ret ParU_C_Init_Control (ParU_C_Control *Control_C);

//------------------------------------------------------------------------------
// ParU_C_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more speciallized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_Factorize calls. 
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Analyze(
        // input:
        cholmod_sparse *A,  // input matrix to analyze ...
        // output:
        ParU_C_Symbolic **Sym_handle,  // output, symbolic analysis
        // control:
        ParU_C_Control *Control);

//------------------------------------------------------------------------------
// ParU_C_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here.
// ParU_C_Symbolic structure is computed ParU_Analyze and is an input in this
// routine.
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Factorize(
        // input:
        cholmod_sparse *A, ParU_C_Symbolic *Sym,
        // output:
        ParU_C_Numeric **Num_handle,
        // control:
        ParU_C_Control *Control);

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// In all the solve routines Num structure must come with the same Sym struct
// that comes from ParU_Factorize

//-------- Ax = b (x is overwritten on b)---------------------------------------
ParU_Ret ParU_C_Solve_Axx(
    // input:
    ParU_C_Symbolic *Sym, ParU_C_Numeric *Num,
    // input/output:
    double *b,              // vector of size m-by-1
    // control:
    ParU_C_Control *Control);

//-------- Ax = b --------------------------------------------------------------
ParU_Ret ParU_C_Solve_Axb(
    // input:
    ParU_C_Symbolic *Sym, ParU_C_Numeric *Num,
    double *b,              // vector of size m-by-1
    // output
    double *x,              // vector of size m-by-1
    // control:
    ParU_C_Control *Control);

//-------- AX = B  (X is overwritten on B, multiple rhs)------------------------
ParU_Ret ParU_C_Solve_AXX(
    // input
    ParU_C_Symbolic *Sym, ParU_C_Numeric *Num, int64_t nrhs,
    // input/output:
    double *B,  // m(num_rows of A) x nrhs
    // control:
    ParU_C_Control *Control);

//-------- AX = B  (multiple rhs)-----------------------------------------------
ParU_Ret ParU_C_Solve_AXB(
    // input
    ParU_C_Symbolic *Sym, ParU_C_Numeric *Num, int64_t nrhs,
    double *B,  // m(num_rows of A) x nrhs
    // output:
    double *X,  // m(num_rows of A) x nrhs
    // control:
    ParU_C_Control *Control);

// FIXME: add Lsolve, perms etc

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// The user provide both x and b
// resid = norm1(b-A*x) / (norm1(A) * norm1 (x))
ParU_Ret ParU_C_Residual_bAx(
    // inputs:
    cholmod_sparse *A, double *x, double *b, int64_t m,
    // output:
    double *resid, double *anorm, double *xnorm,
    // control:
    ParU_C_Control *Control);

// resid = norm1(B-A*X) / (norm1(A) * norm1 (X))
// (multiple rhs)
ParU_Ret ParU_C_Residual_BAX(
    // inputs:
    cholmod_sparse *A, double *X, double *B, int64_t m, int64_t nrhs,
    // output:
    double *resid, double *anorm, double *xnorm,
    // control:
    ParU_C_Control *Control);

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Freenum(ParU_C_Numeric **Num_handle, ParU_C_Control *Control);
ParU_Ret ParU_C_Freesym(ParU_C_Symbolic **Sym_handle, ParU_C_Control *Control);

#ifdef __cplusplus
}
#endif 

#endif //PARU_C_H
