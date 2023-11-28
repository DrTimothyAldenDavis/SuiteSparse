////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// paru_c.cpp ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

//
/*! @brief  This C++ file provides a set of C-callable wrappers so that a C
 *  program can call ParU.
 *
 *  @author Aznaveh
 */
#include "ParU_C.h"

#include "paru_internal.hpp"

extern "C"
{

//------------------------------------------------------------------------------
// ParU_Version: 
//------------------------------------------------------------------------------

// return the version
ParU_Ret ParU_C_Version (int ver [3], char date [128])
    {return ParU_Version (ver ,date);}

//------------------------------------------------------------------------------
// ParU_C_Init_Control: initialize C_Control with the default values
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Init_Control (ParU_C_Control *Control_C)
{
    Control_C->mem_chunk = PARU_MEM_CHUNK ;  // chunk size for memset and memcpy

    Control_C->umfpack_ordering =  UMFPACK_ORDERING_METIS;
    Control_C->umfpack_strategy = 
        UMFPACK_STRATEGY_AUTO;  // symmetric or unsymmetric
    Control_C->umfpack_default_singleton = 1;

    Control_C->relaxed_amalgamation_threshold = 32;

    Control_C->scale = 1;
    Control_C->panel_width = 32;
    Control_C->paru_strategy = PARU_STRATEGY_AUTO;


    Control_C->piv_toler = .1;
    Control_C->diag_toler = .001;
    Control_C->trivial = 4;
    Control_C->worthwhile_dgemm = 512;
    Control_C->worthwhile_trsm = 4096;
    Control_C->paru_max_threads = 0;
    return PARU_SUCCESS;
}

//------------------------------------------------------------------------------
// paru_cp_control: copy the inside of the C structrue to the Cpp structure
//------------------------------------------------------------------------------

void paru_cp_control (ParU_Control *Control, ParU_C_Control *Control_C)
{
    Control->mem_chunk = Control_C->mem_chunk;

    Control->umfpack_ordering = Control_C->umfpack_ordering;
    Control->umfpack_strategy = Control_C->umfpack_strategy;
    Control->umfpack_default_singleton = Control_C->umfpack_default_singleton;

    Control->relaxed_amalgamation_threshold = 
        Control_C->relaxed_amalgamation_threshold;

    Control->scale = Control_C->scale;
    Control->panel_width = Control_C->panel_width;
    Control->paru_strategy = Control_C->paru_strategy;


    Control->piv_toler = Control_C->piv_toler;
    Control->diag_toler = Control_C->diag_toler;
    Control->trivial = Control_C->trivial;
    Control->worthwhile_dgemm = Control_C->worthwhile_dgemm;
    Control->worthwhile_trsm = Control_C->worthwhile_trsm;
    Control->paru_max_threads = Control_C->paru_max_threads;
}

//------------------------------------------------------------------------------
// ParU_C_Analyze: Symbolic analysis is done in this routine. UMFPACK is called
// here and after that some more speciallized symbolic computation is done for
// ParU. ParU_Analyze can be called once and can be used for different
// ParU_C_Factorize calls. 
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Analyze(
        // input:
        cholmod_sparse *A,  // input matrix to analyze ...
        // output:
        ParU_C_Symbolic **Sym_handle_C,  // output, symbolic analysis
        // control:
        ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Symbolic *Sym_C =
        (ParU_C_Symbolic *)paru_alloc(1, sizeof(ParU_C_Symbolic));
    if (!Sym_C)
    {
        return PARU_OUT_OF_MEMORY;
    }
    ParU_Symbolic *Sym;
    ParU_Ret info;
    info = ParU_Analyze(A, &Sym, &Control);
    if (info != PARU_SUCCESS)
        return info; //To avoid playing with wrong ponters
    Sym_C->sym_handle = (void*) Sym;
    *Sym_handle_C = Sym_C;
    Sym_C->m = Sym->m;
    Sym_C->n = Sym->n;
    Sym_C->anz = Sym->anz;
    return info;
}

//------------------------------------------------------------------------------
// ParU_C_Factorize: Numeric factorization is done in this routine. Scaling and
// making Sx matrix, computing factors and permutations is here.
// ParU_C_Symbolic structure is computed by ParU_C_Analyze and is an input in
// this routine.
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Factorize (
        // input:
        cholmod_sparse *A, ParU_C_Symbolic *Sym_C,
        // output:
        ParU_C_Numeric **Num_handle_C,
        // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_Symbolic *Sym = (ParU_Symbolic *) Sym_C->sym_handle;
    ParU_C_Numeric *Num_C = 
        (ParU_C_Numeric *)paru_alloc(1, sizeof(ParU_C_Numeric));
    if (!Num_C)
    {
        return PARU_OUT_OF_MEMORY;
    }

    ParU_Ret info;
    ParU_Numeric *Num;
    info = ParU_Factorize(A, Sym, &Num, &Control);
    if (info != PARU_SUCCESS)
        return info;
    Num_C->num_handle = (void *) Num;
    *Num_handle_C = Num_C;
    Num_C->rcond = Num->rcond;
    return info;
}

//------------------------------------------------------------------------------
//--------------------- Solve routines -----------------------------------------
//------------------------------------------------------------------------------

// In all the solve routines Num structure must come with the same Sym struct
// that comes from ParU_Factorize

//-------- Ax = b (x is overwritten on b)---------------------------------------
ParU_Ret ParU_C_Solve_Axx (
    // input:
    ParU_C_Symbolic *Sym_C, ParU_C_Numeric *Num_C,
    // input/output:
    double *b,
    // control:
    ParU_C_Control *Control_C)
{
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return ParU_Solve ((ParU_Symbolic *) Sym_C->sym_handle, 
            (ParU_Numeric *) Num_C->num_handle, b, &Control);
}

//-------- Ax = b --------------------------------------------------------------
ParU_Ret ParU_C_Solve_Axb (
    // input:
    ParU_C_Symbolic *Sym_C, ParU_C_Numeric *Num_C, double *b,
    // output
    double *x,
    // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return ParU_Solve ((ParU_Symbolic *) Sym_C->sym_handle,
            (ParU_Numeric *)Num_C->num_handle, b, x, &Control);
}

//-------- AX = B  (X is overwritten on B, multiple rhs)------------------------
ParU_Ret ParU_C_Solve_AXX (
    // input
    ParU_C_Symbolic *Sym_C, ParU_C_Numeric *Num_C, int64_t nrhs,
    // input/output:
    double *B,  // m(num_rows of A) x nrhs
    // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return ParU_Solve ((ParU_Symbolic *) Sym_C->sym_handle,
            (ParU_Numeric *)Num_C->num_handle, B, &Control);
}

//-------- AX = B  (multiple rhs)-----------------------------------------------
ParU_Ret ParU_C_Solve_AXB (
    // input
    ParU_C_Symbolic *Sym_C, ParU_C_Numeric *Num_C, int64_t nrhs, double *B,
    // output:
    double *X,
    // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    return ParU_Solve ((ParU_Symbolic *) Sym_C->sym_handle,
            (ParU_Numeric *)Num_C->num_handle, nrhs, B, X, &Control);
}

//------------------------------------------------------------------------------
//-------------- computing residual --------------------------------------------
//------------------------------------------------------------------------------

// The user provide both x and b
// resid = norm1(b-A*x) / norm1(A)

ParU_Ret ParU_C_Residual_bAx (
    // inputs:
    cholmod_sparse *A, double *x, double *b, int64_t m,
    // output:
    double *residc, double *anormc, double *xnormc,
    // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    double resid, anorm, xnorm;
    ParU_Ret info;
    info = ParU_Residual (A, x, b, m, resid, anorm, xnorm, &Control); 
    *residc = resid;
    *anormc = anorm;
    *xnormc = xnorm;
    return info;
}


// resid = norm1(B-A*X) / norm1(A) (multiple rhs)
ParU_Ret ParU_C_Residual_BAX (
    // inputs:
    cholmod_sparse *A, double *X, double *B, int64_t m, int64_t nrhs,
    // output:
    double *residc, double *anormc, double *xnormc,
    // control:
    ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    double resid, anorm, xnorm;
    ParU_Ret info;
    info = ParU_Residual (A, X, B, m, nrhs, resid, anorm, xnorm, &Control); 
    *residc = resid;
    *anormc = anorm;
    *xnormc = xnorm;
    return info;
}

//------------------------------------------------------------------------------
//------------ Free routines----------------------------------------------------
//------------------------------------------------------------------------------

ParU_Ret ParU_C_Freenum (
        ParU_C_Numeric **Num_handle_C, ParU_C_Control *Control_C)
{ 
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Numeric *Num_C = *Num_handle_C;
    ParU_Numeric *Num = (ParU_Numeric *) (Num_C->num_handle);
    ParU_Ret info;
    info = ParU_Freenum(&Num, &Control);
    paru_free(1, sizeof(ParU_C_Numeric), *Num_handle_C);
    return info;
}
 
ParU_Ret ParU_C_Freesym (
        ParU_C_Symbolic **Sym_handle_C, ParU_C_Control *Control_C)
{
    ParU_Control Control;
    paru_cp_control (&Control, Control_C);
    ParU_C_Symbolic *Sym_C = *Sym_handle_C;
    ParU_Symbolic *Sym = (ParU_Symbolic *)(Sym_C->sym_handle);
    ParU_Ret info;
    info = ParU_Freesym(&Sym, &Control);
    paru_free(1, sizeof(ParU_C_Symbolic), *Sym_handle_C);
    return info;
}

} //extern c
