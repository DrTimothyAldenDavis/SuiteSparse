////////////////////////////////////////////////////////////////////////////////
//////////////////////////  ParU_Get ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

// ParU_Get: return contents of Symbolic and Numeric objects

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// ParU_Get: get an int64_t scalar or array
//------------------------------------------------------------------------------

ParU_Info ParU_Get
(
    // input:
    const ParU_Symbolic *Sym,   // symbolic analysis from ParU_Analyze
    const ParU_Numeric *Num,    // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    int64_t *result,            // int64_t result: a scalar or an array
    // control:
    ParU_Control *Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = 0 ;

    if (!Sym || !Control)
    {
        return (PARU_INVALID) ;
    }

    size_t n = Sym->n ;

    switch (field)
    {
        case PARU_GET_N:
            (*result) = Sym->n ;
            break ;

        case PARU_GET_ANZ:
            (*result) = Sym->anz ;
            break ;

        case PARU_GET_NROW_SINGLETONS:
            (*result) = Sym->rs1 ;
            break ;

        case PARU_GET_NCOL_SINGLETONS:
            (*result) = Sym->cs1 ;
            break ;

        case PARU_GET_PARU_STRATEGY:
            (*result) = (int64_t) Sym->paru_strategy ;
            break ;

        case PARU_GET_UMFPACK_STRATEGY:
            (*result) = (int64_t) Sym->umfpack_strategy ;
            break ;

        case PARU_GET_UMFPACK_ORDERING:
            (*result) = (int64_t) Sym->umfpack_ordering ;
            break ;

        case PARU_GET_LNZ:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            (*result) = 0 ;     // FIXME: get nnz(L)
            break ;

        case PARU_GET_UNZ:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            (*result) = 0 ;     // FIXME: get nnz(U)
            break ;

        case PARU_GET_P:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            paru_memcpy (result, Num->Pfin, sizeof (int64_t) * n, Control) ;
            break ;

        case PARU_GET_Q:
            paru_memcpy (result, Sym->Qfill, sizeof (int64_t) * n, Control) ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_Get: get a double scalar or array
//------------------------------------------------------------------------------

ParU_Info ParU_Get
(
    // input:
    const ParU_Symbolic *Sym,   // symbolic analysis from ParU_Analyze
    const ParU_Numeric *Num,    // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    double *result,             // double result: a scalar or an array
    // control:
    ParU_Control *Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = 0 ;

    if (!Sym || !Control || !Num || (Sym->n != Num->sym_m))
    {
        return (PARU_INVALID) ;
    }

    size_t n = Sym->n ;

    switch (field)
    {
        case PARU_GET_FLOP_COUNT:
            (*result) = 0 ;     // FIXME: get flop count
            break ;

        case PARU_GET_RCOND_ESTIMATE:
            (*result) = Num->rcond ;
            break ;

        case PARU_GET_MIN_UDIAG:
            (*result) = Num->min_udiag ;
            break ;

        case PARU_GET_MAX_UDIAG:
            (*result) = Num->max_udiag ;
            break ;

        case PARU_GET_ROW_SCALE_FACTORS:
            paru_memcpy (result, Num->Rs, sizeof (double) * n, Control) ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_Get: get a string
//------------------------------------------------------------------------------

ParU_Info ParU_Get
(
    // input:
    const ParU_Symbolic *Sym,   // symbolic analysis from ParU_Analyze
    const ParU_Numeric *Num,    // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    const char **result,        // string result
    // control:
    ParU_Control *Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = NULL ;

    switch (field)
    {

        case PARU_GET_BLAS_LIBRARY_NAME:
            (*result) = SuiteSparse_BLAS_library ( ) ;
            break ;

        case PARU_GET_FRONT_TREE_TASKING:
            #ifdef PARU_1TASK
            (*result) = "sequential" ;
            #else
            (*result) = "parallel" ;
            #endif
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

