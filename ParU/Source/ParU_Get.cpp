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

ParU_Info ParU_Get              // get an int64_t scalar or array from Sym/Num
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    int64_t *result,            // int64_t result: a scalar or an array
    // control:
    ParU_Control Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = 0 ;

    if (!Sym)
    {
        return (PARU_INVALID) ;
    }

    int64_t n = Sym->n ;

    // get Control
    int32_t nthreads = paru_nthreads (Control) ;
    size_t mem_chunk = PARU_DEFAULT_MEM_CHUNK ;
    if (Control != NULL)
    {
        mem_chunk = Control->mem_chunk ;
    }

    switch (field)
    {
        case PARU_GET_N:
            (*result) = n ;
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

        case PARU_GET_STRATEGY:
            (*result) = (int64_t) Sym->strategy_used ;
            break ;

        case PARU_GET_UMFPACK_STRATEGY:
            (*result) = (int64_t) Sym->umfpack_strategy ;
            break ;

        case PARU_GET_ORDERING:
            (*result) = (int64_t) Sym->ordering_used ;
            break ;

        case PARU_GET_LNZ_BOUND:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            (*result) = (int64_t) Num->nnzL ;
            break ;

        case PARU_GET_UNZ_BOUND:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            (*result) = (int64_t) Num->nnzU ;
            break ;

        case PARU_GET_P:
            if (!Num || Num->sym_m != n) return (PARU_INVALID) ;
            paru_memcpy (result, Num->Pfin, sizeof (int64_t) * n,
                mem_chunk, nthreads) ;
            break ;

        case PARU_GET_Q:
            paru_memcpy (result, Sym->Qfill, sizeof (int64_t) * n,
                mem_chunk, nthreads) ;
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

ParU_Info ParU_Get              // get a double scalar or array from Sym/Num
(
    // input:
    const ParU_Symbolic Sym,    // symbolic analysis from ParU_Analyze
    const ParU_Numeric Num,     // numeric factorization from ParU_Factorize
    ParU_Get_enum field,        // field to get
    // output:
    double *result,             // double result: a scalar or an array
    // control:
    ParU_Control Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = 0 ;

    if (!Sym || !Num || (Sym->n != Num->sym_m))
    {
        return (PARU_INVALID) ;
    }

    int64_t n = Sym->n ;

    // get Control
    int32_t nthreads = paru_nthreads (Control) ;
    size_t mem_chunk = PARU_DEFAULT_MEM_CHUNK ;
    if (Control != NULL)
    {
        mem_chunk = Control->mem_chunk ;
    }

    switch (field)
    {
        case PARU_GET_FLOPS_BOUND:
            (*result) = (double) Num->sfc ;     
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
            paru_memcpy (result, Num->Rs, sizeof (double) * n,
                mem_chunk, nthreads) ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_Get: get a string from the Control object
//------------------------------------------------------------------------------

ParU_Info ParU_Get              // get a string from the Control object
(
    // input:
    ParU_Control_enum field,    // field to get
    // output:
    const char **result,        // string result
    // control:
    ParU_Control Control
)
{

    if (!result)
    {
        return (PARU_INVALID) ;
    }

    (*result) = NULL ;

    switch (field)
    {

        case PARU_CONTROL_BLAS_LIBRARY_NAME:
            (*result) = SuiteSparse_BLAS_library ( ) ;
            break ;

        case PARU_CONTROL_FRONT_TREE_TASKING:
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

//------------------------------------------------------------------------------
// ParU_Get: get an int64_t parameter from the Control object
//------------------------------------------------------------------------------

ParU_Info ParU_Get          // get an int64_t parameter from the Control object
(
    // input
    ParU_Control_enum parameter,    // parameter to get
    // output:
    int64_t *c,                     // value of parameter
    // control:
    ParU_Control Control
)
{

    if (!c)
    {
        return (PARU_INVALID) ;
    }

    (*c) = 0 ;

    switch (parameter)
    {

        case PARU_CONTROL_MAX_THREADS:            // max number of threads
            // 0 is the default, which lets ParU use the # of threads
            // determined by omp_get_max_threads.
            (*c) = (Control == NULL) ? PARU_DEFAULT_MAX_THREADS :
                Control->paru_max_threads ;
            break ;

        case PARU_CONTROL_NUM_THREADS:            // actual # of threads
            // This is the actual # of threads ParU will use.
            (*c) = (int64_t) paru_nthreads (Control) ;
            break ;

        case PARU_CONTROL_OPENMP:                 // 1 if OpenMP, 0 if not
            #if defined ( _OPENMP )
            (*c) = 1 ;
            #else
            (*c) = 0 ;
            #endif
            break ;

        case PARU_CONTROL_STRATEGY:               // ParU strategy
            (*c) = (Control == NULL) ? PARU_DEFAULT_STRATEGY :
                Control->strategy ;
            break ;

        case PARU_CONTROL_UMFPACK_STRATEGY:       // UMFPACK strategy
            (*c) = (Control == NULL) ? PARU_DEFAULT_UMFPACK_STRATEGY :
                Control->umfpack_strategy ;
            break ;

        case PARU_CONTROL_ORDERING:               // ParU ordering
            (*c) = (Control == NULL) ? PARU_DEFAULT_ORDERING :
                Control->ordering ;
            break ;

        case PARU_CONTROL_RELAXED_AMALGAMATION:   // goal # pivots in each front
            (*c) = (Control == NULL) ? PARU_DEFAULT_RELAXED_AMALGAMATION :
                Control->relaxed_amalgamation ;
            break ;

        case PARU_CONTROL_PANEL_WIDTH:            // # of pivots in a panel
            (*c) = (Control == NULL) ? PARU_DEFAULT_PANEL_WIDTH :
                Control->panel_width ;
            break ;

        case PARU_CONTROL_DGEMM_TINY:             // dimension of tiny dgemm's
            (*c) = (Control == NULL) ? PARU_DEFAULT_DGEMM_TINY :
                Control->trivial ;
            break ;

        case PARU_CONTROL_DGEMM_TASKED:           // dimension of tasked dgemm's
            (*c) = (Control == NULL) ? PARU_DEFAULT_DGEMM_TASKED :
                Control->worthwhile_dgemm ;
            break ;

        case PARU_CONTROL_DTRSM_TASKED:           // dimension of tasked dtrsm's
            (*c) = (Control == NULL) ? PARU_DEFAULT_DTRSM_TASKED :
                Control->worthwhile_dtrsm ;
            break ;

        case PARU_CONTROL_PRESCALE:               // prescale input matrix
            (*c) = (Control == NULL) ? PARU_DEFAULT_PRESCALE :
                Control->prescale ;
            break ;

        case PARU_CONTROL_SINGLETONS:             // filter singletons, or not
            (*c) = (Control == NULL) ? PARU_DEFAULT_SINGLETONS :
                Control->filter_singletons ;
            break ;

        case PARU_CONTROL_MEM_CHUNK:              // chunk size memset, memcpy
            (*c) = (Control == NULL) ? PARU_DEFAULT_MEM_CHUNK :
                Control->mem_chunk ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}

//------------------------------------------------------------------------------
// ParU_Get: get a double parameter from the Control object
//------------------------------------------------------------------------------

ParU_Info ParU_Get          // get a double parameter from the Control object
(
    // input
    ParU_Control_enum parameter,    // parameter to get
    // output:
    double *c,                      // value of parameter
    // control:
    ParU_Control Control
)
{

    if (!c)
    {
        return (PARU_INVALID) ;
    }

    (*c) = 0 ;

    switch (parameter)
    {

        case PARU_CONTROL_PIVOT_TOLERANCE:          // pivot tolerance
            (*c) = (Control == NULL) ? PARU_DEFAULT_PIVOT_TOLERANCE :
                Control->piv_toler ;
            break ;

        case PARU_CONTROL_DIAG_PIVOT_TOLERANCE:     // diagonal pivot tolerance
            (*c) = (Control == NULL) ? PARU_DEFAULT_DIAG_PIVOT_TOLERANCE :
                Control->diag_toler ;
            break ;

        default:
            return (PARU_INVALID) ;
            break ;
    }

    return (PARU_SUCCESS) ;
}
