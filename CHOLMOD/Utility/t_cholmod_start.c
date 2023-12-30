//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_start: start CHOLMOD (int32/int64 version)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// cholmod_start or cholmod_l_start must be called once prior to calling any
// other CHOLMOD method.  It contains workspace that must be freed by
// cholmod_finish or cholmod_l_finish.

int CHOLMOD(start) (cholmod_common *Common)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (Common == NULL) return (FALSE) ;

    //--------------------------------------------------------------------------
    // settings required before CHOLMOD(defaults)
    //--------------------------------------------------------------------------

    memset ((void *) Common, 0, sizeof (struct cholmod_common_struct)) ;
    Common->itype = ITYPE ;     // CHOLMOD_INT or CHOLMOD_LONG

    //--------------------------------------------------------------------------
    // set defaults
    //--------------------------------------------------------------------------

    CHOLMOD(defaults) (Common) ;

    //--------------------------------------------------------------------------
    // initialize the rest of Common to various nonzero values
    //--------------------------------------------------------------------------

    Common->gpuMemorySize = 1 ;
    Common->chunk = 128000 ;
    Common->nthreads_max = SUITESPARSE_OPENMP_MAX_THREADS ;

    Common->modfl = EMPTY ;
    Common->aatfl = EMPTY ;
    Common->blas_ok = TRUE ;

    Common->SPQR_grain = 1 ;
    Common->SPQR_small = 1e6 ;
    Common->SPQR_shrink = 1 ;

    Common->mark = EMPTY ;
    Common->fl = EMPTY ;
    Common->lnz = EMPTY ;

    #ifdef BLAS_DUMP
    Common->blas_dump = fopen ("blas_dump.txt", "a") ;
    #endif

    return (TRUE) ;
}

