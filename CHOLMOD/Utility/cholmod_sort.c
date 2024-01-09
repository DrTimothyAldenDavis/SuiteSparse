//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sort: sort a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Sorts the entries in each column of a sparse matrix.

#include "cholmod_internal.h"

//------------------------------------------------------------------------------
// random number generator for cm_qsort
//------------------------------------------------------------------------------

// return a random uint64_t, in range 0 to 2^60
#define CM_RAND_MAX 32767

// return a random number between 0 and CM_RAND_MAX
static inline uint64_t cm_rand15 (uint64_t *seed)
{
   (*seed) = (*seed) * 1103515245 + 12345 ;
   return (((*seed) / 65536) % (CM_RAND_MAX + 1)) ;
}

// return a random uint64_t, in range 0 to 2^60
static inline uint64_t cm_rand (uint64_t *seed)
{
    uint64_t i = cm_rand15 (seed) ;
    i = CM_RAND_MAX * i + cm_rand15 (seed) ;
    i = CM_RAND_MAX * i + cm_rand15 (seed) ;
    i = CM_RAND_MAX * i + cm_rand15 (seed) ;
    return (i) ;
}

// swap two entries A [a] and A [b]
#define SWAP(type,A,a,b)                    \
{                                           \
    type t = A [a] ;                        \
    A [a] = A [b] ;                         \
    A [b] = t ;                             \
}

//------------------------------------------------------------------------------
// t_cholmod_sort_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_sort_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_sort_worker.c"
#define COMPLEX
#include "t_cholmod_sort_worker.c"
#define ZOMPLEX
#include "t_cholmod_sort_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_sort_worker.c"
#define COMPLEX
#include "t_cholmod_sort_worker.c"
#define ZOMPLEX
#include "t_cholmod_sort_worker.c"

//------------------------------------------------------------------------------
// cholmod_sort
//------------------------------------------------------------------------------

int CHOLMOD(sort)
(
    // input/output:
    cholmod_sparse *A,      // input/output matrix to sort
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, FALSE) ;
    ASSERT (CHOLMOD(dump_sparse) (A, "sort:A input", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // sort each column of A
    //--------------------------------------------------------------------------

    switch ((A->xtype + A->dtype) % 8)
    {
        default:
            p_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            cs_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            zs_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            cd_cholmod_sort_worker (A) ;
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            zd_cholmod_sort_worker (A) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (CHOLMOD(dump_sparse) (A, "sort:A output", Common) >= 0) ;
    return (TRUE) ;
}

