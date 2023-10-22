//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_sort_worker: sort all columns of a sparse matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

//------------------------------------------------------------------------------
// variants for each xtype: pattern, real, complex, and zomplex
//------------------------------------------------------------------------------

#if defined ( PATTERN )

    #define CM_PART(Ai,Ax,Az,n,seed)   TEMPLATE(cm_part) (Ai, n, seed)
    #define CM_QSRT(Ai,Ax,Az,p,n,seed) TEMPLATE(cm_qsrt) (Ai+p, n, seed)
    #define CM_SWAP(a,b)                    \
    {                                       \
        SWAP (Int, Ai, a, b) ;              \
    }

#elif defined ( REAL )

    #define CM_PART(Ai,Ax,Az,n,seed)   TEMPLATE(cm_part) (Ai, Ax, n, seed)
    #define CM_QSRT(Ai,Ax,Az,p,n,seed) TEMPLATE(cm_qsrt) (Ai+p, Ax+p, n, seed)
    #define CM_SWAP(a,b)                    \
    {                                       \
        SWAP (Int,  Ai, a, b) ;             \
        SWAP (Real, Ax, a, b) ;             \
    }

#elif defined ( COMPLEX )

    #define CM_PART(Ai,Ax,Az,n,seed)   TEMPLATE(cm_part) (Ai, Ax, n, seed)
    #define CM_QSRT(Ai,Ax,Az,p,n,seed) TEMPLATE(cm_qsrt) (Ai+p,Ax+2*p,n,seed)
    #define CM_SWAP(a,b)                    \
    {                                       \
        SWAP (Int,  Ai, a, b) ;             \
        SWAP (Real, Ax, 2*(a),   2*(b)) ;   \
        SWAP (Real, Ax, 2*(a)+1, 2*(b)+1) ; \
    }

#else

    #define CM_PART(Ai,Ax,Az,n,seed)   TEMPLATE(cm_part) (Ai, Ax, Az, n, seed)
    #define CM_QSRT(Ai,Ax,Az,p,n,seed) TEMPLATE(cm_qsrt) (Ai+p,Ax+p,Az+p,n,seed)
    #define CM_SWAP(a,b)                    \
    {                                       \
        SWAP (Int,  Ai, a, b) ;             \
        SWAP (Real, Ax, a, b) ;             \
        SWAP (Real, Az, a, b) ;             \
    }

#endif

//------------------------------------------------------------------------------
// cm_part: use a pivot to partition an array
//------------------------------------------------------------------------------

// C.A.R Hoare partition method, partitions an array in-place via a pivot.
// k = partition (A, n) partitions A [0:n-1] such that all entries in
// A [0:k] are <= all entries in A [k+1:n-1].

// Ai [0:n-1] is the sort key, and Ax,Az [0:n-1] are satelite data.


static inline Int TEMPLATE (cm_part)
(
    Int  *Ai,               // Ai [0..n-1]: indices to sort
    #if !defined ( PATTERN )
    Real *Ax,               // Ax [0..n-1]: values of A(:,j)
    #endif
    #if defined ( ZOMPLEX )
    Real *Az,               // Az [0..n-1]
    #endif
    const Int n,            // length of Ai, Ax, Az
    uint64_t *seed          // random number seed, modified on output
)
{

    // select a pivot at random
    Int p = ((n < CM_RAND_MAX) ? cm_rand15 (seed) : cm_rand (seed)) % n ;

    // get the pivot value
    Int pivot = Ai [p] ;

    // At the top of the while loop, A [left+1...right-1] is considered, and
    // entries outside this range are in their proper place and not touched.
    // Since the input specification of this function is to partition A
    // [0..n-1], left must start at -1 and right must start at n.
    Int left = -1 ;
    Int right = n ;

    // keep partitioning until the left and right sides meet
    while (true)
    {
        // loop invariant:  A [0..left] < pivot and A [right..n-1] > pivot,
        // so the region to be considered is A [left+1 ... right-1].

        // increment left until finding an entry A [left] >= pivot
        do { left++ ; } while (Ai [left] < pivot) ;

        // decrement right until finding an entry A [right] <= pivot
        do { right-- ; } while (pivot < Ai [right]) ;

        // now A [0..left-1] < pivot and A [right+1..n-1] > pivot, but
        // A [left] > pivot and A [right] < pivot, so these two entries
        // are out of place and must be swapped.

        // However, if the two sides have met, the partition is finished.
        if (left >= right)
        {
            // A has been partitioned into A [0:right] and A [right+1:n-1].
            // k = right+1, so A is split into A [0:k-1] and A [k:n-1].
            return (right + 1) ;
        }

        // since A [left] > pivot and A [right] < pivot, swap them
        CM_SWAP (left, right) ;

        // after the swap this condition holds:
        // A [0..left] < pivot and A [right..n-1] > pivot
    }
}

//------------------------------------------------------------------------------
// cm_qsrt: recursive single-threaded quicksort
//------------------------------------------------------------------------------

static void TEMPLATE (cm_qsrt)     // sort A [0:n-1]
(
    Int  *Ai,               // Ai [0..n-1]: indices to sort
    #if !defined ( PATTERN )
    Real *Ax,               // Ax [0..n-1]: values of A(:,j)
    #endif
    #if defined ( ZOMPLEX )
    Real *Az,               // Az [0..n-1]
    #endif
    const Int n,            // length of Ai, Ax, Az
    uint64_t *seed          // random number seed
)
{

    if (n < 20)
    {
        // in-place insertion sort on A [0:n-1], where n is small
        for (Int k = 1 ; k < n ; k++)
        {
            for (Int j = k ; j > 0 && Ai [j] < Ai [j-1] ; j--)
            {
                // swap A [j-1] and A [j]
                CM_SWAP (j-1, j) ;
            }
        }
    }
    else
    {
        // partition A [0:n-1] into A [0:k-1] and A [k:n-1]
        Int k = CM_PART (Ai, Ax, Az, n, seed) ;

        // sort each partition
        CM_QSRT (Ai, Ax, Ax, 0, k,   seed) ;  // sort A [0:k-1]
        CM_QSRT (Ai, Ax, Az, k, n-k, seed) ;  // sort A [k+1:n-1]
    }
}

//------------------------------------------------------------------------------
// t_cholmod_sort_worker
//------------------------------------------------------------------------------

static void TEMPLATE (cholmod_sort_worker)
(
    cholmod_sparse *A   // matrix to sort in-place
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Ap  = (Int  *) A->p ;
    Int  *Ai  = (Int  *) A->i ;
    Int  *Anz = (Int  *) A->nz ;
    Real *Ax  = (Real *) A->x ;
    Real *Az  = (Real *) A->z ;
    Int ncol = A->ncol ;
    bool packed = A->packed ;
    uint64_t seed = 42 ;

    for (Int j = 0 ; j < ncol ; j++)
    {

        //----------------------------------------------------------------------
        // sort A(:,j) according to its row indices
        //----------------------------------------------------------------------

        Int pa = Ap [j] ;
        Int pend = (packed) ? (Ap [j+1]) : (pa + Anz [j]) ;
        Int ilast = EMPTY ;
        for (Int p = pa ; p < pend ; p++)
        {
            Int i = Ai [p] ;
            if (i < ilast)
            {
                // sort Ai, Ax, Ax [pa:pend-1] according to row index Ai
                Int jnz = pend - pa ;
                CM_QSRT (Ai, Ax, Az, pa, pend - pa, &seed) ;
                break ;
            }
            ilast = i ;
        }
    }

    A->sorted = TRUE ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

#undef  CM_SWAP
#undef  CM_PART
#undef  CM_QSRT

