//------------------------------------------------------------------------------
// SPEX_LU/spex_left_lu_get_pivot: find a pivot entry in a column
//------------------------------------------------------------------------------

// SPEX_LU: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,,
// Erick Moreno-Centeno, and Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* This function performs the pivoting for the SPEX Left LU factorization.
 * The optional Order is:
 *
 *  SPEX_SMALLEST = 0,      Smallest pivot
 *  SPEX_DIAGONAL = 1,      Diagonal pivoting
 *  SPEX_FIRST_NONZERO = 2, First nonzero per column chosen as pivot
 *  SPEX_TOL_SMALLEST = 3,  Diagonal pivoting with tolerance for pivot (default)
 *  SPEX_TOL_LARGEST = 4,   Diagonal pivoting with tolerance for largest pivot
 *  SPEX_LARGEST = 5        Largest pivot
 *
 * Options 2, 4 and 5 are not recommended and may lead to significant drops in
 * performance.
 *
 * On output, the pivs, rhos, pinv, and row_perm arrays are all modified.
 */

#define SPEX_FREE_ALL           \
    SPEX_mpq_clear (tol);       \
    SPEX_mpq_clear (ratio);

#include "spex_lu_internal.h"

SPEX_info spex_left_lu_get_pivot
(
    int64_t *pivot,         // found index of pivot entry
    SPEX_matrix x,          // kth column of L and U
    int64_t *pivs,          // vector indicating which rows have been pivotal
    int64_t n,              // dimension of the problem
    int64_t top,            // nonzero pattern is located in xi[top..n-1]
    int64_t *xi,            // nonzero pattern of x
    int64_t col,            // current column of A (real kth column i.e., q[k])
    int64_t k,              // iteration of the algorithm
    SPEX_matrix rhos,       // vector of pivots
    int64_t *pinv,          // row permutation
    int64_t *row_perm,      // opposite of pinv.
                            // if pinv[i] = j then row_perm[j] = i
    const SPEX_options option // command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    SPEX_info info ;
    SPEX_REQUIRE(rhos, SPEX_DENSE, SPEX_MPZ);
    SPEX_REQUIRE(x, SPEX_DENSE, SPEX_MPZ);

    // pivoting method to use (see above description)
    SPEX_pivot order = SPEX_OPTION_PIVOT(option);
    // tolerance used if some tol-based pivoting is used
    double tolerance = SPEX_OPTION_TOL(option);

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    int sgn, r;
    mpq_t tol, ratio;
    SPEX_mpq_set_null (tol);
    SPEX_mpq_set_null (ratio);

    if (order == SPEX_SMALLEST)
    {

        //----------------------------------------------------------------------
        // Smallest pivot
        //----------------------------------------------------------------------

        SPEX_CHECK(spex_left_lu_get_smallest_pivot(pivot, x, pivs, n, top, xi));

    }
    else if (order == SPEX_DIAGONAL)
    {

        //----------------------------------------------------------------------
        // Diagonal
        //----------------------------------------------------------------------

        // Check if x[col] is eligible. take smallest pivot    if not
        SPEX_MPZ_SGN(&sgn, x->x.mpz[col]);
        if (sgn != 0 && pivs[col] < 0)
        {
            *pivot = col;
        }
        else
        {
            SPEX_CHECK (spex_left_lu_get_smallest_pivot(pivot, x, pivs, n,
                top, xi));
        }

    }
    else if (order == SPEX_FIRST_NONZERO)
    {

        //----------------------------------------------------------------------
        // First nonzero
        //----------------------------------------------------------------------

        SPEX_CHECK (spex_left_lu_get_nonzero_pivot(pivot, x, pivs, n, top, xi));

    }
    else if (order == SPEX_TOL_LARGEST)
    {

        //----------------------------------------------------------------------
        // Tolerance with largest pivot
        //----------------------------------------------------------------------

        SPEX_CHECK (spex_left_lu_get_largest_pivot(pivot, x, pivs, n, top, xi));

        //----------------------------------------------------------------------
        // Check x[col] vs largest potential pivot
        //----------------------------------------------------------------------
        SPEX_MPZ_SGN(&sgn, x->x.mpz[col]);
        if (sgn != 0 && pivs[col] < 0)
        {
            SPEX_MPQ_INIT(tol);
            SPEX_MPQ_INIT(ratio);
            // tol = user specified tolerance
            SPEX_MPQ_SET_D(tol, tolerance);
            // ratio = diagonal/largest
            SPEX_MPQ_SET_NUM(ratio, x->x.mpz[col]);
            SPEX_MPQ_SET_DEN(ratio, x->x.mpz[*pivot]);
            // ratio = |ratio|
            SPEX_MPQ_ABS(ratio, ratio);

            // Is ratio >= tol?
            SPEX_MPQ_CMP(&r, ratio, tol);
            if (r >= 0)
            {
                *pivot = col;
            }
        }

    }
    else if (order == SPEX_LARGEST)
    {

        //----------------------------------------------------------------------
        // Use the largest potential pivot
        //----------------------------------------------------------------------

        SPEX_CHECK (spex_left_lu_get_largest_pivot(pivot, x, pivs, n, top, xi));

    }
    else // if (order == SPEX_TOL_SMALLEST)
    {

        //----------------------------------------------------------------------
        // Tolerance with smallest pivot (default option)
        //----------------------------------------------------------------------

        SPEX_CHECK (spex_left_lu_get_smallest_pivot(pivot, x, pivs, n, top,
            xi));

        //----------------------------------------------------------------------
        // Checking x[col] vs smallest pivot
        //----------------------------------------------------------------------
        SPEX_MPZ_SGN(&sgn, x->x.mpz[col]);
        if (sgn != 0 && pivs[col] < 0)
        {

            // Initialize tolerance and ratio
            SPEX_MPQ_INIT(tol);
            SPEX_MPQ_INIT(ratio);

            // ratio = |smallest/diagonal|
            SPEX_MPZ_ABS(SPEX_MPQ_NUM(ratio), x->x.mpz[*pivot]);
            SPEX_MPZ_ABS(SPEX_MPQ_DEN(ratio), x->x.mpz[col]);

            // Set user specified tolerance
            SPEX_MPQ_SET_D(tol, tolerance);

            // Is ratio >= tol?
            SPEX_MPQ_CMP(&r, ratio, tol);
            if (r >= 0)
            {
                *pivot = col;
            }
        }
    }

    //--------------------------------------------------------------------------
    // Reflect changes in row location & row_perm
    //--------------------------------------------------------------------------

    // Must move pivot into position k
    int64_t intermed = pinv[*pivot];
    int64_t intermed2 = row_perm[k];

    //--------------------------------------------------------------------------
    // Set row_perm[k] = pivot and row_perm[pinv[pivot]] = row_perm[k]
    // Also, set pinv[pivot] = k and pinv[row_perm[k]] = pinv[pivot]
    //--------------------------------------------------------------------------
    row_perm[k] = *pivot;
    row_perm[intermed] = intermed2;
    pinv[*pivot] = k;
    pinv[intermed2] = intermed;
    // Row pivot is now pivotal
    pivs[*pivot] = 1;

    // Set the kth pivot.
    size_t size;
    // Get the size of x[pivot]
    SPEX_MPZ_SIZEINBASE(&size, x->x.mpz[*pivot], 2);
    // GMP manual: Allocated size should be size+2
    SPEX_MPZ_INIT2(rhos->x.mpz[k], size+2);
    // The kth pivot is x[pivot]
    SPEX_MPZ_SET(rhos->x.mpz[k], x->x.mpz[*pivot]);

    // Free memory
    SPEX_FREE_ALL;
    return SPEX_OK;
}

