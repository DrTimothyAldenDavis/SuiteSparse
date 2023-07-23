// =============================================================================
// === spqr_rhpack =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*  Copy the R matrix and optionally the H matrix in the frontal matrix F and
    store them in packed form.  The columns of R and H are interleaved.  F can
    be overwritten with R+H (the pack can occur in-place), but in that case,
    the C matrix is destroyed.  If H is not kept and the pack is done in-place,
    then H is destroyed.  H is not stored in packed form if it is not kept.

    If dead columns appear in F, the leading part of R is a squeezed upper
    triangular matrix.  In the example below, m = 9, n =  6, npiv = 4,
    and the 3rd column is dead (Stair [2] == 0).

        0:  r r r r r r     <- Stair [2] = 0, denotes dead column
        1:  h r r r r r
        2:  h h . r r r
        3:  h h . h c c
        4:  - h . h h c     <- Stair [0] = 4
        5:  . h . h h h
        6:  . - . h h h     <- Stair [1] = 6
        7:  . . . - - h     <- Stair [3] = Stair [4] = 6
        8:  . . . . . h
                      -     <- Stair [5] = 9

    The number of rows of R is equal to the number of leading npiv columns that
    are not dead; this number must be <= m.  In this example, rm = 3.

    The staircase defines the number of entries in each column, or is equal
    to zero to denote a dead column.
*/

#include "spqr.hpp"

template <typename Entry, typename Int> Int spqr_rhpack   // returns # of entries in R+H
(
    // input, not modified
    int keepH,              // if true, then H is packed
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int *Stair,            // size npiv; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.

    // input, not modified (unless the pack occurs in-place)
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *R,               // packed columns of R+H
    Int *p_rm              // number of rows in R block
)
{
    Entry *R0 = R ;
    Int i, k, h, t, rm ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    ASSERT (m >= 0 && n >= 0 && npiv <= n && npiv >= 0) ;

    if (m <= 0 || n <= 0)
    {
        *p_rm = 0 ;                     // no rows in R block
        return (0) ;                    // nothing to do
    }

    ASSERT (R != NULL && F != NULL) ;
    ASSERT (R <= F                      // can be packed in-place, in F
         || R >= F + m*n) ;             // or must appear after F

    // -------------------------------------------------------------------------
    // pack the squeezed upper triangular part of R
    // -------------------------------------------------------------------------

    rm = 0 ;                            // number of rows in R block
    for (k = 0 ; k < npiv ; k++)
    {
        // get the staircase
        t = Stair [k] ;                 // F (0:t-1,k) contains R and H
        ASSERT (t >= 0 && t <= m) ;
        if (t == 0)
        {
            t = rm ;                    // dead col, R (0:rm-1,k) only, no H
        }
        else if (rm < m)
        {
            rm++ ;                      // column k is not dead
        }
        if (keepH)
        {
            // pack R (0:rm-1,k) and H (rm:t-1,k)
            for (i = 0 ; i < t ; i++)
            {
                *(R++) = F [i] ;
            }
        }
        else
        {
            // pack R (0:rm-1,k), discarding H
            for (i = 0 ; i < rm ; i++)
            {
                *(R++) = F [i] ;
            }
        }
        F += m ;                        // advance to the next column of F
    }

    // -------------------------------------------------------------------------
    // pack the rectangular part of R
    // -------------------------------------------------------------------------

    h = rm ;                        // the column of H starts in row h
    for ( ; k < n ; k++)
    {

        // pack R (0:rm-1,k)
        for (i = 0 ; i < rm ; i++)
        {
            *(R++) = F [i] ;
        }

        if (keepH)
        {
            // pack H (h:t-1,k)
            t = Stair [k] ;             // get the staircase
            ASSERT (t >= rm && t <= m) ;
            h = MIN (h+1, m) ;          // one more row of C to skip over
            for (i = h ; i < t ; i++)
            {
                *(R++) = F [i] ;
            }
        }

        F += m ;                    // advance to the next column of F
    }

    *p_rm = rm ;                        // return # of rows in R block
    return (R-R0) ;                     // return # of packed entries
}


// explicit instantiations

template int32_t spqr_rhpack <double, int32_t>
(
    int keepH, int32_t m, int32_t n, int32_t npiv, int32_t *Stair, double *F,
    double *R, int32_t *p_rm
) ;

template int32_t spqr_rhpack <Complex, int32_t> 
(
    int keepH, int32_t m, int32_t n, int32_t npiv, int32_t *Stair, Complex *F,
    Complex *R, int32_t *p_rm
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long spqr_rhpack <double, SuiteSparse_long>
(
    int keepH, SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long npiv,
    SuiteSparse_long *Stair, double *F, double *R, SuiteSparse_long *p_rm
) ;

template SuiteSparse_long spqr_rhpack <Complex, SuiteSparse_long>
(
    int keepH, SuiteSparse_long m, SuiteSparse_long n, SuiteSparse_long npiv,
    SuiteSparse_long *Stair, Complex *F, Complex *R, SuiteSparse_long *p_rm
) ;

#endif
