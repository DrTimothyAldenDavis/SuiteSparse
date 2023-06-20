// =============================================================================
// === spqr_cpack ==============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//  spqr_cpack copies the C matrix from the frontal matrix F and
//  stores it in packed form.  F can be overwritten with C (the pack can occur
//  in-place), but in that case, the R and H matrices are destroyed.
//
//  In this example, m = 5, n = 6, npiv = 2, and rank = 2 (the number of good
//  pivot columns found; equivalently, the number of rows in the R block).  The
//  number of columns in C is cn = n-npiv.  The number of rows is
//  cm = MIN (m-rank,cn).  In this example, MIN (m-rank,cn) = 3.
//
//      . . . . . .
//      . . . . . .
//      . . c c c c     <- rank = 2
//      . . . c c c
//      . . . . c c
//
//  In the next example below, m = 8 instead.  Note that
//  cm = MIN (m-rank,cn) = 4.
//
//      . . . . . .
//      . . . . . .
//      . . c c c c     <- rank = 2
//      . . . c c c
//      . . . . c c
//      . . . . . c
//      . . . . . .
//      . . . . . .

#include "spqr.hpp"

template <typename Entry, typename Int> Int spqr_cpack     // returns # of rows in C
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int rank,              // the C block starts at F (rank,npiv)

    // input, not modified unless the pack occurs in-place
    Entry *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Entry *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
)
{
    Int i, k, cm, cn ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    ASSERT (m >= 0 && n >= 0 && npiv >= 0 && npiv <= n) ;

    ASSERT (rank >= 0 && rank <= MIN (m,npiv)) ;
    cn = n - npiv ;                     // number of columns of C
    cm = MIN (m-rank, cn) ;             // number of rows of C
    ASSERT (cm <= cn) ;
    if (cm <= 0 || cn <= 0)
    {
        return (0) ;                    // nothing to do
    }

    ASSERT (C != NULL && F != NULL) ;
    ASSERT (C <= F                      // C can be packed in-place, in F
         || C >= F + m*n) ;             // or C must appear after F

    F += INDEX (rank,npiv,m) ;          // C starts at F (rank,npiv)

    // -------------------------------------------------------------------------
    // pack the upper triangular part of C
    // -------------------------------------------------------------------------

    for (k = 0 ; k < cm ; k++)
    {
        // pack C (0:k,k)
        for (i = 0 ; i <= k ; i++)
        {
            *(C++) = F [i] ;
        }
        F += m ;                        // advance to the next column of F
    }

    // -------------------------------------------------------------------------
    // pack the rectangular part of C
    // -------------------------------------------------------------------------

    for ( ; k < cn ; k++)
    {
        // pack C (0:cm-1,k)
        for (i = 0 ; i < cm ; i++)
        {
            *(C++) = F [i] ;
        }
        F += m ;                        // advance to the next column of F
    }

    PR (("Cpack rank %ld cm %ld cn %ld\n", rank, cm, cn)) ;
    return (cm) ;                       // return # of rows in C
}
template int32_t spqr_cpack <double, int32_t>     // returns # of rows in C
(
    // input, not modified
    int32_t m,                 // # of rows in F
    int32_t n,                 // # of columns in F
    int32_t npiv,              // number of pivotal columns in F
    int32_t rank,              // the C block starts at F (rank,npiv)

    // input, not modified unless the pack occurs in-place
    double *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    double *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;
template int32_t spqr_cpack <Complex, int32_t>     // returns # of rows in C
(
    // input, not modified
    int32_t m,                 // # of rows in F
    int32_t n,                 // # of columns in F
    int32_t npiv,              // number of pivotal columns in F
    int32_t rank,              // the C block starts at F (rank,npiv)

    // input, not modified unless the pack occurs in-place
    Complex *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Complex *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;
template int64_t spqr_cpack <double, int64_t>     // returns # of rows in C
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t rank,              // the C block starts at F (rank,npiv)

    // input, not modified unless the pack occurs in-place
    double *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    double *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;
template int64_t spqr_cpack <Complex, int64_t>     // returns # of rows in C
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t rank,              // the C block starts at F (rank,npiv)

    // input, not modified unless the pack occurs in-place
    Complex *F,               // m-by-n frontal matrix in column-major order

    // output, contents not defined on input
    Complex *C                // packed columns of C, of size cm-by-cn in upper
                            // trapezoidal form.
) ;
