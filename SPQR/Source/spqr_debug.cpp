// =============================================================================
// === spqr_debug ==============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "spqr.hpp"

// =============================================================================
// This file is for debugging only.
// =============================================================================
#ifndef NDEBUG

#ifndef NPRINT

// =============================================================================
// === spqrDebug_print =========================================================
// =============================================================================

void spqrDebug_print
(
    double x
)
{
    PR ((" %10.4g", x)) ;
}

void spqrDebug_print
(
    Complex x
)
{
    PR ((" (%10.4g + 1i*(%10.4g))", x.real ( ), x.imag ( ))) ;
}

// =============================================================================
// === spqrDebug_dumpdense =====================================================
// =============================================================================

template <typename Entry, typename Int> void spqrDebug_dumpdense
(
    Entry *A,
    Int m,
    Int n,
    Int lda,
    cholmod_common *cc
)
{
    Int i, j ;
    if (cc == NULL) return ;
    PR (("Dense: m %ld n %ld lda %ld p %p\n", m, n, lda, A)) ;
    if (m < 0 || n < 0 || lda < m || A == NULL)
    {
        PR (("bad dense matrix!\n")) ;
        ASSERT (0) ;
        return ;
    }

    for (j = 0 ; j < n ; j++)
    {
        PR (("   --- column %ld of %ld\n", j, n)) ;
        for (i = 0 ; i < m ; i++)
        {
            if (i == j) PR (("      [ diag:     ")) ;
            else        PR (("      row %4ld    ", i)) ;
            spqrDebug_print (A [i + j*lda]) ;
            if (i == j) PR ((" ]\n")) ;
            else        PR (("\n")) ;
        }
        PR (("\n")) ;
    }

#if 0
    for (i = 0 ; i < m ; i++)
    {
        for (j = 0 ; j < n ; j++)
        {
//          if (A [i+j*lda] != (Entry) 0)
//          {
//              PR (("X"))  ;
//          }
//          else
//          {
//              PR (("."))  ;
//          }
            spqrDebug_print (A [i + j*lda]) ;
        }
        PR (("\n")) ;
    }
#endif

}

// =============================================================================
// === spqrDebug_dumpsparse ====================================================
// =============================================================================

template <typename Entry, typename Int> void spqrDebug_dumpsparse
(
    Int *Ap,
    Int *Ai,
    Entry *Ax,
    Int m,
    Int n,
    cholmod_common *cc
)
{
    Int p, i, j ;
    if (cc == NULL) return ;
    PR (("\nSparse: m %ld n %ld nz %ld Ap %p Ai %p Ax %p\n",
        m, n, Ap [n], Ap, Ai,Ax)) ;
    if (m < 0 || n < 0 || Ax == NULL || Ap == NULL || Ai == NULL) return ;
    for (j = 0 ; j < n ; j++)
    {
        PR (("  column %ld\n", j)) ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            PR (("   %ld :", i)) ;
            spqrDebug_print (Ax [p]) ;
            PR (("\n")) ;
            ASSERT (i >= 0 && i < m) ;
        }
    }
}

#endif

// =============================================================================
// === spqrDebug_listcount =====================================================
// =============================================================================

#ifdef DEBUG_EXPENSIVE

// returns # of times x is in the List [0..len-1]
template <typename Int> Int spqrDebug_listcount
(
    Int x, Int *List, Int len, Int what,
    cholmod_common *cc
)
{
    Int k, nfound = 0 ;
    if (cc == NULL) return (EMPTY) ;
    if (what == 0)
    {
        k = 0 ;
        PR (("\nQfill, j %ld len %ld\n", x, len)) ;
    }
    if (what == 1)
    {
        k = 0 ;
        PR (("\nQrows, i %ld len %ld\n", x, len)) ;
    }
    for (k = 0 ; k < len ; k++)
    {
        if (List [k] == x) nfound++ ;
        PR (("   %ld ( %ld ) %ld\n", x, List [k], nfound)) ;
    }
    PR (("total found %ld\n\n", nfound)) ;
    return (nfound) ;
}
#endif

// =============================================================================
// === spqrDebug_rhsize ========================================================
// =============================================================================

// Count the number of entries in the R+H block for a single front.

template <typename Int> Int spqrDebug_rhsize    // returns # of entries in R+H
(
    // input, not modified
    Int m,                 // # of rows in F
    Int n,                 // # of columns in F
    Int npiv,              // number of pivotal columns in F
    Int *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
)
{
    Int k, h, t, rm, rhsize = 0 ;

    ASSERT (m >= 0 && n >= 0 && npiv <= n && npiv >= 0) ;

    if (cc == NULL) return (EMPTY) ;
    if (m <= 0 || n <= 0) return (0) ;                     // nothing to do

    PR (("Try RHSIZE: m %ld n %ld npiv %ld\n", m, n, npiv)) ;

    // -------------------------------------------------------------------------
    // count the squeezed part of R+H
    // -------------------------------------------------------------------------

    rm = 0 ;                            // number of rows in R (:,0:k)
    for (k = 0 ; k < npiv ; k++)
    {
        // get the staircase
        t = Stair [k] ;                 // F (0:t-1,k) contains R and H
        if (t == 0)
        {
            t = rm ;                    // dead col, R (0:rm-1,k) only, no H
        }
        else if (rm < m)
        {
            rm++ ;                      // col k not dead; one more row of R
        }
        PR (("  for RHSIZE, k %ld Stair %ld t %ld (piv)\n", k, Stair[k], t)) ;
        // pack R (0:rm-1,k) and H (rm:t-1,k)
        rhsize += t ;
    }

    // -------------------------------------------------------------------------
    // count the rectangular part of R and trapezoidal part of H
    // -------------------------------------------------------------------------

    h = rm ;                            // the column of H starts in row h
    for ( ; k < n ; k++)
    {
        // get the staircase
        t = Stair [k] ;
        // pack R (0:rm-1,k)
        rhsize += rm ;
        h = MIN (h+1, m) ;              // one more row of C to skip over
        // pack H (h:t-1,k)
        PR (("  for RHSIZE, k %ld Stair %ld t %ld\n", k, Stair[k], t)) ;
        rhsize += (t-h) ;
    }

    PR (("  RHSIZE: m %ld n %ld npiv %ld is %ld\n", m, n, npiv, rhsize)) ;
    return (rhsize) ;                   // return # of entries in R+H
}


template int32_t spqrDebug_rhsize    // returns # of entries in R+H
(
    // input, not modified
    int32_t m,                 // # of rows in F
    int32_t n,                 // # of columns in F
    int32_t npiv,              // number of pivotal columns in F
    int32_t *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
) ;

template int64_t spqrDebug_rhsize    // returns # of entries in R+H
(
    // input, not modified
    int64_t m,                 // # of rows in F
    int64_t n,                 // # of columns in F
    int64_t npiv,              // number of pivotal columns in F
    int64_t *Stair,            // size n; column j is dead if Stair [j] == 0.
                            // Only the first npiv columns can be dead.
    cholmod_common *cc
) ;

// =============================================================================
// === spqrDebug_dump_Parent ===================================================
// =============================================================================

template <typename Int> void spqrDebug_dump_Parent (Int n, Int *Parent, const char *filename)
{
    FILE *pfile = fopen (filename, "w") ;
    if (Parent == NULL)
    {
        fprintf (pfile, "0\n") ;
    }
    else
    {
        for (Int f = 0 ; f < n ; f++)
        {
            fprintf (pfile, "%ld\n", 1+Parent [f]) ;
        }
    }
    fclose (pfile) ;
}

template void spqrDebug_dump_Parent <int32_t> (int32_t n, int32_t *Parent, const char *filename) ;

template void spqrDebug_dump_Parent <int64_t> (int64_t n, int64_t *Parent, const char *filename) ;

#endif
