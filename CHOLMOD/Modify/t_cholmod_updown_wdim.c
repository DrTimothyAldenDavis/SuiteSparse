//------------------------------------------------------------------------------
// CHOLMOD/Modify/t_cholmod_updown_wdim: template for t_cholmod_updown_worker
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Updates/downdates the LDL' factorization, by computing a new factorization of
//
//      Lnew * Dnew * Lnew' = Lold * Dold * Lold' +/- C*C'
//
// This file is not compiled separately.  It is #included twice into
// t_cholmod_updown_worker.c, to create all the kernels for double and single
// precision update/downdate.  There are no user-callable routines in this
// file.
//
// The next #include statements, below, create the numerical update/downdate
// kernels from t_cholmod_updown_numkr.c.  For the double case, there are 4
// compiled versions of this file, one for each value of WDIM in the set 1, 2,
// 4, and 8.  Each calls multiple versions of t_cholmod_updown_numkr; the
// number of versions of each is equal to WDIM.  Each t_cholmod_updown_numkr
// version is #included as a static function within its t_cholmod_updown_wdim.c
// caller routine.  Thus, for the double case:
//
// t*_updown_wdim.c  creates these versions of t_cholmod_updown_numkr.c:
// ----------------  ---------------------------------------------------
//
// d_updown_1_r      d_updown_1_1
//
// d_updown_2_r      d_updown_2_1   d_updown_2_2
//
// d_updown_4_r      d_updown_4_1   d_updown_4_2   d_updown_4_3   d_updown_4_4
//
// d_updown_8_r      d_updown_8_1   d_updown_8_2   d_updown_8_3   d_updown_8_4
//                   d_updown_8_5   d_updown_8_6   d_updown_8_7   d_updown_8_8
//
// For the single (float) case, all the functions are named s_updown_*_*.
//
// workspace: Xwork (nrow*wdim)

//------------------------------------------------------------------------------
// routines for numeric update/downdate along one path
//------------------------------------------------------------------------------

#define RANK 1
#include "t_cholmod_updown_numkr.c"

#if WDIM >= 2
#define RANK 2
#include "t_cholmod_updown_numkr.c"
#endif

#if WDIM >= 4
#define RANK 3
#include "t_cholmod_updown_numkr.c"
#define RANK 4
#include "t_cholmod_updown_numkr.c"
#endif

#if WDIM == 8
#define RANK 5
#include "t_cholmod_updown_numkr.c"
#define RANK 6
#include "t_cholmod_updown_numkr.c"
#define RANK 7
#include "t_cholmod_updown_numkr.c"
#define RANK 8
#include "t_cholmod_updown_numkr.c"
#endif

//------------------------------------------------------------------------------
// numeric update/downdate for all paths
//------------------------------------------------------------------------------

static void UPDOWN (WDIM, r)   // WDIM is a compile-time constant (1,2,4, or 8)
(
    int update,         // TRUE for update, FALSE for downdate
    cholmod_sparse *C,  // in packed or unpacked, and sorted form
                        // no empty columns
    Int rank,           // rank of the update/downdate
    cholmod_factor *L,  // with unit diagonal (diagonal not stored)
                        // temporary workspaces:
    Real W [ ],         // n-by-WDIM dense matrix, initially zero
    Path_type Path [ ],
    Int npaths,
    Int mask [ ],       // size n
    Int maskmark,
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real Alpha [8] ;
    Real *Cx, *Wpath, *W1, *a ;
    Int i, j, p, ccol, pend, wfirst, e, path, packed ;
    Int *Ci, *Cp, *Cnz ;

    Ci = C->i ;
    Cx = C->x ;
    Cp = C->p ;
    Cnz = C->nz ;
    packed = C->packed ;
    ASSERT (IMPLIES (!packed, Cnz != NULL)) ;
    ASSERT (L->n == C->nrow) ;
    DEBUG (CHOLMOD(dump_real) ("num_d: in W:", W, L->dtype,
        WDIM, L->n, FALSE, 1, Common)) ;

    //--------------------------------------------------------------------------
    // scatter C into W
    //--------------------------------------------------------------------------

    for (path = 0 ; path < rank ; path++)
    {
        // W (:, path) = C (:, Path [path].col)
        ccol = Path [path].ccol ;
        Wpath = W + path ;
        PRINT1 (("Ordered Columns [path = "ID"] = "ID"\n", path, ccol)) ;
        p = Cp [ccol] ;
        pend = (packed) ? (Cp [ccol+1]) : (p + Cnz [ccol]) ;
        // column C can be empty
        for ( ; p < pend ; p++)
        {
            i = Ci [p] ;
            ASSERT (i >= 0 && i < (Int) (C->nrow)) ;
            if (mask == NULL || mask [i] < maskmark)
            {
                Wpath [WDIM * i] = Cx [p] ;
            }
            PRINT1 (("    row "ID" : %g mask "ID"\n", i, Cx [p],
                    (mask) ? mask [i] : 0)) ;
        }
        Alpha [path] = 1.0 ;
    }
    DEBUG (CHOLMOD(dump_real) ("num_d: W:", W, L->dtype,
        WDIM, L->n, FALSE, 1,Common)) ;

    //--------------------------------------------------------------------------
    // numeric update/downdate of the paths
    //--------------------------------------------------------------------------

    // for each disjoint subpath in Tbar in DFS order do
    for (path = rank ; path < npaths ; path++)
    {

        // determine which columns of W to use
        wfirst = Path [path].wfirst ;
        e = Path [path].end ;
        j = Path [path].start ;
        ASSERT (e >= 0 && e < (Int) (L->n)) ;
        ASSERT (j >= 0 && j < (Int) (L->n)) ;

        W1 = W + wfirst ;       // pointer to row 0, column wfirst of W
        a = Alpha + wfirst ;    // pointer to Alpha [wfirst]

        PRINT1 (("Numerical update/downdate of path "ID"\n", path)) ;
        PRINT1 (("start "ID" end "ID" wfirst "ID" rank "ID" ccol "ID"\n", j, e,
                wfirst, Path [path].rank, Path [path].ccol)) ;

        #if WDIM == 1

        UPDOWN (WDIM,1) (update, j, e, a, W1, L, Common) ;

        #else

        switch (Path [path].rank)
        {
            case 1:
                UPDOWN (WDIM,1) (update, j, e, a, W1, L, Common) ;
                break ;

            #if WDIM >= 2
            case 2:
                UPDOWN (WDIM,2) (update, j, e, a, W1, L, Common) ;
                break ;
            #endif

            #if WDIM >= 4
            case 3:
                UPDOWN (WDIM,3) (update, j, e, a, W1, L, Common) ;
                break ;
            case 4:
                UPDOWN (WDIM,4) (update, j, e, a, W1, L, Common) ;
                break ;
            #endif

            #if WDIM == 8
            case 5:
                UPDOWN (WDIM,5) (update, j, e, a, W1, L, Common) ;
                break ;
            case 6:
                UPDOWN (WDIM,6) (update, j, e, a, W1, L, Common) ;
                break ;
            case 7:
                UPDOWN (WDIM,7) (update, j, e, a, W1, L, Common) ;
                break ;
            case 8:
                UPDOWN (WDIM,8) (update, j, e, a, W1, L, Common) ;
                break ;
            #endif
        }
        #endif
    }
}

// prepare for the next inclusion of this file in t_cholmod_updown_worker.c
#undef WDIM

