//------------------------------------------------------------------------------
// CHOLMOD/Modify/t_cholmod_rowdel_worker: delete row/col from LDL'
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static int TEMPLATE (cholmod_rowdel_worker)
(
    // input:
    Int k,              // row/column index to delete
    cholmod_sparse *R,  // NULL, or the nonzero pattern of kth row of L
    Real yk [2],        // kth entry in the solution to A*y=b
    Int *colmark,       // Int array of size 1.  See cholmod_updown.c
    // input/output:
    cholmod_factor *L,  // factor to modify
    cholmod_dense *X,   // solution to Lx=b (size n-by-1)
    cholmod_dense *DeltaB,  // change in b, zero on output
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int n = L->n ;
    Real dk, xk, dj ;
    Real *Xx, *Nx ;
    Int *Rj, *Rp ;
    Int j, p, pend, kk, lnz, left, right, middle, i, klast, given_row, rnz ;

    if (R == NULL)
    {
        Rj = NULL ;
        rnz = EMPTY ;
    }
    else
    {
        Rj = R->i ;
        Rp = R->p ;
        rnz = Rp [1] ;
    }

    bool do_solve = (X != NULL) && (DeltaB != NULL) ;
    if (do_solve)
    {
        Xx = X->x ;
        Nx = DeltaB->x ;
    }
    else
    {
        Xx = NULL ;
        Nx = NULL ;
    }

    // inputs, not modified on output:
    Int *Lp = L->p ;         // size n+1

    // outputs, contents defined on input for incremental case only:
    Int *Lnz = L->nz ;       // size n
    Int *Li = L->i ;         // size L->nzmax.  Can change in size.
    Real *Lx = L->x ;         // size L->nzmax.  Can change in size.

    ASSERT (L->nz != NULL) ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    Real *W = Common->Xwork ;   // size n, used only in cholmod_updown
    Real *Cx = W + n ;          // use 2nd column of Xwork for C (size n)
    Int *Iwork = Common->Iwork ;
    Int *Ci = Iwork + n ;       // size n
    // NOTE: cholmod_updown uses Iwork [0..n-1] as Stack

    //--------------------------------------------------------------------------
    // prune row k from all columns of L
    //--------------------------------------------------------------------------

    given_row = (rnz >= 0) ;
    klast = given_row ? rnz : k ;
    PRINT2 (("given_row "ID"\n", given_row)) ;

    for (kk = 0 ; kk < klast ; kk++)
    {
        // either search j = 0:k-1 or j = Rj [0:rnz-1]
        j = given_row ? (Rj [kk]) : (kk) ;

        if (j < 0 || j >= k)
        {
            ERROR (CHOLMOD_INVALID, "R invalid") ;
            return (FALSE) ;
        }

        PRINT2 (("Prune col j = "ID":\n", j)) ;

        lnz = Lnz [j] ;
        dj = Lx [Lp [j]] ;
        ASSERT (Lnz [j] > 0 && Li [Lp [j]] == j) ;

        if (lnz > 1)
        {
            left = Lp [j] ;
            pend = left + lnz ;
            right = pend - 1 ;

            i = Li [right] ;

            if (i < k)
            {
                // row k is not in column j
                continue ;
            }
            else if (i == k)
            {
                // k is the last row index in this column (quick delete)
                if (do_solve)
                {
                    Xx [j] -= yk [0] * dj * Lx [right] ;
                }
                Lx [right] = 0 ;
            }
            else
            {
                // binary search for row k in column j
                PRINT2 (("\nBinary search: lnz "ID" k = "ID"\n", lnz, k)) ;
                while (left < right)
                {
                    middle = (left + right) / 2 ;
                    PRINT2 (("left "ID" right "ID" middle "ID": ["ID" "ID""
                        ""ID"]\n", left, right, middle,
                        Li [left], Li [middle], Li [right])) ;
                    if (k > Li [middle])
                    {
                        left = middle + 1 ;
                    }
                    else
                    {
                        right = middle ;
                    }
                }
                ASSERT (left >= Lp [j] && left < pend) ;

                #ifndef NDEBUG
                // brute force, linear-time search
                {
                    Int p3 = Lp [j] ;
                    i = EMPTY ;
                    PRINT2 (("Brute force:\n")) ;
                    for ( ; p3 < pend ; p3++)
                    {
                        i = Li [p3] ;
                        PRINT2 (("p "ID" ["ID"]\n", p3, i)) ;
                        if (i >= k)
                        {
                            break ;
                        }
                    }
                    if (i == k)
                    {
                        ASSERT (k == Li [p3]) ;
                        ASSERT (p3 == left) ;
                    }
                }
                #endif

                if (k == Li [left])
                {
                    if (do_solve)
                    {
                        Xx [j] -= yk [0] * dj * Lx [left] ;
                    }
                    // found row k in column j.  Prune it from the column.
                    Lx [left] = 0 ;
                }
            }
        }
    }

    #ifndef NDEBUG
    // ensure that row k has been deleted from the matrix L
    for (j = 0 ; j < k ; j++)
    {
        Int lasti ;
        lasti = EMPTY ;
        p = Lp [j] ;
        pend = p + Lnz [j] ;
        // look for row k in column j
        PRINT1 (("Pruned column "ID"\n", j)) ;
        for ( ; p < pend ; p++)
        {
            i = Li [p] ;
            PRINT2 ((" "ID"", i)) ;
            PRINT2 ((" %g\n", Lx [p])) ;
            ASSERT (IMPLIES (i == k, Lx [p] == 0)) ;
            ASSERT (i > lasti) ;
            lasti = i ;
        }
        PRINT1 (("\n")) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // set diagonal and clear column k of L
    //--------------------------------------------------------------------------

    lnz = Lnz [k] - 1 ;
    ASSERT (Lnz [k] > 0) ;

    //--------------------------------------------------------------------------
    // update/downdate
    //--------------------------------------------------------------------------

    // update or downdate L (k+1:n, k+1:n) with the vector
    // C = L (:,k) * sqrt (abs (D [k]))
    // Do a numeric update if D[k] > 0, numeric downdate otherwise.

    PRINT1 (("rowdel downdate lnz = "ID"\n", lnz)) ;

    // store the new unit diagonal
    p = Lp [k] ;
    pend = p + lnz + 1 ;
    dk = Lx [p] ;
    Lx [p++] = 1 ;
    PRINT2 (("D [k = "ID"] = %g\n", k, dk)) ;
    int ok = TRUE ;
    double fl = 0 ;

    if (lnz > 0)
    {
        // compute DeltaB for updown (in DeltaB)
        if (do_solve)
        {
            xk = Xx [k] - yk [0] * dk ;
            for ( ; p < pend ; p++)
            {
                Nx [Li [p]] += Lx [p] * xk ;
            }
        }

        bool do_update = (dk > 0) ;
        if (!do_update)
        {
            dk = -dk ;
        }
        Real sqrt_dk = sqrt (dk) ;
        p = Lp [k] + 1 ;
        for (kk = 0 ; kk < lnz ; kk++, p++)
        {
            Ci [kk] = Li [p] ;
            Cx [kk] = Lx [p] * sqrt_dk ;
            Lx [p] = 0 ;                // clear column k
        }
        fl = lnz + 1 ;

        // create a n-by-1 sparse matrix to hold the single column
        cholmod_sparse *C, Cmatrix ;
        Int Cp [2] ;
        C = &Cmatrix ;
        C->nrow = n ;
        C->ncol = 1 ;
        C->nzmax = lnz ;
        C->sorted = TRUE ;
        C->packed = TRUE ;
        C->p = Cp ;
        C->i = Ci ;
        C->x = Cx ;
        C->nz = NULL ;
        C->itype = L->itype ;
        C->xtype = L->xtype ;
        C->dtype = L->dtype ;
        C->z = NULL ;
        C->stype = 0 ;

        Cp [0] = 0 ;
        Cp [1] = lnz ;

        // numeric update if dk > 0, and with Lx=b change
        // workspace: Flag (nrow), Head (nrow+1), W (nrow), Iwork (2*nrow)
        ok = CHOLMOD(updown_mark) (do_update ? (1) : (0), C, colmark,
                L, X, DeltaB, Common) ;

        // clear workspace
        for (kk = 0 ; kk < lnz ; kk++)
        {
            Cx [kk] = 0 ;
        }
    }

    Common->modfl += fl ;

    if (do_solve)
    {
        // kth equation becomes identity, so X(k) is now Y(k)
        Xx [k] = yk [0] ;
    }

    return (ok) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

