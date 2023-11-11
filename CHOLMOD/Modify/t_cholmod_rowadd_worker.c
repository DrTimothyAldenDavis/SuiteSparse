//------------------------------------------------------------------------------
// CHOLMOD/Modify/t_cholmod_rowadd_worker: add row/col to an LDL' factorization
//------------------------------------------------------------------------------

// CHOLMOD/Modify Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// and William W. Hager. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static int TEMPLATE (cholmod_rowadd_worker)
(
    // input:
    Int k,              // row/column index to add
    cholmod_sparse *R,  // row/column of matrix to factorize (n-by-1)
    Real bk [2],        // kth entry of the right hand side, b
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
    Int *Rj = R->i ;
    Int *Rp = R->p ;
    Int *Rnz = R->nz ;
    Real *Rx = R->x ;
    Int rnz = (R->packed) ? (Rp [1]) : (Rnz [0]) ;
    bool do_solve = (X != NULL) && (DeltaB != NULL) ;
    Real *Xx = NULL ;
    Real *Nx = NULL ;
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
    Int *Lp = L->p ;         // size n+1.  input, not modified on output

    // outputs, contents defined on input for incremental case only:
    Int *Lnz = L->nz ;      // size n
    Int *Li = L->i ;        // size L->nzmax.  Can change in size.
    Real *Lx = L->x ;       // size L->nzmax.  Can change in size.
    Int *Lnext = L->next ;  // size n+2

    ASSERT (L->nz != NULL) ;

    PRINT1 (("rowadd:\n")) ;
    double fl = 0 ;

    //--------------------------------------------------------------------------
    // get workspace
    //--------------------------------------------------------------------------

    Int *Flag = Common->Flag ;      // size n
    Real *W = Common->Xwork ;       // size n
    Real *Cx = W + n ;              // size n (use 2nd column of Xwork for C)
    Int *Stack = Common->Iwork ;    // size n, also in cholmod_updown
    Int *Ci = Stack + n ;           // size n
    // NOTE: cholmod_updown uses Iwork [0..n-1] as Stack as well

    Int mark = Common->mark ;

    // copy Rj/Rx into W/Ci
    for (Int p = 0 ; p < rnz ; p++)
    {
        Int i = Rj [p] ;
        ASSERT (i >= 0 && i < n) ;
        W [i] = Rx [p] ;
        Ci [p] = i ;
    }

    // At this point, W [Ci [0..rnz-1]] holds the sparse vector to add
    // The nonzero pattern of column W is held in Ci (it may be unsorted).

    //--------------------------------------------------------------------------
    // symbolic factorization to get pattern of kth row of L
    //--------------------------------------------------------------------------

    DEBUG (for (Int p = 0 ; p < rnz ; p++)
            PRINT1 (("C ("ID",%g)\n", Ci [p], W [Ci [p]]))) ;
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, 0, Common)) ;

    // flag the diagonal
    Flag [k] = mark ;

    // find the union of all the paths
    Int top = n ;
    Int lnz = 0 ;   // # of nonzeros in column k of L, excluding diagonal
    for (Int p = 0 ; p < rnz ; p++)
    {
        Int i = Ci [p] ;

        if (i < k)
        {

            // walk from i = entry in Ci to root (and stop if i marked)
            PRINT2 (("\nwalk from i = "ID" towards k = "ID"\n", i, k)) ;
            Int len = 0 ;

            // walk up tree, but stop if we go below the diagonal
            while (i < k && i != EMPTY && Flag [i] < mark)
            {
                PRINT2 (("   Add "ID" to path\n", i)) ;
                ASSERT (i >= 0 && i < k) ;
                Stack [len++] = i ;     // place i on the stack
                Flag [i] = mark ;               // mark i as visited
                // parent is the first entry in the column after the diagonal
                ASSERT (Lnz [i] > 0) ;
                Int parent = (Lnz [i] > 1) ? (Li [Lp [i] + 1]) : EMPTY ;
                PRINT2 (("                      parent: "ID"\n", parent)) ;
                i = parent ;    // go up the tree
            }
            ASSERT (len <= top) ;

            // move the path down to the bottom of the stack
            // this shifts Stack [0..len-1] down to [ ... oldtop-1]
            while (len > 0)
            {
                Stack [--top] = Stack [--len] ;
            }
        }
        else if (i > k)
        {
            // prune the diagonal and upper triangular entries from Ci
            Ci [lnz++] = i ;
            Flag [i] = mark ;
        }
    }

    #ifndef NDEBUG
    PRINT1 (("length of S after prune: "ID"\n", lnz)) ;
    for (Int p = 0 ; p < lnz ; p++)
    {
        PRINT1 (("After prune Ci ["ID"] = "ID"\n", p, Ci [p])) ;
        ASSERT (Ci [p] > k) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // ensure each column of L has enough space to grow
    //--------------------------------------------------------------------------

    for (Int kk = top ; kk < n ; kk++)
    {
        // could skip this if we knew column j already included row k
        Int j = Stack [kk] ;
        if (Lp [j] + Lnz [j] >= Lp [Lnext [j]])
        {
            PRINT1 (("Col "ID" realloc, old Lnz "ID"\n", j, Lnz [j])) ;
            if (!CHOLMOD(reallocate_column) (j, Lnz [j] + 1, L, Common))
            {
                // out of memory, L is now simplicial symbolic
                CLEAR_FLAG (Common) ;
                ASSERT (check_flag (Common)) ;
                for (Int i = 0 ; i < n ; i++)
                {
                    W [i] = 0 ;
                }
                return (FALSE) ;
            }
            // L->i and L->x may have moved
            Li = L->i ;
            Lx = L->x ;
        }
        ASSERT (Lp [j] + Lnz [j] < Lp [Lnext [j]]
            || (Lp [Lnext [j]] - Lp [j] == n-j)) ;
    }

    //--------------------------------------------------------------------------
    // compute kth row of L and store in column form
    //--------------------------------------------------------------------------

    // solve L (1:k-1, 1:k-1) * y (1:k-1) = b (1:k-1)
    // where b (1:k) is in W and Ci

    // L (k, 1:k-1) = y (1:k-1) ./ D (1:k-1)
    // D (k) = B (k,k) - L (k, 1:k-1) * y (1:k-1)

    PRINT2 (("\nForward solve: "ID" to "ID"\n", top, n)) ;
    ASSERT (Lnz [k] >= 1 && Li [Lp [k]] == k) ;
    DEBUG (for (Int i = top ; i < n ; i++)
        PRINT2 ((" Path: "ID"\n", Stack [i]))) ;

    Real dk = W [k] ;
    W [k] = 0.0 ;

    // if do_solve: compute x (k) = b (k) - L (k, 1:k-1) * x (1:k-1)
    Real xk = bk [0] ;
    PRINT2 (("B [k] = %g\n", xk)) ;

    for (Int kk = top ; kk < n ; kk++)
    {
        Int j = Stack [kk] ;
        Int i = j ;
        PRINT2 (("Forward solve col j = "ID":\n", j)) ;
        ASSERT (j >= 0 && j < k) ;

        // forward solve using L (j+1:k-1,j)
        Real yj = W [j] ;
        W [j] = 0.0 ;
        Int p = Lp [j] ;
        Int pend = p + Lnz [j] ;
        ASSERT (Lnz [j] > 0) ;
        Real dj = Lx [p++] ;
        for ( ; p < pend ; p++)
        {
            i = Li [p] ;
            PRINT2 (("    row "ID"\n", i)) ;
            ASSERT (i > j) ;
            ASSERT (i < n) ;
            // stop at row k
            if (i >= k)
            {
                break ;
            }
            W [i] -= Lx [p] * yj ;
        }

        // each iteration of the above for loop did 2 flops, and 3 flops
        // are done below.  so: fl += 2 * (Lp [j] - p - 1) + 3 becomes:
        fl += 2 * (Lp [j] - p) + 1 ;

        // scale L (k,1:k-1) and compute dot product for D (k,k)
        Real l_kj = yj / dj ;
        dk -= l_kj * yj ;

        // compute dot product for X(k)
        if (do_solve)
        {
            xk -= l_kj * Xx [j] ;
        }

        // store l_kj in the jth column of L
        // and shift the rest of the column down

        Int li = k ;
        Real lx = l_kj ;

        if (i == k)
        {
            // no need to modify the nonzero pattern of L, since it already
            // contains row index k.
            ASSERT (Li [p] == k) ;
            Lx [p] = l_kj ;

            for (p++ ; p < pend ; p++)
            {
                i = Li [p] ;
                Real l_ij = Lx [p] ;
                ASSERT (i > k && i < n) ;
                PRINT2 (("   apply to row "ID" of column k of L\n", i)) ;

                // add to the pattern of the kth column of L
                if (Flag [i] < mark)
                {
                    PRINT2 (("   add Ci["ID"] = "ID"\n", lnz, i)) ;
                    ASSERT (i > k) ;
                    Ci [lnz++] = i ;
                    Flag [i] = mark ;
                }

                // apply the update to the kth column of L
                // yj is equal to l_kj * d_j
                W [i] -= l_ij * yj ;
            }

        }
        else
        {

            PRINT2 (("Shift col j = "ID", apply saxpy to col k of L\n", j)) ;
            for ( ; p < pend ; p++)
            {
                // swap (Li [p],Lx [p]) with (li,lx)
                i = Li [p] ;
                Real l_ij = Lx [p] ;
                Li [p] = li ;
                Lx [p] = lx ;
                li = i ;
                lx = l_ij ;
                ASSERT (i > k && i < n) ;
                PRINT2 (("   apply to row "ID" of column k of L\n", i)) ;

                // add to the pattern of the kth column of L
                if (Flag [i] < mark)
                {
                    PRINT2 (("   add Ci["ID"] = "ID"\n", lnz, i)) ;
                    ASSERT (i > k) ;
                    Ci [lnz++] = i ;
                    Flag [i] = mark ;
                }

                // apply the update to the kth column of L
                // yj is equal to l_kj * d_j

                W [i] -= l_ij * yj ;
            }

            // store the last value in the jth column of L
            Li [p] = li ;
            Lx [p] = lx ;
            Lnz [j]++ ;
        }
    }

    //--------------------------------------------------------------------------
    // merge C with the pattern of the existing column of L
    //--------------------------------------------------------------------------

    // This column should be zero, but it may contain explicit zero entries.
    // These entries should be kept, not dropped.
    Int p = Lp [k] ;
    Int pend = p + Lnz [k] ;
    for (p++ ; p < pend ; p++)
    {
        Int i = Li [p] ;
        // add to the pattern of the kth column of L
        if (Flag [i] < mark)
        {
            PRINT2 (("   add Ci["ID"] = "ID" from existing col k\n", lnz, i)) ;
            ASSERT (i > k) ;
            Ci [lnz++] = i ;
            Flag [i] = mark ;
        }
    }

    //--------------------------------------------------------------------------
    // update X(k)
    //--------------------------------------------------------------------------

    if (do_solve)
    {
        Xx [k] = xk ;
        PRINT2 (("Xx [k] = %g\n", Xx [k])) ;
    }

    //--------------------------------------------------------------------------
    // ensure abs (dk) >= dbound/sbound, if given
    //--------------------------------------------------------------------------

    #ifdef DOUBLE
    dk = (Common->dbound > 0) ? (CHOLMOD(dbound) (dk, Common)) : dk ;
    #else
    dk = (Common->sbound > 0) ? (CHOLMOD(sbound) (dk, Common)) : dk ;
    #endif

    PRINT2 (("D [k = "ID"] = %g\n", k, dk)) ;

    //--------------------------------------------------------------------------
    // store the kth column of L
    //--------------------------------------------------------------------------

    // ensure the new column of L has enough space
    if (Lp [k] + lnz + 1 > Lp [Lnext [k]])
    {
        PRINT1 (("New Col "ID" realloc, old Lnz "ID"\n", k, Lnz [k])) ;
        if (!CHOLMOD(reallocate_column) (k, lnz + 1, L, Common))
        {
            // out of memory, L is now simplicial symbolic
            CHOLMOD(clear_flag) (Common) ;
            for (Int i = 0 ; i < n ; i++)
            {
                W [i] = 0 ;
            }
            return (FALSE) ;
        }
        // L->i and L->x may have moved
        Li = L->i ;
        Lx = L->x ;
    }
    ASSERT (Lp [k] + lnz + 1 <= Lp [Lnext [k]]) ;

    #ifndef NDEBUG
    PRINT2 (("\nPrior to sort: lnz "ID" (excluding diagonal)\n", lnz)) ;
    for (Int kk = 0 ; kk < lnz ; kk++)
    {
        Int i = Ci [kk] ;
        PRINT2 (("L ["ID"] kept: "ID" %e\n", kk, i, W [i] / dk)) ;
    }
    #endif

    // sort Ci
    qsort (Ci, lnz, sizeof (Int), (int (*) (const void *, const void *)) icomp);

    // store the kth column of L
    DEBUG (Int lastrow = k) ;
    p = Lp [k] ;
    Lx [p++] = dk ;
    Lnz [k] = lnz + 1 ;
    fl += lnz ;
    for (Int kk = 0 ; kk < lnz ; kk++, p++)
    {
        Int i = Ci [kk] ;
        PRINT2 (("L ["ID"] after sort: "ID", %e\n", kk, i, W [i] / dk)) ;
        ASSERT (i > lastrow) ;
        Li [p] = i ;
        Lx [p] = W [i] / dk ;
        W [i] = 0.0 ;
        DEBUG (lastrow = i) ;
    }

    // compute DeltaB for updown (in DeltaB)
    if (do_solve)
    {
        Int p = Lp [k] ;
        Int pend = p + Lnz [k] ;
        for (p++ ; p < pend ; p++)
        {
            ASSERT (Li [p] > k) ;
            Nx [Li [p]] -= Lx [p] * xk ;
        }
    }

    // clear the flag for the update
    mark = CHOLMOD(clear_flag) (Common) ;

    // workspaces are now cleared
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 2*n, R->dtype, Common)) ;

    //--------------------------------------------------------------------------
    // update/downdate
    //--------------------------------------------------------------------------

    // update or downdate L (k+1:n, k+1:n) with the vector
    // C = L (:,k) * sqrt (abs (D [k])).
    // Do a numeric update if D[k] < 0, numeric downdate otherwise.

    int ok = TRUE ;
    Common->modfl = 0 ;

    PRINT1 (("rowadd update lnz = "ID"\n", lnz)) ;
    if (lnz > 0)
    {
        bool do_update = (dk < 0) ;
        if (do_update)
        {
            dk = -dk ;
        }
        Real sqrt_dk = sqrt (dk) ;
        Int p = Lp [k] + 1 ;
        for (Int kk = 0 ; kk < lnz ; kk++, p++)
        {
            Cx [kk] = Lx [p] * sqrt_dk ;
        }
        fl += lnz + 1 ;

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

        // numeric downdate if dk > 0, and optional Lx=b change
        // workspace: Flag (nrow), Head (nrow+1), W (nrow), Iwork (2*nrow)
        ok = CHOLMOD(updown_mark) (do_update ? (1) : (0), C, colmark,
                L, X, DeltaB, Common) ;

        // clear workspace
        for (Int kk = 0 ; kk < lnz ; kk++)
        {
            Cx [kk] = 0 ;
        }
    }

    Common->modfl += fl ;
    return (ok) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

