//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_triplet_to_sparse_worker
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_template.h"

static size_t TEMPLATE (cholmod_triplet_to_sparse_worker)   // return nnz(R)
(
    cholmod_triplet *T,     // input matrix
    cholmod_sparse *R,      // output matrix
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int  *Rp  = (Int  *) R->p ;
    Int  *Ri  = (Int  *) R->i ;
    Int  *Rnz = (Int  *) R->nz ;
    Real *Rx  = (Real *) R->x ;
    Real *Rz  = (Real *) R->z ;

    Int  *Ti = (Int *) T->i ;
    Int  *Tj = (Int *) T->j ;
    Real *Tx = (Real *) T->x ;
    Real *Tz = (Real *) T->z ;
    size_t nrow = T->nrow ;
    size_t ncol = T->ncol ;
    Int nz = T->nnz ;

    //--------------------------------------------------------------------------
    // W [0..nrow-1] = Rp [0..nrow-1] using Iwork workspace
    //--------------------------------------------------------------------------

    // using W [0..nrow-1] as workspace for row pointers [
    Int *W = (Int *) Common->Iwork ;
    memcpy (W, Rp, nrow * sizeof (Int)) ;

    //--------------------------------------------------------------------------
    // construct the matrix R, keeping duplicates for now
    //--------------------------------------------------------------------------

    // R is treated as if it is stored by row, in the comments below

    int stype = T->stype ;

    for (Int k = 0 ; k < nz ; k++)
    {
        // get the T (i,j) entry
        Int i = Ti [k] ;
        Int j = Tj [k] ;
        Int p ;
        if (stype > 0)
        {
            // T represents a symmetric matrix with upper part stored
            if (i < j)
            {
                // R (i,j) = T (i,j), placed in row R (i,:)
                Ri [p = W [i]++] = j ;
            }
            else
            {
                // R (j,i) = T (i,j), placed in row R (j,:)
                Ri [p = W [j]++] = i ;
            }
        }
        else if (stype < 0)
        {
            // T represents a symmetric matrix with lower part stored
            if (i > j)
            {
                // R (i,j) = T (i,j), placed in row R (i,:)
                Ri [p = W [i]++] = j ;
            }
            else
            {
                // R (j,i) = T (i,j), placed in row R (j,:)
                Ri [p = W [j]++] = i ;
            }
        }
        else
        {
            // T represents an unsymmetric matrix
            // R (i,j) = T (i,j), placed in row R (i,:)
            Ri [p = W [i]++] = j ;
        }
        ASSIGN (Rx, Rz, p, Tx, Tz, k) ;     // Rx [p] = Tx [k]
    }

    // no longer using W as temporary workspace for row pointers ]

    //--------------------------------------------------------------------------
    // assemble any duplicate entries
    //--------------------------------------------------------------------------

    // use W [0..ncol-1] for pointers to duplicates in each row of R [
    CHOLMOD(set_empty) (W, ncol) ;

    size_t rnz = 0 ;    // total # of entries in R after assembling duplicates

    for (Int i = 0 ; i < nrow ; i++)
    {

        //----------------------------------------------------------------------
        // get the location of R (i,:) before assemblying duplicates
        //----------------------------------------------------------------------

        // row R (i,:) is in located in Ri [pstart..pend-1].  If duplicates are
        // detected, the new row i will be located in Ri [pstart..pp-1].

        Int pstart = Rp [i] ;
        Int pend = Rp [i+1] ;
        Int pp = pstart ;

        // W [j] is the position in Ri of the last time column j was seen.
        // Here, W [0..ncol-1] < pstart is true because R is stored by row,
        // and any column j already seen will have been seen in an earlier
        // row.  If column j has never been seen, W [j] is EMPTY (-1).

        //----------------------------------------------------------------------
        // assemble duplicates in R (i,:)
        //----------------------------------------------------------------------

        for (Int p = pstart ; p < pend ; p++)
        {

            //------------------------------------------------------------------
            // get R(i,j)
            //------------------------------------------------------------------

            Int j = Ri [p] ;
            Int plastj = W [j] ;   // last seen position of column index j

            //------------------------------------------------------------------
            // assemble R(i,j)
            //------------------------------------------------------------------

            if (plastj < pstart)
            {
                // column j has been seen for the first time in row R (i,:),
                // at position pp.  Move the entry to position pp, and keep
                // track of it in case column j appears again in row R (i,:).
                // Rx [pp] = Rx [p]
                ASSIGN (Rx, Rz, pp, Rx, Rz, p) ;
                Ri [pp] = j ;
                // one more unique entry has been seen in R (i,:)
                W [j] = pp++ ;
            }
            else
            {
                // column j has already been seen in this row R (i,;), at
                // position plastj, so assemble this duplicate entry into that
                // position.
                // Rx [plastj] += Rx [p]
                ASSEMBLE (Rx, Rz, plastj, Rx, Rz, p) ;
            }
        }

        //----------------------------------------------------------------------
        // count the number of entries in R (i,:)
        //----------------------------------------------------------------------

        Int rnz_i = pp - pstart ;
        Rnz [i] = rnz_i ;
        rnz += rnz_i ;
    }

    // done using W [0..ncol-1] workspace ]

    //--------------------------------------------------------------------------
    // return result: # of entries in R after assembling duplicates
    //--------------------------------------------------------------------------

    return (rnz) ;
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

