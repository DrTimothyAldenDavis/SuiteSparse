// =============================================================================
// === spqr_rcount =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Count the number of explicit nonzeros in each column of R.  Exact zero
// entries are excluded.

#include "spqr.hpp"

template <typename Entry, typename Int> void spqr_rcount
(
    // inputs, not modified
    spqr_symbolic <Int> *QRsym,
    spqr_numeric <Entry, Int> *QRnum,

    Int n1rows,        // added to each row index of Ra and Rb
    Int econ,          // only get entries in rows n1rows to econ-1
    Int n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, count Rb' instead of Rb

    // input/output
    // FUTURE : make Ra, Rb, H2 cholmod_sparse
    Int *Ra,           // size n2; Ra [j] += nnz (R (:,j)) if j < n2
    Int *Rb,           // If getT is false: size n-n2 and
                        // Rb [j-n2] += nnz (R (:,j)) if j >= n2.
                        // If getT is true: size econ, and
                        // Rb [i] += nnz (R (i, n2:n-1))
    Int *H2p,          // size rjsize+1.  Column pointers for H.
                        // Only computed if H was kept during factorization.
                        // Only H2p [0..nh] is used.
    Int *p_nh          // number of Householder vectors (nh <= rjsize)
)
{
    Entry **Rblock, *R, *Tau, *HTau ;
    Int *Rp, *Rj, *Super, *HStair, *Stair, *Hm ;
    char *Rdead ;
    Int nf, j, f, col1, fp, pr, fn, rm, k, i, t, fm, h, getRa, getRb, nh,
        row1, keepH, getH, hnz ;

    // -------------------------------------------------------------------------
    // get the contents of the QRsym and QRnum objects
    // -------------------------------------------------------------------------

    keepH = QRnum->keepH ; 

    getRa = (Ra != NULL) ;
    getRb = (Rb != NULL) ;
    getH  = (H2p != NULL && p_nh != NULL) && keepH ;
    if (!(getRa || getRb || getH))
    {
        // nothing to do
        return ;
    }

    nf = QRsym->nf ;
    // n = QRsym->n ;
    Rblock = QRnum->Rblock ;
    Rp = QRsym->Rp ;
    Rj = QRsym->Rj ;
    Super = QRsym->Super ;
    Rdead = QRnum->Rdead ;

    HStair = QRnum->HStair ;
    HTau = QRnum->HTau ;
    Hm = QRnum->Hm ;
    Stair = NULL ;
    Tau = NULL ;
    fm = 0 ;
    h = 0 ;
    t = 0 ;
    nh = 0 ;
    hnz = 0 ;

    // -------------------------------------------------------------------------
    // examine the packed block for each front F
    // -------------------------------------------------------------------------

    row1 = n1rows ;
    for (f = 0 ; f < nf ; f ++)
    {
        R = Rblock [f] ;
        col1 = Super [f] ;                  // first pivot column in front F
        fp = Super [f+1] - col1 ;           // number of pivots in front F
        pr = Rp [f] ;                       // pointer to row indices for F
        fn = Rp [f+1] - pr ;                // # of columns in front F

        if (keepH)
        {
            Stair = HStair + pr ;           // staircase of front F
            Tau = HTau + pr ;               // Householder coeff. for front F
            fm = Hm [f] ;                   // # of rows in front F
            h = 0 ;                         // H vbector starts in row h
        }

        rm = 0 ;                            // number of rows in R block
        for (k = 0 ; k < fn ; k++)
        {
            // -----------------------------------------------------------------
            // get the column and its staircase
            // -----------------------------------------------------------------

            if (k < fp)
            {
                // a pivotal column of front F
                j = col1 + k ;
                ASSERT (Rj [pr + k] == j) ;
                if (keepH)
                {
                    t = Stair [k] ;             // length of R+H vector
                    ASSERT (t >= 0 && t <= fm) ;
                    if (t == 0)
                    {
                        t = rm ;                // dead col, R only, no H
                    }
                    else if (rm < fm)
                    {
                        rm++ ;                  // column k is not dead
                    }
                    h = rm ;                    // H vector starts in row h
                }
                else
                {
                    if (!Rdead [j])
                    {
                        rm++ ;                  // column k is not dead
                    }
                }
            }
            else
            {
                // a non-pivotal column of front F
                j = Rj [pr + k] ;
                ASSERT (j >= Super [f+1] && j < QRsym->n) ;
                if (keepH)
                {
                    t = Stair [k] ;             // length of R+H vector
                    ASSERT (t >= rm && t <= fm) ;
                    h = MIN (h+1, fm) ;         // one more row of C to skip
                }
            }

            // -----------------------------------------------------------------
            // count nnz (R (0:econ-1,j)) for this R block
            // -----------------------------------------------------------------

            for (i = 0 ; i < rm ; i++)
            {
                // R (i,j) is nonzero (i local)
                Entry rij = *(R++) ;
                if (rij != (Entry) 0)
                {
                    if (j < n2)
                    {
                        if (getRa && row1 + i < econ)
                        {
                            Ra [j]++ ;
                        }
                    }
                    else
                    {
                        if (getRb && row1 + i < econ)
                        {
                            if (getT)
                            {
                                Rb [row1+i]++ ;
                            }
                            else
                            {
                                Rb [j-n2]++ ;
                            }
                        }
                    }
                }
            }

            // -----------------------------------------------------------------
            // count nnz (H (:,pr+k))
            // -----------------------------------------------------------------

            if (keepH && t >= h)
            {
                // the Householder reflection is not empty
                PR (("count terms in H k %ld, t %ld h %ld\n", k, t, h)) ;
                if (getH && Tau [k] != (Entry) 0)
                {
                    H2p [nh++] = hnz++ ;    // count the implicit identity
                    for (i = h ; i < t ; i++)
                    {
                        Entry hij = *(R++) ;
                        if (hij != (Entry) 0)
                        {
                            hnz++ ;         // H (i,pr+k) is nonzero
                        }
                    }
                }
                else
                {
                    R += (t-h) ;            // skip over the column of H
                }
            }
        }
        ASSERT (IMPLIES (keepH, QRnum->Hr [f] == rm)) ;
        row1 += rm ;                        // count the squeezed rows of R
    }

    // -------------------------------------------------------------------------
    // finalize the H column pointers
    // -------------------------------------------------------------------------

    if (getH)
    {
        H2p [nh] = hnz ;
        *p_nh = nh ;
    }
}


// explicit instantiations

template void spqr_rcount <double, int32_t>
(
    spqr_symbolic <int32_t> *QRsym, spqr_numeric <double, int32_t> *QRnum,
    int32_t n1rows, int32_t econ, int32_t n2, int getT,
    int32_t *Ra, int32_t *Rb, int32_t *H2p, int32_t *p_nh
) ;

template void spqr_rcount <Complex, int32_t>
(
    spqr_symbolic <int32_t> *QRsym, spqr_numeric <Complex, int32_t> *QRnum,
    int32_t n1rows, int32_t econ, int32_t n2, int getT, 
    int32_t *Ra, int32_t *Rb, int32_t *H2p, int32_t *p_nh
) ;

template void spqr_rcount <double, int64_t>
(
    spqr_symbolic <int64_t> *QRsym, spqr_numeric <double, int64_t> *QRnum,
    int64_t n1rows, int64_t econ, int64_t n2, int getT,
    int64_t *Ra, int64_t *Rb, int64_t *H2p, int64_t *p_nh
) ;

template void spqr_rcount <Complex, int64_t>
(
    spqr_symbolic <int64_t> *QRsym, spqr_numeric <Complex, int64_t> *QRnum,
    int64_t n1rows, int64_t econ, int64_t n2, int getT,
    int64_t *Ra, int64_t *Rb, int64_t *H2p, int64_t *p_nh
) ;
