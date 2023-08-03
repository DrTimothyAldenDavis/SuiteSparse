// =============================================================================
// === spqr_rconvert ===========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Converts the packed supernodal form of R into two MATLAB-style
// compressed-column form matrices, Ra and Rb.  Ra is the first n2 columns
// of R, and Rb is the last n-n2 columns of R.  The matrix Ra is not created
// if any of its arrays (Rap, Rai, Rax) are NULL; likewise for Rb.

#include "spqr.hpp"

template <typename Entry, typename Int> void spqr_rconvert
(
    // inputs, not modified
    spqr_symbolic <Int> *QRsym,
    spqr_numeric <Entry, Int> *QRnum,

    Int n1rows,        // added to each row index of Ra, Rb, and H
    Int econ,          // only get entries in rows n1rows to econ-1
    Int n2,            // Ra = R (:,0:n2-1), Rb = R (:,n2:n-1)
    int getT,           // if true, get Rb' instead of Rb

    // input/output
    // FUTURE : make Ra, Rb, H2 cholmod_sparse:
    Int *Rap,          // size n2+1; on input, Rap [j] is the column pointer
                        // for Ra.  Incremented on output by the number of
                        // entries added to column j of Ra.

    // output, not defined on input
    Int *Rai,          // size rnz1 = nnz(Ra); row indices of Ra
    Entry *Rax,         // size rnz; numerical values of Ra

    // input/output
    Int *Rbp,          // if getT is false:
                        // size (n-n2)+1; on input, Rbp [j] is the column
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to column j of Rb.
                        // if getT is true:
                        // size econ+1; on input, Rbp [i] is the row
                        // pointer for Rb.  Incremented on output by the number
                        // of entries added to row i of Rb.

    // output, not defined on input
    Int *Rbi,          // size rnz2 = nnz(Rb); indices of Rb
    Entry *Rbx,         // size rnz2; numerical values of Rb

    // input
    Int *H2p,          // size nh+1; H2p [j] is the column pointer for H.
                        // H2p, H2i, and H2x are ignored if H was not kept
                        // during factorization.  nh computed by rcount

    // output, not defined on input
    Int *H2i,          // size hnz = nnz(H); indices of H
    Entry *H2x,         // size hnz; numerical values of H

    Entry *H2Tau        // size nh; Householder coefficients
)
{
    Entry rij, hij ;
    Entry **Rblock, *R, *Tau, *HTau ;
    Int *Rp, *Rj, *Super, *HStair, *Hii, *Stair, *Hip, *Hm, *Hi ;
    char *Rdead ;
    Int nf, j, f, col1, fp, pr, fn, rm, k, i, p, getRa, getRb, row1, fm,
        // n, rjsize,
        h, getH, keepH, ph, t, nh ;

    // -------------------------------------------------------------------------
    // get the contents of the QRsym and QRnum objects
    // -------------------------------------------------------------------------

    keepH = QRnum->keepH ;
    getRa = (Rap != NULL && Rai != NULL && Rax != NULL) ;
    getRb = (Rbp != NULL && Rbi != NULL && Rbx != NULL) ;
    getH  = (H2p != NULL && H2i != NULL && H2x != NULL && H2Tau != NULL)
            && keepH ;
    if (!(getRa || getRb || getH))
    {
        // nothing to do
        return ;
    }

#ifndef NDEBUG
    if (getRa)
    {
        for (k = 0 ; k <= n2 ; k++)
        {
            PR (("Rap [%ld] = %ld on input\n", k, Rap [k])) ;
        }
    }
    if (getRb)
    {
        Int n = QRsym->n ;
        Int zn = getT ? econ : n-n2 ;
        for (k = 0 ; k <= zn ; k++)
        {
            PR (("Rbp [%ld] = %ld on input\n", k, Rbp [k])) ;
        }
    }
#endif

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
    Hii = QRnum->Hii ;
    Hip = QRsym->Hip ;
    // rjsize = QRsym->rjsize ;
    Stair = NULL ;
    Hi = NULL ;
    Tau = NULL ;
    fm = 0 ;
    h = 0 ;
    t = 0 ;
    nh = 0 ;

    // -------------------------------------------------------------------------
    // convert the packed block for each front F
    // -------------------------------------------------------------------------

    row1 = n1rows ;
    ph = 0 ;                                // pointer for constructing H
    PR (("rconvert nf : %ld xtype %d\n", nf, spqr_type <Entry> ( ))) ;
    for (f = 0 ; f < nf ; f ++)
    {
        PR (("\n---- rconvert front f %ld\n", f)) ;
        R = Rblock [f] ;
        col1 = Super [f] ;                  // first pivot column in front F
        fp = Super [f+1] - col1 ;           // number of pivots in front F
        pr = Rp [f] ;                       // pointer to row indices for F
        fn = Rp [f+1] - pr ;                // # of columns in front F

        if (keepH)
        {
            Stair = HStair + pr ;           // staircase of front F
            Tau = HTau + pr ;               // Householder coeff. for front F
            Hi = &Hii [Hip [f]] ;           // list of row indices of H
            fm = Hm [f] ;                   // # of rows in front F
            PR (("f %ld fm %ld Hip [f] %ld Hip [f+1] %ld\n",
                f, fm, Hip [f], Hip [f+1])) ;
            ASSERT (fm <= Hip [f+1]-Hip[f]) ;
            h = 0 ;                         // H vector starts in row h
        }

        // ---------------------------------------------------------------------
        // extract each column of the R or R+H block
        // ---------------------------------------------------------------------

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
                PR (("\nk %ld pivotal column of front f %ld, global j %ld\n",
                    k, f, j)) ;
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
                    ASSERT (t >= h) ;
                }
                else
                {
                    PR (("  k %ld j %ld Rdead[j] %d old rm %ld ",
                        k, j, (int) (Rdead [j]), rm)) ;
                    if (!Rdead [j])
                    {
                        rm++ ;                  // column k is not dead
                    }
                    PR ((" new rm %ld\n", rm)) ;
                }
            }
            else
            {
                // a non-pivotal column of front F
                j = Rj [pr + k] ;
                PR (("\nk %ld non-pivotal column of front f %ld, global j %ld\n",
                    k, f, j)) ;
                ASSERT (j >= Super [f+1] && j < QRsym->n) ;
                if (keepH)
                {
                    t = Stair [k] ;             // length of R+H vector
                    ASSERT (t >= rm && t <= fm) ;
                    h = MIN (h+1, fm) ;         // one more row of C to skip
                    ASSERT (t >= h) ;
                }
            }

            // -----------------------------------------------------------------
            // extract the column of R
            // -----------------------------------------------------------------

            PR (("extract column j (%ld) of R (rm %ld entries)\n", j, rm)) ;

            #ifndef NDEBUG
            for (i = 0 ; i < rm ; i++)
            {
                rij = R [i] ;
                PR (("  got row %ld col %ld value", row1+i, j)) ;
                PRVAL ((rij)) ;
                PR (("\n")) ;
            }
            #endif

            for (i = 0 ; i < rm ; i++)
            {
                rij = *(R++) ;

                if (rij != (Entry) 0)
                {
                    // R (row1+i,j) is nonzero, copy into Ra or Rb
                    if (j < n2)
                    {
                        if (getRa && row1 + i < econ)
                        {
                            p = Rap [j]++ ;
                            Rai [p] = row1 + i ;
                            Rax [p] = rij ;
                            ASSERT (p < Rap [j+1]) ;
                        }
                    }
                    else
                    {
                        if (getRb && row1 + i < econ)
                        {
                            if (getT)
                            {
                                p = Rbp [row1+i]++ ;
                                Rbi [p] = j-n2 ;
                                Rbx [p] = spqr_conj (rij) ;
                                ASSERT (p < Rbp [row1+i+1]) ;
                            }
                            else
                            {
                                p = Rbp [j-n2]++ ;
                                Rbi [p] = row1 + i ;
                                Rbx [p] = rij ;
                                ASSERT (p < Rbp [j-n2+1]) ;
                            }
                        }
                    }
                }
            }

            // -----------------------------------------------------------------
            // extract the column of H
            // -----------------------------------------------------------------

            ASSERT (IMPLIES (keepH, t >= h)) ;
            if (keepH && t >= h)
            {
                // skip the Householder reflection if it's empty
                if (getH && Tau [k] != (Entry) 0)
                {
                    H2Tau [nh++] = Tau [k] ;
                    H2i [ph] = Hi [h-1] + n1rows ;  // the implicit identity
                    H2x [ph] = 1 ;
                    ph++ ;
                    for (i = h ; i < t ; i++)
                    {
                        hij = *(R++) ;
                        if (hij != (Entry) 0)
                        {
                            H2i [ph] = Hi [i] + n1rows ;
                            H2x [ph] = hij ;
                            ph++ ;
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
}


// explicit instantiations

template void spqr_rconvert <double, int32_t>
(
    spqr_symbolic <int32_t> *QRsym, spqr_numeric <double, int32_t> *QRnum,
    int32_t n1rows, int32_t econ, int32_t n2, int getT, int32_t *Rap,
    int32_t *Rai, double *Rax, int32_t *Rbp, int32_t *Rbi, double *Rbx,
    int32_t *H2p, int32_t *H2i, double *H2x, double *H2Tau
) ;

template void spqr_rconvert <Complex, int32_t>
(
    spqr_symbolic <int32_t> *QRsym, spqr_numeric <Complex, int32_t> *QRnum,
    int32_t n1rows, int32_t econ, int32_t n2, int getT, int32_t *Rap,
    int32_t *Rai, Complex *Rax, int32_t *Rbp, int32_t *Rbi, Complex *Rbx,
    int32_t *H2p, int32_t *H2i, Complex *H2x, Complex *H2Tau
) ;

template void spqr_rconvert <double, int64_t>
(
    spqr_symbolic <int64_t> *QRsym, spqr_numeric <double, int64_t> *QRnum,
    int64_t n1rows, int64_t econ, int64_t n2, int getT, int64_t *Rap,
    int64_t *Rai, double *Rax, int64_t *Rbp, int64_t *Rbi, double *Rbx,
    int64_t *H2p, int64_t *H2i, double *H2x, double *H2Tau
) ;

template void spqr_rconvert <Complex, int64_t>
(
    spqr_symbolic <int64_t> *QRsym, spqr_numeric <Complex, int64_t> *QRnum,
    int64_t n1rows, int64_t econ, int64_t n2, int getT, int64_t *Rap,
    int64_t *Rai, Complex *Rax, int64_t *Rbp,int64_t *Rbi, Complex *Rbx,
    int64_t *H2p, int64_t *H2i, Complex *H2x, Complex *H2Tau
) ;
