// =============================================================================
// === SuiteSparseQR ===========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//  QR factorization of a sparse matrix, and optionally solve a least squares
//  problem, Q*R = A*E where A is m-by-n, E is a permutation matrix, R is
//  upper triangular if A is full rank, and Q is an orthogonal matrix.
//  R is upper trapezoidal or a "squeezed" upper trapezoidal matrix if A is
//  found to be rank deficient.
//
//  All output arrays are optional.  If you pass NULL for the C, R, E, X,
//  and/or H array handles, those arrays will not be returned.
//
//  The Z output (either sparse or dense) is either C, C', or X where
//  C = Q'*B and X = E*(R\(Q'*B)).  The latter gives the result of X=A\B,
//  which is the least-squares solution if m > n.
//
//  To return full-sized results, set econ = m.  Then C and R will have m rows,
//  and C' will have m columns.
//
//  To return economy-sized results, set econ = n.  Then C and R will have k
//  rows and C' will have k columns, where k = min(m,n).
//
//  To return rank-sized results, set econ = 0.  Then C and R will have k rows
//  and C' will have k columns, where k = r = the estimated rank of A.
//
//  In all three cases, k = max (min (m, econ), r).
//
//  To compute Q, pass in B = speye (m), and then set getCTX to 1
//  (Q is then returned as Zsparse).
//
//  The Householder representation of Q is returned in H, HTau, and HPinv.
//  E is a permutation vector represention of the column permutation, for
//  the factorization Q*R=A*E.

#include "spqr.hpp"

#define XCHUNK 4            // FUTURE: make this a parameter

#define FREE_ALL \
        spqr_freefac (&QR, cc) ; \
        spqr_free <Int> (rjsize+1, sizeof (Int),  H2p, cc) ; \
        spqr_free_dense <Int>  (&HTau, cc) ; \
        spqr_free_sparse <Int> (&H, cc) ; \
        spqr_free_sparse <Int> (&R, cc) ; \
        spqr_free_sparse <Int> (&Xsparse, cc) ; \
        spqr_free_sparse <Int> (&Zsparse, cc) ; \
        spqr_free_dense <Int>  (&Zdense, cc) ; \
        spqr_free <Int> (xsize, sizeof (Entry), Xwork, cc) ; \
        spqr_free <Int> (csize, sizeof (Entry), C, cc) ; \
        spqr_free <Int> (wsize, sizeof (Entry), W, cc) ; \
        spqr_free <Int> (maxfrank, sizeof (Int), Rlive, cc) ; \
        spqr_free <Int> (maxfrank, sizeof (Entry *), Rcolp, cc) ; \
        spqr_free <Int> (n+bncols, sizeof (Int), E, cc) ; \
        spqr_free <Int> (m, sizeof (Int), HP1inv, cc) ;

// forward declarations

extern template SuiteSparseQR_factorization <Complex, int32_t>
*spqr_1factor <Complex, int32_t>
(
    int ordering, double tol, int32_t bncols, int keepH, cholmod_sparse *A,
    int32_t ldb, int32_t *Bp, int32_t *Bi, Complex *Bx, cholmod_common *cc
) ;

extern template SuiteSparseQR_factorization <double, int32_t>
*spqr_1factor <double, int32_t>
(
    int ordering, double tol, int32_t bncols, int keepH, cholmod_sparse *A,
    int32_t ldb, int32_t *Bp, int32_t *Bi, double *Bx, cholmod_common *cc
) ;

extern template int spqr_1fixed <Complex, int32_t>
(
    double tol, int32_t bncols, cholmod_sparse *A, int32_t **p_R1p,
    int32_t **p_P1inv, cholmod_sparse **p_Y, int32_t *p_n1cols,
    int32_t *p_n1rows, cholmod_common *cc
) ;

extern template int spqr_1fixed <double, int32_t>
(
    double tol, int32_t bncols, cholmod_sparse *A, int32_t **p_R1p,
    int32_t **p_P1inv, cholmod_sparse **p_Y, int32_t *p_n1cols,
    int32_t *p_n1rows, cholmod_common *cc
) ;

extern template spqr_symbolic <int32_t> 
*spqr_analyze <int32_t>
(
    cholmod_sparse *A, int ordering, int32_t *Quser, int do_rank_detection,
    int keepH, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

extern template SuiteSparseQR_factorization <Complex, SuiteSparse_long>
*spqr_1factor <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long bncols, int keepH,
    cholmod_sparse *A, SuiteSparse_long ldb, SuiteSparse_long *Bp,
    SuiteSparse_long *Bi, Complex *Bx, cholmod_common *cc
) ;

extern template int spqr_1fixed <Complex, SuiteSparse_long>
(
    double tol, SuiteSparse_long bncols, cholmod_sparse *A,
    SuiteSparse_long **p_R1p, SuiteSparse_long **p_P1inv, cholmod_sparse **p_Y,
    SuiteSparse_long *p_n1cols, SuiteSparse_long *p_n1rows, cholmod_common *cc
) ;

extern template int spqr_1fixed <double, SuiteSparse_long>
(
    double tol, SuiteSparse_long bncols, cholmod_sparse *A,
    SuiteSparse_long **p_R1p, SuiteSparse_long **p_P1inv, cholmod_sparse **p_Y,
    SuiteSparse_long *p_n1cols, SuiteSparse_long *p_n1rows, cholmod_common *cc
) ;

extern template SuiteSparseQR_factorization <double, SuiteSparse_long>
*spqr_1factor <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long bncols, int keepH,
    cholmod_sparse *A, SuiteSparse_long ldb, SuiteSparse_long *Bp,
    SuiteSparse_long *Bi, double *Bx, cholmod_common *cc
) ;

extern template spqr_symbolic <SuiteSparse_long>
*spqr_analyze <SuiteSparse_long>
(
    cholmod_sparse *A, int ordering, SuiteSparse_long *Quser,
    int do_rank_detection, int keepH, cholmod_common *cc
) ;

#endif

// =============================================================================
// === SuiteSparseQR overloaded functions ======================================
// =============================================================================

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // number of rows of C and R to return; a value
                            // less than the rank r of A is treated as r, and
                            // a value greater than m is treated as m.
                            // That is, e = max(min(m,econ),rank(A)) gives the
                            // number of rows of C and R, and columns of C'.

    int getCTX,             // if 0: return Z = C of size econ-by-bncols
                            // if 1: return Z = C' of size bncols-by-econ
                            // if 2: return Z = X of size econ-by-bncols

    cholmod_sparse *A,      // m-by-n sparse matrix

    // B is either sparse or dense.  If Bsparse is non-NULL, B is sparse and
    // Bdense is ignored.  If Bsparse is NULL and Bdense is non-NULL, then B is
    // dense.  B is not present if both are NULL.
    cholmod_sparse *Bsparse,    // B is m-by-bncols
    cholmod_dense *Bdense,

    // output arrays, neither allocated nor defined on input.

    // Z is the matrix C, C', or X, either sparse or dense.  If p_Zsparse is
    // non-NULL, then Z is returned as a sparse matrix.  If p_Zsparse is NULL
    // and p_Zdense is non-NULL, then Z is returned as a dense matrix.  Neither
    // are returned if both arguments are NULL.
    cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense,

    cholmod_sparse **p_R,   // the R factor
    Int **p_E,             // size n; fill-reducing ordering of A.
    cholmod_sparse **p_H,   // the Householder vectors (m-by-nh)
    Int **p_HPinv,         // size m; row permutation for H
    cholmod_dense **p_HTau, // size 1-by-nh, Householder coefficients

    // workspace and parameters
    cholmod_common *cc
)
{
    Int *Q1fill, *R1p, *R1j, *P1inv, *Zp, *Zi, *Rp, *Ri, *Rap, *Hp, *H2p,
        *Hi, *HP1inv, // *Ap, *Ai,
        *Rlive, *E, *Bp, *Bi ;
    Entry *R1x, *B, *Zx, *Rx, *X2, *C, *Hx, *C1, *X1, *Xwork, // *Ax,
        *W, **Rcolp, *Bx ;
    Int i, j, k, p, p2, n1cols, n1rows, B_is_sparse, Z_is_sparse, getC,
        getR, getH, getE, getX, getZ, iold, inew, rank, n2, rnz, znz, zn, zm,
        pr, xsize, getT, nh, k1, k2, xchunk, xncol, m, n, csize, wsize, ldb,
        maxfrank, rjsize, bncols, bnrows ;
    spqr_symbolic <Int> *QRsym ;
    spqr_numeric <Entry, Int> *QRnum ;
    SuiteSparseQR_factorization <Entry, Int> *QR ;
    cholmod_sparse *Xsparse, *Zsparse, *R, *H ;
    cholmod_dense *Zdense, *HTau ;

    double t0 = SuiteSparse_time ( ) ;

    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    if (p_Zsparse != NULL) *p_Zsparse = NULL ;
    if (p_Zdense  != NULL) *p_Zdense  = NULL ;
    if (p_R != NULL) *p_R = NULL ;
    if (p_E  != NULL) *p_E  = NULL ;
    if (p_H != NULL) *p_H = NULL ;
    if (p_HPinv != NULL) *p_HPinv = NULL ;
    if (p_HTau  != NULL) *p_HTau  = NULL ;

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    Int xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, EMPTY) ;
    if (Bsparse != NULL) RETURN_IF_XTYPE_INVALID (Bsparse, EMPTY) ;
    if (Bdense  != NULL) RETURN_IF_XTYPE_INVALID (Bdense,  EMPTY) ;
    cc->status = CHOLMOD_OK ;

    n = 0 ;
    m = 0 ;
    rnz = 0 ;
    nh = 0 ;

    Rap = NULL ;

    R = NULL ;
    Rp = NULL ;
    Ri = NULL ;
    Rx = NULL ;

    Xsparse = NULL ;
    Zsparse = NULL ;
    Zdense = NULL ;

    Zp = NULL ;
    Zi = NULL ;
    Zx = NULL ;

    H = NULL ;
    Hp = NULL ;
    Hi = NULL ;
    Hx = NULL ;
    H2p = NULL ;

    HTau = NULL ;
    HP1inv = NULL ;

    maxfrank = 0 ;
    xsize = 0 ;
    csize = 0 ;
    wsize = 0 ;

    Xwork = NULL ;
    C = NULL ;
    Rcolp = NULL ;
    Rlive = NULL ;

    E = NULL ;
    W = NULL ;

    m = A->nrow ;
    n = A->ncol ;
    // Ap = (Int *) A->p ;
    // Ai = (Int *) A->i ;
    // Ax = (Entry *) A->x ;

    // B is an optional input.  It can be sparse or dense
    B_is_sparse = (Bsparse != NULL) ;
    if (B_is_sparse)
    {
        // B is sparse
        bncols = Bsparse->ncol ;
        bnrows = Bsparse->nrow ;
        Bp = (Int *) Bsparse->p ;
        Bi = (Int *) Bsparse->i ;
        Bx = (Entry *) Bsparse->x ;
        ldb = 0 ;       // unused
    }
    else if (Bdense != NULL)
    {
        // B is dense
        bncols = Bdense->ncol ;
        bnrows = Bdense->nrow ;
        Bp = NULL ;
        Bi = NULL ;
        Bx = (Entry *) Bdense->x ;
        ldb = Bdense->d ;
    }
    else
    {
        // B is not present
        bncols = 0 ;
        bnrows = m ;
        Bp = NULL ;
        Bi = NULL ;
        Bx = NULL ;
        ldb = 0 ;
    }

    if (bnrows != m)
    {
        ERROR (CHOLMOD_INVALID, "A and B must have the same # of rows") ;
        return (EMPTY) ;
    }
    PR (("bncols : %ld\n", bncols)) ;

    // Z = C, C', or X, can be sparse or dense
    Z_is_sparse = (p_Zsparse != NULL) ;
    getZ = (Z_is_sparse || p_Zdense != NULL) && (getCTX >= 0 && getCTX <= 2) ;

    // Z = C is an optional output of size econ-by-bncols, sparse or dense
    getC = getZ && (getCTX == 0) ;

    // Z = C' is an optional output of size bncols-by-econ, sparse or dense
    getT = getZ && (getCTX == 1) ;

    // Z = X is an optional output of size n-by-bncols, sparse or dense
    getX = getZ && (getCTX == 2) ;

    // R is an optional output of size econ-by-n.  It is always sparse
    getR = (p_R != NULL) ;

    // E is an optional output; it is a permutation vector of length n+bncols
    getE = (p_E != NULL) ;

    // H is an optional output, of size m-by-nh.  It is always sparse
    getH = (p_H != NULL && p_HPinv != NULL && p_HTau != NULL) ;

    // at most one of C, C', or X will be returned as the Z matrix
    ASSERT (getC + getT + getX <= 1) ;

    PR (("getCTX %d getZ %ld getC %ld getT %ld getX %ld getR %ld getE %ld"
        "getH %ld\n", getCTX, getZ, getC, getT, getX, getR, getE, getH)) ;

    // -------------------------------------------------------------------------
    // symbolic and numeric QR factorization of [A B], exploiting singletons
    // -------------------------------------------------------------------------

    QR = spqr_1factor (ordering, tol, bncols, getH, A, ldb, Bp, Bi, Bx, cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (EMPTY) ;
    }

    R1p = QR->R1p ;
    R1j = QR->R1j ;
    R1x = QR->R1x ;
    P1inv = QR->P1inv ;
    Q1fill = QR->Q1fill ;
    QRsym = QR->QRsym ;
    QRnum = QR->QRnum ;
    n1rows = QR->n1rows ;
    n1cols = QR->n1cols ;
    n2 = n - n1cols ;
    rank = QR->rank ;
    tol = QR->tol ;

    PR (("Singletons: n1cols %ld n1rows %ld\n", n1cols, n1rows)) ;
    PR (("n2 %ld n %ld QRsym->n %ld\n", n2, n, QRsym->n)) ;

    // -------------------------------------------------------------------------
    // determine the economy size
    // -------------------------------------------------------------------------

    // econ gives the number of rows of C and R to return.  It must be at least
    // as large as the estimated rank of R (all entries in R are always
    // returned).  It can be no larger than m (all of C and R are returned).
    // If X is requested but not C nor R, then only rows 0 to rank-1 of C is
    // extracted from the QR factorization.

    if (getX && !getR)
    {
        econ = rank ;
    }
    else
    {
        // constrain econ to be between rank and m
        econ = MIN (m, econ) ;
        econ = MAX (econ, rank) ;
    }
    PR (("getX %ld econ %ld\n", getX, econ)) ;

    zm = getT ? bncols : econ ;     // number of rows of Z, if present
    zn = getT ? econ : bncols ;     // number of columns of Z, if present

    ASSERT (rank <= econ && econ <= m) ;

    // -------------------------------------------------------------------------
    // count the entries in the R factor of Y
    // -------------------------------------------------------------------------

    if (getZ)
    {
        // Zsparse is zm-by-zn, but with no entries so far
        Zsparse = spqr_allocate_sparse <Int> (zm, zn, 0, TRUE, TRUE, 0, xtype,
            cc) ;
        Zp = Zsparse ? ((Int *) Zsparse->p) : NULL ;
        PR (("Z is zm %ld by zn %ld\n", zm, zn)) ;
    }

    if (getR)
    {
        // R is econ-by-n, but with no entries so far
        R = spqr_allocate_sparse <Int> (econ, n, 0, TRUE, TRUE, 0, xtype, cc) ;
        Rp = R ? ((Int *) R->p) : NULL ;
        Rap = Rp + n1cols ;
    }

    rjsize = QRsym->rjsize ;    // rjsize is an upper bound on nh

    if (getH)
    {
        H2p = (Int *) spqr_malloc <Int> (rjsize+1, sizeof (Int), cc) ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_ALL ;
        return (EMPTY) ;
    }

    // count the number of nonzeros in each column of the R factor of Y,
    // and count the entries in the Householder vectors
    spqr_rcount (QRsym, QRnum, n1rows, econ, n2, getT, Rap, Zp, H2p, &nh) ;

#ifndef NDEBUG
    if (getZ) for (k = 0 ; k < zn ; k++)
    {
        PR (("Zp [%ld] %ld count\n", k, Zp [k])) ;
    }
    if (getR) for (k = 0 ; k < n ; k++)
    {
        PR (("R factor count Rp [%ld] = %ld\n", k, Rp [k])) ;
    }
#endif

    // -------------------------------------------------------------------------
    // count the entries in the singleton rows, R1
    // -------------------------------------------------------------------------

    if (getR)
    {
        // add the entries in the singleton rows R11 and R12
        for (k = 0 ; k < n1rows ; k++)
        {
            for (p = R1p [k] ; p < R1p [k+1] ; p++)
            {
                j = R1j [p] ;
                ASSERT (j >= k && j < n) ;
                Rp [j]++ ;
            }
        }
    }

#ifndef NDEBUG
    if (getR) for (k = 0 ; k < n ; k++)
    {
        PR (("R factor count Rp [%ld] = %ld after R1\n", k, Rp [k])) ;
    }
#endif

    // -------------------------------------------------------------------------
    // count the entries in B1 and add them to the counts of Z
    // -------------------------------------------------------------------------

    if (getZ && n1rows > 0)
    {
        // add the entries in the singleton rows of B1
        if (B_is_sparse)
        {
            // B is sparse
            for (k = 0 ; k < bncols ; k++)
            {
                // count the nonzero entries in column k of B1
                for (p = Bp [k] ; p < Bp [k+1] ; p++)
                {
                    iold = Bi [p] ;
                    inew = P1inv [iold] ;
                    if (inew < n1rows)
                    {
                        Zp [getT ? inew : k]++ ;
                    }
                }
            }
        }
        else
        {
            // B is dense
            B = Bx ;
            for (k = 0 ; k < bncols ; k++)
            {
                // count the nonzero entries in column k of B1
                for (iold = 0 ; iold < m ; iold++)
                {
                    inew = P1inv [iold] ;
                    if (inew < n1rows && B [iold] != (Entry) 0)
                    {
                        Zp [getT ? inew : k]++ ;
                    }
                }
                B += ldb ;
            }
        }
        #ifndef NDEBUG
        for (k = 0 ; k < zn ; k++)
        {
            PR (("after adding singletons Zp [%ld] %ld count\n", k, Zp [k])) ;
        }
        #endif
    }

    // -------------------------------------------------------------------------
    // compute the Rp and Zp column pointers (skip if NULL)
    // -------------------------------------------------------------------------

    // no Int overflow can occur
    rnz = spqr_cumsum (n, Rp) ;     // Rp = cumsum ([0 Rp])
    znz = spqr_cumsum (zn, Zp) ;    // Zp = cumsum ([0 Zp])

    // -------------------------------------------------------------------------
    // allocate R, Z, and H
    // -------------------------------------------------------------------------

    if (getR)
    {
        // R now has space for rnz entries
        spqr_reallocate_sparse <Int> (rnz, R, cc) ;
        Ri = (Int  *) R->i ;
        Rx = (Entry *) R->x ;
    }

    if (getZ)
    {
        // Zsparse now has space for znz entries
        spqr_reallocate_sparse <Int> (znz, Zsparse, cc) ;
        Zi = (Int  *) Zsparse->i ;
        Zx = (Entry *) Zsparse->x ;
    }

    if (getH)
    {
        // H is m-by-nh with hnz entries, where nh <= rjsize
        Int hnz = H2p [nh] ;
        H = spqr_allocate_sparse <Int> (m, nh, hnz, TRUE, TRUE, 0, xtype, cc) ;
        // copy the column pointers from H2p to Hp
        if (cc->status == CHOLMOD_OK)
        {
            Hp = (Int  *) H->p ;
            Hi = (Int  *) H->i ;
            Hx = (Entry *) H->x ;
            for (k = 0 ; k <= nh ; k++)
            {
                Hp [k] = H2p [k] ;
            }
        }
        // free the H2p workspace, and allocate HTau
        spqr_free <Int> (rjsize+1, sizeof (Int), H2p, cc) ;
        H2p = NULL ;
        HTau = spqr_allocate_dense <Int> (1, nh, 1, xtype, cc) ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_ALL ;
        return (EMPTY) ;
    }

    // -------------------------------------------------------------------------
    // place the singleton rows in R
    // -------------------------------------------------------------------------

    if (getR)
    {
        // add the entries in the singleton rows R11 and R12
        for (k = 0 ; k < n1rows ; k++)
        {
            for (p = R1p [k] ; p < R1p [k+1] ; p++)
            {
                j = R1j [p] ;
                ASSERT (j >= k && j < n) ;
                pr = Rp [j]++ ;
                Ri [pr] = k ;
                Rx [pr] = R1x [p] ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // place the B1 entries in C or C' 
    // -------------------------------------------------------------------------

    if (getZ && n1rows > 0)
    {
        if (B_is_sparse)
        {
            // B is sparse
            PR (("B1 sparse:\n")) ;
            for (k = 0 ; k < bncols ; k++)
            {
                // copy the nonzero entries in column k of B1 into C or C'
                for (p = Bp [k] ; p < Bp [k+1] ; p++)
                {
                    iold = Bi [p] ;
                    inew = P1inv [iold] ;
                    if (inew < n1rows)
                    {
                        if (getT)
                        {
                            p2 = Zp [inew]++ ;
                            Zi [p2] = k ;
                            Zx [p2] = spqr_conj (Bx [p]) ;
                        }
                        else
                        {
                            p2 = Zp [k]++ ;
                            Zi [p2] = inew ;
                            Zx [p2] = Bx [p] ;
                        }
                    }
                }
            }
        }
        else
        {
            // B is dense
            B = Bx ;
            PR (("B1 dense:\n")) ;
            for (k = 0 ; k < bncols ; k++)
            {
                // copy the nonzero entries in column k of B1 into C or C'
                for (iold = 0 ; iold < m ; iold++)
                {
                    inew = P1inv [iold] ;
                    if (inew < n1rows && B [iold] != (Entry) 0)
                    {
                        if (getT)
                        {
                            p2 = Zp [inew]++ ;
                            Zi [p2] = k ;
                            Zx [p2] = spqr_conj (B [iold]) ;
                        }
                        else
                        {
                            p2 = Zp [k]++ ;
                            Zi [p2] = inew ;
                            Zx [p2] = B [iold] ;
                        }
                    }
                }
                B += ldb ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // place the R factor entries in Z and R, and get H
    // -------------------------------------------------------------------------

    spqr_rconvert (QRsym, QRnum, n1rows, econ, n2, getT, Rap, Ri, Rx,
        Zp, Zi, Zx, Hp, Hi, Hx, HTau ? ((Entry *) HTau->x) : NULL) ;

    // -------------------------------------------------------------------------
    // finalize the column pointers of Z and R
    // -------------------------------------------------------------------------

    spqr_shift (n, Rp) ;
    spqr_shift (zn, Zp) ;

    // -------------------------------------------------------------------------
    // Finalize Z: either Z = E*(R\Z) or Z = full (Z)
    // -------------------------------------------------------------------------

    if (getX)
    {

        // ---------------------------------------------------------------------
        // solve the least squares problem: X = E*(R\C)
        // ---------------------------------------------------------------------

        // result X is n-by-bncols with leading dimension n, sparse or dense 
        maxfrank = QRnum->maxfrank  ;

        if (Z_is_sparse)
        {

            // -----------------------------------------------------------------
            // create the sparse result X
            // -----------------------------------------------------------------

            // ask for space for n+1 entries; this is increased later if needed
            Xsparse = spqr_allocate_sparse <Int> (n, bncols, n+1, TRUE, TRUE, 0,
                xtype, cc) ;
            xncol = 0 ;

            if (cc->status < CHOLMOD_OK)
            {
                // out of memory
                FREE_ALL ;
                return (EMPTY) ;
            }

            // Xwork will be a temporary dense result
            xchunk = MIN (bncols, XCHUNK) ;

            // xsize = n * xchunk ;
            xsize = spqr_mult (n, xchunk, &ok) ;

            // csize = rank * xchunk ;
            csize = spqr_mult (rank, xchunk, &ok) ;

            // wsize = maxfrank * xchunk ;
            wsize = spqr_mult (maxfrank, xchunk, &ok) ;

            if (ok)
            {
                Xwork = (Entry *) spqr_malloc <Int> (xsize, sizeof (Entry), cc) ;
                C = (Entry *) spqr_calloc <Int> (csize, sizeof (Entry), cc) ;
                W = (Entry *) spqr_malloc <Int> (wsize, sizeof (Entry), cc) ;
            }

            // -----------------------------------------------------------------
            // punt: use less workspace if we ran out of memory
            // -----------------------------------------------------------------

            if (!ok || (cc->status < CHOLMOD_OK && xchunk > 1))
            {
                // PUNT: out of memory; try again with xchunk = 1
                cc->status = CHOLMOD_OK ;
                ok = TRUE ;
                spqr_free <Int> (xsize, sizeof (Entry), Xwork, cc) ;
                spqr_free <Int> (csize, sizeof (Entry), C, cc) ;
                spqr_free <Int> (wsize, sizeof (Entry), W, cc) ;
                xchunk = 1 ;
                xsize = n ;
                csize = rank ;
                wsize = maxfrank ;
                Xwork = (Entry *) spqr_malloc <Int> (xsize, sizeof (Entry), cc) ;
                C = (Entry *) spqr_calloc <Int> (csize, sizeof (Entry), cc) ;
                W = (Entry *) spqr_malloc <Int> (wsize, sizeof (Entry), cc) ;
            }

            // -----------------------------------------------------------------
            // use Xwork for the solve
            // -----------------------------------------------------------------

            X2 = Xwork ;

        }
        else
        {

            // -----------------------------------------------------------------
            // construct the dense X
            // -----------------------------------------------------------------

            xchunk = bncols ;

            // csize = rank * xchunk ;
            csize = spqr_mult (rank, xchunk, &ok) ;

            // wsize = maxfrank * xchunk ;
            wsize = spqr_mult (maxfrank, xchunk, &ok) ;

            if (ok)
            {
                C = (Entry *) spqr_calloc <Int> (csize, sizeof (Entry), cc) ;
                W = (Entry *) spqr_malloc <Int> (wsize, sizeof (Entry), cc) ;
            }

            // allocate the dense X and use it for the solve
            Zdense = spqr_allocate_dense <Int> (n, bncols, n, xtype, cc) ;
            X2 = Zdense ? ((Entry *) Zdense->x) : NULL ;
        }

        Rlive = (Int *)   spqr_malloc <Int> (maxfrank, sizeof (Int),    cc) ;
        Rcolp = (Entry **) spqr_malloc <Int> (maxfrank, sizeof (Entry *), cc) ;

        if (!ok || cc->status < CHOLMOD_OK)
        {
            // still out of memory (or problem too large)
            FREE_ALL ;
            ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
            return (EMPTY) ;
        }

        for (k1 = 0 ; k1 < bncols ; k1 += xchunk)
        {

            // -----------------------------------------------------------------
            // this iteration computes X (:,k1:k2-1)
            // -----------------------------------------------------------------

            k2 = MIN (bncols, k1+xchunk) ;
            PR (("k1 %ld k2 %ld bncols %ld n1rows %ld\n",
                k1, k2, bncols, n1rows)) ;

#ifndef NDEBUG
            for (i = 0 ; i < rank*(k2-k1) ; i++)
            {
                ASSERT (C [i] == (Entry) 0) ;
            }
#endif

            // -----------------------------------------------------------------
            // scatter C
            // -----------------------------------------------------------------

            C1 = C ;
            for (k = k1 ; k < k2 ; k++)
            {
                // scatter C (0:rank-1,k1:k2-1) into workspace
                for (p = Zp [k] ; p < Zp [k+1] ; p++)
                {
                    i = Zi [p] ;
                    if (i < rank)
                    {
                        C1 [i] = Zx [p] ;
                    }
                }
                C1 += rank ;
            }

            // -----------------------------------------------------------------
            // X = E*(R\C)
            // -----------------------------------------------------------------

            spqr_rsolve (QR, TRUE, k2-k1, rank, C, X2, Rcolp, Rlive, W, cc) ;

            // -----------------------------------------------------------------
            // clear workspace C ; skip if this is the last k (C is freed)
            // -----------------------------------------------------------------

            if (k2 < bncols)
            {
                C1 = C ;
                for (k = k1 ; k < k2 ; k++)
                {
                    for (p = Zp [k] ; p < Zp [k+1] ; p++)
                    {
                        i = Zi [p] ;
                        if (i < rank)
                        {
                            C1 [i] = 0 ;
                        }
                    }
                    C1 += rank ;
                }
            }

            // -----------------------------------------------------------------
            // save the result, either sparse or dense
            // -----------------------------------------------------------------

            if (Z_is_sparse)
            {
                // append the columns of X2 onto the sparse X
                X1 = X2 ;
                for (k = 0 ; k < k2-k1 ; k++)
                {
                    spqr_append <Entry, Int> (X1, NULL, Xsparse, &xncol, cc) ;
                    X1 += n ;
                    if (cc->status < CHOLMOD_OK)
                    {
                        // out of memory
                        FREE_ALL ;
                        return (EMPTY) ;
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // free workspace
        // ---------------------------------------------------------------------

        C     = (Entry *)  spqr_free <Int> (csize,    sizeof (Entry), C, cc) ;
        W     = (Entry *)  spqr_free <Int> (wsize,    sizeof (Entry), W, cc) ;
        Rlive = (Int *)   spqr_free <Int> (maxfrank, sizeof (Int),  Rlive, cc);
        Rcolp = (Entry **) spqr_free <Int> (maxfrank, sizeof (Entry *), Rcolp,
            cc) ;

        // ---------------------------------------------------------------------
        // free the sparse Z
        // ---------------------------------------------------------------------

        spqr_free_sparse <Int> (&Zsparse, cc) ;

        // ---------------------------------------------------------------------
        // finalize the sparse X
        // ---------------------------------------------------------------------

        if (Z_is_sparse)
        {

            // -----------------------------------------------------------------
            // reduce X in size so that nnz(X) == nzmax(X)
            // -----------------------------------------------------------------

            znz = spqr_nnz <Int> (Xsparse, cc) ;
            spqr_reallocate_sparse <Int> (znz, Xsparse, cc) ;
            ASSERT (cc->status == CHOLMOD_OK) ;

            // -----------------------------------------------------------------
            // Z becomes the sparse X
            // -----------------------------------------------------------------

            Zsparse = Xsparse ;
            Xsparse = NULL ;

            // -----------------------------------------------------------------
            // free the dense Xwork
            // -----------------------------------------------------------------

            spqr_free <Int> (xsize, sizeof (Entry), Xwork, cc) ;
            Xwork = NULL ;
            xsize = 0 ;
        }

    }
    else if ((getC || getT) && !Z_is_sparse)
    {

        // ---------------------------------------------------------------------
        // convert C or C' to full
        // ---------------------------------------------------------------------

        Zdense = spqr_sparse_to_dense <Int> (Zsparse, cc) ;
        spqr_free_sparse <Int> (&Zsparse, cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            FREE_ALL ;
            return (EMPTY) ;
        }
    }

    // -------------------------------------------------------------------------
    // extract permutations, if requested, from the QR factors
    // -------------------------------------------------------------------------

    if (getE)
    {
        // preserve Q1fill if E has been requested
        E = QR->Q1fill ;
        QR->Q1fill = NULL ;
    }

    if (getH)
    {
        // preserve HP1inv if E has been requested.  If there are no singletons,
        // the QR->HP1inv is NULL, and QR->QRnum->HPinv serves in its place.
        if (n1cols > 0)
        {
            // singletons are present, get E from QR->HP1inv
            HP1inv = QR->HP1inv ;
            QR->HP1inv = NULL ;
        }
        else
        {
            // no singletons are present, get E from QR->QRnum->HPinv
            ASSERT (QR->HP1inv == NULL) ;
            HP1inv = QRnum->HPinv ;
            QRnum->HPinv = NULL ;
        }
    }

    // -------------------------------------------------------------------------
    // free the QR factors
    // -------------------------------------------------------------------------

    spqr_freefac (&QR, cc) ;

    // -------------------------------------------------------------------------
    // convert R to trapezoidal, if rank < n and ordering is not fixed
    // -------------------------------------------------------------------------

    if (getR && ordering != SPQR_ORDERING_FIXED && rank < n && tol >= 0)
    {
        Int *Rtrapp, *Rtrapi, *Qtrap ;
        Entry *Rtrapx ;

        // find Rtrap and Qtrap. This may fail if tol < 0 and the matrix
        // is structurally rank deficient; in that case, k2 = EMPTY and
        // Rtrap is returned NULL.
        k2 = spqr_trapezoidal (n, Rp, Ri, Rx, bncols, Q1fill, TRUE,
            &Rtrapp, &Rtrapi, &Rtrapx, &Qtrap, cc) ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            // This is very unlikely to fail since the QR factorization object
            // has just been freed
            FREE_ALL ;
            return (EMPTY) ;
        }

        ASSERT (k2 == EMPTY || rank == k2) ;

        if (Rtrapp != NULL)
        {
            // Rtrap was created (it was skipped if R was already in upper
            // trapezoidal form)

            // free the old R and Q1fill
            spqr_free <Int> (n+1,      sizeof (Int),  Rp, cc) ;
            spqr_free <Int> (rnz,      sizeof (Int),  Ri, cc) ;
            spqr_free <Int> (rnz,      sizeof (Entry), Rx, cc) ;
            spqr_free <Int> (n+bncols, sizeof (Int),  E,  cc) ;

            // replace R and Q1fill with Rtrap and Qtrap
            R->p = Rtrapp ;
            R->i = Rtrapi ;
            R->x = Rtrapx ;
            E = Qtrap ;
        }
    }

    // -------------------------------------------------------------------------
    // return results
    // -------------------------------------------------------------------------

    if (getZ)
    {
        // return C, C',  or X
        if (Z_is_sparse)
        {
            *p_Zsparse = Zsparse ;
        }
        else
        {
            *p_Zdense = Zdense ;
        }
    }

    if (getH)
    {
        // return H, of size m-by-nh in sparse column-oriented form
        *p_H = H ;
        *p_HTau = HTau ;
        *p_HPinv = HP1inv ;     // this is QR->HP1inv or QR->QRnum->HPinv
    }

    if (getR)
    {
        // return R, of size econ-by-n in sparse column-oriented form
        *p_R = R ;
    }

    if (getE)
    {
        // return E of size n+bncols (NULL if ordering == SPQR_ORDERING_FIXED)
        *p_E = E ;
    }

    double t3 = SuiteSparse_time ( ) ;
    double total_time = t3 - t0 ;
    cc->SPQR_solve_time =
        total_time - cc->SPQR_analyze_time - cc->SPQR_factorize_time ;

    return (rank) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ, int getCTX, cholmod_sparse *A,
    cholmod_sparse *Bsparse, cholmod_dense *Bdense, cholmod_sparse **p_Zsparse,
    cholmod_dense  **p_Zdense, cholmod_sparse **p_R, int32_t **p_E,
    cholmod_sparse **p_H, int32_t **p_HPinv, cholmod_dense **p_HTau,
    cholmod_common *cc
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ, int getCTX, cholmod_sparse *A,
    cholmod_sparse *Bsparse, cholmod_dense *Bdense, cholmod_sparse **p_Zsparse,
    cholmod_dense **p_Zdense, cholmod_sparse **p_R, int32_t **p_E,
    cholmod_sparse **p_H, int32_t **p_HPinv, cholmod_dense **p_HTau,
    cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, int getCTX,
    cholmod_sparse *A, cholmod_sparse *Bsparse, cholmod_dense *Bdense,
    cholmod_sparse **p_Zsparse, cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R, SuiteSparse_long **p_E, cholmod_sparse **p_H,
    SuiteSparse_long **p_HPinv, cholmod_dense **p_HTau, cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, int getCTX,
    cholmod_sparse *A, cholmod_sparse *Bsparse, cholmod_dense *Bdense,
    cholmod_sparse **p_Zsparse, cholmod_dense  **p_Zdense,
    cholmod_sparse **p_R, SuiteSparse_long **p_E, cholmod_sparse **p_H,
    SuiteSparse_long **p_HPinv, cholmod_dense **p_HTau, cholmod_common *cc
) ;

#endif

// -----------------------------------------------------------------------------
// X=A\B where X and B are dense
// -----------------------------------------------------------------------------
template <typename Entry, typename Int> cholmod_dense *SuiteSparseQR
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
)
{
    cholmod_dense *X ;
    SuiteSparseQR <Entry, Int> (ordering, tol, 0, 2, A,   
        NULL, B, NULL, &X, NULL, NULL, NULL, NULL, NULL, cc) ;
    return (X) ;
}


// explicit instantiations

template cholmod_dense *SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_dense  *B,
    cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_dense  *B,
    cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template cholmod_dense *SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_dense  *B,
    cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_dense  *B,
    cholmod_common *cc
) ;

#endif


// -----------------------------------------------------------------------------
// X=A\B where X and B are dense, default ordering and tol
// -----------------------------------------------------------------------------
template <typename Entry, typename Int> cholmod_dense *SuiteSparseQR
(
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
)
{
    cholmod_dense *X ;
    SuiteSparseQR <Entry, Int> (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 0, 2, A,   
        NULL, B, NULL, &X, NULL, NULL, NULL, NULL, NULL, cc) ;
    return (X) ;
}


// explicit instantiations

template cholmod_dense *SuiteSparseQR <double, int32_t>
(
    cholmod_sparse *A, cholmod_dense  *B, cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR <Complex, int32_t>
(
    cholmod_sparse *A, cholmod_dense  *B, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template cholmod_dense *SuiteSparseQR <double, SuiteSparse_long>
(
    cholmod_sparse *A, cholmod_dense  *B, cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR <Complex, SuiteSparse_long>
(
    cholmod_sparse *A, cholmod_dense  *B, cholmod_common *cc
) ;

#endif


// -----------------------------------------------------------------------------
// X=A\B where X and B are sparse 
// -----------------------------------------------------------------------------
template <typename Entry, typename Int> cholmod_sparse *SuiteSparseQR
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs
    cholmod_common *cc      // workspace and parameters
)
{

    cholmod_sparse *X ;
    SuiteSparseQR <Entry, Int> (ordering, tol, 0, 2, A,   
        B, NULL, &X, NULL, NULL, NULL, NULL, NULL, NULL, cc) ;

    return (X) ;
}


// explicit instantiations

template cholmod_sparse *SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_sparse *B,
    cholmod_common *cc
) ;

template cholmod_sparse *SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_sparse *B,
    cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template cholmod_sparse *SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_sparse *B,
    cholmod_common *cc
) ;

template cholmod_sparse *SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, cholmod_sparse *A, cholmod_sparse *B,
    cholmod_common *cc 
) ;

#endif


// -----------------------------------------------------------------------------
// [Q,R,E] = qr(A), returning Q as a sparse matrix
// -----------------------------------------------------------------------------

template <typename Entry, typename Int> Int SuiteSparseQR     // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **Q,     // m-by-e sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    cholmod_sparse *I ;
    Int xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    Int m = A->nrow ;
    I = spqr_speye <Int> (m, m, xtype, cc) ;
    Int rank = (I == NULL) ? EMPTY : SuiteSparseQR <Entry, Int> (ordering, tol,
        econ, 1, A, I, NULL, Q, NULL, R, E, NULL, NULL, NULL, cc) ;
    spqr_free_sparse <Int> (&I, cc) ;
    return (rank) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A,
    cholmod_sparse **Q, cholmod_sparse **R, int32_t **E, cholmod_common *cc 
) ;

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A,
    cholmod_sparse **Q, cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **Q, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **Q, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

#endif


// -----------------------------------------------------------------------------
// [Q,R,E] = qr(A), discarding Q
// -----------------------------------------------------------------------------

template <typename Entry, typename Int> Int SuiteSparseQR // returns rank(A) estimate
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // e-by-n sparse matrix
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, 1, A,   
        NULL, NULL, NULL, NULL, R, E, NULL, NULL, NULL, cc)) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A,
    cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A,
    cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **R, SuiteSparse_long **E, cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **R, SuiteSparse_long **E, cholmod_common *cc
) ;

#endif


// -----------------------------------------------------------------------------
// [C,R,E] = qr(A,B) where C and B are both dense
// -----------------------------------------------------------------------------

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_dense  *B,      // m-by-nrhs dense matrix
    // outputs
    cholmod_dense  **C,     // C = Q'*B, an e-by-nrhs dense matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, 0, A, NULL, B,
        NULL, C, R, E, NULL, NULL, NULL, cc)) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ,
    cholmod_sparse *A, cholmod_dense  *B,
    cholmod_dense  **C, cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ,
    cholmod_sparse *A, cholmod_dense *B, 
    cholmod_dense  **C, cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ,
    cholmod_sparse *A, cholmod_dense  *B,
    cholmod_dense  **C, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ,
    cholmod_sparse *A, cholmod_dense  *B,
    cholmod_dense  **C, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

#endif


// -----------------------------------------------------------------------------
// [C,R,E] = qr(A,B) where C and B are both sparse
// -----------------------------------------------------------------------------

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    cholmod_sparse *B,      // m-by-nrhs sparse matrix
    // outputs
    cholmod_sparse **C,     // C = Q'*B, an e-by-nrhs sparse matrix
    cholmod_sparse **R,     // e-by-n sparse matrix where e=max(econ,rank(A))
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, 0, A, B, NULL,
        C, NULL, R, E, NULL, NULL, NULL, cc)) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ,
    cholmod_sparse *A, cholmod_sparse *B,
    cholmod_sparse **C, cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ,
    cholmod_sparse *A, cholmod_sparse *B, 
    cholmod_sparse **C, cholmod_sparse **R, int32_t **E, cholmod_common *cc
) ;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ,
    cholmod_sparse *A, cholmod_sparse *B,
    cholmod_sparse **C, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ,
    cholmod_sparse *A, cholmod_sparse *B,
    cholmod_sparse **C, cholmod_sparse **R, SuiteSparse_long **E,
    cholmod_common *cc
) ;

#endif

// -----------------------------------------------------------------------------
// [Q,R,E] = qr(A) where Q is returned in Householder form
// -----------------------------------------------------------------------------

// returns rank(A) estimate if successful, EMPTY otherwise
template <typename Entry, typename Int> Int SuiteSparseQR
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // columns with 2-norm <= tol are treated as zero
    Int econ,              // e = max(min(m,econ),rank(A))
    cholmod_sparse *A,      // m-by-n sparse matrix
    // outputs
    cholmod_sparse **R,     // the R factor
    Int **E,               // permutation of 0:n-1, NULL if identity
    cholmod_sparse **H,     // the Householder vectors (m-by-nh)
    Int **HPinv,           // size m; row permutation for H
    cholmod_dense **HTau,   // size 1-by-nh, Householder coefficients
    cholmod_common *cc      // workspace and parameters
)
{
    return (SuiteSparseQR <Entry, Int> (ordering, tol, econ, EMPTY, A,
        NULL, NULL, NULL, NULL, R, E, H, HPinv, HTau, cc)) ;
}


// explicit instantiations

template int32_t SuiteSparseQR <double, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A, 
    cholmod_sparse **R, int32_t **E, cholmod_sparse **H, int32_t **HPinv, 
    cholmod_dense **HTau, cholmod_common *cc
) ;

template int32_t SuiteSparseQR <Complex, int32_t>
(
    int ordering, double tol, int32_t econ, cholmod_sparse *A,
    cholmod_sparse **R, int32_t **E, cholmod_sparse **H, int32_t **HPinv,
    cholmod_dense **HTau, cholmod_common *cc
) ;

template struct spqr_numeric <double, int32_t>;
template struct spqr_numeric <Complex, int32_t>;

#if SuiteSparse_long_max != INT32_MAX

template SuiteSparse_long SuiteSparseQR <double, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **R, SuiteSparse_long **E, cholmod_sparse **H,
    SuiteSparse_long **HPinv, cholmod_dense **HTau, cholmod_common *cc
) ;

template SuiteSparse_long SuiteSparseQR <Complex, SuiteSparse_long>
(
    int ordering, double tol, SuiteSparse_long econ, cholmod_sparse *A,
    cholmod_sparse **R, SuiteSparse_long **E, cholmod_sparse **H,
    SuiteSparse_long **HPinv, cholmod_dense **HTau, cholmod_common *cc
) ;

template struct spqr_numeric <double, SuiteSparse_long>;
template struct spqr_numeric <Complex, SuiteSparse_long>;

#endif
