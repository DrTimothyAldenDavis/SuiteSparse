//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_cm: CHOLMOD test program
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// A program for exhaustive statement-coverage for CHOLMOD, AMD, COLAMD, and
// CCOLAMD.  It tests every line of code in all three packages.
//
// For a complete test, all CHOLMOD modules, AMD, COLAMD, CCOLAMD, METIS,
// LAPACK, and the BLAS are required.  A partial test can be performed without
// the Supernodal and/or Partition modules.  METIS is not required if the
// Partition module is not installed.  LAPACK and the BLAS are not required
// if the Supernodal module is not installed.
//
// Usage:
//
//      cm < input > output
//
// where "input" contains a sparse matrix in triplet form.  The first line of
// the file contains four or five integers:
//
//      nrow ncol nnz stype complex
//
// where the matrix is nrow-by-ncol.  nnz is the number of (i,j,aij) triplets
// in the rest of the file, one triplet per line.  stype is -1 (symmetric
// with entries in lower triangular part provided), 0 (unsymmetric), or 1
// (symmetric upper).  If the 5th entry is missing, or 0, then the matrix is
// real; if 1 the matrix is complex, if 2 the matrix is zomplex.  Each
// subsequent line contains
//
//      i j aij
//
// for the row index, column index, and value of A(i,j).  Duplicate entries
// are summed.  If stype is 2 or 3, the rest of the file is ignored, and a
// special test matrix is constructed (2: arrowhead, 3: tridiagonal plus a
// dense row).  Test matrices are located in the Matrix/ subdirectory.
//
// For complex matrices, each line consists of
//
//      i j xij zij
//
// where xij is the real part of A(i,j) and zij is the imaginary part.
//
// cm takes one optional parameter.  If present (it does not matter what the
// argument is, actually) then extension memory-failure tests are performed.

#include "cm.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

double zero [2], one [2], minusone [2] ;
cholmod_common Common, *cm ;
double Zero [2] ;

//------------------------------------------------------------------------------
// my_rand
//------------------------------------------------------------------------------

// The POSIX example of rand, duplicated here so that the same sequence will
// be generated on different machines.

static unsigned long next = 1 ;

// see MY_RAND_MAX in cm.h
Int my_rand (void)
{
   next = next * 1103515245 + 12345 ;
   return ((unsigned)(next/65536) % (MY_RAND_MAX + 1)) ;
}

void my_srand (unsigned seed)
{
   next = seed ;
}

unsigned long my_seed (void)
{
   return (next) ;
}

//------------------------------------------------------------------------------
// progress
//------------------------------------------------------------------------------

// print a "." on stderr to indicate progress

#define COUNT 100

Int dot = 0 ;
void progress (Int force, char s)
{
    dot++ ;
    if (force)
    {
        dot = 0 ;
    }
    if (dot % COUNT == 0)
    {
        fprintf (stderr, "%c", s) ;
    }
    fflush (stdout) ;
    fflush (stderr) ;
}

//------------------------------------------------------------------------------
// my_handler
//------------------------------------------------------------------------------

// An error occurred that should not have occurred

void my_handler (int status, const char *file, int line, const char *msg)
{
    printf ("Error handler: file %s line %d status %d: %s\n",
            file, line, status, msg) ;
    if (status < CHOLMOD_OK || status > CHOLMOD_DSMALL)
    {
        fprintf (stderr, "\n\n************************************************"
                "********************************\n"
                "*** Test failure: file: %s line: %d\n"
                "*** status: %d message: %s\n"
                "***********************************************************"
                "*********************\n", file, line, status, msg);
        fflush (stderr) ;
        fflush (stdout) ;
        abort ( ) ;
    }
}

//------------------------------------------------------------------------------
// Assert
//------------------------------------------------------------------------------

void Assert (int truth, char *file, int line)
{
    if (!truth)
    {
        my_handler (-1, file, line, "") ;
    }
}

//------------------------------------------------------------------------------
// nrand
//------------------------------------------------------------------------------

// return a random Int between 0 and n-1

Int nrand (Int n)
{
    return ((n <= 0) ? (0) : (rand ( ) % n)) ;
}

//------------------------------------------------------------------------------
// xrand
//------------------------------------------------------------------------------

// return a random double between 0 and x

double xrand (double range)
{
    return ((range * (double) (my_rand ( ))) / MY_RAND_MAX) ;
}

//------------------------------------------------------------------------------
// rand_set
//------------------------------------------------------------------------------

// allocate and construct a random set of 0:n-1, possibly with duplicates

Int *rand_set (Int len, Int n)
{
    Int *cset ;
    Int k ;
    cset = CHOLMOD(malloc) (len, sizeof (Int), cm) ;
    if (cset == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot create random set") ;
        return (NULL) ;
    }
    for (k = 0 ; k < len ; k++)
    {
        cset [k] = nrand (n) ;
    }
    CHOLMOD(print_subset) (cset, len, n, "random cset", cm) ;
    return (cset) ;
}

//------------------------------------------------------------------------------
// zeros
//------------------------------------------------------------------------------

// Same as cholmod_zeros or cholmod_l_zeros, except it allows for a leading
// dimension that is different than nrow

cholmod_dense *zeros (Int nrow, Int ncol, Int d, int xdtype)
{
    cholmod_dense *X ;
    Real *Xx, *Xz ;
    Int i, nz ;
    X = CHOLMOD(allocate_dense) (nrow, ncol, d, xdtype, cm) ;
    if (X == NULL)
    {
        return (NULL) ;
    }
    Xx = X->x ;
    Xz = X->z ;
    nz = MAX (1, X->nzmax) ;
    switch (X->xtype)
    {
        case CHOLMOD_REAL:
            for (i = 0 ; i < nz ; i++)
            {
                Xx [i] = 0 ;
            }
            break ;
        case CHOLMOD_COMPLEX:
            for (i = 0 ; i < 2*nz ; i++)
            {
                Xx [i] = 0 ;
            }
            break ;
        case CHOLMOD_ZOMPLEX:
            for (i = 0 ; i < nz ; i++)
            {
                Xx [i] = 0 ;
            }
            for (i = 0 ; i < nz ; i++)
            {
                Xz [i] = 0 ;
            }
            break ;
    }
    return (X) ;
}

//------------------------------------------------------------------------------
// xtrue
//------------------------------------------------------------------------------

// Allocate and construct a dense matrix, X(i,j) = i+j*d+1

// if tweak is nonzero, X(0,0) is revised.

cholmod_dense *xtrue (Int nrow, Int ncol, Int d, int xdtype, int tweak)
{
    Real *x, *z ;
    cholmod_dense *X ;
    Int j, i ;
    X = zeros (nrow, ncol, d, xdtype) ;
    if (X == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot create dense matrix") ;
        return (NULL) ;
    }
    x = X->x ;
    z = X->z ;

    int xtype = xdtype & 3 ;

    if (xtype == CHOLMOD_REAL)
    {
        for (j = 0 ; j < ncol ; j++)
        {
            for (i = 0 ; i < nrow ; i++)
            {
                x [i+j*d] = i+j*d + 1 ;
            }
        }
    }
    else if (xtype == CHOLMOD_COMPLEX)
    {
        for (j = 0 ; j < ncol ; j++)
        {
            for (i = 0 ; i < nrow ; i++)
            {
                x [2*(i+j*d)  ] = i+j*d + 1 ;
                x [2*(i+j*d)+1] = ((Real) (j+i*d + 1))/10 ;
            }
        }
    }
    else // xtype == CHOLMOD_ZOMPLEX
    {
        for (j = 0 ; j < ncol ; j++)
        {
            for (i = 0 ; i < nrow ; i++)
            {
                x [i+j*d] = i+j*d + 1 ;
                z [i+j*d] = ((Real) (j+i*d + 1))/10 ;
            }
        }
    }

    if (tweak != 0 && nrow > 0 && ncol > 0)
    {
        x [0] += ((double) tweak) / 10.0 ;
        if (xtype == CHOLMOD_COMPLEX)
        {
            x [1] += ((double) tweak) / 20.0 ;
        }
        else if (xtype == CHOLMOD_ZOMPLEX)
        {
            z [0] += ((double) tweak) / 20.0 ;
        }
    }

    return (X) ;
}

//------------------------------------------------------------------------------
// resid
//------------------------------------------------------------------------------

// compute r = norm (A*x-b)/norm(b) or r = norm (A*A'*x-b)/norm(b)

double resid (cholmod_sparse *A, cholmod_dense *X, cholmod_dense *B)
{
    double r, bnorm ;
    cholmod_dense *R, *X2, *B2 ;
    cholmod_sparse *C, *A2 ;
    Int d, n, nrhs, xtype ;

    if (A == NULL || X == NULL || B == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute resid") ;
        return (1) ;
    }

    cm->status = CHOLMOD_OK ;
    n = B->nrow ;

    //--------------------------------------------------------------------------
    // convert all inputs to an identical xtype
    //--------------------------------------------------------------------------

    xtype = MAX (A->xtype, X->xtype) ;
    xtype = MAX (xtype, B->xtype) ;
    A2 = NULL ;
    B2 = NULL ;
    X2 = NULL ;
    if (A->xtype != xtype)
    {
        A2 = CHOLMOD(copy_sparse) (A, cm) ;
        CHOLMOD(sparse_xtype) (xtype + DTYPE, A2, cm) ;
        A = A2 ;
    }
    if (X->xtype != xtype)
    {
        X2 = CHOLMOD(copy_dense) (X, cm) ;
        CHOLMOD(dense_xtype) (xtype + DTYPE, X2, cm) ;
        X = X2 ;
    }
    if (B->xtype != xtype)
    {
        B2 = CHOLMOD(copy_dense) (B, cm) ;
        CHOLMOD(dense_xtype) (xtype + DTYPE, B2, cm) ;
        B = B2 ;
    }

    if (cm->status < CHOLMOD_OK)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute resid") ;
        CHOLMOD(free_sparse) (&A2, cm) ;
        CHOLMOD(free_dense) (&B2, cm) ;
        CHOLMOD(free_dense) (&X2, cm) ;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // get the right-hand-side, B, and its norm
    //--------------------------------------------------------------------------

    nrhs = B->ncol ;
    d = B->d ;
    if (nrhs == 1)
    {
        // inf-norm, 1-norm, or 2-norm (random choice)
        bnorm = CHOLMOD(norm_dense) (B, nrand (2), cm) ;
    }
    else
    {
        // inf-norm or  1-norm (random choice)
        bnorm = CHOLMOD(norm_dense) (B, nrand (1), cm) ;
    }

    //--------------------------------------------------------------------------
    // compute the residual
    //--------------------------------------------------------------------------

    if (A->stype == 0)
    {
        if (n < 10 && A->xtype == CHOLMOD_REAL)
        {
            // test cholmod_aat, C = A*A'
            C = CHOLMOD(aat) (A, NULL, 0, 1, cm) ;

            // R = B
            R = CHOLMOD(copy_dense) (B, cm) ;
            // R = C*X - R
            CHOLMOD(sdmult) (C, FALSE, one, minusone, X, R, cm) ;
            CHOLMOD(free_sparse) (&C, cm) ;

        }
        else
        {
            // W = A'*X
            cholmod_dense *W ;
            W = CHOLMOD(zeros) (A->ncol, nrhs, A->xtype + DTYPE, cm) ;
            CHOLMOD(sdmult) (A, TRUE, one, zero, X, W, cm) ;
            // R = B
            R = CHOLMOD(copy_dense) (B, cm) ;
            // R = A*W - R
            CHOLMOD(sdmult) (A, FALSE, one, minusone, W, R, cm) ;
            CHOLMOD(free_dense) (&W, cm) ;
        }
    }
    else
    {
        // R = B
        R = CHOLMOD(copy_dense) (B, cm) ;
        // R = A*X - R
        CHOLMOD(sdmult) (A, FALSE, one, minusone, X, R, cm) ;
    }

    //--------------------------------------------------------------------------
    // r = norm (R) / norm (B)
    //--------------------------------------------------------------------------

    r = CHOLMOD(norm_dense) (R, 1, cm) ;

    CHOLMOD(free_dense) (&R, cm) ;
    CHOLMOD(free_sparse) (&A2, cm) ;
    CHOLMOD(free_dense) (&B2, cm) ;
    CHOLMOD(free_dense) (&X2, cm) ;

    if (bnorm > 0)
    {
        r /= bnorm ;
    }
    return (r) ;
}

//------------------------------------------------------------------------------
// resid_sparse
//------------------------------------------------------------------------------

// compute r = norm (A*x-b)/norm(b) or r = norm (A*A'*x-b)/norm(b)

double resid_sparse (cholmod_sparse *A, cholmod_sparse *X, cholmod_sparse *B)
{
    double r, bnorm ;
    cholmod_sparse *R, *W, *AT, *W2 ;
    cholmod_dense *X2, *B2 ;
    Int n, nrhs, xtype ;

    if (A == NULL || X == NULL || B == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute resid") ;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // compute the residual
    //--------------------------------------------------------------------------

    xtype = MAX (A->xtype, X->xtype) ;
    xtype = MAX (xtype, B->xtype) ;

    if (xtype > CHOLMOD_REAL)
    {

        //----------------------------------------------------------------------
        // convert X and B to dense if any is complex or zomplex
        //----------------------------------------------------------------------

        X2 = CHOLMOD(sparse_to_dense) (X, cm) ;
        B2 = CHOLMOD(sparse_to_dense) (B, cm) ;
        r = resid (A, X2, B2) ;
        CHOLMOD(free_dense) (&X2, cm) ;
        CHOLMOD(free_dense) (&B2, cm) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // all inputs are real
        //----------------------------------------------------------------------

        n = B->nrow ;
        nrhs = B->ncol ;
        // inf-norm or 1-norm (random choice)
        bnorm = CHOLMOD(norm_sparse) (B, nrand (1), cm) ;

        if (A->stype == 0)
        {
            // W = A'*X
            AT = CHOLMOD(transpose) (A, 1, cm) ;
            W = CHOLMOD(ssmult) (AT, X, 0, TRUE, FALSE, cm) ;
            CHOLMOD(free_sparse) (&AT, cm) ;
            // W2 = A*W
            W2 = CHOLMOD(ssmult) (A, W, 0, TRUE, FALSE, cm) ;
            CHOLMOD(free_sparse) (&W, cm) ;
            // R = W2 - B
            R = CHOLMOD(add) (W2, B, one, minusone, TRUE, FALSE, cm) ;
            CHOLMOD(free_sparse) (&W2, cm) ;
        }
        else
        {
            // W = A*X
            W = CHOLMOD(ssmult) (A, X, 0, TRUE, FALSE, cm) ;
            // R = W - B
            R = CHOLMOD(add) (W, B, one, minusone, TRUE, FALSE, cm) ;
            CHOLMOD(free_sparse) (&W, cm) ;
        }

        r = CHOLMOD(norm_sparse) (R, 1, cm) ;
        CHOLMOD(free_sparse) (&R, cm) ;
        if (bnorm > 0)
        {
            r /= bnorm ;
        }
    }

    return (r) ;
}

//------------------------------------------------------------------------------
// resid3
//------------------------------------------------------------------------------

// r = norm (A1*A2*A3*x - b) /  norm (b)

double resid3 (cholmod_sparse *A1, cholmod_sparse *A2, cholmod_sparse *A3,
    cholmod_dense *X, cholmod_dense *B)
{
    double r, bnorm ;
    cholmod_dense *R, *W1, *W2, *X2, *B2 ;
    cholmod_sparse *C1, *C2, *C3 ;
    Int n, nrhs, d, xtype ;

    if (A1 == NULL || X == NULL || B == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute resid3") ;
        return (1) ;
    }

    cm->status = CHOLMOD_OK ;

    n = B->nrow ;

    //--------------------------------------------------------------------------
    // convert all inputs to an identical xtype
    //--------------------------------------------------------------------------

    xtype = MAX (A1->xtype, X->xtype) ;
    xtype = MAX (xtype, B->xtype) ;
    if (A2 != NULL)
    {
        xtype = MAX (xtype, A2->xtype) ;
    }
    if (A3 != NULL)
    {
        xtype = MAX (xtype, A3->xtype) ;
    }

    C1 = NULL ;
    C2 = NULL ;
    C3 = NULL ;
    B2 = NULL ;
    X2 = NULL ;

    if (A1->xtype != xtype)
    {
        C1 = CHOLMOD(copy_sparse) (A1, cm) ;
        CHOLMOD(sparse_xtype) (xtype + DTYPE, C1, cm) ;
        A1 = C1 ;
    }

    if (A2 != NULL && A2->xtype != xtype)
    {
        C2 = CHOLMOD(copy_sparse) (A2, cm) ;
        CHOLMOD(sparse_xtype) (xtype + DTYPE, C2, cm) ;
        A2 = C2 ;
    }

    if (A3 != NULL && A3->xtype != xtype)
    {
        C3 = CHOLMOD(copy_sparse) (A3, cm) ;
        CHOLMOD(sparse_xtype) (xtype + DTYPE, C3, cm) ;
        A3 = C3 ;
    }

    if (X->xtype != xtype)
    {
        X2 = CHOLMOD(copy_dense) (X, cm) ;
        CHOLMOD(dense_xtype) (xtype + DTYPE, X2, cm) ;
        X = X2 ;
    }

    if (B->xtype != xtype)
    {
        B2 = CHOLMOD(copy_dense) (B, cm) ;
        CHOLMOD(dense_xtype) (xtype + DTYPE, B2, cm) ;
        B = B2 ;
    }

    if (cm->status < CHOLMOD_OK)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute resid3") ;
        CHOLMOD(free_sparse) (&C1, cm) ;
        CHOLMOD(free_sparse) (&C2, cm) ;
        CHOLMOD(free_sparse) (&C3, cm) ;
        CHOLMOD(free_dense) (&B2, cm) ;
        CHOLMOD(free_dense) (&X2, cm) ;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // get B and its norm
    //--------------------------------------------------------------------------

    nrhs = B->ncol ;
    d = B->d ;
    bnorm = CHOLMOD(norm_dense) (B, 1, cm) ;

    //--------------------------------------------------------------------------
    // compute the residual
    //--------------------------------------------------------------------------

    if (A3 != NULL)
    {
        // W1 = A3*X
        W1 = CHOLMOD(zeros) (n, nrhs, xtype + DTYPE, cm) ;
        CHOLMOD(sdmult) (A3, FALSE, one, zero, X, W1, cm) ;
    }
    else
    {
        W1 = X ;
    }

    if (A2 != NULL)
    {
        // W2 = A2*W1
        W2 = CHOLMOD(eye) (n, nrhs, xtype + DTYPE, cm) ;
        CHOLMOD(sdmult) (A2, FALSE, one, zero, W1, W2, cm) ;
    }
    else
    {
        W2 = W1 ;
    }

    // R = B
    R = CHOLMOD(copy_dense) (B, cm) ;

    // R = A1*W2 - R
    CHOLMOD(sdmult) (A1, FALSE, one, minusone, W2, R, cm) ;

    //--------------------------------------------------------------------------
    // r = norm (R) / norm (B)
    //--------------------------------------------------------------------------

    r = CHOLMOD(norm_dense) (R, 1, cm) ;
    CHOLMOD(free_dense) (&R, cm) ;

    CHOLMOD(free_sparse) (&C1, cm) ;
    CHOLMOD(free_sparse) (&C2, cm) ;
    CHOLMOD(free_sparse) (&C3, cm) ;
    CHOLMOD(free_dense) (&B2, cm) ;
    CHOLMOD(free_dense) (&X2, cm) ;

    if (A3 != NULL)
    {
        CHOLMOD(free_dense) (&W1, cm) ;
    }
    if (A2 != NULL)
    {
        CHOLMOD(free_dense) (&W2, cm) ;
    }
    if (bnorm > 0)
    {
        r /= bnorm ;
    }
    return (r) ;
}

//------------------------------------------------------------------------------
// pnorm
//------------------------------------------------------------------------------

// r = norm (x-Pb) or r = norm(x-P'b).  This is lengthy because CHOLMOD does
// not provide any operations on dense matrices, and because it used to test
// the sparse-to-dense conversion routine.  Multiple methods are used to
// compute the same thing.

double pnorm (cholmod_dense *X, Int *P, cholmod_dense *B, Int inv)
{
    cholmod_dense *R, *X2, *B2 ;
    cholmod_factor *L ;
    Real *xx, *xz, *bx, *bz, *rx, *rz ;
    Int *Pinv, *Perm ;
    double rnorm, r ;
    Int i, j, k, n, nrhs, xtype, ok, save, lxtype ;

    if (X == NULL || P == NULL || B == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot compute pnorm") ;
        return (1) ;
    }

    save = cm->prefer_zomplex ;
    n = X->nrow ;
    nrhs = X->ncol ;
    rnorm = 0 ;

    Pinv = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    if (Pinv != NULL)
    {
        for (k = 0 ; k < n ; k++)
        {
            Pinv [P [k]] = k ;
        }
    }

    xtype = MAX (X->xtype, B->xtype) ;

    R = CHOLMOD(zeros) (n, nrhs, CHOLMOD_ZOMPLEX + DTYPE, cm) ;
    B2 = CHOLMOD(copy_dense) (B, cm) ;
    ok = R != NULL && B2 != NULL ;
    ok = ok && CHOLMOD(dense_xtype) (CHOLMOD_ZOMPLEX + DTYPE, B2, cm) ;

    for (lxtype = CHOLMOD_REAL ; ok && lxtype <= CHOLMOD_ZOMPLEX ; lxtype++)
    {
        // create a fake factor object
//      L = CHOLMOD(allocate_factor) (n, cm) ;
        L = CHOLMOD(alloc_factor) (n, DTYPE, cm) ;
        CHOLMOD(change_factor) (lxtype, TRUE, FALSE, TRUE, TRUE, L, cm) ;
        ok = ok && (L != NULL && L->Perm != NULL && Pinv != NULL) ;
        if (ok)
        {
            L->ordering = CHOLMOD_GIVEN ;
            Perm = L->Perm ;
            for (k = 0 ; k < n ; k++)
            {
                Perm [k] = Pinv [k] ;
            }
        }
        for (k = 0 ; k <= 1 ; k++)
        {
            // solve the inverse permutation system, X2 = P*X or X2 = P'*X
            cm->prefer_zomplex = k ;
            X2 = CHOLMOD(solve) (inv ? CHOLMOD_Pt : CHOLMOD_P, L, X, cm) ;

            ok = ok && CHOLMOD(dense_xtype) (CHOLMOD_ZOMPLEX + DTYPE, X2, cm) ;
            if (ok && X2 != NULL)
            {
                rx = R->x ;
                rz = R->z ;
                xx = X2->x ;
                xz = X2->z ;
                bx = B2->x ;
                bz = B2->z ;
                for (j = 0 ; j < nrhs ; j++)
                {
                    for (i = 0 ; i < n ; i++)
                    {
                        rx [i+j*n] = xx [i+j*n] - bx [i+j*n] ;
                        rz [i+j*n] = xz [i+j*n] - bz [i+j*n] ;
                    }
                }
            }
            r = CHOLMOD(norm_dense) (R, 0, cm) ;
            rnorm = MAX (r, rnorm) ;
            CHOLMOD(free_dense) (&X2, cm) ;
        }
        CHOLMOD(free_factor) (&L, cm) ;
    }

    CHOLMOD(free_dense) (&B2, cm) ;
    CHOLMOD(free_dense) (&R, cm) ;

    if (xtype == CHOLMOD_REAL)
    {
        // X and B are both real
        cholmod_sparse *Bs, *Pb ;
        Bs = CHOLMOD(dense_to_sparse) (B, TRUE, cm) ;
        Pb = CHOLMOD(submatrix) (Bs, inv ? Pinv : P, n, NULL, -1, TRUE, TRUE,
            cm) ;
        X2 = CHOLMOD(sparse_to_dense) (Pb, cm) ;
        R = CHOLMOD(zeros) (n, nrhs, CHOLMOD_REAL + DTYPE, cm) ;
        if (R != NULL && X != NULL && X2 != NULL)
        {
            rx = R->x ;
            xx = X->x ;
            bx = X2->x ;
            for (j = 0 ; j < nrhs ; j++)
            {
                for (i = 0 ; i < n ; i++)
                {
                    rx [i+j*n] = xx [i+j*n] - bx [i+j*n] ;
                }
            }
        }
        CHOLMOD(free_sparse) (&Bs, cm) ;
        CHOLMOD(free_sparse) (&Pb, cm) ;
        CHOLMOD(free_dense) (&X2, cm) ;
        r = CHOLMOD(norm_dense) (R, 1, cm) ;
        rnorm = MAX (rnorm, r) ;
        CHOLMOD(free_dense) (&R, cm) ;
    }

    CHOLMOD(free) (n, sizeof (Int), Pinv, cm) ;
    cm->prefer_zomplex = save ;
    return (rnorm) ;
}

//------------------------------------------------------------------------------
// prune_row
//------------------------------------------------------------------------------

// Set row k and column k of a packed matrix A to zero.  Set A(k,k) to 1
// if space is available.

void prune_row (cholmod_sparse *A, Int k)
{
    Real *Ax ;
    Int *Ap, *Ai ;
    Int ncol, p, i, j, nz ;

    if (A == NULL)
    {
        ERROR (CHOLMOD_INVALID, "nothing to prune") ;
        return ;
    }

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    nz = 0 ;
    ncol = A->ncol ;

    for (j = 0 ; j < ncol ; j++)
    {
        p = Ap [j] ;
        Ap [j] = nz ;
        if (j == k && nz < Ap [j+1])
        {
            Ai [nz] = k ;
            Ax [nz] = 1 ;
            nz++ ;
        }
        else
        {
            for ( ; p < Ap [j+1] ; p++)
            {
                i = Ai [p] ;
                if (i != k)
                {
                    Ai [nz] = i ;
                    Ax [nz] = Ax [p] ;
                    nz++ ;
                }
            }
        }
    }
    Ap [ncol] = nz ;
}

//------------------------------------------------------------------------------
// do_matrix
//------------------------------------------------------------------------------

double do_matrix (cholmod_sparse *A)
{
    double err, maxerr = 0 ;
    Int print, precise, maxprint, minprint, nmethods ;

    if (A == NULL)
    {
        ERROR (CHOLMOD_INVALID, "no matrix") ;
        return (1) ;
    }

    //--------------------------------------------------------------------------
    // determine print level, based on matrix size
    //--------------------------------------------------------------------------

    if (A->nrow <= 4)
    {
        minprint = 5 ;
        maxprint = 5 ;
    }
    else if (A->nrow <= 8)
    {
        minprint = 4 ;
        maxprint = 4 ;
    }
    else
    {
        minprint = 1 ;
        maxprint = 1 ;
    }

    //--------------------------------------------------------------------------
    // for all print levels and precisions, do:
    //--------------------------------------------------------------------------

    int psave = cm->print ;

    for (print = minprint ; print <= maxprint ; print++)
    {
        for (precise = 0 ; precise <= (print >= 4) ? 1 : 0 ; precise++)
        {
            Int save1, save2 ;

            maxerr = 0 ;
            my_srand (42) ;                                     // RAND reset
            cm->print = print ;
            cm->precise = precise ;
            printf ("\n----------Print level %d precise: %d\n",
                    cm->print, cm->precise) ;

            save1 = cm->final_asis ;
            save2 = cm->final_super ;
            cm->final_asis = FALSE ;
            cm->final_super = TRUE ;
            OK (CHOLMOD(print_common) ("1:cm", cm)) ;
            cm->final_asis = save1 ;
            cm->final_super = save2 ;

            //------------------------------------------------------------------
            // test various matrix operations
            //------------------------------------------------------------------

            err = test_ops (A) ;                                // RAND
            MAXERR (maxerr, err, 1) ;

            //------------------------------------------------------------------
            // solve the augmented system
            //------------------------------------------------------------------

            err = aug (A) ;                     // no random number use
            MAXERR (maxerr, err, 1) ;

            //------------------------------------------------------------------
            // solve using different methods
            //------------------------------------------------------------------

            printf ("test_solver (1)\n") ;
            cm->nmethods = 9 ;
            cm->final_asis = TRUE ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (2)\n") ;
            cm->final_asis = TRUE ;
            for (nmethods = 0 ; nmethods < 7 ; nmethods++)
            {
                cm->nmethods = nmethods ;
                cm->method [0].ordering = CHOLMOD_NATURAL ;
                err = test_solver (A) ;                         // RAND reset
                MAXERR (maxerr, err, 1) ;
            }

            printf ("test_solver (3)\n") ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_NESDIS ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (3b)\n") ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_NESDIS ;
            cm->method [0].nd_camd = 2 ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (3c)\n") ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_NATURAL ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (4)\n") ;
            cm->nmethods = 1 ;
            cm->method[0].ordering = CHOLMOD_METIS ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (5)\n") ;
            cm->nmethods = 1 ;
            cm->method [0].ordering = CHOLMOD_AMD ;
            CHOLMOD(free_work) (cm) ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            printf ("test_solver (6)\n") ;
            cm->nmethods = 1 ;
            cm->method[0].ordering = CHOLMOD_COLAMD ;
            err = test_solver (A) ;                             // RAND reset
            MAXERR (maxerr, err, 1) ;

            //------------------------------------------------------------------
            // restore default control parameters
            //------------------------------------------------------------------

            OK (CHOLMOD(print_common) ("2:cm", cm)) ;
            CHOLMOD(defaults) (cm) ; cm->useGPU = 0 ;
        }
    }

    printf ("do_matrix max error %.1g\n", maxerr) ;
    cm->print = psave ;

    return (maxerr) ;
}

//------------------------------------------------------------------------------
// main
//------------------------------------------------------------------------------

// Usage:
//      cm < matrix         do not perform intensive memory-failure tests
//      cm -m < matrix      do perform memory tests
//      cm -s < matrix      matrix is singular, nan error expected
//
// (The memory tests are performed if any argument is given to cm).

int main (int argc, char **argv)
{
    cholmod_triplet *T ;
    cholmod_sparse *A, *C, *AT ;
    double err = 0, maxerr = 0 ;
    Int n = 0, nmin = 0, nrow = 0, ncol = 0, save ;
    int singular, do_memory, i, do_nantests, ok ;
    double v = CHOLMOD_VERSION, tic [2], t ;
    int version [3] ;
    char *p ;
    const char* env_use_gpu;

    SuiteSparse_start ( ) ;
    SuiteSparse_tic (tic) ;
    printf ("Testing CHOLMOD (%g): %d ", v, CHOLMOD(version) (version)) ;
    printf ("(%d.%d.%d)\n", version [0], version [1], version [2]) ;
    OK (CHOLMOD_VERSION == CHOLMOD(version) (NULL)) ;
    v = SUITESPARSE_VERSION ;
    printf ("for SuiteSparse (%g): %d ", v, SuiteSparse_version (version)) ;
    printf ("(%d.%d.%d)\n", version [0], version [1], version [2]) ;
    printf ("%s: argc: %d\n", argv [0], argc) ;
    my_srand (42) ;                                             // RAND

    // Ignore floating point exceptions.  Infs and NaNs are generated
    // on purpose.
    signal (SIGFPE, SIG_IGN) ;

    // query the CHOLMOD_USE_GPU environment variable
    env_use_gpu = getenv ("CHOLMOD_USE_GPU") ;
    if ( env_use_gpu )
    {
        // CHOLMOD_USE_GPU environment variable is set to something
        if ( atoi ( env_use_gpu ) == 0 )
        {
            printf ("CHOLMOD_USE_GPU 0\n") ;
        }
        else
        {
            printf ("CHOLMOD_USE_GPU 1 (ignored for this test)\n") ;
        }
    }
    else
    {
        printf ("CHOLMOD_USE_GPU not present\n") ;
    }

    fflush (stdout) ;

    double maxerr_allowed = 0.1 ;

    singular = FALSE ;
    do_memory = FALSE ;
    do_nantests = FALSE ;
    for (i = 1 ; i < argc ; i++)
    {
        char *s ;
        s = argv [i] ;
        if (s [0] == '-' && s [1] == 'm')
        {
            // -m   do memory tests
            do_memory = TRUE ;
        }
        if (s [0] == '-' && s [1] == 's')
        {
            // -s   matrix is singular, maxerr not important (likely nan)
            singular = TRUE ;
        }
        if (s [0] == '-' && s [1] == 'n')
        {
            // -n   do tests with nans
            do_nantests = TRUE ;
        }
        if (s [0] == '-' && s [1] == 'e')
        {
            // -e# maxerr must be < 1e-#
            // for example, -e-10 means maxerr < 1e-10 must hold for the
            // test to pass.
            long int ee = strtol (s + 2, NULL, 10) ;
            if (ee > 0)
            {
                maxerr_allowed = pow ((double) 10, (double) (-ee)) ;
                printf ("maxerr allowed: %g\n", maxerr_allowed) ;
            }
        }
    }

    printf ("do_memory: %d singular: %d\n", do_memory, singular) ;

    //--------------------------------------------------------------------------
    // test SuiteSparse_config
    //--------------------------------------------------------------------------

    suitesparse_tests ( ) ;

    p = SuiteSparse_malloc (0, 0) ;
    OKP (p) ;
    p [0] = 'a' ;
    SuiteSparse_free (p) ;
    p = SuiteSparse_calloc (0, 0) ;
    OKP (p) ;
    p [0] = 'a' ;
    p = SuiteSparse_realloc (0, 0, 0, p, &ok) ;
    OK (ok) ;
    OKP (p) ;
    p [0] = 'a' ;
    SuiteSparse_free (p) ;
    p = SuiteSparse_malloc (INT64_MAX, 1024) ;
    NOP (p) ;
    p = SuiteSparse_calloc (INT64_MAX, 1024) ;
    NOP (p) ;
    p = SuiteSparse_realloc (0, 0, 0, NULL, &ok) ;
    OK (ok) ;
    OKP (p) ;
    p [0] = 'a' ;
    SuiteSparse_free (p) ;
    p = SuiteSparse_realloc (INT64_MAX, 0, 1024, NULL, &ok) ;
    NOP (p) ;
    NOT (ok) ;

    //--------------------------------------------------------------------------
    // initialize CHOLMOD
    //--------------------------------------------------------------------------

    cm = &Common ;
    OK (CHOLMOD(start) (cm)) ; cm->useGPU = 0 ;

    //--------------------------------------------------------------------------
    // test all methods with NULL common
    //--------------------------------------------------------------------------

    // no user error handler, since lots of errors will be raised here
    cm->error_handler = NULL ;
    null_test (NULL) ;
    save = cm->itype ;
    cm->itype = -999 ;
    null_test (cm) ;
    cm->itype = save ;
    null_test2 ( ) ;
    CHOLMOD(finish) (cm) ;
    OK (cm->malloc_count == 0) ;
    OK (CHOLMOD(start) (cm)) ; cm->useGPU = 0 ;

    //--------------------------------------------------------------------------
    // create basic scalars
    //--------------------------------------------------------------------------

    Zero [0] = 0 ;
    Zero [1] = 0 ;

    zero [0] = 0 ;
    zero [1] = 0 ;
    one [0] = 1 ;
    one [1] = 0 ;
    minusone [0] = -1 ;
    minusone [1] = 0 ;

    //--------------------------------------------------------------------------
    // basic tests
    //--------------------------------------------------------------------------

    basic1 (cm) ;
    printf ("basic1 tests OK\n") ;

    //--------------------------------------------------------------------------
    // integer overflow tests
    //--------------------------------------------------------------------------

    overflow_tests (cm) ;
    printf ("overflow tests OK\n") ;

    //--------------------------------------------------------------------------
    // read in a triplet matrix
    //--------------------------------------------------------------------------

    T = read_triplet (stdin) ;                                  // RAND
    OKP (T) ;
    int test_result = 0 ;
    ok = true ;

    //--------------------------------------------------------------------------
    // test CHOLMOD with the triplet matrix T
    //--------------------------------------------------------------------------

    if (T->nrow > 1000000)
    {

        //----------------------------------------------------------------------
        // do huge-problem tests only
        //----------------------------------------------------------------------

        huge ( ) ;                                              // RAND
        CHOLMOD(free_triplet) (&T, cm) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // test CHOLMOD with this matrix T
        //----------------------------------------------------------------------

        maxerr = 0 ;
        CHOLMOD(defaults) (cm) ; cm->useGPU = 0 ;
        cm->error_handler = my_handler ;
        cm->nmethods = 8 ;

        //----------------------------------------------------------------------
        // convert triplet to sparse matrix
        //----------------------------------------------------------------------

        A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
        AT = CHOLMOD(transpose) (A, 0, cm) ;
        C = unpack (A) ;                                        // RAND

        cm->print = 4 ;
        cm->precise = FALSE ;
        OK (CHOLMOD(print_sparse) (A, "Test matrix, A", cm)) ;
        OK (CHOLMOD(print_sparse) (C, "Unpacked/unsorted version of A", cm)) ;
        cm->print = 1 ;     // default cm->print for all remaining tests

        if (T != NULL)
        {
            nrow = T->nrow ;
            ncol = T->ncol ;
            n = MAX (nrow, ncol) ;
            nmin = MIN (nrow, ncol) ;
        }

        //----------------------------------------------------------------------
        // basic error tests, possibily with nans
        //----------------------------------------------------------------------

        null2 (T, do_nantests) ;                                // RAND

        printf ("Null2 OK : no error\n") ;
        maxerr = 0 ;

        //----------------------------------------------------------------------
        // remaining tests (skip if nantests were just done)
        //----------------------------------------------------------------------

        if (!do_nantests)
        {

            //------------------------------------------------------------------
            // raw factorization tests
            //------------------------------------------------------------------

            cm->error_handler = NULL ;
            err = raw_factor (A, 2) ;                           // RAND
            cm->error_handler = my_handler ;
            MAXERR (maxerr, err, 1) ;
            printf ("raw factorization error %.1g\n", err) ;

            err = raw_factor2 (A, 0., 0) ;
            MAXERR (maxerr, err, 1) ;
            printf ("raw factorization error2 %.1g\n", err) ;

            err = raw_factor2 (A, 1e-16, 0) ;
            MAXERR (maxerr, err, 1) ;
            printf ("raw factorization error3 %.1g\n", err) ;

            if (n < 1000 && A && T && A->stype == 1)
            {
                // factorize a symmetric matrix (upper part stored), possibly
                // with ignored entries in lower triangular part.
                cholmod_sparse *F ;
                int save = T->stype ;

                printf ("triplet to sparse:\n") ;
                T->stype = 0 ;
                F = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;
                T->stype = save ;

                if (F) F->stype = 1 ;

                err = raw_factor2 (F, 0., 0) ;
                MAXERR (maxerr, err, 1) ;
                printf ("raw factorization error4 %.1g\n", err) ;

                err = raw_factor2 (F, 1e-16, 0) ;
                MAXERR (maxerr, err, 1) ;
                printf ("raw factorization error5 %.1g\n", err) ;

                cm->dbound = 1e-15 ;
                cm->sbound = 1e-6 ;

                #ifdef DOUBLE
                double alpha = 1e-16 ;
                #else
                double alpha = 1e-8 ;
                #endif

                err = raw_factor2 (F, 0., 0) ;
                MAXERR (maxerr, err, 1) ;
                printf ("raw factorization error6 %.1g\n", err) ;

                err = raw_factor2 (F, alpha, 0) ;
                MAXERR (maxerr, err, 1) ;
                printf ("raw factorization error7 %.1g\n", err) ;

                err = raw_factor2 (F, alpha, 1) ;
                MAXERR (maxerr, err, 1) ;
                printf ("raw factorization error8 %.1g\n", err) ;

                cm->dbound = 0 ;
                cm->sbound = 0 ;

                CHOLMOD(free_sparse) (&F, cm) ;
            }

            //------------------------------------------------------------------
            // matrix ops
            //------------------------------------------------------------------

            err = test_ops (A) ;                                // RAND
            MAXERR (maxerr, err, 1) ;
            printf ("initial testops error %.1g\n", err) ;

            //------------------------------------------------------------------
            // analyze, factorize, solve
            //------------------------------------------------------------------

            err = solve (A) ;                                   // RAND
            MAXERR (maxerr, err, 1) ;
            printf ("initial solve error %.1g\n", err) ;

            //------------------------------------------------------------------
            // CCOLAMD tests
            //------------------------------------------------------------------

            cctest (A) ;                                        // RAND reset
            cctest (AT) ;                                       // RAND reset

            //------------------------------------------------------------------
            // COLAMD tests
            //------------------------------------------------------------------

            ctest (A) ;
            ctest (AT) ;

            //------------------------------------------------------------------
            // AMD tests
            //------------------------------------------------------------------

            if (n < NLARGE || A->stype)
            {
                // for unsymmetric matrices, this forms A*A' and A'*A, which
                // can fail if A has a dense row or column.  So it is only done
                // for modest-sized unsymmetric matrices.
                amdtest (A) ;
                amdtest (AT) ;
                camdtest (A) ;                                  // RAND
                camdtest (AT) ;                                 // RAND
            }

            if (n < NSMALL)
            {

                //--------------------------------------------------------------
                // do_matrix with an unpacked matrix
                //--------------------------------------------------------------

                // try with an unpacked matrix, and a non-default dbound
                cm->dbound = 1e-15 ;
                cm->sbound = 1e-6 ;
                err = do_matrix (C) ;                           // RAND reset
                MAXERR (maxerr, err, 1) ;
                cm->dbound = 0 ;
                cm->sbound = 0 ;

                //--------------------------------------------------------------
                // do_matrix: analyze, factorize, and solve, with many options
                //--------------------------------------------------------------

                err = do_matrix (A) ;                           // RAND reset
                MAXERR (maxerr, err, 1) ;

                //--------------------------------------------------------------
                // pretend to solve an LP
                //--------------------------------------------------------------

                if (nrow != ncol)
                {
                    int psave = cm->print ;
                    cm->print = 2 ;
                    err = lpdemo (T) ;                          // RAND
                    cm->print = psave ;
                    MAXERR (maxerr, err, 1) ;
                    cm->nmethods = 1 ;
                    cm->method [0].ordering = CHOLMOD_COLAMD ;
                    err = lpdemo (T) ;                          // RAND
                    MAXERR (maxerr, err, 1) ;
                    #ifdef DOUBLE
                    printf ("initial lp error %.1g, dbound %g\n",
                        err, cm->dbound) ;
                    #else
                    printf ("initial lp error %.1g, sbound %g\n",
                        err, cm->sbound) ;
                    #endif
                    cm->nmethods = 0 ;
                    cm->method [0].ordering = CHOLMOD_GIVEN ;
                }
            }

            //------------------------------------------------------------------
            // test more ops
            //------------------------------------------------------------------

            my_srand (42) ;                                     // RAND reset
            double err = test_ops2 (A) ;
            MAXERR (maxerr, err, 1) ;

            err = cat_tests (A, cm) ;
            MAXERR (maxerr, err, 1) ;

            err = dense_tests (A, cm) ;
            MAXERR (maxerr, err, 1) ;

            dtype_tests (A, cm) ;

            common_tests (cm) ;

            error_tests (A, cm) ;

            err = tofrom_tests (A, cm) ;
            MAXERR (maxerr, err, 1) ;

            //------------------------------------------------------------------
            // exhaustive memory-error handling for small matrices
            //------------------------------------------------------------------

            progress (1, '|') ;
            if (n < NSMALL && do_memory)
            {
                memory_tests (T) ;                              // RAND
            }
        }

        //----------------------------------------------------------------------
        // free matrices and print results
        //----------------------------------------------------------------------

        CHOLMOD(free_sparse) (&C, cm) ;
        CHOLMOD(free_sparse) (&AT, cm) ;
        CHOLMOD(free_sparse) (&A, cm) ;
        CHOLMOD(free_triplet) (&T, cm) ;

        ok = true ;
        if (nrow <= ncol && !singular)
        {
            ok = (maxerr <= maxerr_allowed) && !isnan (maxerr) ;
        }
    }

    //--------------------------------------------------------------------------
    // report results
    //--------------------------------------------------------------------------

    t = SuiteSparse_toc (tic) ;
    fprintf (stderr, "\n%8.2f sec", t) ;
    if (nrow <= ncol && !singular)
    {
        // maxerr should be NaN if nrow > ncol, so don't print it
        fprintf (stderr, ", maxerr %.1e <= %.1e", maxerr, maxerr_allowed) ;
    }
    if (ok)
    {
        fprintf (stderr, ", Test OK\n") ;
        test_result = 0 ;
    }
    else
    {
        fprintf (stderr, ", Test FAIL\n") ;
        test_result = 1 ;
    }

    //--------------------------------------------------------------------------
    // finalize CHOLMOD
    //--------------------------------------------------------------------------

    CHOLMOD(finish) (cm) ;

    int psave = cm->print ;
    cm->print = 5 ;
    OK (CHOLMOD(print_common) ("4:cm", cm)) ;
    cm->print = psave ;

    printf ("FINAL::malloc count "ID" inuse "ID"\n",
            (Int) cm->malloc_count,
            (Int) cm->memory_inuse) ;

    OK (cm->malloc_count == 0) ;
    OK (cm->memory_inuse == 0) ;
    if (nrow > ncol)
    {
        // maxerr should be NaN, so don't print it
        printf ("All tests passed: time %.1g\n", t) ;
    }
    else
    {
        printf ("All tests passed: max error %.1g time: %.1g\n", maxerr, t);
    }

    SuiteSparse_finish ( ) ;
    return (test_result) ;
}

