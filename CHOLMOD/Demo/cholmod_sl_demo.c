//------------------------------------------------------------------------------
// CHOLMOD/Demo/cholmod_sl_demo: demo program for CHOLMOD
//------------------------------------------------------------------------------

// CHOLMOD/Demo Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Read in a matrix from a file, and use CHOLMOD to solve Ax=b if A is
// symmetric, or (AA'+beta*I)x=b otherwise.  The file format is a simple
// triplet format, compatible with most files in the Matrix Market format.
// See cholmod_read.c for more details.  The readhb.f program reads a
// Harwell/Boeing matrix (excluding element-types) and converts it into the
// form needed by this program.  reade.f reads a matrix in Harwell/Boeing
// finite-element form.
//
// Usage:
//      cholmod_sl_demo matrixfile
//      cholmod_sl_demo < matrixfile
//
// The matrix is assumed to be positive definite (a supernodal LL' or simplicial
// LDL' factorization is used).
//
// Requires the Utility, Cholesky, MatrixOps, and Check Modules.
// Optionally uses the Partition and Supernodal Modules.
// Does not use the Modify Module.
//
// See cholmod_simple.c for a simpler demo program.
//
// There are 4 versions of this demo:
//
// cholmod_di_demo: double or double complex values, int32_t integers
// cholmod_dl_demo: double or double complex values, int64_t integers
// cholmod_si_demo: float or float complex values, int32_t integers
// cholmod_sl_demo: float or float complex values, int64_t integers

#include "cholmod_demo.h"
#define NTRIALS 100

// ff is a global variable so that it can be closed by my_handler
FILE *ff ;

// halt if an error occurs
static void my_handler (int status, const char *file, int line,
    const char *message)
{
    printf ("cholmod error: file: %s line: %d status: %d: %s\n",
            file, line, status, message) ;
    if (status < 0)
    {
        if (ff != NULL) fclose (ff) ;
        exit (1) ;
    }
}

static void check_version (char *package, int ver [3],
    int major, int minor, int patch)
{
    printf ("%s version %d.%d.%d\n", package, ver [0], ver [1], ver [2]) ;
    #ifndef TEST_COVERAGE
    if (ver [0] != major || ver [1] != minor || ver [2] != patch)
    {
        printf ("header version differs (%d,%d,%d) from library\n",
            major, minor, patch) ;
        my_handler (CHOLMOD_INVALID, __FILE__, __LINE__,
            "version mismatch") ;
    }
    #endif
}

int main (int argc, char **argv)
{
    double
        resid [4], t, ta, tf, ts [3], tot, bnorm, xnorm, anorm, rnorm, fl,
        anz, axbnorm, rnorm2, resid2, rcond,
        one [2], zero [2], minusone [2], beta [2], xlnz,
        isize, xsize, s, ss, lnz ;

    float
        *Bx, *Rx, *Xx, *Bz, *Xz, *Rz, xn, *X1x, *X2x, *B2x ;
    int64_t
        i, n, *Bsetp, *Bseti, *Xsetp, *Xseti, xlen, j, k, *Lnz ;

    FILE *f ;
    cholmod_sparse *A ;
    cholmod_dense *X = NULL, *B, *W, *R = NULL ;
    cholmod_common Common, *cm ;
    cholmod_factor *L ;
    int trial, method, L_is_super ;
    int ver [3] ;
    int prefer_zomplex, nmethods ;

    ts[0] = 0.;
    ts[1] = 0.;
    ts[2] = 0.;

    //--------------------------------------------------------------------------
    // get the file containing the input matrix
    //--------------------------------------------------------------------------

    ff = NULL ;
    prefer_zomplex = 0 ;
    if (argc > 1)
    {
        if ((f = fopen (argv [1], "r")) == NULL)
        {
            my_handler (CHOLMOD_INVALID, __FILE__, __LINE__,
                    "unable to open file") ;
        }
        ff = f ;
        prefer_zomplex = (argc > 2) ;
    }
    else
    {
        f = stdin ;
    }

    //--------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    cm->print = 3 ;

    cm->prefer_zomplex = prefer_zomplex ;

    // use default parameter settings, except for the error handler.  This
    // demo program terminates if an error occurs (out of memory, not positive
    // definite, ...).  It makes the demo program simpler (no need to check
    // CHOLMOD error conditions).  This non-default parameter setting has no
    // effect on performance.
    cm->error_handler = my_handler ;

    // Note that CHOLMOD will do a supernodal LL' or a simplicial LDL' by
    // default, automatically selecting the latter if flop/nnz(L) < 40.

    //--------------------------------------------------------------------------
    // create basic scalars
    //--------------------------------------------------------------------------

    zero [0] = 0 ;
    zero [1] = 0 ;
    one [0] = 1 ;
    one [1] = 0 ;
    minusone [0] = -1 ;
    minusone [1] = 0 ;
    beta [0] = 1e-6 ;
    beta [1] = 0 ;

    //--------------------------------------------------------------------------
    // check versions
    //--------------------------------------------------------------------------

    printf ("\n---------------------------------- cholmod_sl_demo:\n") ;

    cholmod_l_version (ver) ;
    check_version ("cholmod", ver, CHOLMOD_MAIN_VERSION,
        CHOLMOD_SUB_VERSION, CHOLMOD_SUBSUB_VERSION) ;

    SuiteSparse_version (ver) ;
    check_version ("SuiteSparse", ver, SUITESPARSE_MAIN_VERSION,
        SUITESPARSE_SUB_VERSION, SUITESPARSE_SUBSUB_VERSION) ;

    amd_version (ver) ;
    check_version ("AMD", ver, AMD_MAIN_VERSION,
        AMD_SUB_VERSION, AMD_SUBSUB_VERSION) ;

    colamd_version (ver) ;
    check_version ("COLAMD", ver, COLAMD_MAIN_VERSION,
        COLAMD_SUB_VERSION, COLAMD_SUBSUB_VERSION) ;

    #ifndef NCAMD
    camd_version (ver) ;
    check_version ("CAMD", ver, CAMD_MAIN_VERSION,
        CAMD_SUB_VERSION, CAMD_SUBSUB_VERSION) ;

    ccolamd_version (ver) ;
    check_version ("CCOLAMD", ver, CCOLAMD_MAIN_VERSION,
        CCOLAMD_SUB_VERSION, CCOLAMD_SUBSUB_VERSION) ;
    #endif

    //--------------------------------------------------------------------------
    // read in a matrix
    //--------------------------------------------------------------------------

    A = cholmod_l_read_sparse2 (f, CHOLMOD_SINGLE, cm) ;
    if (ff != NULL)
    {
        fclose (ff) ;
        ff = NULL ;
    }
    anorm = 1 ;
#ifndef NMATRIXOPS
    anorm = cholmod_l_norm_sparse (A, 0, cm) ;
    printf ("norm (A,inf) = %g\n", anorm) ;
    printf ("norm (A,1)   = %g\n", cholmod_l_norm_sparse (A, 1, cm)) ;
#endif

    if (prefer_zomplex && A->xtype == CHOLMOD_COMPLEX)
    {
        // Convert to zomplex, just for testing.  In a zomplex matrix,
        // the real and imaginary parts are in separate arrays.  MATLAB
        // uses zomplex matrix exclusively.
        cholmod_l_sparse_xtype (CHOLMOD_ZOMPLEX + A->dtype, A, cm) ;
    }

    int xtype = A->xtype ;  // real, complex, or zomplex
    int dtype = A->dtype ;  // single or double
    int xdtype = xtype + dtype ;
    cholmod_l_print_sparse (A, "A", cm) ;

    if (A->nrow > A->ncol)
    {
        // Transpose A so that A'A+beta*I will be factorized instead
        cholmod_sparse *C = cholmod_l_transpose (A, 2, cm) ;
        cholmod_l_free_sparse (&A, cm) ;
        A = C ;
        printf ("transposing input matrix\n") ;
    }

    //--------------------------------------------------------------------------
    // create an arbitrary right-hand-side
    //--------------------------------------------------------------------------

    n = A->nrow ;
    xn = n ;
    B = cholmod_l_zeros (n, 1, xdtype, cm) ;
    Bx = B->x ;
    Bz = B->z ;

#if GHS
    {
        // b = A*ones(n,1), used by Gould, Hu, and Scott in their experiments
        cholmod_dense *X0 ;
        X0 = cholmod_l_ones (A->ncol, 1, xdtype, cm) ;
        cholmod_l_sdmult (A, 0, one, zero, X0, B, cm) ;
        cholmod_l_free_dense (&X0, cm) ;
    }
#else
    if (xtype == CHOLMOD_REAL)
    {
        // real case
        for (i = 0 ; i < n ; i++)
        {
            Bx [i] = 1 + i / xn ;
        }
    }
    else if (xtype == CHOLMOD_COMPLEX)
    {
        // complex case
        for (i = 0 ; i < n ; i++)
        {
            Bx [2*i  ] = 1 + i / xn ;           // real part of B(i)
            Bx [2*i+1] = (xn/2 - i) / (3*xn) ;  // imag part of B(i)
        }
    }
    else // (xtype == CHOLMOD_ZOMPLEX)
    {
        // zomplex case
        for (i = 0 ; i < n ; i++)
        {
            Bx [i] = 1 + i / xn ;               // real part of B(i)
            Bz [i] = (xn/2 - i) / (3*xn) ;      // imag part of B(i)
        }
    }
#endif

    cholmod_l_print_dense (B, "B", cm) ;
    bnorm = 1 ;
#ifndef NMATRIXOPS
    bnorm = cholmod_l_norm_dense (B, 0, cm) ;   // max norm
    printf ("bnorm %g\n", bnorm) ;
#endif

    //--------------------------------------------------------------------------
    // analyze and factorize
    //--------------------------------------------------------------------------

    // The analysis and factorizations are repeated, to get accurate timings
    // and to exercise that particular feature.

    double maxresid = 0 ;

for (int overall_trials = 0 ; overall_trials <= 1 ; overall_trials++)
{
    printf ("\n=== Overall Trial: %d\n", overall_trials) ;

    t = CPUTIME ;
    L = cholmod_l_analyze (A, cm) ;
    ta = CPUTIME - t ;
    ta = MAX (ta, 0) ;

    printf ("Analyze: flop %g lnz %g, time: %g\n", cm->fl, cm->lnz, ta) ;

    for (int factor_trials = 0 ; factor_trials <= 2 ; factor_trials++)
    {
        if (A->stype == 0)
        {
            t = CPUTIME ;
            cholmod_l_factorize_p (A, beta, NULL, 0, L, cm) ;
            tf = CPUTIME - t ;
            tf = MAX (tf, 0) ;
            printf ("factor_trial: %d, Factorizing A*A'+beta*I, time: %g\n",
                factor_trials, tf) ;
        }
        else
        {
            t = CPUTIME ;
            cholmod_l_factorize (A, L, cm) ;
            tf = CPUTIME - t ;
            tf = MAX (tf, 0) ;
            printf ("factor trial: %d, Factorizing A, time: %g\n",
                factor_trials, tf) ;
        }
    }

    cholmod_l_print_factor (L, "L", cm) ;

    // determine the # of integers's and reals's in L.  See cholmod_free
    if (L->is_super)
    {
        s = L->nsuper + 1 ;
        xsize = L->xsize ;
        ss = L->ssize ;
        isize =
            n       // L->Perm
            + n     // L->ColCount, nz in each column of 'pure' L
            + s     // L->pi, column pointers for L->s
            + s     // L->px, column pointers for L->x
            + s     // L->super, starting column index of each supernode
            + ss ;  // L->s, the pattern of the supernodes
    }
    else
    {
        // this space can increase if you change parameters to their non-
        // default values (cm->final_pack, for example).
        lnz = L->nzmax ;
        xsize = lnz ;
        isize =
            n       // L->Perm
            + n     // L->ColCount, nz in each column of 'pure' L
            + n+1   // L->p, column pointers
            + lnz   // L->i, integer row indices
            + n     // L->nz, nz in each column of L
            + n+2   // L->next, link list
            + n+2 ; // L->prev, link list
    }

    // solve with Bset will change L from simplicial to supernodal
    rcond = cholmod_l_rcond (L, cm) ;
    L_is_super = L->is_super ;

    //--------------------------------------------------------------------------
    // solve
    //--------------------------------------------------------------------------

    if (n >= 1000)
    {
        nmethods = 1 ;
    }
    else if (xtype == CHOLMOD_ZOMPLEX)
    {
        nmethods = 2 ;
    }
    else
    {
        nmethods = 3 ;
    }
    printf ("nmethods: %d\n", nmethods) ;

    for (method = 0 ; method <= nmethods ; method++)
    {
        resid [method] = -1 ;       // not yet computed

        if (method == 0)
        {
            // basic solve, just once
            t = CPUTIME ;
            X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
            ts [0] = CPUTIME - t ;
            ts [0] = MAX (ts [0], 0) ;
        }
        else if (method == 1)
        {
            // basic solve, many times, but keep the last one
            t = CPUTIME ;
            for (trial = 0 ; trial < NTRIALS ; trial++)
            {
                cholmod_l_free_dense (&X, cm) ;
                Bx [0] = 1 + trial / xn ;       // tweak B each iteration
                X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
            }
            ts [1] = CPUTIME - t ;
            ts [1] = MAX (ts [1], 0) / NTRIALS ;
        }
        else if (method == 2)
        {
            // solve with reused workspace
            cholmod_dense *Ywork = NULL, *Ework = NULL ;
            cholmod_l_free_dense (&X, cm) ;

            t = CPUTIME ;
            for (trial = 0 ; trial < NTRIALS ; trial++)
            {
                Bx [0] = 1 + trial / xn ;       // tweak B each iteration
                cholmod_l_solve2 (CHOLMOD_A, L, B, NULL, &X, NULL,
                    &Ywork, &Ework, cm) ;
            }
            cholmod_l_free_dense (&Ywork, cm) ;
            cholmod_l_free_dense (&Ework, cm) ;
            ts [2] = CPUTIME - t ;
            ts [2] = MAX (ts [2], 0) / NTRIALS ;
        }
        else
        {
            // solve with reused workspace and sparse Bset
            if (xtype == CHOLMOD_ZOMPLEX) continue ;
            cholmod_dense *Ywork = NULL, *Ework = NULL ;
            cholmod_dense *X2 = NULL, *B2 = NULL ;
            cholmod_sparse *Bset, *Xset = NULL ;

            double err ;
            FILE *timelog = fopen ("timelog.m", "w") ;
            if (timelog) fprintf (timelog, "results = [\n") ;

            // xtype is CHOLMOD_REAL or CHOLMOD_COMPLEX, not CHOLMOD_ZOMPLEX
            B2 = cholmod_l_zeros (n, 1, xdtype, cm) ;
            B2x = B2->x ;

            Bset = cholmod_l_allocate_sparse (n, 1, 1, FALSE, TRUE, 0,
                CHOLMOD_PATTERN + dtype, cm) ;
            Bsetp = Bset->p ;
            Bseti = Bset->i ;
            Bsetp [0] = 0 ;     // nnz(B) is 1 (it can be anything)
            Bsetp [1] = 1 ;
            resid [3] = 0 ;

            for (i = 0 ; i < MIN (100,n) ; i++)
            {
                // B (i) is nonzero, all other entries are ignored
                // (implied to be zero)
                Bseti [0] = i ;
                if (xtype == CHOLMOD_REAL)
                {
                    B2x [i] = 3.1 * i + 0.9 ;
                }
                else // (xtype == CHOLMOD_COMPLEX)
                {
                    B2x [2*i  ] = i + 0.042 ;
                    B2x [2*i+1] = i - 92.7 ;
                }

                // first get the entire solution, to compare against
                cholmod_l_solve2 (CHOLMOD_A, L, B2, NULL, &X, NULL,
                    &Ywork, &Ework, cm) ;

                // now get the sparse solutions; this will change L from
                // supernodal to simplicial

                if (i == 0)
                {
                    // first solve can be slower because it has to allocate
                    // space for X2, Xset, etc, and change L.
                    // So don't time it
                    cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                        &Ywork, &Ework, cm) ;
                }

                t = CPUTIME ;
                for (trial = 0 ; trial < NTRIALS ; trial++)
                {
                    // solve Ax=b but only to get x(i).
                    // b is all zero except for b(i).
                    // This takes O(xlen) time
                    cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                        &Ywork, &Ework, cm) ;
                }
                t = CPUTIME - t ;
                t = MAX (t, 0) / NTRIALS ;

                // check the solution and log the time
                Xsetp = Xset->p ;
                Xseti = Xset->i ;
                xlen = Xsetp [1] ;
                X1x = X->x ;
                X2x = X2->x ;
                Lnz = L->nz ;

                if (xtype == CHOLMOD_REAL)
                {
                    fl = 2 * xlen ;
                    for (k = 0 ; k < xlen ; k++)
                    {
                        j = Xseti [k] ;
                        fl += 4 * Lnz [j] ;
                        err = X1x [j] - X2x [j] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                    }
                }
                else // (xtype == CHOLMOD_COMPLEX)
                {
                    fl = 16 * xlen ;
                    for (k = 0 ; k < xlen ; k++)
                    {
                        j = Xseti [k] ;
                        fl += 16 * Lnz [j] ;
                        err = X1x [2*j  ] - X2x [2*j  ] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                        err = X1x [2*j+1] - X2x [2*j+1] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                    }
                }

                if (timelog) fprintf (timelog, "%g %g %g %g\n",
                    (double) i, (double) xlen, fl, t);

                // clear B for the next test
                if (xtype == CHOLMOD_REAL)
                {
                    B2x [i] = 0 ;
                }
                else // (xtype == CHOLMOD_COMPLEX)
                {
                    B2x [2*i  ] = 0 ;
                    B2x [2*i+1] = 0 ;
                }
            }

            if (timelog)
            {
                fprintf (timelog, "] ; resid = %g ;\n", resid [3]) ;
                fprintf (timelog, "lnz = %g ;\n", cm->lnz) ;
                fprintf (timelog, "t = %g ;   %% dense solve time\n", ts [2]) ;
                fclose (timelog) ;
            }

#ifndef NMATRIXOPS
            double xnorm1 = cholmod_l_norm_dense (X, 1, cm) + ((n==0)?1:0) ;
            resid [3] = resid [3] / xnorm1 ;
#endif

            cholmod_l_free_dense (&Ywork, cm) ;
            cholmod_l_free_dense (&Ework, cm) ;
            cholmod_l_free_dense (&X2, cm) ;
            cholmod_l_free_dense (&B2, cm) ;
            cholmod_l_free_sparse (&Xset, cm) ;
            cholmod_l_free_sparse (&Bset, cm) ;
        }

        //----------------------------------------------------------------------
        // compute the residual
        //----------------------------------------------------------------------

        if (method < 3)
        {
#ifndef NMATRIXOPS
            if (A->stype == 0)
            {
                // (AA'+beta*I)x=b is the linear system that was solved
                // W = A'*X
                W = cholmod_l_allocate_dense (A->ncol, 1, A->ncol, xtype+dtype,
                    cm) ;
                cholmod_l_sdmult (A, 2, one, zero, X, W, cm) ;
                // R = B - beta*X
                cholmod_l_free_dense (&R, cm) ;
                R = cholmod_l_zeros (n, 1, xdtype, cm) ;
                Rx = R->x ;
                Rz = R->z ;
                Xx = X->x ;
                Xz = X->z ;
                if (xtype == CHOLMOD_REAL)
                {
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                    }
                }
                else if (xtype == CHOLMOD_COMPLEX)
                {
                    // complex case
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [2*i  ] = Bx [2*i  ] - beta [0] * Xx [2*i  ] ;
                        Rx [2*i+1] = Bx [2*i+1] - beta [1] * Xx [2*i+1] ;
                    }
                }
                else // (xtype == CHOLMOD_ZOMPLEX)
                {
                    // zomplex case
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                        Rz [i] = Bz [i] - beta [1] * Xz [i] ;
                    }
                }

                // R = A*W - R
                cholmod_l_sdmult (A, 0, one, minusone, W, R, cm) ;
                cholmod_l_free_dense (&W, cm) ;
            }
            else
            {
                // Ax=b was factorized and solved, R = B-A*X
                cholmod_l_free_dense (&R, cm) ;
                R = cholmod_l_copy_dense (B, cm) ;
                cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;
            }
            rnorm = cholmod_l_norm_dense (R, 0, cm) ;       // max abs. entry
            xnorm = cholmod_l_norm_dense (X, 0, cm) ;       // max abs. entry
            axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0)) ;
            resid [method] = rnorm / axbnorm ;
#else
            printf ("residual not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
        }
    }

    tot = ta + tf + ts [0] ;

    //--------------------------------------------------------------------------
    // iterative refinement (real symmetric case only)
    //--------------------------------------------------------------------------

    resid2 = -1 ;
#ifndef NMATRIXOPS
    if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
    {
        cholmod_dense *R2 ;

        // x = A\b
        cholmod_l_free_dense (&X, cm) ;
        X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;

        // R = B-A*X
        cholmod_l_free_dense (&R, cm) ;
        R = cholmod_l_copy_dense (B, cm) ;
        cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;

        // R2 = A\(B-A*X)
        R2 = cholmod_l_solve (CHOLMOD_A, L, R, cm) ;

        // compute X = X + A\(B-A*X)
        Xx = X->x ;
        Rx = R2->x ;
        for (i = 0 ; i < n ; i++)
        {
            Xx [i] = Xx [i] + Rx [i] ;
        }
        cholmod_l_free_dense (&R2, cm) ;

        // compute the new residual, R = B-A*X
        cholmod_l_free_dense (&R, cm) ;
        R = cholmod_l_copy_dense (B, cm) ;
        cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;
        rnorm2 = cholmod_l_norm_dense (R, 0, cm) ;
        resid2 = rnorm2 / axbnorm ;
    }
#endif

    cholmod_l_free_dense (&R, cm) ;

    //--------------------------------------------------------------------------
    // print results
    //--------------------------------------------------------------------------

    anz = cm->anz ;
    for (i = 0 ; i < CHOLMOD_MAXMETHODS ; i++)
    {
        fl = cm->method [i].fl ;
        xlnz = cm->method [i].lnz ;
        cm->method [i].fl = -1 ;
        cm->method [i].lnz = -1 ;
        int ordering = cm->method [i].ordering ;
        if (fl >= 0)
        {
            printf ("Ordering: ") ;
            if (ordering == CHOLMOD_POSTORDERED) printf ("postordered ") ;
            if (ordering == CHOLMOD_NATURAL)     printf ("natural ") ;
            if (ordering == CHOLMOD_GIVEN)       printf ("user    ") ;
            if (ordering == CHOLMOD_AMD)         printf ("AMD     ") ;
            if (ordering == CHOLMOD_METIS)       printf ("METIS   ") ;
            if (ordering == CHOLMOD_NESDIS)      printf ("NESDIS  ") ;
            if (xlnz > 0)
            {
                printf ("fl/lnz %10.1f", fl / xlnz) ;
            }
            if (anz > 0)
            {
                printf ("  lnz/anz %10.1f", xlnz / anz) ;
            }
            printf ("\n") ;
        }
    }

    printf ("ints in L: %15.0f, reals in L: %15.0f\n",
        (double) isize, (double) xsize) ;
    printf ("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
            cm->fl, cm->lnz) ;
    if (A->stype == 0)
    {
        printf ("nnz(A):    %15.0f\n", cm->anz) ;
    }
    else
    {
        printf ("nnz(A*A'): %15.0f\n", cm->anz) ;
    }
    if (cm->lnz > 0)
    {
        printf ("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz) ;
    }
    if (anz > 0)
    {
        printf ("nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz) ;
    }
    printf ("analyze time:  %12.4f\n", ta) ;
    printf ("factor  time:   %12.4f mflop: %8.1f\n", tf,
        (tf == 0) ? 0 : (1e-6*cm->fl / tf)) ;
    printf ("solve   time:   %12.4f mflop: %8.1f\n", ts [0],
        (ts [0] == 0) ? 0 : (1e-6*4*cm->lnz / ts [0])) ;
    printf ("overall time:   %12.4f mflop: %8.1f\n",
            tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot)) ;
    printf ("solve   time:   %12.4f mflop: %8.1f (%d trials)\n", ts [1],
        (ts [1] == 0) ? 0 : (1e-6*4*cm->lnz / ts [1]), NTRIALS) ;
    printf ("solve2  time:   %12.4f mflop: %8.1f (%d trials)\n", ts [2],
        (ts [2] == 0) ? 0 : (1e-6*4*cm->lnz / ts [2]), NTRIALS) ;
    printf ("peak memory usage: %12.0f (MB)\n",
            (double) (cm->memory_usage) / 1048576.) ;
    printf ("residual (|Ax-b|/(|A||x|+|b|)): ") ;
    for (method = 0 ; method <= nmethods ; method++)
    {
        printf ("%8.2e ", resid [method]) ;
        if (isnan (resid [method]) || resid [method] >= 0)
        {
            maxresid = MAX (maxresid, resid [method]) ;
        }
    }
    printf ("\n") ;
    if (isnan (resid2) || resid2 >= 0)
    {
        printf ("residual %8.1e (|Ax-b|/(|A||x|+|b|))"
                " after iterative refinement\n", resid2) ;
        maxresid = MAX (maxresid, resid2) ;
    }
    printf ("rcond    %8.1e\n\n", rcond) ;

    if (L_is_super)
    {
        cholmod_l_gpu_stats (cm) ;
    }

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_free_dense (&X, cm) ;
}

    //--------------------------------------------------------------------------
    // free matrices and finish CHOLMOD
    //--------------------------------------------------------------------------

    cholmod_l_free_sparse (&A, cm) ;
    cholmod_l_free_dense (&B, cm) ;
    cholmod_l_finish (cm) ;

    bool ok = !isnan (maxresid) &&
        maxresid < ((dtype == CHOLMOD_DOUBLE) ? 1e-7 : 1e-4) ;
    printf ("maxresid: %g : %s\n", maxresid, 
        ok ? "All tests passed" : "test failure") ;
    fprintf (stderr, "%s maxresid: %10.2g %s : %s\n",
        ok ? "OK" : "FAIL", maxresid,
        (dtype == CHOLMOD_DOUBLE) ? "(double)" : "(single)",
        (argc > 1) ? argv [1] : "stdin") ;
    return (ok ? 0 : 1) ;
}

