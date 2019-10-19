// =============================================================================
// === spqr_solve mexFunction ==================================================
// =============================================================================

#include "spqr_mx.hpp"

/*  x = A\b using a sparse QR factorization.

    x = spqr_solve (A,b,opts) ;

    A 2nd optional output gives statistics, in a self-explainable struct.

    Let [m n] = size (A).

            opts is an optional struct with the following fields.  Fields not
            present are set to their defaults.  The defaults are:

                opts.tol = 20 * (m+n) * eps * sqrt(max(diag(A'*A)))
                opts.ordering = "colamd"

            opts.tol:  Default 20 * (m+n) * eps * sqrt(max(diag(A'*A))),
                that is, 20 * (m+n) * eps * (max 2-norm of the columns of A).

                If tol >= 0, then an approximate rank-revealing QR factorization
                is used.  A diagonal entry of R with magnitude < tol is treated
                as zero; the matrix A is considered rank deficient.

                If tol = -1, then no rank detection is attempted.  This allows
                the sparse QR to use less memory than when attempting to detect
                rank, even for full-rank matrices.  Thus, if you know your
                matrix has full rank, tol = -1 will reduce memory usage.  Zeros
                may appear on the diagonal of R if A is rank deficient.

                If tol <= -2, then the default tolerance is used.

            opts.ordering:  Default is "default"
                See spqr.m for info.

            opts.solution: default "basic"
                Defines how an underdetermined system should be solved (m < n).
                "basic": compute a basic solution, using
                        [C,R,E]=spqr(A,B) ; X=E*(R\C) ; where C=Q'*B
                "min2norm": compute a minimum 2-norm solution, using
                        [C,R,E]=spqr(A') ; X = Q*(R'\(E'*B))
                        A min2norm solution is more costly to compute, in time
                        and memory, than a basic solution.
                This option is ignored if m >= n.
*/

#define MIN(a,b) (((a) < (b)) ? (a) : (b))

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Int *Bp, *Bi ;
    double *Ax, *Bx, dummy, flops = 0 ;
    Int m, n, k, bncols, p, i, rank, A_complex, B_complex, is_complex,
        anz, bnz ;
    spqr_mx_options opts ;
    cholmod_sparse *A, Amatrix, *Xsparse ; 
    cholmod_dense *Xdense ;
    cholmod_common Common, *cc ;
    char msg [LEN+1] ;

#ifdef TIMING
    double t0 = spqr_time ( ) ;
#endif

    // -------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    // -------------------------------------------------------------------------

    cc = &Common ;
    cholmod_l_start (cc) ;
    spqr_mx_config (SPUMONI, cc) ;

    // -------------------------------------------------------------------------
    // check inputs
    // -------------------------------------------------------------------------

    if (nargout > 2)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxlhs", "Too many output arguments") ;
    }
    if (nargin < 2)
    {
        mexErrMsgIdAndTxt ("MATLAB:minrhs", "Not enough input arguments") ;
    }
    if (nargin > 3)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxrhs", "Too many input arguments") ;
    }

    // -------------------------------------------------------------------------
    // get the input matrix A (must be sparse)
    // -------------------------------------------------------------------------

    if (!mxIsSparse (pargin [0]))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "A must be sparse") ;
    }

    A = spqr_mx_get_sparse (pargin [0], &Amatrix, &dummy) ;
    m = A->nrow ;
    n = A->ncol ;
    A_complex = mxIsComplex (pargin [0]) ;

    B_complex = mxIsComplex (pargin [1]) ;
    is_complex = (A_complex || B_complex) ;
    Ax = spqr_mx_merge_if_complex (pargin [0], is_complex, &anz, cc) ; 
    if (is_complex)
    {
        // A has been converted from real or zomplex to complex
        A->x = Ax ;
        A->z = NULL ;
        A->xtype = CHOLMOD_COMPLEX ;
    }

    // -------------------------------------------------------------------------
    // determine usage and parameters
    // -------------------------------------------------------------------------

    spqr_mx_get_options ((nargin < 3) ? NULL : pargin [2], &opts, m, 3, cc) ;

    opts.Qformat = SPQR_Q_DISCARD ;
    opts.econ = 0 ;
    opts.permvector = TRUE ;
    opts.haveB = TRUE ;

    // -------------------------------------------------------------------------
    // get the input matrix B (sparse or dense)
    // -------------------------------------------------------------------------

    if (!mxIsNumeric (pargin [1]))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "invalid non-numeric B") ;
    }
    if (mxGetM (pargin [1]) != m)
    {
        mexErrMsgIdAndTxt ("QR:invalidInput",
            "A and B must have the same number of rows") ;
    }

    cholmod_sparse Bsmatrix, *Bsparse ;
    cholmod_dense  Bdmatrix, *Bdense ;

    // convert from real or zomplex to complex
    Bx = spqr_mx_merge_if_complex (pargin [1], is_complex, &bnz, cc) ;

    int B_is_sparse = mxIsSparse (pargin [1]) ;
    if (B_is_sparse)
    {
        Bsparse = spqr_mx_get_sparse (pargin [1], &Bsmatrix, &dummy) ;
        Bdense = NULL ;
        if (is_complex)
        {
            // Bsparse has been converted from real or zomplex to complex
            Bsparse->x = Bx ;
            Bsparse->z = NULL ;
            Bsparse->xtype = CHOLMOD_COMPLEX ;
        }
    }
    else
    {
        Bsparse = NULL ;
        Bdense = spqr_mx_get_dense (pargin [1], &Bdmatrix, &dummy) ;
        if (is_complex)
        {
            // Bdense has been converted from real or zomplex to complex
            Bdense->x = Bx ;
            Bdense->z = NULL ;
            Bdense->xtype = CHOLMOD_COMPLEX ;
        }
    }

    // -------------------------------------------------------------------------
    // X = A\B
    // -------------------------------------------------------------------------

    if (opts.min2norm && m < n)
    {
#ifndef NEXPERT
        // This requires SuiteSparseQR_expert.cpp
        if (is_complex)
        {
            if (B_is_sparse)
            {
                // X and B are both sparse and complex
                Xsparse = SuiteSparseQR_min2norm <Complex> (opts.ordering,
                    opts.tol, A, Bsparse, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Xsparse, cc) ;
            }
            else
            {
                // X and B are both dense and complex
                Xdense = SuiteSparseQR_min2norm <Complex> (opts.ordering,
                    opts.tol, A, Bdense, cc) ;
                pargout [0] = spqr_mx_put_dense (&Xdense, cc) ;
            }
        }
        else
        {
            if (B_is_sparse)
            {
                // X and B are both sparse and real
                Xsparse = SuiteSparseQR_min2norm <double> (opts.ordering,
                    opts.tol, A, Bsparse, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Xsparse, cc) ;
            }
            else
            {
                // X and B are both dense and real
                Xdense = SuiteSparseQR_min2norm <double> (opts.ordering,
                    opts.tol, A, Bdense, cc) ;
                pargout [0] = spqr_mx_put_dense (&Xdense, cc) ;
            }
        }
#else
        mexErrMsgIdAndTxt ("QR:notInstalled", "min2norm method not installed") ;
#endif

    }
    else
    {

        if (is_complex)
        {
            if (B_is_sparse)
            {
                // X and B are both sparse and complex
                Xsparse = SuiteSparseQR <Complex> (opts.ordering, opts.tol,
                    A, Bsparse, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Xsparse, cc) ;
            }
            else
            {
                // X and B are both dense and complex
                Xdense = SuiteSparseQR <Complex> (opts.ordering, opts.tol,
                    A, Bdense, cc) ;
                pargout [0] = spqr_mx_put_dense (&Xdense, cc) ;
            }
        }
        else
        {
            if (B_is_sparse)
            {
                // X and B are both sparse and real
                Xsparse = SuiteSparseQR <double> (opts.ordering, opts.tol,
                    A, Bsparse, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Xsparse, cc) ;
            }
            else
            {
                // X and B are both dense and real
                Xdense = SuiteSparseQR <double> (opts.ordering, opts.tol,
                    A, Bdense, cc) ;
                pargout [0] = spqr_mx_put_dense (&Xdense, cc) ;
            }
        }
    }

#ifdef TIMING
    flops = cc->other1 [0] ;
    double t = spqr_time ( ) - t0 ;
#endif

    // -------------------------------------------------------------------------
    // info output
    // -------------------------------------------------------------------------

    if (nargout > 1)
    {
        const char *info_struct [ ] =
        {
            "nnzR_upper_bound",             // 0: nnz(R) bound
            "nnzH_upper_bound",             // 1: nnz(H) bound
            "number_of_frontal_matrices",   // 2: nf
            "number_of_TBB_tasks",          // 3: ntasks
            "rank_A_estimate",              // 4: rank
            "number_of_column_singletons",  // 5: n1cols
            "number_of_singleton_rows",     // 6: n1rows
            "ordering",                     // 7: ordering used
            "memory_usage_in_bytes",        // 8: memory usage
            "flops_upper_bound",            // 9: upper bound on flop count
                                            //    (excluding backsolve)
            "tol",                          // 10: column norm tolerance used
            "number_of_TBB_threads"         // 11: # threads used
#ifdef TIMING
            , "analyze_time"                // 12: analyze time
            , "factorize_time"              // 13: factorize time (and apply Q')
            , "solve_time"                  // 14: R\C backsolve only
            , "total_time"                  // 15: total x=A\b in seconds
            , "flops"                       // 16: actual flops (incl backsolve)
#endif
        } ;

#ifdef TIMING
#define NINFO 17
#else
#define NINFO 12
#endif

        pargout [1] = mxCreateStructMatrix (1, 1, NINFO, info_struct) ;

        for (Int k = 0 ; k <= 6 ; k++)
        {
            mxSetFieldByNumber (pargout [1], 0, k,
                mxCreateDoubleScalar ((double) cc->SPQR_istat [k])) ;
        }

        // get the ordering used.  Note that "default", "best", and "cholmod"
        // are not among the possible results, since they are meta-orderings
        // that select among AMD, COLAMD, and/or METIS.
        mxArray *ord ;
        switch (cc->SPQR_istat [7])
        {
            case SPQR_ORDERING_FIXED:
            case SPQR_ORDERING_GIVEN: 
                ord = mxCreateString ("fixed") ;
                break ;
            case SPQR_ORDERING_NATURAL:
                ord = mxCreateString ("natural") ;
                break ;
            case SPQR_ORDERING_COLAMD:
                ord = mxCreateString ("colamd") ;
                break ;
            case SPQR_ORDERING_AMD:
                ord = mxCreateString ("amd") ;
                break ;
#ifndef NPARTITION
            case SPQR_ORDERING_METIS:
                ord = mxCreateString ("metis") ;
                break ;
#endif
            default:
                ord = mxCreateString ("unknown") ;
                break ;
        }
        mxSetFieldByNumber (pargout [1], 0, 7, ord) ;

        mxSetFieldByNumber (pargout [1], 0, 8,
            mxCreateDoubleScalar ((double) cc->memory_usage)) ;
        mxSetFieldByNumber (pargout [1], 0, 9,
            mxCreateDoubleScalar (cc->SPQR_xstat [0])) ;
        mxSetFieldByNumber (pargout [1], 0, 10,
            mxCreateDoubleScalar (cc->SPQR_xstat [1])) ;

        int nthreads = cc->SPQR_nthreads ;
        if (nthreads <= 0)
        {
            mxSetFieldByNumber (pargout [1], 0, 11, mxCreateString ("default"));
        }
        else
        {
            mxSetFieldByNumber (pargout [1], 0, 11,
                mxCreateDoubleScalar ((double) nthreads)) ;
        }

#ifdef TIMING
        for (Int k = 1 ; k <= 3 ; k++)
        {
            mxSetFieldByNumber (pargout [1], 0, 11+k,
                mxCreateDoubleScalar (cc->other1 [k])) ;
        }
        mxSetFieldByNumber (pargout [1], 0, 15, mxCreateDoubleScalar (t)) ;
        mxSetFieldByNumber (pargout [1], 0, 16, mxCreateDoubleScalar (flops)) ;
#endif            
    }

    // -------------------------------------------------------------------------
    // warn if rank deficient
    // -------------------------------------------------------------------------

    rank = cc->SPQR_istat [4] ;
    if (rank < MIN (m,n))
    {
        // snprintf would be safer, but Windows is oblivious to safety ...
        // (Visual Studio C++ 2008 does not recognize snprintf!)
        sprintf (msg, "rank deficient. rank = %ld tol = %g\n", rank,
            cc->SPQR_xstat [1]) ;
        mexWarnMsgIdAndTxt ("MATLAB:rankDeficientMatrix", msg) ;
    }

    if (is_complex)
    {
        // free the merged complex copies of A and B
        cholmod_l_free (anz, sizeof (Complex), Ax, cc) ;
        cholmod_l_free (bnz, sizeof (Complex), Bx, cc) ;
    }

    cholmod_l_finish (cc) ;
    if (opts.spumoni > 0) spqr_mx_spumoni (&opts, is_complex, cc) ;
}
