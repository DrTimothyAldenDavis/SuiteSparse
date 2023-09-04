// =============================================================================
// === qrest_template.hpp ======================================================
// =============================================================================



// =============================================================================
// === SPQR_qmult ==============================================================
// =============================================================================

// wrapper for SuiteSparseQR_qmult (dense), optionally testing memory alloc.

template <typename Entry, typename Int> cholmod_dense *SPQR_qmult
(
    // arguments for SuiteSparseQR_qmult: 
    int method,
    cholmod_sparse *H,
    cholmod_dense *Tau,
    Int *HPinv,
    cholmod_dense *X,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_dense *Y = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        Y = SuiteSparseQR_qmult <Entry,Int> (method, H, Tau, HPinv, X, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            Y = SuiteSparseQR_qmult <Entry,Int> (method, H, Tau, HPinv, X, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (Y) ;
}


// =============================================================================
// === SPQR_qmult_sparse =======================================================
// =============================================================================

// wrapper for SuiteSparseQR_qmult (sparse), optionally testing memory alloc.

template <typename Entry, typename Int> cholmod_sparse *SPQR_qmult
(
    // arguments for SuiteSparseQR_qmult: 
    int method,
    cholmod_sparse *H,
    cholmod_dense *Tau,
    Int *HPinv,
    cholmod_sparse *X,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_sparse *Y = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        Y = SuiteSparseQR_qmult <Entry,Int> (method, H, Tau, HPinv, X, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            Y = SuiteSparseQR_qmult <Entry,Int> (method, H, Tau, HPinv, X, cc);
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (Y) ;
}

#ifndef NEXPERT

// =============================================================================
// === SPQR_qmult (dense case) =================================================
// =============================================================================

// wrapper for SuiteSparseQR_qmult (dense), optionally testing memory alloc.

template <typename Entry, typename Int> cholmod_dense *SPQR_qmult
(
    // arguments for SuiteSparseQR_qmult: 
    int method,
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_dense *X,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_dense *Y = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        Y = SuiteSparseQR_qmult <Entry,Int> (method, QR, X, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            Y = SuiteSparseQR_qmult <Entry,Int> (method, QR, X, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (Y) ;
}

// =============================================================================
// === SPQR_qmult (sparse case) ================================================
// =============================================================================

// wrapper for SuiteSparseQR_qmult (sparse), optionally testing memory alloc.

template <typename Entry, typename Int> cholmod_sparse *SPQR_qmult
(
    // arguments for SuiteSparseQR_qmult: 
    int method,
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_sparse *X,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_sparse *Y = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        Y = SuiteSparseQR_qmult <Entry,Int> (method, QR, X, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            Y = SuiteSparseQR_qmult <Entry,Int> (method, QR, X, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (Y) ;
}


// =============================================================================
// === SPQR_solve (dense case) =================================================
// =============================================================================

// Wrapper for testing SuiteSparseQR_solve

template <typename Entry, typename Int> cholmod_dense *SPQR_solve
(
    // arguments for SuiteSparseQR_solve: 
    int system,
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_dense *B,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_dense *X = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        X = SuiteSparseQR_solve (system, QR, B, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            X = SuiteSparseQR_solve (system, QR, B, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (X) ;
}


// =============================================================================
// === SPQR_solve (sparse case) ================================================
// =============================================================================

// Wrapper for testing SuiteSparseQR_solve

template <typename Entry, typename Int> cholmod_sparse *SPQR_solve
(
    // arguments for SuiteSparseQR_solve: 
    int system,
    SuiteSparseQR_factorization <Entry, Int> *QR,
    cholmod_sparse *B,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_sparse *X = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        X = SuiteSparseQR_solve (system, QR, B, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            X = SuiteSparseQR_solve (system, QR, B, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (X) ;
}


// =============================================================================
// === SPQR_min2norm (dense case) ==============================================
// =============================================================================

// Wrapper for testing SuiteSparseQR_min2norm

template <typename Entry, typename Int> cholmod_dense *SPQR_min2norm
(
    // arguments for SuiteSparseQR_min2norm: 
    int ordering,
    double tol,
    cholmod_sparse *A,
    cholmod_dense *B,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_dense *X = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        X = SuiteSparseQR_min2norm <Entry,Int> (ordering, tol, A, B, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            X = SuiteSparseQR_min2norm <Entry,Int> (ordering, tol, A, B, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (X) ;
}

// =============================================================================
// === SPQR_min2norm (sparse case) =============================================
// =============================================================================

// Wrapper for testing SuiteSparseQR_min2norm

template <typename Entry, typename Int> cholmod_sparse *SPQR_min2norm
(
    // arguments for SuiteSparseQR_min2norm: 
    int ordering,
    double tol,
    cholmod_sparse *A,
    cholmod_sparse *B,
    cholmod_common *cc,

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    cholmod_sparse *X = NULL ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        X = SuiteSparseQR_min2norm <Entry,Int> (ordering, tol, A, B, cc) ;
    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            X = SuiteSparseQR_min2norm <Entry,Int> (ordering, tol, A, B, cc) ;
            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }
    return (X) ;
}

// =============================================================================
// === SPQR_factorize ==========================================================
// =============================================================================

// Wrapper for testing SuiteSparseQR_factorize or
// SuiteSparseQR_symbolic and SuiteSparseQR_numeric

template <typename Entry, typename Int>
SuiteSparseQR_factorization <Entry, Int> *SPQR_factorize
(
    // arguments for SuiteSparseQR_factorize: 
    int ordering,
    double tol,
    cholmod_sparse *A,
    cholmod_common *cc,

    // method to use
    int split,              // if 1 use SuiteSparseQR_symbolic followed by
                            // SuiteSparseQR_numeric, if 0 use
                            // SuiteSparseQR_factorize, if 2, do the
                            // numeric factorization twice, just for testing.
                            // if 3 use SuiteSparseQR_C_factorize
                            // if 3 use SuiteSparseQR_C_symbolic / _C_numeric

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    SuiteSparseQR_factorization <Entry, Int> *QR ;
    SuiteSparseQR_C_factorization *C_QR ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing

        if (split == 4)
        {

            C_QR = SuiteSparseQR_C_symbolic (ordering, tol >= 0, A, cc) ;
            SuiteSparseQR_C_numeric (tol, A, C_QR, cc) ;
            if (C_QR == NULL)
            {
                cc->status = CHOLMOD_OUT_OF_MEMORY ;
                QR = NULL ;
            }
            else
            {
                QR = (SuiteSparseQR_factorization <Entry,Int> *)
                    (C_QR->factors) ;
                if (QR == NULL || QR->QRnum == NULL)
                {
                    cc->status = CHOLMOD_OUT_OF_MEMORY ;
                    QR = NULL  ;
                }
                else
                {
                    // QR itself will be kept; free the C wrapper
                    C_QR->factors = NULL ;
                    cc->status = CHOLMOD_OK ;
                    if (QR == NULL) printf ("Hey!\n") ;
                }
            }
            SuiteSparseQR_C_free (&C_QR, cc) ;

        }
        else if (split == 3)
        {

            C_QR = SuiteSparseQR_C_factorize (ordering, tol, A, cc) ;
            int save = cc->status ;
            if (C_QR == NULL)
            {
                QR = NULL ;
            }
            else
            {
                QR = (SuiteSparseQR_factorization <Entry,Int> *)
                    (C_QR->factors) ;
                C_QR->factors = NULL ;
                SuiteSparseQR_C_free (&C_QR, cc) ;
            }
            cc->status = save ;

        }
        else if (split == 2)
        {
            QR = SuiteSparseQR_symbolic <Entry,Int> (ordering,
                tol >= 0 || tol <= SPQR_DEFAULT_TOL, A, cc) ;
            ASSERT (QR != NULL) ;
            SuiteSparseQR_numeric <Entry,Int> (tol, A, QR, cc) ;
            // just for testing
            SuiteSparseQR_numeric <Entry,Int> (tol, A, QR, cc) ;
        }
        else if (split == 1)
        {
            // split symbolic/numeric, no singletons exploited
            QR = SuiteSparseQR_symbolic <Entry,Int> (ordering,
                tol >= 0 || tol <= SPQR_DEFAULT_TOL, A, cc) ;
            ASSERT (QR != NULL) ;
            SuiteSparseQR_numeric <Entry,Int> (tol, A, QR, cc) ;
        }
        else
        {
            QR = SuiteSparseQR_factorize <Entry,Int> (ordering, tol, A, cc) ;
        }

    }
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;

            if (split == 4)
            {

                C_QR = SuiteSparseQR_C_symbolic (ordering, tol >= 0, A, cc) ;
                SuiteSparseQR_C_numeric (tol, A, C_QR, cc) ;
                if (C_QR == NULL)
                {
                    cc->status = CHOLMOD_OUT_OF_MEMORY ;
                    QR = NULL ;
                }
                else
                {
                    QR = (SuiteSparseQR_factorization <Entry,Int> *)
                        (C_QR->factors) ;
                    if (QR == NULL || QR->QRnum == NULL)
                    {
                        cc->status = CHOLMOD_OUT_OF_MEMORY ;
                        QR = NULL  ;
                    }
                    else
                    {
                        // QR itself will be kept; free the C wrapper
                        C_QR->factors = NULL ;
                        cc->status = CHOLMOD_OK ;
                        if (QR == NULL) printf ("Hey!!\n") ;
                    }
                }
                SuiteSparseQR_C_free (&C_QR, cc) ;


            }
            else if (split == 3)
            {

                C_QR = SuiteSparseQR_C_factorize (ordering, tol, A, cc) ;
                int save = cc->status ;
                if (C_QR == NULL)
                {
                    QR = NULL ;
                }
                else
                {
                    QR = (SuiteSparseQR_factorization <Entry,Int> *)
                        (C_QR->factors) ;
                    C_QR->factors = NULL ;
                    SuiteSparseQR_C_free (&C_QR, cc) ;
                }
                cc->status = save ;

            }
            else if (split == 1 || split == 2)
            {
                // split symbolic/numeric, no singletons exploited
                QR = SuiteSparseQR_symbolic <Entry,Int> (ordering,
                    tol >= 0 || tol <= SPQR_DEFAULT_TOL, A, cc) ;
                if (cc->status < CHOLMOD_OK)
                {
                    continue ;
                }
                ASSERT (QR != NULL) ;
                SuiteSparseQR_numeric <Entry,Int> (tol, A, QR, cc) ;
                if (cc->status < CHOLMOD_OK)
                {
                    SuiteSparseQR_free <Entry,Int> (&QR, cc) ;
                }
            }
            else
            {
                QR = SuiteSparseQR_factorize <Entry,Int> (ordering, tol, A, cc) ;
            }

            if (cc->status == CHOLMOD_OK) break ;
        }
        normal_memory_handler (cc, true) ;
    }

    if (QR == NULL) printf ("huh?? split: %d ordering %d tol %g\n", split,
        ordering, tol) ;
    return (QR) ;
}


#endif

// =============================================================================
// === SPQR_qr =================================================================
// =============================================================================

// wrapper for SuiteSparseQR, optionally testing memory allocation

template <typename Entry, typename Int> Int SPQR_qr
(
    // arguments for SuiteSparseQR: 
    int ordering,
    double tol,
    Int econ,
    int getCTX,
    cholmod_sparse *A,
    cholmod_sparse *Bsparse,
    cholmod_dense *Bdense,
    cholmod_sparse **Zsparse,
    cholmod_dense  **Zdense,
    cholmod_sparse **R,
    Int **E,
    cholmod_sparse **H,
    Int **HPinv,
    cholmod_dense **HTau,
    cholmod_common *cc,

    // which version to use (C or C++)
    int use_c_version,      // if TRUE use C version, otherwise use C++

    // malloc control
    int memory_test,        // if TRUE, test malloc error handling
    int memory_punt         // if TRUE, test punt case
)
{
    Int rank ;
    if (!memory_test)
    {
        // just call the method directly; no memory testing
        if (use_c_version)
        {
            if (sizeof (Int) == sizeof (int64_t))
            {
                rank = SuiteSparseQR_C (ordering, tol, econ, getCTX, A,
                    Bsparse, Bdense, Zsparse, Zdense, R,
                    (int64_t **) E, H, (int64_t **) HPinv, HTau, cc) ;
            }
            else
            {
                rank = SuiteSparseQR_i_C (ordering, tol, econ, getCTX, A,
                    Bsparse, Bdense, Zsparse, Zdense, R,
                    (int32_t **) E, H, (int32_t **) HPinv, HTau, cc) ;
            }
        }
        else
        {
            rank = SuiteSparseQR <Entry, Int> (ordering, tol, econ, getCTX, A,
                Bsparse, Bdense, Zsparse, Zdense, R, E, H, HPinv, HTau, cc) ;
        }
    } 
    else
    {
        // test malloc error handling
        int64_t tries ;
        test_memory_handler (cc, true) ;
        my_punt = memory_punt ;
        for (tries = 0 ; my_tries < 0 ; tries++)
        {
            my_tries = tries ;
            if (use_c_version)
            {
                if (sizeof (Int) == sizeof (int64_t))
                {
                    rank = SuiteSparseQR_C (ordering, tol, econ, getCTX, A,
                        Bsparse, Bdense, Zsparse, Zdense, R,
                        (int64_t **) E, H, (int64_t **) HPinv, HTau, cc) ;
                }
                else
                {
                    rank = SuiteSparseQR_i_C (ordering, tol, econ, getCTX, A,
                        Bsparse, Bdense, Zsparse, Zdense, R,
                        (int32_t **) E, H, (int32_t **) HPinv, HTau, cc) ;
                }
            }
            else
            {
                rank = SuiteSparseQR <Entry, Int> (ordering, tol, econ,
                    getCTX, A, Bsparse, Bdense, Zsparse, Zdense, R,
                    E, H, HPinv, HTau, cc);
            }
            if (cc->status == CHOLMOD_OK)
            {
                break ;
            }
        }
        normal_memory_handler (cc, true) ;
    }
    return (rank) ;
}


// =============================================================================
// === my_rand =================================================================
// =============================================================================

// The POSIX example of rand, duplicated here so that the same sequence will
// be generated on different machines.

static unsigned long next = 1 ;

#define MY_RAND_MAX 32767

// RAND_MAX assumed to be 32767
int64_t my_rand (void)
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

int64_t nrand (int64_t n)       // return a random int64_t between 0 and n-1
{
    return ((n <= 0) ? 0 : (my_rand ( ) % n)) ;
}

double xrand ( )        // return a random double between -1 and 1
{
    double x = ((double) my_rand ( )) / MY_RAND_MAX ;
    return (2*x-1) ;
}

double erand (double range)
{
    return (range * xrand ( )) ;
}

Complex erand (Complex range)
{
    /*
    Complex x ;
    x.real ( ) = xrand ( ) ;
    x.imag ( ) = xrand ( ) ;
    return (range * x) ;
    */
    Complex i = Complex (0,1) ;
    return (range * (xrand ( ) + i * xrand ( ))) ;
}


// =============================================================================
// === getreal =================================================================
// =============================================================================

// return real part of a scalar x

inline double getreal (double x)
{
    return (x) ;
}

inline double getreal (Complex x)
{
    return (x.real ( )) ;
}

// =============================================================================
// === getimag =================================================================
// =============================================================================

// return imaginary part of a scalar x
inline double getimag (double x)        // x is an unused parameter
{
    return (0) ;
}

inline double getimag (Complex x)
{
    return (x.imag ( )) ;
}


// =============================================================================
// === dense_wrapper ===========================================================
// =============================================================================

// place a column-oriented matrix in a cholmod_dense wrapper

template <typename Entry, typename Int> cholmod_dense *dense_wrapper
(
    cholmod_dense *X,
    Int nrow,
    Int ncol,
    Entry *Xx
)
{
    X->xtype = spqr_type <Entry> ( ) ;
    X->nrow = nrow ;
    X->ncol = ncol ;
    X->d = nrow ;           // leading dimension = nrow
    X->nzmax = nrow * ncol ;
    X->x = Xx ;
    X->z = NULL ;           // ZOMPLEX case not supported
    X->dtype = CHOLMOD_DOUBLE ;
    return (X) ;
}

// =============================================================================
// === sparse_split ============================================================
// =============================================================================

// Return the real or imaginary part of a packed complex sparse matrix

template <typename Entry, typename Int>
cholmod_sparse *sparse_split
(
    cholmod_sparse *A,
    int part,
    cholmod_common *cc
)
{
    cholmod_sparse *C ;
    if (!A || A->xtype != CHOLMOD_COMPLEX || A->nz != NULL) return (NULL) ;
    if (! (part == 0 || part == 1)) return (NULL) ;

    int64_t nz = spqr_nnz <Int> (A, cc) ;
    C = spqr_allocate_sparse <Int> (A->nrow, A->ncol, nz, TRUE, TRUE, 0,
        CHOLMOD_REAL, cc) ;

    Int *Ap = (Int *) A->p ;
    Int *Ai = (Int *) A->i ;
    double *Ax = (double *) A->x ;

    Int *Cp = (Int *) C->p ;
    Int *Ci = (Int *) C->i ;
    double *Cx = (double *) C->x ;

    Int n = A->ncol ;

    for (Int k = 0 ; k < n+1 ; k++)
    {
        Cp [k] = Ap [k] ;
    }

    for (int64_t k = 0 ; k < nz ; k++)
    {
        Ci [k] = Ai [k] ;
    }

    for (Int k = 0 ; k < nz ; k++)
    {
        Cx [k] = Ax [2*k + part] ;
    }

    return (C) ;
}

template <typename Entry, typename Int>
cholmod_sparse *sparse_real (cholmod_sparse *A, cholmod_common *cc)
{
    return (sparse_split <Entry,Int> (A, 0, cc)) ;
}

template <typename Entry, typename Int>
cholmod_sparse *sparse_imag (cholmod_sparse *A, cholmod_common *cc)
{
    return (sparse_split <Entry,Int> (A, 1, cc)) ;
}

// =============================================================================
// === sparse_merge ============================================================
// =============================================================================

// Add the real and imaginary parts of a matrix (both stored in real form)
// into a single matrix.  The two parts must have the same nonzero pattern.
// A is CHOLMOD_REAL on input and holds the real part, it is CHOLMOD_COMPLEX
// on output.  A_imag is CHOLMOD_REAL on input; it holds the imaginary part
// of A as a real matrix.

template <typename Entry, typename Int> int sparse_merge
(
    cholmod_sparse *A,          // input/output
    cholmod_sparse *A_imag,     // input only
    cholmod_common *cc
)
{
    if (A == NULL || A_imag == NULL)
    {
        return (FALSE) ;
    }
    int64_t nz1 = spqr_nnz <Int> (A, cc) ;
    int64_t nz2 = spqr_nnz <Int> (A_imag, cc) ;
    if (A->xtype != CHOLMOD_REAL || A_imag->xtype != CHOLMOD_REAL || nz1 != nz2)
    {
        return (FALSE) ;
    }

    // change A from real to complex
    spqr_sparse_xtype <Int> (CHOLMOD_COMPLEX, A, cc) ;

    double *Ax = (double *) A->x ;
    double *Az = (double *) A_imag->x ;

    // merge in the imaginary part from A_imag into A
    for (int64_t k = 0 ; k < nz1 ; k++)
    {
        Ax [2*k+1] = Az [k] ;
    }
    return (TRUE) ;
}

// =============================================================================
// === sparse_diff =============================================================
// =============================================================================

// Compute C = A-B where A and B are either both real, or both complex

template <typename Entry, typename Int> cholmod_sparse *sparse_diff
(
    cholmod_sparse *A,
    cholmod_sparse *B,
    cholmod_common *cc
)
{
    cholmod_sparse *C ;
    double one [2] = {1,0}, minusone [2] = {-1,0} ;

    if (spqr_type <Entry> ( ) == CHOLMOD_REAL)
    {
        // C = A - B
        C = spqr_ssadd <Int> (A, B, one, minusone, TRUE, TRUE, cc) ;
    }
    else
    {
        cholmod_sparse *A_real, *A_imag, *B_real, *B_imag, *C_imag ;

        A_real = sparse_real <Entry,Int> (A, cc) ;
        A_imag = sparse_imag <Entry,Int> (A, cc) ;

        B_real = sparse_real <Entry,Int> (B, cc) ;
        B_imag = sparse_imag <Entry,Int> (B, cc) ;

        // real(C) = real(A) - real (B)
        C = spqr_ssadd <Int> (A_real, B_real, one, minusone, TRUE, TRUE, cc) ;

        // imag(C) = imag(A) - imag (B)
        C_imag = spqr_ssadd <Int> (A_imag, B_imag, one, minusone, TRUE, TRUE, cc);

        // C = real(C) + 1i*imag(C)
        sparse_merge <Entry,Int> (C, C_imag, cc) ;
        spqr_free_sparse <Int> (&C_imag, cc) ;

        spqr_free_sparse <Int> (&A_real, cc) ;
        spqr_free_sparse <Int> (&A_imag, cc) ;
        spqr_free_sparse <Int> (&B_real, cc) ;
        spqr_free_sparse <Int> (&B_imag, cc) ;
    }

    return (C) ;
}


// =============================================================================
// === permute_columns =========================================================
// =============================================================================

// compute A(:,P)

template <typename Entry, typename Int> cholmod_sparse *permute_columns
(
    cholmod_sparse *A,
    Int *P,
    cholmod_common *cc
) 
{
    Int m = A->nrow ;
    Int n = A->ncol ;
    Int nz = spqr_nnz <Int> (A, cc) ;
    int xtype = spqr_type <Entry> ( ) ;
    Int *Ap = (Int *) A->p ;
    Int *Ai = (Int *) A->i ;
    Entry *Ax = (Entry *) A->x ;
    cholmod_sparse *C ;

    // allocate empty matrix C with space for nz entries
    C = spqr_allocate_sparse <Int> (m, n, nz, TRUE, TRUE, 0, xtype, cc) ;
    Int *Cp = (Int *) C->p ;
    Int *Ci = (Int *) C->i ;
    Entry *Cx = (Entry *) C->x ;

    // construct column pointers for C
    for (Int k = 0 ; k < n ; k++)
    {
        // column j of A becomes column k of C
        Int j = P ? P [k] : k ;
        Cp [k] = Ap [j+1] - Ap [j] ;
    }
    spqr_cumsum (n, Cp) ;

    // copy columns from A to C
    for (Int k = 0 ; k < n ; k++)
    {
        // copy column k of A into column j of C
        Int j = P ? P [k] : k ;
        Int pdest = Cp [k] ;
        Int psrc = Ap [j] ;
        Int len = Ap [j+1] - Ap [j] ;
        for (Int t = 0 ; t < len ; t++)
        {
            Ci [pdest + t] = Ai [psrc + t] ;
            Cx [pdest + t] = Ax [psrc + t] ;
        }
    }

    return (C) ;
}


// =============================================================================
// === sparse_multiply =========================================================
// =============================================================================

// compute A*B where A and B can be both real or both complex

template <typename Entry, typename Int> cholmod_sparse *sparse_multiply
(
    cholmod_sparse *A,
    cholmod_sparse *B,
    cholmod_common *cc
)
{
    cholmod_sparse *C ;

    if (spqr_type <Entry> ( ) == CHOLMOD_REAL)
    {
        // C = A*B
        C = spqr_ssmult <Int> (A, B, 0, TRUE, TRUE, cc) ;
    }
    else
    {
        // cholmod_ssmult and cholmod_add only work for real matrices
        cholmod_sparse *A_real, *A_imag, *B_real, *B_imag, *C_imag, *T1, *T2 ;
        double one [2] = {1,0}, minusone [2] = {-1,0} ;

        A_real = sparse_real <Entry,Int> (A, cc) ;
        A_imag = sparse_imag <Entry,Int> (A, cc) ;

        B_real = sparse_real <Entry,Int> (B, cc) ;
        B_imag = sparse_imag <Entry,Int> (B, cc) ;

        // real(C) = real(A)*real(B) - imag(A)*imag(B)
        T1 = spqr_ssmult <Int> (A_real, B_real, 0, TRUE, TRUE, cc) ;
        T2 = spqr_ssmult <Int> (A_imag, B_imag, 0, TRUE, TRUE, cc) ;
        C = spqr_ssadd <Int> (T1, T2, one, minusone, TRUE, TRUE, cc) ;
        spqr_free_sparse <Int> (&T1, cc) ;
        spqr_free_sparse <Int> (&T2, cc) ;

        // imag(C) = imag(A)*real(B) + real(A)*imag(B)
        T1 = spqr_ssmult <Int> (A_imag, B_real, 0, TRUE, TRUE, cc) ;
        T2 = spqr_ssmult <Int> (A_real, B_imag, 0, TRUE, TRUE, cc) ;
        C_imag = spqr_ssadd <Int> (T1, T2, one, one, TRUE, TRUE, cc) ;
        spqr_free_sparse <Int> (&T1, cc) ;
        spqr_free_sparse <Int> (&T2, cc) ;

        // C = real(C) + 1i*imag(C)
        sparse_merge <Entry,Int> (C, C_imag, cc) ;
        spqr_free_sparse <Int> (&C_imag, cc) ;

        spqr_free_sparse <Int> (&A_real, cc) ;
        spqr_free_sparse <Int> (&A_imag, cc) ;
        spqr_free_sparse <Int> (&B_real, cc) ;
        spqr_free_sparse <Int> (&B_imag, cc) ;
    }

    return (C) ;
}


// =============================================================================
// === sparse_resid ============================================================
// =============================================================================

// compute norm (A*x-b,1) for A,x, and b all sparse

template <typename Entry, typename Int> double sparse_resid
(
    cholmod_sparse *A,
    double anorm,
    cholmod_sparse *X,
    cholmod_sparse *B,
    cholmod_common *cc
)
{
    cholmod_sparse *AX, *Resid ;
    // AX = A*X
    AX = sparse_multiply <Entry,Int> (A, X, cc) ;
    // Resid = AX - B
    Resid = sparse_diff <Entry,Int> (AX, B, cc) ;
    // resid = norm (Resid,1)
    double resid = spqr_norm_sparse <Int> (Resid, 1, cc) ;
    resid = CHECK_NAN (resid) ;
    spqr_free_sparse <Int> (&AX, cc) ;
    spqr_free_sparse <Int> (&Resid, cc) ;
    return (CHECK_NAN (resid / anorm)) ;
}

// =============================================================================
// === dense_resid =============================================================
// =============================================================================

// compute norm (A*x-b,1) for A sparse, x and b dense

template <typename Entry, typename Int> double dense_resid
(
    cholmod_sparse *A,
    double anorm,
    cholmod_dense *X,
    Int nb,
    Entry *Bx,
    cholmod_common *cc
)
{
    cholmod_dense *B, Bmatrix, *Resid ;
    double one [2] = {1,0}, minusone [2] = {-1,0} ;

    B = dense_wrapper <Entry,Int> (&Bmatrix, A->nrow, nb, Bx) ;
    // Resid = B
    Resid = spqr_copy_dense <Int> (B, cc) ;

    // Resid = A*X - Resid
    spqr_sdmult <Int> (A, FALSE, one, minusone, X, Resid, cc) ;

    // resid = norm (Resid,1)
    double resid = spqr_norm_dense <Int> (Resid, 1, cc) ;
    resid = CHECK_NAN (resid) ;
    spqr_free_dense <Int> (&Resid, cc) ;
    return (CHECK_NAN (resid / anorm)) ;
}

// =============================================================================
// === check_r_factor ==========================================================
// =============================================================================

// compute norm (R'*R - (A(:,P))'*(A(:,P)), 1) / norm (A'*A,1)

template <typename Entry, typename Int> double check_r_factor
(
    cholmod_sparse *R,
    cholmod_sparse *A,
    Int *P,
    cholmod_common *cc
)
{
    cholmod_sparse *RTR, *RT, *C, *CT, *CTC, *D ;

    // RTR = R'*R
    RT = spqr_transpose <Int> (R, 2, cc) ;
    RTR = sparse_multiply <Entry,Int> (RT, R, cc) ;
    spqr_free_sparse <Int> (&RT, cc) ;

    // C = A(:,P)
    C = permute_columns<Entry,Int> (A, P, cc) ;

    // CTC = C'*C
    CT = spqr_transpose <Int> (C, 2, cc) ;
    CTC = sparse_multiply <Entry,Int> (CT, C, cc) ;
    spqr_free_sparse <Int> (&CT, cc) ;
    spqr_free_sparse <Int> (&C, cc) ;

    double ctcnorm = spqr_norm_sparse <Int> (CTC, 1, cc) ;
    if (ctcnorm == 0) ctcnorm = 1 ;
    ctcnorm = CHECK_NAN (ctcnorm) ;

    // D = RTR - CTC
    D = sparse_diff <Entry,Int> (RTR, CTC, cc) ;
    spqr_free_sparse <Int> (&CTC, cc) ;
    spqr_free_sparse <Int> (&RTR, cc) ;

    // err = norm (D,1)
    double err = spqr_norm_sparse <Int> (D, 1, cc) / ctcnorm ;
    err = CHECK_NAN (err) ;

    spqr_free_sparse <Int> (&D, cc) ;
    return (CHECK_NAN (err)) ;
}


// =============================================================================
// === check_qr ================================================================
// =============================================================================

// compute norm (Q*R - A(:,P)) / norm (A)

template <typename Entry, typename Int> double check_qr
(
    cholmod_sparse *Q,
    cholmod_sparse *R,
    cholmod_sparse *A,
    Int *P,
    double anorm,
    cholmod_common *cc
)
{
    cholmod_sparse *QR, *C, *D ;

    // C = A(:,P)
    C = permute_columns<Entry,Int> (A, P, cc) ;

    // QR = Q*R
    QR = sparse_multiply <Entry,Int> (Q, R, cc) ;

    // D = RTR - CTC
    D = sparse_diff <Entry,Int> (QR, C, cc) ;
    spqr_free_sparse <Int> (&QR, cc) ;
    spqr_free_sparse <Int> (&C, cc) ;

    // err = norm (D,1)
    double err = spqr_norm_sparse <Int> (D, 1, cc) / anorm ;
    err = CHECK_NAN (err) ;

    spqr_free_sparse <Int> (&D, cc) ;
    return (CHECK_NAN (err)) ;
}

// =============================================================================
// === Rsolve ==================================================================
// =============================================================================

template <typename Entry, typename Int> int Rsolve
(
    // R is n-by-n, upper triangular with zero-free diagonal
    Int n,
    cholmod_sparse *R,
    Entry *X,       // X is n-by-nx, leading dimension n, overwritten with soln
    Int nx,
    cholmod_common *cc
)
{
    // Int n = R->n ;
    Int *Rp = (Int *) R->p ; 
    Int *Ri = (Int *) R->i ; 
    Entry *Rx = (Entry *) R->x ; 

    // check the diagonal
    for (Int j = 0 ; j < n ; j++)
    {
        if (Rp [j] == Rp [j+1] || Ri [Rp [j+1]-1] != j)
        {
            printf ("Rsolve: R not upper triangular w/ zero-free diagonal\n") ;
            return (FALSE) ;
        }
    }

    // do the backsolve
    for (Int k = 0 ; k < nx ; k++)
    {
        for (Int j = n-1 ; j >= 0 ; j--)
        {
            Entry rjj = Rx [Rp [j+1]-1] ;
            if (rjj == (Entry) 0)
            {
                printf ("Rsolve: R has an explicit zero on the diagonal\n") ;
                return (FALSE) ;
            }
            X [j] /= rjj ;
            for (Int p = Rp [j] ; p < Rp [j+1]-1 ; p++)
            {
                X [Ri [p]] -= Rx [p] * X [j] ;
            }
        }
        X += n ;
    }

    return (TRUE) ;
}


// =============================================================================
// === create_Q ================================================================
// =============================================================================

// create a sparse Q
template <typename Entry, typename Int> cholmod_sparse *create_Q
(
    cholmod_sparse *H,
    cholmod_dense *HTau,
    Int *HPinv,
    cholmod_common *cc
)
{
    cholmod_sparse *Q, *I ;
    Int m = H->nrow ;
    int xtype = spqr_type <Entry> ( ) ;
    I = spqr_speye <Int> (m, m, xtype, cc) ;
    Q = SPQR_qmult <Entry,Int> (1, H, HTau, HPinv, I, cc, m<300, nrand (2)) ;
    spqr_free_sparse <Int> (&I, cc) ;
    return (Q) ;
}


// =============================================================================
// === QRsolve =================================================================
// =============================================================================

// solve Ax=b using H, R, and E

template <typename Entry, typename Int> double QRsolve
(
    cholmod_sparse *A,
    double anorm,
    Int rank,
    int method,
    cholmod_sparse *H,
    cholmod_dense *HTau,
    Int *HPinv,
    cholmod_sparse *R,
    Int *Qfill,
    cholmod_dense *Bdense,
    cholmod_common *cc
)
{
    double one [2] = {1,0}, zero [2] = {0,0}, resid = EMPTY ;
    int xtype = spqr_type <Entry> ( ) ;
    Int n = A->ncol ;
    Int m = A->nrow ;
    Int nrhs = Bdense->ncol ;
    Entry *X, *Y = NULL, *B ;
    cholmod_dense *Ydense = NULL ;
    cholmod_dense *Xdense ;

    B = (Entry *) Bdense->x ;
    Xdense = spqr_zeros <Int> (n, nrhs, xtype, cc) ;
    X = (Entry *) Xdense->x ;

    if (method == 0)
    {
        // solve Ax=b using H, R, and E
        // Y = Q'*B
        Ydense = SPQR_qmult <Entry,Int> (0, H, HTau, HPinv, Bdense, cc,
            m < 300, nrand (2)) ;
    }
    else
    {
        cholmod_sparse *Q, *QT ;
        // solve using Q instead of qmult
        Q = create_Q <Entry,Int> (H, HTau, HPinv, cc) ;
        QT = spqr_transpose <Int> (Q, 2, cc) ;
        // Y = Q'*B
        Ydense = spqr_zeros <Int> (m, nrhs, xtype, cc) ;
        spqr_sdmult <Int> (QT, FALSE, one, zero, Bdense, Ydense, cc) ;
        spqr_free_sparse <Int> (&Q, cc) ;
        spqr_free_sparse <Int> (&QT, cc) ;
    }

    // Y (1:rank) = R (1:rank,1:rank) \ Y (1:rank)
    Y = (Entry *) Ydense->x ;
    Int ok = Rsolve <Entry,Int> (rank, R, Y, nrhs, cc) ;
    // X = E*Y
    if (ok)
    {
        for (Int kk = 0 ; kk < nrhs ; kk++)
        {
            for (Int k = 0 ; k < rank ; k++)
            {
                Int j = Qfill ? Qfill [k] : k ;
                X [j] = Y [k] ;
            }
            X += n ;
            Y += m ;
        }
        // check norm (A*x-b), x and b dense
        resid = dense_resid <Entry,Int> (A, anorm, Xdense, nrhs, B, cc) ;
    }

    spqr_free_dense <Int> (&Ydense, cc) ;
    spqr_free_dense <Int> (&Xdense, cc) ;
    return (CHECK_NAN (resid)) ;
}

// =============================================================================
// === check_qmult =============================================================
// =============================================================================

// Test qmult

template <typename Entry, typename Int> double check_qmult
(
    cholmod_sparse *H,
    cholmod_dense *HTau,
    Int *HPinv,
    int test_errors,
    cholmod_common *cc
)
{
    cholmod_sparse *Q, *QT, *Xsparse, *Ssparse, *Zsparse ;
    cholmod_dense *Xdense, *Zdense, *Sdense, *Ydense ;
    int xtype = spqr_type <Entry> ( ) ;
    Entry *X, *Y, *Z, *S ;
    double err, maxerr = 0 ;
    double one [2] = {1,0}, zero [2] = {0,0} ;
    Entry range = (Entry) 1.0 ;
    Int k ;

    Int m = H->nrow ;
    Q = create_Q <Entry,Int> (H, HTau, HPinv, cc) ;     // construct Q from H

    QT = spqr_transpose <Int> (Q, 2, cc) ;           // QT = Q'

    // compare Q with qmult for sparse and dense X
    for (Int nx = 0 ; nx < 5 ; nx++)                // # of columns of X
    {
        Int xsize = m * nx ;                        // size of X
        for (Int nz = 1 ; nz <= xsize+1 ; nz *= 16) // # of nonzeros in X
        {

            // -----------------------------------------------------------------
            // create X as m-by-nx, both sparse and dense
            // -----------------------------------------------------------------

            Xdense = spqr_zeros <Int> (m, nx, xtype, cc) ;
            X = (Entry *) Xdense->x ;
            for (k = 0 ; k < nz ; k++)
            {
                X [nrand (xsize)] += erand (range) ;
            }
            Xsparse = spqr_dense_to_sparse <Int> (Xdense, TRUE, cc) ;

            // -----------------------------------------------------------------
            // Y = Q'*X for method 0, Y = Q*X for method 1
            // -----------------------------------------------------------------

            for (int method = 0 ; method <= 1 ; method++)
            {
                // Y = Q'*X or Q*X
                Ydense = SPQR_qmult <Entry,Int> (method, H, HTau, HPinv,
                    Xdense, cc, m < 300, nrand (2)) ;
                Y = (Entry *) Ydense->x ;

                Zdense = spqr_zeros <Int> (m, nx, xtype, cc) ;
                spqr_sdmult <Int> (Q, (method == 0), one, zero,
                    Xdense, Zdense, cc) ;
                Z = (Entry *) Zdense->x ;

                for (err = 0, k = 0 ; k < xsize ; k++)
                {
                    double e1 = spqr_abs (Y [k] - Z [k], cc) ;
                    e1 = CHECK_NAN (e1) ;
                    err = MAX (err, e1) ;
                }
                maxerr = MAX (maxerr, err) ;

                // S = Q'*Xsparse or Q*Xsparse
                Ssparse = SPQR_qmult <Entry,Int> (
                    method, H, HTau, HPinv, Xsparse, cc, m < 300, nrand (2)) ;
                Sdense = spqr_sparse_to_dense <Int> (Ssparse, cc) ;
                S = (Entry *) Sdense->x ;

                for (err = 0, k = 0 ; k < xsize ; k++)
                {
                    double e1 = spqr_abs (S [k] - Y [k], cc) ;
                    e1 = CHECK_NAN (e1) ;
                    err = MAX (err, e1) ;
                }
                maxerr = MAX (maxerr, err) ;

                spqr_free_dense <Int> (&Ydense, cc) ;
                spqr_free_dense <Int> (&Zdense, cc) ;
                spqr_free_sparse <Int> (&Ssparse, cc) ;
                spqr_free_dense <Int> (&Sdense, cc) ;
            }

            spqr_free_dense <Int> (&Xdense, cc) ;
            spqr_free_sparse <Int> (&Xsparse, cc) ;

            // -----------------------------------------------------------------
            // create X as nx-by-m, both sparse and dense
            // -----------------------------------------------------------------

            Xdense = spqr_zeros <Int> (nx, m, xtype, cc) ;
            X = (Entry *) Xdense->x ;
            for (k = 0 ; k < nz ; k++)
            {
                X [nrand (xsize)] += erand (range) ;
            }
            Xsparse = spqr_dense_to_sparse <Int> (Xdense, TRUE, cc) ;

            // -----------------------------------------------------------------
            // Y = X*Q' for method 2, Y = X*Q for method 3
            // -----------------------------------------------------------------

            for (int method = 2 ; method <= 3 ; method++)
            {
                // Y = X*Q' or X*Q
                Ydense = SPQR_qmult <Entry,Int> (method, H, HTau, HPinv,
                    Xdense, cc, m < 300, nrand (2)) ;
                Y = (Entry *) Ydense->x ;

                if (method == 2)
                {
                    // Zsparse = (X*Q')
                    Zsparse = sparse_multiply <Entry,Int> (Xsparse, QT, cc) ;
                }
                else
                {
                    // Zsparse = (X*Q)
                    Zsparse = sparse_multiply <Entry,Int> (Xsparse, Q, cc) ;
                }
                Zdense = spqr_sparse_to_dense <Int> (Zsparse, cc) ;

                Z = (Entry *) Zdense->x ;

                for (err = 0, k = 0 ; k < xsize ; k++)
                {
                    double e1 = spqr_abs (Y [k] - Z [k], cc) ;
                    e1 = CHECK_NAN (e1) ;
                    err = MAX (err, e1) ;
                }
                maxerr = MAX (maxerr, err) ;

                // S = X*Q' or X*Q
                Ssparse = SPQR_qmult <Entry,Int> (
                    method, H, HTau, HPinv, Xsparse, cc, m < 300, nrand (2)) ;
                Sdense = spqr_sparse_to_dense <Int> (Ssparse, cc) ;
                S = (Entry *) Sdense->x ;

                for (err = 0, k = 0 ; k < xsize ; k++)
                {
                    double e1 = spqr_abs (S [k] - Y [k], cc) ;
                    e1 = CHECK_NAN (e1) ;
                    err = MAX (err, e1) ;
                }
                maxerr = MAX (maxerr, err) ;

                spqr_free_dense <Int> (&Ydense, cc) ;
                spqr_free_dense <Int> (&Zdense, cc) ;
                spqr_free_sparse <Int> (&Ssparse, cc) ;
                spqr_free_sparse <Int> (&Zsparse, cc) ;
                spqr_free_dense <Int> (&Sdense, cc) ;
            }

            spqr_free_dense <Int> (&Xdense, cc) ;
            spqr_free_sparse <Int> (&Xsparse, cc) ;

        }
    }
    spqr_free_sparse <Int> (&Q, cc) ;
    spqr_free_sparse <Int> (&QT, cc) ;

    // -------------------------------------------------------------------------
    // qmult error conditions
    // -------------------------------------------------------------------------

    // These should fail; expect 6 error messages in the output

    if (test_errors)
    {
        err = 0 ;
        printf ("The following six errors are expected:\n") ;
        Xdense = spqr_zeros <Int> (m+1, 1, xtype,cc) ;
        Ydense = SuiteSparseQR_qmult <Entry,Int> (0, H, HTau, HPinv, Xdense, cc) ;
        if (Ydense != NULL || cc->status != CHOLMOD_INVALID) err++ ;
        spqr_free_dense <Int> (&Xdense, cc) ;

        Xdense = spqr_zeros <Int> (1, m+1, xtype,cc) ;
        Ydense = SuiteSparseQR_qmult <Entry,Int> (2, H, HTau, HPinv, Xdense, cc) ;
        if (Ydense != NULL || cc->status != CHOLMOD_INVALID) err++ ;
        Ydense = SuiteSparseQR_qmult <Entry,Int> (42, H, HTau, HPinv, Xdense, cc) ;
        if (Ydense != NULL || cc->status != CHOLMOD_INVALID) err++ ;
        spqr_free_dense <Int> (&Xdense, cc) ;

        Xsparse = spqr_speye <Int> (m+1, m+1, xtype, cc) ;

        Q = SuiteSparseQR_qmult <Entry,Int> (0, H, HTau, HPinv, Xsparse, cc);
        if (Q != NULL || cc->status != CHOLMOD_INVALID) err++ ;
        Q = SuiteSparseQR_qmult <Entry,Int> (2, H, HTau, HPinv, Xsparse, cc);
        if (Q != NULL || cc->status != CHOLMOD_INVALID) err++ ;
        Q = SuiteSparseQR_qmult <Entry,Int> (9, H, HTau, HPinv, Xsparse, cc);
        if (Q != NULL || cc->status != CHOLMOD_INVALID) err++ ;

        printf (" ... error handling done\n\n") ;
        spqr_free_sparse <Int> (&Xsparse, cc) ;

        maxerr = MAX (maxerr, err) ;
    }

    return (CHECK_NAN (maxerr)) ;
}


// =============================================================================
// === check_rc ================================================================
// =============================================================================

// X = Q'*B has been done, continue with C=R\C and 

template <typename Entry, typename Int> double check_rc
(
    Int rank,
    cholmod_sparse *R,
    cholmod_sparse *A,
    Entry *B,
    cholmod_dense *X,
    Int nrhs,
    double anorm,
    Int *Qfill,
    cholmod_common *cc
)
{
    double resid = EMPTY ;
    int xtype = spqr_type <Entry> ( ) ;
    cholmod_dense *W ;
    Int n ;
    int ok ;
    Entry *W1, *X1 ;
    if (!R || !X)
    {
        ok = 0 ;
    }
    else
    {
        n = X->nrow ;
        X1 = (Entry *) X->x ;
        // solve X = R\X, overwriting X with solution
        ok = Rsolve <Entry,Int> (rank, R, X1, nrhs, cc) ;
    }
    if (ok)
    {
        // W = E*X
        // W = (Entry *) spqr_calloc <Int> (A->ncol * nrhs, sizeof (Entry), cc);
        W = spqr_zeros <Int> (A->ncol, nrhs, xtype, cc) ;
        W1 = (Entry *) W->x ;
        for (Int col = 0 ; col < nrhs ; col++)
        {
            for (Int k = 0 ; k < rank ; k++)
            {
                Int j = Qfill ? Qfill [k] : k ;
                if (j < (Int) A->ncol) W1 [j] = X1 [k] ;
            }
            W1 += A->ncol ;
            X1 += n ;
        }
        // check norm (A*x-b), x and b dense
        resid = dense_resid <Entry,Int> (A, anorm, W, nrhs, B, cc) ;
        // spqr_free <Int> (A->ncol * nrhs, sizeof (Entry), W, cc) ;
        spqr_free_dense <Int> (&W, cc) ;
    }
    return (CHECK_NAN (resid)) ;
}


// =============================================================================
// === transpose ===============================================================
// =============================================================================

// Transpose a dense matrix.

template <typename Entry, typename Int> cholmod_dense *transpose
(
    cholmod_dense *Xdense,
    cholmod_common *cc
)
{
    Entry *X, *Y ;
    cholmod_dense *Ydense ;
    if (Xdense == NULL)
    {
        printf ("transpose failed!\n") ;
        return (NULL) ;
    }
    Int m = Xdense->nrow ;
    Int n = Xdense->ncol ;
    Int ldx = Xdense->d ;
    int xtype = spqr_type <Entry> ( ) ;
    Ydense = spqr_allocate_dense <Int> (n, m, n, xtype, cc) ;
    X = (Entry *) Xdense->x ;
    Y = (Entry *) Ydense->x ;
    for (Int i = 0 ; i < m ; i++)
    {
        for (Int j = 0 ; j < n ; j++)
        {
            Y [j+i*n] = spqr_conj (X [i+j*ldx]) ;
        }
    }
    return (Ydense) ;
}

// =============================================================================
// === qrtest ==================================================================
// =============================================================================

template <typename Entry, typename Int> void qrtest
(
    cholmod_sparse *A,
    double errs [5],
    cholmod_common *cc
)
{
    cholmod_sparse *H, *I, *R, *Q, *Csparse, *Xsparse, *AT, *Bsparse ;
    cholmod_dense *Cdense, *Xdense, *Bdense, *HTau ; ;
    double tol = DBL_EPSILON, err, resid, maxerr, maxresid [2][2] ;
    double tols [ ] = { SPQR_DEFAULT_TOL, -1, 0, DBL_EPSILON } ;
    Int n, m, nz, *HPinv, ntol, *Ai, *Ap, k, *Qfill, rank, nb, *Cp, *Ci, econ ;
    int which ;
    Entry *B, *Ax, *Cx ;
    int xtype = spqr_type <Entry> ( ) ;
    int ordering ;
    Entry range = (Entry) 1.0 ;

    errs [0] = EMPTY ;
    errs [1] = EMPTY ;
    errs [2] = EMPTY ;
    errs [3] = EMPTY ;
    errs [4] = EMPTY ;
    if (A == NULL)
    {
        fprintf (stderr, "qrtest: no input matrix\n") ;
        return ;
    }

    m = A->nrow ;
    n = A->ncol ;
    Ap = (Int *) A->p ;
    Ai = (Int *) A->i ;
    Ax = (Entry *) A->x ;
    double anorm = spqr_norm_sparse <Int> (A, 1, cc) ;
    anorm = CHECK_NAN (anorm) ;
    printf ("\n===========================================================\n") ;
    printf ("Matrix: %d by %d, nnz(A) = %ld, norm(A,1) = %g\n",
        (int) m, (int) n, spqr_nnz <Int> (A, cc), anorm) ;
    printf (  "===========================================================\n") ;

    if (anorm == 0) anorm = 1 ;

    // these should all be zero, no matter what the matrix
    maxerr = 0 ;

    // residuals for well-determined or under-determined systems.  If
    // not rank deficient, these should be zero
    maxresid [0][0] = 0 ;      // for m <= n, default tol, ordering not fixed
    maxresid [0][1] = 0 ;      // for m <= n, all other cases

    // residuals for least-squares systems (these will not be zero)
    maxresid [1][0] = 0 ;      // for m > n, default tol, ordering not fixed
    maxresid [1][1] = 0 ;      // for m > n, all other cases

    my_srand (m+n) ;

    if (m > 45000)
    {
        // The only thing returned are the statistics (rank, etc)
        printf ("\n=== QR huge (expect int overflow):\n") ;
        rank = SPQR_qr <Entry,Int> (
            0, 0, 0, 0, A,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            cc, FALSE, FALSE, FALSE) ;
        return ;
    }

    if (MAX (m,n) >= 300) fprintf (stderr, "please wait ... ") ;

    AT = spqr_transpose <Int> (A, 2, cc) ;

    // -------------------------------------------------------------------------
    // test QR routines
    // -------------------------------------------------------------------------

    econ = m ;

    for (ordering = 0 ; ordering <= 9 ; ordering++)
    {

        // skip SPQR_ORDERING_GIVEN, unless the matrix is tiny
        if (ordering == SPQR_ORDERING_GIVEN && MAX (m,n) > 10) continue ;

        for (ntol = 0 ; ntol < NTOL ; ntol++)
        {

            tol = tols [ntol] ;
            if // (ntol == 0)                   // this old test is fragile ...
               (tol == SPQR_DEFAULT_TOL)        // use this instead
            {
                // with default tolerance, the fixed ordering can sometimes
                // fail if the matrix is rank deficient (R cannot be permuted
                // to upper trapezoidal form).
                which = (ordering == 0) ;
            }
            else
            {
                // with non-default tolerance, the solution can sometimes be
                // poor; this is expected.
                which = 1 ;
            }
            printf ("\n=== QR with ordering %d tol %g:\n", ordering, tol) ;

            // -----------------------------------------------------------------
            // create dense and sparse right-hand-sides
            // -----------------------------------------------------------------

            nb = 5 ;
            Bdense = spqr_zeros <Int> (m, nb, xtype, cc) ;
            B = (Entry *) Bdense->x ;
            for (k = 0 ; k < m*nb ; k++)
            {
                B [k] = ((double) (my_rand ( ) % 2)) * erand (range) ;
            }
            Bsparse = spqr_dense_to_sparse <Int> (Bdense, TRUE, cc) ;

            // -----------------------------------------------------------------
            // X = qrsolve(A,B) where X and B are dense
            // -----------------------------------------------------------------

            // X = A\B
            if (ordering == SPQR_ORDERING_DEFAULT && tol == SPQR_DEFAULT_TOL)
            {
                printf ("[ backslach, A and B and X dense: defaults\n") ;
                Xdense = SuiteSparseQR <Entry, Int> (A, Bdense, cc) ;
                printf ("] done backslach, A and B and X dense: defaults\n") ;
            }
            else
            {
                printf ("[ backslach, A and B and X dense: tol %g order %d\n",
                    tol, ordering) ;
                Xdense = SuiteSparseQR <Entry, Int> (ordering, tol, A, Bdense, cc) ;
                printf ("] done backslach, A B X dense: tol %g order %d\n",
                    tol, ordering) ;
            }

            // check norm (A*x-b), x and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid0b %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_dense <Int>  (&Xdense, cc) ;

            if (cc->useGPU)
            {
                // error testing for infeasible GPU memory
                size_t save = cc->gpuMemorySize ;
                cc->gpuMemorySize = 1 ;
                printf ("[ Pretend GPU memory is too small:\n") ;
                Xdense = SuiteSparseQR <Entry, Int> (ordering, tol, A, Bdense, cc) ;
                cc->gpuMemorySize = save ;
                printf ("] test done infeasible GPU, status %2d, useGPU: %d\n",
                    cc->status, cc->useGPU) ;
                spqr_free_dense <Int> (&Xdense, cc) ;
            }

            spqr_free_dense <Int>  (&Bdense, cc) ;

            // -----------------------------------------------------------------
            // X = qrsolve(A,B) where X and B are sparse
            // -----------------------------------------------------------------

            // X = A\B
            printf ("[ backslash with sparse B: tol  %g\n", tol) ;
            Xsparse = SuiteSparseQR <Entry, Int> (ordering, tol, A, Bsparse, cc) ;
            printf ("] did tol %g\n", tol) ;

            // check norm (A*x-b), x and b sparse
            resid = sparse_resid <Entry,Int> (A, anorm, Xsparse, Bsparse, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid0 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_sparse <Int> (&Xsparse, cc) ;
            spqr_free_sparse <Int> (&Bsparse, cc) ;
            
            // -----------------------------------------------------------------
            // X = qrsolve (A,B) where X and B are sparse, with memory test
            // -----------------------------------------------------------------

            // use B = A and solve AX=B where X is sparse
            cc->SPQR_shrink = 0 ;         // shrink = 0 ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 2, A,
                A, NULL, &Xsparse, NULL, NULL, &Qfill, NULL, NULL, NULL,
                cc, TRUE, m < 300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            if // (ntol == 0)               // old test is fragile ...
               (tol == SPQR_DEFAULT_TOL)    // use this instead.
            {
                printf ("using default tol: %g\n", cc->SPQR_tol_used) ;
            }

            // check norm (A*x-b), x and b sparse
            resid = sparse_resid <Entry,Int> (A, anorm, Xsparse, A, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid1 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_sparse <Int> (&Xsparse, cc) ;
            spqr_free <Int> (n+n, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // X = qrsolve (A,B) where X and B are dense, with memory test
            // -----------------------------------------------------------------

            // use B = dense m-by-2 matrix with some zeros
            nb = 5 ;
            Bdense = spqr_zeros <Int> (m, nb, xtype, cc) ;
            B = (Entry *) Bdense->x ;
            for (k = 0 ; k < m*nb ; k++)
            {
                B [k] = (k+1) % 7 ;
            }

            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 2, A,
                NULL, Bdense, NULL, &Xdense, NULL, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;

            // check norm (A*x-b), x and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid2 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_dense <Int> (&Xdense, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // X = qrsolve (A,B) where X and B are full and H is kept
            // -----------------------------------------------------------------

            cc->SPQR_shrink = 2 ;         // shrink = 2 ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 2, A,
                NULL, Bdense, NULL, &Xdense, NULL, &Qfill, &H, &HPinv, &HTau,
                cc, FALSE, m < 300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            spqr_free_dense <Int> (&HTau, cc) ;
            spqr_free <Int> (m, sizeof (Int), HPinv, cc) ;
            spqr_free_sparse <Int> (&H, cc) ;

            // check norm (A*x-b), x and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid3 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_dense <Int> (&Xdense, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B) where C is sparse and B is full
            // -----------------------------------------------------------------

            cc->SPQR_shrink = 2 ;         // shrink = 2 ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 0, A,
                NULL, Bdense, &Csparse, NULL, &R, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            // compute x=R\C and check norm (A*x-b)
            Cdense = spqr_sparse_to_dense <Int> (Csparse, cc) ;
            resid = check_rc <Entry,Int> (rank, R, A, B, Cdense, nb, anorm, Qfill, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid4 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;
            spqr_free_dense <Int> (&Cdense, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err1:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&Csparse, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B) where C and B are full
            // -----------------------------------------------------------------

            cc->SPQR_shrink = 0 ;         // shrink = 0 ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 0, A,
                NULL, Bdense, NULL, &Cdense, &R, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err2:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_dense <Int> (&Cdense, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B) where C and B are full, simple wrapper
            // -----------------------------------------------------------------

            SuiteSparseQR <Entry, Int> (ordering, tol, econ, A, Bdense,
                &Cdense, &R, &Qfill, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err3:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_dense <Int> (&Cdense, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B) where C and B are sparse, simple wrapper
            // -----------------------------------------------------------------
            
            Bsparse = spqr_dense_to_sparse <Int> (Bdense, TRUE, cc) ;

            SuiteSparseQR <Entry, Int> (ordering, tol, econ, A, Bsparse,
                &Csparse, &R, &Qfill, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err4:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&Csparse, cc) ;
            spqr_free_sparse <Int> (&Bsparse, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [CT,R,E] = qr (A,B), but do not return R
            // -----------------------------------------------------------------

            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 1, A,
                NULL, Bdense, &Csparse, NULL, NULL, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;

            spqr_free_sparse <Int> (&Csparse, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [Q,R,E] = qr (A), Q in Householder form
            // -----------------------------------------------------------------

            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, -1, A,
                NULL, NULL, NULL, NULL, &R, &Qfill, &H, &HPinv, &HTau,
                cc, FALSE, m < 300, nrand (2)) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err5:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            // solve Ax=b using Householder form
            resid = QRsolve <Entry,Int> (A, anorm, rank, 0, H, HTau, HPinv, R,
                Qfill, Bdense, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid5 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            // solve Ax=b using Q matrix form
            resid = QRsolve <Entry,Int> (A, anorm, rank, 1, H, HTau, HPinv, R,
                Qfill, Bdense, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid6 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_dense <Int> (&HTau, cc) ;
            spqr_free_sparse <Int> (&H, cc) ;
            spqr_free <Int> (m, sizeof (Int), HPinv, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qfill, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;

            // -----------------------------------------------------------------
            // [Q,R,E] = qr (A), non-economy
            // -----------------------------------------------------------------

            cc->SPQR_shrink = 0 ;         // shrink = 0 ;
            I = spqr_speye <Int> (m, m, xtype, cc) ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, m, 1, A,
                I, NULL, &Q, NULL, &R, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m<300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            // ensure norm (Q*R - A*E) is small
            err = check_qr <Entry,Int> (Q, R, A, Qfill, anorm, cc) ;
            printf ("order %d : Q*R-A*E           Err6:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&I, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free_sparse <Int> (&Q, cc) ;
            spqr_free <Int> (n+m, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [Q,R,E] = qr (A), non-economy, using simple wrapper
            // -----------------------------------------------------------------

            if (nrand (2))
            {
                // use C version
                if (sizeof (Int) == sizeof (int64_t))
                {
                    SuiteSparseQR_C_QR (ordering, tol, m, A, &Q, &R,
                        (int64_t **) &Qfill, cc) ;
                }
                else
                {
                    SuiteSparseQR_i_C_QR (ordering, tol, m, A, &Q, &R,
                        (int32_t **) &Qfill, cc) ;
                }
            }
            else
            {
                // use C++ version
                SuiteSparseQR <Entry, Int> (ordering, tol, m, A, &Q, &R,
                    &Qfill, cc) ;
            }

            // ensure norm (Q*R - A*E) is small
            err = check_qr <Entry,Int> (Q, R, A, Qfill, anorm, cc) ;
            printf ("order %d : Q*R-A*E           Err7:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free_sparse <Int> (&Q, cc) ;
            spqr_free <Int> (n+m, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [R,E] = qr (A)
            // -----------------------------------------------------------------

            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 0, A,
                NULL, NULL, NULL, NULL, &R, &Qfill, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err8:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;
            Int rank1 = rank ;

            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [R,E] = qr (A) using simple wrapper
            // -----------------------------------------------------------------

            SuiteSparseQR <Entry, Int> (ordering, tol, econ, A, &R, &Qfill, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err9:  %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qfill, cc) ;

            // -----------------------------------------------------------------
            // [ ] = qr (A)
            // -----------------------------------------------------------------

            // The only thing returned are the statistics (rank, etc)
            cc->SPQR_shrink = 0 ;         // shrink = 0 ;
            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 0, A,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                cc, FALSE, m < 300, nrand (2)) ;
            cc->SPQR_shrink = 1 ;         // restore default shrink = 1 ;

            err = (rank != rank1) ;
            printf ("order %d : rank %6d %6d Err10: %g\n", ordering,
                (int) rank, (int) rank1, err) ;
            maxerr = MAX (maxerr, err) ;

            // -----------------------------------------------------------------
            // [C,H,R,E] = qr (A)
            // -----------------------------------------------------------------

            rank = SPQR_qr <Entry,Int> (
                ordering, tol, econ, 0, A,
                NULL, Bdense, &Csparse, NULL, &R, &Qfill, &H, &HPinv, &HTau,
                cc, FALSE, m < 300, nrand (2)) ;

            // compute x=R\C and check norm (A*x-b)
            Cdense = spqr_sparse_to_dense <Int> (Csparse, cc) ;
            resid = check_rc <Entry,Int> (rank, R, A, B, Cdense, nb, anorm, Qfill, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("Resid7 %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;
            spqr_free_dense <Int> (&Cdense, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err11: %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&Csparse, cc) ;
            spqr_free_sparse <Int> (&R, cc) ;

            // compare Q with qmult
            err = check_qmult <Entry,Int> (H, HTau, HPinv,
                ordering == 2 && /* ntol == 0 */ tol == SPQR_DEFAULT_TOL, cc) ;
            printf ("order %d : check qmult       Err12: %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_dense <Int> (&HTau, cc) ;
            spqr_free_sparse <Int> (&H, cc) ;
            spqr_free <Int> (m, sizeof (Int), HPinv, cc) ;
            spqr_free <Int> (n+nb, sizeof (Int), Qfill, cc) ;
            spqr_free_dense <Int> (&Bdense, cc) ;

            // -----------------------------------------------------------------
            // [H,R,E] = qr (A), simple wrapper
            // -----------------------------------------------------------------

            SuiteSparseQR <Entry, Int> (ordering, tol, econ, A,
                &R, &Qfill, &H, &HPinv, &HTau, cc) ;

            // check that R'*R = (A*E)'*(A*E)
            err = check_r_factor <Entry,Int> (R, A, Qfill, cc) ;
            printf ("order %d : R'R-(A*E)'*(A*E), Err13: %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            // compare Q with qmult
            err = check_qmult <Entry,Int> (H, HTau, HPinv, FALSE, cc) ;
            printf ("order %d : check qmult       Err14: %g\n", ordering, err) ;
            maxerr = MAX (maxerr, err) ;

            spqr_free_sparse <Int> (&R, cc) ;
            spqr_free_dense <Int> (&HTau, cc) ;
            spqr_free_sparse <Int> (&H, cc) ;
            spqr_free <Int> (m, sizeof (Int), HPinv, cc) ;
            spqr_free <Int> (n, sizeof (Int), Qfill, cc) ;

#ifndef NEXPERT

            // =================================================================
            // === expert routines =============================================
            // =================================================================

            SuiteSparseQR_factorization <Entry, Int> *QR ;
            cholmod_dense *XT, *Zdense, *BT, *Ydense ;
            cholmod_sparse *Ysparse ;

            // -----------------------------------------------------------------
            // QR = qr (A), then solve
            // -----------------------------------------------------------------

            for (int split = 0 ; split <= 4 ; split++)
            {

                QR = SPQR_factorize <Entry,Int> (ordering, tol, A, cc,
                    split, m < 300, nrand (2)) ;

                // split == 4 does not use singletons, so it can fail if
                // rank < n
                int wh = which || (split == 4) ;

                // solve Ax=b
                nb = 5 ;
                Bdense = spqr_zeros <Int> (m, nb, xtype, cc) ;
                B = (Entry *) Bdense->x ;
                for (k = 0 ; k < m*nb ; k++)
                {
                    B [k] = erand (range) ;
                }

                // Y = Q'*B
                Ydense = SPQR_qmult <Entry,Int> (SPQR_QTX, QR, Bdense, cc,
                        m < 300, nrand (2)) ;

                // X = R\(E*Y)
                Xdense = SPQR_solve <Entry,Int> (SPQR_RETX_EQUALS_B, QR,
                        Ydense, cc, m < 300, nrand (2)) ;
                // check norm (A*x-b), x and b dense
                resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
                maxresid [m>n][wh] = MAX (maxresid [m>n][wh], resid) ;
                printf ("Resid8_%d %d %d %d : %g (%d) tol %g\n",
                    split, m>n, (int) ntol, ordering, resid, wh, tol) ;
                spqr_free_dense <Int> (&Xdense, cc) ;
                spqr_free_dense <Int> (&Ydense, cc) ;

                // Y = (B'*Q)'
                BT = transpose <Entry,Int> (Bdense, cc) ;
                XT = SPQR_qmult <Entry,Int> (SPQR_XQ, QR, BT, cc,
                        m < 300, nrand (2)) ;

                Ydense = transpose <Entry,Int> (XT, cc) ;
                spqr_free_dense <Int> (&XT, cc) ;
                spqr_free_dense <Int> (&BT, cc) ;

                // X = R\(E*Y)
                Xdense = SPQR_solve <Entry,Int> (SPQR_RETX_EQUALS_B, QR,
                        Ydense, cc, m < 300, nrand (2)) ;
                // check norm (A*x-b), x and b dense
                resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
                maxresid [m>n][wh] = MAX (maxresid [m>n][wh], resid) ;
                printf ("Resid9_%d %d %d %d : %g (%d) tol %g\n",
                    split, m>n, (int) ntol, ordering, resid, wh, tol) ;
                spqr_free_dense <Int> (&Xdense, cc) ;
                spqr_free_dense <Int> (&Ydense, cc) ;

                // -------------------------------------------------------------
                // error testing
                // -------------------------------------------------------------

                if (ordering == 0 && /* ntol == 0 */ tol == SPQR_DEFAULT_TOL)
                {
                    printf ("Error handling ... expect 3 error messages: \n") ;
                    err = (SuiteSparseQR_qmult <Entry,Int> (-1, QR, Bdense, cc)
                        != NULL) ;
                    spqr_free_dense <Int> (&Bdense, cc) ;
                    Bdense = spqr_zeros <Int> (m+1, 1, xtype, cc) ;
                    err += (SuiteSparseQR_qmult <Entry,Int> (SPQR_QX,QR,Bdense,cc)
                        != NULL);
                    spqr_free_dense <Int> (&Bdense, cc) ;
                    Bdense = spqr_zeros <Int> (1, m+1, xtype, cc) ;
                    err += (SuiteSparseQR_qmult <Entry,Int> (SPQR_XQ,QR,Bdense,cc)
                        != NULL);
                    if (QR->n1cols > 0)
                    {
                        // this will fail; cannot refactorize with singletons
                        printf ("Error handling ... expect error message:\n") ;
                        err += (SuiteSparseQR_numeric <Entry,Int>
                            (tol, A, QR, cc) != FALSE) ;
                    }
                    printf ("order %d : error handling    Err15: %g\n",
                        ordering, err) ;
                    maxerr = MAX (maxerr, err) ;
                    printf (" ... error handling done\n\n") ;
                }

                spqr_free_dense <Int> (&Bdense, cc) ;

                // -------------------------------------------------------------

                // solve A'x=b
                nb = 5 ;
                Bdense = spqr_zeros <Int> (n, nb, xtype, cc) ;
                B = (Entry *) Bdense->x ;
                for (k = 0 ; k < n*nb ; k++)
                {
                    B [k] = erand (range) ;
                }
                // Y = R'\(E'*B)
                Ydense = SPQR_solve <Entry,Int> (SPQR_RTX_EQUALS_ETB, QR,
                        Bdense, cc, m < 300, nrand (2)) ;
                // X = Q*Y
                Xdense = SPQR_qmult <Entry,Int> (SPQR_QX, QR, Ydense, cc,
                        m < 300, nrand (2)) ;
                // check norm (A'*x-b), x and b dense
                resid = dense_resid <Entry,Int> (AT, anorm, Xdense, nb, B, cc) ;
                maxresid [m<n][wh] = MAX (maxresid [m<n][wh], resid) ;
                printf ("ResidA_%d %d %d %d : %g (%d) tol %g\n",
                    split, m<n, (int) ntol, ordering, resid, wh, tol) ;
                spqr_free_dense <Int> (&Xdense, cc) ;
                spqr_free_dense <Int> (&Ydense, cc) ;

                // -------------------------------------------------------------
                // error testing
                // -------------------------------------------------------------

                if (!split && ordering == 0 && /* ntol == 0 */
                    tol == SPQR_DEFAULT_TOL)
                {
                    printf ("Error testing ... expect 3 error messages:\n") ;
                    err = (SuiteSparseQR_solve <Entry,Int> (-1, QR, Bdense, cc)
                        != NULL) ;
                    spqr_free_dense <Int> (&Bdense, cc) ;
                    err += (SuiteSparseQR_solve <Entry,Int>
                        (SPQR_RTX_EQUALS_ETB, QR, Bdense, cc) != NULL) ;
                    Bdense = spqr_zeros <Int> (n+1, 1, xtype, cc) ;
                    err += (SuiteSparseQR_solve (SPQR_RTX_EQUALS_ETB,
                        QR, Bdense, cc) != NULL) ;
                    printf ("order %d : error handling    Err16: %g\n",
                        ordering, err) ;
                    maxerr = MAX (maxerr, err) ;
                    printf (" ... error handling done\n\n") ;
                }

                SuiteSparseQR_free (&QR, cc) ;
                spqr_free_dense <Int> (&Bdense, cc) ;
            }

            // -----------------------------------------------------------------
            // QR = qr (A'), then solve
            // -----------------------------------------------------------------

            // use qmult to solve min-2-norm problem
            QR = SuiteSparseQR_factorize <Entry,Int> (ordering, tol, AT, cc) ;

            nb = 5 ;
            Bdense = spqr_zeros <Int> (m, nb, xtype, cc) ;
            B = (Entry *) Bdense->x ;
            for (k = 0 ; k < m*nb ; k++)
            {
                B [k] = erand (range) ;
            }

            // solve X = R'\B
            Xdense = SPQR_solve <Entry,Int> (SPQR_RTX_EQUALS_B, QR, Bdense, cc,
                    m < 300, nrand (2)) ;
            spqr_free_dense <Int> (&Xdense, cc) ;

            // solve X = R'\(E'*B)
            Xdense = SPQR_solve <Entry,Int> (SPQR_RTX_EQUALS_ETB, QR, Bdense,
                    cc, m < 300, nrand (2)) ;

            // Y = Q*X
            Ydense = SPQR_qmult <Entry,Int> (SPQR_QX, QR, Xdense, cc,
                    m < 300, nrand (2)) ;

            // check norm (A*y-b), y and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Ydense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("ResidB %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;
            spqr_free_dense <Int> (&Ydense, cc) ;

            // Y = (X'*Q')'
            XT = transpose <Entry,Int> (Xdense, cc) ;
            Zdense = SPQR_qmult <Entry,Int> (SPQR_XQT, QR, XT, cc,
                    m < 300, nrand (2)) ;
            Ydense = transpose <Entry,Int> (Zdense, cc) ;
            spqr_free_dense <Int> (&XT, cc) ;
            spqr_free_dense <Int> (&Zdense, cc) ;

            // check norm (A*y-b), y and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Ydense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("ResidC %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;
            spqr_free_dense <Int> (&Ydense, cc) ;
            spqr_free_dense <Int> (&Xdense, cc) ;

            // -----------------------------------------------------------------
            // min 2-norm solution using min2norm
            // -----------------------------------------------------------------

            Xdense = SPQR_min2norm <Entry,Int> (ordering, tol, A, Bdense, cc,
                    m < 300, nrand (2)) ;

            // check norm (A*x-b), y and b dense
            resid = dense_resid <Entry,Int> (A, anorm, Xdense, nb, B, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("ResidD %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;
            spqr_free_dense <Int> (&Xdense, cc) ;

            spqr_free_dense <Int> (&Bdense, cc) ;

            // -----------------------------------------------------------------
            // sparse case
            // -----------------------------------------------------------------

            nb = 5 ;
            Bdense = spqr_zeros <Int> (m, nb, xtype, cc) ;
            B = (Entry *) Bdense->x ;
            for (k = 0 ; k < m*nb ; k++)
            {
                B [k] = ((double) (my_rand ( ) % 2)) * erand (range) ;
            }
            Bsparse = spqr_dense_to_sparse <Int> (Bdense, TRUE, cc) ;
            spqr_free_dense <Int>  (&Bdense, cc) ;

            // solve X = R'\(E'*B)
            Xsparse = NULL ;
            Xsparse = SPQR_solve <Entry,Int> (SPQR_RTX_EQUALS_ETB, QR,
                Bsparse, cc, m < 300, nrand (2)) ;

            // Y = Q*X
            Ysparse = NULL ;
            Ysparse = SPQR_qmult <Entry,Int> (SPQR_QX, QR, Xsparse, cc,
                    m < 300, nrand (2)) ;

            // check norm (A*y-b), y and b sparse
            resid = sparse_resid <Entry,Int> (A, anorm, Ysparse, Bsparse, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("ResidE %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_sparse <Int> (&Xsparse, cc) ;
            spqr_free_sparse <Int> (&Ysparse, cc) ;

            Xsparse = SPQR_min2norm <Entry,Int> (ordering, tol, A, Bsparse, cc,
                    m < 300, nrand (2)) ;

            // check norm (A*x-b), x and b sparse
            resid = sparse_resid <Entry,Int> (A, anorm, Xsparse, Bsparse, cc) ;
            maxresid [m>n][which] = MAX (maxresid [m>n][which], resid) ;
            printf ("ResidF %d %d %d : %g\n", m>n, (int) ntol,
                ordering, resid) ;

            spqr_free_sparse <Int> (&Xsparse, cc) ;
            spqr_free_sparse <Int> (&Bsparse, cc) ;

            SuiteSparseQR_free (&QR, cc) ;
#endif
        }
    }

    // -------------------------------------------------------------------------
    // check error handling
    // -------------------------------------------------------------------------

    printf ("Check error handling, one error message is expected:\n") ;
    cholmod_dense *Bgunk = spqr_ones <Int> (m+1, 1, xtype, cc) ;
    rank = SuiteSparseQR <Entry, Int> (
        0, 0, econ, -1, A,
        NULL, Bgunk, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        cc) ;
    spqr_free_dense <Int> (&Bgunk, cc) ;
    err = (rank != EMPTY) ;
    maxerr = MAX (maxerr, err) ;     // rank should be EMPTY
    printf (" ... error handling done\n\n") ;

    // -------------------------------------------------------------------------
    // test non-user callable functions
    // -------------------------------------------------------------------------

    // attempt to permute A to upper triangular form
    Int *Qtrap ;
    rank = spqr_trapezoidal <Entry, Int> (n, Ap, Ai, Ax, 0, NULL, FALSE, &Cp, &Ci, &Cx,
        &Qtrap, cc) ;
    printf ("Rank of A, if A*P permutable to upper trapezoidal: %d\n",
        (int) rank) ;
    if (Cp != NULL)
    {
        nz = Cp [n] ;
        spqr_free <Int> (n+1, sizeof (Int), Cp, cc) ;
        spqr_free <Int> (nz, sizeof (Int), Ci, cc) ;
        spqr_free <Int> (nz, sizeof (Entry), Cx, cc) ;
        spqr_free <Int> (n, sizeof (Int), Qtrap, cc) ;
    }
    spqr_free_sparse <Int> (&AT, cc) ;

    // -------------------------------------------------------------------------
    // test the C API
    // -------------------------------------------------------------------------

    qrtest_C (A, anorm, errs, maxresid, cc) ;

    // -------------------------------------------------------------------------
    // final results
    // -------------------------------------------------------------------------

    errs [0] = CHECK_NAN (maxerr) ;
    errs [1] = CHECK_NAN (maxresid [0][0]) ;
    errs [2] = CHECK_NAN (maxresid [0][1]) ;
    errs [3] = CHECK_NAN (maxresid [1][0]) ;
    errs [4] = CHECK_NAN (maxresid [1][1]) ;
}

// =============================================================================
// === do_matrix2 ==============================================================
// =============================================================================

// Read in a matrix, and use it to test SuiteSparseQR
// If kind == 0, then the first two residuals should be low.

template <typename Int>
int do_matrix2 (int kind, cholmod_sparse *A, cholmod_common *cc)
{
    double errs [5] = {0,0,0,0,0} ;
    Int m = A->nrow ;
    Int n = A->ncol ;

    // -------------------------------------------------------------------------
    // use it to test SuiteSparseQR
    // -------------------------------------------------------------------------

    if (A->xtype == CHOLMOD_COMPLEX && A->stype == 0)
    {
        qrtest <Complex,Int> (A, errs, cc) ;
    }
    else if (A->xtype == CHOLMOD_REAL)
    {
        if (A->stype != 0)
        {
            cholmod_sparse *A1 ;
            A1 = spqr_copy <Int> (A, 0, 1, cc) ;
            qrtest <double,Int> (A1, errs, cc) ;
            spqr_free_sparse <Int> (&A1, cc) ;
        }
        else
        {
            qrtest <double,Int> (A, errs, cc) ;
        }
    }
    else
    {
        // cannot handle ZOMPLEX, PATTERN, or symmetric/Hermitian COMPLEX
        fprintf (stderr, "invalid matrix\n") ;
        errs [0] = 1 ;
    }

    // -------------------------------------------------------------------------
    // report the results
    // -------------------------------------------------------------------------

    if (kind == 0)
    {
        printf ("First Resid and ") ;
    }
    printf ("Err should be low:\n") ;
    printf ("RESULT:  Err %8.1e Resid %8.1e %8.1e", errs [0],
        errs [1], errs [2]) ;
    if (m == n)
    {
        printf ("                  ") ;
    }
    else
    {
        printf (" %8.1e %8.1e", errs [3], errs [4]) ;
    }

    if (errs [0] > 1e-10)
    {
        printf (" : FAIL\n") ;
        fprintf (stderr, "Error: %g FAIL\n", errs [0]) ;
        return (1) ;
    }

    // if kind == 0, then this full-rank matrix should have low residual
    if (kind == 0 && (errs [1] > 1e-10))
    {
        printf (" : FAIL\n") ;
        fprintf (stderr, "error: %g FAIL\n", errs [1]) ;
        return (1) ;
    }

    printf (" : OK.") ;
    fprintf (stderr, "OK.") ;
    return (0) ;
}

// =============================================================================
// === do_matrix ===============================================================
// =============================================================================

// Read in a matrix, and use it to test SuiteSparseQR
// If kind == 0, then the first two residuals should be low.

template <typename Int>
int do_matrix (int kind, FILE *file, cholmod_common *cc)
{
    cholmod_sparse *A ;

    int nfail0 = 0 ;
    int nfail1 = 0 ;
    int nfail2 = 0 ;
    int nfail3 = 0 ;

    // -------------------------------------------------------------------------
    // read in the matrix
    // -------------------------------------------------------------------------

    A = spqr_read_sparse <Int> (file, cc) ;
    if (A == NULL)
    {
        fprintf (stderr, "Unable to read matrix\n") ;
        return (1) ;
    }
    Int m = A->nrow ;
    Int n = A->ncol ;
    fprintf (stderr, "%5d by %5d : ", (int) m, (int) n) ;
    if (sizeof (int64_t) > sizeof (int) && (m > 10000 || n > 10000))
    {
        fprintf (stderr, "(test skipped on 64-bit systems)\n") ;
        spqr_free_sparse <Int> (&A, cc) ;
        return (0) ;
    }

    // defaults
    cc->SPQR_grain = 1 ;         // no parallel analysis
    printf ("\nBeginning CPU tests [\n") ;
    fprintf (stderr, " CPU ") ;
    nfail0 = do_matrix2 <Int> (kind, A, cc) ;

    // non-defaults (will not use the GPU)
    cc->SPQR_grain = 4 ;         // grain size relative to total work
    nfail2 = do_matrix2 <Int> (kind, A, cc) ;
    cc->SPQR_grain = 1 ;         // no parallel analysis
    printf ("\nCPU tests done ]\n") ;

    // test the GPU, if installed
    #ifdef SUITESPARSE_CUDA
    cc->useGPU = TRUE ;
    // was 3.5 * ((size_t) 1024 * 1024 * 1024) ;
    size_t totmem, availmem ;
    double t = SuiteSparse_time ( ) ;
    spqr_gpu_memorysize (&totmem, &availmem, cc) ;
    t = SuiteSparse_time ( ) - t ;
    cc->gpuMemorySize = availmem ;
    printf ("\nBeginning GPU tests, GPU memory %g MB warmup time %g[\n",
        (double) (cc->gpuMemorySize) / (1024*1024), t) ;
    fprintf (stderr, " GPU ") ;
    nfail1 = do_matrix2 <Int> (kind, A, cc) ;
    printf ("\nGPU tests done ]\n") ;
    if (m > 200)
    {
        // try with a tiny GPU memory size, but only for a few matrices
        // in the test set.  Each front will go in its own stage.
        printf ("\nBeginning GPU tests with tiny GPU memory [\n") ;
        cc->gpuMemorySize = 0 ;
        nfail3 = do_matrix2 <Int> (kind, A, cc) ;
        // restore defaults
        cc->useGPU = FALSE ;
        printf ("\nGPU tests done (tiny memory) ]\n") ;
    }
    #endif

    spqr_free_sparse <Int> (&A, cc) ;

    printf ("\n") ;
    fprintf (stderr, "\n") ;
    return (nfail0 + nfail1 + nfail2 + nfail3) ;
}

