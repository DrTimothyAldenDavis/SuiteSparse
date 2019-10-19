// =============================================================================
// === spqr_mx_matlab.hpp ======================================================
// =============================================================================

// utility functions and definitions solely for use in a MATLAB mexFunction

#ifndef SPQR_MX_MATLAB_H
#define SPQR_MX_MATLAB_H

#include "mex.h"
#include "SuiteSparseQR.hpp"

// UF_long is defined in UFconfig.h, included by SuiteSparseQR.hpp:
#define Int UF_long
#define Int_max UF_long_max

// get the BLAS_INT definition from CHOLMOD (this is for spumoni output only)
#include "cholmod_blas.h"

#include <complex>
typedef std::complex<double> Complex ;

#define TRUE 1
#define FALSE 0
#define EMPTY (-1)
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

typedef struct spqr_mx_options_struct
{
    double tol ;            // <= -2 means to use default tol
    Int econ ;
    int ordering ;
    int permvector ;
    int Qformat ;
    int haveB ;
    int spumoni ;
    int min2norm ;

} spqr_mx_options ;

// for error and warning messages
#define LEN 200

// default value of spumoni
#define SPUMONI 0

// values for opts.Qformat
#define SPQR_Q_DISCARD 0
#define SPQR_Q_MATRIX 1
#define SPQR_Q_HOUSEHOLDER 2

int spqr_mx_get_options
(
    const mxArray *mxopts,
    spqr_mx_options *opts,
    Int m,
    int nargout,

    // workspace and parameters
    cholmod_common *cc
)  ;

mxArray *spqr_mx_put_sparse
(
    cholmod_sparse **Ahandle,	// CHOLMOD version of the matrix
    cholmod_common *cc
) ;

mxArray *spqr_mx_put_dense2
(
    Int m,
    Int n,
    double *Ax,         // size nz if real; size 2*nz if complex (and freed)
    int is_complex,

    // workspace and parameters
    cholmod_common *cc
) ;

mxArray *spqr_mx_put_dense
(
    cholmod_dense **Ahandle,	// CHOLMOD version of the matrix
    cholmod_common *cc
) ;

mxArray *spqr_mx_put_permutation
(
    Int *P,
    Int n,
    int vector,

    // workspace and parameters
    cholmod_common *cc
) ;

double *spqr_mx_merge_if_complex
(
    // inputs, not modified
    const mxArray *A,
    int make_complex,

    // output
    Int *p_nz,              // number of entries in A

    // workspace and parameters
    cholmod_common *cc
) ;

int spqr_mx_config (Int spumoni, cholmod_common *cc) ;

cholmod_sparse *spqr_mx_get_sparse
(
    const mxArray *Amatlab, // MATLAB version of the matrix
    cholmod_sparse *A,	    // CHOLMOD version of the matrix
    double *dummy 	    // a pointer to a valid scalar double
) ;

cholmod_dense *spqr_mx_get_dense
(
    const mxArray *Amatlab, // MATLAB version of the matrix
    cholmod_dense *A,	    // CHOLMOD version of the matrix
    double *dummy	    // a pointer to a valid scalar double
) ;

void spqr_mx_get_usage
(
    mxArray *A,         // mxArray to check
    int tight,          // if true, then nnz(A) must equal nzmax(A)
    Int *p_usage,       // bytes used
    Int *p_count,       // # of malloc'd blocks
    cholmod_common *cc
) ;

extern "C" {
extern int spqr_spumoni ;
void spqr_mx_error (int status, const char *file, int line, const char *msg) ;
}

void spqr_mx_spumoni
(
    spqr_mx_options *opts,
    int is_complex,             // TRUE if complex, FALSE if real
    cholmod_common *cc
) ;

#endif
