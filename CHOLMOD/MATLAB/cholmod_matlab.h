//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/cholmod_matlab.h: include file for CHOLMOD' MATLAB interface
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2022, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* Shared prototypes and definitions for CHOLMOD mexFunctions */

#include "SuiteSparse_config.h"

#include "cholmod.h"
#include <float.h>
#include "mex.h"
#define EMPTY (-1)
#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define LEN 16

#define MXFREE(a) { \
    void *ptr ; \
    ptr = (void *) (a) ; \
    if (ptr != NULL) mxFree (ptr) ; \
}

#define ERROR_TOO_SMALL 0
#define ERROR_HUGE 1
#define ERROR_NOT_INTEGER 2
#define ERROR_TOO_LARGE 3
#define ERROR_USAGE 4
#define ERROR_LENGTH 5
#define ERROR_INVALID_TYPE 6
#define ERROR_OUT_OF_MEMORY 7

/* getting spumoni at run-time takes way too much time */
#ifndef SPUMONI
#define SPUMONI 0
#endif

/* closed by sputil_error_handler if not NULL */
extern FILE *sputil_file ;

void sputil_error   /* reports an error */
(
    int64_t error,     /* kind of error */
    int64_t is_index   /* TRUE if a matrix index, FALSE if a matrix dimension */
) ;

int64_t sputil_double_to_int   /* returns integer value of x */
(
    double x,       /* double value to convert */
    int64_t is_index,  /* TRUE if a matrix index, FALSE if a matrix dimension */
    int64_t n          /* if a matrix index, x cannot exceed this dimension */
) ;

double sputil_get_double (const mxArray *arg) ; /* like mxGetScalar */

int64_t sputil_get_integer  /* returns the integer value of a MATLAB argument */
(
    const mxArray *arg,     /* MATLAB argument to convert */
    int64_t is_index,       /* TRUE if an index, FALSE if a matrix dimension */
    int64_t n               /* maximum value, if an index */
) ;


int64_t sputil_copy_ij     /* returns the dimension, n */
(
    int64_t is_scalar,  /* TRUE if argument is a scalar, FALSE otherwise */
    int64_t scalar,     /* scalar value of the argument */
    void *vector,       /* vector value of the argument */
    mxClassID category, /* type of vector */
    int64_t nz,         /* length of output vector I */
    int64_t n,          /* maximum dimension, EMPTY if not yet known */
    int64_t *I          /* vector of length nz to copy into */
) ;

/* converts a triplet matrix to a compressed-column matrix */
cholmod_sparse *sputil_triplet_to_sparse
(
    int64_t nrow, int64_t ncol, int64_t nz, int64_t nzmax,
    int64_t i_is_scalar, int64_t i, void *i_vector, mxClassID i_class,
    int64_t j_is_scalar, int64_t j, void *j_vector, mxClassID j_class,
    int64_t s_is_scalar, double x, double z, void *x_vector, double *z_vector,
    mxClassID s_class, int64_t s_complex,
    cholmod_common *cm
) ;

mxArray *sputil_copy_sparse (const mxArray *A) ;    /* copy a sparse matrix */

int64_t sputil_nelements (const mxArray *arg) ; /* like mxGetNumberOfElements */

void sputil_sparse      /* top-level wrapper for "sparse" function */
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
) ;

void sputil_error_handler (int status, const char *file, int line,
    const char *message) ;

void sputil_config (int64_t spumoni, cholmod_common *cm) ;

mxArray *sputil_sparse_to_dense (const mxArray *S) ;

cholmod_sparse *sputil_get_sparse
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_sparse *A,      /* CHOLMOD version of the matrix */
    double *dummy,          /* a pointer to a valid scalar double */
    int64_t stype           /* -1: lower, 0: unsymmetric, 1: upper */
) ;

cholmod_dense *sputil_get_dense
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_dense *A,       /* CHOLMOD version of the matrix */
    double *dummy           /* a pointer to a valid scalar double */
) ;

mxArray *sputil_put_dense   /* returns the MATLAB version */
(
    cholmod_dense **Ahandle,    /* CHOLMOD version of the matrix */
    cholmod_common *cm
) ;

mxArray *sputil_put_sparse
(
    cholmod_sparse **Ahandle,   /* CHOLMOD version of the matrix */
    cholmod_common *cm
) ;

int64_t sputil_drop_zeros      /* drop numerical zeros from a CHOLMOD matrix */
(
    cholmod_sparse *S
) ;

mxArray *sputil_put_int     /* copy int64_t vector to mxArray */
(
    int64_t *P,             /* vector to convert */
    int64_t n,              /* length of P */
    int64_t one_based       /* 1 if convert from 0-based to 1-based, else 0 */
) ;

mxArray *sputil_dense_to_sparse (const mxArray *arg) ;

void sputil_check_ijvector (const mxArray *arg) ;

void sputil_trim
(
    cholmod_sparse *S,
    int64_t k,
    cholmod_common *cm
) ;

cholmod_sparse *sputil_get_sparse_pattern
(
    const mxArray *Amatlab,     /* MATLAB version of the matrix */
    cholmod_sparse *Ashallow,   /* shallow CHOLMOD version of the matrix */
    double *dummy,              /* a pointer to a valid scalar double */
    cholmod_common *cm
) ;

cholmod_sparse *sputil_extract_zeros
(
    cholmod_sparse *A,
    cholmod_common *cm
) ;
