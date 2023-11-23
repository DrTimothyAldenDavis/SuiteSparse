//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/sputil2.h: include file for CHOLMOD' MATLAB interface
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Shared prototypes and definitions for CHOLMOD mexFunctions

#include "SuiteSparse_config.h"
#include "cholmod.h"
#include <float.h>
#include "mex.h"

#if MX_HAS_INTERLEAVED_COMPLEX
// -R2018a is required
#else
#error "CHOLMOD must be compiled with 'mex -R2018a'"
#endif

#define EMPTY (-1)
#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define LEN 16

#define MXFREE(a)                       \
{                                       \
    void *ptr ;                         \
    ptr = (void *) (a) ;                \
    if (ptr != NULL) mxFree (ptr) ;     \
}

// getting spumoni at run-time takes way too much time
#ifndef SPUMONI
#define SPUMONI 0
#endif

// closed by sputil2_error_handler if not NULL
extern FILE *sputil2_file ;

void sputil2_error_handler
(
    int status,
    const char *file,
    int line,
    const char *message
) ;

void sputil2_config (int64_t spumoni, cholmod_common *cm) ;

cholmod_sparse *sputil2_get_sparse  // create a CHOLMOD copy of a MATLAB matrix
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse or dense matrix
    int stype,              // assumed A->stype (-1: lower, 0: unsym, 1: upper)
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output, if the MATLAB input matrix is sparse.
    // output:
    size_t *xsize,          // if > 0, A->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_sparse.
    cholmod_common *cm
) ;

cholmod_sparse *sputil2_get_sparse_only     // returns A = Astatic
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse matrix
    int stype,              // assumed A->stype (-1: lower, 0: unsym, 1: upper)
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output.
    // output:
    size_t *xsize,          // if > 0, A->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_sparse.
    cholmod_common *cm
) ;

void sputil2_free_sparse    // free a matrix created by sputil2_get_sparse
(
    // input/output:
    cholmod_sparse **A,     // a CHOLMOD matrix created by sputil2_get_sparse
    cholmod_sparse *Astatic,
    // input:
    size_t xsize,           // from sputil2_get_sparse when A was created
    cholmod_common *cm
) ;

cholmod_dense *sputil2_get_dense // CHOLMOD copy of a MATLAB dense matrix
(
    // input:
    const mxArray *Xmatlab, // MATLAB dense matrix
    int dtype,              // requested X->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_dense *Xstatic, // the header of X on input, contents not
                            // initialized.  Contains the CHOLMOD dense X on
                            // output.
    // output:
    size_t *xsize,          // if > 0, X->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_dense.
    cholmod_common *cm
) ;

void sputil2_free_dense    // free a matrix created by sputil2_get_dense
(
    // input/output:
    cholmod_dense **Xhandle,    // matrix created by sputil2_get_dense
    cholmod_dense *Xstatic,
    // input:
    size_t xsize,               // from sputil2_get_dense when X was created
    cholmod_common *cm
) ;

mxArray *sputil2_put_dense      // return MATLAB version of the matrix
(
    cholmod_dense **Xhandle,    // CHOLMOD version of the matrix
    mxClassID mxclass,          // requested class of the MATLAB matrix:
                                // mxDOUBLE_CLASS, mxSINGLE_CLASS, or
                                // mxLOGICAL_CLASS
    cholmod_common *cm
) ;

mxArray *sputil2_put_sparse     // return MATLAB version of the matrix
(
    cholmod_sparse **Ahandle,   // CHOLMOD version of the matrix
    mxClassID mxclass,          // requested class of the MATLAB matrix:
                                // mxDOUBLE_CLASS, mxSINGLE_CLASS, or
                                // mxLOGICAL_CLASS
    bool drop,                  // if true, drop explicit zeros
    cholmod_common *cm
) ;

mxArray *sputil2_put_int    // copy int64_t vector to mxArray
(
    int64_t *P,             // vector to convert
    int64_t n,              // length of P
    int64_t one_based       // 1 if convert from 0-based to 1-based, else 0
) ;

void sputil2_trim
(
    cholmod_sparse *S,
    int64_t k,
    cholmod_common *cm
) ;

cholmod_sparse *sputil2_get_sparse_pattern
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse or dense matrix
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output, if the MATLAB input matrix is sparse.
    cholmod_common *cm
) ;

cholmod_sparse *sputil2_extract_zeros
(
    cholmod_sparse *A,
    cholmod_common *cm
) ;

