// CXSparse/MATLAB/CSparse/cs_mex.h: include file for MATLAB interface
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
#include "mex.h"

void cs_mex_check (int64_t nel, int64_t m, int64_t n, int square,
    int sparse, int values, const mxArray *A) ;

int64_t *cs_dl_mex_get_int (int64_t n, const mxArray *Imatlab, int64_t *imax,
    int lo);
mxArray *cs_dl_mex_put_int (int64_t *p, int64_t n, int64_t offset,
    int do_free) ;

double *cs_dl_mex_get_double (int64_t n, const mxArray *X) ;
cs_dl *cs_dl_mex_get_sparse (cs_dl *A, int square, int values,
    const mxArray *Amatlab) ;
double *cs_dl_mex_put_double (int64_t n, const double *b, mxArray **X) ;
mxArray *cs_dl_mex_put_sparse (cs_dl **A) ;

#ifndef NCOMPLEX
cs_complex_t *cs_cl_mex_get_double (int64_t n, const mxArray *X) ;
cs_cl *cs_cl_mex_get_sparse (cs_cl *A, int square, const mxArray *Amatlab) ;
mxArray *cs_cl_mex_put_double (int64_t n, cs_complex_t *b) ;
mxArray *cs_cl_mex_put_sparse (cs_cl **Ahandle) ;
#endif
