/* ========================================================================== */
/* === umfpack_global ======================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License for License.                      */
/* -------------------------------------------------------------------------- */

/* prototypes for global variables, and basic operators for complex values  */

#ifndef EXTERN
#define EXTERN extern
#endif

EXTERN double (*umfpack_hypot) (double, double) ;
EXTERN int (*umfpack_divcomplex) (double, double, double, double, double *, double *) ;

double umf_hypot (double x, double y) ;
int umf_divcomplex (double, double, double, double, double *, double *) ;

