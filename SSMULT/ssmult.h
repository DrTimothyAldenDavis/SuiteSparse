/* ========================================================================== */
/* === ssmult.h ============================================================= */
/* ========================================================================== */

/* Include file for ssmult.c and ssmultsym.c
 * Copyright 2007, Timothy A. Davis, University of Florida
 */

#include "mex.h"
#include <stdlib.h>

/* define the MATLAB integer */
#ifdef IS64
#define Int mwSignedIndex
#else
#define Int int
#endif

/* turn off debugging */
#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
