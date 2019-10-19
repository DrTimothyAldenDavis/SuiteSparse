/* ========================================================================= */
/* === camd_global ========================================================= */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD Version 2.1, Copyright (c) 2006 by Timothy A. Davis, Yanqing Chen,   */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.        */
/* web: http://www.cise.ufl.edu/research/sparse/camd                         */
/* ------------------------------------------------------------------------- */

#include <stdlib.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#include "matrix.h"
#endif

#ifndef NULL
#define NULL 0
#endif

/* ========================================================================= */
/* === Default CAMD memory manager ========================================= */
/* ========================================================================= */

/* The user can redefine these global pointers at run-time to change the memory
 * manager used by CAMD.  CAMD only uses malloc and free; realloc and calloc are
 * include for completeness, in case another package wants to use the same
 * memory manager as CAMD.
 *
 * If compiling as a MATLAB mexFunction, the default memory manager is mxMalloc.
 * You can also compile CAMD as a standard ANSI-C library and link a mexFunction
 * against it, and then redefine these pointers at run-time, in your
 * mexFunction.
 *
 * If -DNMALLOC is defined at compile-time, no memory manager is specified at
 * compile-time.  You must then define these functions at run-time, before
 * calling CAMD, for CAMD to work properly.
 */

#ifndef NMALLOC
#ifdef MATLAB_MEX_FILE
/* MATLAB mexFunction: */
void *(*camd_malloc) (size_t) = mxMalloc ;
void (*camd_free) (void *) = mxFree ;
void *(*camd_realloc) (void *, size_t) = mxRealloc ;
void *(*camd_calloc) (size_t, size_t) = mxCalloc ;
#else
/* standard ANSI-C: */
void *(*camd_malloc) (size_t) = malloc ;
void (*camd_free) (void *) = free ;
void *(*camd_realloc) (void *, size_t) = realloc ;
void *(*camd_calloc) (size_t, size_t) = calloc ;
#endif
#else
/* no memory manager defined at compile-time; you MUST define one at run-time */
void *(*camd_malloc) (size_t) = NULL ;
void (*camd_free) (void *) = NULL ;
void *(*camd_realloc) (void *, size_t) = NULL ;
void *(*camd_calloc) (size_t, size_t) = NULL ;
#endif

/* ========================================================================= */
/* === Default CAMD printf routine ========================================= */
/* ========================================================================= */

/* The user can redefine this global pointer at run-time to change the printf
 * routine used by CAMD.  If NULL, no printing occurs.  
 *
 * If -DNPRINT is defined at compile-time, stdio.h is not included.  Printing
 * can then be enabled at run-time by setting camd_printf to a non-NULL function.
 */

#ifndef NPRINT
#ifdef MATLAB_MEX_FILE
int (*camd_printf) (const char *, ...) = mexPrintf ;
#else
#include <stdio.h>
int (*camd_printf) (const char *, ...) = printf ;
#endif
#else
int (*camd_printf) (const char *, ...) = NULL ;
#endif
