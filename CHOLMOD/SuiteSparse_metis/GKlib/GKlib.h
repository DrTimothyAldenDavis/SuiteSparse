/*
 * GKlib.h
 * 
 * George's library of most frequently used routines
 *
 * $Id: GKlib.h 13005 2012-10-23 22:34:36Z karypis $
 *
 */

#ifndef _GKLIB_H_
#define _GKLIB_H_ 1

#define GKMSPACE

#if defined(_MSC_VER)
#define __MSC__
#endif
#if defined(__ICC)
#define __ICC__
#endif


#include "gk_arch.h" /*!< This should be here, prior to the includes */


/*************************************************************************
* Header file inclusion section
**************************************************************************/
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
/* -------------------------------------------------------------------------- */
/* Added for SuiteSparse, to disable signal handling when incorporated into
 * a MATLAB mexFunction.  Tim Davis, Jan 30, 2016, Texas A&M University. */
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define SIGABRT 6
#define SIGTERM 15
#define jmp_buf int
#define raise(sig) { mexErrMsgTxt ("METIS error") ; }
#define signal(sig,func) NULL
#define longjmp(env,val) { mexErrMsgTxt ("METIS error") ; }
#define setjmp(x) (0)
#define exit(x) { mexErrMsgTxt ("METIS error") ; }
#else
#include <signal.h>
#include <setjmp.h>
#endif
/* -------------------------------------------------------------------------- */
#include <assert.h>
#include <sys/stat.h>

#if 0
// regex.h and gk_regex.h disabled for SuiteSparse, Jan 1, 2023.
#if defined(__WITHPCRE__)
  #include <pcreposix.h>
#else
  #if defined(USE_GKREGEX)
    #include "gkregex.h"
  #else
    #include <regex.h>
  #endif /* defined(USE_GKREGEX) */
#endif /* defined(__WITHPCRE__) */
#endif



#if defined(__OPENMP__) 
#include <omp.h>
#endif

/* -------------------------------------------------------------------------- */
/* Added for incorporation into SuiteSparse.
   Tim Davis, Oct 31, 2022, Texas A&M University. */
#include "SuiteSparse_config.h"
#define malloc  SuiteSparse_config_malloc
#define calloc  SuiteSparse_config_calloc
#define realloc SuiteSparse_config_realloc
#define free(p)                                 \
{                                               \
    if ((p) != NULL)                            \
    {                                           \
        SuiteSparse_config_free (p) ;           \
        (p) = NULL ;                            \
    }                                           \
}

/* -------------------------------------------------------------------------- */




#include <gk_types.h>
#include <gk_struct.h>
#include <gk_externs.h>
#include <gk_defs.h>
#include <gk_macros.h>
#include <gk_getopt.h>

#include <gk_mksort.h>
#include <gk_mkblas.h>
#include <gk_mkmemory.h>
#include <gk_mkpqueue.h>
#include <gk_mkpqueue2.h>
#include <gk_mkrandom.h>
#include <gk_mkutils.h>

#include <gk_proto.h>


#endif  /* GKlib.h */


