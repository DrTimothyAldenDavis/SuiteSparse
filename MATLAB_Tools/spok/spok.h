#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define SPOK_INT mwSignedIndex
#else
/* for use outside of MATLAB, use with -DSPOK_INT=long to get long version of
   code */
#ifndef SPOK_INT
#define SPOK_INT int
#endif
#endif

#define SPOK_OK 1
#define SPOK_WARNING 0

#define SPOK_FATAL_M (-1)
#define SPOK_FATAL_N (-2)
#define SPOK_FATAL_NZMAX (-3)
#define SPOK_FATAL_P (-4)
#define SPOK_FATAL_I (-5)
