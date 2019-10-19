/* ========================================================================== */
/* === ccolamd_global.c ===================================================== */
/* ========================================================================== */

/* ----------------------------------------------------------------------------
 * CCOLAMD version 2.5.
 * Copyright (C) 2005, Univ. of Florida.  Authors: Timothy A. Davis,
 * Sivasankaran Rajamanickam, and Stefan Larimore
 * See License.txt for the Version 2.1 of the GNU Lesser General Public License
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Global variables for CCOLAMD */

#ifndef NPRINT
#ifdef MATLAB_MEX_FILE
#include "mex.h"
int (*ccolamd_printf) (const char *, ...) = mexPrintf ;
#else
#include <stdio.h>
int (*ccolamd_printf) (const char *, ...) = printf ;
#endif
#else
int (*ccolamd_printf) (const char *, ...) = ((void *) 0) ;
#endif

