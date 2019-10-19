/* ========================================================================== */
/* === Demo/cholmod_demo.h ================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Demo Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * The CHOLMOD/Demo Module is licensed under Version 2.0 of the GNU
 * General Public License.  See gpl.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

#include "cholmod.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define TRUE 1
#define FALSE 0

#define CPUTIME ((double) (clock ( )) / CLOCKS_PER_SEC)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

