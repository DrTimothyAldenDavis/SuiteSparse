//------------------------------------------------------------------------------
// CHOLMOD/Demo/cholmod_demo.h: include file for CHOLMOD demos
//------------------------------------------------------------------------------

// CHOLMOD/Demo Module.  Copyright (C) 2005-2022, Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod.h"
#include <time.h>
#define TRUE 1
#define FALSE 0

#define CPUTIME (SuiteSparse_time ( ))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(a)   (((a) >= (0)) ? (a) : -(a))

