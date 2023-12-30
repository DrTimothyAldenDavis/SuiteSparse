//------------------------------------------------------------------------------
// CHOLMOD/Demo/cholmod_demo.h: include file for CHOLMOD demos
//------------------------------------------------------------------------------

// CHOLMOD/Demo Module.  Copyright (C) 2005-2023, Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cholmod.h"
#include "amd.h"
#include "colamd.h"
#ifndef NCAMD
#include "camd.h"
#include "ccolamd.h"
#endif
#include <time.h>
#define TRUE 1
#define FALSE 0

#define CPUTIME SUITESPARSE_TIME

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(a)   (((a) >= (0)) ? (a) : -(a))

