//------------------------------------------------------------------------------
// CHOLMOD/Tcov/di_read: double/int32_t version of cmread
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define DOUBLE
#define CHOLMOD_INT32
#define DTYPE CHOLMOD_DOUBLE

#include "t_cmread.c"
