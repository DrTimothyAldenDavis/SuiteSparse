//------------------------------------------------------------------------------
// CHOLMOD/Tcov/dl_read: double/int64_t version of cmread
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define DOUBLE
#define CHOLMOD_INT64
#define DTYPE CHOLMOD_DOUBLE

#include "t_cmread.c"
