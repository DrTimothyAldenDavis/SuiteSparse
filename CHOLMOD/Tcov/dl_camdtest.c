//------------------------------------------------------------------------------
// CHOLMOD/Tcov/dl_camdtest: double/int64_t version of camdtest
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define DOUBLE
#define CHOLMOD_INT64
#define DTYPE CHOLMOD_DOUBLE

#define DLONG
#include "t_camdtest.c"

