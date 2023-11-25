//------------------------------------------------------------------------------
// CHOLMOD/Tcov/sl_camdtest: single/int64_t version of camdtest
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define SINGLE
#define CHOLMOD_INT64
#define DTYPE CHOLMOD_SINGLE

#define DLONG
#include "t_camdtest.c"

