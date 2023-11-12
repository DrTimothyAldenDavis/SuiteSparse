//------------------------------------------------------------------------------
// CHOLMOD/Tcov/dl_test: double/int64_t version of Tcov/cm test program
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define DOUBLE
#define CHOLMOD_INT64
#define DTYPE CHOLMOD_DOUBLE

#include "t_cm.c"
#include "t_test_ops.c"
#include "t_null.c"
#include "t_null2.c"
#include "t_lpdemo.c"
#include "t_memory.c"
#include "t_solve.c"
#include "t_aug.c"
#include "t_unpack.c"
#include "t_raw_factor.c"
#include "t_cctest.c"
#include "t_ctest.c"
