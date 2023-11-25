//------------------------------------------------------------------------------
// CHOLMOD/Tcov/sl_test: single/int64_t version of Tcov/cm test program
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#define SINGLE
#define CHOLMOD_INT64
#define DTYPE CHOLMOD_SINGLE

#include "t_cm.c"
#include "t_test_ops.c"
#include "t_test_ops2.c"
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
#include "t_basic.c"
#include "t_overflow_tests.c"
#include "t_dump.c"
#include "t_read_triplet.c"
#include "t_rhs.c"
#include "t_znorm_diag.c"
#include "t_prand.c"
#include "t_perm_matrix.c"
#include "t_ptest.c"
#include "t_rand_dense.c"
#include "t_cat_tests.c"
#include "t_dense_tests.c"
#include "t_dtype_tests.c"
#include "t_common_tests.c"
#include "t_error_tests.c"
#include "t_tofrom_tests.c"
#include "t_suitesparse.c"
