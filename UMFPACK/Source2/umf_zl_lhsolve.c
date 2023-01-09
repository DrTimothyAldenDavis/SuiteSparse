//------------------------------------------------------------------------------
// UMFPACK/Source2/umf_zl_lhsolve.c:
// double complex int64_t version of umf_ltsolve
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

#define ZLONG
#define CONJUGATE_SOLVE
#include "umf_ltsolve.c"

