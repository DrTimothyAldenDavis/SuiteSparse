// CXSparse/Tcov/cstcov_malloc_test.h: include file for testing
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs.h"
#define malloc_count CS_NAME (_malloc_count)
#ifndef EXTERN
#define EXTERN extern
#endif
EXTERN int malloc_count ;

