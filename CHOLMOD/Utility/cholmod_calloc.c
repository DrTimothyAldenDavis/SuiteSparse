//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_calloc: calloc (int32 version)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#define CHOLMOD_ALLOC_FUNCTION      cholmod_calloc
#define SUITESPARSE_ALLOC_FUNCTION  SuiteSparse_calloc
#define CHOLMOD_INT32
#include "t_cholmod_malloc.c"

