//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_malloc: malloc (int32 version)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#define CHOLMOD_ALLOC_FUNCTION      cholmod_malloc
#define SUITESPARSE_ALLOC_FUNCTION  SuiteSparse_malloc
#define CHOLMOD_INT32
#include "t_cholmod_malloc.c"

