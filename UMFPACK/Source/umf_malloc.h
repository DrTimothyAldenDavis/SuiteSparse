//------------------------------------------------------------------------------
// UMFPACK/Source/umf_malloc.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef _UMF_MALLOC
#define _UMF_MALLOC

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
extern Int UMF_malloc_count ;
#endif

void *UMF_malloc
(
    Int n_objects,
    size_t size_of_object
) ;

#endif
