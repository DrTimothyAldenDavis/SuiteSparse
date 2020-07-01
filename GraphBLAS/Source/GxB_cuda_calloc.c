//------------------------------------------------------------------------------
// GxB_cuda_calloc.c: wrapper for cudaMallocManaged, or just calloc if no CUDA
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// DRAFT: in progress

#include "GB.h"

void *GxB_cuda_calloc (size_t n, size_t size)   // standard calloc signature
{
    #if defined ( GBCUDA )
    return (GB_cuda_calloc (n, size)) ;
    #else
    return (calloc (n, size)) ;
    #endif
}

