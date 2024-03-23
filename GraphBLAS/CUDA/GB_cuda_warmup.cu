//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_warmup.cu: warmup the GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_warmup (int device)
{
    
    //--------------------------------------------------------------------------
    // set the device
    //--------------------------------------------------------------------------

    if (!GB_cuda_set_device (device))
    {
        // invalid device
        return (false) ;
    }

    // FIXME: why do we need this?
    double gpu_memory_size = GB_Global_gpu_memorysize_get (device) ;

    //--------------------------------------------------------------------------
    // allocate two small blocks just to load the drivers
    //--------------------------------------------------------------------------

    size_t size = 0 ;
    void *p = GB_malloc_memory (1, 1, &size) ;
    if (p == NULL)
    {
        // no memory on the device
        return (false) ;
    }
    GB_free_memory (&p, size) ;

    cudaMalloc (&p, size ) ;
    if (p == NULL)
    {
        // no memory on the device
        return (false) ;
    }
    cudaFree (p) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (true) ;
}

