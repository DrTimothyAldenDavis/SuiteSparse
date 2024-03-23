//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_init: initialize the GPUs for use by GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_cuda_init queries the system for the # of GPUs available, their memory
// sizes, SM counts, and other capabilities.  Unified Memory support is
// assumed.  Then each GPU is "warmed up" by allocating a small amount of
// memory.

#include "GB.h"

GrB_Info GB_cuda_init (void)
{

    // get the GPU properties
    if (!GB_Global_gpu_count_set (true)) return (GrB_PANIC) ;
    int gpu_count = GB_Global_gpu_count_get ( ) ;
    for (int device = 0 ; device < 1 ; device++) // TODO for GPU: gpu_count
    {
        // query the GPU and then warm it up
        if (!GB_Global_gpu_device_properties_get (device))
        {
            return (GrB_PANIC) ;
        }
    }

    // initialize RMM if necessary
    if (!rmm_wrap_is_initialized ())
    {
        rmm_wrap_initialize_all_same (rmm_wrap_managed,
            // FIXME ask the GPU(s) for good default values.  This might be
            // found by GB_cuda_init.  Perhaps GB_cuda_init needs to be split
            // into 2 methods: one to query the sizes(s) of the GPU(s) then
            // call rmm_wrap_initialize_all_same, and the other for the rest
            // of the work.  Alternatively, move GB_cuda_init here (if so,
            // ensure that it doesn't depend on any other initializations
            // below).
            256 * 1000000L, 256 * 100000000L, 1) ;
    }

    // warm up the GPUs
    for (int device = 0 ; device < 1 ; device++) // TODO for GPU: gpu_count
    {
        if (!GB_cuda_warmup (device))
        {
            return (GrB_PANIC) ;
        }
    }

    GB_cuda_set_device (0) ;            // make GPU 0 the default device
    GB_Context_gpu_id_set (NULL, 0) ;   // set GxB_CONTEXT_WORLD->gpu_id to 0

    // also check for jit cache, pre-load library of common kernels ...
    return (GrB_SUCCESS) ;
}

