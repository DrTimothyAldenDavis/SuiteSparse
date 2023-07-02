//------------------------------------------------------------------------------
// GPUQREngine/Source/Scheduler/ssgpu_maxQueueSize.cpp
//------------------------------------------------------------------------------

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
#ifdef SUITESPARSE_CUDA
#include "GPUQREngine_Scheduler.hpp"
#define MIN_QUEUE_SIZE 50000

size_t ssgpu_maxQueueSize       // return size of scheduler queue
(
    size_t gpuMemorySize        // size of GPU memory, in bytes
)
{
    size_t maxQueueSize ;
    size_t tileSizeBytes = TILESIZE * TILESIZE * sizeof(double);
    if (gpuMemorySize <= 1)
    {
        // GPU memory size not specified, use minimal amount.
        // This case is for testing with small matrices.
        maxQueueSize = MIN_QUEUE_SIZE ;
    }
    else
    {
        maxQueueSize = 2 * (gpuMemorySize/tileSizeBytes) / SSGPU_MINAPPLYGRANULARITY ;
    }

    // ensure the queue size is reasonably large enough
    maxQueueSize = MAX (maxQueueSize, MIN_QUEUE_SIZE) ;
    return (maxQueueSize) ;
}

#endif
