// =============================================================================
// === GPUQREngine/Include/GPUQREngine_Stats.hpp ===============================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// The QREngineStats structure wraps data members responsible for capturing
// runtime characteristics of the factorization.
//
// =============================================================================

#ifndef GPUQRENGINE_STATS_HPP
#define GPUQRENGINE_STATS_HPP

template <typename Int = int64_t> struct QREngineStats
{
    float kernelTime;           // The total time spent in the kernel.
                                // This time is accumulated across multiple
                                // kernel invocations.

    Int numLaunches;            // The total number of kernel launches.

    int64_t flopsActual;        // The total number of flops performed.

    QREngineStats()
    {
        kernelTime = 0.0;
        numLaunches = 0;
        flopsActual = 0;
    }
};

#endif
