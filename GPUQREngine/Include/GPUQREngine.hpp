// =============================================================================
// === GPUQREngine/Include/GPUQREngine.hpp =====================================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// This is the main user-level include file.
//
// =============================================================================

#ifndef GPUQRENGINE_HPP
#define GPUQRENGINE_HPP

#include "SuiteSparseGPU_Runtime.hpp"
#include "GPUQREngine_Front.hpp"
#include "GPUQREngine_Stats.hpp"

enum QREngineResultCode
{
    QRENGINE_SUCCESS,           // GPU QR was successfull
    QRENGINE_OUTOFMEMORY,       // GPU QR ran out of memory
    QRENGINE_GPUERROR           // failed to communicated with the GPU
};

// Use C++ Polymorphism to provide many different function signatures and
// call patterns.
QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,
    Front *userFronts,
    Int numFronts,
    QREngineStats *stats = NULL
);

QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,
    Front *userFronts,
    Int numFronts,
    Int *Parent,
    Int *Childp,
    Int *Child,
    QREngineStats *stats = NULL
);

Int *GPUQREngine_FindStaircase
(
    Front *front                // The front whose staircase we are computing
);

// Version information:
#define GPUQRENGINE_DATE "Nov 12, 2022"
#define GPUQRENGINE_MAIN_VERSION   2
#define GPUQRENGINE_SUB_VERSION    0
#define GPUQRENGINE_SUBSUB_VERSION 0

#define GPUQRENGINE_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define GPUQRENGINE_VERSION \
    GPUQRENGINE_VER_CODE(GPUQRENGINE_MAIN_VERSION,GPUQRENGINE_SUB_VERSION)

#endif
