// =============================================================================
// === GPUQREngine/Source/GPUQREngine_ExpertSparse.cpp =========================
// =============================================================================

// GPUQREngine, Copyright (c) 2013, Timothy A Davis, Sencer Nuri Yeralan,
// and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------
//
// This file contains the sparse GPUQREngine wrapper that simply calls down into
// the Internal GPUQREngine factorization routine.
//
// =============================================================================

#ifdef SUITESPARSE_CUDA

#include "GPUQREngine_Internal.hpp"

template <typename Int>
QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <Int> *userFronts,      // The list of fronts to factorize
    Int numFronts,          // The number of fronts to factorize
    Int *Parent,            // The front-to-parent mapping
    Int *Childp,            // Front-to-child column pointers
    Int *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <Int> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
)
{
    return (GPUQREngine_Internal (gpuMemorySize, userFronts, numFronts, Parent,
        Childp, Child, stats)) ;
}

template QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <int32_t> *userFronts,      // The list of fronts to factorize
    int32_t numFronts,          // The number of fronts to factorize
    int32_t *Parent,            // The front-to-parent mapping
    int32_t *Childp,            // Front-to-child column pointers
    int32_t *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <int32_t> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
) ;
template QREngineResultCode GPUQREngine
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front <int64_t> *userFronts,      // The list of fronts to factorize
    int64_t numFronts,          // The number of fronts to factorize
    int64_t *Parent,            // The front-to-parent mapping
    int64_t *Childp,            // Front-to-child column pointers
    int64_t *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats <int64_t> *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
) ;
#endif

