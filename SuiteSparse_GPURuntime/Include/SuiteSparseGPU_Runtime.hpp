// =============================================================================
// === SuiteSparse_GPURuntime/Include/SuiteSparse_GPURuntime.hpp ===============
// =============================================================================

// SuiteSparse_GPURuntime, Copyright (c) 2013-2016, Timothy A Davis,
// Sencer Nuri Yeralan, and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef SUITESPARSEGPU_RUNTIME_HPP
#define SUITESPARSEGPU_RUNTIME_HPP

#ifdef SUITESPARSE_CUDA

    #include "cuda_runtime.h"
    #include "SuiteSparse_config.h"

    #include <stdlib.h>

    #include "SuiteSparseGPU_macros.hpp"

    #if DEBUG_ATLEAST_ERRORONLY
    #include <stdio.h>
    #endif

    class Workspace;

    #include "SuiteSparseGPU_Workspace.hpp"

#endif

// Version information:
#define SUITESPARSE_GPURUNTIME_DATE "Nov 4, 2022"
#define SUITESPARSE_GPURUNTIME_MAIN_VERSION   2
#define SUITESPARSE_GPURUNTIME_SUB_VERSION    0
#define SUITESPARSE_GPURUNTIME_SUBSUB_VERSION 0

#define SUITESPARSE_GPURUNTIME_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define SUITESPARSE_GPURUNTIME_VERSION \
    SUITESPARSE_GPURUNTIME_VER_CODE(SUITESPARSE_GPURUNTIME_MAIN_VERSION,SUITESPARSE_GPURUNTIME_SUB_VERSION)

#endif
