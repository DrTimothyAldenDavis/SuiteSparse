// =============================================================================
// SuiteSparse_GPURuntime/Include/SuiteSparseGPU_internal.hpp
// =============================================================================

// SuiteSparse_GPURuntime, Copyright (c) 2013-2016, Timothy A Davis,
// Sencer Nuri Yeralan, and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef SUITESPARSEGPU_INTERNAL_HPP
#define SUITESPARSEGPU_INTERNAL_HPP

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

#include "SuiteSparse_GPURuntime.hpp"

#endif
