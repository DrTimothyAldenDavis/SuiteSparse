// =============================================================================
// SuiteSparse_GPURuntime/Include/SuiteSparseGPU_workspace_macros.hpp
// =============================================================================

// SuiteSparse_GPURuntime, Copyright (c) 2013-2016, Timothy A Davis,
// Sencer Nuri Yeralan, and Sanjay Ranka.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#ifndef SUITESPARSE_GPURUNTIME_WORKSPACE_MACROS_HPP
#define SUITESPARSE_GPURUNTIME_WORKSPACE_MACROS_HPP

#ifndef GPU_REFERENCE
#define GPU_REFERENCE(WORKSPACE, TYPE) \
    ((TYPE) (WORKSPACE != NULL ? (WORKSPACE)->gpu() : NULL))
#endif

#ifndef CPU_REFERENCE
#define CPU_REFERENCE(WORKSPACE, TYPE) \
    ((TYPE) (WORKSPACE != NULL ? (WORKSPACE)->cpu() : NULL))
#endif

#endif
