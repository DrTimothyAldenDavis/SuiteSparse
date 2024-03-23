//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda.hpp: include file for host CUDA methods (not for JIT)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_HPP
#define GB_CUDA_HPP

extern "C"
{ 
    #include "GB_dev.h"
    #include "GB_compiler.h"
    #include "GB_cpu_features.h"
    #include "GB_warnings.h"
}

#include "GraphBLAS_cuda.hpp"

extern "C"
{
    #include <cassert>
    #include <cmath>
    #include "GB.h"
    #include "GB_stringify.h"
    #include "GB_iceil.h"
}

// Finally, include the CUDA definitions
#include "cuda_runtime.h"
#include "cuda.h"

#include <limits>
#include <iostream>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>

#include "GB_cuda_error.hpp"
#include "GB_cuda_timer.hpp"

//------------------------------------------------------------------------------
// prefetch and memadvise
//------------------------------------------------------------------------------

// for the "which" parameter of GB_cuda_matrix_prefetch:
// FIXME: rename this to GB_WHATEVER_P for GB_cuda_matrix_advise

#define GB_PREFETCH_P   1
#define GB_PREFETCH_H   2
#define GB_PREFETCH_Y   4
#define GB_PREFETCH_B   8
#define GB_PREFETCH_I  16
#define GB_PREFETCH_X  32
#define GB_PREFETCH_PIX   (GB_PREFETCH_P + GB_PREFETCH_I + GB_PREFETCH_X)
#define GB_PREFETCH_PYI   (GB_PREFETCH_P + GB_PREFETCH_Y + GB_PREFETCH_I)
#define GB_PREFETCH_PYBI  (GB_PREFETCH_PYI + GB_PREFETCH_B)
#define GB_PREFETCH_PYBIX (GB_PREFETCH_PYBI + GB_PREFETCH_X)
#define GB_PREFETCH_PHI   (GB_PREFETCH_P + GB_PREFETCH_H + GB_PREFETCH_I)
#define GB_PREFETCH_PHBI  (GB_PREFETCH_PHI + GB_PREFETCH_B)
#define GB_PREFETCH_PHBIX (GB_PREFETCH_PHBI + GB_PREFETCH_X)

GrB_Info GB_cuda_matrix_prefetch
(
    GrB_Matrix A,
    int which,              // which components to prefetch (phybix control)
    int device,             // GPU device or cudaCpuDeviceId
    cudaStream_t stream
) ;

#if 0
// do we need this function too?
GrB_Info GB_cuda_matrix_advise
(
    GrB_Matrix A,

    p, h, y, b, i, x?   6 bools

    what to do:  advise (prefer location? access by)?  prefetch? nothing?
        avdice: enum (1 to 6)

    int device,             // GPU device or cudaCpuDeviceId
) ;
#endif

void GB_cuda_upscale_identity
(
    GB_void *identity_upscaled,     // output: at least sizeof (uint32_t)
    GrB_Monoid monoid               // input: monoid to upscale
) ;

#endif

