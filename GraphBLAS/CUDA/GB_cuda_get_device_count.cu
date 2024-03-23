//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_get_device_count.cu: find out how many GPUs exist
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_get_device_count   // true if OK, false if failure
(
    int *gpu_count              // return # of GPUs in the system
)
{
    cudaError_t err = cudaGetDeviceCount (gpu_count) ;
    return (err == cudaSuccess) ;
}

