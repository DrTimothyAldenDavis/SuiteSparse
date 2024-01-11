//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_common_jitFactory.hpp: for all jitFactory classes
//------------------------------------------------------------------------------

// (c) Nvidia Corp. 2023 All rights reserved
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Common defines for all jitFactory classes:
// iostream callback to deliver the buffer to jitify as if read from a file
// compiler flags
// Include this file along with any jitFactory you need.

// NOTE: do not edit the GB_cuda_common_jitFactory.hpp directly.  It is
// configured by cmake from the following file:
// GraphBLAS/CUDA/Config/GB_cuda_common_jitFactory.hpp.in

#ifndef GB_CUDA_COMMON_JITFACTORY_HPP
#define GB_CUDA_COMMON_JITFACTORY_HPP

#pragma once

#include "GraphBLAS_cuda.h"

extern "C"
{
    #include "GB.h"
    #include "GB_stringify.h"
}

#include <iostream>
#include <cstdint>
#include "GB_cuda_jitify_cache.h"
#include "GB_cuda_jitify_launcher.h"
#include "GB_cuda_mxm_factory.hpp"
#include "GB_cuda_error.h"
#include "../rmm_wrap/rmm_wrap.h"
#include "GB_iceil.h"

// amount of shared memory to use in CUDA kernel launches
constexpr unsigned int SMEM = 0 ;

#if 0

static const std::vector<std::string> GB_jit_cuda_compiler_flags{   // OLD
   "-std=c++17",
   //"-G",
   "-remove-unused-globals",
   "-w",
   "-D__CUDACC_RTC__",
// "-I" + jit::get_user_home_cache_dir(),   // FIXME: add +/cu/00
// "-I" + jit::get_user_home_cache_dir() + "/src",
   "-I/usr/local/cuda/include",
   // FIXME: add SUITESPARSE_CUDA_ARCHITECTURES here, via config
};

#endif

inline std::vector<std::string> GB_cuda_jit_compiler_flags ( )
{
    return (
        std::vector<std::string>  (
        {"-std=c++17",
        //"-G",
        "-remove-unused-globals",
        "-w",
        "-D__CUDACC_RTC__",
        "-I" + jit::get_user_home_cache_dir(),   // FIXME: add +/cu/00
        "-I" + jit::get_user_home_cache_dir() + "/src",
        "-I/usr/local/cuda/include"
        // FIXME: add SUITESPARSE_CUDA_ARCHITECTURES here, via config
        })) ;
} ;

// FIXME: rename GB_jit_cuda_header_names or something
static const std::vector<std::string> header_names ={};

// FIXME: rename GB_jit_cuda_file_callback
inline std::istream* (*file_callback)(std::string, std::iostream&);

#endif
