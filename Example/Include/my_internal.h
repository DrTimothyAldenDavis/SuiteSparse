//------------------------------------------------------------------------------
// SuiteSparse/Example/Include/my_internal.h
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.

// SuiteSparse include files for C/C++:
#include "SuiteSparse_config.h"
#if !defined (SUITESPARSE_VERSION) || \
    (SUITESPARSE_VERSION < SUITESPARSE_VER_CODE(7,4))
#error "This library requires SuiteSparse_config 7.4.0 or later"
#endif

#include "amd.h"
#if AMD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires AMD 3.3.0 or later"
#endif

#include "btf.h"
#if AMD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires BTF 3.3.0 or later"
#endif

#include "camd.h"
#if CAMD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires CAMD 3.3.0 or later"
#endif

#include "ccolamd.h"
#if CCOLAMD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires CCOLAMD 3.3.0 or later"
#endif

#include "cholmod.h"
#if CHOLMOD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires CHOLMOD 5.1.0 or later"
#endif

#include "colamd.h"
#if COLAMD_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires COLAMD 3.3.0 or later"
#endif

#include "cs.h"
// #if ! defined (CXSPARSE) || (CS_VERSION < SUITESPARSE_VER_CODE(3,3))
// #error "This library requires CXSparse 3.3.0 or later"
// #endif

#if ! defined (NO_GRAPHBLAS)
    #include "GraphBLAS.h"
    #if !defined ( GxB_SUITESPARSE_GRAPHBLAS ) || \
        GxB_IMPLEMENTATION < GxB_VERSION(8,3,0)
    #error "This library requires SuiteSparse:GraphBLAS 8.3.0 or later"
    #endif
#endif

#if ! defined (NO_LAGRAPH)
    #include "LAGraph.h"
    #if GxB_VERSION (LAGRAPH_VERSION_MAJOR,LAGRAPH_VERSION_MINOR,   \
        LAGRAPH_VERSION_UPDATE) < GxB_VERSION(1,1,0)
    #error "This library requires LAGraph 1.1.0 or later"
    #endif
#endif

#include "klu.h"
#if KLU_VERSION < SUITESPARSE_VER_CODE(2,3)
#error "This library requires KLU 2.3.0 or later"
#endif

#include "ldl.h"
#if LDL_VERSION < SUITESPARSE_VER_CODE(3,3)
#error "This library requires LDL 3.3.0 or later"
#endif

#include "RBio.h"
#if RBIO_VERSION < SUITESPARSE_VER_CODE(4,3)
#error "This library requires RBio 4.3.0 or later"
#endif

#include "SPEX.h"
#if GxB_VERSION (SPEX_VERSION_MAJOR,SPEX_VERSION_MINOR,   \
        SPEX_VERSION_SUB) < GxB_VERSION(2,3,0)
#error "This library requires SPEX 2.3.0 or later"
#endif

#include "SuiteSparseQR_C.h"
#if SPQR_VERSION < SUITESPARSE_VER_CODE(4,3)
#error "This library requires SPQR 4.3.0 or later"
#endif

#include "umfpack.h"
#if UMFPACK_VER < SUITESPARSE_VER_CODE(4,3)
#error "This library requires UMFPACK 6.3.0 or later"
#endif

// SuiteSparse include files for C++:
#ifdef __cplusplus
    #include "SuiteSparseQR.hpp"

    #include "Mongoose.hpp"
    #if GxB_VERSION (Mongoose_VERSION_MAJOR,Mongoose_VERSION_MINOR,   \
        Mongoose_VERSION_PATCH) < GxB_VERSION(3,3,0)
    #error "This library requires Mongoose 3.3.0 or later"
    #endif

#endif

// OpenMP include file:
#include <omp.h>

#include "my.h"

