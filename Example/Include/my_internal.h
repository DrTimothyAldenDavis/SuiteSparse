//------------------------------------------------------------------------------
// SuiteSparse/Example/Include/my_internal.h
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.

#ifndef MY_INTERNAL_H
#define MY_INTERNAL_H

// SuiteSparse include files for C/C++:
#include "SuiteSparse_config.h"
#if !defined (SUITESPARSE__VERSION) || SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,5,1)
#error "This library requires SuiteSparse_config 7.5.1 or later"
#endif

#include "amd.h"
#if !defined (AMD__VERSION) || AMD__VERSION < SUITESPARSE__VERCODE(3,3,1)
#error "This library requires AMD 3.3.1 or later"
#endif

#include "btf.h"
#if !defined (BTF__VERSION) || BTF__VERSION < SUITESPARSE__VERCODE(2,3,1)
#error "This library requires BTF 2.3.0 or later"
#endif

#include "camd.h"
#if !defined (CAMD__VERSION) || CAMD__VERSION < SUITESPARSE__VERCODE(3,3,1)
#error "This library requires CAMD 3.3.1 or later"
#endif

#include "ccolamd.h"
#if !defined (CCOLAMD__VERSION) || CCOLAMD__VERSION < SUITESPARSE__VERCODE(3,3,1)
#error "This library requires CCOLAMD 3.3.1 or later"
#endif

#include "cholmod.h"
#if !defined (CHOLMOD__VERSION) || CHOLMOD__VERSION < SUITESPARSE__VERCODE(5,1,1)
#error "This library requires CHOLMOD 5.1.1 or later"
#endif

#include "colamd.h"
#if !defined (COLAMD__VERSION) || COLAMD__VERSION < SUITESPARSE__VERCODE(3,3,1)
#error "This library requires COLAMD 3.3.1 or later"
#endif

#include "cs.h"
#if !defined (CXSPARSE__VERSION) || CXSPARSE__VERSION < SUITESPARSE__VERCODE(4,3,1)
#error "This library requires CXSparse 4.3.1 or later"
#endif

#if ! defined (NO_GRAPHBLAS)
    #include "GraphBLAS.h"
    #if !defined ( GxB_SUITESPARSE_GRAPHBLAS ) || \
        GxB_IMPLEMENTATION < GxB_VERSION (9,0,0)
    #error "This library requires SuiteSparse:GraphBLAS 9.0.0 or later"
    #endif
#endif

#if ! defined (NO_LAGRAPH)
    #include "LAGraph.h"
    #if SUITESPARSE__VERCODE(LAGRAPH_VERSION_MAJOR,LAGRAPH_VERSION_MINOR,LAGRAPH_VERSION_UPDATE) < SUITESPARSE__VERCODE(1,1,1)
    #error "This library requires LAGraph 1.1.1 or later"
    #endif
#endif

#include "klu.h"
#if !defined (KLU__VERSION) || KLU__VERSION < SUITESPARSE__VERCODE(2,3,1)
#error "This library requires KLU 2.3.1 or later"
#endif

#include "ldl.h"
#if !defined (LDL__VERSION) || LDL__VERSION < SUITESPARSE__VERCODE(3,3,1)
#error "This library requires LDL 3.3.1 or later"
#endif

#include "RBio.h"
#if !defined (RBIO__VERSION) || RBIO__VERSION < SUITESPARSE__VERCODE(4,3,1)
#error "This library requires RBio 4.3.1 or later"
#endif

#include "SPEX.h"
#if !defined (SPEX__VERSION) || SPEX__VERSION < SUITESPARSE__VERCODE(2,3,1)
#error "This library requires SPEX 2.3.1 or later"
#endif

#include "SuiteSparseQR_C.h"
#if !defined (SPQR__VERSION) || SPQR__VERSION < SUITESPARSE__VERCODE(4,3,1)
#error "This library requires SPQR 4.3.1 or later"
#endif

#include "umfpack.h"
#if !defined (UMFPACK__VERSION) || UMFPACK__VERSION < SUITESPARSE__VERCODE(6,3,1)
#error "This library requires UMFPACK 6.3.1 or later"
#endif

// SuiteSparse include files for C++:
#ifdef __cplusplus
    #include "SuiteSparseQR.hpp"

    #include "Mongoose.hpp"
    #if !defined (Mongoose__VERSION) || Mongoose__VERSION < SUITESPARSE__VERCODE(3,3,1)
    #error "This library requires Mongoose 3.3.1 or later"
    #endif

#endif

// OpenMP include file:
#include <omp.h>

#include "my.h"

#endif
