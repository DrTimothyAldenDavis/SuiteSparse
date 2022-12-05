//------------------------------------------------------------------------------
// SuiteSparse/Example/[Config or Include]/my.h
//------------------------------------------------------------------------------

// Copyright (c) 2022, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.  Do not edit the Include/my.h
// file, since it is constructed from Config/my.h.in by cmake.

#ifdef __cplusplus
extern "C" {
#endif
    // SuiteSparse include files for C:
    #include "amd.h"
    #include "btf.h"
    #include "camd.h"
    #include "ccolamd.h"
    #include "cholmod.h"
    #include "colamd.h"
    #include "cs.h"
    #include "GraphBLAS.h"
    #include "klu.h"
    #include "ldl.h"
    #include "RBio.h"
    #include "SPEX.h"
    #include "SuiteSparseQR_C.h"
    #include "umfpack.h"
    // ANSI C include files:
    #include <stdio.h>
    #include <string.h>
    #include <math.h>
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// SuiteSparse include files for C++:
#include "SuiteSparseQR.hpp"
#include "Mongoose.hpp"
#endif

// OpenMP include file:
#include <omp.h>

// version and date for example user library
#define MY_DATE "Dec 9, 2022"
#define MY_MAJOR_VERSION 1
#define MY_MINOR_VERSION 2
#define MY_PATCH_VERSION 3

void my_library (int version [3], char date [128]) ;
void my_function (void) ;

