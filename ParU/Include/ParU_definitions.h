// ============================================================================/
// ======================= ParU_definitions.h =================================/
// ============================================================================/

// ParU, Mohsen Aznaveh and Timothy A. Davis, (c) 2023, All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0
// some defintions that are used both in C and C++

#ifndef PARU_DEFINITIONS_H
#define PARU_DEFINITIONS_H


#include "SuiteSparse_config.h"
#include "cholmod.h"
#include "umfpack.h"

typedef enum ParU_Ret
{
    PARU_SUCCESS = 0,
    PARU_OUT_OF_MEMORY = -1,  
    PARU_INVALID = -2,
    PARU_SINGULAR = -3,
    PARU_TOO_LARGE = -4
} ParU_Ret;

#define PARU_MEM_CHUNK (1024*1024)

#define PARU_DATE "Jan 10, 2024"
#define PARU_VERSION_MAJOR  0
#define PARU_VERSION_MINOR  1
#define PARU_VERSION_UPDATE 1

#define PARU__VERSION SUITESPARSE__VERCODE(0,1,1)
#if !defined (SUITESPARSE__VERSION) || \
    (SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,5,0))
#error "ParU 0.1.1 requires SuiteSparse_config 7.5.0 or later"
#endif

#if !defined (UMFPACK__VERSION) || \
    (UMFPACK__VERSION < SUITESPARSE__VERCODE(6,3,1))
#error "ParU 0.1.1 requires UMFPACK 6.3.1 or later"
#endif

#if !defined (CHOLMOD__VERSION) || \
    (CHOLMOD__VERSION < SUITESPARSE__VERCODE(5,1,1))
#error "ParU 0.1.1 requires CHOLMOD 5.1.1 or later"
#endif

//  the same values as UMFPACK_STRATEGY defined in UMFPACK/Include/umfpack.h
#define PARU_STRATEGY_AUTO 0         // decided to use sym. or unsym. strategy
#define PARU_STRATEGY_UNSYMMETRIC 1  // COLAMD(A), metis, ...
#define PARU_STRATEGY_SYMMETRIC 3    // prefer diagonal

#endif //PARU_DEFINITIONS_H
