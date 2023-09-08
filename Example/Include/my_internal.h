//------------------------------------------------------------------------------
// SuiteSparse/Example/Include/my_internal.h
//------------------------------------------------------------------------------

// Copyright (c) 2022-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Example include file for a user library.

// SuiteSparse include files for C/C++:
#include "amd.h"
#include "btf.h"
#include "camd.h"
#include "ccolamd.h"
#include "cholmod.h"
#include "colamd.h"
#include "cs.h"
#if ! defined (NO_GRAPHBLAS)
#  include "GraphBLAS.h"
#endif
#include "klu.h"
#include "ldl.h"
#include "RBio.h"
#include "SPEX.h"
#include "SuiteSparseQR_C.h"
#include "umfpack.h"

#ifdef __cplusplus
// SuiteSparse include files for C++:
#  include "SuiteSparseQR.hpp"
#  include "Mongoose.hpp"
#endif

// OpenMP include file:
#include <omp.h>

#include "my.h"
