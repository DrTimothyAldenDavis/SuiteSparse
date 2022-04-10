
//------------------------------------------------------------------------------
// GB_reduce_to_scalar_cuda.cu: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

extern "C"
{
#include "GB_reduce.h"
}

//#include "GB_cuda.h"
//#include "GB_jit_cache.h"
//
#include "jitFactory.hpp"
//#include "type_name.hpp"

GrB_Info GB_reduce_to_scalar_cuda
(
    GB_void *s,
    const GrB_Monoid reduce,
    const GrB_Matrix A,
    GB_Context Context
)
{

    //----------------------------------------------------------------------
    // reduce C to a scalar, just for testing:
    //----------------------------------------------------------------------

    int64_t nz = GB_nnz(A);

    GB_cuda_reduce( A, s, reduce);

    printf("num_triangles = %d\n",  s[0] );

    return GrB_SUCCESS ;
}

