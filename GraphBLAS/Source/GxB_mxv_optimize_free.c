//------------------------------------------------------------------------------
// GxB_mxv_optimize: optimize a matrix for matrix-vector multiply
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_mkl.h"

GrB_Info GxB_mxv_optimize_free      // analyze A for subsequent use in mxv
(
    GrB_Matrix A                    // input/output matrix
)
{
#if GB_HAS_MKL_GRAPH

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE ("GxB_mxv_optimize_free (A, desc)") ;
    GB_BURBLE_START ("GxB_mxv_optimize_free") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    //--------------------------------------------------------------------------
    // free any existing MKL version of the matrix A and its optimization
    //--------------------------------------------------------------------------

    GB_MKL_GRAPH_MATRIX_DESTROY (A->mkl) ;

    A->mkl = NULL ;
#endif
    return (GrB_SUCCESS) ;
}

