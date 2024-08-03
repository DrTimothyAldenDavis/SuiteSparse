//------------------------------------------------------------------------------
// GB_reduce_to_scalar_iso: reduce an iso matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "reduce/GB_reduce.h"

void GB_reduce_to_scalar_iso        // s = reduce (A) where A is iso
(
    GB_void *restrict s,            // output scalar of type reduce->op->ztype
    GrB_Monoid monoid,              // monoid to use for the reduction
    GrB_Matrix A                    // matrix to reduce
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A->iso) ;
    ASSERT_MATRIX_OK (A, "A for reduce_to_scalar_iso", GB0) ;
    ASSERT_MONOID_OK (monoid, "monoid for reduce_to_scalar_iso", GB0) ;
    ASSERT (s != NULL) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // get input matrix and the monoid
    //--------------------------------------------------------------------------

    // A consists of n entries, all equal to Ax [0]
    uint64_t n = (uint64_t) (GB_nnz (A) - A->nzombies) ;
    ASSERT (n > 0) ;

    // get the monoid
    GxB_binary_function freduce = monoid->op->binop_function ;
    GrB_Type ztype = monoid->op->ztype ;
    size_t zsize = ztype->size ;
    GB_Type_code zcode = ztype->code ;

    // a = (ztype) Ax [0]
    GB_void a [GB_VLA(zsize)] ;
    GB_cast_scalar (a, zcode, A->x, A->type->code, zsize) ;

    //--------------------------------------------------------------------------
    // reduce n entries, all equal to a, to the scalar s, in O(log(n)) time
    //--------------------------------------------------------------------------

    if (n == INT64_MAX)
    { 
        // A has too many entries to reduce in a single step.  The only way
        // this can occur is if A is a huge full iso-valued matrix, where vlen
        // * vdim caused uint64_t overflow in GB_nnz_full and returned
        // INT64_MAX.  Reduce the matrix in two steps: first reducing each
        // vector of size vlen to a scalar t, obtainting an implicit iso full
        // vector T of size vdim.  Each entry in this vector T has the value t,
        // and then this vector T is reduced to the result s.
        GBURBLE ("(reduce huge iso full matrix to scalar) ") ;
        GB_void t [GB_VLA(zsize)] ;
        GB_reduce_worker_iso (t, freduce, a, A->vlen, zsize) ;
        GB_reduce_worker_iso (s, freduce, t, A->vdim, zsize) ;
    }
    else
    { 
        GBURBLE ("(reduce iso matrix to scalar) ") ;
        GB_reduce_worker_iso (s, freduce, a, n, zsize) ;
    }
}

