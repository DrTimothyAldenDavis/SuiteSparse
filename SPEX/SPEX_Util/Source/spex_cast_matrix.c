//------------------------------------------------------------------------------
// SPEX_Util/spex_cast_matrix: create a dense typecasted matrix
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// spex_cast_matrix constructs a dense nz-by-1 matrix Y that holds the
// typecasted values of the input matrix A.  The input matrix A can be of any
// kind (CSC, triplet, or dense) and any type.

#define SPEX_FREE_ALL                   \
    SPEX_matrix_free (&Y, option) ;

#include "spex_util_internal.h"

SPEX_info spex_cast_matrix
(
    SPEX_matrix **Y_handle,     // nz-by-1 dense matrix to create
    SPEX_type Y_type,           // type of Y
    SPEX_matrix *A,             // matrix with nz entries
    const SPEX_options *option  // Command options, if NULL defaults are used
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // inputs have been checked in the only caller SPEX_matrix_copy
#if 0
    if (Y_handle == NULL || A == NULL)
    {
        return (SPEX_INCORRECT_INPUT) ;
    }
    if (nz < 0)
    {
        return (SPEX_INCORRECT_INPUT) ;
    }
    (*Y_handle) = NULL ;
#endif

    int64_t nz;
    SPEX_info info = SPEX_OK ;
    SPEX_matrix *Y = NULL ;
    SPEX_CHECK (SPEX_matrix_nnz (&nz, A, option)) ;


    //--------------------------------------------------------------------------
    // allocate Y (shallow if Y_type is the same as A->type)
    //--------------------------------------------------------------------------

    SPEX_CHECK (SPEX_matrix_allocate (&Y, SPEX_DENSE, Y_type,
        nz, 1, nz, Y_type == A->type, true, option)) ;

    //--------------------------------------------------------------------------
    // typecast the values from A into Y
    //--------------------------------------------------------------------------

    if (Y_type == A->type)
    {

        //----------------------------------------------------------------------
        // Y is shallow; just copy in the pointer of the values of A
        //----------------------------------------------------------------------

        switch (Y_type) // checked in SPEX_matrix_copy
        {
            case SPEX_MPZ:   Y->x.mpz   = A->x.mpz   ;
            break;
            case SPEX_MPQ:   Y->x.mpq   = A->x.mpq   ;
            break;
            case SPEX_MPFR:  Y->x.mpfr  = A->x.mpfr  ;
            break;
            case SPEX_INT64: Y->x.int64 = A->x.int64 ;
            break;
            case SPEX_FP64:  Y->x.fp64  = A->x.fp64  ;
            break;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // Y is deep; typecast the values from A into Y
        //----------------------------------------------------------------------

        SPEX_CHECK (spex_cast_array (SPEX_X (Y), Y->type,
            SPEX_X (A), A->type, nz, Y->scale, A->scale, option)) ;

    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*Y_handle) = Y;
    SPEX_CHECK (info) ;
    return (SPEX_OK) ;
}

