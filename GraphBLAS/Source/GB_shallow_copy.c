//------------------------------------------------------------------------------
// GB_shallow_copy: create a shallow copy of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a purely shallow copy of a matrix.  No typecasting is done.

// The CSR/CSC format of C and A can differ, but they have they same vlen and
// vdim.  This function is CSR/CSC agnostic, except that C_is_csc is used to
// set the C->is_csc state in C.

// No errors are checked except for out-of-memory conditions.  This function is
// not user-callable.  Shallow matrices are never passed back to the user.

// Compare this function with GB_shallow_op.c
// A has any sparsity structure (hypersparse, sparse, bitmap, or full)

#include "GB_transpose.h"

#define GB_FREE_ALL ;

GB_PUBLIC                   // used by GraphBLAS MATLAB interface
GrB_Info GB_shallow_copy    // create a purely shallow matrix
(
    GrB_Matrix C,           // output matrix C, with a static header
    const bool C_is_csc,    // desired CSR/CSC format of C
    const GrB_Matrix A,     // input matrix
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (C != NULL) ;
    ASSERT (C->static_header) ;
    ASSERT_MATRIX_OK (A, "A for shallow copy", GB0) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    //--------------------------------------------------------------------------
    // construct a shallow copy of A for the pattern of C
    //--------------------------------------------------------------------------

    // allocate the struct for C, but do not allocate C->[p,h,b,i,x].
    // C has the exact same sparsity structure as A.
    GrB_Info info ;
    info = GB_new (&C, true, // sparse or hyper, static header
        A->type, A->vlen, A->vdim, GB_Ap_null, C_is_csc,
        GB_sparsity (A), A->hyper_switch, 0, Context) ;
    ASSERT (info == GrB_SUCCESS) ;

    //--------------------------------------------------------------------------
    // make a shallow copy of the vector pointers
    //--------------------------------------------------------------------------

    ASSERT (C->magic == GB_MAGIC2) ;    // C not yet initialized
    C->p_shallow = (A->p != NULL) ;     // C->p not freed when freeing C
    C->h_shallow = (A->h != NULL) ;     // C->h not freed when freeing C
    C->p = A->p ;                       // C->p is of size A->plen + 1
    C->h = A->h ;                       // C->h is of size A->plen
    C->plen = A->plen ;                 // C and A have the same hyperlist size
    C->nvec = A->nvec ;
    C->nvec_nonempty = A->nvec_nonempty ;
    C->jumbled = A->jumbled ;           // C is jumbled if A is jumbled
    C->nvals = A->nvals ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // check for empty matrix
    //--------------------------------------------------------------------------

    if (A->nzmax == 0)
    { 
        // C->p and C->h are shallow but the rest is empty
        C->nzmax = 0 ;
        C->b = NULL ;
        C->i = NULL ;
        C->x = NULL ;
        C->b_shallow = false ;
        C->i_shallow = false ;
        C->x_shallow = false ;
        C->jumbled = false ;
        ASSERT_MATRIX_OK (C, "C = quick copy of empty A", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // make a shallow copy of the pattern
    //--------------------------------------------------------------------------

    C->b = A->b ;                   // of size A->nzmax
    C->b_shallow = (A->b != NULL) ; // C->b will not be freed when freeing C

    C->i = A->i ;                   // of size A->nzmax
    C->i_shallow = (A->i != NULL) ; // C->i will not be freed when freeing C

    //--------------------------------------------------------------------------
    // make a shallow copy of the values
    //--------------------------------------------------------------------------

    C->nzmax = A->nzmax ;
    C->x = A->x ;
    C->x_shallow = (A->x != NULL) ; // C->x will not be freed when freeing C
    ASSERT (C->x_size == 0) ;       // C->x is shallow
    ASSERT_MATRIX_OK (C, "C = pure shallow (A)", GB0) ;
    return (GrB_SUCCESS) ;
}

