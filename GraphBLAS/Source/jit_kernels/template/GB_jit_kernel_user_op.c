//------------------------------------------------------------------------------
// GB_jit_kernel_user_op.c: JIT kernel for a user-defined operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GB_JIT_GLOBAL GB_JIT_KERNEL_USER_OP_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_USER_OP_PROTO (GB_jit_kernel)
{
    (*user_function) = ((void *) GB_USER_OP_FUNCTION) ;
    (*defn) = GB_USER_OP_DEFN ;
    return (GrB_SUCCESS) ;
}

