//------------------------------------------------------------------------------
// GB_jit_kernel_split_full: split full A into a full tile C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// cij = op (aij)
#define GB_COPY(pC,pA) GB_UNOP (Cx, pC, Ax, pA, false, i, j, y)

GB_JIT_GLOBAL GB_JIT_KERNEL_SPLIT_FULL_PROTO (GB_jit_kernel) ;
GB_JIT_GLOBAL GB_JIT_KERNEL_SPLIT_FULL_PROTO (GB_jit_kernel)
{
    #include "template/GB_split_full_template.c"
    return (GrB_SUCCESS) ;
}

