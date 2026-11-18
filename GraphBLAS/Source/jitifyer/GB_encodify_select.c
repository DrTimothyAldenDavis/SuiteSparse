//------------------------------------------------------------------------------
// GB_encodify_select: encode a select problem, including types and op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

uint64_t GB_encodify_select     // encode an select problem
(
    // output:
    GB_jit_encoding *encoding,  // unique encoding of the entire problem,
                                // except for the suffix
    char **suffix,              // suffix for user-defined kernel
    // input:
    const GB_jit_kcode kcode,   // kernel to encode
    const GrB_Matrix C,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A
)
{ 

    //--------------------------------------------------------------------------
    // check if the operator and copied matrix values are JIT'able
    //--------------------------------------------------------------------------

    if ((op != NULL && op->hash == UINT64_MAX) ||
        A->type->hash == UINT64_MAX)
    { 
        // A positional operator may ignore x, but phase2 still copies A values.
        memset (encoding, 0, sizeof (GB_jit_encoding)) ;
        (*suffix) = NULL ;
        return (UINT64_MAX) ;
    }

    //--------------------------------------------------------------------------
    // primary encoding of the problem
    //--------------------------------------------------------------------------

    GB_encodify_kcode (encoding, kcode) ;
    GB_enumify_select (&encoding->code, C, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // determine the suffix and its length
    //--------------------------------------------------------------------------

    uint64_t hash = 0 ;
    if (op->hash != 0)
    { 
        // user-defined operator, which must already include the xtype of
        // its input, with no typecasting so its xtype must match the A type.
        // The suffix is the name of the operator, without the need for
        // any type name in the suffix.
        hash = op->hash ;
        encoding->suffix_len = op->name_len ;
        (*suffix) = op->name ;
    }
    else
    { 
        // built-in operator; hash the matrix type.  Positional operators
        // (notably NONZOMBIE) have no xtype. Include A's actual type so
        // kernels copying distinct UDT layouts cannot be reused.  A UDT suffix
        // also enables definition checks when loading cached kernels.
        // If the A->type->hash is zero, it denotes a built-in type,
        // which is not included in the hash or the suffix.
        hash = A->type->hash ;
        encoding->suffix_len = (hash == 0) ? 0 : A->type->name_len ;
        (*suffix) = (hash == 0) ? NULL : A->type->name ;
    }

    //--------------------------------------------------------------------------
    // compute the hash of the entire problem
    //--------------------------------------------------------------------------

    hash = hash ^ GB_jitifyer_hash_encoding (encoding) ;
    return ((hash == 0 || hash == UINT64_MAX) ? GB_MAGIC : hash) ;
}

