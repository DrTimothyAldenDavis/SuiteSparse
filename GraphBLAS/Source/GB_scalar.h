//------------------------------------------------------------------------------
// GB_scalar.h: definitions for GxB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_SCALAR_H
#define GB_SCALAR_H

GxB_Scalar GB_Scalar_wrap   // create a new GxB_Scalar with one entry
(
    GxB_Scalar s,           // GxB_Scalar to create
    GrB_Type type,          // type of GxB_Scalar to create
    int64_t *Sp,            // becomes S->p, an array of size 2
    int64_t *Si,            // becomes S->i, an array of size 1
    void *Sx                // becomes S->x, an array of size 1 * type->size
) ;

// wrap a bare scalar inside a statically-allocated GxB_Scalar
#define GB_SCALAR_WRAP(scalar,prefix,T,ampersand,bare,stype)        \
    struct GB_Scalar_opaque scalar ## _struct ;                     \
    int64_t Sp [2], Si [1] ;                                        \
    size_t ssize = stype->size ;                                    \
    GB_void Sx [GB_VLA (ssize)] ;                                   \
    GxB_Scalar scalar = GB_Scalar_wrap (& scalar ## _struct,        \
        stype, Sp, Si, Sx) ;                                        \
    memcpy (Sx, ampersand bare, ssize) ;

#endif

