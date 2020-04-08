//------------------------------------------------------------------------------
// GB_mex_reduce_to_scalar: c = accum(c,reduce_to_scalar(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Reduce a matrix or vector to a scalar

#include "GB_mex.h"

#define USAGE "c = GB_mex_reduce_to_scalar (c, accum, reduce, A)"

#define FREE_ALL                        \
{                                       \
    GB_MATRIX_FREE (&A) ;               \
    if (!reduce_is_complex)             \
    {                                   \
        GrB_Monoid_free (&reduce) ;     \
    }                                   \
    if (ctype == Complex)               \
    {                                   \
        GB_FREE_MEMORY (c, 1, 2 * sizeof (double)) ; \
    }                                   \
    GB_mx_put_global (true, 0) ;        \
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Monoid reduce = NULL ;
    bool reduce_is_complex = false ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 1 || nargin != 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get the scalar c
    GB_void *c ;
    int64_t cnrows, cncols ;
    mxClassID cclass ;
    GrB_Type ctype ;

    GB_mx_mxArray_to_array (pargin [0], &c, &cnrows, &cncols, &cclass,
        &ctype) ;
    if (cnrows != 1 || cncols != 1)
    {
        mexErrMsgTxt ("c must be a scalar") ;
    }
    if (ctype == NULL)
    {
        mexErrMsgTxt ("c must be numeric") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce; default: NOP, default class is class(C)
    GrB_BinaryOp reduceop ;
    if (!GB_mx_mxArray_to_BinaryOp (&reduceop, pargin [2], "reduceop",
        GB_NOP_opcode, cclass, ctype == Complex, ctype == Complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("reduceop failed") ;
    }

    // get the reduce monoid
    if (reduceop == Complex_plus)
    {
        reduce_is_complex = true ;
        reduce = Complex_plus_monoid ;
    }
    else if (reduceop == Complex_times)
    {
        reduce_is_complex = true ;
        reduce = Complex_times_monoid ;
    }
    else
    {
        // create the reduce monoid
        if (!GB_mx_Monoid (&reduce, reduceop, malloc_debug))
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
    }

    // get accum; default: NOP, default class is class(C)
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [1], "accum",
        GB_NOP_opcode, cclass, ctype == Complex, reduce_is_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    GrB_Descriptor d = NULL ;

    // c = accum(C,A*B)

    // test both Vector and Matrix methods.  The typecast is not necessary,
    // just to test.

    if (A->type == Complex)
    {
        if (A->vdim == 1)
        {
            GrB_Vector V ;
            V = (GrB_Vector) A ;
            METHOD (GrB_Vector_reduce_UDT (c, accum, reduce, V, d)) ;
        }
        else
        {
            METHOD (GrB_Matrix_reduce_UDT (c, accum, reduce, A, d)) ;
        }
    }
    else
    {
        if (A->vdim == 1)
        {
            GrB_Vector V ;
            V = (GrB_Vector) A ;

            #define REDUCE(suffix,type,X) \
                METHOD (GrB_Vector_reduce ## suffix \
                    ((type *) c, accum, reduce, X, d)) ;

            switch (cclass)
            {
                // all GraphBLAS built-in types are supported

                case mxLOGICAL_CLASS : REDUCE (_BOOL,   bool     , V) ; break ;
                case mxINT8_CLASS    : REDUCE (_INT8,   int8_t   , V) ; break ;
                case mxUINT8_CLASS   : REDUCE (_UINT8,  uint8_t  , V) ; break ;
                case mxINT16_CLASS   : REDUCE (_INT16,  int16_t  , V) ; break ;
                case mxUINT16_CLASS  : REDUCE (_UINT16, uint16_t , V) ; break ;
                case mxINT32_CLASS   : REDUCE (_INT32,  int32_t  , V) ; break ;
                case mxUINT32_CLASS  : REDUCE (_UINT32, uint32_t , V) ; break ;
                case mxINT64_CLASS   : REDUCE (_INT64,  int64_t  , V) ; break ;
                case mxUINT64_CLASS  : REDUCE (_UINT64, uint64_t , V) ; break ;
                case mxSINGLE_CLASS  : REDUCE (_FP32,   float    , V) ; break ;
                case mxDOUBLE_CLASS  : REDUCE (_FP64,   double   , V) ; break ;

                case mxCELL_CLASS    :
                case mxCHAR_CLASS    :
                case mxUNKNOWN_CLASS :
                case mxFUNCTION_CLASS:
                case mxSTRUCT_CLASS  :
                default              :
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported class") ;
            }
        }
        else
        {

            #undef  REDUCE
            #define REDUCE(suffix,type,X) \
                METHOD (GrB_Matrix_reduce ## suffix \
                    ((type *) c, accum, reduce, X, d)) ;

            switch (cclass)
            {
                // all GraphBLAS built-in types are supported

                case mxLOGICAL_CLASS : REDUCE (_BOOL,   bool     , A) ; break ;
                case mxINT8_CLASS    : REDUCE (_INT8,   int8_t   , A) ; break ;
                case mxUINT8_CLASS   : REDUCE (_UINT8,  uint8_t  , A) ; break ;
                case mxINT16_CLASS   : REDUCE (_INT16,  int16_t  , A) ; break ;
                case mxUINT16_CLASS  : REDUCE (_UINT16, uint16_t , A) ; break ;
                case mxINT32_CLASS   : REDUCE (_INT32,  int32_t  , A) ; break ;
                case mxUINT32_CLASS  : REDUCE (_UINT32, uint32_t , A) ; break ;
                case mxINT64_CLASS   : REDUCE (_INT64,  int64_t  , A) ; break ;
                case mxUINT64_CLASS  : REDUCE (_UINT64, uint64_t , A) ; break ;
                case mxSINGLE_CLASS  : REDUCE (_FP32,   float    , A) ; break ;
                case mxDOUBLE_CLASS  : REDUCE (_FP64,   double   , A) ; break ;

                case mxCELL_CLASS    :
                case mxCHAR_CLASS    :
                case mxUNKNOWN_CLASS :
                case mxFUNCTION_CLASS:
                case mxSTRUCT_CLASS  :
                default              :
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported class") ;
            }
        }
    }

    // return C to MATLAB as a scalar
    if (ctype == Complex)
    {
        pargout [0] = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxCOMPLEX) ;
        GB_mx_complex_split (1, c, pargout [0]) ;
    }
    else
    {
        pargout [0] = mxCreateNumericMatrix (1, 1, cclass, mxREAL) ;
        GB_void *p = mxGetData (pargout [0]) ;
        memcpy (p, c, ctype->size) ;
    }

    FREE_ALL ;
}

