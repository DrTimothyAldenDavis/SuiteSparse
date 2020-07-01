//------------------------------------------------------------------------------
// GB_mex_Matrix_eWiseAdd: C<M> = accum(C,A+B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE \
    "C = GB_mex_Matrix_eWiseAdd (C, M, accum, add, A, B, desc, test)"

#define FREE_ALL                        \
{                                       \
    GB_MATRIX_FREE (&A) ;               \
    GB_MATRIX_FREE (&B) ;               \
    GB_MATRIX_FREE (&C) ;               \
    GrB_Descriptor_free_(&desc) ;       \
    GB_MATRIX_FREE (&M) ;               \
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
    GrB_Matrix B = NULL ;
    GrB_Matrix C = NULL ;
    GrB_Matrix M = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 1 || nargin < 6 || nargin > 8)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    #define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
    if (nargin > 7 && C != NULL) C->nvec_nonempty = -1 ;
    #define FREE_DEEP_COPY GB_MATRIX_FREE (&C) ;
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get M (shallow copy)
    M = GB_mx_mxArray_to_Matrix (pargin [1], "M", false, false) ;
    if (M == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("M failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [5], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // get add operator
    bool user_complex = (Complex != GxB_FC64)
        && (A->type == Complex || B->type == Complex) ;
    GrB_BinaryOp add ;
    if (!GB_mx_mxArray_to_BinaryOp (&add, pargin [3], "add",
        C->type, user_complex) || add == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("add failed") ;
    }

    // get accum, if present
    user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || add->ztype == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // just for testing
    if (nargin > 7)
    {
        if (M != NULL) M->nvec_nonempty = -1 ;
        A->nvec_nonempty = -1 ;
        B->nvec_nonempty = -1 ;
        C->nvec_nonempty = -1 ;
    }

    // GxB_print (A, 3) ;
    // GxB_print (B, 3) ;
    // if (M != NULL) GxB_print (M, 3) ;
    // GxB_print (C, 3) ;
    // GxB_print (accum, 3) ;
    // GxB_print (add, 3) ;

    // C<M> = accum(C,A+B)
    METHOD (GrB_Matrix_eWiseAdd_BinaryOp_(C, M, accum, add, A, B, desc)) ;

    // printf ("result: \n") ; GxB_print (C, 3) ;

    // return C to MATLAB as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

