//------------------------------------------------------------------------------
// GB_mex_subassign: C(I,J)<Mask> = accum (C (I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

// This function is a wrapper for all GrB_*_subassign functions.
// For these uses, the Mask must always be the same size as C(I,J) and A.

// GrB_Matrix_subassign: Mask has the same size as C(I,J) and A
// GrB_Matrix_subassign_TYPE: Mask has the same size as C(I,J).  A is scalar.

// GrB_Vector_subassign: Mask has the same size as C(I,J) and A
// GrB_Vector_subassign_TYPE: Mask has the same size as C(I,J).  A is scalar.

// GrB_Col_subassign: on input to GB_mex_subassign, Mask and A are a single
// columns, the same size as the single subcolumn C(I,j).  They are not column
// vectors.

// GrB_Row_subassign: on input to GB_mex_subassign, Mask and A are single ROWS,
// the same size as the single subrow C(i,J).  They are not column vectors.
// Before GrB_Row_subassign is called, the A and the Mask (if present) are
// transposed.

// Thus, in all cases, A and the Mask (if present) have the same size as C(I,J),
// except in the case where A is a scalar.  In that case, A is implicitly
// expanded into a matrix the same size as C(I,J), but this occurs inside
// GrB_*subassign, not here.

// This function does the same thing as the MATLAB mimic GB_spec_subassign.m.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define FREE_ALL                        \
{                                       \
    GB_MATRIX_FREE (&A) ;               \
    GB_MATRIX_FREE (&Mask) ;            \
    GB_MATRIX_FREE (&C) ;               \
    GrB_free (&desc) ;                  \
    GB_mx_put_global (malloc_debug) ;   \
}

#define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true) ;

#define FREE_DEEP_COPY GB_MATRIX_FREE (&C) ;

GrB_Matrix C = NULL ;
GrB_Matrix Mask = NULL ;
GrB_Matrix A = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
GrB_Index *I = NULL, ni ;
GrB_Index *J = NULL, nj ;
bool malloc_debug = false ;
GrB_Info info = GrB_SUCCESS ;

//------------------------------------------------------------------------------
// assign: perform a single assignment
//------------------------------------------------------------------------------

#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        GB_MATRIX_FREE (&mask) ;        \
        GB_MATRIX_FREE (&u) ;           \
        return (info) ;                 \
    }                                   \
}

GrB_Info assign ( )
{
    bool at = (desc != NULL && desc->in0 == GrB_TRAN) ;
    GrB_Info info ;
    GrB_Matrix mask = NULL, u = NULL ;

    if (A->nrows == 1 && A->ncols == 1 && NNZ (A) == 1)
    {
        // scalar expansion to matrix or vector
        void *Ax = A->x ;

        if (ni == 1 && nj == 1 && Mask == NULL && I != GrB_ALL && J != GrB_ALL
            && GB_op_is_second (accum, C->type) && A->type->code != GB_UDT_code
            && desc == NULL)
        {
            // printf ("setElement\n") ;
            // test GrB_Matrix_setElement
            #define ASSIGN(type)                                        \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (GrB_Matrix_setElement (C, x, I [0], J [0])) ;       \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (bool) ;
                case GB_INT8_code   : ASSIGN (int8_t) ;
                case GB_UINT8_code  : ASSIGN (uint8_t) ;
                case GB_INT16_code  : ASSIGN (int16_t) ;
                case GB_UINT16_code : ASSIGN (uint16_t) ;
                case GB_INT32_code  : ASSIGN (int32_t) ;
                case GB_UINT32_code : ASSIGN (uint32_t) ;
                case GB_INT64_code  : ASSIGN (int64_t) ;
                case GB_UINT64_code : ASSIGN (uint64_t) ;
                case GB_FP32_code   : ASSIGN (float) ;
                case GB_FP64_code   : ASSIGN (double) ;
                case GB_UDT_code    :
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported class") ;
            }
            #undef ASSIGN

        }
        if (C->ncols == 1)
        {

            // test GrB_Vector_subassign_scalar functions
            // printf ("scalar assign to vector\n") ;
            #define ASSIGN(type)                                        \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (GrB_subassign ((GrB_Vector) C, (GrB_Vector) Mask,   \
                    accum, x, I, ni, desc)) ;      \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (bool) ;
                case GB_INT8_code   : ASSIGN (int8_t) ;
                case GB_UINT8_code  : ASSIGN (uint8_t) ;
                case GB_INT16_code  : ASSIGN (int16_t) ;
                case GB_UINT16_code : ASSIGN (uint16_t) ;
                case GB_INT32_code  : ASSIGN (int32_t) ;
                case GB_UINT32_code : ASSIGN (uint32_t) ;
                case GB_INT64_code  : ASSIGN (int64_t) ;
                case GB_UINT64_code : ASSIGN (uint64_t) ;
                case GB_FP32_code   : ASSIGN (float) ;
                case GB_FP64_code   : ASSIGN (double) ;
                case GB_UDT_code    :
                {
                    OK (GrB_subassign ((GrB_Vector) C, (GrB_Vector) Mask,
                        accum, Ax, I, ni, desc)) ;
                }
                break ;
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported class") ;
            }
            #undef ASSIGN

        }
        else
        {

            // test Matrix_subassign_scalar functions
            // printf ("scalar assign to matrix\n") ;
            #define ASSIGN(type)                                            \
            {                                                               \
                type x = ((type *) Ax) [0] ;                                \
                OK (GrB_subassign (C, Mask, accum, x, I, ni, J, nj,desc)) ; \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (bool) ;
                case GB_INT8_code   : ASSIGN (int8_t) ;
                case GB_UINT8_code  : ASSIGN (uint8_t) ;
                case GB_INT16_code  : ASSIGN (int16_t) ;
                case GB_UINT16_code : ASSIGN (uint16_t) ;
                case GB_INT32_code  : ASSIGN (int32_t) ;
                case GB_UINT32_code : ASSIGN (uint32_t) ;
                case GB_INT64_code  : ASSIGN (int64_t) ;
                case GB_UINT64_code : ASSIGN (uint64_t) ;
                case GB_FP32_code   : ASSIGN (float) ;
                case GB_FP64_code   : ASSIGN (double) ;
                case GB_UDT_code    :
                {
                    OK (GrB_subassign (C, Mask, accum, Ax, I, ni, J, nj, desc));
                }
                break ;

                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported class") ;
            }
            #undef ASSIGN

        }
    }
    else if (C->ncols == 1 && A->ncols == 1 &&
        (Mask == NULL || Mask->ncols == 1) && !at)
    {
        // test GrB_Vector_subassign
        // printf ("vector assign\n") ;
        OK (GrB_subassign ((GrB_Vector) C, (GrB_Vector) Mask, accum,
            (GrB_Vector) A, I, ni, desc)) ;
    }
    else if (A->ncols == 1 && nj == 1 &&
        (Mask == NULL || Mask->ncols == 1) && !at)
    {
        // test GrB_Col_subassign
        // printf ("col assign\n") ;
        OK (GrB_subassign (C, (GrB_Vector) Mask, accum, (GrB_Vector) A,
            I, ni, J [0], desc)) ;
    }
    else if (A->nrows == 1 && ni == 1 &&
        (Mask == NULL || Mask->nrows == 1) && !at)
    {
        // test GrB_Row_subassign; this is not meant to be efficient,
        // just for testing
        // printf ("row assign\n") ;
        if (Mask != NULL)
        {
            GB_NEW (&mask, GrB_BOOL, Mask->ncols, Mask->nrows, false, true) ;
            OK (info) ;
            OK (GB_Matrix_transpose (mask, Mask, NULL, true)) ;
        }
        GB_NEW (&u, A->type, A->ncols, A->nrows, false, true) ;
        OK (info) ;
        OK (GB_Matrix_transpose (u, A, NULL, true)) ;
        OK (GrB_subassign (C, (GrB_Vector) mask, accum, (GrB_Vector) u,
            I [0], J, nj, desc)) ;
        GB_MATRIX_FREE (&mask) ;
        GB_MATRIX_FREE (&u) ;
    }
    else
    {
        // standard submatrix assignment
        // printf ("submatrix assign\n") ;
        OK (GrB_subassign (C, Mask, accum, A, I, ni, J, nj, desc)) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// many_subassign: do a sequence of assignments
//------------------------------------------------------------------------------

// The list of assignments is in a struct array

GrB_Info many_subassign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fMask,
    int fdesc,
    mxClassID cclass,
    const mxArray *pargin [ ]
)
{
    GrB_Info info = GrB_SUCCESS ;

    for (int64_t k = 0 ; k < nwork ; k++)
    {

        //----------------------------------------------------------------------
        // get the kth work to do
        //----------------------------------------------------------------------

        // each struct has fields A, I, J, and optionally Mask, accum, and desc

        mxArray *p ;

        // [ turn off malloc debugging
        bool save = GB_thread_local.malloc_debug ;
        GB_thread_local.malloc_debug = false ;

        // get Mask (shallow copy)
        Mask = NULL ;
        if (fMask >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fMask) ;
            Mask = GB_mx_mxArray_to_Matrix (p, "Mask", false) ;
            if (Mask == NULL && !mxIsEmpty (p))
            {
                FREE_ALL ;
                mexErrMsgTxt ("Mask failed") ;
            }
        }

        // get A (shallow copy)
        p = mxGetFieldByNumber (pargin [1], k, fA) ;
        A = GB_mx_mxArray_to_Matrix (p, "A", false) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }

        // get accum; default: NOP, default class is class(C)
        accum = NULL ;
        if (faccum >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, faccum) ;
            if (!GB_mx_mxArray_to_BinaryOp (&accum, p, "accum",
                GB_NOP_opcode, cclass,
                C->type == Complex, A->type == Complex))
            {
                FREE_ALL ;
                mexErrMsgTxt ("accum failed") ;
            }
        }

        // get I
        p = mxGetFieldByNumber (pargin [1], k, fI) ;
        if (!GB_mx_mxArray_to_indices (&I, p, &ni))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J
        p = mxGetFieldByNumber (pargin [1], k, fJ) ;
        if (!GB_mx_mxArray_to_indices (&J, p, &nj))
        {
            FREE_ALL ;
            mexErrMsgTxt ("J failed") ;
        }

        // get desc
        desc = NULL ;
        if (fdesc > 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fdesc) ;
            if (!GB_mx_mxArray_to_Descriptor (&desc, p, "desc"))
            {
                FREE_ALL ;
                mexErrMsgTxt ("desc failed") ;
            }
        }
        // restore malloc debugging to test the method
        GB_thread_local.malloc_debug = save ;   // ]

        // GB_check (desc, "desc", 3) ;

        //----------------------------------------------------------------------
        // C(I,J)<Mask> = A
        //----------------------------------------------------------------------

        info = assign ( ) ;

        GB_MATRIX_FREE (&A) ;
        GB_MATRIX_FREE (&Mask) ;
        GrB_free (&desc) ;

        if (info != GrB_SUCCESS)
        {
            return (info) ;
        }
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_mex_subassign mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global ( ) ;
    A = NULL ;
    C = NULL ;
    Mask = NULL ;
    desc = NULL ;

    if (nargout > 1 || ! (nargin == 2 || nargin == 6 || nargin == 7))
    {
        mexErrMsgTxt ("Usage: C = GB_mex_subassign "
       "(C, Mask, accum, A, I, J, desc) or (C, Work)");
    }

    //--------------------------------------------------------------------------
    // get C (make a deep copy)
    //--------------------------------------------------------------------------

    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }
    mxClassID cclass = GB_mx_Type_to_classID (C->type) ;

    if (nargin == 2)
    {

        //----------------------------------------------------------------------
        // get a list of work to do: a struct array of length nwork
        //----------------------------------------------------------------------

        // each entry is a struct with fields:
        // Mask, accum, A, I, J, desc

        if (!mxIsStruct (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("2nd argument must be a struct") ;
        }

        int nwork = mxGetNumberOfElements (pargin [1]) ;
        int nf = mxGetNumberOfFields (pargin [1]) ;
        for (int f = 0 ; f < nf ; f++)
        {
            mxArray *p ;
            for (int k = 0 ; k < nwork ; k++)
            {
                p = mxGetFieldByNumber (pargin [1], k, f) ;
            }
        }

        int fA = mxGetFieldNumber (pargin [1], "A") ;
        int fI = mxGetFieldNumber (pargin [1], "I") ;
        int fJ = mxGetFieldNumber (pargin [1], "J") ;
        int faccum = mxGetFieldNumber (pargin [1], "accum") ;
        int fMask = mxGetFieldNumber (pargin [1], "Mask") ;
        int fdesc = mxGetFieldNumber (pargin [1], "desc") ;

        if (fA < 0 || fI < 0 || fJ < 0) mexErrMsgTxt ("A,I,J required") ;

        METHOD (many_subassign (nwork, fA, fI, fJ, faccum, fMask, fdesc, cclass,
            pargin)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C(I,J)<Mask> = A, with a single assignment
        //----------------------------------------------------------------------

        // get Mask (shallow copy)
        Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", false) ;
        if (Mask == NULL && !mxIsEmpty (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("Mask failed") ;
        }

        // get A (shallow copy)
        A = GB_mx_mxArray_to_Matrix (pargin [3], "A", false) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }

        // get accum; default: NOP, default class is class(C)
        accum = NULL ;
        if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
            GB_NOP_opcode, cclass, C->type == Complex, A->type == Complex))
        {
            FREE_ALL ;
            mexErrMsgTxt ("accum failed") ;
        }

        // get I
        if (!GB_mx_mxArray_to_indices (&I, pargin [4], &ni))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J
        if (!GB_mx_mxArray_to_indices (&J, pargin [5], &nj))
        {
            FREE_ALL ;
            mexErrMsgTxt ("J failed") ;
        }

        // get desc
        if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
        {
            FREE_ALL ;
            mexErrMsgTxt ("desc failed") ;
        }

        // C(I,J)<Mask> = A

        METHOD (assign ( )) ;
    }

    //--------------------------------------------------------------------------
    // return C to MATLAB as a struct
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (C, "Final C before wait", 0)) ;
    GrB_wait ( ) ;
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C assign result", true) ;
    FREE_ALL ;
}

