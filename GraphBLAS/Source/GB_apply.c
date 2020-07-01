//------------------------------------------------------------------------------
// GB_apply: apply a unary operator; optionally transpose a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C<M> = accum (C, op(A)) or accum (C, op(A)')

// GB_apply does the work for GrB_*_apply, including the binary op variants.

#include "GB_apply.h"
#include "GB_transpose.h"
#include "GB_accum_mask.h"

#define GB_FREE_ALL ;

GrB_Info GB_apply                   // C<M> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // M descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
        const GrB_UnaryOp op1_in,       // unary operator to apply
        const GrB_BinaryOp op2_in,      // binary operator to apply
        const GxB_Scalar scalar,        // scalar to bind to binary operator
        bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    const GrB_Matrix A,             // first input:  matrix A
    bool A_transpose,               // A matrix descriptor
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M and/or A

    GB_RETURN_IF_FAULTY (accum) ;
    ASSERT_MATRIX_OK (C, "C input for GB_apply", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_apply", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_apply", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_apply", GB0) ;

    GrB_UnaryOp  op1 = op1_in ;
    GrB_BinaryOp op2 = op2_in ;
    GB_Opcode opcode ;
    GrB_Type T_type ;
    if (op1 != NULL)
    { 
        // apply a unary operator
        GB_RETURN_IF_FAULTY (op1) ;
        ASSERT_UNARYOP_OK (op1, "op1 for GB_apply", GB0) ;
        T_type = op1->ztype ;
        opcode = op1->opcode ;
        // A must also be compatible with op1->xtype
        if (!GB_Type_compatible (A->type, op1->xtype))
        { 
            return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
                "Incompatible type for z=%s(x):\n"
                "input A of type [%s]\n"
                "cannot be typecast to x input of type [%s]",
                op1->name, A->type->name, op1->xtype->name))) ;
        }
    }
    else if (op2 != NULL)
    {
        // apply a binary operator, with one input bound to a scalar
        GB_RETURN_IF_FAULTY (op2) ;
        ASSERT_BINARYOP_OK (op2, "op2 for GB_apply", GB0) ;
        ASSERT_SCALAR_OK (scalar, "scalar for GB_apply", GB0) ;
        T_type = op2->ztype ;
        opcode = op2->opcode ;
        bool op_is_first  = opcode == GB_FIRST_opcode ;
        bool op_is_second = opcode == GB_SECOND_opcode ;
        bool op_is_pair   = opcode == GB_PAIR_opcode ;
        if (binop_bind1st)
        {
            // C = op (scalar,A)
            // A must be compatible with op2->ytype
            if (!(op_is_first || op_is_pair ||
                  GB_Type_compatible (A->type, op2->ytype)))
            { 
                return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
                    "Incompatible type for z=%s(x,y):\n"
                    "input A of type [%s]\n"
                    "cannot be typecast to y input of type [%s]",
                    op2->name, A->type->name, op2->ytype->name))) ;
            }
            // scalar must be compatible with op2->xtype
            if (!(op_is_second || op_is_pair ||
                  GB_Type_compatible (scalar->type, op2->xtype)))
            { 
                return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
                    "Incompatible type for z=%s(x,y):\n"
                    "input scalar of type [%s]\n"
                    "cannot be typecast to x input of type [%s]",
                    op2->name, scalar->type->name, op2->xtype->name))) ;
            }
        }
        else
        {
            // C = op (A,scalar)
            // A must be compatible with op2->xtype
            if (!(op_is_first || op_is_pair ||
                  GB_Type_compatible (A->type, op2->xtype)))
            { 
                return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
                    "Incompatible type for z=%s(x,y):\n"
                    "input scalar of type [%s]\n"
                    "cannot be typecast to x input of type [%s]",
                    op2->name, A->type->name, op2->xtype->name))) ;
            }
            // scalar must be compatible with op2->ytype
            if (!(op_is_second || op_is_pair
                  || GB_Type_compatible (scalar->type, op2->ytype)))
            { 
                return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
                    "Incompatible type for z=%s(x,y):\n"
                    "input A of type [%s]\n"
                    "cannot be typecast to y input of type [%s]",
                    op2->name, scalar->type->name, op2->ytype->name))) ;
            }
        }
    }
    else
    {
        return (GB_ERROR (GrB_NULL_POINTER, (GB_LOG,
            "Required argument is null: [op]"))) ;
    }

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Info info = GB_compatible (C->type, C, M, accum, T_type, Context) ;
    if (info != GrB_SUCCESS)
    { 
        return (info) ;
    }

    // check the dimensions
    int64_t tnrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t tncols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    if (GB_NROWS (C) != tnrows || GB_NCOLS (C) != tncols)
    { 
        return (GB_ERROR (GrB_DIMENSION_MISMATCH, (GB_LOG,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "input is " GBd "-by-" GBd "%s",
            GB_NROWS (C), GB_NCOLS (C),
            tnrows, tncols, A_transpose ? " (transposed)" : ""))) ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_MATRIX_WAIT (M) ;
    GB_MATRIX_WAIT (A) ;
    GB_SCALAR_WAIT (scalar) ;

    if (op2 != NULL && GB_NNZ (scalar) != 1)
    {
        // the scalar entry must be present
        return (GB_ERROR (GrB_INVALID_VALUE, (GB_LOG,
            "Scalar is missing; it must contain a single entry"))) ;
    }

    //--------------------------------------------------------------------------
    // rename first, second, any, and pair operators
    //--------------------------------------------------------------------------

    if (op2 != NULL)
    { 
        // GxB_print (op2, 3) ;
        // first(A,x), second(y,A), and any(...) become identity(A)
        if ((opcode == GB_ANY_opcode) ||
            (opcode == GB_FIRST_opcode  && !binop_bind1st) ||
            (opcode == GB_SECOND_opcode &&  binop_bind1st))
        { 
            switch (op2->xtype->code)
            {
                default              :
                case GB_BOOL_code    : op1 = GrB_IDENTITY_BOOL   ; break ;
                case GB_INT8_code    : op1 = GrB_IDENTITY_INT8   ; break ;
                case GB_INT16_code   : op1 = GrB_IDENTITY_INT16  ; break ;
                case GB_INT32_code   : op1 = GrB_IDENTITY_INT32  ; break ;
                case GB_INT64_code   : op1 = GrB_IDENTITY_INT64  ; break ;
                case GB_UINT8_code   : op1 = GrB_IDENTITY_UINT8  ; break ;
                case GB_UINT16_code  : op1 = GrB_IDENTITY_UINT16 ; break ;
                case GB_UINT32_code  : op1 = GrB_IDENTITY_UINT32 ; break ;
                case GB_UINT64_code  : op1 = GrB_IDENTITY_UINT64 ; break ;
                case GB_FP32_code    : op1 = GrB_IDENTITY_FP32   ; break ;
                case GB_FP64_code    : op1 = GrB_IDENTITY_FP64   ; break ;
                case GB_FC32_code    : op1 = GxB_IDENTITY_FC32   ; break ;
                case GB_FC64_code    : op1 = GxB_IDENTITY_FC64   ; break ;
            }
            // printf ("renamed: ") ; GxB_print (op1, 3) ;
            op2 = NULL ;
        }
        else if (opcode == GB_PAIR_opcode)
        { 
            // pair (...) becomes one(A)
            switch (op2->xtype->code)
            {
                default              :
                case GB_BOOL_code    : op1 = GxB_ONE_BOOL   ; break ;
                case GB_INT8_code    : op1 = GxB_ONE_INT8   ; break ;
                case GB_INT16_code   : op1 = GxB_ONE_INT16  ; break ;
                case GB_INT32_code   : op1 = GxB_ONE_INT32  ; break ;
                case GB_INT64_code   : op1 = GxB_ONE_INT64  ; break ;
                case GB_UINT8_code   : op1 = GxB_ONE_UINT8  ; break ;
                case GB_UINT16_code  : op1 = GxB_ONE_UINT16 ; break ;
                case GB_UINT32_code  : op1 = GxB_ONE_UINT32 ; break ;
                case GB_UINT64_code  : op1 = GxB_ONE_UINT64 ; break ;
                case GB_FP32_code    : op1 = GxB_ONE_FP32   ; break ;
                case GB_FP64_code    : op1 = GxB_ONE_FP64   ; break ;
                case GB_FC32_code    : op1 = GxB_ONE_FC32   ; break ;
                case GB_FC64_code    : op1 = GxB_ONE_FC64   ; break ;
            }
            // printf ("renamed: ") ; GxB_print (op1, 3) ;
            op2 = NULL ;
        }

#if 0
        else
        {
            switch (opcode)
            {
                // commutative operators, no need for bind1st workers:
                case PLUS_opcode      :
                case TIMES_opcode     :
                case PAIR_opcode      :
                case ANY_opcode       :
                case ISEQ_opcode      :
                case ISNE_opcode      :
                case EQ_opcode        :
                case NE_opcode        :
                case MIN_opcode       :
                case MAX_opcode       :
                case LOR_opcode       :
                case LAND_opcode      :
                case LXOR_opcode      :
                case LXNOR_opcode     :
                case HYPOT_opcode     :
                case BOR_opcode       :
                case BAND_opcode      :
                case BXOR_opcode      :
                case BXNOR_opcode     : binop_bind1st = false ;
                default : ;
            }
        }
#endif

    }

    //--------------------------------------------------------------------------
    // T = op(A) or op(A')
    //--------------------------------------------------------------------------

    bool C_is_csc = C->is_csc ;
    if (C_is_csc != A->is_csc)
    { 
        // Flip the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    GrB_Matrix T = NULL ;

    if (A_transpose)
    { 
        // T = op (A'), typecasting to op*->ztype
        // transpose: typecast, apply an op, not in place
        GBBURBLE ("(transpose-op) ") ;
        info = GB_transpose (&T, T_type, C_is_csc, A,
            op1, op2, scalar, binop_bind1st,
            Context) ;
    }
    else if (M == NULL && accum == NULL && (C == A) && C->type == T_type)
    { 
        GBBURBLE ("(inplace-op) ") ;
        // C = op (C), operating on the values in place, with no typecasting
        // of the output of the operator with the matrix C.  Always succeeds.
        // No work to do if the op is identity.
        // FUTURE::: also handle C += op(C), with accum.
        if (opcode != GB_IDENTITY_opcode)
        { 
            GB_void *Cx = (GB_void *) C->x ;
            GB_apply_op (Cx,
                op1, op2, scalar, binop_bind1st,
                Cx, C->type, GB_NNZ (C), Context) ;
        }
        return (GrB_SUCCESS) ;
    }
    else
    { 
        // T = op (A), pattern is a shallow copy of A, type is op*->ztype.
        GBBURBLE ("(shallow-op) ") ;
        info = GB_shallow_op (&T, C_is_csc,
            op1, op2, scalar, binop_bind1st,
            A, Context) ;
    }

    if (info != GrB_SUCCESS)
    { 
        GB_MATRIX_FREE (&T) ;
        return (info) ;
    }

    ASSERT (T->is_csc == C->is_csc) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the M
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Context)) ;
}

