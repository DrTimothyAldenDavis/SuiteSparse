//------------------------------------------------------------------------------
// GB_opaque.h: definitions of opaque objects
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_OPAQUE_H
#define GB_OPAQUE_H

//------------------------------------------------------------------------------
// pending tuples
//------------------------------------------------------------------------------

// Pending tuples are a list of unsorted (i,j,x) tuples that have not yet been
// added to a matrix.  The data structure is defined in GB_Pending.h.

typedef struct GB_Pending_struct *GB_Pending ;

//------------------------------------------------------------------------------
// type codes for GrB_Type
//------------------------------------------------------------------------------

typedef enum
{
    // the 14 scalar types: 13 built-in types, and one user-defined type code
    GB_ignore_code  = 0,
    GB_BOOL_code    = 0,        // 'logical' in MATLAB
    GB_INT8_code    = 1,
    GB_UINT8_code   = 2,
    GB_INT16_code   = 3,
    GB_UINT16_code  = 4,
    GB_INT32_code   = 5,
    GB_UINT32_code  = 6,
    GB_INT64_code   = 7,
    GB_UINT64_code  = 8,
    GB_FP32_code    = 9,        // float ('single' in MATLAB)
    GB_FP64_code    = 10,       // double
    GB_FC32_code    = 11,       // float complex ('single complex' in MATLAB)
    GB_FC64_code    = 12,       // double complex
    GB_UDT_code     = 13        // void *, user-defined type
}
GB_Type_code ;                  // enumerated type code

//------------------------------------------------------------------------------
// operator codes used in GrB_BinaryOp and GrB_UnaryOp
//------------------------------------------------------------------------------

typedef enum
{
    //--------------------------------------------------------------------------
    // NOP
    //--------------------------------------------------------------------------

    GB_NOP_opcode = 0,  // no operation

    //--------------------------------------------------------------------------
    // primary unary operators x=f(x)
    //--------------------------------------------------------------------------

    GB_ONE_opcode,      // z = 1
    GB_IDENTITY_opcode, // z = x
    GB_AINV_opcode,     // z = -x
    GB_ABS_opcode,      // z = abs(x) ; except z is real if x is complex
    GB_MINV_opcode,     // z = 1/x ; special cases for bool and integers
    GB_LNOT_opcode,     // z = !x
    GB_BNOT_opcode,     // z = ~x (bitwise complement)

    //--------------------------------------------------------------------------
    // unary operators for floating-point types (real and complex)
    //--------------------------------------------------------------------------

    GB_SQRT_opcode,     // z = sqrt (x)
    GB_LOG_opcode,      // z = log (x)
    GB_EXP_opcode,      // z = exp (x)

    GB_SIN_opcode,      // z = sin (x)
    GB_COS_opcode,      // z = cos (x)
    GB_TAN_opcode,      // z = tan (x)

    GB_ASIN_opcode,     // z = asin (x)
    GB_ACOS_opcode,     // z = acos (x)
    GB_ATAN_opcode,     // z = atan (x)

    GB_SINH_opcode,     // z = sinh (x)
    GB_COSH_opcode,     // z = cosh (x)
    GB_TANH_opcode,     // z = tanh (x)

    GB_ASINH_opcode,    // z = asinh (x)
    GB_ACOSH_opcode,    // z = acosh (x)
    GB_ATANH_opcode,    // z = atanh (x)

    GB_SIGNUM_opcode,   // z = signum (x)
    GB_CEIL_opcode,     // z = ceil (x)
    GB_FLOOR_opcode,    // z = floor (x)
    GB_ROUND_opcode,    // z = round (x)
    GB_TRUNC_opcode,    // z = trunc (x)

    GB_EXP2_opcode,     // z = exp2 (x)
    GB_EXPM1_opcode,    // z = expm1 (x)
    GB_LOG10_opcode,    // z = log10 (x)
    GB_LOG1P_opcode,    // z = log1P (x)
    GB_LOG2_opcode,     // z = log2 (x)

    //--------------------------------------------------------------------------
    // unary operators for real floating-point types
    //--------------------------------------------------------------------------

    GB_LGAMMA_opcode,   // z = lgamma (x)
    GB_TGAMMA_opcode,   // z = tgamma (x)
    GB_ERF_opcode,      // z = erf (x)
    GB_ERFC_opcode,     // z = erfc (x)
    GB_FREXPX_opcode,   // z = frexpx (x), mantissa from ANSI C11 frexp
    GB_FREXPE_opcode,   // z = frexpe (x), exponent from ANSI C11 frexp

    //--------------------------------------------------------------------------
    // unary operators for complex types only
    //--------------------------------------------------------------------------

    GB_CONJ_opcode,     // z = conj (x)

    //--------------------------------------------------------------------------
    // unary operators where z is real and x is complex
    //--------------------------------------------------------------------------

    GB_CREAL_opcode,    // z = creal (x)
    GB_CIMAG_opcode,    // z = cimag (x)
    GB_CARG_opcode,     // z = carg (x)

    //--------------------------------------------------------------------------
    // unary operators where z is bool and x is any floating-point type
    //--------------------------------------------------------------------------

    GB_ISINF_opcode,    // z = isinf (x)
    GB_ISNAN_opcode,    // z = isnan (x)
    GB_ISFINITE_opcode, // z = isfinite (x)

    //--------------------------------------------------------------------------
    // binary operators z=f(x,y) that return the same type as their inputs
    //--------------------------------------------------------------------------

    GB_FIRST_opcode,    // z = x
    GB_SECOND_opcode,   // z = y
    GB_ANY_opcode,      // z = x or y, selected arbitrarily
    GB_PAIR_opcode,     // z = 1
    GB_MIN_opcode,      // z = min(x,y)
    GB_MAX_opcode,      // z = max(x,y)
    GB_PLUS_opcode,     // z = x + y
    GB_MINUS_opcode,    // z = x - y
    GB_RMINUS_opcode,   // z = y - x
    GB_TIMES_opcode,    // z = x * y
    GB_DIV_opcode,      // z = x / y ; special cases for bool and ints
    GB_RDIV_opcode,     // z = y / x ; special cases for bool and ints
    GB_POW_opcode,      // z = pow (x,y)

    GB_ISEQ_opcode,     // z = (x == y)
    GB_ISNE_opcode,     // z = (x != y)
    GB_ISGT_opcode,     // z = (x >  y)
    GB_ISLT_opcode,     // z = (x <  y)
    GB_ISGE_opcode,     // z = (x >= y)
    GB_ISLE_opcode,     // z = (x <= y)

    GB_LOR_opcode,      // z = (x != 0) || (y != 0)
    GB_LAND_opcode,     // z = (x != 0) && (y != 0)
    GB_LXOR_opcode,     // z = (x != 0) != (y != 0)

    GB_BOR_opcode,      // z = (x | y), bitwise or
    GB_BAND_opcode,     // z = (x & y), bitwise and
    GB_BXOR_opcode,     // z = (x ^ y), bitwise xor
    GB_BXNOR_opcode,    // z = ~(x ^ y), bitwise xnor
    GB_BGET_opcode,     // z = bitget (x,y)
    GB_BSET_opcode,     // z = bitset (x,y)
    GB_BCLR_opcode,     // z = bitclr (x,y)
    GB_BSHIFT_opcode,   // z = bitshift (x,y)

    //--------------------------------------------------------------------------
    // binary operators z=f(x,y) that return bool (TxT -> bool)
    //--------------------------------------------------------------------------

    GB_EQ_opcode,       // z = (x == y)
    GB_NE_opcode,       // z = (x != y)
    GB_GT_opcode,       // z = (x >  y)
    GB_LT_opcode,       // z = (x <  y)
    GB_GE_opcode,       // z = (x >= y)
    GB_LE_opcode,       // z = (x <= y)

    //--------------------------------------------------------------------------
    // binary operators for real floating-point types (TxT -> T)
    //--------------------------------------------------------------------------

    GB_ATAN2_opcode,        // z = atan2 (x,y)
    GB_HYPOT_opcode,        // z = hypot (x,y)
    GB_FMOD_opcode,         // z = fmod (x,y)
    GB_REMAINDER_opcode,    // z = remainder (x,y)
    GB_COPYSIGN_opcode,     // z = copysign (x,y)
    GB_LDEXP_opcode,        // z = ldexp (x,y)

    //--------------------------------------------------------------------------
    // binary operator z=f(x,y) where z is complex, x,y real:
    //--------------------------------------------------------------------------

    GB_CMPLX_opcode,        // z = cmplx (x,y)

    //--------------------------------------------------------------------------
    // user-defined: unary and binary operators
    //--------------------------------------------------------------------------

    GB_USER_opcode          // user-defined operator
}
GB_Opcode ;

//------------------------------------------------------------------------------
// select opcodes
//------------------------------------------------------------------------------

// operator codes used in GrB_SelectOp structures
typedef enum
{
    // built-in select operators: thunk optional; defaults to zero
    GB_TRIL_opcode      = 0,
    GB_TRIU_opcode      = 1,
    GB_DIAG_opcode      = 2,
    GB_OFFDIAG_opcode   = 3,
    GB_RESIZE_opcode    = 4,

    // built-in select operators, no thunk used
    GB_NONZOMBIE_opcode = 5,
    GB_NONZERO_opcode   = 6,
    GB_EQ_ZERO_opcode   = 7,
    GB_GT_ZERO_opcode   = 8,
    GB_GE_ZERO_opcode   = 9,
    GB_LT_ZERO_opcode   = 10,
    GB_LE_ZERO_opcode   = 11,

    // built-in select operators, thunk optional; defaults to zero
    GB_NE_THUNK_opcode  = 12,
    GB_EQ_THUNK_opcode  = 13,
    GB_GT_THUNK_opcode  = 14,
    GB_GE_THUNK_opcode  = 15,
    GB_LT_THUNK_opcode  = 16,
    GB_LE_THUNK_opcode  = 17,

    // for all user-defined select operators:  thunk is optional
    GB_USER_SELECT_opcode = 18
}
GB_Select_Opcode ;


//------------------------------------------------------------------------------
// opaque content of GraphBLAS objects
//------------------------------------------------------------------------------

#define GB_LEN 128

struct GB_Type_opaque       // content of GrB_Type
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t size ;           // size of the type
    GB_Type_code code ;     // the type code
    char name [GB_LEN] ;    // name of the type
} ;

struct GB_UnaryOp_opaque    // content of GrB_UnaryOp
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Type xtype ;        // type of x
    GrB_Type ztype ;        // type of z
    GxB_unary_function function ;        // a pointer to the unary function
    char name [GB_LEN] ;    // name of the unary operator
    GB_Opcode opcode ;      // operator opcode
} ;

struct GB_BinaryOp_opaque   // content of GrB_BinaryOp
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Type xtype ;        // type of x
    GrB_Type ytype ;        // type of y
    GrB_Type ztype ;        // type of z
    GxB_binary_function function ;        // a pointer to the binary function
    char name [GB_LEN] ;    // name of the binary operator
    GB_Opcode opcode ;      // operator opcode
} ;

struct GB_SelectOp_opaque   // content of GxB_SelectOp
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Type xtype ;        // type of x, or NULL if generic
    GrB_Type ttype ;        // type of thunk, or NULL if not used or generic
    GxB_select_function function ;        // a pointer to the select function
    char name [GB_LEN] ;    // name of the select operator
    GB_Select_Opcode opcode ;   // operator opcode
} ;

struct GB_Monoid_opaque     // content of GrB_Monoid
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_BinaryOp op ;       // binary operator of the monoid
    void *identity ;        // identity of the monoid
    size_t op_ztype_size ;  // size of the type (also is op->ztype->size)
    void *terminal ;        // value that triggers early-exit (NULL if no value)
    bool builtin ;          // built-in or user defined
} ;

struct GB_Semiring_opaque   // content of GrB_Semiring
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Monoid add ;        // add operator of the semiring
    GrB_BinaryOp multiply ; // multiply operator of the semiring
    bool builtin ;          // built-in or user defined
} ;

struct GB_Scalar_opaque     // content of GxB_Scalar: 1-by-1 standard CSC matrix
{
    #include "GB_matrix.h"
} ;

struct GB_Vector_opaque     // content of GrB_Vector: m-by-1 standard CSC matrix
{
    #include "GB_matrix.h"
} ;

struct GB_Matrix_opaque     // content of GrB_Matrix
{
    #include "GB_matrix.h"
} ;

struct GB_Descriptor_opaque // content of GrB_Descriptor
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Desc_Value out ;    // output descriptor
    GrB_Desc_Value mask ;   // mask descriptor
    GrB_Desc_Value in0 ;    // first input descriptor (A for C=A*B, for example)
    GrB_Desc_Value in1 ;    // second input descriptor (B for C=A*B)
    GrB_Desc_Value axb ;    // for selecting the method for C=A*B
    int nthreads_max ;      // max # threads to use in this call to GraphBLAS
    double chunk ;          // chunk size for # of threads for small problems
    bool predefined ;       // if true, descriptor is predefined
    bool use_mkl ;          // if true, use the Intel MKL
} ;

#endif

