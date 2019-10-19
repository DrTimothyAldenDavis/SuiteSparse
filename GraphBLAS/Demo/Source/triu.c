//------------------------------------------------------------------------------
// GraphBLAS/Demo/triu.c: extract the lower triangular part of a matrix
//------------------------------------------------------------------------------

// C = triu (A,1).  The matrix A can be of any built-in type.

// This method makes two copies of the matrix: (1) the set of triplets, and (2)
// the final output via GrB_Matrix_build.  As a result, it is about 4 times
// slower than triu(A,1) in MATLAB, and it is a little cumbersome for the user
// to write a method that works on any type.  A better method would be to
// create the output matrix directly, but this would require access to the
// internal data structure of GraphBLAS.

// FUTURE:  GraphBLAS could use a function, say 'GrB_select'.  It would be like
// GrB_apply, but rather than applying a unary operator, it would take in a
// function pointer than defines what entries should appear in the output.  A
// possible prototype:

// GrB_Info info = GrB_select (C, Mask, accum, A, f, desc) ;

// where f is a function pointer with the following prototype:

//  bool f                      // return true if A(i,j) is to be kept in C
//  (
//      const GrB_Index i,      // row index of entry A(i,j)
//      const GrB_Index j,      // column index of entry A(i,j)
//      const GrB_Index m,      // # of rows of A
//      const GrB_Index n,      // # of cols of A
//      const void *x           // value of A, must match type of A
//  ) ;

// If this function were available, then triu.c would be very simple:

//      bool f ( ... ) { return (i > j) ; }
//      ...
//      GrB_Matrix_new (&C, atype, m, n) ;
//      GrB_select (C, NULL, NULL, A, f, NULL) ;

// The method would be much faster since GrB_select would have access to the
// internal data structure of A.  It would also be very flexible since f can
// be any boolean test on the entry A(i,j).  It could be used to drop specific
// numeric values, for example, like explicit zeros.

// There would be two versions of this operation, GrB_Matrix_select and
// GrB_Vector_select.  The output C would have the same size as A (or A' if
// GrB_TRAN is specified), and its entries would be a subset of A.  The values
// of the entries would not be modified (except perhaps by typecasting).  It
// would also be possible to generalize the operation and pass in a unary
// operator, like GrB_apply.

// Let f(A) denote the pruned copy of A, after deleting entries via the
// function f.  The entire descriptor and accum/mask phase would be used.  The
// operation would be C<Mask>=f(A) or C<Mask>=f(A') if the input descriptor
// inp0 is GrB_TRAN.  If accum is not NULL, it would be C<Mask>=accum(C,f(A)).
// The GrB_SCMP and GrB_REPLACE descriptors would also modify the operation.

// This suggestion for GrB_select would be like adding a Mask that is defined
// not by a matrix, but by a function f(i,j,m,n,x).  So a more general
// suggestion would be to define a Mask function that could take the place of
// any Mask matrix.  This would be a significant extension of GraphBLAS.

#include "demos.h"

#define FREE_ALL                \
    GrB_free (&C) ;             \
    if (I != NULL) free (I) ;   \
    if (J != NULL) free (J) ;   \
    if (X != NULL) free (X) ;

//------------------------------------------------------------------------------
// triu: extract the upper triangular part of a matrix, excl. the diagonal
//------------------------------------------------------------------------------

GrB_Info triu               // C = triu (A,1)
(
    GrB_Matrix *C_output,   // output matrix
    const GrB_Matrix A      // input matrix, boolean or double
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Index m, n, nnz ;
    GrB_Matrix C = NULL ;
    GrB_Type atype ;
    GrB_Index *I = NULL, *J = NULL ;
    void *X = NULL ;
    size_t asize ;
    *C_output = NULL ;

    OK (GrB_Matrix_type (&atype, A)) ;
    OK (GrB_Type_size (&asize, atype)) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_nrows (&m, A)) ;
    OK (GrB_Matrix_ncols (&n, A)) ;
    OK (GrB_Matrix_new (&C, atype, m, n)) ;
    OK (GrB_Matrix_nvals (&nnz, A)) ;

    //--------------------------------------------------------------------------
    // allocate space for the tuples
    //--------------------------------------------------------------------------

    I = (GrB_Index *) malloc ((nnz + 1) * sizeof (int64_t)) ;
    J = (GrB_Index *) malloc ((nnz + 1) * sizeof (int64_t)) ;
    X = malloc ((nnz + 1) * asize) ;
    if (I == NULL || J == NULL || X == NULL)
    {
        FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // C = triu (A,1)
    //--------------------------------------------------------------------------

    int64_t lnz = 0 ;

    #define TRIU(xtype,op)                                      \
    {                                                           \
        xtype *Y = (xtype *) X ;                                \
        OK (GrB_Matrix_extractTuples (I, J, Y, &nnz, A)) ;      \
        for (int64_t k = 0 ; k < nnz ; k++)                     \
        {                                                       \
            if (I [k] < J [k])                                  \
            {                                                   \
                I [lnz] = I [k] ;                               \
                J [lnz] = J [k] ;                               \
                Y [lnz] = Y [k] ;                               \
                lnz++ ;                                         \
            }                                                   \
        }                                                       \
        OK (GrB_Matrix_build (C, I, J, Y, lnz, op)) ;           \
    }

    if      (atype == GrB_BOOL  ) { TRIU (bool,     GrB_FIRST_BOOL)   ; }
    else if (atype == GrB_INT8  ) { TRIU (int8_t,   GrB_FIRST_INT8)   ; }
    else if (atype == GrB_UINT8 ) { TRIU (uint8_t,  GrB_FIRST_UINT8)  ; }
    else if (atype == GrB_INT16 ) { TRIU (int16_t,  GrB_FIRST_INT16)  ; }
    else if (atype == GrB_UINT16) { TRIU (uint16_t, GrB_FIRST_UINT16) ; }
    else if (atype == GrB_INT32 ) { TRIU (int32_t,  GrB_FIRST_INT32)  ; }
    else if (atype == GrB_UINT32) { TRIU (uint32_t, GrB_FIRST_UINT32) ; }
    else if (atype == GrB_INT64 ) { TRIU (int64_t,  GrB_FIRST_INT64)  ; }
    else if (atype == GrB_UINT64) { TRIU (uint64_t, GrB_FIRST_UINT64) ; }
    else if (atype == GrB_FP32  ) { TRIU (float,    GrB_FIRST_FP32)   ; }
    else if (atype == GrB_FP64  ) { TRIU (double,   GrB_FIRST_FP64)   ; }
    else
    {
        FREE_ALL ;
        printf ("triu: invalid type, must be built-in type\n") ;
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    free (I) ;
    free (J) ;
    free (X) ;
    *C_output = C ;
    return (GrB_SUCCESS) ;
}

#undef FREE_ALL
#undef TRIU

