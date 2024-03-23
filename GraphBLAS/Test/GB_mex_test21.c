//------------------------------------------------------------------------------
// GB_mex_test21: test JIT functionality currently not used by GraphBLAS itself
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The enumfy/macrofy methods handle more cases than are currently used by
// the JIT kernels, such as (1) typecasting the output of the accum/monoid/op
// to the type of C, (2) positional select operators, (3) positional index
// unary operators, and (4) the case when the output C matrix is iso-valued.
// These features will be needed when the JIT is extended to allow for
// typecasing of the output.  Positional operators and iso-valued outputs are
// not needed in the JIT on the CPU; these are done with specialized kernels in
// GrB_apply and GrB_select.  These cases will be needed for CUDA however.

// These tests exercise those parts of the JIT that GraphBLAS does not yet use.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_test21"
#define HEADER fprintf (fp, "\n\n================================================================================\n") ;

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void opi32func (GxB_FC32_t *z, const GxB_FC32_t *x, GrB_Index i, GrB_Index j,
    const GxB_FC32_t *y) ;
void opi32func (GxB_FC32_t *z, const GxB_FC32_t *x, GrB_Index i, GrB_Index j,
    const GxB_FC32_t *y)
{
    (*z) = (*x) ;
}
#define OPI32_DEFN \
"void opi32func (GxB_FC32_t *z, const GxB_FC32_t *x, GrB_Index i, GrB_Index j, \n" \
"    const GxB_FC32_t *y)   \n" \
"{                          \n" \
"    (*z) = (*x) ;          \n" \
"}"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GB_WERK ("GB_mex_test21") ;

    //--------------------------------------------------------------------------
    // results are written to a single log file
    //--------------------------------------------------------------------------

    FILE *fp = fopen ("log_GB_mex_test21.txt", "w") ;
    CHECK (fp != NULL) ;

    //--------------------------------------------------------------------------
    // GB_enumify_cuda_atomic
    //--------------------------------------------------------------------------

    const char *a, *cuda_type ;
    bool user_monoid_atomically ;
    bool has_cheeseburger = GB_enumify_cuda_atomic (&a,
        &user_monoid_atomically, &cuda_type, NULL, 0, sizeof (uint32_t), 0) ;
    CHECK (!has_cheeseburger) ;
    CHECK (user_monoid_atomically) ;
    CHECK (cuda_type == NULL) ;
    CHECK (a == NULL) ;

    uint64_t scode ;
    GrB_Matrix A, B, C, C_iso, H ;
    OK (GrB_Matrix_new (&A, GrB_BOOL, 5, 5)) ;
    OK (GrB_Matrix_new (&B, GrB_BOOL, 5, 5)) ;
    OK (GrB_Matrix_new (&C, GrB_BOOL, 5, 5)) ;
    OK (GrB_Matrix_new (&H, GrB_INT32, 5, 5)) ;
    OK (GrB_Matrix_new (&C_iso, GrB_BOOL, 5, 5)) ;
    OK (GrB_assign (C_iso, NULL, NULL, true, GrB_ALL, 5, GrB_ALL, 5, NULL)) ;
    OK (GrB_wait (C, GrB_MATERIALIZE)) ;
    // GxB_print (C_iso, 3) ;

    //--------------------------------------------------------------------------
    // GB_macrofy_cast_output
    //--------------------------------------------------------------------------

    HEADER ;
    fprintf (fp, "GB_macrofy_cast_output, ztype NULL\n") ;
    printf ("GB_macrofy_cast_output, ztype NULL\n") ;
    GB_macrofy_cast_output (fp, "GB_PUTC", "z", "Cx,p", "Cx [p]", NULL, NULL) ;

    HEADER ;
    fprintf (fp, "GB_macrofy_cast_output, cast FC64 to bool\n") ;
    printf ("GB_macrofy_cast_output, cast FC64 to bool\n") ;
    GB_macrofy_cast_output (fp, "GB_PUTC", "z", "Cx,p", "Cx [p]", 
        GxB_FC64, GrB_BOOL) ;

    //--------------------------------------------------------------------------
    // GB_assign_describe
    //--------------------------------------------------------------------------

    HEADER ;
    fprintf (fp, "GB_assign_describe\n") ;
    printf ("GB_assign_describe\n") ;
    char str [2048] ;
    GB_assign_describe (str, 2048, false, GB_ALL, GB_ALL, 
        /* M_is_null: */ true, /* M_sparsity: */ GxB_SPARSE,
        /* Mask_comp: */ true, /* Mask_struct: */ true,
        /* accum: */ NULL, /* A_is_null: */ false, GB_ASSIGN) ;
    fprintf (fp, "%s\n", str) ;
    printf ("%s\n", str) ;

    //--------------------------------------------------------------------------
    // GB_enumify_ewise / GB_macrofy_ewise
    //--------------------------------------------------------------------------

    HEADER ;
    fprintf (fp, "GB_enumify_ewise / GB_macrofy_ewise, C iso\n") ;
    printf ("GB_enumify_ewise / GB_macrofy_ewise, C iso\n") ;
    GB_enumify_ewise (&scode, false, false, true, /* C_iso: */ true,
        /* C_in_iso: */ false, GxB_SPARSE, GrB_BOOL, /* M: */ NULL,
        false, false, GrB_LAND, false, A, B) ;
//  printf ("ewise  scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_ewise (fp, scode, GrB_LAND, GrB_BOOL, GrB_BOOL, GrB_BOOL) ;

    HEADER ;
    fprintf (fp, "GB_enumify_ewise / GB_macrofy_ewise, C non iso\n") ;
    printf ("GB_enumify_ewise / GB_macrofy_ewise, C non iso\n") ;
    GB_enumify_ewise (&scode, false, false, true, /* C_iso: */ false,
        /* C_in_iso: */ false, GxB_SPARSE, GrB_BOOL, /* M: */ NULL,
        false, false, GrB_LAND, false, A, B) ;
//  printf ("ewise  scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_ewise (fp, scode, GrB_LAND, GrB_BOOL, GrB_BOOL, GrB_BOOL) ;

    //--------------------------------------------------------------------------
    // GB_enumify_mxm / GB_macrofy_mxm
    //--------------------------------------------------------------------------

    HEADER ;
    fprintf (fp, "GB_enumify_mxm / GB_macrofy_mxm, C iso\n") ;
    printf ("GB_enumify_mxm / GB_macrofy_mxm, C iso\n") ;
    GB_enumify_mxm (&scode, /* C_iso: */ true, /* C_in_iso: */ true,
        GxB_SPARSE, GrB_BOOL, /* M: */ NULL, false, false,
        GrB_LAND_LOR_SEMIRING_BOOL, /* flipxy: */ true, A, B) ;
//  printf ("mxm    scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_mxm (fp, scode, GrB_LAND_LOR_SEMIRING_BOOL,
        GrB_BOOL, GrB_BOOL, GrB_BOOL) ;

    HEADER ;
    fprintf (fp, "GB_enumify_mxm / GB_macrofy_mxm, any_pair, flipxy\n") ;
    printf ("GB_enumify_mxm / GB_macrofy_mxm, any_pair, flipxy\n") ;
    GB_enumify_mxm (&scode, /* C_iso: */ true, /* C_in_iso: */ false,
        GxB_SPARSE, GrB_BOOL, /* M: */ NULL, false, false,
        GxB_ANY_PAIR_BOOL, /* flipxy: */ true, A, B) ;
//  printf ("mxm    scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_mxm (fp, scode, GxB_ANY_PAIR_BOOL,
        GrB_BOOL, GrB_BOOL, GrB_BOOL) ;

    HEADER ;
    fprintf (fp, "GB_enumify_mxm / GB_macrofy_mxm, any_pair fp32\n") ;
    printf ("GB_enumify_mxm / GB_macrofy_mxm, any_pair fp32\n") ;
    GB_enumify_mxm (&scode, /* C_iso: */ false, /* C_in_iso: */ false,
        GxB_SPARSE, GrB_FP32, /* M: */ NULL, false, false,
        GxB_ANY_PAIR_FP32, /* flipxy: */ true, A, B) ;
//  printf ("mxm    scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_mxm (fp, scode, GxB_ANY_PAIR_FP32,
        GrB_FP32, GrB_FP32, GrB_FP32) ;

    //--------------------------------------------------------------------------
    // GB_enumify_select / GB_macrofy_select
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp idxops [16] = {
        GrB_ROWINDEX_INT32,  GrB_ROWINDEX_INT64,
        GrB_COLINDEX_INT32,  GrB_COLINDEX_INT64,
        GrB_DIAGINDEX_INT32, GrB_DIAGINDEX_INT64,
        GrB_TRIL, GrB_TRIU, GrB_DIAG, GrB_OFFDIAG,
        GrB_COLLE, GrB_COLGT, GrB_ROWLE, GrB_ROWGT,
        GxB_FLIPDIAGINDEX_INT32, GxB_FLIPDIAGINDEX_INT64 } ;

    for (int k = 0 ; k < 16 ; k++)
    {
        HEADER ;
        GrB_IndexUnaryOp op = idxops [k] ;
        fprintf (fp, "GB_enumify_select / GB_macrofy_select: %s\n", op->name) ;
        printf ("GB_enumify_select / GB_macrofy_select: %s\n", op->name) ;
        // GxB_print (op, 3) ;
        GB_enumify_select (&scode, /* C iso: */ false, /* inplace A: */ false,
            op, /* flipij: */ false, A) ;
//      printf ("select scode: %016" PRIx64 "\n", scode) ;
        GB_macrofy_select (fp, scode, op, GrB_BOOL) ;
    }

    HEADER ;
    fprintf (fp, "GB_enumify_select / GB_macrofy_select: opi32\n") ;
    printf ("GB_enumify_select / GB_macrofy_select: opi32\n") ;
    GrB_IndexUnaryOp opi ;
    OK (GxB_IndexUnaryOp_new (&opi, (GxB_index_unary_function) opi32func,
        GxB_FC32, GxB_FC32, GxB_FC32, "opi32func", OPI32_DEFN)) ;
//  GxB_print (opi, 3) ;
    GB_enumify_select (&scode, /* C iso: */ false, /* inplace A: */ false,
        opi, /* flipij: */ false, A) ;
//  printf ("select scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_select (fp, scode, opi, GxB_FC32) ;
    GrB_free (&opi) ;

    //--------------------------------------------------------------------------
    // GB_enumify_apply / GB_macrofy_apply
    //--------------------------------------------------------------------------

    GrB_UnaryOp unops [9] = { GxB_ONE_BOOL,
        GxB_POSITIONI_INT32,  GxB_POSITIONI_INT64,
        GxB_POSITIONI1_INT32, GxB_POSITIONI1_INT64,
        GxB_POSITIONJ_INT32,  GxB_POSITIONJ_INT64,
        GxB_POSITIONJ1_INT32, GxB_POSITIONJ1_INT64 } ;

    for (int k = 0 ; k < 9 ; k++)
    {
        HEADER ;
        GrB_UnaryOp op = unops [k] ;
        fprintf (fp, "GB_enumify_apply / GB_macrofy_apply: %s\n", op->name) ;
        printf ("GB_enumify_apply / GB_macrofy_apply: %s\n", op->name) ;
        GB_enumify_apply (&scode, GxB_SPARSE, true, GrB_INT32,
            (GB_Operator) op, false, A) ;
//      printf ("apply  scode: %016" PRIx64 "\n", scode) ;
        GB_macrofy_apply (fp, scode, (GB_Operator) op, GrB_INT32, GrB_INT32) ;
    }

    HEADER ;
    GrB_UnaryOp op1 = GxB_SQRT_FC64 ;
    fprintf (fp, "GB_enumify_apply / GB_macrofy_apply: %s\n", op1->name) ;
    printf ("GB_enumify_apply / GB_macrofy_apply: %s\n", op1->name) ;
//  printf ("apply  scode: %016" PRIx64 "\n", scode) ;
    GB_enumify_apply (&scode, GxB_SPARSE, true, GrB_INT32,
            (GB_Operator) op1, false, A) ;
    GB_macrofy_apply (fp, scode, (GB_Operator) op1, GrB_INT32, GrB_INT32) ;

    //--------------------------------------------------------------------------
    // GB_enumify_build / GB_macrofy_build
    //--------------------------------------------------------------------------

    HEADER ;
    GrB_BinaryOp op2 = GxB_TIMES_FC32 ;
    fprintf (fp, "GB_enumify_build / GB_macrofy_build: %s\n", op2->name) ;
    printf ("GB_enumify_build / GB_macrofy_build: %s\n", op2->name) ;
    GB_enumify_build (&scode, op2, GrB_BOOL, GrB_BOOL) ;
//  printf ("build  scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_build (fp, scode, op2, GrB_BOOL, GrB_BOOL) ;

    HEADER ;
    op2 = GrB_LAND ;
    fprintf (fp, "GB_enumify_build / GB_macrofy_build: %s\n", op2->name) ;
    printf ("GB_enumify_build / GB_macrofy_build: %s\n", op2->name) ;
    GB_enumify_build (&scode, op2, GxB_FC32, GxB_FC32) ;
//  printf ("build  scode: %016" PRIx64 "\n", scode) ;
    GB_macrofy_build (fp, scode, op2, GxB_FC32, GxB_FC32) ;

    //--------------------------------------------------------------------------
    // GB_enumify_assign / GB_macrofy_assign
    //--------------------------------------------------------------------------

    HEADER ;
    GrB_BinaryOp accum = NULL ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)=A (assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        A, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = NULL ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:s:hi,lo:s:hi)=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:s:hi,lo:s:hi)=A (assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_STRIDE, GB_STRIDE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        A, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = NULL ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(i,J)=s (row assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(i,J)=s (row assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_ALL, GB_LIST, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ NULL, /* scalar_type: */ GxB_FC32,
        /* assign_kind: */ GB_ROW_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = NULL ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(I,j)=s (col assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(I,j)=s (col assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_LIST, GB_ALL, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ NULL, /* scalar_type: */ GxB_FC32,
        /* assign_kind: */ GB_COL_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = NULL ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)=A (assign) \n") ;
    GB_enumify_assign (&scode, C_iso, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ NULL, /* scalar_type: */ GrB_FP32, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = GrB_PLUS_FP32 ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)+=s (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)+=s (assign) \n") ;
    GB_enumify_assign (&scode, C_iso, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ NULL, /* scalar_type: */ GrB_FP32, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = GrB_PLUS_FP32 ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)+=s (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)+=s (assign) \n") ;
    GB_enumify_assign (&scode, C_iso, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ NULL, /* scalar_type: */ GrB_INT32, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = GrB_PLUS_FP32 ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)+=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)+=A (assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ H, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = GrB_LAND ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)&=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)&=A (assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ A, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    // accum ztype == ctype
    // accum xtype != ctype

    HEADER ;
    accum = GrB_LT_FP32 ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)<=A (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C(lo:hi,lo:hi)<=A (assign) \n") ;
    GB_enumify_assign (&scode, C, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ A, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

    HEADER ;
    accum = GrB_LT_FP32 ;
    fprintf (fp, "GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)<=H (assign) \n") ;
    printf ("GB_enumify_assign / GB_macrofy_assign: "
        "C_iso(lo:hi,lo:hi)<=H (assign) \n") ;
    GB_enumify_assign (&scode, C_iso, false, GB_RANGE, GB_RANGE, /* M: */ NULL,
        /* Mask_struct: */ false, /* Mask_comp: */ false, accum,
        /* A: */ H, /* scalar_type: */ NULL, /* assign_kind: */ GB_ASSIGN) ;
    GB_macrofy_assign (fp, scode, accum,
        /* ctype: */ GrB_BOOL, /* atype: */ GrB_BOOL) ;

#if 0
void GB_enumify_assign      // enumerate a GrB_assign problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    GrB_Matrix C,
    bool C_replace,
    // index types:
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 3: list
    int Jkind,              // ditto
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp accum,     // the accum operator (may be NULL)
    // A matrix or scalar
    GrB_Matrix A,           // NULL for scalar assignment
    GrB_Type scalar_type,
    int assign_kind         // 0: assign, 1: subassign, 2: row, 3: col
)
#endif


    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    fclose (fp) ;
    GrB_free (&A) ;
    GrB_free (&B) ;
    GrB_free (&C) ;
    GrB_free (&H) ;
    GrB_free (&C_iso) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test21:  all tests passed\n\n") ;
}

