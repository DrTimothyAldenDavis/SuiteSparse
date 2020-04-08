//------------------------------------------------------------------------------
// GB_mex_about: print the 'about' information
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_about"

GrB_Info ack (int64_t *stuff, GrB_Matrix GunkIt) ;

GrB_Info ack (int64_t *stuff, GrB_Matrix GunkIt)
{
    GB_WHERE ("ack") ;
    GB_RETURN_IF_NULL (stuff) ;
    GB_RETURN_IF_NULL_OR_FAULTY (GunkIt) ;
    return (GrB_SUCCESS) ;
}

bool select_plus_one (GrB_Index i, GrB_Index j, GrB_Index nrows,
    GrB_Index ncols, const double *x, const double *thunk) ;

bool select_nothing (GrB_Index i, GrB_Index j, GrB_Index nrows,
    GrB_Index ncols, const void *x, const void *thunk) ;

bool select_plus_one (GrB_Index i, GrB_Index j, GrB_Index nrows,
    GrB_Index ncols, const double *x, const double *thunk)
{
    // return true if x >= thunk+1
    return ((*x) >= ((*thunk)+1)) ;
}

bool select_nothing (GrB_Index i, GrB_Index j, GrB_Index nrows,
    GrB_Index ncols, const void *x, const void *thunk)
{
    return (false) ;
}

typedef int16_t user_int ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // start error log
    //--------------------------------------------------------------------------

    FILE *f = fopen ("errlog2.txt", "w") ;
    printf ("in %s\n", __FILE__) ;

    //--------------------------------------------------------------------------
    // test GrB_init with invalid mode
    //--------------------------------------------------------------------------

    GB_Global_GrB_init_called_set (false) ;
    GrB_Info info = GrB_init (911) ;
    printf ("expected error: [%d]\n", info) ;
    mxAssert (info == GrB_INVALID_VALUE, "error must be 'invalid value'") ;

    bool malloc_debug = GB_mx_get_global (true) ;

    GB_WHERE (USAGE) ;

    printf ("table of codes:\n") ;
    printf ("bool  code: %d class: %d\n", GB_BOOL_code   , mxLOGICAL_CLASS) ;
    printf ("int8  code: %d class: %d\n", GB_INT8_code   , mxINT8_CLASS   ) ;
    printf ("uint8 code: %d class: %d\n", GB_UINT8_code  , mxUINT8_CLASS  ) ;
    printf ("int16 code: %d class: %d\n", GB_INT16_code  , mxINT16_CLASS  ) ;
    printf ("uin16 code: %d class: %d\n", GB_UINT16_code , mxUINT16_CLASS ) ;
    printf ("int32 code: %d class: %d\n", GB_INT32_code  , mxINT32_CLASS  ) ;
    printf ("uin32 code: %d class: %d\n", GB_UINT32_code , mxUINT32_CLASS ) ;
    printf ("in64  code: %d class: %d\n", GB_INT64_code  , mxINT64_CLASS  ) ;
    printf ("uin64 code: %d class: %d\n", GB_UINT64_code , mxUINT64_CLASS ) ;
    printf ("fp32  code: %d class: %d\n", GB_FP32_code   , mxSINGLE_CLASS ) ;
    printf ("fp64  code: %d class: %d\n", GB_FP64_code   , mxDOUBLE_CLASS ) ;
    printf ("struct   class: %d\n", mxSTRUCT_CLASS) ;
    printf ("cell     class: %d\n", mxCELL_CLASS) ;
    printf ("void     class: %d\n", mxVOID_CLASS) ;
    printf ("function class: %d\n", mxFUNCTION_CLASS) ;
    printf ("unknown mxClassID: %d\n", mxUNKNOWN_CLASS ) ;

    printf ("in %s:\n%s", __FILE__, GrB_error ( )) ;

    printf ("sizeof (struct GB_Type_opaque) %d\n",
             sizeof (struct GB_Type_opaque)) ;
    printf ("sizeof (struct GB_UnaryOp_opaque) %d\n",
             sizeof (struct GB_UnaryOp_opaque)) ;
    printf ("sizeof (struct GB_BinaryOp_opaque) %d\n",
             sizeof (struct GB_BinaryOp_opaque)) ;
    printf ("sizeof (struct GB_SelectOp_opaque) %d\n",
             sizeof (struct GB_SelectOp_opaque)) ;
    printf ("sizeof (struct GB_Monoid_opaque) %d\n",
             sizeof (struct GB_Monoid_opaque)) ;
    printf ("sizeof (struct GB_Semiring_opaque) %d\n",
             sizeof (struct GB_Semiring_opaque)) ;
    printf ("sizeof (struct GB_Vector_opaque) %d\n",
             sizeof (struct GB_Vector_opaque)) ;
    printf ("sizeof (struct GB_Matrix_opaque) %d\n",
             sizeof (struct GB_Matrix_opaque)) ;
    printf ("sizeof (struct GB_Descriptor_opaque) %d\n",
             sizeof (struct GB_Descriptor_opaque)) ;

    info = GB_ERROR (GrB_PANIC, (GB_LOG,
        "just testing the error log ... not really a panic\n"
        "hello world, the answer is %d", 42)) ;

    printf ("%s", GrB_error ( )) ;

    size_t s ;
    GxB_Type_size (&s, GrB_BOOL  ) ; printf ("%d %d\n", s, sizeof (bool      ));
    GxB_Type_size (&s, GrB_INT8  ) ; printf ("%d %d\n", s, sizeof (int8_t    ));
    GxB_Type_size (&s, GrB_UINT8 ) ; printf ("%d %d\n", s, sizeof (uint8_t   ));
    GxB_Type_size (&s, GrB_INT16 ) ; printf ("%d %d\n", s, sizeof (int16_t   ));
    GxB_Type_size (&s, GrB_UINT16) ; printf ("%d %d\n", s, sizeof (uint16_t  ));
    GxB_Type_size (&s, GrB_INT32 ) ; printf ("%d %d\n", s, sizeof (int32_t   ));
    GxB_Type_size (&s, GrB_UINT32) ; printf ("%d %d\n", s, sizeof (uint32_t  ));
    GxB_Type_size (&s, GrB_INT64 ) ; printf ("%d %d\n", s, sizeof (int64_t   ));
    GxB_Type_size (&s, GrB_UINT64) ; printf ("%d %d\n", s, sizeof (uint64_t  ));
    GxB_Type_size (&s, GrB_FP32  ) ; printf ("%d %d\n", s, sizeof (float     ));
    GxB_Type_size (&s, GrB_FP64  ) ; printf ("%d %d\n", s, sizeof (double    ));

    printf ("info is %d\n", info) ;

    GrB_Type t ;

    GB_UnaryOp_check (GrB_LNOT, "LNOT", GxB_COMPLETE, stdout, Context) ;
    GxB_UnaryOp_ztype (&t, GrB_LNOT) ;
    GB_UnaryOp_check (t, "ztype", GxB_COMPLETE, stdout, Context) ;
    GxB_UnaryOp_xtype (&t, GrB_LNOT) ;
    GB_UnaryOp_check (t, "xtype", GxB_COMPLETE, stdout, Context) ;

    GB_UnaryOp_check (GxB_LNOT_FP32, "LNOT_FP32", GxB_COMPLETE, stdout, Context) ;
    GxB_UnaryOp_ztype (&t, GxB_LNOT_FP32) ;
    GB_UnaryOp_check (t, "ztype", GxB_COMPLETE, stdout, Context) ;
    GxB_UnaryOp_xtype (&t, GxB_LNOT_FP32) ;
    GB_UnaryOp_check (t, "xtype", GxB_COMPLETE, stdout, Context) ;

    GB_BinaryOp_check (GxB_ISEQ_INT32, "ISEQ_INT32", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_ztype (&t, GxB_ISEQ_INT32) ;
    GB_BinaryOp_check (t, "ztype", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_xtype (&t, GxB_ISEQ_INT32) ;
    GB_BinaryOp_check (t, "xtype", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_ytype (&t, GxB_ISEQ_INT32) ;
    GB_BinaryOp_check (t, "ytype", GxB_COMPLETE, stdout, Context) ;

    GB_BinaryOp_check (GrB_EQ_INT32, "EQ_INT32", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_ztype (&t, GrB_EQ_INT32) ;
    GB_BinaryOp_check (t, "ztype", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_xtype (&t, GrB_EQ_INT32) ;
    GB_BinaryOp_check (t, "xtype", GxB_COMPLETE, stdout, Context) ;
    GxB_BinaryOp_ytype (&t, GrB_EQ_INT32) ;
    GB_BinaryOp_check (t, "ytype", GxB_COMPLETE, stdout, Context) ;

    GrB_Monoid m ;
    GrB_BinaryOp op ;

    GrB_Monoid_new_UINT16 (&m, GrB_PLUS_UINT16, (uint16_t) 0) ;
    GB_Monoid_check (m, "plus uint16 monoid", GxB_COMPLETE, stdout, Context) ;
    uint16_t id ;
    GxB_Monoid_identity (&id, m) ;
    printf ("id is %d\n", id) ;
    GxB_Monoid_operator (&op, m) ;
    GB_Monoid_check (op, "plus op from monoid", GxB_COMPLETE, stdout, Context) ;

    GrB_Monoid_free (&m) ;

    int16_t id0 = INT16_MIN ;

    GrB_Monoid_new_INT16 (&m, GrB_MAX_INT16, id0) ;
    GB_Monoid_check (m, "max int16 monoid", GxB_COMPLETE, stdout, Context) ;
    int16_t id1 ;
    GxB_Monoid_identity (&id1, m) ;
    printf ("id1 is %d\n", id1) ;
    GxB_Monoid_operator (&op, m) ;
    GB_BinaryOp_check (op, "plus op from monoid", GxB_COMPLETE, stdout,
        Context) ;

    GrB_Semiring sem ;
    GrB_Semiring_new (&sem, m, GrB_TIMES_INT16) ;
    GB_Semiring_check (sem, "\nnew sem", GxB_COMPLETE, stdout, Context) ;

    GrB_Monoid mm ;
    GxB_Semiring_add (&mm, sem) ;
    GB_Monoid_check (mm, "sem mm", GxB_COMPLETE, stdout, Context) ;
    GxB_Semiring_multiply (&op, sem) ;
    GB_BinaryOp_check (op, "sem mult", GxB_COMPLETE, stdout, Context) ;

    GrB_Monoid_free (&m) ;
    GrB_Semiring_free (&sem) ;

    int64_t *stuff = NULL ;
    int64_t morestuff = 44 ;
    GrB_Matrix Gunk ;
    GrB_Matrix_new (&Gunk, GrB_FP64, 5, 5) ;
    info = ack (&morestuff, Gunk) ;

    GxB_Matrix_type (&t, Gunk) ;
    GB_Type_check (t, "matrix Gunk type is:", GxB_COMPLETE, stdout, Context) ;

    GrB_Vector victor ;
    GrB_Vector_new (&victor, GrB_UINT32, 43) ;
    GxB_Vector_type (&t, victor) ;
    GB_Type_check (t, "victor type is:", GxB_COMPLETE, stdout, Context) ;
    GxB_Type_size (&s, t) ;
    printf ("and its size of type is %d\n", s) ;
    GrB_Vector_free (&victor) ;

    //--------------------------------------------------------------------------
    // descriptor
    //--------------------------------------------------------------------------

    GrB_Descriptor Duh ;
    GrB_Desc_Value val ;

    GrB_Descriptor_new (&Duh) ;
    GB_Descriptor_check (Duh, "\n---------------------------------- Duh:",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    GxB_Desc_set (Duh, GrB_INP0, GrB_TRAN) ;
    GB_Descriptor_check (Duh, "\n------------------------------- Duh set:",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    GxB_Desc_set (Duh, GrB_MASK, GrB_COMP) ;
    GB_Descriptor_check (Duh, "\n-----Duh set mask",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    GxB_Desc_set (Duh, GrB_OUTP, GrB_REPLACE) ;
    GB_Descriptor_check (Duh, "\n-----Duh set out",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    GxB_Desc_set (Duh, GrB_MASK, GxB_DEFAULT) ;
    GB_Descriptor_check (Duh, "\n-----Duh set mask back",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    info = GxB_Desc_set (Duh, GrB_INP1, GrB_REPLACE) ;
    printf ("%s\n", GrB_error () ) ;
    GB_Descriptor_check (Duh, "\n-----Duh set in1",
        GxB_COMPLETE, stdout, Context) ;
    GxB_Desc_get (Duh, GrB_OUTP, &val) ; printf ("got outp %d\n", val) ;
    GxB_Desc_get (Duh, GrB_MASK, &val) ; printf ("got mask %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP0, &val) ; printf ("got inp0 %d\n", val) ;
    GxB_Desc_get (Duh, GrB_INP1, &val) ; printf ("got inp1 %d\n", val) ;

    GrB_Descriptor_free (&Duh) ;

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    printf ("ok:\n%s", GrB_error ( )) ;

    info = ack (NULL, Gunk) ;

    printf ("%s", GrB_error ( )) ;

    Gunk->magic = 999 ;
    info = ack (&morestuff, Gunk) ;
    printf ("%s", GrB_error ( )) ;

    Gunk->magic = GB_MAGIC ;
    GrB_Matrix_free (&Gunk) ;

    GB_Type_check (Complex, "user Complex type", GxB_COMPLETE, stdout, Context);
    GxB_Type_size (&s, Complex) ;
    printf ("size is %d\n", (int) s) ;

    //--------------------------------------------------------------------------
    // about
    //--------------------------------------------------------------------------

    #ifdef GxB_SUITESPARSE_GRAPHBLAS

    printf ("\nAbout:\n%s\n", GxB_IMPLEMENTATION_ABOUT) ;
    printf ("Date: %s\n", GxB_IMPLEMENTATION_DATE) ;
    printf ("Implementation: %d.%d.%d ("GBu")\n",
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB,
        GxB_IMPLEMENTATION) ;
    printf ("License:%s\n", GxB_IMPLEMENTATION_LICENSE) ;

    printf ("Spec: %d.%d.%d ("GBu")\n",
        GxB_SPEC_MAJOR, GxB_SPEC_MINOR, GxB_SPEC_SUB, GxB_SPEC_VERSION) ;
    printf ("Spec Date: %s\n", GxB_SPEC_DATE) ;
    printf ("About the spec:\n%s\n", GxB_SPEC_ABOUT) ;

    #if GxB_SPEC_VERSION >= GxB_VERSION(1,0,0)
    printf ("The spec is >= version 1.0.0\n") ;
    #else
    printf ("The spec is < version 1.0.0\n") ;
    #endif

    #if GxB_SPEC_VERSION < GxB_VERSION(2,3,0)
    printf ("The spec is < version 2.3.0\n") ;
    #else
    printf ("The spec is >= version 2.3.0\n") ;
    #endif

    #if GxB_IMPLEMENTATION < GxB_VERSION(1,0,0)
    printf ("This implementation is <  version 1.0.0\n") ;
    #else
    printf ("This implementation is >= version 1.0.0\n") ;
    #endif

    #endif

    //--------------------------------------------------------------------------
    // types
    //--------------------------------------------------------------------------

    printf ("built-in types:\n") ;
    GB_Type_check (GrB_BOOL, "bool", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_INT8, "int8", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_UINT8, "uint8", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_INT16, "int16", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_UINT16, "uint16", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_INT32, "int32", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_UINT32, "uint32", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_INT64, "int64", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_UINT64, "uint64", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_FP32, "fp32", GxB_COMPLETE, stdout, Context) ;
    GB_Type_check (GrB_FP64, "fp64", GxB_COMPLETE, stdout, Context) ;

    printf ("\nprinting built-in types:\n") ;
    bool       b = true ;
    int8_t    i8 = 22   ;
    uint8_t   u8 = 44   ;
    int16_t  i16 = 909  ;
    uint16_t u16 = 777  ;
    int32_t  i32 = 3203 ;
    uint32_t u32 = 8080 ;
    int64_t  i64 = -987 ;
    uint64_t u64 = 987  ;
    float    f32 = 3.14 ;
    double   f64 = 99.4 ;

    GB_code_check (GB_BOOL_code,   &b  , stdout, Context) ; printf ("\n");
    GB_code_check (GB_INT8_code,   &i8 , stdout, Context) ; printf ("\n");
    GB_code_check (GB_UINT8_code,  &u8 , stdout, Context) ; printf ("\n");
    GB_code_check (GB_INT16_code,  &i16, stdout, Context) ; printf ("\n");
    GB_code_check (GB_UINT16_code, &u16, stdout, Context) ; printf ("\n");
    GB_code_check (GB_INT32_code,  &i32, stdout, Context) ; printf ("\n");
    GB_code_check (GB_UINT32_code, &u32, stdout, Context) ; printf ("\n");
    GB_code_check (GB_INT64_code,  &i64, stdout, Context) ; printf ("\n");
    GB_code_check (GB_UINT64_code, &u64, stdout, Context) ; printf ("\n");
    GB_code_check (GB_FP32_code,   &f32, stdout, Context) ; printf ("\n");
    GB_code_check (GB_FP64_code,   &f64, stdout, Context) ; printf ("\n");
    GB_code_check (GB_UDT_code,    &f64, stdout, Context) ; printf ("\n");

    for (int i = 0 ; i <= GrB_PANIC + 1 ; i++)
    {
        printf ("info: %2d %s\n", i, GB_status_code (i)) ;
    }

    //--------------------------------------------------------------------------
    // threading
    //--------------------------------------------------------------------------

    #if defined (USER_POSIX_THREADS)
    printf ("User threads: POSIX\n") ;
    #elif defined (USER_WINDOWS_THREADS)
    printf ("User threads: Windows\n") ;
    #elif defined (USER_ANSI_THREADS)
    printf ("User threads: ANSI\n") ;
    #elif defined (USER_OPENMP_THREADS)
    printf ("User threads: OpenMP\n") ;
    #elif defined (USER_NO_THREADS)
    printf ("User threads: none\n") ;
    #else
    printf ("User threads: not specific (none)\n") ;
    #endif

    //--------------------------------------------------------------------------
    // global get/set
    //--------------------------------------------------------------------------

    double h ;
    GxB_Format_Value ff ;
    GxB_Global_Option_get (GxB_HYPER, &h) ;
    GxB_Global_Option_get (GxB_FORMAT, &ff) ;
    printf ("hyper_ratio %g csc %d\n", h, (ff == GxB_BY_COL)) ;

    GrB_Mode mode ;
    GxB_Global_Option_get (GxB_MODE, &mode) ;
    printf ("mode: %d\n", mode) ;

    GxB_Thread_Model threading ;
    GxB_Global_Option_get (GxB_THREAD_SAFETY, &threading) ;
    printf ("thread safety: %d\n", threading) ;
    GxB_Global_Option_get (GxB_THREADING, &threading) ;
    printf ("threading: %d\n", threading) ;

    int nthreads ;
    GxB_Global_Option_get (GxB_NTHREADS, &nthreads) ;
    printf ("# threads: %d\n", nthreads) ;

    double chunk ;
    GxB_Global_Option_get (GxB_CHUNK, &chunk) ;
    printf ("chunk: %g\n", chunk) ;

    //--------------------------------------------------------------------------
    // check A and B aliased
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    OK (GrB_Matrix_new (&A, GrB_BOOL, 10000, 10000)) ;
    OK (GrB_Matrix_new (&B, GrB_BOOL, 10000, 10000)) ;
    OK (GrB_Matrix_setElement_BOOL (A, true, 0, 0)) ;
    OK (GrB_Matrix_setElement_BOOL (B, true, 0, 0)) ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_nvals (&nvals, B)) ;
    CHECK (!GB_aliased (A, B)) ;
    int64_t *Bh_save = B->h ;
    B->h = A->h ; B->h_shallow = true ;
    CHECK (GB_aliased (A, B)) ;
    B->h = Bh_save ; B->h_shallow = false ;
    CHECK (!GB_aliased (A, B)) ;
    OK (GxB_Matrix_fprint (A, "A", 3, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", 3, NULL)) ;
    GrB_Matrix_free (&A) ;
    GrB_Matrix_free (&B) ;

    //--------------------------------------------------------------------------
    // check descripter set/get for nthreads and chunk
    //--------------------------------------------------------------------------

    GrB_Descriptor desc ;
    OK (GrB_Descriptor_new (&desc)) ;
    OK (GxB_Desc_set (desc, GxB_NTHREADS, 42)) ;
    OK (GxB_Desc_set (desc, GxB_CHUNK, (double) 12345)) ;
    OK (GxB_Desc_get (desc, GxB_CHUNK, &chunk)) ;
    OK (GxB_Desc_get (desc, GxB_NTHREADS, &nthreads)) ;
    OK (GxB_Descriptor_fprint (desc, "desc", GxB_COMPLETE, NULL)) ;
    CHECK (chunk == 12345) ;
    CHECK (nthreads == 42) ;
    GrB_Descriptor_free (&desc) ;

    //--------------------------------------------------------------------------
    // make a shallow copy of an empty matrix
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_BOOL, 10000, 10000)) ;
    OK (GB_shallow_copy (&C, A->is_csc, A, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free (&A) ;
    GrB_Matrix_free (&C) ;

    //--------------------------------------------------------------------------
    // tests with memory tracking off
    //--------------------------------------------------------------------------

    GB_Global_malloc_tracking_set (false) ;
    GB_void *p = GB_malloc_memory (4, sizeof (int64_t)) ;
    CHECK (p != NULL) ;
    GB_free_memory (p, 4, sizeof (int64_t)) ;
    p = GB_calloc_memory (4, sizeof (int64_t)) ;
    CHECK (p != NULL) ;
    bool ok = true ;
    p = GB_realloc_memory (6, 4, sizeof (int64_t), p, &ok) ;
    CHECK (p != NULL) ;
    CHECK (ok) ;
    GB_free_memory (p, 6, sizeof (int64_t)) ;
    p = NULL ;
    printf ("in use:   "GBd"\n", GB_Global_inuse_get ( )) ;
    printf ("max used: "GBd"\n", GB_Global_maxused_get ( )) ;

    CHECK (!GB_Global_malloc_is_thread_safe_get ( )) ;
    GB_Global_malloc_is_thread_safe_set (true) ;
    CHECK (GB_Global_malloc_is_thread_safe_get ( )) ;
    GB_Global_malloc_is_thread_safe_set (false) ;
    CHECK (!GB_Global_malloc_is_thread_safe_get ( )) ;

    GB_Global_malloc_tracking_set (true) ;

    //--------------------------------------------------------------------------
    // other global settings
    //--------------------------------------------------------------------------

    GB_Global_hack_set (90123) ;
    CHECK (GB_Global_hack_get ( ) == 90123) ;

    GrB_Info expected = GrB_INVALID_VALUE ;

    //--------------------------------------------------------------------------
    // GB_pslice
    //--------------------------------------------------------------------------

    int64_t *Slice = NULL ;
    GB_pslice (&Slice, NULL, 0, 4) ;
    for (int t = 0 ; t < 4 ; t++) CHECK (Slice [t] == 0) ;
    GB_free_memory (Slice, 5, sizeof (int64_t)) ;
    Slice = NULL ;

    //--------------------------------------------------------------------------
    // renamed boolean monoids
    //--------------------------------------------------------------------------

    GrB_Monoid mono = NULL ;

    // DIV renamed to FIRST
    OK (GrB_Monoid_new_BOOL (&mono, GrB_DIV_BOOL, (bool) false)) ;
    printf ("\ndiv_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    // RDIV renamed to SECOND
    OK (GrB_Monoid_new_BOOL (&mono, GxB_RDIV_BOOL, (bool) false)) ;
    printf ("\nrdiv_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    // ISGT renamed to GT
    OK (GrB_Monoid_new_BOOL (&mono, GxB_ISGT_BOOL, (bool) false)) ;
    printf ("\nisgt_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    // ISLT renamed to LT
    OK (GrB_Monoid_new_BOOL (&mono, GxB_ISLT_BOOL, (bool) false)) ;
    printf ("\nislt_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    // ISGE renamed to GE
    OK (GrB_Monoid_new_BOOL (&mono, GxB_ISGE_BOOL, (bool) false)) ;
    printf ("\nisge_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    // ISLE renamed to LE
    OK (GrB_Monoid_new_BOOL (&mono, GxB_ISLE_BOOL, (bool) false)) ;
    printf ("\nisle_bool monoid:\n") ;
    OK (GxB_Monoid_fprint (mono, "mono", GxB_COMPLETE, NULL)) ;
    GrB_Monoid_free (&mono) ;

    //--------------------------------------------------------------------------
    // select
    //--------------------------------------------------------------------------

    GrB_Type user_type = NULL ;
    OK (GrB_Type_new (&user_type, sizeof (user_int))) ;
    OK (GrB_Matrix_new (&A, user_type, 10, 10)) ;
    OK (GrB_Matrix_new (&B, GrB_INT16, 10, 10)) ;
    user_int value ;
    for (int i = 0 ; i < 10 ; i++)
    {
        value = (int64_t) i ;
        OK (GrB_Matrix_setElement_UDT (A, &value, i, i)) ;
        OK (GrB_Matrix_setElement_INT16 (B, i, i, i)) ;
    }
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GrB_Matrix_nvals (&nvals, B)) ;
    OK (GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_fprint (B, "A", GxB_COMPLETE, NULL)) ;

    // expected = GrB_DIMENSION_MISMATCH ;
    // ERR (GxB_Matrix_select (A, NULL, NULL, GxB_NE_THUNK, A, A, NULL)) ;
    // printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    GxB_Scalar thunk = NULL ;
    OK (GxB_Scalar_new (&thunk, user_type)) ;
    GrB_Type type2 = NULL ;
    OK (GxB_Scalar_type (&type2, thunk)) ;
    CHECK (type2 == user_type) ;
    OK (GxB_Scalar_fprint (thunk, "thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_select (A, NULL, NULL, GxB_NE_THUNK, A, thunk, NULL)) ;
    // printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    value = (int64_t) 4 ;
    OK (GxB_Scalar_setElement_UDT (thunk, &value)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GxB_Matrix_select (A, NULL, NULL, GxB_GE_THUNK, A, thunk, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    GxB_Scalar thunk2 = NULL ;
    OK (GxB_Scalar_new (&thunk2, GrB_INT16)) ;
    OK (GxB_Scalar_setElement_INT16 (thunk2, 4)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GxB_Matrix_select (A, NULL, NULL, GxB_GE_ZERO, A, NULL, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    ERR (GxB_Matrix_select (A, NULL, NULL, GxB_GT_ZERO, A, NULL, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    ERR (GxB_Matrix_select (A, NULL, NULL, GxB_LT_ZERO, A, NULL, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    ERR (GxB_Matrix_select (A, NULL, NULL, GxB_LE_ZERO, A, NULL, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;

    ERR (GxB_Matrix_select (B, NULL, NULL, GxB_LE_THUNK, B, thunk, NULL)) ;
    printf ("Expected error: info: %d\n%s\n", info, GrB_error ( )) ;
    GrB_Matrix_free (&B) ;

    OK (GrB_Matrix_new (&B, user_type, 10, 10)) ;
    printf ("\n============== B = select (A != 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_NONZERO, A, NULL, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A == 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_EQ_ZERO, A, NULL, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A != 4)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_NE_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A == 4)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_EQ_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;

    GrB_Matrix_free (&B) ;
    GrB_Matrix_free (&A) ;
    GxB_Scalar_free (&thunk) ;
    GxB_Scalar_free (&thunk2) ;
    GrB_Type_free (&user_type) ;

    OK (GrB_Matrix_new (&A, GrB_BOOL, 10, 10)) ;
    OK (GrB_Matrix_new (&B, GrB_BOOL, 10, 10)) ;
    OK (GxB_Scalar_new (&thunk, GrB_BOOL)) ;
    OK (GxB_Scalar_setElement_BOOL (thunk, 0)) ;
    for (int i = 0 ; i < 10 ; i++)
    {
        OK (GrB_Matrix_setElement_BOOL (A, (bool) (i % 2), i, i)) ;
    }
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    OK (GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL)) ;

    printf ("\n============== B = select (A > 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_GT_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A >= 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_GE_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A < 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_LT_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;
    printf ("\n============== B = select (A <= 0)\n") ;
    OK (GxB_Matrix_select (B, NULL, NULL, GxB_LE_THUNK, A, thunk, NULL)) ;
    OK (GxB_Matrix_fprint (B, "B", GxB_COMPLETE, NULL)) ;

    GrB_Matrix_free (&B) ;
    GrB_Matrix_free (&A) ;
    GxB_Scalar_free (&thunk) ;

    //--------------------------------------------------------------------------
    // GxB_Matrix_fprint for a slice or hyperslice
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP64, 8, 8)) ;
    for (int i = 0 ; i < 8 ; i++)
    {
        for (int j = 0 ; j < 8 ; j++)
        {
            OK (GrB_Matrix_setElement_FP64 (A, i*100+j, i, j)) ;
        }
    }
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    expected = GrB_INVALID_OBJECT ;

    for (int hyper = 0 ; hyper <= 1 ; hyper++)
    {
        if (hyper)
        {
            OK (GxB_Matrix_Option_set (A, GxB_HYPER, GxB_ALWAYS_HYPER)) ;
        }

        GrB_Matrix Aslice [2] = { NULL, NULL } ;
        int64_t Slice [8] ;
        Slice [0] = 0 ;
        Slice [1] = 4 ;
        Slice [2] = 8 ;
        OK (GB_slice (A, 2, Slice, Aslice, Context)) ;
        OK (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_COMPLETE, NULL)) ;
        OK (GxB_Matrix_fprint (Aslice [1], "Aslice [1]", GxB_COMPLETE, NULL)) ;

        GB_Pending gunk ;
        Aslice [0]->Pending = &gunk ;
        ERR (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SHORT, NULL)) ;
        Aslice [0]->Pending = NULL ;
        OK (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SILENT, NULL)) ;

        int64_t a1save = Aslice [0]->nvec ;
        Aslice [0]->nvec = 999999 ;
        ERR (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SHORT, NULL)) ;
        Aslice [0]->nvec = a1save ;
        OK (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SILENT, NULL)) ;

        Aslice [0]->i_shallow = false ;
        ERR (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SHORT, NULL)) ;
        Aslice [0]->i_shallow = true ;
        OK (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SILENT, NULL)) ;

        int64_t hfirst = Aslice [0]->hfirst ;
        Aslice [0]->hfirst = -1 ;
        ERR (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SHORT, NULL)) ;
        Aslice [0]->hfirst = 0 ;
        OK (GxB_Matrix_fprint (Aslice [0], "Aslice [0]", GxB_SILENT, NULL)) ;

        GrB_Matrix_free (&Aslice [0]) ;
        GrB_Matrix_free (&Aslice [1]) ;
    }

    //--------------------------------------------------------------------------
    // pending tuples
    //--------------------------------------------------------------------------

    GrB_Matrix_free (&A) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 8, 8)) ;

    GrB_Index I [1] = { 0 }, J [1] = { 0 } ;
    OK (GrB_Matrix_assign_FP64 (A, NULL, GrB_PLUS_FP64,
        (double) 2, I, 1, J, 1, NULL)) ;
    GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 3, 0, 0)) ;
    GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL) ;

    GrB_Matrix_free (&A) ;

    //--------------------------------------------------------------------------
    // select error handling
    //--------------------------------------------------------------------------

    GxB_SelectOp selectop = NULL ;
    OK (GxB_SelectOp_new (&selectop, select_plus_one, GrB_FP64, GrB_FP64)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 8, 8)) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, 8, 8)) ;
    for (int i = 0 ; i < 8 ; i++)
    {
        OK (GrB_Matrix_setElement_FP64 (A, i, i, i)) ;
    }
    OK (GxB_Matrix_fprint (A, "A", GxB_COMPLETE, NULL)) ;
    OK (GxB_Scalar_new (&thunk, GrB_FP64)) ;
    OK (GxB_Scalar_setElement_FP64 (thunk, 4)) ;
    OK (GxB_Matrix_select (C, NULL, NULL, selectop, A, thunk, NULL)) ;

    printf ("\nprint in one-based, long format:\n") ;
    GB_Global_print_one_based_set (true) ;
    GB_Global_print_format_set (1) ;
    OK (GxB_Matrix_fprint (C, "C", GxB_COMPLETE, NULL)) ;
    GB_Global_print_one_based_set (false) ;
    GB_Global_print_format_set (0) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_select (C, NULL, NULL, selectop, A, NULL, NULL)) ;
    printf ("Error expected: %d\n%s\n", info, GrB_error ( )) ;

    expected = GrB_INVALID_VALUE ;
    OK (GxB_Scalar_clear (thunk)) ;
    ERR (GxB_Matrix_select (C, NULL, NULL, selectop, A, thunk, NULL)) ;
    printf ("Error expected: %d\n%s\n", info, GrB_error ( )) ;

    expected = GrB_DOMAIN_MISMATCH ;
    GxB_Scalar_free (&thunk) ;
    OK (GxB_Scalar_new (&thunk, GrB_FP32)) ;
    ERR (GxB_Matrix_select (C, NULL, NULL, selectop, A, thunk, NULL)) ;
    printf ("Error expected: %d\n%s\n", info, GrB_error ( )) ;

    GxB_SelectOp_free (&selectop) ;
    OK (GxB_SelectOp_new (&selectop, select_nothing, GrB_FP64, NULL)) ;
    ERR (GxB_Matrix_select (C, NULL, NULL, selectop, A, thunk, NULL)) ;
    printf ("Error expected: %d\n%s\n", info, GrB_error ( )) ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    OK (GrB_Type_new (&user_type, sizeof (user_int))) ;
    user_type->magic = 0xDEAD ;
    ERR (GxB_Type_fprint (user_type, "user_type", GxB_COMPLETE, NULL)) ;
    expected = GrB_INVALID_OBJECT ;
    selectop->ttype = user_type ;
    ERR (GxB_SelectOp_fprint (selectop, "selectop", GxB_COMPLETE, NULL)) ;
    user_type->magic = GB_MAGIC ;

    expected = GrB_UNINITIALIZED_OBJECT ;
    thunk->magic = 0xDEAD ;
    ERR (GxB_Scalar_fprint (thunk, "thunk", GxB_COMPLETE, NULL)) ;
    thunk->magic = GB_MAGIC ;
    printf ("Error expected: %d\n%s\n", info, GrB_error ( )) ;

    GrB_Type_free (&user_type) ;
    GrB_Matrix_free (&A) ;
    GrB_Matrix_free (&C) ;
    GxB_Scalar_free (&thunk) ;
    GxB_SelectOp_free (&selectop) ;

    //--------------------------------------------------------------------------
    // slice vector
    //--------------------------------------------------------------------------

    // GB_wait constructs a slice, Aslice [1], that is then added to the
    // pending tuples, B = T.  It then calls GB_add to compute Aslice [1] + T,
    // where Aslice [1] can either be hypersparse or a hyperslice.  Need to
    // trigger the condition that the index i appears after all entries in the
    // implicit hyperlist Ah.  It's hard to test this case directly, via
    // GB_wait and GB_add.

    int64_t i, pA = -1, pB = -1 ;
    int64_t Bh [10] ;
    for (int i = 0 ; i < 10 ; i++)
    {
        Bh [i] = 1000 + i ;
    }
    GB_slice_vector (&i, NULL, &pA, &pB,
        0, 0, NULL,     // Mi is empty
        0, 10, NULL, 1, // Ah is an implicit hyperlist: [1 2 3 4 5 6 7 8 9 10]
        0, 10, Bh,      // Bh is an explicit hyperlist
        2001,           // n
        (double) 10) ;  // target_work
    printf ("slice_vector: i "GBd" pA "GBd" pB "GBd"\n", i, pA, pB) ;
    OK (i == 1000) ;    // n is cut in half, i = floor ((0+(n-1))/2) 
    OK (pA == 10) ;     // first task does all of A
    OK (pB == 0) ;      // second task does all of B

    //--------------------------------------------------------------------------
    // GxB_Scalar
    //--------------------------------------------------------------------------

    GxB_Scalar scalar = NULL, scalar2 = NULL ;
    OK (GxB_Scalar_new (&scalar, GrB_FP64)) ;
    OK (GxB_Scalar_nvals (&nvals, scalar)) ;
    CHECK (nvals == 0) ;

    bool     b_8 = 0 ;
    int8_t   i_8 = 0 ;
    int16_t  i_16 = 0 ;
    int32_t  i_32 = 0 ;
    int64_t  i_64 = 0 ;
    uint8_t  u_8 = 0 ;
    uint16_t u_16 = 0 ;
    uint32_t u_32 = 0 ;
    uint64_t u_64 = 0 ;
    float    x_32 = 0 ;
    double   x_64 = 0 ;

    OK (GxB_Scalar_setElement_FP64 (scalar, (double) 1.25)) ;
    OK (GxB_Scalar_nvals (&nvals, scalar)) ;
    CHECK (nvals == 1) ;

    OK (GxB_Scalar_dup (&scalar2, scalar)) ;
    OK (GxB_Scalar_fprint (scalar2, "scalar2", GxB_COMPLETE, NULL)) ;

    OK (GxB_Scalar_extractElement_BOOL (&b_8,  scalar)) ; CHECK (b_8 == 1) ;

    OK (GxB_Scalar_extractElement_INT8  (&i_8,  scalar)) ; CHECK (i_8  == 1) ;
    OK (GxB_Scalar_extractElement_INT16 (&i_16, scalar)) ; CHECK (i_16 == 1) ;
    OK (GxB_Scalar_extractElement_INT32 (&i_32, scalar)) ; CHECK (i_32 == 1) ;
    OK (GxB_Scalar_extractElement_INT64 (&i_64, scalar)) ; CHECK (i_64 == 1) ;

    OK (GxB_Scalar_extractElement_UINT8  (&u_8,  scalar)) ; CHECK (u_8  == 1) ;
    OK (GxB_Scalar_extractElement_UINT16 (&u_16, scalar)) ; CHECK (u_16 == 1) ;
    OK (GxB_Scalar_extractElement_UINT32 (&u_32, scalar)) ; CHECK (u_32 == 1) ;
    OK (GxB_Scalar_extractElement_UINT64 (&u_64, scalar)) ; CHECK (u_64 == 1) ;

    OK (GxB_Scalar_extractElement_FP32 (&x_32, scalar)) ; CHECK (x_32 == 1.25) ;
    OK (GxB_Scalar_extractElement_FP64 (&x_64, scalar)) ; CHECK (x_64 == 1.25) ;

    OK (GxB_Scalar_clear (scalar)) ;
    info = GxB_Scalar_extractElement_FP64 (&x_64, scalar) ;
    CHECK (info == GrB_NO_VALUE) ;
    CHECK (x_64 == 1.25) ;

    u_64 = 0 ;
    nvals = 0 ;
    OK (GxB_Scalar_extractElement_UINT64 (&u_64, scalar2)) ; CHECK (u_64 == 1) ;
    OK (GxB_Scalar_nvals (&nvals, scalar2)) ;
    CHECK (nvals == 1) ;

    expected = GrB_INVALID_OBJECT ;
    scalar2->vlen = 2 ;
    ERR (GxB_Scalar_fprint (scalar2, "scalar2", GxB_COMPLETE, NULL)) ;
    scalar2->vlen = 1 ;
    OK (GxB_Scalar_fprint (scalar2, "scalar2", GxB_COMPLETE, NULL)) ;

    GxB_Scalar_free (&scalar) ;
    GxB_Scalar_free (&scalar2) ;

    //--------------------------------------------------------------------------
    // predefined descriptors
    //--------------------------------------------------------------------------

    OK (GxB_Descriptor_fprint (GrB_DESC_T1     , "T1    ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_T0     , "T0    ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_T0T1   , "T0T1  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_C      , "C     ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_CT1    , "CT1   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_CT0    , "CT0   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_CT0T1  , "CT0T1 ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_S      , "S     ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_ST1    , "ST1   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_ST0    , "ST0   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_ST0T1  , "ST0T1 ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_SC     , "SC    ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_SCT1   , "SCT1  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_SCT0   , "SCT0  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_SCT0T1 , "SCT0T1", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_R      , "R     ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RT1    , "RT1   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RT0    , "RT0   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RT0T1  , "RT0T1 ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RC     , "RC    ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RCT1   , "RCT1  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RCT0   , "RCT0  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RCT0T1 , "RCT0T1", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RS     , "RS    ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RST1   , "RST1  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RST0   , "RST0  ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RST0T1 , "RST0T1", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RSC    , "RSC   ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RSCT1  , "RSCT1 ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RSCT0  , "RSCT0 ", GxB_COMPLETE, NULL));
    OK (GxB_Descriptor_fprint (GrB_DESC_RSCT0T1, "RSCT0T1",GxB_COMPLETE, NULL));

    GrB_Descriptor_new (&Duh) ;
    OK (GxB_Desc_set (Duh, GxB_AxB_METHOD, GxB_AxB_SAXPY)) ;
    OK (GxB_Descriptor_fprint (Duh, "Duh", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_set (Duh, GxB_AxB_METHOD, GxB_AxB_HASH)) ;
    OK (GxB_Descriptor_fprint (Duh, "Duh", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_set (Duh, GxB_AxB_METHOD, GxB_AxB_HEAP)) ;
    OK (GxB_Descriptor_fprint (Duh, "Duh", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_set (Duh, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON)) ;
    OK (GxB_Descriptor_fprint (Duh, "Duh", GxB_COMPLETE, NULL)) ;
    OK (GxB_Desc_set (Duh, GxB_AxB_METHOD, GxB_AxB_DOT)) ;
    OK (GxB_Descriptor_fprint (Duh, "Duh", GxB_COMPLETE, NULL)) ;
    GrB_Descriptor_free (&Duh) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Desc_set (GrB_DESC_S, GrB_INP0, GrB_TRAN)) ;
    printf ("\nExpected error: %s\n", GrB_error ( )) ;

    ERR (GrB_Descriptor_set (GrB_DESC_S, GrB_INP0, GrB_TRAN)) ;
    printf ("\nExpected error: %s\n", GrB_error ( )) ;

    //--------------------------------------------------------------------------
    // burble
    //--------------------------------------------------------------------------

    bool burble ;
    OK (GxB_Global_Option_get (GxB_BURBLE, &burble)) ;
    printf ("burble: %d\n", burble) ;

    //--------------------------------------------------------------------------
    // select ops
    //--------------------------------------------------------------------------

    OK (GxB_SelectOp_fprint (GxB_TRIL,     "tril"    ,  GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_TRIU,     "triu"    ,  GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_DIAG,     "diag"    ,  GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_OFFDIAG,  "offidiag", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_NONZERO,  "nonzero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_EQ_ZERO,  "eq_zero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_GT_ZERO,  "gt_zero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_GE_ZERO,  "ge_zero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_LT_ZERO,  "lt_zero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_LE_ZERO,  "le_zero" , GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_NE_THUNK, "ne_thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_EQ_THUNK, "eq_thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_GT_THUNK, "gt_thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_GE_THUNK, "ge_thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_LT_THUNK, "lt_thunk", GxB_COMPLETE, NULL)) ;
    OK (GxB_SelectOp_fprint (GxB_LE_THUNK, "le_thunk", GxB_COMPLETE, NULL)) ;

    //--------------------------------------------------------------------------
    // assign scalar into hypersparse
    //--------------------------------------------------------------------------

    GrB_Index n = INT32_MAX ;
    n = n * 1024 ;
    OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;
    expected = GrB_OUT_OF_MEMORY ;
    ERR (GrB_Matrix_assign_FP64 (A, NULL, NULL, (double) 1,
        GrB_ALL, n, GrB_ALL, n, NULL)) ;
    printf ("\nproblem too large, expected error: %s\n", GrB_error ( )) ;
    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true, 0) ;
    fclose (f) ;
    printf ("\nAll errors printed above were expected.\n") ;
    printf ("GB_mex_about: all tests passed\n\n") ;
}

