//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Source/all_user_objects.c
//------------------------------------------------------------------------------

// This file is constructed automatically by cmake and m4 when GraphBLAS is
// compiled, from the Config/user_def*.m4 and *.m4 files in User/.  Do not edit
// this file directly.  It contains references to internally-defined functions
// and objects inside GraphBLAS, which are not user-callable.

#include "GB.h"

//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def1.m4: define user-defined objects
//------------------------------------------------------------------------------


















//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_band.m4: example user built-in objects
//------------------------------------------------------------------------------

// user-defined functions for GxB_select, to choose entries within a band

#ifdef GxB_USER_INCLUDE

    #define MY_BAND

    static inline bool myband (GrB_Index i, GrB_Index j, GrB_Index nrows,
        GrB_Index ncols, const void *x, const void *thunk)
    {
        int64_t *lohi = (int64_t *) thunk ;
        int64_t i2 = (int64_t) i ;
        int64_t j2 = (int64_t) j ;
        return ((lohi [0] <= (j2-i2)) && ((j2-i2) <= lohi [1])) ;
    }

#endif

// Select operator to compute C = tril (triu (A, k1), k2)

    #define GB_DEF_My_band_function myband
    extern bool myband
    (
        GrB_Index i,
        GrB_Index j,
        GrB_Index nrows,
        GrB_Index ncols,
        const void *x,
        const void *thunk
    ) ;
    struct GB_SelectOp_opaque GB_opaque_My_band =
    {
        GB_MAGIC,           // object is defined
        NULL,  // x not used
        myband,                 // pointer to the C function
        "myband",
        GB_USER_SELECT_C_opcode // user-defined at compile-time
    } ;
    GxB_SelectOp My_band = & GB_opaque_My_band ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_complex.m4: example user built-in objects
//------------------------------------------------------------------------------

// user-defined functions for a double complex type

#ifdef GxB_USER_INCLUDE

    // Get the complex.h definitions, but remove "I" since it is used elsewhere
    // in GraphBLAS.
    #include <complex.h>
    #undef I

    // Not all complex.h definitions include the CMPLX macro
    #ifndef CMPLX
    #define CMPLX(real,imag) \
        ( \
        (double complex)((double)(real)) + \
        (double complex)((double)(imag) * _Complex_I) \
        )
    #endif

    // define a token so a user application can check for existence 
    #define MY_COMPLEX

    static inline void my_complex_plus
    (
        double complex *z,
        const double complex *x,
        const double complex *y
    )
    {
        (*z) = (*x) + (*y) ;
    }

    static inline void my_complex_times
    (
        double complex *z,
        const double complex *x,
        const double complex *y
    )
    {
        (*z) = (*x) * (*y) ;
    }

#endif

// GraphBLAS does not have a complex type; this defines one:

    #define GB_DEF_My_Complex_type double complex
    struct GB_Type_opaque GB_opaque_My_Complex =
    {
        GB_MAGIC,           // object is defined
        sizeof (double complex),        // size of the type
        GB_UCT_code,        // user-defined at compile-time
        "double complex"
    } ;
    GrB_Type My_Complex = & GB_opaque_My_Complex ;

// The two operators, complex add and multiply:

    #define GB_DEF_My_Complex_plus_function my_complex_plus
    #define GB_DEF_My_Complex_plus_ztype GB_DEF_My_Complex_type
    #define GB_DEF_My_Complex_plus_xtype GB_DEF_My_Complex_type
    #define GB_DEF_My_Complex_plus_ytype GB_DEF_My_Complex_type
    extern void my_complex_plus
    (
        GB_DEF_My_Complex_plus_ztype *z,
        const GB_DEF_My_Complex_plus_xtype *x,
        const GB_DEF_My_Complex_plus_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_My_Complex_plus =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_My_Complex,     // type of x
        & GB_opaque_My_Complex,     // type of y
        & GB_opaque_My_Complex,     // type of z
        my_complex_plus,                 // pointer to the C function
        "my_complex_plus",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp My_Complex_plus = & GB_opaque_My_Complex_plus ;


    #define GB_DEF_My_Complex_times_function my_complex_times
    #define GB_DEF_My_Complex_times_ztype GB_DEF_My_Complex_type
    #define GB_DEF_My_Complex_times_xtype GB_DEF_My_Complex_type
    #define GB_DEF_My_Complex_times_ytype GB_DEF_My_Complex_type
    extern void my_complex_times
    (
        GB_DEF_My_Complex_times_ztype *z,
        const GB_DEF_My_Complex_times_xtype *x,
        const GB_DEF_My_Complex_times_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_My_Complex_times =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_My_Complex,     // type of x
        & GB_opaque_My_Complex,     // type of y
        & GB_opaque_My_Complex,     // type of z
        my_complex_times,                 // pointer to the C function
        "my_complex_times",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp My_Complex_times = & GB_opaque_My_Complex_times ;

// The plus monoid:

    #define GB_DEF_My_Complex_plus_monoid_add GB_DEF_My_Complex_plus_function
    #define GB_DEF_My_Complex_plus_monoid_zsize sizeof (GB_DEF_My_Complex_plus_ztype)
    GB_DEF_My_Complex_plus_ztype GB_DEF_My_Complex_plus_monoid_identity = CMPLX(0,0) ;
    struct GB_Monoid_opaque GB_opaque_My_Complex_plus_monoid =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_My_Complex_plus,     // binary operator
        & GB_DEF_My_Complex_plus_monoid_identity,   // identity value
        GB_DEF_My_Complex_plus_monoid_zsize,    // identity size
        GB_USER_COMPILED,   // user-defined at compile-time
        NULL                // no terminal value
    } ;
    GrB_Monoid My_Complex_plus_monoid = & GB_opaque_My_Complex_plus_monoid ;

// the conventional plus-times semiring for C=A*B for the complex case
 
    #define GB_AgusB    GB_AxB_user_gus_My_Complex_plus_times
    #define GB_AdotB    GB_AxB_user_dot_My_Complex_plus_times
    #define GB_AheapB   GB_AxB_user_heap_My_Complex_plus_times
    #define GB_identity    GB_DEF_My_Complex_plus_monoid_identity
    #define GB_ADD(z,y)    GB_DEF_My_Complex_plus_monoid_add (&(z), &(z), &(y))
    #if defined ( GB_DEF_My_Complex_plus_monoid_is_user_terminal )
        #define GB_terminal if (memcmp (&cij, &GB_DEF_My_Complex_plus_monoid_user_terminal, GB_DEF_My_Complex_plus_monoid_zsize) == 0) break ;
    #elif defined ( GB_DEF_My_Complex_plus_monoid_terminal )
        #define GB_terminal if (cij == GB_DEF_My_Complex_plus_monoid_terminal) break ;
    #else
        #define GB_terminal ;
    #endif
    #define GB_MULT(z,x,y) GB_DEF_My_Complex_times_function (&(z), &(x), &(y))
    #define GB_ztype       GB_DEF_My_Complex_times_ztype
    #define GB_xtype       GB_DEF_My_Complex_times_xtype
    #define GB_ytype       GB_DEF_My_Complex_times_ytype
    #define GB_handle_flipxy 1
    #undef GBCOMPACT
    #include "GB_AxB.c"
    #undef GB_identity
    #undef GB_terminal
    #undef GB_ADD
    #undef GB_xtype
    #undef GB_ytype
    #undef GB_ztype
    #undef GB_MULT
    #undef GB_AgusB
    #undef GB_AdotB
    #undef GB_AheapB
    struct GB_Semiring_opaque GB_opaque_My_Complex_plus_times =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_My_Complex_plus_monoid,     // add monoid
        & GB_opaque_My_Complex_times,     // multiply operator
        GB_USER_COMPILED    // user-defined at compile-time
    } ;
    GrB_Semiring My_Complex_plus_times = & GB_opaque_My_Complex_plus_times ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_max.m4: example user built-in objects
//------------------------------------------------------------------------------

// user-defined MAX functions for GxB_Monoid_terminal_new, to choose a
// non-default terminal value

#ifdef GxB_USER_INCLUDE

    #define MY_MAX

    static inline void my_maxdouble
    (
        double *z,
        const double *x,
        const double *y
    )
    {
        // this is not safe with NaNs
        (*z) = ((*x) > (*y)) ? (*x) : (*y) ;
    }

#endif

// max operator

    #define GB_DEF_My_Max_function my_maxdouble
    #define GB_DEF_My_Max_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_Max_xtype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_Max_ytype GB_DEF_GrB_FP64_type
    extern void my_maxdouble
    (
        GB_DEF_My_Max_ztype *z,
        const GB_DEF_My_Max_xtype *x,
        const GB_DEF_My_Max_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_My_Max =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_GrB_FP64,     // type of y
        & GB_opaque_GrB_FP64,     // type of z
        my_maxdouble,                 // pointer to the C function
        "my_maxdouble",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp My_Max = & GB_opaque_My_Max ;

// The max monoid, with terminal value of 1

    #define GB_DEF_My_Max_Terminal1_add GB_DEF_My_Max_function
    #define GB_DEF_My_Max_Terminal1_zsize sizeof (GB_DEF_My_Max_ztype)
    #define GB_DEF_My_Max_Terminal1_is_user_terminal
    GB_DEF_My_Max_ztype GB_DEF_My_Max_Terminal1_identity = (-INFINITY) ;
    GB_DEF_My_Max_ztype GB_DEF_My_Max_Terminal1_user_terminal = 1 ;
    struct GB_Monoid_opaque GB_opaque_My_Max_Terminal1 =
    {
        GB_MAGIC,                   // object is defined
        & GB_opaque_My_Max,             // binary operator
        & GB_DEF_My_Max_Terminal1_identity,       // identity value
        GB_DEF_My_Max_Terminal1_zsize,            // identity and terminal size
        GB_USER_COMPILED,           // user-defined at compile-time
        & GB_DEF_My_Max_Terminal1_user_terminal   // terminal value
    } ;
    GrB_Monoid My_Max_Terminal1 = & GB_opaque_My_Max_Terminal1 ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_pagerank.m4: PageRank semiring
//------------------------------------------------------------------------------

// Defines a PageRank type, operators, monoid, and semiring for the method in
// Demo/Source/dpagerank2.c.

#ifdef GxB_USER_INCLUDE

// Define a token that dpagerank2.c can use to determine if these definitions
// are available at compile-time.
#define PAGERANK_PREDEFINED

// each node has a rank value, and a constant which is 1/outdegree
typedef struct
{
    double rank ;
    double invdegree ;
}
pagerank_type ;

// global variable declarations
extern
double pagerank_damping, pagerank_teleport, pagerank_rdiff,
    pagerank_init_rank, pagerank_rsum ;

// for thread safety if the user application uses OpenMP, with parallel calls
// to dpagerank2 on independent problems.
#pragma omp threadprivate(pagerank_damping, pagerank_teleport, pagerank_rdiff, pagerank_init_rank, pagerank_rsum)

// The identity value for the pagerank_add monoid is {0,0}. For the
// GxB_*_define macro that defines the GrB_Monoid, the identity argument must
// be a compile-time constant (for the C definition), and it must also be
// parsable as an argument to the m4 macro.  If the user-defined type is a
// struct, the initializer uses curly brackets, but this causes a parsing error
// for m4.  The solution is to define a C macro with the initialization
// constant, and to use it in the GxB*define m4 macro.
#define PAGERANK_ZERO {0,0}

// unary operator to divide a double entry by the scalar pagerank_rsum
static inline
void pagerank_div (double *z, const double *x)
{
    (*z) = (*x) / pagerank_rsum ;
}

// unary operator that typecasts PageRank_type to double, extracting the rank
static inline
void pagerank_get_rank (double *z, const pagerank_type *x)
{
    (*z) = (x->rank) ;
}

// unary operator to initialize a node
static inline
void init_page (pagerank_type *z, const double *x)
{
    z->rank = pagerank_init_rank ;  // all nodes start with rank 1/n
    z->invdegree = 1. / (*x) ;      // set 1/outdegree of this node 
}

//------------------------------------------------------------------------------
// PageRank semiring
//------------------------------------------------------------------------------

// In MATLAB notation, the new rank is computed with:
// newrank = pagerank_damping * (rank * D * A) + pagerank_teleport

// where A is a square binary matrix of the original graph, and A(i,j)=1 if
// page i has a link to page j.  rank is a row vector of size n.  The matrix D
// is diagonal, with D(i,i)=1/outdegree(i), where outdegree(i) = the outdegree
// of node i, or equivalently, outdegree(i) = sum (A (i,:)).

// That is, if newrank(j) were computed with a dot product:
//      newrank (j) = 0
//      for all i:
//          newrank (j) = newrank (j) + (rank (i) * D (i,i)) * A (i,j)

// To accomplish this computation in a single vector-matrix multiply, the value
// of D(i,i) is held as component of a combined data type, the pagerank_type,
// which has both the rank(i) and the entry D(i,i).

// binary multiplicative operator for the pagerank semiring
static inline
void pagerank_multiply
(
    pagerank_type *z,
    const pagerank_type *x,
    const bool *y
)
{
    // y is the boolean entry of the matrix, A(i,j)
    // x->rank is the rank of node i, and x->invdegree is 1/outdegree(i)
    // note that z->invdegree is left unchanged
    z->rank = (*y) ? ((x->rank) * (x->invdegree)) : 0 ;
}

// binary additive operator for the pagerank semiring
static inline
void pagerank_add
(
    pagerank_type *z,
    const pagerank_type *x,
    const pagerank_type *y
)
{
    // note that z->invdegree is left unchanged; it is unused
    z->rank = (x->rank) + (y->rank) ;
}

//------------------------------------------------------------------------------
// pagerank accumulator
//------------------------------------------------------------------------------

// The semiring computes the vector newrank = rank*D*A.  To complete the page
// rank computation, the new rank must be scaled by the
// pagerank_damping, and the pagerank_teleport must be included, which is
// done in the page rank accumulator:

// newrank = pagerank_damping * newrank + pagerank_teleport

// The PageRank_semiring does not construct the entire pagerank_type of
// rank*D*A, since the vector that holds newrank(i) must also keep the
// 1/invdegree(i), unchanged.  This is restored in the accumulator operator.

// The PageRank_accum operator can also compute pagerank_rdiff = norm (r-rnew),
// as a side effect.  This is unsafe but faster (see the comments below);
// uncomment the following #define to enable the unsafe method, or comment it
// out to use the safe method:
//
    #define PAGERANK_UNSAFE

// binary operator to accumulate the new rank from the old
static inline
void pagerank_accum
(
    pagerank_type *z,
    const pagerank_type *x,
    const pagerank_type *y
)
{
    // note that this formula does not use the old rank:
    // new rank = pagerank_damping * (rank*A ) + pagerank_teleport
    double rnew = pagerank_damping * (y->rank) + pagerank_teleport ;

    #ifdef PAGERANK_UNSAFE

    // This computation of pagerank_rdiff is not guaranteed to work per the
    // GraphBLAS spec, but it does work with the current implementation of
    // SuiteSparse:GraphBLAS.  The reason is that there is no guarantee that
    // the accumulator step of a GraphBLAS operation is computed sequentially.
    // If computed in parallel, a race condition would occur.

    // This step uses the old rank, to compute the stopping criterion:
    // pagerank_rdiff = sum (ranknew - rankold)
    double delta = rnew - (x->rank) ;
    pagerank_rdiff += delta * delta ;

    #endif

    // update the rank, and copy over the inverse degree from the old page info
    z->rank = rnew ;
    z->invdegree = x->invdegree ;
}

//------------------------------------------------------------------------------
// pagerank_diff: compute the change in the rank
//------------------------------------------------------------------------------

// This is safer than computing pagerank_rdiff via pagerank_accum, and is
// compliant with the GraphBLAS spec.

static inline
void pagerank_diff
(
    pagerank_type *z,
    const pagerank_type *x,
    const pagerank_type *y
)
{
    double delta = (x->rank) - (y->rank) ;
    z->rank = delta * delta ;
}

#else

// global variable definitions
double pagerank_damping, pagerank_teleport, pagerank_rdiff,
    pagerank_init_rank, pagerank_rsum ;

#endif

// create the new Page type

    #define GB_DEF_PageRank_type_type pagerank_type
    struct GB_Type_opaque GB_opaque_PageRank_type =
    {
        GB_MAGIC,           // object is defined
        sizeof (pagerank_type),        // size of the type
        GB_UCT_code,        // user-defined at compile-time
        "pagerank_type"
    } ;
    GrB_Type PageRank_type = & GB_opaque_PageRank_type ;

// create the unary operator to initialize the PageRank_type of each node

    #define GB_DEF_PageRank_init_function init_page
    #define GB_DEF_PageRank_init_ztype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_init_xtype GB_DEF_GrB_FP64_type
    extern void init_page
    (
        GB_DEF_PageRank_init_ztype *z,
        const GB_DEF_PageRank_init_xtype *x
    ) ;
    struct GB_UnaryOp_opaque GB_opaque_PageRank_init =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_PageRank_type,     // type of z
        init_page,                 // pointer to the C function
        "init_page",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_UnaryOp PageRank_init = & GB_opaque_PageRank_init ;

// create PageRank_accum

    #define GB_DEF_PageRank_accum_function pagerank_accum
    #define GB_DEF_PageRank_accum_ztype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_accum_xtype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_accum_ytype GB_DEF_PageRank_type_type
    extern void pagerank_accum
    (
        GB_DEF_PageRank_accum_ztype *z,
        const GB_DEF_PageRank_accum_xtype *x,
        const GB_DEF_PageRank_accum_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_PageRank_accum =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_type,     // type of x
        & GB_opaque_PageRank_type,     // type of y
        & GB_opaque_PageRank_type,     // type of z
        pagerank_accum,                 // pointer to the C function
        "pagerank_accum",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp PageRank_accum = & GB_opaque_PageRank_accum ;

// create PageRank_add operator and monoid

    #define GB_DEF_PageRank_add_function pagerank_add
    #define GB_DEF_PageRank_add_ztype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_add_xtype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_add_ytype GB_DEF_PageRank_type_type
    extern void pagerank_add
    (
        GB_DEF_PageRank_add_ztype *z,
        const GB_DEF_PageRank_add_xtype *x,
        const GB_DEF_PageRank_add_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_PageRank_add =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_type,     // type of x
        & GB_opaque_PageRank_type,     // type of y
        & GB_opaque_PageRank_type,     // type of z
        pagerank_add,                 // pointer to the C function
        "pagerank_add",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp PageRank_add = & GB_opaque_PageRank_add ;

// create PageRank_monoid.  See the discussion above for PAGERANK_ZERO.

    #define GB_DEF_PageRank_monoid_add GB_DEF_PageRank_add_function
    #define GB_DEF_PageRank_monoid_zsize sizeof (GB_DEF_PageRank_add_ztype)
    GB_DEF_PageRank_add_ztype GB_DEF_PageRank_monoid_identity = PAGERANK_ZERO ;
    struct GB_Monoid_opaque GB_opaque_PageRank_monoid =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_add,     // binary operator
        & GB_DEF_PageRank_monoid_identity,   // identity value
        GB_DEF_PageRank_monoid_zsize,    // identity size
        GB_USER_COMPILED,   // user-defined at compile-time
        NULL                // no terminal value
    } ;
    GrB_Monoid PageRank_monoid = & GB_opaque_PageRank_monoid ;

// create PageRank_multiply operator

    #define GB_DEF_PageRank_multiply_function pagerank_multiply
    #define GB_DEF_PageRank_multiply_ztype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_multiply_xtype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_multiply_ytype GB_DEF_GrB_BOOL_type
    extern void pagerank_multiply
    (
        GB_DEF_PageRank_multiply_ztype *z,
        const GB_DEF_PageRank_multiply_xtype *x,
        const GB_DEF_PageRank_multiply_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_PageRank_multiply =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_type,     // type of x
        & GB_opaque_GrB_BOOL,     // type of y
        & GB_opaque_PageRank_type,     // type of z
        pagerank_multiply,                 // pointer to the C function
        "pagerank_multiply",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp PageRank_multiply = & GB_opaque_PageRank_multiply ;

// create PageRank_semiring
 
    #define GB_AgusB    GB_AxB_user_gus_PageRank_semiring
    #define GB_AdotB    GB_AxB_user_dot_PageRank_semiring
    #define GB_AheapB   GB_AxB_user_heap_PageRank_semiring
    #define GB_identity    GB_DEF_PageRank_monoid_identity
    #define GB_ADD(z,y)    GB_DEF_PageRank_monoid_add (&(z), &(z), &(y))
    #if defined ( GB_DEF_PageRank_monoid_is_user_terminal )
        #define GB_terminal if (memcmp (&cij, &GB_DEF_PageRank_monoid_user_terminal, GB_DEF_PageRank_monoid_zsize) == 0) break ;
    #elif defined ( GB_DEF_PageRank_monoid_terminal )
        #define GB_terminal if (cij == GB_DEF_PageRank_monoid_terminal) break ;
    #else
        #define GB_terminal ;
    #endif
    #define GB_MULT(z,x,y) GB_DEF_PageRank_multiply_function (&(z), &(x), &(y))
    #define GB_ztype       GB_DEF_PageRank_multiply_ztype
    #define GB_xtype       GB_DEF_PageRank_multiply_xtype
    #define GB_ytype       GB_DEF_PageRank_multiply_ytype
    #define GB_handle_flipxy 1
    #undef GBCOMPACT
    #include "GB_AxB.c"
    #undef GB_identity
    #undef GB_terminal
    #undef GB_ADD
    #undef GB_xtype
    #undef GB_ytype
    #undef GB_ztype
    #undef GB_MULT
    #undef GB_AgusB
    #undef GB_AdotB
    #undef GB_AheapB
    struct GB_Semiring_opaque GB_opaque_PageRank_semiring =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_monoid,     // add monoid
        & GB_opaque_PageRank_multiply,     // multiply operator
        GB_USER_COMPILED    // user-defined at compile-time
    } ;
    GrB_Semiring PageRank_semiring = & GB_opaque_PageRank_semiring ;

// create unary operator that typecasts the PageRank_type to double

    #define GB_DEF_PageRank_get_function pagerank_get_rank
    #define GB_DEF_PageRank_get_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_PageRank_get_xtype GB_DEF_PageRank_type_type
    extern void pagerank_get_rank
    (
        GB_DEF_PageRank_get_ztype *z,
        const GB_DEF_PageRank_get_xtype *x
    ) ;
    struct GB_UnaryOp_opaque GB_opaque_PageRank_get =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_type,     // type of x
        & GB_opaque_GrB_FP64,     // type of z
        pagerank_get_rank,                 // pointer to the C function
        "pagerank_get_rank",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_UnaryOp PageRank_get = & GB_opaque_PageRank_get ;

// create unary operator that scales the rank by pagerank_rsum

    #define GB_DEF_PageRank_div_function pagerank_div
    #define GB_DEF_PageRank_div_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_PageRank_div_xtype GB_DEF_GrB_FP64_type
    extern void pagerank_div
    (
        GB_DEF_PageRank_div_ztype *z,
        const GB_DEF_PageRank_div_xtype *x
    ) ;
    struct GB_UnaryOp_opaque GB_opaque_PageRank_div =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_GrB_FP64,     // type of z
        pagerank_div,                 // pointer to the C function
        "pagerank_div",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_UnaryOp PageRank_div = & GB_opaque_PageRank_div ;

// create PageRank_diff operator

    #define GB_DEF_PageRank_diff_function pagerank_diff
    #define GB_DEF_PageRank_diff_ztype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_diff_xtype GB_DEF_PageRank_type_type
    #define GB_DEF_PageRank_diff_ytype GB_DEF_PageRank_type_type
    extern void pagerank_diff
    (
        GB_DEF_PageRank_diff_ztype *z,
        const GB_DEF_PageRank_diff_xtype *x,
        const GB_DEF_PageRank_diff_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_PageRank_diff =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_PageRank_type,     // type of x
        & GB_opaque_PageRank_type,     // type of y
        & GB_opaque_PageRank_type,     // type of z
        pagerank_diff,                 // pointer to the C function
        "pagerank_diff",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp PageRank_diff = & GB_opaque_PageRank_diff ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_plus_rdiv.m4: example user built-in objects
//------------------------------------------------------------------------------

#ifdef GxB_USER_INCLUDE

#define MY_RDIV

    static inline void my_rdiv
    (
        double *z,
        const double *x,
        const double *y
    )
    {
        (*z) = (*y) / (*x) ;
    }

#endif

// rdiv operator

    #define GB_DEF_My_rdiv_function my_rdiv
    #define GB_DEF_My_rdiv_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_rdiv_xtype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_rdiv_ytype GB_DEF_GrB_FP64_type
    extern void my_rdiv
    (
        GB_DEF_My_rdiv_ztype *z,
        const GB_DEF_My_rdiv_xtype *x,
        const GB_DEF_My_rdiv_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_My_rdiv =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_GrB_FP64,     // type of y
        & GB_opaque_GrB_FP64,     // type of z
        my_rdiv,                 // pointer to the C function
        "my_rdiv",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp My_rdiv = & GB_opaque_My_rdiv ;

// plus-rdiv semiring
 
    #define GB_AgusB    GB_AxB_user_gus_My_plus_rdiv
    #define GB_AdotB    GB_AxB_user_dot_My_plus_rdiv
    #define GB_AheapB   GB_AxB_user_heap_My_plus_rdiv
    #define GB_identity    GB_DEF_GxB_PLUS_FP64_MONOID_identity
    #define GB_ADD(z,y)    GB_DEF_GxB_PLUS_FP64_MONOID_add (&(z), &(z), &(y))
    #if defined ( GB_DEF_GxB_PLUS_FP64_MONOID_is_user_terminal )
        #define GB_terminal if (memcmp (&cij, &GB_DEF_GxB_PLUS_FP64_MONOID_user_terminal, GB_DEF_GxB_PLUS_FP64_MONOID_zsize) == 0) break ;
    #elif defined ( GB_DEF_GxB_PLUS_FP64_MONOID_terminal )
        #define GB_terminal if (cij == GB_DEF_GxB_PLUS_FP64_MONOID_terminal) break ;
    #else
        #define GB_terminal ;
    #endif
    #define GB_MULT(z,x,y) GB_DEF_My_rdiv_function (&(z), &(x), &(y))
    #define GB_ztype       GB_DEF_My_rdiv_ztype
    #define GB_xtype       GB_DEF_My_rdiv_xtype
    #define GB_ytype       GB_DEF_My_rdiv_ytype
    #define GB_handle_flipxy 1
    #undef GBCOMPACT
    #include "GB_AxB.c"
    #undef GB_identity
    #undef GB_terminal
    #undef GB_ADD
    #undef GB_xtype
    #undef GB_ytype
    #undef GB_ztype
    #undef GB_MULT
    #undef GB_AgusB
    #undef GB_AdotB
    #undef GB_AheapB
    struct GB_Semiring_opaque GB_opaque_My_plus_rdiv =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GxB_PLUS_FP64_MONOID,     // add monoid
        & GB_opaque_My_rdiv,     // multiply operator
        GB_USER_COMPILED    // user-defined at compile-time
    } ;
    GrB_Semiring My_plus_rdiv = & GB_opaque_My_plus_rdiv ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_plus_rdiv2.m4: example user built-in objects
//------------------------------------------------------------------------------

// This version tests the case when the user-defined multiply operator
// has a different type for x and y.

#ifdef GxB_USER_INCLUDE

    #define MY_RDIV2

    static inline void my_rdiv2
    (
        double *z,
        const double *x,
        const float *y
    )
    {
        (*z) = (*y) / (*x) ;
    }

#endif

// rdiv2 operator

    #define GB_DEF_My_rdiv2_function my_rdiv2
    #define GB_DEF_My_rdiv2_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_rdiv2_xtype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_rdiv2_ytype GB_DEF_GrB_FP32_type
    extern void my_rdiv2
    (
        GB_DEF_My_rdiv2_ztype *z,
        const GB_DEF_My_rdiv2_xtype *x,
        const GB_DEF_My_rdiv2_ytype *y
    ) ;
    struct GB_BinaryOp_opaque GB_opaque_My_rdiv2 =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_GrB_FP32,     // type of y
        & GB_opaque_GrB_FP64,     // type of z
        my_rdiv2,                 // pointer to the C function
        "my_rdiv2",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_BinaryOp My_rdiv2 = & GB_opaque_My_rdiv2 ;

// plus-rdiv2 semiring
 
    #define GB_AgusB    GB_AxB_user_gus_My_plus_rdiv2
    #define GB_AdotB    GB_AxB_user_dot_My_plus_rdiv2
    #define GB_AheapB   GB_AxB_user_heap_My_plus_rdiv2
    #define GB_identity    GB_DEF_GxB_PLUS_FP64_MONOID_identity
    #define GB_ADD(z,y)    GB_DEF_GxB_PLUS_FP64_MONOID_add (&(z), &(z), &(y))
    #if defined ( GB_DEF_GxB_PLUS_FP64_MONOID_is_user_terminal )
        #define GB_terminal if (memcmp (&cij, &GB_DEF_GxB_PLUS_FP64_MONOID_user_terminal, GB_DEF_GxB_PLUS_FP64_MONOID_zsize) == 0) break ;
    #elif defined ( GB_DEF_GxB_PLUS_FP64_MONOID_terminal )
        #define GB_terminal if (cij == GB_DEF_GxB_PLUS_FP64_MONOID_terminal) break ;
    #else
        #define GB_terminal ;
    #endif
    #define GB_MULT(z,x,y) GB_DEF_My_rdiv2_function (&(z), &(x), &(y))
    #define GB_ztype       GB_DEF_My_rdiv2_ztype
    #define GB_xtype       GB_DEF_My_rdiv2_xtype
    #define GB_ytype       GB_DEF_My_rdiv2_ytype
    #define GB_handle_flipxy 1
    #undef GBCOMPACT
    #include "GB_AxB.c"
    #undef GB_identity
    #undef GB_terminal
    #undef GB_ADD
    #undef GB_xtype
    #undef GB_ytype
    #undef GB_ztype
    #undef GB_MULT
    #undef GB_AgusB
    #undef GB_AdotB
    #undef GB_AheapB
    struct GB_Semiring_opaque GB_opaque_My_plus_rdiv2 =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GxB_PLUS_FP64_MONOID,     // add monoid
        & GB_opaque_My_rdiv2,     // multiply operator
        GB_USER_COMPILED    // user-defined at compile-time
    } ;
    GrB_Semiring My_plus_rdiv2 = & GB_opaque_My_plus_rdiv2 ;

//------------------------------------------------------------------------------
// GraphBLAS/User/Example/my_scale.m4: example user built-in objects
//------------------------------------------------------------------------------

// user-defined unary operator: z = f(x) = my_scalar*x and its global scalar

#ifdef GxB_USER_INCLUDE

    //--------------------------------------------------------------------------
    // declarations: for GraphBLAS.h
    //--------------------------------------------------------------------------

    // The following are declarations that are enabled in GraphBLAS.h and
    // appear in all user codes that #include "GraphBLAS.h", and also in all
    // internal GraphBLAS codes.  All user declarations (not definitions)
    // should appear here.

    #define MY_SCALE

    extern double my_scalar ;
    // for thread safety if the user application uses OpenMP
    #pragma omp threadprivate(my_scalar)

    static inline void my_scale
    (
        double *z,
        const double *x
    )
    {
        (*z) = my_scalar * (*x) ;
    }

#else

    //--------------------------------------------------------------------------
    // definitions: code appears just once, in Source/all_user_objects.c
    //--------------------------------------------------------------------------

    // The following defintions are enabled in only a single place:
    // SuiteSparse/GraphBLAS/Source/all_user_objects.c.  This is the place
    // where all user-defined global variables should be defined.

    double my_scalar = 0 ;

#endif


//------------------------------------------------------------------------------
// define/declare the GrB_UnaryOp My_scale
//------------------------------------------------------------------------------

// Unary operator to compute z = my_scalar*x


    #define GB_DEF_My_scale_function my_scale
    #define GB_DEF_My_scale_ztype GB_DEF_GrB_FP64_type
    #define GB_DEF_My_scale_xtype GB_DEF_GrB_FP64_type
    extern void my_scale
    (
        GB_DEF_My_scale_ztype *z,
        const GB_DEF_My_scale_xtype *x
    ) ;
    struct GB_UnaryOp_opaque GB_opaque_My_scale =
    {
        GB_MAGIC,           // object is defined
        & GB_opaque_GrB_FP64,     // type of x
        & GB_opaque_GrB_FP64,     // type of z
        my_scale,                 // pointer to the C function
        "my_scale",
        GB_USER_C_opcode    // user-defined at compile-time
    } ;
    GrB_UnaryOp My_scale = & GB_opaque_My_scale ;

//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def2.m4: code to call user semirings
//------------------------------------------------------------------------------

GrB_Info GB_AxB_user
(
    const GrB_Desc_Value GB_AxB_method,
    const GrB_Semiring GB_s,

    GrB_Matrix *GB_Chandle,
    const GrB_Matrix GB_M,
    const GrB_Matrix GB_A,
    const GrB_Matrix GB_B,
    bool GB_flipxy,

    // for dot method only:
    const bool GB_mask_comp,

    // for heap method only:
    int64_t *restrict GB_List,
    GB_pointer_pair *restrict GB_pA_pair,
    GB_Element *restrict GB_Heap,
    const int64_t GB_bjnz_max,

    // for Gustavson method only:
    GB_Sauna GB_C_Sauna
)
{
    GrB_Info GB_info = GrB_SUCCESS ;
    if (0)
    {
        ;
    }
    else if (GB_s == My_Complex_plus_times)
    {
        if (GB_AxB_method == GxB_AxB_GUSTAVSON)
        { 
            GB_info = GB_AxB_user_gus_My_Complex_plus_times
                (*GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy, GB_C_Sauna) ;
        }
        else if (GB_AxB_method == GxB_AxB_DOT)
        { 
            GB_info = GB_AxB_user_dot_My_Complex_plus_times
                (GB_Chandle, GB_M, GB_mask_comp, GB_A, GB_B, GB_flipxy) ;
        }
        else // (GB_AxB_method == GxB_AxB_HEAP)
        { 
            GB_info = GB_AxB_user_heap_My_Complex_plus_times
                (GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy,
                GB_List, GB_pA_pair, GB_Heap, GB_bjnz_max) ;
        }
    } 
    else if (GB_s == PageRank_semiring)
    {
        if (GB_AxB_method == GxB_AxB_GUSTAVSON)
        { 
            GB_info = GB_AxB_user_gus_PageRank_semiring
                (*GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy, GB_C_Sauna) ;
        }
        else if (GB_AxB_method == GxB_AxB_DOT)
        { 
            GB_info = GB_AxB_user_dot_PageRank_semiring
                (GB_Chandle, GB_M, GB_mask_comp, GB_A, GB_B, GB_flipxy) ;
        }
        else // (GB_AxB_method == GxB_AxB_HEAP)
        { 
            GB_info = GB_AxB_user_heap_PageRank_semiring
                (GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy,
                GB_List, GB_pA_pair, GB_Heap, GB_bjnz_max) ;
        }
    } 
    else if (GB_s == My_plus_rdiv)
    {
        if (GB_AxB_method == GxB_AxB_GUSTAVSON)
        { 
            GB_info = GB_AxB_user_gus_My_plus_rdiv
                (*GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy, GB_C_Sauna) ;
        }
        else if (GB_AxB_method == GxB_AxB_DOT)
        { 
            GB_info = GB_AxB_user_dot_My_plus_rdiv
                (GB_Chandle, GB_M, GB_mask_comp, GB_A, GB_B, GB_flipxy) ;
        }
        else // (GB_AxB_method == GxB_AxB_HEAP)
        { 
            GB_info = GB_AxB_user_heap_My_plus_rdiv
                (GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy,
                GB_List, GB_pA_pair, GB_Heap, GB_bjnz_max) ;
        }
    } 
    else if (GB_s == My_plus_rdiv2)
    {
        if (GB_AxB_method == GxB_AxB_GUSTAVSON)
        { 
            GB_info = GB_AxB_user_gus_My_plus_rdiv2
                (*GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy, GB_C_Sauna) ;
        }
        else if (GB_AxB_method == GxB_AxB_DOT)
        { 
            GB_info = GB_AxB_user_dot_My_plus_rdiv2
                (GB_Chandle, GB_M, GB_mask_comp, GB_A, GB_B, GB_flipxy) ;
        }
        else // (GB_AxB_method == GxB_AxB_HEAP)
        { 
            GB_info = GB_AxB_user_heap_My_plus_rdiv2
                (GB_Chandle, GB_M, GB_A, GB_B, GB_flipxy,
                GB_List, GB_pA_pair, GB_Heap, GB_bjnz_max) ;
        }
    } 
    return (GB_info) ;
}

