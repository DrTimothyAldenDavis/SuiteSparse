//------------------------------------------------------------------------------
// GB_mex_test53: integer hints with zombies, pending tuples, and hyper_hash
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Changing GxB_ROWINDEX_INTEGER_HINT, GxB_COLINDEX_INTEGER_HINT, or
// GxB_OFFSET_INTEGER_HINT on a matrix converts its integer arrays at once
// (GB_convert_int), without finishing any pending work first.

// This test changes each hint, on sparse and hypersparse matrices, held by row
// and by column, and on vectors, while the matrix has zombies, pending tuples,
// both, or neither.  The result is checked against the same matrix whose
// pending work was finished before the hint was changed.

// Regression test for a bug in v10.0.0 to v10.5.x.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_Matrix_free (&A) ;          \
    GrB_Matrix_free (&B) ;          \
    GrB_Matrix_free (&E) ;          \
    GrB_Matrix_free (&F) ;          \
    GrB_Vector_free (&v) ;          \
    GrB_Vector_free (&e) ;          \
}

// An N-by-N random matrix with NZ entries is spread out with a stride of
// STRIDE, so that only every STRIDE-th vector of the matrix is non-empty.  It
// still has more than 1024 non-empty vectors, so that a hypersparse matrix
// gets a hyper_hash (A->Y).
#define N  4000
#define NZ 20000
#define STRIDE 10
// a smaller random matrix is assigned into A(0:NS-1,0:NS-1)
#define NS  1000
#define NZS 2000
// entries with values less than CUTOFF are deleted.  GB_mx_random_matrix
// creates FP64 matrices with values uniformly distributed in [0,1], so this
// deletes about 10% of the entries.
#define CUTOFF 0.10

//------------------------------------------------------------------------------
// check_state: assert the sparsity and pending work of a matrix
//------------------------------------------------------------------------------

static void check_state
(
    GrB_Matrix A,
    int sparsity,       // expected sparsity: GxB_SPARSE or GxB_HYPERSPARSE
    bool zombies,       // true if A must have zombies, false if it must not
    bool pending,       // true if A must have pending tuples, false if not
    bool hyper_hash     // true if A must have a hyper_hash A->Y, false if not
)
{
    GrB_Info info ;
    int32_t s ;
    OK (GrB_Matrix_get_INT32 (A, &s, GxB_SPARSITY_STATUS)) ;
    CHECK (s == sparsity) ;
    CHECK (zombies == (A->nzombies > 0)) ;
    CHECK (pending == (A->Pending != NULL && A->Pending->n > 0)) ;
    CHECK (hyper_hash == (A->Y != NULL)) ;
}

//------------------------------------------------------------------------------
// make_matrix: create a matrix, edit it, and change one integer hint
//------------------------------------------------------------------------------

// A is a random matrix, held by row or column, sparse or hypersparse, with all
// integers of size "from".  A random N-by-N matrix R is assigned into A with a
// stride of STRIDE in the vector dimension: A(0:STRIDE:end,:) = R if A is held
// by row, or A(:,0:STRIDE:end) = R if held by column, so that most vectors of
// A are empty.  If zombies is true, the entries of A with
// values less than CUTOFF are deleted (M = select (A < CUTOFF) followed by
// A<M,struct> = empty scalar), which leaves them as zombies.  GrB_select
// itself does not leave zombies: it always builds a new result.  If pending is
// true, A(0:NS-1,0:NS-1) += S is done with a smaller random matrix S, which
// adds the entries of S not already in A as pending tuples.  If wait_first is
// true, the pending work is finished before the hint is changed to "to".  The
// pending work is not finished afterwards.

static void make_matrix
(
    GrB_Matrix *A_handle,
    bool by_col,
    bool hyper,
    bool zombies,
    bool pending,
    bool wait_first,
    int hint,           // the hint to change
    int bits,           // the field to query the # of bits of that hint
    int from,           // initial # of bits of all integers
    int to              // new # of bits for the hint
)
{
    GrB_Info info ;
    GrB_Matrix A = NULL, M = NULL, R = NULL, S = NULL ;
    GrB_Scalar empty = NULL ;
    int sparsity = hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;
    // the stride makes A hypersparse with the default sparsity control;
    // the sparse case is forced
    int control = hyper ? GxB_AUTO_SPARSITY : GxB_SPARSE ;

    // A(0:STRIDE:end,:) = R if by row, or A(:,0:STRIDE:end) = R if by col
    GrB_Index nrows = by_col ? N : (STRIDE * N) ;
    GrB_Index ncols = by_col ? (STRIDE * N) : N ;
    GrB_Index stride [3] ;
    stride [GxB_BEGIN] = 0 ;
    stride [GxB_INC  ] = STRIDE ;
    stride [GxB_END  ] = STRIDE * (N-1) ;
    simple_rand_seed (1) ;
    OK (GB_mx_random_matrix (&R, false, false, N, N, NZ, 1, false)) ;
    OK (GrB_Matrix_new (&A, GrB_FP64, nrows, ncols)) ;
    OK (GrB_Matrix_set_INT32 (A, by_col ? GxB_BY_COL : GxB_BY_ROW,
        GxB_FORMAT)) ;
    OK (GrB_Matrix_set_INT32 (A, control, GxB_SPARSITY_CONTROL)) ;
    if (by_col)
    { 
        OK (GrB_Matrix_assign (A, NULL, NULL, R, GrB_ALL, 0, stride,
            GxB_STRIDE, NULL)) ;
    }
    else
    { 
        OK (GrB_Matrix_assign (A, NULL, NULL, R, stride, GxB_STRIDE, GrB_ALL,
            0, NULL)) ;
    }
    // finish the assignment before setting the hints: GrB_wait on a matrix
    // with no prior entries does not use its integer hints
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_set_INT32 (A, from, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, from, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, from, GxB_OFFSET_INTEGER_HINT)) ;
    int32_t b ;
    OK (GrB_Matrix_get_INT32 (A, &b, bits)) ;
    CHECK (b == from) ;
    check_state (A, sparsity, false, false, hyper) ;

    // zombies: A<M,struct> = empty, where M = (A < CUTOFF)
    if (zombies)
    { 
        OK (GrB_Scalar_new (&empty, GrB_FP64)) ;
        OK (GrB_Matrix_new (&M, GrB_BOOL, nrows, ncols)) ;
        OK (GrB_Matrix_select_FP64 (M, NULL, NULL, GrB_VALUELT_FP64, A,
            CUTOFF, NULL)) ;
        OK (GrB_Matrix_assign_Scalar (A, M, NULL, empty, GrB_ALL, 0,
            GrB_ALL, 0, GrB_DESC_S)) ;
    }

    // pending tuples: A(0:NS-1,0:NS-1) += S
    if (pending)
    { 
        GrB_Index range [2] = { 0, NS-1 } ;
        simple_rand_seed (2) ;
        OK (GB_mx_random_matrix (&S, false, false, NS, NS, NZS, 1, false)) ;
        OK (GrB_Matrix_assign (A, NULL, GrB_PLUS_FP64, S, range, GxB_RANGE,
            range, GxB_RANGE, NULL)) ;
    }
    check_state (A, sparsity, zombies, pending, hyper) ;

    if (wait_first)
    { 
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
        check_state (A, sparsity, false, false, hyper) ;
    }

    // change the hint; this converts the integers of A immediately
    OK (GrB_Matrix_set_INT32 (A, to, hint)) ;
    OK (GrB_Matrix_get_INT32 (A, &b, bits)) ;
    CHECK (b == to) ;

    // A must be valid, with its zombies, pending tuples, and hyper_hash still
    // present
    OK (GxB_Matrix_fprint (A, "A after hint change", GxB_SILENT, NULL)) ;
    check_state (A, sparsity, zombies && !wait_first, pending && !wait_first,
        hyper) ;

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&R)) ;
    OK (GrB_Matrix_free (&S)) ;
    OK (GrB_Scalar_free (&empty)) ;
    (*A_handle) = A ;
}

//------------------------------------------------------------------------------
// make_vector: create a vector, edit it, and change one integer hint
//------------------------------------------------------------------------------

// v is the first column of a random N-by-2 matrix, sparse, with all integers
// of size "from".  Zombies and pending tuples are made as in make_matrix,
// with v(0:NS-1) += s for a smaller random vector s.

static void make_vector
(
    GrB_Vector *v_handle,
    bool zombies,
    bool pending,
    bool wait_first,
    int hint,
    int bits,
    int from,
    int to
)
{
    GrB_Info info ;
    GrB_Matrix R = NULL ;
    GrB_Vector v = NULL, m = NULL, s = NULL ;
    GrB_Scalar empty = NULL ;

    // v = R (:,0) for a random matrix R
    simple_rand_seed (3) ;
    OK (GB_mx_random_matrix (&R, false, false, N, 2, N/2, 1, false)) ;
    OK (GrB_Vector_new (&v, GrB_FP64, N)) ;
    OK (GrB_Vector_set_INT32 (v, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Col_extract (v, NULL, NULL, R, GrB_ALL, N, 0, NULL)) ;
    OK (GrB_Matrix_free (&R)) ;
    OK (GrB_Vector_set_INT32 (v, from, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Vector_set_INT32 (v, from, GxB_OFFSET_INTEGER_HINT)) ;
    OK (GrB_Vector_wait (v, GrB_MATERIALIZE)) ;
    int32_t b ;
    OK (GrB_Vector_get_INT32 (v, &b, bits)) ;
    CHECK (b == from) ;
    GrB_Matrix V = (GrB_Matrix) v ;
    check_state (V, GxB_SPARSE, false, false, false) ;

    // zombies: v<m,struct> = empty, where m = (v < CUTOFF)
    if (zombies)
    { 
        OK (GrB_Scalar_new (&empty, GrB_FP64)) ;
        OK (GrB_Vector_new (&m, GrB_BOOL, N)) ;
        OK (GrB_Vector_select_FP64 (m, NULL, NULL, GrB_VALUELT_FP64, v,
            CUTOFF, NULL)) ;
        OK (GrB_Vector_assign_Scalar (v, m, NULL, empty, GrB_ALL, 0,
            GrB_DESC_S)) ;
    }

    // pending tuples: v(0:NS-1) += s
    if (pending)
    { 
        GrB_Index range [2] = { 0, NS-1 } ;
        simple_rand_seed (4) ;
        OK (GB_mx_random_matrix (&R, false, false, NS, 2, NS/2, 1, false)) ;
        OK (GrB_Vector_new (&s, GrB_FP64, NS)) ;
        OK (GrB_Col_extract (s, NULL, NULL, R, GrB_ALL, NS, 0, NULL)) ;
        OK (GrB_Matrix_free (&R)) ;
        OK (GrB_Vector_assign (v, NULL, GrB_PLUS_FP64, s, range, GxB_RANGE,
            NULL)) ;
    }
    check_state (V, GxB_SPARSE, zombies, pending, false) ;

    if (wait_first)
    { 
        OK (GrB_Vector_wait (v, GrB_MATERIALIZE)) ;
        check_state (V, GxB_SPARSE, false, false, false) ;
    }

    // change the hint
    OK (GrB_Vector_set_INT32 (v, to, hint)) ;
    OK (GrB_Vector_get_INT32 (v, &b, bits)) ;
    CHECK (b == to) ;
    OK (GxB_Vector_fprint (v, "v after hint change", GxB_SILENT, NULL)) ;
    check_state (V, GxB_SPARSE, zombies && !wait_first, pending && !wait_first,
        false) ;

    OK (GrB_Vector_free (&m)) ;
    OK (GrB_Vector_free (&s)) ;
    OK (GrB_Scalar_free (&empty)) ;
    (*v_handle) = v ;
}

//------------------------------------------------------------------------------
// GB_mex_test53 mexFunction
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
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, B = NULL, E = NULL, F = NULL ;
    GrB_Vector v = NULL, e = NULL ;
    int ntests = 0 ;

    //--------------------------------------------------------------------------
    // matrices
    //--------------------------------------------------------------------------

    // which = 0: row index hint, 1: column index hint, 2: offset hint
    int hints [3] = { GxB_ROWINDEX_INTEGER_HINT, GxB_COLINDEX_INTEGER_HINT,
                      GxB_OFFSET_INTEGER_HINT } ;
    int bits  [3] = { GxB_ROWINDEX_INTEGER_BITS, GxB_COLINDEX_INTEGER_BITS,
                      GxB_OFFSET_INTEGER_BITS } ;

    for (int by_col = 0 ; by_col <= 1 ; by_col++)
    for (int hyper = 0 ; hyper <= 1 ; hyper++)
    for (int zombies = 0 ; zombies <= 1 ; zombies++)
    for (int pending = 0 ; pending <= 1 ; pending++)
    for (int which = 0 ; which <= 2 ; which++)
    for (int from = 32 ; from <= 64 ; from += 32)
    {
        int to = (from == 32) ? 64 : 32 ;

        // A: hint changed with pending work in A;
        // E: hint changed after finishing the work
        make_matrix (&A, by_col, hyper, zombies, pending,
            false, hints [which], bits [which], from, to) ;
        make_matrix (&E, by_col, hyper, zombies, pending,
            true,  hints [which], bits [which], from, to) ;
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
        OK (GxB_Matrix_fprint (A, "A after wait", GxB_SILENT, NULL)) ;
        int sparsity = hyper ? GxB_HYPERSPARSE : GxB_SPARSE ;
        check_state (A, sparsity, false, false, hyper) ;
        check_state (E, sparsity, false, false, hyper) ;
        CHECK (GB_mx_isequal (A, E, 0)) ;

        // use A in a computation: B = A+A and F = E+E
        GrB_Index nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        OK (GrB_Matrix_new (&B, GrB_FP64, nrows, ncols)) ;
        OK (GrB_Matrix_new (&F, GrB_FP64, nrows, ncols)) ;
        OK (GrB_Matrix_eWiseAdd_BinaryOp (B, NULL, NULL, GrB_PLUS_FP64,
            A, A, NULL)) ;
        OK (GrB_Matrix_eWiseAdd_BinaryOp (F, NULL, NULL, GrB_PLUS_FP64,
            E, E, NULL)) ;
        OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
        OK (GrB_Matrix_wait (F, GrB_MATERIALIZE)) ;
        CHECK (GB_mx_isequal (B, F, 0)) ;

        OK (GrB_Matrix_free (&A)) ;
        OK (GrB_Matrix_free (&B)) ;
        OK (GrB_Matrix_free (&E)) ;
        OK (GrB_Matrix_free (&F)) ;
        ntests++ ;
    }

    //--------------------------------------------------------------------------
    // vectors
    //--------------------------------------------------------------------------

    // which = 0: row index hint, 2: offset hint
    for (int zombies = 0 ; zombies <= 1 ; zombies++)
    for (int pending = 0 ; pending <= 1 ; pending++)
    for (int which = 0 ; which <= 2 ; which += 2)
    for (int from = 32 ; from <= 64 ; from += 32)
    {
        int to = (from == 32) ? 64 : 32 ;
        make_vector (&v, zombies, pending, false,
            hints [which], bits [which], from, to) ;
        make_vector (&e, zombies, pending, true,
            hints [which], bits [which], from, to) ;
        OK (GrB_Vector_wait (v, GrB_MATERIALIZE)) ;
        check_state ((GrB_Matrix) v, GxB_SPARSE, false, false, false) ;
        check_state ((GrB_Matrix) e, GxB_SPARSE, false, false, false) ;
        OK (GxB_Vector_fprint (v, "v after wait", GxB_SILENT, NULL)) ;
        CHECK (GB_mx_isequal ((GrB_Matrix) v, (GrB_Matrix) e, 0)) ;
        OK (GrB_Vector_free (&v)) ;
        OK (GrB_Vector_free (&e)) ;
        ntests++ ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test53:  all %d tests passed\n\n", ntests) ;
}

