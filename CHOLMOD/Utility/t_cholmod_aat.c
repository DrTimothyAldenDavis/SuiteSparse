//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_aat: compute A*A' or A(:,f)*A(:,f)'
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// C = A*A' or C = A(:,f)*A(:,f)', where for complex/zomplex matrices, A' can
// be computed as a matrix (complex conjugate) transpose, or an array (complex
// non-conjugate) transpose.  On output the C->stype is zero.  The input matrix
// A must have A->stype = 0.

// workspace:
//      Iwork of size max (nrow,ncol)
//      Xwork of size n if real and mode > 0, or 2*n if complex and mode > 0

// The mode parameter:
//  2       numerical, with conjugate transpose
//  1       numerical, with non-conjugate transpose
//  0       pattern, keeping the diagonal
//  -1      pattern, remove the diagonal
//  -2      pattern, and add 50% + n extra space to C 
//          as elbow room for AMD and CAMD, when converting
//          a symmetric matrix A to an unsymmetric matrix C

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_sparse) (&C, Common) ;         \
        CHOLMOD(free_sparse) (&F, Common) ;         \
        return (NULL) ;                             \
    }

//------------------------------------------------------------------------------
// t_cholmod_aat_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_aat_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_aat_worker.c"
#define COMPLEX
#include "t_cholmod_aat_worker.c"
#define ZOMPLEX
#include "t_cholmod_aat_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_aat_worker.c"
#define COMPLEX
#include "t_cholmod_aat_worker.c"
#define ZOMPLEX
#include "t_cholmod_aat_worker.c"

//------------------------------------------------------------------------------
// cholmod_aat: C=A*A' or A(:,f)*A(:,f)'
//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(aat)
(
    cholmod_sparse *A,  // input matrix
    Int *fset,          // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    int mode,           // 2: numerical (conj), 1: numerical (non-conj.),
                        // 0: pattern (with diag), -1: pattern (remove diag),
                        // -2: pattern (remove diag; add ~50% extra space in C)
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;
    cholmod_sparse *C = NULL, *F = NULL ;
    ASSERT (CHOLMOD(dump_sparse) (A, "aat:A", Common) >= 0) ;

    if (A->stype != 0)
    {
        ERROR (CHOLMOD_INVALID, "input matrix must be unsymmetric") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    bool ignore_diag = (mode < 0) ;
    bool values = (mode > 0) && (A->xtype != CHOLMOD_PATTERN) ;
    int axtype = (values) ? A->xtype : CHOLMOD_PATTERN ;
    bool conj = (mode >= 2) ;

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;
    Int n = A->nrow ;

    Int  *Ap  = (Int  *) A->p ;
    Int  *Anz = (Int  *) A->nz ;
    Int  *Ai  = (Int  *) A->i ;
    bool packed = A->packed ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries of C and A
    //--------------------------------------------------------------------------

    size_t ei = sizeof (Int) ;
    size_t e = (A->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t ew =  ((axtype == CHOLMOD_PATTERN) ? 0 :
                 ((axtype == CHOLMOD_COMPLEX) ? 2 : 1)) ;
    size_t ex = e * ew ;
    size_t ez = e * ((axtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    CHOLMOD(alloc_work) (0, MAX (ncol, nrow), ew, A->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // construct the transpose F = A' or F = A(:,f)', or complex conjugate
    //--------------------------------------------------------------------------

    // no row permutation (Perm is NULL)
    // uses Common->Iwork if fset is not NULL
    F = CHOLMOD(ptranspose) (A, mode, /* Perm: */ NULL, fset, fsize, Common) ;
    RETURN_IF_ERROR ;

    Int *Fp = (Int *) F->p ;
    Int *Fi = (Int *) F->i ;

    Int *W = Common->Iwork ;        // size n, not yet initialized
    CHOLMOD(set_empty) (W, n) ;     // W [0..n-1] = EMPTY

    //--------------------------------------------------------------------------
    // cnz = nnz (C)
    //--------------------------------------------------------------------------

    int64_t cnz = 0 ;
    for (Int j = 0 ; j < n ; j++)
    {

        //----------------------------------------------------------------------
        // If W [i] != jmark then row i has not yet been seen in C(:,j)
        //----------------------------------------------------------------------

        Int jmark = -j-2 ;

        //----------------------------------------------------------------------
        // compute # of entries in C(:,j) = A*F(:,j)
        //----------------------------------------------------------------------

        for (Int pf = Fp [j] ; pf < Fp [j+1] ; pf++)
        {

            //------------------------------------------------------------------
            // get the F(t,j) entry
            //------------------------------------------------------------------

            Int t = Fi [pf] ;

            //------------------------------------------------------------------
            // add any entries A(i,t) not already in C(:,j) pattern
            //------------------------------------------------------------------

            Int p = Ap [t] ;
            Int pend = (packed) ? (Ap [t+1]) : (p + Anz [t]) ;
            for ( ; p < pend ; p++)
            {
                // get A(i,t); it is seen for first time if W [i] != jmark
                Int i = Ai [p] ;
                if (ignore_diag && i == j) continue ;
                if (W [i] != jmark)
                {
                    W [i] = jmark ;
                    cnz++ ;
                }
            }
        }

        //----------------------------------------------------------------------
        // check for integer overflow
        //----------------------------------------------------------------------

        if (cnz < 0 || cnz >= Int_max / sizeof (Int))
        {
            Common->status = CHOLMOD_TOO_LARGE ;
            break ;
        }
    }

    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    size_t cnzmax = cnz + ((mode == -2) ? (cnz/2 + n) : 0) ;
    C = CHOLMOD(allocate_sparse) (n, n, cnzmax,
        /* C is not sorted: */ FALSE, /* C is packed: */ TRUE,
        /* C stype: */ 0, axtype + A->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // compute the pattern and values of C
    //--------------------------------------------------------------------------

    switch ((C->xtype + C->dtype) % 8)
    {

        default:
            p_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_REAL:
            r_s_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_COMPLEX:
            c_s_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_SINGLE + CHOLMOD_ZOMPLEX:
            z_s_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_REAL:
            r_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_COMPLEX:
            c_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;

        case CHOLMOD_DOUBLE + CHOLMOD_ZOMPLEX:
            z_cholmod_aat_worker (C, A, F, ignore_diag, Common) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    DEBUG (size_t nnzdiag = CHOLMOD(dump_sparse) (C, "aat:C", Common)) ;
    ASSERT (IMPLIES (ignore_diag, nnzdiag == 0)) ;
    CHOLMOD(free_sparse) (&F, Common) ;
    return (C) ;
}

