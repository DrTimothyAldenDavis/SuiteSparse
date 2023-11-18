//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_copy: copy a sparse matrix (with change of stype)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Create a copy (C) of a sparse matrix A, with change of stype.
// A has any xtype, dtype, and stype, and can be packed or unpacked.
// C has the same xtype and dtype, but any stype (unless A is rectangular,
// in which case C->stype must be zero).  C is packed.

// A->stype == 0: A is unsymmetric
// A->stype < 0: A is symmetric with lower triangular part stored.  Any entries
//      above the diagonal are ignored.
// A->stype > 0: A is symmetric with upper triangular part stored.  Any entries
//      below the diagonal are ignored.

// C can be returned as numerical or pattern-only; and in the latter case,
// the diagonal entries can be ignored.

// In MATLAB:                                       Using cholmod_copy:
// ---------                                        ----------------------------
// C = A ;                                          A unsymmetric, C unsymmetric
// C = tril (A) ;                                   A unsymmetric, C sym lower
// C = triu (A) ;                                   A unsymmetric, C sym upper
// U = triu (A) ; L = tril (U',-1) ; C = L+U ;      A sym upper, C unsymmetric
// C = triu (A)' ;                                  A sym upper, C sym lower
// C = triu (A) ;                                   A sym upper, C sym upper
// L = tril (A) ; U = triu (L',1) ; C = L+U ;       A sym lower, C unsymmetric
// C = tril (A) ;                                   A sym lower, C sym lower
// C = tril (A)' ;                                  A sym lower, C sym upper

// If A is complex or zomplex, then U' or L' above can be computed as an
// array (non-conjugate) transpose, or a matrix (conjugate) transpose.

// The mode parameter:
//  2       numerical, with conjugate transpose
//  1       numerical, with non-conjugate transpose
//  0       pattern, keeping the diagonal
//  -1      pattern, remove the diagonal
//  -2      pattern, and add 50% + n extra space to C
//          as elbow room for AMD and CAMD, when converting
//          a symmetric matrix A to an unsymmetric matrix C

// workspace: Iwork (max (nrow,ncol))

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_sparse) (&C, Common) ;         \
        return (NULL) ;                             \
    }

//------------------------------------------------------------------------------
// t_cholmod_copy_worker template
//------------------------------------------------------------------------------

#define PATTERN
#include "t_cholmod_copy_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_copy_worker.c"

#define COMPLEX
#include "t_cholmod_copy_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_copy_worker.c"

#define ZOMPLEX
#include "t_cholmod_copy_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_copy_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_copy_worker.c"

#define COMPLEX
#include "t_cholmod_copy_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_copy_worker.c"

#define ZOMPLEX
#include "t_cholmod_copy_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_copy_worker.c"

//------------------------------------------------------------------------------

cholmod_sparse *CHOLMOD(copy)
(
    // input:
    cholmod_sparse *A,  // input matrix, not modified
    int stype,          // stype of C
    int mode,           // 2: numerical (conj)
                        // 1: numerical (non-conj.)
                        // 0: pattern (with diag)
                        // -1: pattern (remove diag)
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
    cholmod_sparse *C = NULL ;

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;
    int astype = SIGN (A->stype) ;
    stype = SIGN (stype) ;
    if ((stype || astype) && nrow != ncol)
    {
        ERROR (CHOLMOD_INVALID, "matrix invalid") ;
        return (NULL) ;
    }

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    mode = RANGE (mode, -2, 2) ;
    bool ignore_diag = (mode < 0) ;
    bool up = (astype > 0) ;
    bool lo = (astype < 0) ;
    bool values = (mode > 0) && (A->xtype != CHOLMOD_PATTERN) ;
    bool conj = (mode == 2) ;

    //--------------------------------------------------------------------------
    // copy the matrix
    //--------------------------------------------------------------------------

    if (astype == stype)
    {

        //----------------------------------------------------------------------
        // no change of stype, but possible remove of diagonal
        //----------------------------------------------------------------------

        C = CHOLMOD(band) (A, -nrow, ncol, mode, Common) ;
        RETURN_IF_ERROR ;

    }
    else if (astype == 0)
    {

        //----------------------------------------------------------------------
        // A is unsymmetric; C is symmetric upper or lower
        //----------------------------------------------------------------------

        size_t nlo = (stype > 0) ? 0 : (-nrow) ;
        size_t nup = (stype > 0) ? ncol : 0 ;
        C = CHOLMOD(band) (A, nlo, nup, mode, Common) ;
        RETURN_IF_ERROR ;
        C->stype = stype ;

    }
    else if (astype == -stype)
    {

        //----------------------------------------------------------------------
        // both A and C are symmetric, but opposite, so a transpose is needed
        //----------------------------------------------------------------------

        C = CHOLMOD(transpose) (A, mode, Common) ;
        RETURN_IF_ERROR ;
        if (ignore_diag)
        {
            // remove diagonal, if requested
            CHOLMOD(band_inplace) (-nrow, ncol, -1, C, Common) ;
            RETURN_IF_ERROR ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A is symmetric and C is unsymmetric
        //----------------------------------------------------------------------

        Int  *Ap  = (Int  *) A->p ;
        Int  *Anz = (Int  *) A->nz ;
        Int  *Ai  = (Int  *) A->i ;
        bool packed = A->packed ;

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        ASSERT (nrow == ncol) ;
        CHOLMOD(allocate_work) (0, ncol, 0, Common) ;
        RETURN_IF_ERROR ;
        Int *Wj = (Int *) Common->Iwork ;

        //----------------------------------------------------------------------
        // count entries in each column of C
        //----------------------------------------------------------------------

        memset (Wj, 0, ncol * sizeof (Int)) ;
        size_t cnz = 0 ;

        for (Int j = 0 ; j < ncol ; j++)
        {
            Int p = Ap [j] ;
            Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
            for ( ; p < pend ; p++)
            {
                Int i = Ai [p] ;
                if (i == j)
                {
                    // diagonal entry A(i,i)
                    if (ignore_diag) continue ;
                    Wj [j]++ ;
                    cnz++ ;
                }
                else if ((up && i < j) || (lo && i > j))
                {
                    // A(i,j) is placed in both upper and lower part of C
                    Wj [j]++ ;
                    Wj [i]++ ;
                    cnz += 2 ;
                }
            }
        }

        //----------------------------------------------------------------------
        // allocate C
        //----------------------------------------------------------------------

        size_t cnzmax = cnz + ((mode == -2) ? (cnz/2 + ncol) : 0) ;
        C = CHOLMOD(allocate_sparse) (nrow, ncol, cnzmax,
            /* C is sorted if A is sorted: */ A->sorted,
            /* C is packed: */ TRUE, /* C stype: */ 0,
            (values ? A->xtype : CHOLMOD_PATTERN) + A->dtype, Common) ;
        RETURN_IF_ERROR ;

        //----------------------------------------------------------------------
        // Cp = cumsum of column counts (Wj), and then copy Cp back to Wj
        //----------------------------------------------------------------------

        Int *Cp = C->p ;
        CHOLMOD(cumsum) (Cp, Wj, ncol) ;
        memcpy (Wj, Cp, ncol * sizeof (Int)) ;

        //----------------------------------------------------------------------
        // construct C via the template function
        //----------------------------------------------------------------------

        switch ((C->xtype + C->dtype) % 8)
        {

            default:
                p_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                break ;

            case CHOLMOD_REAL    + CHOLMOD_SINGLE:
                rs_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
                if (conj)
                {
                    cs_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                else
                {
                    cs_t_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
                if (conj)
                {
                    zs_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                else
                {
                    zs_t_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                break ;

            case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
                rd_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                break ;

            case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
                if (conj)
                {
                    cd_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                else
                {
                    cd_t_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                break ;

            case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
                if (conj)
                {
                    zd_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                else
                {
                    zd_t_cholmod_copy_worker (C, A, ignore_diag, Common) ;
                }
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    DEBUG (size_t nnzdiag = CHOLMOD(dump_sparse) (C, "copy:C", Common)) ;
    ASSERT (IMPLIES (ignore_diag, nnzdiag == 0)) ;
    return (C) ;
}

