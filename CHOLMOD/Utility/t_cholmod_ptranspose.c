//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_ptranspose: permuted transpose
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// C = A' or A(p,p)' if A is symmetric. C = A', A(:,f)' or A(p,f)' if A is
// unsymmetric.
//
// workspace: at most nrow+ncol

#include "cholmod_internal.h"

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        CHOLMOD(free_sparse) (&C, Common) ;         \
        return (NULL) ;                             \
    }

cholmod_sparse *CHOLMOD(ptranspose)
(
    // input:
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj)
                        // 1: numerical (non-conj.)
                        // <= 0: pattern (with diag)
    Int *Perm,          // permutation for C=A(p,f)' or C=A(p,p)', or NULL
    Int *fset,          // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, NULL) ;
    Common->status = CHOLMOD_OK ;
    mode = RANGE (mode, -1, 2) ;

    //--------------------------------------------------------------------------
    // count # of entries in C
    //--------------------------------------------------------------------------

    Int cnz = 0 ;
    if ((A->stype == 0) && (fset != NULL))
    {
        // C = A(p,f)' or A(:,f)' where A is unsymmetric and fset is present
        Int ncol = A->ncol ;
        if (A->packed)
        {
            Int *Ap  = (Int *) A->p ;
            for (Int k = 0 ; k < fsize ; k++)
            {
                Int j = fset [k] ;
                if (j < 0 || j >= ncol) continue ;
                cnz += (Ap [j+1] - Ap [j]) ;
            }
        }
        else
        {
            Int *Anz = (Int *) A->nz ;
            for (Int k = 0 ; k < fsize ; k++)
            {
                Int j = fset [k] ;
                if (j < 0 || j >= ncol) continue ;
                cnz += Anz [j] ;
            }
        }
    }
    else
    {
        // C = A(p,p)' A is symmetric, or C=A' where A is any matrix
        cnz = CHOLMOD(nnz) (A, Common) ;
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    cholmod_sparse *C = CHOLMOD(allocate_sparse) (A->ncol, A->nrow, cnz,
        /* C is sorted: */ TRUE, /* C is packed */ TRUE,
        /* C has the opposite stype as A: */ -(A->stype),
        ((mode > 0) ? A->xtype : CHOLMOD_PATTERN) + A->dtype, Common) ;
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    if (A->stype != 0)
    {
        // C = A (p,p)' or A' when A is symmetric (upper or lower)
        CHOLMOD(transpose_sym) (A, mode, Perm, C, Common) ;
    }
    else
    {
        // C = A (p,f)' or A(:,f)' when A is unsymmetric
        CHOLMOD(transpose_unsym) (A, mode, Perm, fset, fsize, C, Common) ;
    }
    RETURN_IF_ERROR ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (C) ;
}

