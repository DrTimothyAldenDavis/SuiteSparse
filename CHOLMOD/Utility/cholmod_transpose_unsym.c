//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_transpose_unsym: unsymmetric permuted transpose
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// Creates C = A', A(:,f)' or A(p,f)' C must be already allocated on input.
// The matrices A and C are unsymmetric (with stype of zero).  See
// cholmod_transpose and cholmod_ptranspose for methods with a simpler
// interface.

// The notation A(:,f) and A(p,f) refers to a matrix of the same size as A,
// where p is a permutation vector and f is a vector fset of size fsize.
// Any column j not in the fset list still appears in A(:,f), just empty.
// p must be a permutation of 0:A->nrow-1, and f must be a permutation of
// a subset of 0:A->ncol-1.

#define RETURN_IF_ERROR                             \
    if (Common->status < CHOLMOD_OK)                \
    {                                               \
        return (FALSE) ;                            \
    }

//------------------------------------------------------------------------------
// t_cholmod_transpose_unsym_worker template
//------------------------------------------------------------------------------

#define NUMERIC

#define PATTERN
#include "t_cholmod_transpose_unsym_worker.c"

#define DOUBLE
#define REAL
#include "t_cholmod_transpose_unsym_worker.c"

#define COMPLEX
#include "t_cholmod_transpose_unsym_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_unsym_worker.c"

#define ZOMPLEX
#include "t_cholmod_transpose_unsym_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_unsym_worker.c"

#undef  DOUBLE
#define SINGLE
#define REAL
#include "t_cholmod_transpose_unsym_worker.c"

#define COMPLEX
#include "t_cholmod_transpose_unsym_worker.c"
#define COMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_unsym_worker.c"

#define ZOMPLEX
#include "t_cholmod_transpose_unsym_worker.c"
#define ZOMPLEX
#define NCONJUGATE
#include "t_cholmod_transpose_unsym_worker.c"

#undef NUMERIC

//------------------------------------------------------------------------------
// cm_copy_Cnz: copy Wi into Cnz
//------------------------------------------------------------------------------

static void cm_copy_Cnz (Int *Cnz, Int *Wi, Int *Perm, Int nrow)
{
    if (Perm == NULL)
    {
        // Cnz [0..nrow-1] = Wi [0..nrow-1]
        memcpy (Cnz, Wi, nrow * sizeof (Int)) ;
    }
    else
    {
        // Cnz [0..nrow-1] = Wi [Perm [0..nrow-1]]
        for (Int i = 0 ; i < nrow ; i++)
        {
            Cnz [i] = Wi [Perm [i]] ;
        }
    }
}

//------------------------------------------------------------------------------
// cholmod_transpose_unsym
//------------------------------------------------------------------------------

int CHOLMOD(transpose_unsym)
(
    // input:
    cholmod_sparse *A,  // input matrix
    int mode,           // 2: numerical (conj)
                        // 1: numerical (non-conj.)
                        // 0: pattern (with diag)
    Int *Perm,          // permutation for C=A(p,f)', or NULL
    Int *fset,          // a list of column indices in range 0:A->ncol-1
    size_t fsize,       // # of entries in fset
    // input/output:
    cholmod_sparse *C,  // output matrix, must be allocated on input
    cholmod_common *Common
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_SPARSE_MATRIX_INVALID (A, FALSE) ;
    RETURN_IF_NULL (C, FALSE) ;
    Common->status = CHOLMOD_OK ;

    mode = RANGE (mode, 0, 2) ;

    if (A->xtype == CHOLMOD_PATTERN || C->xtype == CHOLMOD_PATTERN)
    {
        // A or C is pattern: C must be pattern, so mode can only be zero
        mode = 0 ;
    }

    Int nrow = A->nrow ;
    Int ncol = A->ncol ;

    if (A->stype != 0)
    {
        ERROR (CHOLMOD_INVALID, "A is invalid") ;
        return (FALSE) ;
    }

    if ((C->xtype != ((mode == 0) ? CHOLMOD_PATTERN : A->xtype)) ||
        (C->dtype != A->dtype) || (nrow != C->ncol) || (ncol != C->nrow) ||
        (C->stype != 0))
    {
        ERROR (CHOLMOD_INVALID, "C is invalid") ;
        return (FALSE) ;
    }

    ASSERT (CHOLMOD(dump_sparse) (A, "transpose_unsym:A", Common) >= 0) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    size_t iworksize = (fset == NULL) ? nrow : MAX (nrow, ncol) ;
    CHOLMOD(allocate_work) (0, iworksize, 0, Common) ;
    RETURN_IF_ERROR ;
    Int *Wi = (Int *) Common->Iwork ;   // size n integers

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Int *Ap  = (Int *) A->p ;
    Int *Ai  = (Int *) A->i ;
    Int *Anz = (Int *) A->nz ;

    Int *Cp  = (Int *) C->p ;
    Int *Cnz = (Int *) C->nz ;

    //--------------------------------------------------------------------------
    // check Perm if present
    //--------------------------------------------------------------------------

    if (Perm != NULL)
    {
        memset (Wi, 0, nrow * sizeof (Int)) ;
        for (Int k = 0 ; k < nrow ; k++)
        {
            Int i = Perm [k] ;
            if (i < 0 || i > nrow || Wi [i] == 1)
            {
                ERROR (CHOLMOD_INVALID, "invalid permutation") ;
                return (FALSE) ;
            }
            Wi [i] = 1 ;
        }
    }

    ASSERT (CHOLMOD(dump_perm) (Perm, nrow, nrow, "Perm", Common)) ;

    //--------------------------------------------------------------------------
    // check fset if present, and also determine if it's sorted
    //--------------------------------------------------------------------------

    Int nf = (Int) fsize ;
    bool fsorted = true ;
    if (fset != NULL)
    {
        Int jlast = EMPTY ;
        memset (Wi, 0, ncol * sizeof (Int)) ;
        for (Int k = 0 ; k < nf ; k++)
        {
            Int j = fset [k] ;
            if (j < 0 || j > ncol || Wi [j] == 1)
            {
                ERROR (CHOLMOD_INVALID, "invalid fset") ;
                return (FALSE) ;
            }
            Wi [j] = 1 ;
            fsorted = fsorted && (j > jlast) ;
            jlast = j ;
        }
    }

    ASSERT (CHOLMOD(dump_perm) (fset, fsize, ncol, "fset", Common)) ;

    //--------------------------------------------------------------------------
    // count entries in each row of A or A(:,f)
    //--------------------------------------------------------------------------

    memset (Wi, 0, nrow * sizeof (Int)) ;

    if (fset != NULL)
    {

        //----------------------------------------------------------------------
        // count entries in rows of (A:,f)
        //----------------------------------------------------------------------

        if (A->packed)
        {
            #define PACKED
            #define FSET
            #include "t_cholmod_transpose_unsym_template.c"
        }
        else
        {
            #define FSET
            #include "t_cholmod_transpose_unsym_template.c"
        }

        //----------------------------------------------------------------------
        // save the nz counts if C is unpacked, and recount all of A
        //----------------------------------------------------------------------

        if (!(C->packed))
        {

            cm_copy_Cnz (Cnz, Wi, Perm, nrow) ;

            //------------------------------------------------------------------
            // count entries in rows of A
            //------------------------------------------------------------------

            memset (Wi, 0, nrow * sizeof (Int)) ;
            if (A->packed)
            {
                #define PACKED
                #include "t_cholmod_transpose_unsym_template.c"
            }
            else
            {
                #include "t_cholmod_transpose_unsym_template.c"
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // count entries in rows of A
        //----------------------------------------------------------------------

        if (A->packed)
        {
            #define PACKED
            #include "t_cholmod_transpose_unsym_template.c"
        }
        else
        {
            #include "t_cholmod_transpose_unsym_template.c"
        }

        //----------------------------------------------------------------------
        // save the nz counts if C is unpacked, and recount all of A
        //----------------------------------------------------------------------

        if (!(C->packed))
        {
            cm_copy_Cnz (Cnz, Wi, Perm, nrow) ;
        }
    }

    //--------------------------------------------------------------------------
    // Compute Cp
    //--------------------------------------------------------------------------

    Int p = 0 ;
    if (Perm == NULL)
    {
        // Cp = cumsum (Wi)
        p = CHOLMOD(cumsum) (Cp, Wi, nrow) ;
        // Wi [0..nrow-1] = Cp [0..nrow-1]
        memcpy (Wi, Cp, nrow * sizeof (Int)) ;
    }
    else
    {
        // Cp = cumsum (Wi [Perm])
        for (Int i = 0 ; i < nrow ; i++)
        {
            Cp [i] = p ;
            p += Wi [Perm [i]] ;
        }
        Cp [nrow] = p ;
        // Wi [Perm [0..nrow-1]] = Cp [0..nrow-1]
        for (Int i = 0 ; i < nrow ; i++)
        {
            Wi [Perm [i]] = Cp [i] ;
        }
    }

    if (p > (Int) C->nzmax)
    {
        ERROR (CHOLMOD_INVALID, "C is too small") ;
        return (FALSE) ;
    }

    //--------------------------------------------------------------------------
    // compute the pattern and values of C
    //--------------------------------------------------------------------------

    bool conj = (mode == 2) ;

    switch ((C->xtype + C->dtype) % 8)
    {
        default:
            p_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            break ;

        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            rs_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            if (conj)
            {
                cs_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            else
            {
                cs_t_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            if (conj)
            {
                zs_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            else
            {
                zs_t_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            rd_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            if (conj)
            {
                cd_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            else
            {
                cd_t_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            if (conj)
            {
                zd_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            else
            {
                zd_t_cholmod_transpose_unsym_worker (A, fset, nf, C, Wi) ;
            }
            break ;
    }

    //--------------------------------------------------------------------------
    // finalize C and return result
    //--------------------------------------------------------------------------

    C->sorted = fsorted ;
    ASSERT (CHOLMOD(dump_sparse) (C, "transpose_unsym:C", Common) >= 0) ;
    return (TRUE) ;
}

