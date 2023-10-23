//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_dense2_worker: copy a dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

// Copies a dense matrix X into Y, with change of leading dimension.  If the
// leading dimensions are the same, the copy is done in the caller,
// t_cholmod_copy_dense2.

#include "cholmod_template.h"

static void TEMPLATE (cholmod_copy_dense2_worker)
(
    cholmod_dense *X,
    cholmod_dense *Y
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (X->d != Y->d) ;
    ASSERT (X->nrow == Y->nrow) ;
    ASSERT (X->ncol == Y->ncol) ;
    ASSERT (X->dtype == Y->dtype) ;
    ASSERT (X->xtype == Y->xtype) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    Real *Xx = (Real *) X->x ;
    Real *Xz = (Real *) X->z ;
    Real *Yx = (Real *) Y->x ;
    Real *Yz = (Real *) Y->z ;
    size_t nrow = X->nrow ;
    size_t ncol = X->ncol ;
    size_t xd = X->d ;
    size_t yd = Y->d ;

    //--------------------------------------------------------------------------
    // get the sizes of the entries
    //--------------------------------------------------------------------------

    size_t e = (X->dtype == CHOLMOD_SINGLE) ? sizeof (float) : sizeof (double) ;
    size_t fx = ((X->xtype == CHOLMOD_COMPLEX) ? 2 : 1) ;
    size_t fz = ((X->xtype == CHOLMOD_ZOMPLEX) ? 1 : 0) ;
    size_t e_fx_nrow = e * fx * nrow ;
    size_t e_fz_nrow = e * fz * nrow ;

    //--------------------------------------------------------------------------
    // copy X = Y
    //--------------------------------------------------------------------------

    for (size_t j = 0 ; j < ncol ; j++)
    {

        //----------------------------------------------------------------------
        // Y (:,j) = X (:,j)
        //----------------------------------------------------------------------

        memcpy (Yx, Xx, e_fx_nrow) ;
        Xx += xd * fx ;
        Yx += yd * fx ;

        #if defined ( ZOMPLEX )
        memcpy (Yz, Xz, e_fz_nrow) ;
        Xz += xd * fz ;
        Yz += yd * fz ;
        #endif
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX

