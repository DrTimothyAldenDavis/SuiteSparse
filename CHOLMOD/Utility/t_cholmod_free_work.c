//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_free_work: free workspace in Common (int32/64)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(free_work) (cholmod_common *Common)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (FALSE) ;

    //--------------------------------------------------------------------------
    // free Flag and Head, of size nrow and nrow+1 Ints
    //--------------------------------------------------------------------------

    size_t nrow = Common->nrow ;
    Common->Flag = CHOLMOD(free) (nrow,   sizeof (Int), Common->Flag, Common) ;
    Common->Head = CHOLMOD(free) (nrow+1, sizeof (Int), Common->Head, Common) ;
    Common->nrow = 0 ;

    //--------------------------------------------------------------------------
    // free Iwork, of size iworksize Ints
    //--------------------------------------------------------------------------

    Common->Iwork = CHOLMOD(free) (Common->iworksize,
        sizeof (Int), Common->Iwork, Common) ;
    Common->iworksize = 0 ;

    //--------------------------------------------------------------------------
    // free Xwork, of size xworkbytes
    //--------------------------------------------------------------------------

    Common->Xwork = CHOLMOD(free) (Common->xworkbytes, sizeof (uint8_t),
        Common->Xwork, Common) ;
    Common->xworkbytes = 0 ;

    //--------------------------------------------------------------------------
    // free GPU workspace
    //--------------------------------------------------------------------------

    #ifdef CHOLMOD_HAS_CUDA
        CHOLMOD(gpu_deallocate) (Common) ;
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (TRUE) ;
}

