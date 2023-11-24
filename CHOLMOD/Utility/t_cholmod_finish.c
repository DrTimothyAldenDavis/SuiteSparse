//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_finish: finish CHOLMOD (int32/int64 version)
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

// cholmod_start or cholmod_l_start must be called once prior to calling any
// other CHOLMOD method.  It contains workspace that must be freed by
// cholmod_finish or cholmod_l_finish (which is just another name for
// cholmod_free_work or cholmod_l_free_work, respetively),

int CHOLMOD(finish) (cholmod_common *Common)
{

    #ifdef BLAS_DUMP
    if (Common->blas_dump != NULL)
    {
        fclose (Common->blas_dump) ;
        Common->blas_dump = NULL ;
    }
    #endif

    return (CHOLMOD(free_work) (Common)) ;
}

