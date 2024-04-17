//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_query: CHOLMOD query
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2024, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

bool CHOLMOD(query)
(
    cholmod_query_t feature
)
{
    switch (feature)
    {
        case CHOLMOD_QUERY_HAS_GPL         :
            #ifdef CHOLMOD_HAS_GPL
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_CHECK       :
            #ifdef CHOLMOD_HAS_CHECK
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_CHOLESKY    :
            #ifdef CHOLMOD_HAS_CHOLESKY
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_CAMD        :
            #ifdef CHOLMOD_HAS_CAMD
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_PARTITION   :
            #ifdef CHOLMOD_HAS_PARTITION
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_MATRIXOPS   :
            #ifdef CHOLMOD_HAS_MATRIXOPS
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_MODIFY      :
            #ifdef CHOLMOD_HAS_MODIFY
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_SUPERNODAL  :
            #ifdef CHOLMOD_HAS_SUPERNODAL
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_CUDA        :
            #ifdef CHOLMOD_HAS_CUDA
                return (true) ;
            #else
                return (false) ;
            #endif

        case CHOLMOD_QUERY_HAS_OPENMP      :
            #ifdef CHOLMOD_HAS_OPENMP
                return (true) ;
            #else
                return (false) ;
            #endif

        default :
            // undefined feature
            return (false) ;
    }
}

