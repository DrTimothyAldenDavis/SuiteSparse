////////////////////////////////////////////////////////////////////////////////
/////////////////////////// paru_umfpack_info //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

// Translate UMFPACK status into ParU_Info

#include "paru_internal.hpp"

ParU_Info paru_umfpack_info (int status)
{
    switch (status)
    {
        case UMFPACK_OK:
        case UMFPACK_WARNING_determinant_underflow:
        case UMFPACK_WARNING_determinant_overflow:
            return (PARU_SUCCESS) ;

        case UMFPACK_WARNING_singular_matrix:
            return (PARU_SINGULAR) ;

        case UMFPACK_ERROR_out_of_memory:
            return (PARU_OUT_OF_MEMORY) ;

        default:
            return (PARU_INVALID) ;
    }
}

