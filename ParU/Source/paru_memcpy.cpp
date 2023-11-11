////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_memcpy ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*!  @brief  wrapper around memcpy
 *
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

void paru_memcpy(void *destination, const void *source, 
        size_t num, ParU_Control *Control)
{

    int nthreads = control_nthreads (Control) ;
    size_t mem_chunk = (size_t) control_mem_chunk (Control) ;

    if (num < mem_chunk || nthreads == 1)
    {  // single task memcpy
        memcpy(destination, source, num);
    }
    else
    {  // multiple task memcpy
        size_t nchunks = 1 + (num / mem_chunk);
        if (((size_t) nthreads) > nchunks)
        { 
            nthreads = (int) nchunks ;
        }

        int64_t k;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (k = 0; k < (int64_t)nchunks; k++)
        {
            size_t start = k * mem_chunk;
            if (start < num)
            {
                size_t chunk = std::min(num - start, mem_chunk);
                // void* arithmetic is illegal it is why I am using this
                unsigned char *pdest = static_cast<unsigned char*>(destination) + start;
                const unsigned char *psrc = static_cast<const unsigned char*>(source) + start;
                memcpy(pdest, psrc, chunk);
            }
        }
    }
}
