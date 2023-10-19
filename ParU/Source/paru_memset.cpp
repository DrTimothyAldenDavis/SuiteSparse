////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_memset  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*!  @brief  wrapper around memset
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"

void paru_memset(void *ptr, int64_t value, size_t num, ParU_Control *Control)
{
    int nthreads = control_nthreads (Control) ;
    size_t mem_chunk = (size_t) control_mem_chunk (Control) ;

    if (num < mem_chunk)
    {  // single task memse
        memset(ptr, value, num);
    }
    else
    {  // multiple task memset
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
                size_t chunk = MIN(num - start, mem_chunk);
                // void* arithmetic is illegal it is why I am using this
                unsigned char *ptr_chunk = (unsigned char *)ptr + start;
                memset(ptr_chunk, value, chunk);
            }
        }
    }
}

