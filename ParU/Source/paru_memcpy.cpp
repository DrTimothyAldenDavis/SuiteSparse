////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_memcpy ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*!  @brief  wrapper around memcpy
 *
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

void paru_memcpy
(
    void *destination,      // output array of size nbytes
    const void *source,     // input array of size nbytes
    size_t nbytes,          // # of bytes to copy
    size_t mem_chunk,
    int32_t nthreads
)
{

    if (destination == NULL || source == NULL) return ;

    if (nbytes < mem_chunk || nthreads == 1)
    {
        // single task memcpy
        memcpy(destination, source, nbytes);
    }
    else
    {

        // multiple task memcpy
        size_t nchunks = 1 + (nbytes / mem_chunk);
        if (((size_t) nthreads) > nchunks)
        {
            nthreads = (int) nchunks ;
        }

        int64_t k;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (k = 0; k < (int64_t)nchunks; k++)
        {
            size_t start = k * mem_chunk;
            if (start < nbytes)
            {
                size_t chunk = std::min(nbytes - start, mem_chunk);
                // void* arithmetic is illegal it is why I am using this
                unsigned char *pdest =
                    static_cast<unsigned char*>(destination) + start;
                const unsigned char *psrc =
                    static_cast<const unsigned char*>(source) + start;
                memcpy(pdest, psrc, chunk);
            }
        }
    }
}

