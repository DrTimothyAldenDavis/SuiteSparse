////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_memset  ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*!  @brief  wrapper around memset
 *
 * @author Aznaveh
 */
#include <algorithm>

#include "paru_internal.hpp"

void paru_memset
(
    void *ptr,
    int64_t value,
    size_t nbytes,
    size_t mem_chunk,
    int32_t nthreads
)
{

    if (nbytes < mem_chunk)
    {
        // single task memset
        memset(ptr, value, nbytes);
    }
    else
    {
        // multiple task memset
        size_t nchunks = 1 + (nbytes / mem_chunk);
        if (static_cast<size_t>(nthreads) > nchunks)
        {
            nthreads = static_cast<int>(nchunks);
        }

        int64_t k;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
        for (k = 0; k < static_cast<int64_t>(nchunks); k++)
        {
            size_t start = k * mem_chunk;
            if (start < nbytes)
            {
                size_t chunk = std::min(nbytes - start, mem_chunk);
                // void* arithmetic is illegal it is why I am using this
                unsigned char *ptr_chunk =
                    static_cast<unsigned char*>(ptr) + start;
                memset(ptr_chunk, value, chunk);
            }
        }
    }
}

