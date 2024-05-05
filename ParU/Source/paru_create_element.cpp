////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_create_element  ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief Initializing an empty element
 *    if (Init) use calloc .i.e x = 0
 *
 *                    V RRRRRRRRRRRRRR
 *                      IIIIIIIIIIIIII            I = global index
 *                  V                             r = relative index
 *                  R I  xxxxxxxxxxxxxx           V = row/col valid bit
 *                  R I  xxxxxxxxxxxxxx            if V == f it is valid for
 *                  R I  xxxxxxxxxxxxxx             current front
 *                  R I  xxxxxxxxxxxxxx
 *                  R I  xxxxxxxxxxxxxx
 *
 * @author Aznaveh
 *  */
#include "paru_internal.hpp"
paru_element *paru_create_element(int64_t nrows, int64_t ncols)
{
    DEBUGLEVEL(0);

    PRLEVEL(1, ("%% creating " LD "x" LD " element ", nrows, ncols));
    paru_element *curEl;
    size_t tot_size = sizeof(paru_element) +
                      sizeof(int64_t) * (2 * (nrows + ncols)) +
                      sizeof(double) * nrows * ncols;

    curEl = static_cast<paru_element*>
        #if defined ( PARU_MALLOC_DEBUG )
        (paru_malloc_debug (1, tot_size, __FILE__, __LINE__)) ;
        #else
        (paru_malloc (1, tot_size)) ;
        #endif

    if (curEl == NULL) return NULL;  // do not do error checking

    PRLEVEL(1, (" with size of " LD " in %p\n", tot_size, curEl));

    // Initializing current element
    curEl->nrowsleft = curEl->nrows = nrows;
    curEl->ncolsleft = curEl->ncols = ncols;
    curEl->rValid = -1;
    curEl->cValid = -1;

    curEl->lac = 0;
    return curEl;
}
