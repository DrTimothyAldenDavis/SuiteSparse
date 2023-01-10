//------------------------------------------------------------------------------
// UMFPACK/Source/umf_mem_init_memoryspace: initialize Numeric->Memory
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/* The UMF_mem_* routines manage the Numeric->Memory memory space. */

#include "umf_internal.h"
#include "umf_mem_init_memoryspace.h"

/* initialize the LU and element workspace (Numeric->Memory) */

void UMF_mem_init_memoryspace
(
    NumericType *Numeric
)
{
    Unit *p ;

    ASSERT (Numeric != (NumericType *) NULL) ;
    ASSERT (Numeric->Memory != (Unit *) NULL) ;
    ASSERT (Numeric->size >= 3) ;
    DEBUG0 (("Init memory space, size "ID"\n", Numeric->size)) ;

    Numeric->ngarbage = 0 ;
    Numeric->nrealloc = 0 ;
    Numeric->ncostly = 0 ;
    Numeric->ibig = EMPTY ;
    Numeric->ihead = 0 ;
    Numeric->itail = Numeric->size ;

#ifndef NDEBUG
    UMF_allocfail = FALSE ;
#endif

    /* allocate the 2-unit tail marker block and initialize it */
    Numeric->itail -= 2 ;
    p = Numeric->Memory + Numeric->itail ;
    DEBUG2 (("p "ID" tail "ID"\n", (Int) (p-Numeric->Memory), Numeric->itail)) ;
    Numeric->tail_usage = 2 ;
    p->header.prevsize = 0 ;
    p->header.size = 1 ;

    /* allocate a 1-unit head marker block at the head of memory */
    /* this is done so that an offset of zero is treated as a NULL pointer */
    Numeric->ihead++ ;

    /* initial usage in Numeric->Memory */
    Numeric->max_usage = 3 ;
    Numeric->init_usage = Numeric->max_usage ;

    /* Note that UMFPACK_*symbolic ensures that Numeric->Memory is of size */
    /* at least 3, so this initialization will always succeed. */

#ifndef NDEBUG
    DEBUG2 (("init_memoryspace, all free (except one unit at head\n")) ;
    UMF_dump_memory (Numeric) ;
#endif

}
