//------------------------------------------------------------------------------
// CHOLMOD/Utility/cholmod_memdebug: memory debugging
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module.  Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

#ifndef NDEBUG

    #define CM_MEMTABLE_SIZE 10000
    static void    *memtable_p [CM_MEMTABLE_SIZE] ;
    static size_t   memtable_s [CM_MEMTABLE_SIZE] ;
    static int nmemtable = 0 ;

//------------------------------------------------------------------------------
// CM_memtable_dump: dump the memtable
//------------------------------------------------------------------------------

void CM_memtable_dump (void)
{
    printf ("\nmemtable dump: %d\n", nmemtable) ;
    for (int k = 0 ; k < nmemtable ; k++)
    {
        printf ("  %4d: %12p : %ld\n", k, memtable_p [k], memtable_s [k]) ;
    }
}

//------------------------------------------------------------------------------
// CM_memtable_n: return # of items in the memtable
//------------------------------------------------------------------------------

int CM_memtable_n (void)
{
    return (nmemtable) ;
}

//------------------------------------------------------------------------------
// CM_memtable_clear: clear the memtable
//------------------------------------------------------------------------------

void CM_memtable_clear (void)
{
    nmemtable = 0 ;
}

//------------------------------------------------------------------------------
// CM_memtable_add: add a pointer to the memtable of malloc'd blocks
//------------------------------------------------------------------------------

void CM_memtable_add (void *p, size_t size)
{
    if (p == NULL) return ;

    bool fail = false ;
    #ifdef CM_MEMDUMP
    printf ("memtable add %p size %ld\n", p, size) ;
    #endif

    int n = nmemtable ;
    fail = (n > CM_MEMTABLE_SIZE) ;
    if (!fail)
    {
        for (int i = 0 ; i < n ; i++)
        {
            if (p == memtable_p [i])
            {
                printf ("\nadd duplicate %p size %ld\n", p, size) ;
                CM_memtable_dump ( ) ;
                fail = true ;
                break ;
            }
        }
    }
    if (!fail && p != NULL)
    {
        memtable_p [n] = p ;
        memtable_s [n] = size ;
        nmemtable++ ;
    }

    ASSERT (!fail) ;
    #ifdef CM_MEMDUMP
    CM_memtable_dump ( ) ;
    #endif
}

//------------------------------------------------------------------------------
// CM_memtable_size: get the size of a malloc'd block
//------------------------------------------------------------------------------

size_t CM_memtable_size (void *p)
{
    size_t size = 0 ;

    if (p == NULL) return (0) ;
    bool found = false ;

    int n = nmemtable ;
    for (int i = 0 ; i < n ; i++)
    {
        if (p == memtable_p [i])
        {
            size = memtable_s [i] ;
            found = true ;
            break ;
        }
    }

    if (!found)
    {
        printf ("\nFAIL: %p not found\n", p) ;
        CM_memtable_dump ( ) ;
        ASSERT (0) ;
    }
    return (size) ;
}

//------------------------------------------------------------------------------
// CM_memtable_find: test if a malloc'd block is in the table
//------------------------------------------------------------------------------

bool CM_memtable_find (void *p)
{
    bool found = false ;
    if (p == NULL) return (false) ;
    int n = nmemtable ;
    for (int i = 0 ; i < n ; i++)
    {
        if (p == memtable_p [i])
        {
            found = true ;
            break ;
        }
    }
    return (found) ;
}

//------------------------------------------------------------------------------
// CM_memtable_remove: remove a pointer from the table of malloc'd blocks
//------------------------------------------------------------------------------

void CM_memtable_remove (void *p)
{
    if (p == NULL) return ;

    bool found = false ;
    #ifdef CM_MEMDUMP
    printf ("memtable remove %p ", p) ;
    #endif

    int n = nmemtable ;
    for (int i = 0 ; i < n ; i++)
    {
        if (p == memtable_p [i])
        {
            // found p in the table; remove it
            memtable_p [i] = memtable_p [n-1] ;
            memtable_s [i] = memtable_s [n-1] ;
            nmemtable -- ;
            found = true ;
            break ;
        }
    }

    if (!found)
    {
        printf ("remove %p NOT FOUND\n", p) ;
        CM_memtable_dump ( ) ;
    }
    ASSERT (found) ;
    #ifdef CM_MEMDUMP
    CM_memtable_dump ( ) ;
    #endif
}

#endif

