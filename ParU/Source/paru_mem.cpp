////////////////////////////////////////////////////////////////////////////////
////////////////////////// paru_mem.cpp ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief  Wrappers for managing memory
 *  allocating and freeing is done through SuiteSparse and these wrappers
 *
 * @author Aznaveh
 *
 */

#include "paru_internal.hpp"

////////////////////////////////////////////////////////////////////////////////
// methods for testing memory allocation
////////////////////////////////////////////////////////////////////////////////

#ifdef MATLAB_MEX_FILE
    #undef printf
    #define ABORT { mexErrMsgIdAndTxt ("ParU:abort", \
        "abort %s %d", __FILE__, __LINE__) ; }
#else
    #define ABORT { printf ("line %d abort:\n", __LINE__) ; \
        fflush (stdout) ; fflush (stderr) ; abort ( ) ; }
#endif
#define MEMASSERT(ok) { if (!(ok)) ABORT ; }

#ifdef PARU_ALLOC_TESTING

// global variables for testing only
bool paru_malloc_tracking = false;
int64_t paru_nmalloc = 0;

#ifdef PARU_MEMTABLE_TESTING

typedef struct
{
    #define PARU_MEMTABLE_SIZE 1000000
    bool malloc_tracking ;          // true if allocations are being tracked
    int64_t nmalloc ;               // number of blocks allocated but not freed
    int8_t *memtable_p [PARU_MEMTABLE_SIZE] ;
    size_t  memtable_s [PARU_MEMTABLE_SIZE] ;
    int nmemtable ;
}
paru_Global_struct ;

static paru_Global_struct paru_Global =
{
    // malloc tracking, for testing, statistics, and debugging only
    .malloc_tracking = false,
    .nmalloc = 0,                // memory block counter
    .nmemtable = 0,             // memtable is empty
} ;

//------------------------------------------------------------------------------
// malloc debuging
//------------------------------------------------------------------------------

// These functions keep a separate record of the pointers to all allocated
// blocks of memory and their sizes, just for sanity checks.

void paru_memtable_dump (void)
{
    #pragma omp critical(paru_memdump)
    {
        printf ("\nmemtable dump: %d nmalloc " LD "\n",    // MEMDUMP
            paru_Global.nmemtable, paru_Global.nmalloc) ;
        for (int k = 0 ; k < paru_Global.nmemtable ; k++)
        {
            printf ("  %4d: %12p : %ld\n", k,               // MEMDUMP
                paru_Global.memtable_p [k],
                paru_Global.memtable_s [k]) ;
        }
    }
}

int paru_memtable_n (void)
{
    return (paru_Global.nmemtable) ;
}

void paru_memtable_clear (void)
{
    paru_Global.nmemtable = 0 ;
}

// add a pointer to the table of malloc'd blocks
void paru_memtable_add (void *p, size_t size)
{
    if (p == NULL) return ;
    if (paru_Global.malloc_tracking)
    {
        #pragma omp atomic
        paru_Global.nmalloc++ ;
    }

    bool fail = false ;
    #pragma omp critical(paru_memtable)
    {
        #ifdef PARU_MEMDUMP
        #pragma omp critical(paru_memdump)
        {
            printf ("memtable add %p size %ld\n", p, size) ;    // MEMDUMP
        }
        #endif
        int n = paru_Global.nmemtable ;
        fail = (n > PARU_MEMTABLE_SIZE) ;
        if (!fail)
        {
            for (int i = 0 ; i < n ; i++)
            {
                if (p == (void *) paru_Global.memtable_p [i])
                {
                    printf ("\nadd duplicate %p size %ld\n",    // MEMDUMP
                        p, size) ;
                    paru_memtable_dump ( ) ;
                    fail = true ;
                    break ;
                }
            }
        }
        if (!fail && p != NULL)
        {
            paru_Global.memtable_p [n] = (int8_t *) p ;
            paru_Global.memtable_s [n] = size ;
            paru_Global.nmemtable++ ;
        }
        #ifdef PARU_MEMDUMP
        paru_memtable_dump ( ) ;
        #endif
    }
    MEMASSERT (!fail) ;
}

// get the size of a malloc'd block
size_t paru_memtable_size (void *p)
{
    size_t size = 0 ;

    if (p == NULL) return (0) ;
    bool found = false ;
    #pragma omp critical(paru_memtable)
    {
        int n = paru_Global.nmemtable ;
        for (int i = 0 ; i < n ; i++)
        {
            if (p == paru_Global.memtable_p [i])
            {
                size = paru_Global.memtable_s [i] ;
                found = true ;
                break ;
            }
        }
    }
    if (!found)
    {
        printf ("\nFAIL: %p not found\n", p) ;      // MEMDUMP
        paru_memtable_dump ( ) ;
        ABORT ;
    }

    return (size) ;
}

// test if a malloc'd block is in the table
bool paru_memtable_find (void *p)
{
    bool found = false ;

    if (p == NULL) return (false) ;
    #pragma omp critical(paru_memtable)
    {
        int n = paru_Global.nmemtable ;
        for (int i = 0 ; i < n ; i++)
        {
            if (p == (void *) paru_Global.memtable_p [i])
            {
                found = true ;
                break ;
            }
        }
    }

    return (found) ;
}

// remove a pointer from the table of malloc'd blocks
void paru_memtable_remove (void *p)
{
    if (p == NULL) return ;
    if (paru_Global.malloc_tracking)
    {
        #pragma omp atomic
        paru_Global.nmalloc-- ;
    }

    bool found = false ;
    #pragma omp critical(paru_memtable)
    {
        #ifdef PARU_MEMDUMP
        #pragma omp critical(paru_memdump)
        {
            printf ("memtable remove %p ", p) ;             // MEMDUMP
        }
        #endif
        int n = paru_Global.nmemtable ;
        for (int i = 0 ; i < n ; i++)
        {
            if (p == (void *) paru_Global.memtable_p [i])
            {
                // found p in the table; remove it
                paru_Global.memtable_p [i] = paru_Global.memtable_p [n-1] ;
                paru_Global.memtable_s [i] = paru_Global.memtable_s [n-1] ;
                paru_Global.nmemtable -- ;
                found = true ;
                break ;
            }
        }
        #ifdef PARU_MEMDUMP
        paru_memtable_dump ( ) ;
        #endif
    }
    if (!found)
    {
        printf ("remove %p NOT FOUND\n", p) ;       // MEMDUMP
        paru_memtable_dump ( ) ;
    }
    MEMASSERT (found) ;
}

#endif

//------------------------------------------------------------------------------

bool paru_get_malloc_tracking(void)
{
    bool track;
    #pragma omp critical (paru_malloc_testing)
    {
        track = paru_malloc_tracking;
    }
    return (track);
}

void paru_set_malloc_tracking(bool track)
{
    #pragma omp critical (paru_malloc_testing)
    {
        paru_malloc_tracking = track;
    }
}

void paru_set_nmalloc(int64_t nmalloc)
{
    #pragma omp critical (paru_malloc_testing)
    {
        paru_nmalloc = nmalloc;
    }
}

int64_t paru_decr_nmalloc(void)
{
    int64_t nmalloc = 0;
    #pragma omp critical (paru_malloc_testing)
    {
        if (paru_nmalloc > 0)
        {
            nmalloc = paru_nmalloc--;
        }
    }
    return (nmalloc);
}

int64_t paru_get_nmalloc(void)
{
    int64_t nmalloc = 0;
    #pragma omp critical (paru_malloc_testing)
    {
        nmalloc = paru_nmalloc;
    }
    return (nmalloc);
}

#endif

////////////////////////////////////////////////////////////////////////////////
// wrappers for malloc, calloc, realloc, and free
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// paru_malloc: wrapper around malloc routine
//------------------------------------------------------------------------------

#ifdef PARU_MALLOC_DEBUG
void *paru_malloc_debug
(
    size_t n,
    size_t size,
    const char *filename,
    int line
)
{
    void *p = NULL ;
    #pragma omp critical(paru_memdebug)
    {
        printf ("paru_malloc: n %zu size %zu file: %s line: %d\n",
            n, size, filename, line) ;
        p = paru_malloc (n, size) ;
        printf ("paru_malloc: got %p file: %s line: %d\n",
            p, filename, line) ;
    }
    return (p) ;
}
#endif

void *paru_malloc(size_t n, size_t size)
{
    DEBUGLEVEL(0);
    void *p = NULL;
    if (size == 0)
    {
        PRLEVEL(1, ("ParU: size must be > 0\n"));
        return NULL;
    }
    else if (n >= (Size_max / size) || n >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
        p = NULL;
    }
    else
    {

        #ifdef PARU_ALLOC_TESTING
        {
            // brutal memory testing only
            if (paru_get_malloc_tracking())
            {
                int64_t nmalloc = paru_decr_nmalloc();
                if (nmalloc > 0)
                {
                    p = SuiteSparse_malloc(n, size);
                }
            }
            else
            {
                p = SuiteSparse_malloc(n, size);
            }
        }
        #else
        {
            // in production
            p = SuiteSparse_malloc(n, size);
        }
        #endif

        if (p == NULL)
        {
            // out of memory
            PRLEVEL(1, ("ParU: out of memory\n"));
        }
        else
        {
            #if defined ( PARU_ALLOC_TESTING ) && defined ( PARU_MEMTABLE_TESTING )
            paru_memtable_add (p, n * size) ;
            #endif
        }
    }
    return p;
}

//------------------------------------------------------------------------------
// paru_calloc: wrapper around calloc routine
//------------------------------------------------------------------------------

#ifdef PARU_MALLOC_DEBUG
void *paru_calloc_debug
(
    size_t n,
    size_t size,
    const char *filename,
    int line
)
{
    void *p = NULL ;
    #pragma omp critical(paru_memdebug)
    {
        printf ("paru_calloc: n %zu size %zu file: %s line: %d\n",
            n, size, filename, line) ;
        p = paru_calloc (n, size) ;
        printf ("paru_calloc: got %p file: %s line: %d\n",
            p, filename, line) ;
    }
    return (p) ;
}
#endif

void *paru_calloc(size_t n, size_t size)
{
    DEBUGLEVEL(0);
    void *p = NULL;
    if (size == 0)
    {
        PRLEVEL(1, ("ParU: size must be > 0\n"));
    }
    else if (n >= (Size_max / size) || n >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
    }
    else
    {

        #ifdef PARU_ALLOC_TESTING
        {
            // brutal memory testing only
            if (paru_get_malloc_tracking())
            {
                int64_t nmalloc = paru_decr_nmalloc();
                if (nmalloc > 0)
                {
                    p = SuiteSparse_calloc(n, size);
                }
            }
            else
            {
                p = SuiteSparse_calloc(n, size);
            }
        }
        #else
        {
            // in production
            p = SuiteSparse_calloc(n, size);
        }
        #endif

        if (p == NULL)
        {
            // out of memory
            PRLEVEL(1, ("ParU: out of memory\n"));
        }
        else
        {
            #if defined ( PARU_ALLOC_TESTING ) && defined ( PARU_MEMTABLE_TESTING )
            paru_memtable_add (p, n * size) ;
            #endif
        }
    }
    return p;
}

//------------------------------------------------------------------------------
// paru_realloc: wrapper around realloc routine
//------------------------------------------------------------------------------

#ifdef PARU_MALLOC_DEBUG
void *paru_realloc_debug
(
    size_t nnew,
    size_t size_Entry,
    void *p,
    size_t *n,
    const char *filename,
    int line
)
{
    void *pnew = NULL ;
    #pragma omp critical(paru_memdebug)
    {
        printf ("paru_realloc: nnew %zu size %zu p %p n %zu file: %s line: %d\n",
            nnew, size_Entry, p, *n, filename, line) ;
        pnew = paru_realloc (nnew, size_Entry, p, n) ;
        printf ("paru_realloc: got %p file: %s line: %d\n",
            pnew, filename, line) ;
    }
    return (pnew) ;
}
#endif

void *paru_realloc
(
    size_t nnew,     // requested # of items
    size_t size_Entry,  // size of each Entry
    void *p,         // block memory to realloc
    size_t *n        // current size on input, nnew output if successful
)
{
    DEBUGLEVEL(0);
    void *pnew;
    if (size_Entry == 0)
    {
        PRLEVEL(1, ("ParU: sizeof(entry)  must be > 0\n"));
        p = NULL;
    }
    else if (p == NULL)
    {
        // A new alloc
        p = paru_malloc (nnew, size_Entry) ;
        *n = (p == NULL) ? 0 : nnew;
    }
    else if (nnew == *n )
    {
        PRLEVEL(1, ("%% reallocating nothing " LD ", " LD " in %p \n", nnew, *n,
                    p));
    }
    else if (nnew >= (Size_max / size_Entry) || nnew >= INT_MAX)
    {
        // object is too big to allocate without causing integer overflow
        PRLEVEL(1, ("ParU: problem too large\n"));
    }
    else
    {
        // The object exists, and is changing to some other nonzero size.
        PRLEVEL(1, ("realloc : " LD " to " LD ", " LD "\n", *n, nnew, size_Entry));
        int ok = TRUE;

        #ifdef PARU_ALLOC_TESTING
        {
            // brutal memory testing only
            if (paru_get_malloc_tracking())
            {
                int64_t nmalloc = paru_decr_nmalloc();
                if (nmalloc > 0)
                {
                    pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
                }
                else
                {
                    // pretend to fail
                    ok = FALSE;
                }
            }
            else
            {
                pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
            }
        }
        #else
        {
            // in production
            pnew = SuiteSparse_realloc(nnew, *n, size_Entry, p, &ok);
        }
        #endif

        if (ok)
        {
            #if defined ( PARU_ALLOC_TESTING ) && defined ( PARU_MEMTABLE_TESTING )
            paru_memtable_remove (p) ;
            paru_memtable_add (pnew, nnew*size_Entry) ;
            #endif
            p = pnew ;
            *n = nnew;
        }
    }
    return p;
}

//------------------------------------------------------------------------------
// paru_free: Wrapper around free routine
//------------------------------------------------------------------------------

#ifdef PARU_MALLOC_DEBUG
void paru_free_debug
(
    size_t n,
    size_t size,
    void *p,
    const char *filename,
    int line
)
{
    #pragma omp critical(paru_memdebug)
    {
        printf ("paru_free: n %zu size %zu p %p file: %s line: %d\n",
            n, size, p, filename, line) ;
        paru_free (n, size, p) ;
    }
}
#endif

void paru_free(size_t n, size_t size, void *p)
{
    DEBUGLEVEL(0);
    if (p != NULL)
    {
        SuiteSparse_free (p) ;
        #if defined ( PARU_ALLOC_TESTING ) && defined ( PARU_MEMTABLE_TESTING )
        paru_memtable_remove (p) ;
        #endif
    }
    else
    {
        PRLEVEL(1, ("%% freeing a NULL pointer  \n"));
    }
}

//------------------------------------------------------------------------------
// new/delete: a wrapper for paru_malloc and paru_free
//------------------------------------------------------------------------------

//  Global replacement of new and delete

void *operator new(size_t size)
{
    // no inline, required by [replacement.functions]/3
    DEBUGLEVEL(0);

    if (size == 0)
    {
        ++size;  // make sure at least one byte is allocated
    }

    #if defined ( PARU_MALLOC_DEBUG )
    void *ptr = paru_malloc_debug (1, size, __FILE__, __LINE__) ;
    #else
    void *ptr = paru_malloc (1, size) ;
    #endif

    if (ptr != nullptr)
    {
        return ptr;
    }

    throw std::bad_alloc{};
}

void operator delete(void *ptr) noexcept
{
    DEBUGLEVEL(0);

    #if defined ( PARU_MALLOC_DEBUG )
    paru_free_debug (0, 0, ptr, __FILE__, __LINE__) ;
    #else
    paru_free (0, 0, ptr) ;
    #endif

}

//------------------------------------------------------------------------------
// paru_free_el: free element e from elementList
//------------------------------------------------------------------------------

// Free element e, created by paru_create_element

void paru_free_el(int64_t e, paru_element **elementList)
{
    DEBUGLEVEL(0);
    paru_element *el = elementList[e];
    if (el == NULL) return;

    int64_t nrows = el->nrows, ncols = el->ncols;
    size_t tot_size = sizeof(paru_element) +
                      sizeof(int64_t) * (2 * (nrows + ncols)) +
                      sizeof(double) * nrows * ncols;

    #if defined ( PARU_MALLOC_DEBUG )
    paru_free_debug (1, tot_size, elementList [e], __FILE__, __LINE__) ;
    #else
    paru_free (1, tot_size, elementList [e]) ;
    #endif
    elementList [e] = NULL ;

}

