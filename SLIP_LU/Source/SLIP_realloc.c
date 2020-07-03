//------------------------------------------------------------------------------
// SLIP_LU/SLIP_realloc: wrapper for realloc
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

#include "slip_internal.h"

// SLIP_realloc is a wrapper for realloc.  If p is non-NULL on input, it points
// to a previously allocated object of size nitems_old * size_of_item.  The
// object is reallocated to be of size nitems_new * size_of_item.  If p is NULL
// on input, then a new object of that size is allocated.  On success, a
// pointer to the new object is returned.  If the reallocation fails, p is not
// modified, and a flag is returned to indicate that the reallocation failed.
// If the size decreases or remains the same, then the method always succeeds
// (ok is returned as true).

// Typical usage:  the following code fragment allocates an array of 10 int's,
// and then increases the size of the array to 20 int's.  If the SLIP_malloc
// succeeds but the SLIP_realloc fails, then the array remains unmodified,
// of size 10.
//
//      int *p ;
//      p = SLIP_malloc (10 * sizeof (int)) ;
//      if (p == NULL) { error here ... }
//      printf ("p points to an array of size 10 * sizeof (int)\n") ;
//      bool ok ;
//      p = SLIP_realloc (20, 10, sizeof (int), p, &ok) ;
//      if (ok) printf ("p has size 20 * sizeof (int)\n") ;
//      else printf ("realloc failed; p still has size 10 * sizeof (int)\n") ;
//      SLIP_FREE (p) ;

void *SLIP_realloc      // pointer to reallocated block, or original block
                        // if the realloc failed
(
    int64_t nitems_new,     // new number of items in the object
    int64_t nitems_old,     // old number of items in the object
    size_t size_of_item,    // sizeof each item
    void *p,                // old object to reallocate
    bool *ok                // true if success, false on failure
)
{
    if (!slip_initialized ( ))
    {
        (*ok) = false ;
        return (p) ;
    }

    int result ;
    void *pnew = SuiteSparse_realloc ((size_t) nitems_new, (size_t) nitems_old,
        size_of_item, p, &result) ;
    (*ok) = (result != 0) ;
    return (pnew) ;
}

