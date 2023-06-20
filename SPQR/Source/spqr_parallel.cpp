// =============================================================================
// === spqr_parallel ===========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Factorize all the tasks in parallel with TBB.
// The GPU is not used.

#if 0 /* ifdef HAVE_TBB (TBB removed) */
#include "spqr.hpp"
#include <tbb/task_scheduler_init.h>
#include <tbb/task.h>

using namespace tbb ;

// =============================================================================
// === spqr_zippy ==============================================================
// =============================================================================

template <typename Entry, typename Int> class spqr_zippy: public task
{
  public:

    // -------------------------------------------------------------------------
    // spqr_zippy state
    // -------------------------------------------------------------------------

    const Int id ;
    spqr_blob <Entry> *Blob ;

    // -------------------------------------------------------------------------
    // spqr_zippy constructor
    // -------------------------------------------------------------------------

    spqr_zippy (Int id_, spqr_blob <Entry> *Blob_) : id (id_), Blob (Blob_) { }

    // -------------------------------------------------------------------------
    // spqr_zippy task
    // -------------------------------------------------------------------------

    task* execute ( )
    {

        // ---------------------------------------------------------------------
        // spawn my children
        // ---------------------------------------------------------------------

        Int *TaskChildp = Blob->QRsym->TaskChildp ;
        Int *TaskChild  = Blob->QRsym->TaskChild ;
        Int pfirst = TaskChildp [id] ;
        Int plast  = TaskChildp [id+1] ;
        Int nchildren = plast - pfirst ;

        if (nchildren > 0)
        {
            // create a list of TBB tasks, one for each child
            task_list TasksToDo ;
            for (Int i = 0 ; i < nchildren ; i++)
            {
                Int child = TaskChild [pfirst+i] ;
                TasksToDo.push_back (*new (allocate_child ( ))
                    spqr_zippy (child, Blob)) ;
            }
            // spawn all children and wait for all of them to finish
            set_ref_count (nchildren + 1) ;
            spawn_and_wait_for_all (TasksToDo) ;
        }

        // ---------------------------------------------------------------------
        // chilren are done, do my own task
        // ---------------------------------------------------------------------

        spqr_kernel (id, Blob) ;

        return (NULL) ;
    }
} ;


// =============================================================================
// === spqr_parallel ===========================================================
// =============================================================================

template <typename Entry, typename Int> void spqr_parallel
(
    Int ntasks,
    int nthreads,
    spqr_blob <Entry, Int> *Blob
)
{
    // fire up TBB on the task tree, starting at the root id = ntasks-1
    task_scheduler_init
        init (nthreads <= 0 ? (task_scheduler_init::automatic) : nthreads) ;
    spqr_zippy <Entry> & a = *new (task::allocate_root ( ))
        spqr_zippy <Entry, Int> (ntasks-1, Blob) ;
    task::spawn_root_and_wait (a) ;
}
template void spqr_parallel <double, int32_t>
(
    int32_t ntasks,
    int nthreads,
    spqr_blob <double, int32_t> *Blob
) ;
template void spqr_parallel <Complex, int32_t>
(
    int32_t ntasks,
    int nthreads,
    spqr_blob <Complex, int32_t> *Blob
) ;
template void spqr_parallel <double, int64_t>
(
    int64_t ntasks,
    int nthreads,
    spqr_blob <double, int64_t> *Blob
) ;
template void spqr_parallel <Complex, int64_t>
(
    int64_t ntasks,
    int nthreads,
    spqr_blob <Complex, int64_t> *Blob
) ;
#endif
