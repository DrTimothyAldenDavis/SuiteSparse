// =============================================================================
// === spqr_factorize ==========================================================
// =============================================================================

// Given an m-by-n sparse matrix A and its symbolic analsys, compute its
// numeric QR factorization.
//
// Note that the first part of the code allocates workspace, and the resulting
// QR factors.  It then does all its work in that amount of space, with no
// reallocations.  The size of the memory blocks that are allocated are known
// a priori, from the symbolic analysis, and do not depend upon the specific
// numerical values.  When done, it frees its workspace.
//
// Furthermore, if rank detection is disabled (with tol < 0), the actual flops
// become perfectly repeatable; the code does the same thing regardless of
// the actual numerical values.
//
// These characteristics make this QR factorization a candidate for a
// hardware-in-the-loop solution, for control applications (for example).
// However, the Stacks must not be shrunk when the factorization is done.
//
// FUTURE: split this function into three parts:
//     (a) allocate workspace and resulting QR factors
//     (b) factorize (no malloc/free)
//     (c) free workspace and optionally shrink the Stacks
// Then part (b) could be called many times with different values but same
// nonzero pattern.

// =============================================================================
// === macros ==================================================================
// =============================================================================

#include "spqr.hpp"

#define FCHUNK 32        // FUTURE: make a parameter; Householder block size

// Free Sx, A, and all of Work except the Work [0:ns-1] array itself
#define FREE_WORK_PART1 \
{ \
    free_Work <Entry> (Work, ns, n, maxfn, wtsize, cc) ; \
    if (freeA) cholmod_l_free_sparse (Ahandle, cc) ; \
    cholmod_l_free (anz, sizeof (Entry), Sx, cc) ; \
    Sx = NULL ; \
}

// Free Cblock and the Work array itself
#define FREE_WORK_PART2 \
{ \
    cholmod_l_free (ns, sizeof (spqr_work <Entry>), Work, cc) ; \
    Work = NULL ; \
    cholmod_l_free (nf+1, sizeof (Entry *), Cblock, cc) ; \
    Cblock = NULL ; \
}

// Free all workspace
#define FREE_WORK \
{ \
    FREE_WORK_PART1 ; \
    FREE_WORK_PART2 ; \
}


// =============================================================================
// === get_Work ================================================================
// =============================================================================

// Allocate the Work object, which is an array of structs.  The entry Work [s]
// contains workspace for the Stack s, for each Stack 0 to ns-1.

template <typename Entry> spqr_work <Entry> *get_Work
(
    Int ns,             // number of stacks
    Int n,              // number of columns of A
    Int maxfn,          // largest number of columns in any front
    Int keepH,          // if true, H is kept
    Int fchunk,
    Int *p_wtsize,      // size of WTwork for each 
    cholmod_common *cc
)
{
    int ok = TRUE ;
    spqr_work <Entry> *Work ;
    Int wtsize ;
    *p_wtsize = 0 ;

    // wtsize = (fchunk + (keepH ? 0:1)) * maxfn ;
    wtsize = spqr_mult (fchunk + (keepH ? 0:1), maxfn, &ok) ;

    Work = (spqr_work <Entry> *)    
        cholmod_l_malloc (ns, sizeof (spqr_work <Entry>), cc) ;

    if (!ok || cc->status < CHOLMOD_OK)
    {
        // out of memory or Int overflow
        cholmod_l_free (ns, sizeof (spqr_work <Entry>), Work, cc) ;
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        return (NULL) ;
    }

    for (Int stack = 0 ; stack < ns ; stack++)
    {
        Work [stack].Fmap = (Int *) cholmod_l_malloc (n, sizeof (Int), cc) ;
        Work [stack].Cmap = (Int *) cholmod_l_malloc (maxfn, sizeof (Int), cc) ;
        if (keepH)
        {
            // Staircase is a permanent part of H
            Work [stack].Stair1 = NULL ;
        }
        else
        {
            // Staircase workspace reused for each front
            Work [stack].Stair1 =
                (Int *) cholmod_l_malloc (maxfn, sizeof (Int), cc) ;
        }
        Work [stack].WTwork =
            (Entry *) cholmod_l_malloc (wtsize, sizeof (Entry), cc) ;
        Work [stack].sumfrank = 0 ;
        Work [stack].maxfrank = 0 ;
    }

    *p_wtsize = wtsize ;
    return (Work) ;
}


// =============================================================================
// === free_Work ===============================================================
// =============================================================================

// Free the contents of Work, but not the Work array itself

template <typename Entry> void free_Work
(
    spqr_work <Entry> *Work,
    Int ns,             // number of stacks
    Int n,              // number of columns of A
    Int maxfn,          // largest number of columns in any front
    Int wtsize,         // size of WTwork array for each Stack
    cholmod_common *cc
)
{
    if (Work != NULL)
    {
        for (Int stack = 0 ; stack < ns ; stack++)
        {
            cholmod_l_free (n,      sizeof (Int),   Work [stack].Fmap,   cc) ;
            cholmod_l_free (maxfn,  sizeof (Int),   Work [stack].Cmap,   cc) ;
            cholmod_l_free (maxfn,  sizeof (Int),   Work [stack].Stair1, cc) ;
            cholmod_l_free (wtsize, sizeof (Entry), Work [stack].WTwork, cc) ;
            Work [stack].Fmap = NULL ;
            Work [stack].Cmap = NULL ;
            Work [stack].Stair1 = NULL ;
            Work [stack].WTwork = NULL ;
        }
    }
}


// =============================================================================
// === spqr_factorize ==========================================================
// =============================================================================

template <typename Entry> spqr_numeric <Entry> *spqr_factorize
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Int freeA,                      // if TRUE, free A on output
    double tol,                     // for rank detection
    Int ntol,                       // apply tol only to first ntol columns
    spqr_symbolic *QRsym,

    // workspace and parameters
    cholmod_common *cc
)
{
    Int *Wi, *Qfill, *PLinv, *Cm, *Sp, *Stack_size,
        *TaskFront, *TaskFrontp, *TaskStack, *Stack_maxstack ;
    Entry *Sx, **Rblock, **Cblock, **Stacks ;
    spqr_numeric <Entry> *QRnum ;
    Int nf, m, n, anz, fchunk, maxfn, rank, maxfrank, rjsize, rank1,
        maxstack,j, wtsize, stack, ns, ntasks, keepH, hisize ;
    char *Rdead ;
    cholmod_sparse *A ;
    spqr_work <Entry> *Work ;

    // -------------------------------------------------------------------------
    // get inputs and contents of symbolic object
    // -------------------------------------------------------------------------

    if (QRsym == NULL)
    {
        // out of memory in caller
        if (freeA)
        {
            // if freeA is true, A must always be freed, even on error
            cholmod_l_free_sparse (Ahandle, cc) ;
        }
        return (NULL) ;
    }

    A = *Ahandle ;

    nf = QRsym->nf ;                // number of frontal matrices
    m = QRsym->m ;                  // A is m-by-n
    n = QRsym->n ;
    anz = QRsym->anz ;              // nnz (A)

    keepH = QRsym->keepH ;

    rjsize = QRsym->rjsize ;

    Sp = QRsym->Sp ;                // size m+1, row pointers for S
    Qfill = QRsym->Qfill ;          // fill-reducing ordering
    PLinv = QRsym->PLinv ;          // size m, leftmost column sort

    ns = QRsym->ns ;                // number of stacks
    ntasks = QRsym->ntasks ;        // number of tasks

    // FUTURE: compute a unique maxfn for each stack.  Current maxfn is OK, but
    // it's a global max of the fn of all fronts, and need only be max fn of
    // the fronts in any given stack.

    maxfn  = QRsym->maxfn ;         // max # of columns in any front
    ASSERT (maxfn <= n) ;
    hisize = QRsym->hisize ;        // # of integers in Hii, Householder vectors

    TaskFrontp = QRsym->TaskFrontp ;
    TaskFront  = QRsym->TaskFront ;
    TaskStack  = QRsym->TaskStack ;

    maxstack = QRsym->maxstack ;
    Stack_maxstack = QRsym->Stack_maxstack ;

    if (!(QRsym->do_rank_detection))
    {
        // disable rank detection if not accounted for in analysis
        tol = -1 ;
    }

    // If there is one task, there is only one stack, and visa versa
    ASSERT ((ns == 1) == (ntasks == 1)) ;

    PR (("factorize with ns %ld ntasks %ld\n", ns, ntasks)) ;

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    cholmod_l_allocate_work (0, MAX (m,nf), 0, cc) ;

    // shared Int workspace
    Wi = (Int *) cc->Iwork ;    // size m, aliased with the rest of Iwork
    Cm = Wi ;                   // size nf

    // Cblock is workspace shared by all threads
    Cblock = (Entry **) cholmod_l_malloc (nf+1, sizeof (Entry *), cc) ;

    Work = NULL ;               // Work and its contents not yet allocated
    fchunk = MIN (m, FCHUNK) ;
    wtsize = 0 ;

    // -------------------------------------------------------------------------
    // create S
    // -------------------------------------------------------------------------

    // create numeric values of S = A(p,q) in row-form in Sx
    Sx = (Entry *) cholmod_l_malloc (anz, sizeof (Entry), cc) ;

    if (cc->status == CHOLMOD_OK)
    {
        // use Wi as workspace (Iwork (0:m-1)) [
        spqr_stranspose2 (A, Qfill, Sp, PLinv, Sx, Wi) ;
        // Wi no longer needed ]
    }

    PR (("status after creating Sx: %d\n", cc->status)) ;

    // -------------------------------------------------------------------------
    // input matrix A no longer needed; free it if the user doesn't need it
    // -------------------------------------------------------------------------

    if (freeA)
    {
        // this is done even if out of memory, above
        cholmod_l_free_sparse (Ahandle, cc) ;
        ASSERT (*Ahandle == NULL) ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // allocate numeric object
    // -------------------------------------------------------------------------

    QRnum = (spqr_numeric<Entry> *)
        cholmod_l_malloc (1, sizeof (spqr_numeric<Entry>), cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK ;
        return (NULL) ;
    }

    Rblock     = (Entry **) cholmod_l_malloc (nf, sizeof (Entry *), cc) ;
    Rdead      = (char *)   cholmod_l_calloc (n,  sizeof (char),    cc) ;

    // these may be revised (with ns=1) if we run out of memory
    Stacks     = (Entry **) cholmod_l_calloc (ns, sizeof (Entry *), cc) ;
    Stack_size = (Int *)    cholmod_l_calloc (ns, sizeof (Int),     cc) ;

    QRnum->Rblock     = Rblock ;
    QRnum->Rdead      = Rdead ;
    QRnum->Stacks     = Stacks ;
    QRnum->Stack_size = Stack_size ;

    if (keepH)
    {
        // allocate permanent space for Stair, Tau, Hii for each front
        QRnum->HStair= (Int *)   cholmod_l_malloc (rjsize, sizeof (Int),   cc) ;
        QRnum->HTau  = (Entry *) cholmod_l_malloc (rjsize, sizeof (Entry), cc) ;
        QRnum->Hii   = (Int *)   cholmod_l_malloc (hisize, sizeof (Int),   cc) ;
        QRnum->Hm    = (Int *)   cholmod_l_malloc (nf,     sizeof (Int),   cc) ;
        QRnum->Hr    = (Int *)   cholmod_l_malloc (nf,     sizeof (Int),   cc) ;
        QRnum->HPinv = (Int *)   cholmod_l_malloc (m,      sizeof (Int),   cc) ;
    }
    else
    {
        // H is not kept; this part of the numeric object is not used
        QRnum->HStair = NULL ;
        QRnum->HTau = NULL ;
        QRnum->Hii = NULL ;
        QRnum->Hm = NULL ;
        QRnum->Hr = NULL ;
        QRnum->HPinv = NULL ;
    }

    QRnum->n = n ;
    QRnum->m = m ;
    QRnum->nf = nf ;
    QRnum->rjsize = rjsize ;
    QRnum->hisize = hisize ;
    QRnum->keepH = keepH ;
    QRnum->maxstack = maxstack ;
    QRnum->ns = ns ;
    QRnum->ntasks = ntasks ;
    QRnum->maxfm = EMPTY ;      // max (Hm [0:nf-1]), computed only if H is kept

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freenum (&QRnum, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    Work = get_Work <Entry> (ns, n, maxfn, keepH, fchunk, &wtsize, cc) ;

    // -------------------------------------------------------------------------
    // allocate and initialize each Stack
    // -------------------------------------------------------------------------

    if (cc->status == CHOLMOD_OK)
    {
        for (stack = 0 ; stack < ns ; stack++)
        {
            Entry *Stack ;
            size_t stacksize = (ntasks == 1) ?
                maxstack : Stack_maxstack [stack] ;
            Stack_size [stack] = stacksize ;
            Stack = (Entry *) cholmod_l_malloc (stacksize, sizeof (Entry), cc) ;
            Stacks [stack] = Stack ;
            Work [stack].Stack_head = Stack ;
            Work [stack].Stack_top  = Stack + stacksize ;
        }
    }

    // -------------------------------------------------------------------------
    // punt to sequential case and fchunk = 1 if out of memory
    // -------------------------------------------------------------------------

    if (cc->status < CHOLMOD_OK)
    {
        // PUNT: ran out of memory; try again with smaller workspace
        // out of memory; free any stacks that were successfully allocated
        if (Stacks != NULL)
        {
            for (stack = 0 ; stack < ns ; stack++)
            {
                size_t stacksize = (ntasks == 1) ?
                    maxstack : Stack_maxstack [stack] ;
                cholmod_l_free (stacksize, sizeof (Entry), Stacks [stack], cc) ;
            }
        }
        cholmod_l_free (ns, sizeof (Entry *), Stacks,     cc) ;
        cholmod_l_free (ns, sizeof (Int),     Stack_size, cc) ;

        // free the contents of Work, and the Work array itself
        free_Work <Entry> (Work, ns, n, maxfn, wtsize, cc) ;
        cholmod_l_free (ns, sizeof (spqr_work <Entry>), Work, cc) ;

        // punt to a single stack, a single task, and fchunk of 1
        ns = 1 ;
        ntasks = 1 ;
        fchunk = 1 ;
        cc->status = CHOLMOD_OK ;
        Work = get_Work <Entry> (ns, n, maxfn, keepH, fchunk, &wtsize, cc) ;
        Stacks     = (Entry **) cholmod_l_calloc (ns, sizeof (Entry *), cc) ;
        Stack_size = (Int *)    cholmod_l_calloc (ns, sizeof (Int),     cc) ;
        QRnum->Stacks     = Stacks ;
        QRnum->Stack_size = Stack_size ;
        if (cc->status == CHOLMOD_OK)
        {
            Entry *Stack ;
            Stack_size [0] = maxstack ;
            Stack = (Entry *) cholmod_l_malloc (maxstack, sizeof (Entry), cc) ;
            Stacks [0] = Stack ;
            Work [0].Stack_head = Stack ;
            Work [0].Stack_top  = Stack + maxstack ;
        }
    }

    // actual # of stacks and tasks used
    QRnum->ns = ns ;
    QRnum->ntasks = ntasks ;

    // -------------------------------------------------------------------------
    // check if everything was allocated OK
    // -------------------------------------------------------------------------

    if (cc->status < CHOLMOD_OK)
    {
        spqr_freenum (&QRnum, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // At this point, the factorization is guaranteed to succeed, unless
    // sizeof (BLAS_INT) < sizeof (Int), in which case, you really should get
    // a 64-bit BLAS.

    // -------------------------------------------------------------------------
    // create the Blob : everything the numeric factorization kernel needs
    // -------------------------------------------------------------------------

    spqr_blob <Entry> Blob ;
    Blob.QRsym = QRsym ;
    Blob.QRnum = QRnum ;
    Blob.tol = tol ;
    Blob.Work = Work ;
    Blob.Cm = Cm ;
    Blob.Cblock = Cblock ;
    Blob.Sx = Sx ;
    Blob.ntol = ntol ;
    Blob.fchunk = fchunk ;
    Blob.cc = cc ;

    // -------------------------------------------------------------------------
    // initialize the "pure" flop count (for performance testing only)
    // -------------------------------------------------------------------------

#ifdef TIMING
    cc->other1 [0] = 0 ;
#endif

    // -------------------------------------------------------------------------
    // numeric QR factorization
    // -------------------------------------------------------------------------

    if (ntasks == 1)
    {
        // Just one task, with or without TBB installed: don't use TBB
        spqr_kernel (0, &Blob) ;        // sequential case
    }
    else
    {
#ifdef HAVE_TBB
        // parallel case: TBB is installed, and there is more than one task
        int nthreads = MAX (0, cc->SPQR_nthreads) ;
        spqr_parallel (ntasks, nthreads, &Blob) ;
#else
        // TBB not installed, but the work is still split into multiple tasks.
        // do tasks 0 to ntasks-2 (skip the placeholder root task id = ntasks-1)
        for (Int id = 0 ; id < ntasks-1 ; id++)
        {
            spqr_kernel (id, &Blob) ;
        }
#endif
    }

    // -------------------------------------------------------------------------
    // check for BLAS Int overflow
    // -------------------------------------------------------------------------

    if (CHECK_BLAS_INT && cc->status < CHOLMOD_OK)
    {
        // problem too large for the BLAS.  This can only occur if, for example
        // you're on a 64-bit platform (with sizeof (Int) = 8) and using a
        // 32-bit BLAS (with sizeof (BLAS_INT) = 4).  If sizeof (BLAS_INT) is
        // equal to sizeof (Int), then CHECK_BLAS_INT is FALSE at compile-time,
        // and this entire code is removed as dead code by the compiler.
        spqr_freenum (&QRnum, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // finalize the rank
    // -------------------------------------------------------------------------

    rank = 0 ;
    maxfrank = 1 ;
    for (stack = 0 ; stack < ns ; stack++)
    {
        rank += Work [stack].sumfrank ;
        maxfrank = MAX (maxfrank, Work [stack].maxfrank) ;
    }
    QRnum->rank = rank ;                    // required by spqr_hpinv
    QRnum->maxfrank = maxfrank ;
    PR (("m %ld n %ld my QR rank %ld\n", m, n, rank)) ;

    // -------------------------------------------------------------------------
    // free all workspace, except Cblock and Work
    // -------------------------------------------------------------------------

    FREE_WORK_PART1 ;

    // -------------------------------------------------------------------------
    // shrink the Stacks to hold just R (and H, if H kept)
    // -------------------------------------------------------------------------

    // If shrink is <= 0, then the Stacks are not modified.
    // If shrink is 1, each Stack is realloc'ed to the right size (default)
    // If shrink is > 1, then each Stack is forcibly moved and shrunk.
    // This option is mainly meant for testing, not production use.
    // It should be left at 1 for production use.

    Int any_moved = FALSE ;

    int shrink = cc->SPQR_shrink ;

    if (shrink > 0)
    {
        for (stack = 0 ; stack < ns ; stack++)
        {
            // stacksize is the current size of the this Stack
            size_t stacksize = Stack_size [stack] ;
            Entry *Stack = Stacks [stack] ;
            // Work [stack].Stack_head points to the first empty slot in stack,
            // so newstacksize is the size of the space in use by R and H.
            size_t newstacksize = Work [stack].Stack_head - Stack ;
            ASSERT (newstacksize <= stacksize) ;
            // Reduce the size of this stack.  Cblock [0:nf-1] is no longer
            // needed for holding pointers to the C blocks of each frontal
            // matrix.  Reuse it to hold the reallocated stacks. 
            if (shrink > 1)
            {
                // force the block to move by malloc'ing a new one;
                // this option is mainly for testing only.
                Cblock [stack] = (Entry *) cholmod_l_malloc (newstacksize,
                    sizeof (Entry), cc) ;
                if (Cblock [stack] == NULL)
                {
                    // oops, the malloc failed; just use the old block
                    cc->status = CHOLMOD_OK ;
                    Cblock [stack] = Stack ;
                    // update the memory usage statistics, however
		    cc->memory_inuse +=
                        ((newstacksize-stacksize) * sizeof (Entry)) ;
                }
                else
                {
                    // malloc is OK; copy the block over and free the old one
                    memcpy (Cblock [stack], Stack, newstacksize*sizeof(Entry)) ;
                    cholmod_l_free (stacksize, sizeof (Entry), Stack, cc) ;
                }
                // the Stack has been shrunk to the new size
                stacksize = newstacksize ;
            }
            else
            {
                // normal method; just realloc the block
                Cblock [stack] =    // pointer to the new Stack
                    (Entry *) cholmod_l_realloc (
                    newstacksize,   // requested size of Stack, in # of Entries
                    sizeof (Entry), // size of each Entry in the Stack
                    Stack,          // pointer to the old Stack
                    &stacksize,     // input: old stack size; output: new size
                    cc) ;
            }
            Stack_size [stack] = stacksize ;
            any_moved = any_moved || (Cblock [stack] != Stack) ;
            // reducing the size of a block of memory always succeeds
            ASSERT (cc->status == CHOLMOD_OK) ;
        }
    }

    // -------------------------------------------------------------------------
    // adjust the Rblock pointers if the Stacks have been moved
    // -------------------------------------------------------------------------

    if (any_moved)
    {
        // at least one Stack has moved; check all fronts and adjust them
        for (Int task = 0 ; task < ntasks ; task++)
        {
            Int kfirst, klast ;
            if (ntasks == 1)
            {
                // sequential case
                kfirst = 0 ;
                klast = nf ;
                stack = 0 ;
            }
            else
            {
                kfirst = TaskFrontp [task] ;
                klast  = TaskFrontp [task+1] ;
                stack  = TaskStack [task] ;
            }
            ASSERT (stack >= 0 && stack < ns) ;
            Entry *Old_Stack = Stacks [stack] ;
            Entry *New_Stack = Cblock [stack] ;
            if (New_Stack != Old_Stack)
            {
                for (Int kf = kfirst ; kf < klast ; kf++)
                {
                    Int f = (ntasks == 1) ? kf : TaskFront [kf] ;
                    Rblock [f] = New_Stack + (Rblock [f] - Old_Stack) ;
                }
            }
        }
        // finalize the Stacks
        for (stack = 0 ; stack < ns ; stack++)
        {
            Stacks [stack] = Cblock [stack] ;
        }
    }

    // -------------------------------------------------------------------------
    // free the rest of the workspace
    // -------------------------------------------------------------------------
    
    FREE_WORK_PART2 ;

    // -------------------------------------------------------------------------
    // extract the implicit row permutation for H
    // -------------------------------------------------------------------------

    // this must be done sequentially, when all threads are finished
    if (keepH)
    {
        // use Wi as workspace (Iwork (0:m-1)) [
        spqr_hpinv (QRsym, QRnum, Wi) ;
        // Wi no longer needed ]
    }

    // -------------------------------------------------------------------------
    // find the rank and return the result
    // -------------------------------------------------------------------------

    // find the rank of the first ntol columns of A
    if (ntol >= n)
    {
        rank1 = rank ;
    }
    else
    {
        rank1 = 0 ;
        for (j = 0 ; j < ntol ; j++)
        {
            if (!Rdead [j])
            {
                rank1++ ;
            }
        }
    }
    QRnum->rank1 = rank1 ;
    return (QRnum) ;
}


// =============================================================================

template spqr_numeric <double> *spqr_factorize <double>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Int freeA,                      // if TRUE, free A on output
    double tol,                     // for rank detection
    Int ntol,                       // apply tol only to first ntol columns
    spqr_symbolic *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template spqr_numeric <Complex> *spqr_factorize <Complex>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Int freeA,                      // if TRUE, free A on output
    double tol,                     // for rank detection
    Int ntol,                       // apply tol only to first ntol columns
    spqr_symbolic *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;
