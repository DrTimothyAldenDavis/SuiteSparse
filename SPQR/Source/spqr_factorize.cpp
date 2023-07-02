// =============================================================================
// === spqr_factorize ==========================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

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
    free_Work <Entry, Int> (Work, ns, n, maxfn, wtsize, cc) ; \
    if (freeA) spqr_free_sparse <Int> (Ahandle, cc) ; \
    spqr_free <Int> (anz, sizeof (Entry), Sx, cc) ; \
    Sx = NULL ; \
}

// Free Cblock and the Work array itself
#define FREE_WORK_PART2 \
{ \
    spqr_free <Int> (ns, sizeof (spqr_work <Entry, Int>), Work, cc) ; \
    Work = NULL ; \
    spqr_free <Int> (nf+1, sizeof (Entry *), Cblock, cc) ; \
    Cblock = NULL ; \
}

// Free all workspace
#define FREE_WORK_FACTORIZE \
{ \
    FREE_WORK_PART1 ; \
    FREE_WORK_PART2 ; \
}


// =============================================================================
// === get_Work ================================================================
// =============================================================================

// Allocate the Work object, which is an array of structs.  The entry Work [s]
// contains workspace for the Stack s, for each Stack 0 to ns-1.

template <typename Entry, typename Int> spqr_work <Entry, Int> *get_Work
(
    Int ns,            // number of stacks
    Int n,             // number of columns of A
    Int maxfn,         // largest number of columns in any front
    Int keepH,         // if true, H is kept
    Int fchunk,
    Int *p_wtsize,     // size of WTwork for each 
    cholmod_common *cc
)
{
    int ok = TRUE ;
    spqr_work <Entry, Int> *Work ;
    Int wtsize ;
    *p_wtsize = 0 ;

    // wtsize = (fchunk + (keepH ? 0:1)) * maxfn ;
    wtsize = spqr_mult (fchunk + (keepH ? 0:1), maxfn, &ok) ;

    Work = (spqr_work <Entry, Int> *)    
        spqr_malloc <Int> (ns, sizeof (spqr_work <Entry, Int>), cc) ;

    if (!ok || cc->status < CHOLMOD_OK)
    {
        // out of memory or Int overflow
        spqr_free <Int> (ns, sizeof (spqr_work <Entry, Int>), Work, cc) ;
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        return (NULL) ;
    }

    for (Int stack = 0 ; stack < ns ; stack++)
    {
        Work [stack].Fmap = (Int *) spqr_malloc <Int> (n, sizeof (Int), cc) ;
        Work [stack].Cmap = (Int *) spqr_malloc <Int> (maxfn, sizeof(Int), cc);
        if (keepH)
        {
            // Staircase is a permanent part of H
            Work [stack].Stair1 = NULL ;
        }
        else
        {
            // Staircase workspace reused for each front
            Work [stack].Stair1 =
                (Int *) spqr_malloc <Int> (maxfn, sizeof (Int), cc) ;
        }
        Work [stack].WTwork =
            (Entry *) spqr_malloc <Int> (wtsize, sizeof (Entry), cc) ;
        Work [stack].sumfrank = 0 ;
        Work [stack].maxfrank = 0 ;

        Work [stack].wscale = 0 ;
        Work [stack].wssq   = 0 ;
    }

    *p_wtsize = wtsize ;
    return (Work) ;
}


// =============================================================================
// === free_Work ===============================================================
// =============================================================================

// Free the contents of Work, but not the Work array itself

template <typename Entry, typename Int> void free_Work
(
    spqr_work <Entry, Int> *Work,
    Int ns,            // number of stacks
    Int n,             // number of columns of A
    Int maxfn,         // largest number of columns in any front
    Int wtsize,        // size of WTwork array for each Stack
    cholmod_common *cc
)
{
    if (Work != NULL)
    {
        for (Int stack = 0 ; stack < ns ; stack++)
        {
            spqr_free <Int> (n,      sizeof (Int),   Work [stack].Fmap,   cc) ;
            spqr_free <Int> (maxfn,  sizeof (Int),   Work [stack].Cmap,   cc) ;
            spqr_free <Int> (maxfn,  sizeof (Int),   Work [stack].Stair1, cc) ;
            spqr_free <Int> (wtsize, sizeof (Entry), Work [stack].WTwork, cc) ;
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

template <typename Entry, typename Int> spqr_numeric <Entry, Int> *spqr_factorize
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    Int freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    Int ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <Int> *QRsym,

    // workspace and parameters
    cholmod_common *cc
)
{
    Int *Wi, *Qfill, *PLinv, *Cm, *Sp, *Stack_size,
        *TaskFront, *TaskFrontp, *TaskStack, *Stack_maxstack ;
    Entry *Sx, **Rblock, **Cblock, **Stacks ;
    spqr_numeric <Entry, Int> *QRnum ;
    Int nf, m, n, anz, fchunk, maxfn, rank, maxfrank, rjsize, rank1,
        maxstack,j, wtsize, stack, ns, ntasks, keepH, hisize ;
    char *Rdead ;
    cholmod_sparse *A ;
    spqr_work <Entry, Int> *Work ;

    // -------------------------------------------------------------------------
    // get inputs and contents of symbolic object
    // -------------------------------------------------------------------------

    if (QRsym == NULL)
    {
        // out of memory in caller
        if (freeA)
        {
            // if freeA is true, A must always be freed, even on error
            spqr_free_sparse <Int> (Ahandle, cc) ;
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

    spqr_allocate_work <Int> (0, MAX (m,nf), 0, cc) ;

    // shared Int workspace
    Wi = (Int *) cc->Iwork ;   // size m, aliased with the rest of Iwork
    Cm = Wi ;                   // size nf

    // Cblock is workspace shared by all threads
    Cblock = (Entry **) spqr_malloc <Int> (nf+1, sizeof (Entry *), cc) ;

    Work = NULL ;               // Work and its contents not yet allocated
    fchunk = MIN (m, FCHUNK) ;
    wtsize = 0 ;

    // -------------------------------------------------------------------------
    // create S
    // -------------------------------------------------------------------------

    // create numeric values of S = A(p,q) in row-form in Sx
    Sx = (Entry *) spqr_malloc <Int> (anz, sizeof (Entry), cc) ;

    if (cc->status == CHOLMOD_OK)
    {
        // use Wi as workspace (Iwork (0:m-1)) [
        spqr_stranspose2 (A, Qfill, Sp, PLinv, Sx, Wi) ;
        // Wi no longer needed ]
    }

    PR (("in spqr_factorize, status after creating Sx: %d\n", cc->status)) ;

    // -------------------------------------------------------------------------
    // input matrix A no longer needed; free it if the user doesn't need it
    // -------------------------------------------------------------------------

    if (freeA)
    {
        // this is done even if out of memory, above
        spqr_free_sparse <Int> (Ahandle, cc) ;
        ASSERT (*Ahandle == NULL) ;
    }
    PR (("in spqr_factorize, freed A, status %d\n", cc->status)) ;

    if (cc->status < CHOLMOD_OK)
    {
        PR (("in spqr_factorize, failure %d\n", cc->status)) ;
        // out of memory
        FREE_WORK_FACTORIZE ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // allocate numeric object
    // -------------------------------------------------------------------------

    QRnum = (spqr_numeric<Entry, Int> *)
        spqr_malloc <Int> (1, sizeof (spqr_numeric<Entry, Int>), cc) ;
    PR (("after allocating numeric object header, status %d\n", cc->status)) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        FREE_WORK_FACTORIZE ;
        return (NULL) ;
    }

    Rblock     = (Entry **) spqr_malloc <Int> (nf, sizeof (Entry *), cc) ;
    Rdead      = (char *)   spqr_calloc <Int> (n,  sizeof (char),    cc) ;

    // these may be revised (with ns=1) if we run out of memory
    Stacks     = (Entry **) spqr_calloc <Int> (ns, sizeof (Entry *), cc) ;
    Stack_size = (Int *)   spqr_calloc <Int> (ns, sizeof (Int),    cc) ;

    QRnum->Rblock     = Rblock ;
    QRnum->Rdead      = Rdead ;
    QRnum->Stacks     = Stacks ;
    QRnum->Stack_size = Stack_size ;

    if (keepH)
    {
        // allocate permanent space for Stair, Tau, Hii for each front
        QRnum->HStair= (Int *)  spqr_malloc <Int> (rjsize, sizeof (Int),  cc) ;
        QRnum->HTau  = (Entry *) spqr_malloc <Int> (rjsize, sizeof (Entry), cc) ;
        QRnum->Hii   = (Int *)  spqr_malloc <Int> (hisize, sizeof (Int),  cc) ;
        QRnum->Hm    = (Int *)  spqr_malloc <Int> (nf,     sizeof (Int),  cc) ;
        QRnum->Hr    = (Int *)  spqr_malloc <Int> (nf,     sizeof (Int),  cc) ;
        QRnum->HPinv = (Int *)  spqr_malloc <Int> (m,      sizeof (Int),  cc) ;
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
        FREE_WORK_FACTORIZE ;
        return (NULL) ;
    }

    PR (("after allocating rest of numeric object, status %d\n", cc->status)) ;

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    Work = get_Work <Entry, Int> (ns, n, maxfn, keepH, fchunk, &wtsize, cc) ;
    PR (("after allocating work, status %d\n", cc->status)) ;

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
            Stack = (Entry *) spqr_malloc <Int> (stacksize, sizeof (Entry), cc) ;
            Stacks [stack] = Stack ;
            Work [stack].Stack_head = Stack ;
            Work [stack].Stack_top  = Stack + stacksize ;
        }
    }

    PR (("after allocating the stacks, status %d\n", cc->status)) ;

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
                spqr_free <Int> (stacksize, sizeof (Entry), Stacks [stack], cc) ;
            }
        }
        spqr_free <Int> (ns, sizeof (Entry *), Stacks,     cc) ;
        spqr_free <Int> (ns, sizeof (Int),    Stack_size, cc) ;

        // free the contents of Work, and the Work array itself
        free_Work <Entry, Int> (Work, ns, n, maxfn, wtsize, cc) ;
        spqr_free <Int> (ns, sizeof (spqr_work <Entry, Int>), Work, cc) ;

        // punt to a single stack, a single task, and fchunk of 1
        ns = 1 ;
        ntasks = 1 ;
        fchunk = 1 ;
        cc->status = CHOLMOD_OK ;
        Work = get_Work <Entry, Int> (ns, n, maxfn, keepH, fchunk, &wtsize, cc) ;
        Stacks     = (Entry **) spqr_calloc <Int> (ns, sizeof (Entry *), cc) ;
        Stack_size = (Int *)   spqr_calloc <Int> (ns, sizeof (Int),    cc) ;
        QRnum->Stacks     = Stacks ;
        QRnum->Stack_size = Stack_size ;
        if (cc->status == CHOLMOD_OK)
        {
            Entry *Stack ;
            Stack_size [0] = maxstack ;
            Stack = (Entry *) spqr_malloc <Int> (maxstack, sizeof (Entry), cc) ;
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
        FREE_WORK_FACTORIZE ;
        return (NULL) ;
    }

    // At this point, the factorization is guaranteed to succeed, unless sizeof
    // (SUITESPARSE_BLAS_INT) < sizeof (Int), in which case, you really
    // should get a 64-bit BLAS.

    // -------------------------------------------------------------------------
    // create the Blob : everything the numeric factorization kernel needs
    // -------------------------------------------------------------------------

    spqr_blob <Entry, Int> Blob ;
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

    cc->SPQR_flopcount = 0 ;

    // -------------------------------------------------------------------------
    // numeric QR factorization
    // -------------------------------------------------------------------------

    PR (("[ calling the kernel\n")) ;

    if (ntasks == 1)
    {
        // Just one task, with or without TBB installed: don't use TBB
        spqr_kernel <Entry, Int> (0, &Blob) ;        // sequential case
    }
    else
    {
#if 0 /* ifdef HAVE_TBB (TBB removed) */
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

    PR (("] did the kernel\n")) ;

    // -------------------------------------------------------------------------
    // check for BLAS Int overflow on the CPU, or GPU failure
    // -------------------------------------------------------------------------

    if (cc->status < CHOLMOD_OK)
    {
        // On the CPU, this case can only occur if the problem is too large for
        // the BLAS.  This can only occur if, for example you're on a 64-bit
        // platform (with sizeof (Int) == 8) and using a 32-bit BLAS (with
        // sizeof (SUITESPARSE_BLAS_INT) == 4).  On the GPU, this case can more
        // easily occur, if we run out of memory in spqrgpu_kernel, or if we
        // fail to communicate with the GPU.
        spqr_freenum (&QRnum, cc) ;
        FREE_WORK_FACTORIZE ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // finalize the rank
    // -------------------------------------------------------------------------

    PR (("finalize the rank\n")) ;
    rank = 0 ;
    maxfrank = 1 ;
    for (stack = 0 ; stack < ns ; stack++)
    {
        PR (("stack: %ld Work [stack].sumfrank: %ld\n", stack, Work [stack].sumfrank)) ;
        rank += Work [stack].sumfrank ;
        maxfrank = MAX (maxfrank, Work [stack].maxfrank) ;
    }
    QRnum->rank = rank ;                    // required by spqr_hpinv
    QRnum->maxfrank = maxfrank ;
    PR (("m %ld n %ld my QR rank %ld\n", m, n, rank)) ;

    // -------------------------------------------------------------------------
    // finalize norm(w) for the dead column 2-norms
    // -------------------------------------------------------------------------

    double wscale = 0 ;
    double wssq = 1 ;
    for (stack = 0 ; stack < ns ; stack++)
    {
        // norm_E_fro = norm (s.*sqrt(q)) ; see also LAPACK's dnrm2
        double ws = Work [stack].wscale ;
        double wq = Work [stack].wssq ;
        if (wq != 0)
        {
            double wk = ws * sqrt (wq) ;
            if (wscale < wk)
            {
                double rr = wscale / wk ;
                wssq = 1 + wssq * rr * rr ;
                wscale = wk ;
            }
            else
            {
                double rr = wk / wscale ;
                wssq += rr * rr ;
            }
        }
    }
    QRnum->norm_E_fro = wscale * sqrt (wssq) ;
    cc->SPQR_norm_E_fro = QRnum->norm_E_fro ;

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
                Cblock [stack] = (Entry *) spqr_malloc <Int> (newstacksize,
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
                    spqr_free <Int> (stacksize, sizeof (Entry), Stack, cc) ;
                }
                // the Stack has been shrunk to the new size
                stacksize = newstacksize ;
            }
            else
            {
                // normal method; just realloc the block
                PR (("Normal shrink of the stack: %ld to %ld\n",
                    stacksize, newstacksize)) ;
                Cblock [stack] =    // pointer to the new Stack
                    (Entry *) spqr_realloc <Int> (
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
    PR (("find rank of first ntol cols of A: ntol %ld n %ld\n", ntol, n)) ;
    if (ntol >= n)
    {
        rank1 = rank ;
        PR (("rank1 is rank: %ld\n", rank1)) ;
    }
    else
    {
        rank1 = 0 ;
        for (j = 0 ; j < ntol ; j++)
        {
            PR (("column %ld Rdead: %d\n", j, (int) Rdead [j])) ;
            if (!Rdead [j])
            {
                rank1++ ;
            }
        }
        PR (("rank1 is sum of non-Rdead: %ld\n", rank1)) ;
    }
    QRnum->rank1 = rank1 ;
    return (QRnum) ;
}

template spqr_numeric <double, int32_t> *spqr_factorize <double, int32_t>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    int32_t freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    int32_t ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <int32_t> *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;
template spqr_numeric <Complex, int32_t> *spqr_factorize <Complex, int32_t>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    int32_t freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    int32_t ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <int32_t> *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;
template spqr_numeric <double, int64_t> *spqr_factorize <double, int64_t>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    int64_t freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    int64_t ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <int64_t> *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;
template spqr_numeric <Complex, int64_t> *spqr_factorize <Complex, int64_t>
(
    // input, optionally freed on output
    cholmod_sparse **Ahandle,

    // inputs, not modified
    int64_t freeA,                     // if TRUE, free A on output
    double tol,                     // for rank detection
    int64_t ntol,                      // apply tol only to first ntol columns
    spqr_symbolic <int64_t> *QRsym,

    // workspace and parameters
    cholmod_common *cc
) ;
