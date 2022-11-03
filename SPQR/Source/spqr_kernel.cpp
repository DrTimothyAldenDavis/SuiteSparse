// =============================================================================
// === spqr_kernel =============================================================
// =============================================================================

// SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Factorize all the fronts in a single task

#include "spqr.hpp"

template <typename Entry> void spqr_kernel // _worker
(
    int64_t task,
    spqr_blob <Entry> *Blob
)
{

    // -------------------------------------------------------------------------
    // get the Blob
    // -------------------------------------------------------------------------

    spqr_symbolic *          QRsym = Blob->QRsym ;
    spqr_numeric <Entry> *   QRnum = Blob->QRnum ;
    double                   tol = Blob->tol ;
    int64_t                     ntol = Blob->ntol ;
    int64_t                     fchunk = Blob->fchunk ;
    spqr_work <Entry> *      Work = Blob->Work ;
    int64_t *                   Cm = Blob->Cm ;
    Entry **                 Cblock = Blob->Cblock ;
    Entry *                  Sx = Blob->Sx ;
    cholmod_common *         cc = Blob->cc ;

    // -------------------------------------------------------------------------
    // if we're using the GPU, reroute into the gpu-accelerated kernel code
    // -------------------------------------------------------------------------

#ifdef SUITESPARSE_CUDA
    if (QRsym->QRgpu != NULL)
    {
        ASSERT (task == 0) ;
        spqrgpu_kernel (Blob) ;
        return ;
    }
#endif

    // -------------------------------------------------------------------------
    // get the contents of the QR symbolic object
    // -------------------------------------------------------------------------

    int64_t *  Super = QRsym->Super ;      // size nf+1, gives pivot columns in F
    int64_t *  Rp = QRsym->Rp ;            // size nf+1, pointers for pattern of R
    int64_t *  Rj = QRsym->Rj ;            // size QRsym->rjsize, col indices of R
    int64_t *  Sleft = QRsym->Sleft ;      // size n+2, leftmost column sets
    int64_t *  Sp = QRsym->Sp ;            // size m+1, row pointers for S
    int64_t *  Sj = QRsym->Sj ;            // size anz, column indices for S
    int64_t *  Child = QRsym->Child ;      // size nf, for lists of children
    int64_t *  Childp = QRsym->Childp ;    // size nf+1, for lists of children
    int64_t    maxfn  = QRsym->maxfn ;     // max # of columns in any front
    int64_t    nf = QRsym->nf ;            // number of fronts

    int64_t *  Hip = QRsym->Hip ;          // only used if H is kept

    // these arrays are all NULL if QRsym->ntasks:
    int64_t *  TaskFront = QRsym->TaskFront ;      // size nf+1
    int64_t *  TaskFrontp = QRsym->TaskFrontp ;    // size ntasks+1
    int64_t *  TaskStack = QRsym->TaskStack ;      // size ntasks+1
    int64_t *  On_stack = QRsym->On_stack ;        // size nf+1

    // used for sequential case (when QRnum->ntasks == 1)
    int64_t *  Post = QRsym->Post ;                // size nf

    // -------------------------------------------------------------------------
    // get the contents of the QR numeric object
    // -------------------------------------------------------------------------

    Entry ** Rblock = QRnum->Rblock ;
    char *   Rdead = QRnum->Rdead ;
    int64_t *   HStair = QRnum->HStair ;
    Entry *  HTau = QRnum->HTau ;
    int64_t *   Hii = QRnum->Hii ;          // only used if H is kept
    int64_t *   Hm = QRnum->Hm ;
    int64_t *   Hr = QRnum->Hr ;
    int64_t     keepH = QRnum->keepH ;
    int64_t     ntasks = QRnum->ntasks ;    // number of tasks

    // -------------------------------------------------------------------------
    // get the stack for this task and the head/top pointers
    // -------------------------------------------------------------------------

    int64_t stack, kfirst, klast ;

    if (ntasks == 1)
    {
        // sequential case
        kfirst = 0 ;
        klast  = nf ;
        stack  = 0 ;
    }
    else
    {
        kfirst = TaskFrontp [task] ;
        klast  = TaskFrontp [task+1] ;
        stack  = TaskStack [task] ;
    }

    Entry * Stack_head = Work [stack].Stack_head ;
    Entry * Stack_top = Work [stack].Stack_top ;

#ifndef NDEBUG
    Entry **Stacks = QRnum->Stacks ;
    Entry *Stack = Stacks [stack] ;    // stack being used
    int64_t stacksize = (ntasks == 1) ?
            QRsym->maxstack :
            QRsym->Stack_maxstack [stack] ;
#endif

    // if H kept, Tau and Stair will point to permanent space in QRnum
    Entry * Tau = keepH ? NULL : Work [stack].WTwork ;
    int64_t *  Stair = keepH ? NULL : Work [stack].Stair1 ;
    Entry * W = Work [stack].WTwork + (keepH ? 0 : maxfn) ;

    int64_t *  Fmap = Work [stack].Fmap ;
    int64_t *  Cmap = Work [stack].Cmap ;

    int64_t    sumfrank = Work [stack].sumfrank ;
    int64_t    maxfrank = Work [stack].maxfrank ;

    // for keeping track of norm(w) for dead column 2-norms
    double wscale = Work [stack].wscale ;
    double wssq   = Work [stack].wssq   ;

    // -------------------------------------------------------------------------
    // factorize all the fronts in this task
    // -------------------------------------------------------------------------

    for (int64_t kf = kfirst ; kf < klast ; kf++)
    {

        // ---------------------------------------------------------------------
        // factorize front F
        // ---------------------------------------------------------------------

        int64_t f = (ntasks == 1) ? Post [kf] : TaskFront [kf] ;

#ifndef NDEBUG
        ASSERT (f >= 0 && f < QRsym->nf) ;
        for (int64_t col = 0 ; col < QRsym->n ; col++) Fmap [col] = EMPTY ;
#endif

        if (keepH)
        {
            // get the permanent Stair and Tau vectors for this front
            Stair = HStair + Rp [f] ;
            Tau = HTau + Rp [f] ;
        }

        // ---------------------------------------------------------------------
        // determine the size of F, its staircase, and its Fmap
        // ---------------------------------------------------------------------

        int64_t fm = spqr_fsize (f, Super, Rp, Rj, Sleft, Child, Childp, Cm,
            Fmap, Stair) ;
        int64_t fn = Rp [f+1] - Rp [f] ;        // F is fm-by-fn
        int64_t col1 = Super [f] ;              // first global pivot column in F
        int64_t fp = Super [f+1] - col1 ;       // with fp pivot columns
        int64_t fsize = fm * fn ;
        if (keepH)
        {
            Hm [f] = fm ;
        }

#ifndef NDEBUG
        PR (("\n --- Front f %ld fm %ld fn %ld fp %ld\n", f, fm, fn, fp)) ;
        ASSERT (Stack_head <= Stack_top) ;
#endif

        // ---------------------------------------------------------------------
        // allocate F
        // ---------------------------------------------------------------------

        // allocate F; this will reduce in size to hold just R or RH when done
        Entry *F = Stack_head ;
        Rblock [f] = F ;
        Stack_head += fsize ;

#ifndef NDEBUG
        PR (("Stack head %ld top %ld total %ld stacksize %ld\n",
            (int64_t) (Stack_head - Stack),
            (int64_t) (Stack_top - Stack),
            (Stack_head - Stack) +
            stacksize - (Stack_top - Stack),
            stacksize)) ;
        ASSERT (Stack_head <= Stack_top) ;
#endif

        // ---------------------------------------------------------------------
        // assemble the C blocks of the children of F
        // ---------------------------------------------------------------------

        spqr_assemble (f, fm, keepH,
            Super, Rp, Rj, Sp, Sj, Sleft, Child, Childp,
            Sx, Fmap, Cm, Cblock,
#ifndef NDEBUG
            Rdead,
#endif
            Hr, Stair, Hii, Hip, F, Cmap) ;

#ifndef NDEBUG
#ifndef NPRINT
        PR (("\n --- Front assembled: f %ld\n", f)) ;
        spqrDebug_dumpdense (F, fm, fn, fm, cc) ;
        PR (("\n --- Freeing children from stack %ld\n", stack)) ;
#endif
#endif

        // ---------------------------------------------------------------------
        // free the C blocks of the children of F
        // ---------------------------------------------------------------------

        for (int64_t p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            int64_t c = Child [p] ;
            ASSERT (c >= 0 && c < f) ;
            PR (("   child %ld on stack %ld\n", c,
                (ntasks == 1) ? 0 : On_stack [c])) ;
            if (ntasks == 1 || On_stack [c] == stack)
            {
                int64_t ccsize = spqr_csize (c, Rp, Cm, Super) ;
                Stack_top = MAX (Stack_top, Cblock [c] + ccsize) ;
            }
        }

        ASSERT (Stack_head >= Stack && Stack_head <= Stack_top) ;
        ASSERT (Stack_top <= Stack + stacksize) ;

        // ---------------------------------------------------------------------
        // factorize the front F
        // ---------------------------------------------------------------------

        int64_t frank = spqr_front (fm, fn, fp, tol, ntol - col1,
            fchunk, F, Stair, Rdead + col1, Tau, W,
            &wscale, &wssq, cc) ;

#ifndef NDEBUG
#ifndef NPRINT
        PR (("\n F factorized: f %ld fp %ld frank %ld \n", f, fp, frank)) ;
        for (int64_t jj = col1 ; jj < Super [f+1] ; jj++)
            PR (("Rdead [%ld] = %d\n", jj, Rdead [jj])) ;
        PR (("\n ::: Front factorized:\n")) ;
        spqrDebug_dumpdense (F, fm, fn, fm, cc) ;
#endif
#endif

        // ---------------------------------------------------------------------
        // keep track of the rank of fronts (just for this stack)
        // ---------------------------------------------------------------------

        sumfrank += frank ;
        maxfrank = MAX (maxfrank, frank) ;

        // ---------------------------------------------------------------------
        // pack the C block of front F on stack
        // ---------------------------------------------------------------------

        int64_t csize = spqr_fcsize (fm, fn, fp, frank) ;
        Stack_top -= csize ;

#ifndef NDEBUG
        PR (("Front f %ld csize %ld piv rank %ld\n", f, csize, frank)) ;
        PR (("Stack head %ld top %ld total %ld stacksize %ld\n",
            (int64_t) (Stack_head - Stack),
            (int64_t) (Stack_top - Stack),
            (Stack_head - Stack) +
            stacksize - (Stack_top - Stack),
            stacksize)) ;
        ASSERT (Stack_head <= Stack_top) ;
#endif

        Cblock [f] = Stack_top ;
        Cm [f] = spqr_cpack (fm, fn, fp, frank, F, Cblock [f]) ;

        // ---------------------------------------------------------------------
        // pack R or RH of front F in place
        // ---------------------------------------------------------------------

        int64_t rm ;
        int64_t rsize = spqr_rhpack (keepH, fm, fn, fp, Stair, F, F, &rm) ;
        if (keepH)
        {
            Hr [f] = rm ;
        }

#ifndef NDEBUG
        PR (("Cm of f %ld is %ld, csize: %ld\n", f, Cm [f], csize)) ;
        if (keepH)
        {
            ASSERT (rsize <= fsize - csize) ;
            PR (("Front %ld RHSIZE actual: %ld\n", f, rsize)) ;
            PR (("Hr [ %ld ] =  %ld\n", f, Hr [f])) ;
            ASSERT (spqrDebug_rhsize (fm, fn, fp, Stair, cc) == rsize) ;
        }
        else
        {
            ASSERT (rsize <= frank*(frank+1)/2 + frank*(fn-frank)) ;
            ASSERT (rsize <= fsize) ;
            ASSERT (spqrDebug_rhsize (fm, fn, fp, Stair, cc) >= rsize) ;
        }
#endif

        // ---------------------------------------------------------------------
        // free F, leaving R or RH in its place
        // ---------------------------------------------------------------------

        Stack_head -= fsize ;             // free F
        Stack_head += rsize ;             // allocate R or RH

#ifndef NDEBUG
        PR (("front %ld r/hsize %ld fp %ld fn %ld if trapezoidal %ld\n",
            f, rsize, fp, fn, fp*(fp+1)/2 + fp*(fn-fp))) ;
        PR (("rsize %ld frank %ld fn %ld\n", rsize, frank, fn)) ;
        ASSERT (Stack_head <= Stack_top) ;
#endif
    }

    // -------------------------------------------------------------------------
    // save the stack top & head pointers for the next task on this stack
    // -------------------------------------------------------------------------

    Work [stack].Stack_head = Stack_head  ;
    Work [stack].Stack_top = Stack_top ;
    Work [stack].sumfrank = sumfrank ;  // sum rank of fronts for this stack
    Work [stack].maxfrank = maxfrank ;  // max rank of fronts for this stack

    // for keeping track of norm(w) for dead column 2-norms
    Work [stack].wscale = wscale ;
    Work [stack].wssq   = wssq   ;
}

// =============================================================================

template void spqr_kernel <double>
(
    int64_t task,
    spqr_blob <double> *Blob
) ;

template void spqr_kernel <Complex>
(
    int64_t task,
    spqr_blob <Complex> *Blob
) ;
