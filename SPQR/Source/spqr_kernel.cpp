// =============================================================================
// === spqr_kernel =============================================================
// =============================================================================

// Factorize all the fronts in a single task

#include "spqr.hpp"

template <typename Entry> void spqr_kernel
(
    Int task,
    spqr_blob <Entry> *Blob
)
{

    // -------------------------------------------------------------------------
    // get the Blob
    // -------------------------------------------------------------------------

    spqr_symbolic *          QRsym = Blob->QRsym ;
    spqr_numeric <Entry> *   QRnum = Blob->QRnum ;
    double                   tol = Blob->tol ;
    Int                      ntol = Blob->ntol ;
    Int                      fchunk = Blob->fchunk ;
    spqr_work <Entry> *      Work = Blob->Work ;
    Int *                    Cm = Blob->Cm ;
    Entry **                 Cblock = Blob->Cblock ;
    Entry *                  Sx = Blob->Sx ;
    cholmod_common *         cc = Blob->cc ;

    // -------------------------------------------------------------------------
    // get the contents of the QR symbolic object
    // -------------------------------------------------------------------------

    Int *   Super = QRsym->Super ;      // size nf+1, gives pivot columns in F
    Int *   Rp = QRsym->Rp ;            // size nf+1, pointers for pattern of R
    Int *   Rj = QRsym->Rj ;            // size QRsym->rjsize, col indices of R
    Int *   Sleft = QRsym->Sleft ;      // size n+2, leftmost column sets
    Int *   Sp = QRsym->Sp ;            // size m+1, row pointers for S
    Int *   Sj = QRsym->Sj ;            // size anz, column indices for S
    Int *   Child = QRsym->Child ;      // size nf, for lists of children
    Int *   Childp = QRsym->Childp ;    // size nf+1, for lists of children
    Int     maxfn  = QRsym->maxfn ;     // max # of columns in any front
    Int     nf = QRsym->nf ;            // number of fronts

    Int *   Hip = QRsym->Hip ;          // only used if H is kept

    // these arrays are all NULL if QRsym->ntasks:
    Int *   TaskFront = QRsym->TaskFront ;      // size nf+1
    Int *   TaskFrontp = QRsym->TaskFrontp ;    // size ntasks+1
    Int *   TaskStack = QRsym->TaskStack ;      // size ntasks+1
    Int *   On_stack = QRsym->On_stack ;        // size nf+1

    // used for sequential case (when QRnum->ntasks == 1)
    Int *   Post = QRsym->Post ;                // size nf

    // -------------------------------------------------------------------------
    // get the contents of the QR numeric object
    // -------------------------------------------------------------------------

    Entry **    Rblock = QRnum->Rblock ;
    char *      Rdead = QRnum->Rdead ;
    Int *       HStair = QRnum->HStair ;
    Entry *     HTau = QRnum->HTau ;
    Int *       Hii = QRnum->Hii ;          // only used if H is kept
    Int *       Hm = QRnum->Hm ;
    Int *       Hr = QRnum->Hr ;
    Int         keepH = QRnum->keepH ;
    Int         ntasks = QRnum->ntasks ;    // number of tasks

    // -------------------------------------------------------------------------
    // get the stack for this task and the head/top pointers
    // -------------------------------------------------------------------------

    Int stack, kfirst, klast ;

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
    Int stacksize = (ntasks == 1) ?
            QRsym->maxstack :
            QRsym->Stack_maxstack [stack] ;
#endif

    // if H kept, Tau and Stair will point to permanent space in QRnum
    Entry * Tau = keepH ? NULL : Work [stack].WTwork ;
    Int *   Stair = keepH ? NULL : Work [stack].Stair1 ;
    Entry * W = Work [stack].WTwork + (keepH ? 0 : maxfn) ;

    Int *   Fmap = Work [stack].Fmap ;
    Int *   Cmap = Work [stack].Cmap ;

    Int     sumfrank = Work [stack].sumfrank ;
    Int     maxfrank = Work [stack].maxfrank ;

    // -------------------------------------------------------------------------
    // factorize all the fronts in this task
    // -------------------------------------------------------------------------

    for (Int kf = kfirst ; kf < klast ; kf++)
    {

        // ---------------------------------------------------------------------
        // factorize front F
        // ---------------------------------------------------------------------

        Int f = (ntasks == 1) ? Post [kf] : TaskFront [kf] ;

#ifndef NDEBUG
        ASSERT (f >= 0 && f < QRsym->nf) ;
        for (Int col = 0 ; col < QRsym->n ; col++) Fmap [col] = EMPTY ;
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

        Int fm = spqr_fsize (f, Super, Rp, Rj, Sleft, Child, Childp, Cm,
            Fmap, Stair) ;
        Int fn = Rp [f+1] - Rp [f] ;        // F is fm-by-fn
        Int col1 = Super [f] ;              // first global pivot column in F
        Int fp = Super [f+1] - col1 ;       // with fp pivot columns
        Int fsize = fm * fn ;
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
            (Int) (Stack_head - Stack),
            (Int) (Stack_top - Stack),
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

        for (Int p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            Int c = Child [p] ;
            ASSERT (c >= 0 && c < f) ;
            PR (("   child %ld on stack %ld\n", c,
                (ntasks == 1) ? 0 : On_stack [c])) ;
            if (ntasks == 1 || On_stack [c] == stack)
            {
                Int ccsize = spqr_csize (c, Rp, Cm, Super) ;
                Stack_top = MAX (Stack_top, Cblock [c] + ccsize) ;
            }
        }

        ASSERT (Stack_head >= Stack && Stack_head <= Stack_top) ;
        ASSERT (Stack_top <= Stack + stacksize) ;

        // ---------------------------------------------------------------------
        // factorize the front F
        // ---------------------------------------------------------------------

        Int frank = spqr_front (fm, fn, fp, tol, ntol - col1,
            fchunk, F, Stair, Rdead + col1, Tau, W, cc) ;

#ifndef NDEBUG
#ifndef NPRINT
        PR (("\n F factorized: f %ld fp %ld frank %ld \n", f, fp, frank)) ;
        for (Int jj = col1 ; jj < Super [f+1] ; jj++)
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

        Int csize = spqr_fcsize (fm, fn, fp, frank) ;
        Stack_top -= csize ;

#ifndef NDEBUG
        PR (("Front f %ld csize %ld piv rank %ld\n", f, csize, frank)) ;
        PR (("Stack head %ld top %ld total %ld stacksize %ld\n",
            (Int) (Stack_head - Stack),
            (Int) (Stack_top - Stack),
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

        Int rm ;
        Int rsize = spqr_rhpack (keepH, fm, fn, fp, Stair, F, F, &rm) ;
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
}


// =============================================================================

template void spqr_kernel <double>
(
    Int task,
    spqr_blob <double> *Blob
) ;

template void spqr_kernel <Complex>
(
    Int task,
    spqr_blob <Complex> *Blob
) ;
