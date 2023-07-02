// =============================================================================
// === spqrgpu_buildAssemblyMaps ===============================================
// =============================================================================

// SPQRGPU, Copyright (c) 2008-2022, Timothy A Davis, Sanjay Ranka,
// Sencer Nuri Yeralan, and Wissam Sid-Lakhdar, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------


#include "spqr.hpp"

#ifdef SUITESPARSE_CUDA

template <typename Int>
void spqrgpu_buildAssemblyMaps
(
    Int numFronts,
    Int n,
    Int *Fmap,
    Int *Post,
    Int *Super,
    Int *Rp,
    Int *Rj,
    Int *Sleft,
    Int *Sp,
    Int *Sj,
    double *Sx,
    Int *Fm,
    Int *Cm,
    Int *Childp,
    Int *Child,
    Int *CompleteStair,
    int *CompleteRjmap,
    Int *RjmapOffsets,
    int *CompleteRimap,
    Int *RimapOffsets,
    SEntry *cpuS
)
{

    PR (("GPU: building assembly maps:\n")) ;

    /* Use Fmap and Stair to map a front's local rows to global rows. */
    Int sindex = 0;

    for(Int pf=0; pf<numFronts; pf++) // iterate in post-order
    {
        Int f = Post[pf];

        /* Build Fmap for front f. */
        Int pstart = Rp[f], pend = Rp[f+1];
        for (Int p=pstart; p<pend ; p++)
        {
            Fmap[Rj[p]] = p - pstart;
        }

        /* Get workspaces for offset front members */
        Int *Stair = CompleteStair + Rp[f];

        // ---------------------------------------------------------------------
        // initialize the staircase for front F
        // ---------------------------------------------------------------------

        // initialize the staircase with original rows of S
        Int col1 = Super[f], col2 = Super[f+1];
        Int fp = col2 - col1;
        Int fn = Rp[f+1] - Rp[f];

        for (Int j = 0 ; j < fp ; j++)
        {
            // global column j+col1 is the jth pivot column of front F
            Int col = j + col1 ;
            Stair[j] = Sleft [col+1] - Sleft [col] ;
            PR (("GPU init rows, j: %ld count %ld\n", j, Stair[j])) ;
        }

        // contribution blocks from children will be added here
        for (Int j = fp ; j < fn ; j++){ Stair[j] = 0; }

        /* Build Rjmap and Stair for each child of the current front. */
        for(Int p=Childp[f]; p<Childp[f+1]; p++)
        {
            Int c = Child[p];
            PR (("child %ld\n", c)) ;
            ASSERT (c >= 0 && c < f) ;
            int *Rjmap = CompleteRjmap + RjmapOffsets[c];
            Int pc = Rp [c] ;
            Int pcend = Rp [c+1] ;
            Int fnc = pcend - pc ;              // total # cols in child F
            Int fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            Int cm = Cm [c] ;                   // # of rows in child C
            Int cn = fnc - fpc ;                // # of cols in child C

            // Create the relative column indices of the child's contributions
            // to the parent.  The global column 'col' in the child is a
            // contribution to the jth column of its parent front.  j is
            // a relative column index.
            for (Int pp=0 ; pp<cn ; pp++)
            {
                Int col = Rj [pc+fpc+pp] ;    // global column index
                Int j = Fmap [col] ;          // relative column index
                Rjmap [pp] = j ;               // Save this column
            }

            ASSERT (cm >= 0 && cm <= cn) ;
            pc += fpc ;                        // pointer to column indices in C
            ASSERT (pc + cn == Rp [c+1]) ;
            PR (("  cm %ld cn %ld\n", cm, cn)) ;

            // add the child rows to the staircase
            for (Int ci = 0 ; ci < cm ; ci++)
            {
                Int col = Rj [pc + ci] ;      // leftmost col of this row of C
                Int j = Fmap [col] ;          // global col is jth col of F
                PR (("  child col %ld j %ld\n", col, j)) ;
                ASSERT (j >= 0 && j < fn) ;
                Stair[j]++ ;                   // add this row to jth staircase
            }
        }

        // ---------------------------------------------------------------------
        // replace Stair with cumsum ([0 Stair]), and find # rows of F
        // ---------------------------------------------------------------------

        Int fm = 0 ;
        for (Int j = 0 ; j < fn ; j++)
        {
            Int t = fm ;
            fm += Stair[j] ;
            Stair[j] = t ;
        }
        PR (("fm %ld %ld\n", fm, Stair[fn-1])) ;

        // ---------------------------------------------------------------------
        // pack all the S values into the cpuS workspace & advance scalar stair
        // ---------------------------------------------------------------------

        Int Scount = MAX(0, Sp[Sleft[Super[f+1]]] - Sp[Sleft[Super[f]]]);
        if(Scount > 0)
        {
            for(Int k=0 ; k<fp ; k++)
            {
                /* pack all rows whose leftmost global column index is k+col1 */
                Int leftcol = k + col1 ;

                /* for each row of S whose leftmost column is leftcol */
                for (Int row = Sleft [leftcol]; row < Sleft [leftcol+1]; row++)
                {
                    // get the location of this row in F & advance the staircase
                    Int i = Stair[k]++;

                    /* Pack into S */
                    for (Int p=Sp[row] ; p<Sp[row+1] ; p++)
                    {
                        Int j = Sj[p];
                        cpuS[sindex].findex = i*fn + j;
                        cpuS[sindex].value = Sx[p];
                        sindex++;
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // build Rimap
        // ---------------------------------------------------------------------

        for (Int p = Childp [f] ; p < Childp [f+1] ; p++)
        {
            /* Get child details */
            Int c = Child [p] ;                 // get the child c of front F
            int *Rimap = CompleteRimap + RimapOffsets[c];
            int *Rjmap = CompleteRjmap + RjmapOffsets[c];
            Int pc = Rp [c] ;
            // Int pcend = Rp [c+1] ;
            // Int fnc = pcend - pc ;           // total # cols in child F
            Int fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
            Int cm = Cm [c] ;                   // # of rows in child C
            // Int cn = (fnc - fpc) ;           // cn =# of cols in child C
            // ASSERT (cm >= 0 && cm <= cn) ;
            pc += fpc ;                        // pointer to column indices in C
            // ASSERT (pc + cn == Rp [c+1]) ;

            /* -------------------------------------------------------------- */
            /* construct the Rimap                                            */
            /* -------------------------------------------------------------- */

            for (Int ci = 0 ; ci < cm ; ci++)
            {
                Int j = Rjmap[ci] ;          // global col is jth col of F
                ASSERT (j >= 0 && j < fn) ;
                Int i = Stair[j]++ ;         // add row F(i,:) to jth staircase
                ASSERT (i >= 0 && i < fm) ;
                Rimap[ci] = i ;               // keep track of the mapping
            }
        }
    }
}

template void spqrgpu_buildAssemblyMaps
(
    int32_t numFronts,
    int32_t n,
    int32_t *Fmap,
    int32_t *Post,
    int32_t *Super,
    int32_t *Rp,
    int32_t *Rj,
    int32_t *Sleft,
    int32_t *Sp,
    int32_t *Sj,
    double *Sx,
    int32_t *Fm,
    int32_t *Cm,
    int32_t *Childp,
    int32_t *Child,
    int32_t *CompleteStair,
    int *CompleteRjmap,
    int32_t *RjmapOffsets,
    int *CompleteRimap,
    int32_t *RimapOffsets,
    SEntry *cpuS
) ;

template void spqrgpu_buildAssemblyMaps
(
    int64_t numFronts,
    int64_t n,
    int64_t *Fmap,
    int64_t *Post,
    int64_t *Super,
    int64_t *Rp,
    int64_t *Rj,
    int64_t *Sleft,
    int64_t *Sp,
    int64_t *Sj,
    double *Sx,
    int64_t *Fm,
    int64_t *Cm,
    int64_t *Childp,
    int64_t *Child,
    int64_t *CompleteStair,
    int *CompleteRjmap,
    int64_t *RjmapOffsets,
    int *CompleteRimap,
    int64_t *RimapOffsets,
    SEntry *cpuS
) ;
#endif
