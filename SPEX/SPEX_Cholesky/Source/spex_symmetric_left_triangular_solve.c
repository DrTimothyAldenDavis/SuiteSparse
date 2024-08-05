//------------------------------------------------------------------------------
// SPEX_Cholesky/spex_symmetric_left_triangular_solve: sparse symmetric
// left-looking triangular solve
//------------------------------------------------------------------------------

// SPEX_Cholesky: (c) 2020-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#define SPEX_FREE_ALL ;

#include "spex_cholesky_internal.h"

/* Purpose: This function performs the symmetric sparse REF triangular solve.
 * i.e.,(LD) x = A(:,k).
 *
 * At iteration k, it solves the linear system LD x = A(:,k). Upon completion
 * of this function, x contains the kth column of L.
 *
 * Input arguments of the function:
 *
 * top_output:      A pointer to the beginning of the nonzero pattern of L(:,k)
 *                  Undefined on input. On output xi[top_output..n-1] contains
 *                  the beginning of the nonzero pattern.
 *
 * x:               Solution of linear system. Undefined on input. On output
 *                  contains the kth column of L.
 *
 * xi:              Nonzero pattern. Undefined on input. On output contains the
 *                  nonzero pattern of the kth row of L
 *
 * L:               Lower triangular matrix.
 *
 * A:               Input matrix
 *
 * k:               Current iteration of the algorithm (i.e., column of A)
 *
 * rhos:            Sequence of pivots used in factorization
 *
 * h:               History vector.
 *
 * parent:          Elimination tree
 *
 * c:               Column pointers of L but they don't point to the top
 *                  position of each column of L. Instead they point to the
 *                  position on each column where the next value of L will be
 *                  grabbed, since at iteration k we need to grab the kth of L
 *                  in order to not recompute those values.
 */

// Comparison function used for the quicksort in the factorization
// Each iteration of the triangular solve requires that the nonzero pattern
// is sorted prior to numeric operations. This is the helper function for
// c's default qsort
static inline int compar (const void * a, const void * b)
{
    int64_t x = (* ((int64_t *) a)) ;
    int64_t y = (* ((int64_t *) b)) ;
    return (x < y ? -1 : ((x == y) ? 0 : 1)) ;
}

SPEX_info spex_symmetric_left_triangular_solve
(
    // Output
    int64_t *top_output,     // On output: the beginning of nonzero pattern of
                             // L(:,k). The nonzero pattern is contained in
                             // xi[top_output...n-1]
                             // On input: undefined
    SPEX_matrix x,           // On output: Solution of LD x = A(:,k) ==> kth row
                             // of L but really, the ONLY valid values of x are
                             // those in x[xi] since x is a working vector its
                             // other positions are jumbled.
    int64_t *xi,             // On output: Nonzero pattern vector
    // Input
    const SPEX_matrix L,     // Partial L matrix
    const SPEX_matrix A,     // Input matrix
    const int64_t k,         // Iteration of algorithm
    const SPEX_matrix rhos,  // Partial sequence of pivots
    int64_t *h,              // History vector
    const int64_t *parent,   // Elimination tree
    int64_t *c               // Column pointers of L but they don't point to the
                             // top position of each column of L. Instead they
                             // point to the position on each column where the
                             // next value of L will be grabbed, since at
                             // iteration k we need to grab the kth of L in
                             // order to not recompute those values.
)
{

    SPEX_info info;

    // Input checks. All pointers are checked by the callersm these are here to
    // remind us of the correct formats of each matrix
    ASSERT(top_output != NULL);
    ASSERT (x != NULL);
    ASSERT (xi != NULL);
    ASSERT (L != NULL);
    ASSERT (A != NULL);
    ASSERT (rhos != NULL);
    ASSERT (h != NULL);
    ASSERT (parent != NULL);
    ASSERT (c != NULL);
    ASSERT(L->type == SPEX_MPZ);
    ASSERT(L->kind == SPEX_CSC);
    ASSERT(A->type == SPEX_MPZ);
    ASSERT(A->kind == SPEX_CSC);
    ASSERT(rhos->type == SPEX_MPZ);
    ASSERT(rhos->kind == SPEX_DENSE);
    ASSERT(x->type == SPEX_MPZ);
    ASSERT(x->kind == SPEX_DENSE);

    int64_t j, i, p, m, top, n;
    int sgn;

    // row_top is the start of the nonzero pattern obtained after analyzing the
    // elimination tree xi[row_top..n-1] contains the nonzero pattern of the kth
    // row of L which is the first k-1 entries of the kth column of L
    int64_t row_top;

    //----------------------------------------------------------------
    // Initialize REF Triangular Solve by getting nonzero patern of x
    // This is done in two steps:
    // 1) Obtain the nonzero pattern of L[1:k-1,k]
    //    We could get this from L[k,1:k], but it is cheaper to do it with
    //    ereach.
    // 2) Obtain the nonzero pattern of L[k:n,k]
    //    This is obtained from the preallocation of L
    //----------------------------------------------------------------

    // Obtain dimension of the matrix, which is also the dimension of the dense
    // vectors
    n = A->n;
    ASSERT(n >= 0);

    //----------------------------------------------------------------
    // 1) Obtain the nonzero pattern of L[1:k-1,k]
    //----------------------------------------------------------------

    // Obtain the nonzero pattern of the kth row of L which is entries
    // L(1:k-1,k) Note that the left-looking Cholesky factorization performs
    // two elimination tree analyses. The first is done prior to here in the
    // preallocation of the L matrix. The second, performed here, gets the
    // nonzero pattern of L(k,:) (To compute L(:,k) you need the prealocation
    // first).

    SPEX_CHECK(spex_symmetric_ereach(&row_top, xi, A, k, parent, c));

    // After eReach, xi[rowtop..n-1] stores the location of the nonzeros
    // located in rows 1:k-1.  Note that the values of these nonzeros have
    // already been computed by the left-looking algorithm as they lie in row k
    // of columns 1:k-1 of L, so we do not need to compute these values from
    // scratch however we need to obtain their values.

    //----------------------------------------------------------------
    // 2) Obtain the nonzero pattern of L[1:k-1,k]
    //----------------------------------------------------------------
    // Now we populate the remainder of the nonzero pattern
    // (i.e., the indices of the nonzeros on rows k:n of L).
    // Note that these indices are known
    // because L was preallocated prior to factorization. Thus, we simply need
    // to iterate across the remainder of the kth column of L.
    // Oddly, this 'second' part of the nonzero pattern will be stored in the
    // top part of xi. (becuase of how the ereach works, it is easier to store
    // the nonzero pattern in rows 1:k-1 in the 'bottom' part of xi).
    top = row_top; //top starts in the last position, and it is decreased until
                   // it reaches the actual top position of xi
    for (i = L->p[k]; i < L->p[k+1]; i++)
    {
        top -= 1;           // One more nonzero in column k
        xi[top] = L->i[i];  // Index of the new nonzero
    }
    // At this point xi[top..n-1] contains the FULL nonzero pattern of column k.
    // Any entry lying in rows 1:k-1 of L already have their correct final value
    // currently stored in L. Any entry lying in rows k:n should take their
    // default value in A prior to the left-looking solve. We need the entries
    // in rows 1:k-1 of L in order to perform the IPGE_Updates & History_Updates
    // that are needed to compute the values of the entries in rows k:n of L.

    //----------------------------------------------------------------
    // Initialize x (only the positions of its nonzeros)
    //----------------------------------------------------------------

    // Reset only the essential positions in the x working vector.  That is, so
    // that x[i] = 0 for all row indices i located in xi (i.e., the nonzero
    // pattern of L(:,k)) (HOWEVER, you don't need to zero out the nonzeros
    // that will be grabbed directly from L[k,1:k-1] that is why the loop ends
    // in row_top and not in n)

    for (i = top; i < row_top; i++)
    {
        SPEX_mpz_set_ui(x->x.mpz[ xi[i] ], 0);
    }

    SPEX_mpz_set_ui(x->x.mpz[k], 0);

    // Now x[xi] has been zeroed. We obtain the values of any nonzero located in
    // L[k,1:k-1] which already reside in the previously computed kth row of L.
    // This is done by using the column pointer vector and a helper index p.
    for (i = row_top; i < n; i++)
    {
        m = xi[i];   // m is the row index of the current nonzero.
        p = ++c[m];  // this increases the column pointer of the mth column by
                     // one; because c[m] needs to be pointing to the next place
                     // on column m where a value will be taken from (when we
                     // grab another row of L)
        mpz_set(x->x.mpz[m], L->x.mpz[p]);
    }

    //--------------------------------------------------------------------------
    // Obtain A(:,k) to finish populating L(k+1:n,k) with its starting values.
    //--------------------------------------------------------------------------
    for (i = A->p[k]; i < A->p[k+1]; i++)
    {
        if ( A->i[i] >= k)
        {
            SPEX_MPZ_SET(x->x.mpz[A->i[i]], A->x.mpz[i]);
        }
    }
    // Sort the nonzero pattern xi using quicksort
    qsort (&xi[top], n-top, sizeof (int64_t), compar) ;

    // Reset the history vector h
    for (i = top; i < n; i++)
    {
        h[xi[i]] = -1;
    }

    //--------------------------------------------------------------------------
    // Iterate accross nonzeros in x
    //--------------------------------------------------------------------------
    for ( p = top; p < n; p++)
    {
        /* Finalize x[j] */
        j = xi[p];                              // Current nonzero term
        // If x[j] == 0 no work must be done (this zero is due to numerical
        // cancellation, not a structural/symbolic zero)
        SPEX_MPZ_SGN(&sgn, x->x.mpz[j]);
        if (sgn == 0) continue;
        if (j < k)                    // j < k implies already computed entries
        {
            //------------------------------------------------------------------
            // IPGE updates
            //------------------------------------------------------------------
            // ----------- Iterate accross nonzeros in Lij ---------------------
            for (m = L->p[j]; m < L->p[j+1]; m++)
            {
                i = L->i[m];            // i value of Lij
                if (i >= k)
                {
                    /*************** If lij==0 then no update******************/
                    SPEX_MPZ_SGN(&sgn, L->x.mpz[m]);
                    if (sgn == 0) continue;

                    //----------------------------------------------------------
                    /************* lij is nonzero, x[i] is zero****************/
                    // x[i] = 0 then only perform IPGE update subtraction/div
                    //----------------------------------------------------------
                    SPEX_MPZ_SGN(&sgn, x->x.mpz[i]);
                    if (sgn == 0)
                    {
                        // No previous pivot (because this entry has never been
                        // updated before)
                        if (j < 1)
                        {
                            SPEX_MPZ_SUBMUL(x->x.mpz[i],L->x.mpz[m],
                                            x->x.mpz[j]);// x[i] = 0 - lij*x[j]
                            h[i] = j;                  // Entry is up to date
                        }
                        // Previous pivot exists
                        else
                        {
                            SPEX_MPZ_SUBMUL(x->x.mpz[i],L->x.mpz[m],
                                            x->x.mpz[j]);// x[i] = 0 - lij*x[j]
                            SPEX_MPZ_DIVEXACT(x->x.mpz[i],
                                    x->x.mpz[i],
                                    rhos->x.mpz[j-1]);// x[i] = x[i] / rho[j-1]
                            h[i] = j;                  // Entry is up to date
                        }
                    }

                    //----------------------------------------------------------
                    /************ Both lij and x[i] are nonzero****************/
                    // x[i] != 0 --> History & IPGE update on x[i]
                    //----------------------------------------------------------
                    else
                    {
                        // No previous pivot in this case
                        if (j < 1)
                        {
                            SPEX_MPZ_MUL(x->x.mpz[i],x->x.mpz[i],
                                        rhos->x.mpz[0]); // x[i] = x[i]*rho[0]
                            SPEX_MPZ_SUBMUL(x->x.mpz[i], L->x.mpz[m],
                                         x->x.mpz[j]);// x[i] = x[i] - lij*xj
                            h[i] = j;                 // Entry is now up to date
                        }
                        // There is a previous pivot
                        else
                        {
                            // History update if necessary
                            if (h[i] < j - 1)
                            {
                                SPEX_MPZ_MUL(x->x.mpz[i],x->x.mpz[i],
                                    rhos->x.mpz[j-1]);// x[i] = x[i] * rho[j-1]
                                if (h[i] > -1)
                                {
                                    SPEX_MPZ_DIVEXACT(x->x.mpz[i],
                                                x->x.mpz[i],rhos->x.mpz[h[i]]);
                                                    // x[i] = x[i] / rho[h[i]]
                                }
                            }
                            SPEX_MPZ_MUL(x->x.mpz[i],x->x.mpz[i],
                                        rhos->x.mpz[j]);// x[i] = x[i] * rho[j]
                            SPEX_MPZ_SUBMUL(x->x.mpz[i], L->x.mpz[m],
                                        x->x.mpz[j]);// x[i] = x[i] - lij*xj
                            SPEX_MPZ_DIVEXACT(x->x.mpz[i],
                                    x->x.mpz[i],
                                    rhos->x.mpz[j-1]);// x[i] = x[i] / rho[j-1]
                            h[i] = j;                  // Entry is up to date
                        }
                    }
                }
            }
        }
        else                                              // Entries of L
        {
            //------------------------------------------------------------------
            // History update
            //------------------------------------------------------------------
            if (h[j] < k-1)
            {
                SPEX_MPZ_MUL(x->x.mpz[j],x->x.mpz[j],
                                rhos->x.mpz[k-1]); // x[j] = x[j] * rho[k-1]
                if (h[j] > -1)
                {
                    SPEX_MPZ_DIVEXACT(x->x.mpz[j],x->x.mpz[j],
                                rhos->x.mpz[h[j]]);// x[j] = x[j] / rho[h[j]]
                }
            }
        }
    }
    // Output the beginning of nonzero pattern
    (*top_output) = top;
    return SPEX_OK;
}
