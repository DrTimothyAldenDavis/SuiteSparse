//------------------------------------------------------------------------------
// SPEX_Left_LU/spex_left_lu_ref_triangular_solve: sparse REF triangular solve
//------------------------------------------------------------------------------

// SPEX_Left_LU: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------


/*
 * Purpose: This function performs the sparse REF triangular solve. i.e.,
 * (LD) x = A(:,k). The algorithm is described in the paper; however in essence
 * it computes the nonzero pattern xi, then performs a sequence of IPGE
 * operations on the nonzeros to obtain their final value. All operations are
 * guaranteed to be integral (mpz_t). There are various enhancements in this
 * code used to reduce the overall cost of the operations and minimize
 * operations as much as possible.
 */

/* Description of input/output
 *
 *  top_output: An int64_t scalar which on input is uninitialized. On output
 *              contains the contains the beginning of the nonzero pattern.
 *              The nonzero pattern is contained in xi[top_output...n-1].
 *
 *  L:          The partial L matrix. On input contains columns 1:k-1 of L.
 *              Unmodified on on output.
 *
 *  A:          The input matrix. Unmodified on input/output
 *
 *  k:          Unmodified int64_t which indicates which column of L and U is
 *              being computed.  That is, this triangular solve computes L(:,k)
 *              and U(:,k).
 *
 *  xi:         A worspace array of size 2n, unitialized on input. On output,
 *              xi[top...n-1] contains the nonzero pattern of L(:,k) and U(:,k)
 *              in strictly sorted order.  The row indices in xi[top...n-1]
 *              reflect the row indices in the original A matrix not the final
 *              L and U because of permutation.
 *
 *  q:          An array of size n+1 which is unmodified on input/output.
 *              j = q[k] if the original column j is the kth column of the
 *              LU factorization.
 *
 *  rhos:       Sequence of pivots, unmodified on input/output. This matrix is
 *              utilized to perform history and IPGE updates in the triangular
 *              solve.
 *
 *  pinv:       An array of size n which contains the inverse row permutation,
 *              unmodified on input/ output. This vector is utilized in the
 *              factorization as well as for sorting.
 *
 *  row_perm:   An array of size n which contains the row permutation,
 *              unmodified on input/output.  row_perm is the inverse of pinv
 *              and is utilized for the sorting routine.
 *
 *  h:          A workspace array of size n, unitialized on input and undefined
 *              on output. This vector is utilized to perform history updates
 *              on entries in x. During the triangular solve, h[i] indicates
 *              the last pivot in which entry x[i] was IPGE updated.
 *
 *  x:          Dense mpz_t matrix Workspace of size n*1, unitialized on input.
 *              On output, x[i] is the value of L(i,k) here i is in the nonzero
 *              pattern xi[top...n-1]. Other entries of x are undefined on
 *              output.
 */

#include "spex_left_lu_internal.h"

// Sorting function
static inline int compare (const void * a, const void * b)
{
    int64_t delta = ( *(int64_t*)a - *(int64_t*)b ) ;
    //return value for delta==0 won't matter since it's not happening here
    if (delta < 0)
    {
        return (-1) ;
    }
    else// if (delta >= 0)
    {
        return (1) ;
    }
}

SPEX_info spex_left_lu_ref_triangular_solve // performs the sparse REF triangular solve
(
    int64_t *top_output,      // Output the beginning of nonzero pattern
    SPEX_matrix* L,           // partial L matrix
    const SPEX_matrix* A,     // input matrix
    int64_t k,                // constructing L(:,k)
    int64_t* xi,              // nonzero pattern vector
    const int64_t* q,         // column permutation, not modified
    SPEX_matrix* rhos,        // sequence of pivots
    const int64_t* pinv,      // inverse row permutation
    const int64_t* row_perm,  // row permutation
    int64_t* h,               // history vector
    SPEX_matrix* x            // solution of system ==> kth column of L and U
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // inputs have been validated in SPEX_Left_LU_factorize.c
    SPEX_info info ;
    SPEX_REQUIRE(L, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(A, SPEX_CSC, SPEX_MPZ);
    SPEX_REQUIRE(rhos, SPEX_DENSE, SPEX_MPZ);

    int64_t j, jnew, i, inew, p, m, n, col, top ;
    int sgn ;
    mpz_t *x_mpz = x->x.mpz, *Ax_mpz = A->x.mpz, *Lx_mpz = L->x.mpz,
          *rhos_mpz = rhos->x.mpz;

    //--------------------------------------------------------------------------
    // Begin the REF triangular solve by obtaining the nonzero pattern, and
    // reseting the vectors x, xi, and h
    //--------------------------------------------------------------------------

    // Size of matrix and the dense vectors
    n = A->n;

    // This triangular solve is computing L(:,k) and U(:,k) for the
    // factorization L*D*U = A*Q.  The column permutation is held in the
    // permutation vector q, and col = q [k] is the column of A that has been
    // permuted to the kth column of A*Q.
    col = q[k];

    // Obtain nonzero pattern of L(:,k) in xi[top..n]
    spex_left_lu_reach(&top, L, A, col, xi, pinv);

    // Sort xi [top..n-1] wrt sequence of pivots
    // Convert xi vector with respect to pinv
    for (j = top; j < n; j++)
    {
        xi[j] = pinv[xi[j]];
    }

    // Sort xi[top..n-1]
    qsort(&xi[top], n-top, sizeof(int64_t), compare);

    // Place xi back in original value
    for (j = top; j < n; j++)
    {
        xi[j] = row_perm[xi[j]];
    }

    // Reset x[i] = 0 for all i in nonzero pattern xi [top..n-1]
    for (i = top; i < n; i++)
    {
        SPEX_CHECK (SPEX_mpz_set_ui (x_mpz[xi [i]], 0)) ;
    }
    // Set x[col] = 0.  A(col,col) is the diagonal entry in the original
    // matrix.  The pivot search prefers to select the diagonal, if it is
    // present.  It may be not present in A itself, at all, and thus also
    // not in the pattern xi [top..n-1].  The value x[col] is set to zero
    // here, in case the entry A(col,col) is not present, so that the pivot
    // search query the value of the diagonal.
    SPEX_CHECK(SPEX_mpz_set_ui(x_mpz[col], 0));

    // Reset h[i] = -1 for all i in nonzero pattern xi [top..n-1]
    for (i = top; i < n; i++)
    {
        h[xi[i]] = -1;
    }

    // Set x = A(:,q(k))
    for (i = A->p[col]; i < A->p[col + 1]; i++)
    {
        // Value of the ith nonzero
        SPEX_CHECK(SPEX_mpz_set(x_mpz[A->i[i]], Ax_mpz[i]));
    }


    //--------------------------------------------------------------------------
    // Iterate across nonzeros in x
    //--------------------------------------------------------------------------
    for (p = top; p < n; p++)
    {
        /* Finalize x[j] */
        j = xi[p];                         // First nonzero term
        jnew = pinv[j];                    // Location of nonzero term
        // Check if x[j] == 0, if so continue to next nonzero
        SPEX_CHECK(SPEX_mpz_sgn(&sgn, x_mpz[j]));
        if (sgn == 0) {continue;}          // x[j] = 0 no work must be done

        // x[j] is nonzero
        if (jnew < k)                      // jnew < k implies entries in U
        {
            //------------------------------------------------------------------
            // History update: Bring x[j] to its final value
            //------------------------------------------------------------------
            if (h[j] < jnew - 1)           // HU must be performed
            {
                // x[j] = x[j] * rho[j-1]
                SPEX_CHECK(SPEX_mpz_mul(x_mpz[j],x_mpz[j],rhos_mpz[jnew-1]));

                if (h[j] > -1)
                {
                    // x[j] = x[j] / rho[h[j]]
                    SPEX_CHECK(SPEX_mpz_divexact(x_mpz[j],x_mpz[j],
                        rhos_mpz[h[j]]));
                }
            }

            //------------------------------------------------------------------
            // IPGE updates
            //------------------------------------------------------------------

            // ----------- Iterate across nonzeros in Lij ---------------------
            for (m = L->p[jnew]; m < L->p[jnew+1]; m++)
            {
                i = L->i[m];               // i value of Lij
                inew = pinv[i];            // i location of Lij
                if (inew > jnew)
                {
                    /*************** If lij==0 then no update******************/
                    SPEX_CHECK(SPEX_mpz_sgn(&sgn, Lx_mpz[m]));
                    if (sgn == 0) {continue;}

                    // lij is nonzero. Check if x[i] is nonzero
                    SPEX_CHECK(SPEX_mpz_sgn(&sgn, x_mpz[i]));

                    //----------------------------------------------------------
                    /************* lij is nonzero, x[i] is zero****************/
                    // x[i] = 0 then only perform IPGE update
                    // subtraction/division
                    //----------------------------------------------------------

                    if (sgn == 0)
                    {
                        // No previous pivot
                        if (jnew < 1)
                        {
                            // x[i] = 0 - lij*x[j]
                            SPEX_CHECK(SPEX_mpz_submul(x_mpz[i], Lx_mpz[m],
                                x_mpz[j]));
                            h[i] = jnew;   // Entry is up to date
                        }

                        // Previous pivot exists
                        else
                        {
                            // x[i] = 0 - lij*x[j]
                            SPEX_CHECK(SPEX_mpz_submul(x_mpz[i], Lx_mpz[m],
                                x_mpz[j]));

                            // x[i] = x[i] / rho[j-1]
                            SPEX_CHECK(SPEX_mpz_divexact(x_mpz[i], x_mpz[i],
                                rhos_mpz[jnew-1]));
                            h[i] = jnew;   // Entry is up to date
                        }
                    }

                    //----------------------------------------------------------
                    /************ Both lij and x[i] are nonzero****************/
                    // x[i] != 0 --> History & IPGE update on x[i]
                    //----------------------------------------------------------
                    else
                    {
                        // No previous pivot in this case
                        if (jnew < 1)
                        {
                            // x[i] = x[i]*rho[0]
                            SPEX_CHECK(SPEX_mpz_mul(x_mpz[i],x_mpz[i],
                                rhos_mpz[0]));

                            // x[i] = x[i] - lij*xj
                            SPEX_CHECK(SPEX_mpz_submul(x_mpz[i], Lx_mpz[m],
                                x_mpz[j]));
                            h[i] = jnew;   // Entry is now up to date
                        }
                        // There is a previous pivot
                        else
                        {
                            // History update if necessary
                            if (h[i] < jnew - 1)
                            {
                                // x[i] = x[i] * rho[j-1]
                                SPEX_CHECK(SPEX_mpz_mul(x_mpz[i], x_mpz[i],
                                    rhos_mpz[jnew-1]));
                                if (h[i] > -1)
                                {
                                    // x[i] = x[i] / rho[h[i]]
                                    SPEX_CHECK(SPEX_mpz_divexact(x_mpz[i],
                                        x_mpz[i], rhos_mpz[h[i]]));
                                }
                            }
                            // x[i] = x[i] * rho[j]
                            SPEX_CHECK(SPEX_mpz_mul(x_mpz[i],x_mpz[i],
                                rhos_mpz[jnew]));
                            // x[i] = x[i] - lij*xj
                            SPEX_CHECK(SPEX_mpz_submul(x_mpz[i], Lx_mpz[m],
                                x_mpz[j]));
                            // x[i] = x[i] / rho[j-1]
                            SPEX_CHECK(SPEX_mpz_divexact(x_mpz[i], x_mpz[i],
                                rhos_mpz[jnew-1]));
                            h[i] = jnew;   // Entry is up to date
                        }
                    }
                }
            }
        }
        else                               // Entries of L
        {
            //------------------------------------------------------------------
            // History update
            //------------------------------------------------------------------
            if (h[j] < k-1)
            {
                // x[j] = x[j] * rho[k-1]
                SPEX_CHECK(SPEX_mpz_mul(x_mpz[j], x_mpz[j], rhos_mpz[k-1]));
                if (h[j] > -1)
                {
                    // x[j] = x[j] / rho[h[j]]
                    SPEX_CHECK(SPEX_mpz_divexact(x_mpz[j], x_mpz[j],
                        rhos_mpz[h[j]]));
                }
            }
        }
    }
    // Output the beginning of nonzero pattern
    *top_output = top;
    return SPEX_OK;
}

