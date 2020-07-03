//------------------------------------------------------------------------------
// SLIP_LU/SLIP_LU_factorize: exact sparse LU factorization
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function performs the SLIP LU factorization. This factorization
 * is done via n iterations of the sparse REF triangular solve function. The
 * overall factorization is PAQ = LDU
 * The determinant of A can be obtained as determinant = rhos[n-1]
 *
 *  L: undefined on input, created on output
 *  U: undefined on input, created on output
 *  rhos: undefined on input, created on output
 *  pinv: undefined on input, created on output
 *
 *  A: input only, not modified
 *  S: input only, not modified
 *  option: input only, not modified
 */

#define SLIP_FREE_WORK              \
    SLIP_matrix_free(&x, NULL);     \
    SLIP_FREE(xi);                  \
    SLIP_FREE(h);                   \
    SLIP_FREE(pivs);                \
    SLIP_FREE(row_perm);            \

#define SLIP_FREE_ALL               \
    SLIP_FREE_WORK                  \
    SLIP_matrix_free(&L, NULL);     \
    SLIP_matrix_free(&U, NULL);     \
    SLIP_matrix_free(&rhos, NULL);  \
    SLIP_FREE(pinv);

#include "slip_internal.h"

SLIP_info SLIP_LU_factorize
(
    // output:
    SLIP_matrix **L_handle,    // lower triangular matrix
    SLIP_matrix **U_handle,    // upper triangular matrix
    SLIP_matrix **rhos_handle, // sequence of pivots
    int64_t **pinv_handle,     // inverse row permutation
    // input:
    const SLIP_matrix *A,      // matrix to be factored
    const SLIP_LU_analysis *S, // column permutation and estimates
                               // of nnz in L and U 
    const SLIP_options* option // command options
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!slip_initialized ( )) return (SLIP_PANIC) ;

    SLIP_REQUIRE (A, SLIP_CSC, SLIP_MPZ) ;
    int64_t anz = SLIP_matrix_nnz (A, option) ;

    if (!L_handle || !U_handle || !rhos_handle || !pinv_handle || !S || anz < 0)
    {
        return SLIP_INCORRECT_INPUT;
    }

    (*L_handle) = NULL ;
    (*U_handle) = NULL ;
    (*rhos_handle) = NULL ;
    (*pinv_handle) = NULL ;

    //--------------------------------------------------------------------------
    // Declare and initialize workspace
    //--------------------------------------------------------------------------

    SLIP_matrix *L = NULL ;
    SLIP_matrix *U = NULL ;
    SLIP_matrix *rhos = NULL ;
    int64_t *pinv = NULL ;
    int64_t *xi = NULL ;
    int64_t *h = NULL ;
    int64_t *pivs = NULL ;
    int64_t *row_perm = NULL ;
    SLIP_matrix *x = NULL ;

    SLIP_info info ;
    int64_t n = A->n ;

    int64_t k = 0, top, i, j, col, loc, lnz = 0, unz = 0, pivot, jnew ;
    size_t size ;

    // Inverse pivot ordering
    pinv = (int64_t *) SLIP_malloc (n * sizeof (int64_t)) ;

    // Indicator of which rows have been pivotal
    // pivs[i] = 1 if row i has been selected as a pivot
    // row, otherwise, pivs[i] < 0
    pivs = (int64_t*) SLIP_malloc(n* sizeof(int64_t));

    // h is the history vector utilized for the sparse REF
    // triangular solve algorithm. h serves as a global
    // vector which is repeatedly passed into the triangular
    // solve algorithm
    h = (int64_t*) SLIP_malloc(n* sizeof(int64_t));

    // xi is the global nonzero pattern vector. It stores
    // the pattern of nonzeros of the kth column of L and U
    // for the triangular solve.
    xi = (int64_t*) SLIP_malloc(2*n* sizeof(int64_t));

    // Actual row permutation, the inverse of pinv. This
    // is used for sorting
    row_perm = (int64_t*) SLIP_malloc(n* sizeof(int64_t));

    if (!pivs || !h || !xi || !row_perm || !pinv)
    {
        // out of memory: free everything and return
        SLIP_FREE_ALL  ;
        return SLIP_OUT_OF_MEMORY;
    }

    // initialize workspace and pivot status
    for (i = 0; i < n; i++)
    {
        h[i] = -1;
        pivs[i] = -1;
        // Initialize location based vectors
        pinv[i] = i;
        row_perm[i] = i;
    }

    //--------------------------------------------------------------------------
    // Declare memory for rhos, L, and U
    //--------------------------------------------------------------------------

    // Create rhos, a global dense mpz_t matrix of dimension n*1
    SLIP_CHECK (SLIP_matrix_allocate(&rhos, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, false, option));

    // Allocate L and U without initializing each entry.
    // L and U are allocated to have nnz(L) which is estimated by the symbolic
    // analysis. However, unlike traditional matrix allocation, the second
    // boolean parameter here is set to false, so the individual values of
    // L and U are not allocated. Instead, a more efficient method to
    // allocate these values is done in the factorization to reduce
    // memory usage.
    SLIP_CHECK (SLIP_matrix_allocate(&L, SLIP_CSC, SLIP_MPZ, n, n, S->lnz,
        false, false, option));
    SLIP_CHECK (SLIP_matrix_allocate(&U, SLIP_CSC, SLIP_MPZ, n, n, S->unz,
        false, false, option));

    //--------------------------------------------------------------------------
    // allocate and initialize the workspace x
    //--------------------------------------------------------------------------

    // SLIP LU utilizes arbitrary sized integers which can grow beyond the
    // default 64 bits allocated by GMP. If the integers frequently grow, GMP
    // can get bogged down by performing intermediate reallocations. Instead,
    // we utilize a larger estimate on the workspace x vector so that computing
    // the values in L and U do not require too many extra intemediate calls to
    // realloc.
    //
    // Note that the estimate presented here is not an upper bound nor a lower
    // bound.  It is still possible that more bits will be required which is
    // correctly handled internally.
    int64_t estimate = 64 * SLIP_MAX (2, ceil (log2 ((double) n))) ;

    // Create x, a global dense mpz_t matrix of dimension n*1. Unlike rhos, the
    // second boolean parameter is set to false to avoid initializing
    // each mpz entry of x with default size.  It is intialized below.
    SLIP_CHECK (SLIP_matrix_allocate(&x, SLIP_DENSE, SLIP_MPZ, n, 1, n,
        false, /* do not initialize the entries of x: */ false, option));

    // initialize the entries of x
    for (i = 0; i < n; i++)
    {
        // Allocate memory for entries of x
        SLIP_CHECK(SLIP_mpz_init2(x->x.mpz[i], estimate));
    }

    //--------------------------------------------------------------------------
    // Iterations 0:n-1 (1:n in standard)
    //--------------------------------------------------------------------------

    for (k = 0; k < n; k++)
    {
        // Column pointers for column k of L and U
        L->p[k] = lnz;
        U->p[k] = unz;
        col = S->q[k];

        //----------------------------------------------------------------------
        // Reallocate memory if necessary
        // if lnz+n > L->nzmax, L needs to expand to accomodate new nonzeros.
        // To do so, we double the size of the L and U matrices.
        //----------------------------------------------------------------------
        if (lnz + n > L->nzmax)
        {
            // Double the size of L
            SLIP_CHECK(slip_sparse_realloc(L));
        }
        if (unz + n > U->nzmax)
        {
            // Double the size of U
            SLIP_CHECK(slip_sparse_realloc(U));
        }

        //----------------------------------------------------------------------
        // Triangular solve to compute LDx = A(:,k)
        //----------------------------------------------------------------------
        SLIP_CHECK(slip_ref_triangular_solve(&top, L, A, k, xi,
            (const int64_t *) (S->q),
            rhos,
            (const int64_t *) pinv,
            (const int64_t *) row_perm,
            h, x)) ;

        //----------------------------------------------------------------------
        // Obtain pivot
        //----------------------------------------------------------------------
        SLIP_CHECK(slip_get_pivot(&pivot, x, pivs, n, top, xi,
            col, k, rhos, pinv, row_perm, option));

        //----------------------------------------------------------------------
        // Populate L and U. We iterate across all nonzeros in x
        //----------------------------------------------------------------------
        for (j = top; j < n; j++)
        {
            jnew = xi[j];
            // Location of x[j] in final matrix
            loc = pinv[jnew];

            //------------------------------------------------------------------
            // loc <= k are rows above k, thus go to U
            //------------------------------------------------------------------
            if (loc <= k)
            {
                // Place the i location of the unz nonzero
                U->i[unz] = jnew;
                // Find the size in bits of x[j]
                SLIP_CHECK(SLIP_mpz_sizeinbase(&size, x->x.mpz[jnew], 2));
                // GMP manual: Allocated size should be size+2
                SLIP_CHECK(SLIP_mpz_init2(U->x.mpz[unz], size+2));
                // Place the x value of the unz nonzero
                SLIP_CHECK(SLIP_mpz_set(U->x.mpz[unz], x->x.mpz[jnew]));
                // Increment unz
                unz++;
            }

            //------------------------------------------------------------------
            // loc >= k are rows below k, thus go to L
            //------------------------------------------------------------------
            if (loc >= k)
            {
                // Place the i location of the lnz nonzero
                L->i[lnz] = jnew;
                // Set the size of x[j]
                SLIP_CHECK(SLIP_mpz_sizeinbase(&size, x->x.mpz[jnew], 2));
                // GMP manual: Allocated size should be size+2
                SLIP_CHECK(SLIP_mpz_init2(L->x.mpz[lnz], size+2));
                // Place the x value of the lnz nonzero
                SLIP_CHECK(SLIP_mpz_set(L->x.mpz[lnz], x->x.mpz[jnew]));
                // Increment lnz
                lnz++;
            }
        }
    }

    // Finalize L->p, U->p
    L->p[n] = lnz;
    U->p[n] = unz;

    //--------------------------------------------------------------------------
    // Free memory
    //--------------------------------------------------------------------------

    // free everything, but keep L, U, rhos, and pinv
    SLIP_FREE_WORK ;

    // This cannot fail since the size of L and U are shrinking.
    // Collapse L
    slip_sparse_collapse(L);
    // Collapse U
    slip_sparse_collapse(U);

    //--------------------------------------------------------------------------
    // finalize the row indices in L and U
    //--------------------------------------------------------------------------

    // Permute entries in L
    for (i = 0; i < lnz; i++)
    {
        L->i[i] = pinv[L->i[i]];
    }
    // Permute entries in U
    for (i = 0; i < unz; i++)
    {
        U->i[i] = pinv[U->i[i]];
    }

    //--------------------------------------------------------------------------
    // check the LU factorization (debugging only)
    //--------------------------------------------------------------------------

    #if 0
    SLIP_CHECK (SLIP_matrix_check (L, option)) ;
    SLIP_CHECK (SLIP_matrix_check (U, option)) ;
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*L_handle) = L ;
    (*U_handle) = U ;
    (*rhos_handle) = rhos ;
    (*pinv_handle) = pinv ;
    return (SLIP_OK) ;
}

