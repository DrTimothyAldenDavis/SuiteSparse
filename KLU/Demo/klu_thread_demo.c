//------------------------------------------------------------------------------
// KLU/Demo/klu_thread_demo: thread-safety regression test for klu_*solve_ws
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2026, University of Florida.  All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

/* Regression test for the thread-safe solve entry points (klu_solve_ws and
 * klu_tsolve_ws).  Multiple threads concurrently solve distinct right-hand
 * sides against a single shared Numeric, each thread using its own scratch
 * buffer (sized via klu_solve_worksize) and its own klu_common.  Every
 * thread's result is compared against a serial reference solve.
 *
 * The legacy klu_solve / klu_tsolve entry points are NOT thread-safe against
 * a shared Numeric (they write to Numeric->Xwork); using them here would
 * fail this test, which is exactly what the _ws variants exist to fix. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "klu.h"

#define N      200    /* matrix size                              */
#define NTHR   8      /* worker threads                           */
#define ITERS  200    /* solves per thread                        */

/* Build a non-singular SPD tridiagonal: diag = 4, off-diag = -1. */
static void build_tridiag (int n, int **Ap_out, int **Ai_out, double **Ax_out)
{
    int nz = 3*n - 2 ;
    int *Ap = (int *) malloc ((n+1) * sizeof (int)) ;
    int *Ai = (int *) malloc (nz * sizeof (int)) ;
    double *Ax = (double *) malloc (nz * sizeof (double)) ;
    int p = 0, j ;
    for (j = 0 ; j < n ; j++)
    {
        Ap [j] = p ;
        if (j > 0)   { Ai [p] = j-1 ; Ax [p] = -1.0 ; p++ ; }
        Ai [p] = j ;   Ax [p] =  4.0 ; p++ ;
        if (j < n-1) { Ai [p] = j+1 ; Ax [p] = -1.0 ; p++ ; }
    }
    Ap [n] = p ;
    *Ap_out = Ap ; *Ai_out = Ai ; *Ax_out = Ax ;
}

typedef struct
{
    int tid ;
    klu_symbolic *Symbolic ;
    klu_numeric  *Numeric ;
    const double *rhs ;        /* this thread's RHS, read-only      */
    const double *expected ;   /* serial reference, read-only       */
    const double *expected_t ; /* serial transpose reference        */
    int fail ;
} thread_arg_t ;

static void *worker (void *p)
{
    thread_arg_t *a = (thread_arg_t *) p ;
    klu_common Common ;
    double b [N] ;
    void *Work ;
    size_t ws ;
    int it, i ;

    klu_defaults (&Common) ;
    ws = klu_solve_worksize (a->Symbolic, &Common) ;
    if (ws == 0) { a->fail = 1 ; return NULL ; }
    Work = malloc (ws) ;
    if (Work == NULL) { a->fail = 1 ; return NULL ; }

    for (it = 0 ; it < ITERS ; it++)
    {
        /* Ax = b */
        memcpy (b, a->rhs, sizeof (b)) ;
        if (!klu_solve_ws (a->Symbolic, a->Numeric, N, 1, b, Work, &Common))
        {
            fprintf (stderr, "thread %d: klu_solve_ws failed (status %d)\n",
                a->tid, Common.status) ;
            a->fail = 1 ; goto done ;
        }
        for (i = 0 ; i < N ; i++)
        {
            if (b [i] != a->expected [i])
            {
                fprintf (stderr,
                    "thread %d iter %d: solve mismatch at %d: %g vs %g\n",
                    a->tid, it, i, b [i], a->expected [i]) ;
                a->fail = 1 ; goto done ;
            }
        }

        /* A'x = b */
        memcpy (b, a->rhs, sizeof (b)) ;
        if (!klu_tsolve_ws (a->Symbolic, a->Numeric, N, 1, b, Work, &Common))
        {
            fprintf (stderr, "thread %d: klu_tsolve_ws failed (status %d)\n",
                a->tid, Common.status) ;
            a->fail = 1 ; goto done ;
        }
        for (i = 0 ; i < N ; i++)
        {
            if (b [i] != a->expected_t [i])
            {
                fprintf (stderr,
                    "thread %d iter %d: tsolve mismatch at %d: %g vs %g\n",
                    a->tid, it, i, b [i], a->expected_t [i]) ;
                a->fail = 1 ; goto done ;
            }
        }
    }
done:
    free (Work) ;
    return NULL ;
}

int main (void)
{
    int *Ap, *Ai ;
    double *Ax ;
    klu_symbolic *Symbolic ;
    klu_numeric  *Numeric ;
    klu_common Common ;
    double rhs [NTHR][N], ref [NTHR][N], ref_t [NTHR][N] ;
    pthread_t th [NTHR] ;
    thread_arg_t args [NTHR] ;
    int t, i, fail = 0 ;

    build_tridiag (N, &Ap, &Ai, &Ax) ;

    klu_defaults (&Common) ;
    Symbolic = klu_analyze (N, Ap, Ai, &Common) ;
    Numeric  = klu_factor  (Ap, Ai, Ax, Symbolic, &Common) ;
    if (Symbolic == NULL || Numeric == NULL)
    {
        fprintf (stderr, "factorization failed (status %d)\n", Common.status) ;
        return 1 ;
    }

    /* Distinct RHS per thread; compute serial references for solve and
     * tsolve using the legacy entry points (single-threaded == safe). */
    for (t = 0 ; t < NTHR ; t++)
    {
        for (i = 0 ; i < N ; i++)
        {
            rhs [t][i] = (double) ((t + 1) * (i + 1)) ;
        }
        memcpy (ref   [t], rhs [t], sizeof (rhs [t])) ;
        memcpy (ref_t [t], rhs [t], sizeof (rhs [t])) ;
        klu_solve  (Symbolic, Numeric, N, 1, ref   [t], &Common) ;
        klu_tsolve (Symbolic, Numeric, N, 1, ref_t [t], &Common) ;
    }

    for (t = 0 ; t < NTHR ; t++)
    {
        args [t].tid        = t ;
        args [t].Symbolic   = Symbolic ;
        args [t].Numeric    = Numeric ;
        args [t].rhs        = rhs   [t] ;
        args [t].expected   = ref   [t] ;
        args [t].expected_t = ref_t [t] ;
        args [t].fail       = 0 ;
        pthread_create (&th [t], NULL, worker, &args [t]) ;
    }
    for (t = 0 ; t < NTHR ; t++)
    {
        pthread_join (th [t], NULL) ;
        if (args [t].fail) fail = 1 ;
    }

    klu_free_numeric  (&Numeric,  &Common) ;
    klu_free_symbolic (&Symbolic, &Common) ;
    free (Ap) ; free (Ai) ; free (Ax) ;

    printf ("klu_thread_demo: %s (%d threads x %d iters x 2 solves, n=%d)\n",
        fail ? "FAIL" : "OK", NTHR, ITERS, N) ;
    return fail ;
}
