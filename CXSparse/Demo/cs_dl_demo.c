// CSparse/Demo/cs_dl_demo: demo utilities for CXSparse (double, SuiteSparse_long)
// CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1+
#include "cs_dl_demo.h"
#include <time.h>
/* 1 if A is square & upper tri., -1 if square & lower tri., 0 otherwise */
static SuiteSparse_long is_sym (cs_dl *A)
{
    SuiteSparse_long is_upper, is_lower, j, p, n = A->n, m = A->m, *Ap = A->p, *Ai = A->i ;
    if (m != n) return (0) ;
    is_upper = 1 ;
    is_lower = 1 ;
    for (j = 0 ; j < n ; j++)
    {
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            if (Ai [p] > j) is_upper = 0 ;
            if (Ai [p] < j) is_lower = 0 ;
        }
    }
    return (is_upper ? 1 : (is_lower ? -1 : 0)) ;
}

/* true for off-diagonal entries */
static SuiteSparse_long dropdiag (SuiteSparse_long i, SuiteSparse_long j,
    double aij, void *other)
{ return (i != j) ;}

/* C = A + triu(A,1)' */
static cs_dl *make_sym (cs_dl *A)
{
    cs_dl *AT, *C ;
    AT = cs_dl_transpose (A, 1) ;          /* AT = A' */
    cs_dl_fkeep (AT, &dropdiag, NULL) ;    /* drop diagonal entries from AT */
    C = cs_dl_add (A, AT, 1, 1) ;          /* C = A+AT */
    cs_dl_spfree (AT) ;
    return (C) ;
}

/* create a right-hand side */
static void rhs (double *x, double *b, SuiteSparse_long m)
{
    SuiteSparse_long i ;
    for (i = 0 ; i < m ; i++) b [i] = 1 + ((double) i) / m ;
    for (i = 0 ; i < m ; i++) x [i] = b [i] ;
}

/* infinity-norm of x */
static double norm (double *x, SuiteSparse_long n)
{
    SuiteSparse_long i ;
    double normx = 0 ;
    for (i = 0 ; i < n ; i++) normx = CS_MAX (normx, fabs (x [i])) ;
    return (normx) ;
}

/* compute residual, norm(A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf)) */
static void print_resid (SuiteSparse_long ok, cs_dl *A, double *x, double *b, double *resid)
{
    SuiteSparse_long i, m, n ;
    if (!ok) { printf ("    (failed)\n") ; return ; }
    m = A->m ; n = A->n ;
    for (i = 0 ; i < m ; i++) resid [i] = -b [i] ;  /* resid = -b */
    cs_dl_gaxpy (A, x, resid) ;                        /* resid = resid + A*x  */
    printf ("resid: %8.2e\n", norm (resid,m) / ((n == 0) ? 1 :
        (cs_dl_norm (A) * norm (x,n) + norm (b,m)))) ;
}

static double tic (void) { return (clock () / (double) CLOCKS_PER_SEC) ; }
static double toc (double t) { double s = tic () ; return (CS_MAX (0, s-t)) ; }

static void print_order (SuiteSparse_long order)
{
    switch (order)
    {
        case 0: printf ("natural    ") ; break ;
        case 1: printf ("amd(A+A')  ") ; break ;
        case 2: printf ("amd(S'*S)  ") ; break ;
        case 3: printf ("amd(A'*A)  ") ; break ;
    }
}

/* read a problem from a file; use %g for integers to avoid int64_t conflicts */
problem *get_problem (FILE *f, double tol)
{
    cs_dl *T, *A, *C ;
    SuiteSparse_long sym, m, n, mn, nz1, nz2 ;
    problem *Prob ;
    Prob = cs_dl_calloc (1, sizeof (problem)) ;
    if (!Prob) return (NULL) ;
    T = cs_dl_load (f) ;                   /* load triplet matrix T from a file */
    Prob->A = A = cs_dl_compress (T) ;     /* A = compressed-column form of T */
    cs_dl_spfree (T) ;                     /* clear T */
    if (!cs_dl_dupl (A)) return (free_problem (Prob)) ; /* sum up duplicates */
    Prob->sym = sym = is_sym (A) ;      /* determine if A is symmetric */
    m = A->m ; n = A->n ;
    mn = CS_MAX (m,n) ;
    nz1 = A->p [n] ;
    cs_dl_dropzeros (A) ;                  /* drop zero entries */
    nz2 = A->p [n] ;
    if (tol > 0) cs_dl_droptol (A, tol) ;  /* drop tiny entries (just to test) */
    Prob->C = C = sym ? make_sym (A) : A ;  /* C = A + triu(A,1)', or C=A */
    if (!C) return (free_problem (Prob)) ;
    printf ("\n--- Matrix: %g-by-%g, nnz: %g (sym: %g: nnz %g), norm: %8.2e\n",
            (double) m, (double) n, (double) (A->p [n]), (double) sym,
            (double) (sym ? C->p [n] : 0), cs_dl_norm (C)) ;
    if (nz1 != nz2) printf ("zero entries dropped: %g\n", (double) (nz1 - nz2));
    if (nz2 != A->p [n]) printf ("tiny entries dropped: %g\n",
            (double) (nz2 - A->p [n])) ;
    Prob->b = cs_dl_malloc (mn, sizeof (double)) ;
    Prob->x = cs_dl_malloc (mn, sizeof (double)) ;
    Prob->resid = cs_dl_malloc (mn, sizeof (double)) ;
    return ((!Prob->b || !Prob->x || !Prob->resid) ? free_problem (Prob) : Prob) ;
}

/* free a problem */
problem *free_problem (problem *Prob)
{
    if (!Prob) return (NULL) ;
    cs_dl_spfree (Prob->A) ;
    if (Prob->sym) cs_dl_spfree (Prob->C) ;
    cs_dl_free (Prob->b) ;
    cs_dl_free (Prob->x) ;
    cs_dl_free (Prob->resid) ;
    return (cs_dl_free (Prob)) ;
}

/* solve a linear system using Cholesky, LU, and QR, with various orderings */
SuiteSparse_long demo2 (problem *Prob)
{
    cs_dl *A, *C ;
    double *b, *x, *resid,  t, tol ;
    SuiteSparse_long k, m, n, ok, order, nb, ns, *r, *s, *rr, sprank ;
    cs_dld *D ;
    if (!Prob) return (0) ;
    A = Prob->A ; C = Prob->C ; b = Prob->b ; x = Prob->x ; resid = Prob->resid;
    m = A->m ; n = A->n ;
    tol = Prob->sym ? 0.001 : 1 ;               /* partial pivoting tolerance */
    D = cs_dl_dmperm (C, 1) ;                      /* randomized dmperm analysis */
    if (!D) return (0) ;
    nb = D->nb ; r = D->r ; s = D->s ; rr = D->rr ;
    sprank = rr [3] ;
    for (ns = 0, k = 0 ; k < nb ; k++)
    {
        ns += ((r [k+1] == r [k]+1) && (s [k+1] == s [k]+1)) ;
    }
    printf ("blocks: %g singletons: %g structural rank: %g\n",
        (double) nb, (double) ns, (double) sprank) ;
    cs_dl_dfree (D) ;
    for (order = 0 ; order <= 3 ; order += 3)   /* natural and amd(A'*A) */
    {
        if (!order && m > 1000) continue ;
        printf ("QR   ") ;
        print_order (order) ;
        rhs (x, b, m) ;                         /* compute right-hand side */
        t = tic () ;
        ok = cs_dl_qrsol (order, C, x) ;           /* min norm(Ax-b) with QR */
        printf ("time: %8.2f ", toc (t)) ;
        print_resid (ok, C, x, b, resid) ;      /* print residual */
    }
    if (m != n || sprank < n) return (1) ;      /* return if rect. or singular*/
    for (order = 0 ; order <= 3 ; order++)      /* try all orderings */
    {
        if (!order && m > 1000) continue ;
        printf ("LU   ") ;
        print_order (order) ;
        rhs (x, b, m) ;                         /* compute right-hand side */
        t = tic () ;
        ok = cs_dl_lusol (order, C, x, tol) ;      /* solve Ax=b with LU */
        printf ("time: %8.2f ", toc (t)) ;
        print_resid (ok, C, x, b, resid) ;      /* print residual */
    }
    if (!Prob->sym) return (1) ;
    for (order = 0 ; order <= 1 ; order++)      /* natural and amd(A+A') */
    {
        if (!order && m > 1000) continue ;
        printf ("Chol ") ;
        print_order (order) ;
        rhs (x, b, m) ;                         /* compute right-hand side */
        t = tic () ;
        ok = cs_dl_cholsol (order, C, x) ;         /* solve Ax=b with Cholesky */
        printf ("time: %8.2f ", toc (t)) ;
        print_resid (ok, C, x, b, resid) ;      /* print residual */
    }
    return (1) ;
} 

/* free workspace for demo3 */
static SuiteSparse_long done3 (SuiteSparse_long ok, cs_dls *S, cs_dln *N,
    double *y, cs_dl *W, cs_dl *E, SuiteSparse_long *p)
{
    cs_dl_sfree (S) ;
    cs_dl_nfree (N) ;
    cs_dl_free (y) ;
    cs_dl_spfree (W) ;
    cs_dl_spfree (E) ;
    cs_dl_free (p) ;
    return (ok) ;
}

/* Cholesky update/downdate */
SuiteSparse_long demo3 (problem *Prob)
{
    cs_dl *A, *C, *W = NULL, *WW, *WT, *E = NULL, *W2 ;
    SuiteSparse_long n, k, *Li, *Lp, *Wi, *Wp, p1, p2, *p = NULL, ok ;
    double *b, *x, *resid, *y = NULL, *Lx, *Wx, s,  t, t1 ;
    cs_dls *S = NULL ;
    cs_dln *N = NULL ;
    if (!Prob) return (0) ;
    if (!Prob->sym || Prob->A->n == 0) return (1) ;
    A = Prob->A ; C = Prob->C ; b = Prob->b ; x = Prob->x ; resid = Prob->resid;
    n = A->n ;
    if (!Prob->sym || n == 0) return (1) ;
    rhs (x, b, n) ;                             /* compute right-hand side */
    printf ("\nchol then update/downdate ") ;
    print_order (1) ;
    y = cs_dl_malloc (n, sizeof (double)) ;
    t = tic () ;
    S = cs_dl_schol (1, C) ;                       /* symbolic Chol, amd(A+A') */
    printf ("\nsymbolic chol time %8.2f\n", toc (t)) ;
    t = tic () ;
    N = cs_dl_chol (C, S) ;                        /* numeric Cholesky */
    printf ("numeric  chol time %8.2f\n", toc (t)) ;
    if (!S || !N || !y) return (done3 (0, S, N, y, W, E, p)) ;
    t = tic () ;
    cs_dl_ipvec (S->pinv, b, y, n) ;               /* y = P*b */
    cs_dl_lsolve (N->L, y) ;                       /* y = L\y */
    cs_dl_ltsolve (N->L, y) ;                      /* y = L'\y */
    cs_dl_pvec (S->pinv, y, x, n) ;                /* x = P'*y */
    printf ("solve    chol time %8.2f\n", toc (t)) ;
    printf ("original: ") ;
    print_resid (1, C, x, b, resid) ;           /* print residual */
    k = n/2 ;                                   /* construct W  */
    W = cs_dl_spalloc (n, 1, n, 1, 0) ;
    if (!W) return (done3 (0, S, N, y, W, E, p)) ;
    Lp = N->L->p ; Li = N->L->i ; Lx = N->L->x ;
    Wp = W->p ; Wi = W->i ; Wx = W->x ;
    Wp [0] = 0 ;
    p1 = Lp [k] ;
    Wp [1] = Lp [k+1] - p1 ;
    s = Lx [p1] ;
    srand (1) ;
    for ( ; p1 < Lp [k+1] ; p1++)
    {
        p2 = p1 - Lp [k] ;
        Wi [p2] = Li [p1] ;
        Wx [p2] = s * rand () / ((double) RAND_MAX) ;
    }
    t = tic () ;
    ok = cs_dl_updown (N->L, +1, W, S->parent) ;   /* update: L*L'+W*W' */
    t1 = toc (t) ;
    printf ("update:   time: %8.2f\n", t1) ;
    if (!ok) return (done3 (0, S, N, y, W, E, p)) ;
    t = tic () ;
    cs_dl_ipvec (S->pinv, b, y, n) ;               /* y = P*b */
    cs_dl_lsolve (N->L, y) ;                       /* y = L\y */
    cs_dl_ltsolve (N->L, y) ;                      /* y = L'\y */
    cs_dl_pvec (S->pinv, y, x, n) ;                /* x = P'*y */
    t = toc (t) ;
    p = cs_dl_pinv (S->pinv, n) ;
    W2 = cs_dl_permute (W, p, NULL, 1) ;           /* E = C + (P'W)*(P'W)' */
    WT = cs_dl_transpose (W2,1) ;
    WW = cs_dl_multiply (W2, WT) ;
    cs_dl_spfree (WT) ;
    cs_dl_spfree (W2) ;
    E = cs_dl_add (C, WW, 1, 1) ;
    cs_dl_spfree (WW) ;
    if (!E || !p) return (done3 (0, S, N, y, W, E, p)) ;
    printf ("update:   time: %8.2f (incl solve) ", t1+t) ;
    print_resid (1, E, x, b, resid) ;           /* print residual */
    cs_dl_nfree (N) ;                              /* clear N */
    t = tic () ;
    N = cs_dl_chol (E, S) ;                        /* numeric Cholesky */
    if (!N) return (done3 (0, S, N, y, W, E, p)) ;
    cs_dl_ipvec (S->pinv, b, y, n) ;               /* y = P*b */
    cs_dl_lsolve (N->L, y) ;                       /* y = L\y */
    cs_dl_ltsolve (N->L, y) ;                      /* y = L'\y */
    cs_dl_pvec (S->pinv, y, x, n) ;                /* x = P'*y */
    t = toc (t) ;
    printf ("rechol:   time: %8.2f (incl solve) ", t) ;
    print_resid (1, E, x, b, resid) ;           /* print residual */
    t = tic () ;
    ok = cs_dl_updown (N->L, -1, W, S->parent) ;   /* downdate: L*L'-W*W' */
    t1 = toc (t) ;
    if (!ok) return (done3 (0, S, N, y, W, E, p)) ;
    printf ("downdate: time: %8.2f\n", t1) ;
    t = tic () ;
    cs_dl_ipvec (S->pinv, b, y, n) ;               /* y = P*b */
    cs_dl_lsolve (N->L, y) ;                       /* y = L\y */
    cs_dl_ltsolve (N->L, y) ;                      /* y = L'\y */
    cs_dl_pvec (S->pinv, y, x, n) ;                /* x = P'*y */
    t = toc (t) ;
    printf ("downdate: time: %8.2f (incl solve) ", t1+t) ;
    print_resid (1, C, x, b, resid) ;           /* print residual */
    return (done3 (1, S, N, y, W, E, p)) ;
} 
