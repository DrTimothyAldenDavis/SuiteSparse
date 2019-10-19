#include "cs_mex.h"
/* check MATLAB input argument */
void cs_mex_check (int nel, int m, int n, int square, int sparse, int values,
    const mxArray *A)
{
    int nnel, mm = mxGetM (A), nn = mxGetN (A) ;
    if (values)
    {
        if (mxIsComplex (A))
        {
            mexErrMsgTxt ("matrix must be real; try CXSparse instead") ;
        }
    }
    if (sparse && !mxIsSparse (A)) mexErrMsgTxt ("matrix must be sparse") ;
    if (!sparse)
    {
        if (mxIsSparse (A)) mexErrMsgTxt ("matrix must be full") ;
        if (values && !mxIsDouble (A)) mexErrMsgTxt ("matrix must be double") ;
    }
    if (nel)
    {
        /* check number of elements */
        nnel = mxGetNumberOfElements (A) ;
        if (m >= 0 && n >= 0 && m*n != nnel) mexErrMsgTxt ("wrong length") ;
    }
    else
    {
        /* check row and/or column dimensions */
        if (m >= 0 && m != mm) mexErrMsgTxt ("wrong dimension") ;
        if (n >= 0 && n != nn) mexErrMsgTxt ("wrong dimension") ;
    }
    if (square && mm != nn) mexErrMsgTxt ("matrix must be square") ;
}

/* get a MATLAB sparse matrix and convert to cs */
cs *cs_mex_get_sparse (cs *A, int square, int values, const mxArray *Amatlab)
{
    cs_mex_check (0, -1, -1, square, 1, values, Amatlab) ;
    A->m = mxGetM (Amatlab) ;
    A->n = mxGetN (Amatlab) ;
    A->p = mxGetJc (Amatlab) ;
    A->i = mxGetIr (Amatlab) ;
    A->x = values ? mxGetPr (Amatlab) : NULL ;
    A->nzmax = mxGetNzmax (Amatlab) ;
    A->nz = -1 ;    /* denotes a compressed-col matrix, instead of triplet */
    return (A) ;
}

/* return a sparse matrix to MATLAB */
mxArray *cs_mex_put_sparse (cs **Ahandle)
{
    cs *A ;
    mxArray *Amatlab ;
    A = *Ahandle ;
    Amatlab = mxCreateSparse (0, 0, 0, mxREAL) ;
    mxSetM (Amatlab, A->m) ;
    mxSetN (Amatlab, A->n) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    cs_free (mxGetJc (Amatlab)) ;
    cs_free (mxGetIr (Amatlab)) ;
    cs_free (mxGetPr (Amatlab)) ;
    mxSetJc (Amatlab, A->p) ;           /* assign A->p pointer to MATLAB A */
    mxSetIr (Amatlab, A->i) ;
    mxSetPr (Amatlab, A->x) ;
    mexMakeMemoryPersistent (A->p) ;    /* ensure MATLAB does not free A->p */
    mexMakeMemoryPersistent (A->i) ;
    mexMakeMemoryPersistent (A->x) ;
    cs_free (A) ;                       /* frees A struct only, not A->p, etc */
    *Ahandle = NULL ;
    return (Amatlab) ;
}

/* get a MATLAB dense column vector */
double *cs_mex_get_double (int n, const mxArray *X)
{
    cs_mex_check (0, n, 1, 0, 0, 1, X) ;
    return (mxGetPr (X)) ;
}

/* return a double vector to MATLAB */
double *cs_mex_put_double (int n, const double *b, mxArray **X)
{
    double *x ;
    int k ;
    *X = mxCreateDoubleMatrix (n, 1, mxREAL) ;      /* create x */
    x = mxGetPr (*X) ;
    for (k = 0 ; k < n ; k++) x [k] = b [k] ;       /* copy x = b */
    return (x) ;
}

/* get a MATLAB flint array and convert to int */
int *cs_mex_get_int (int n, const mxArray *Imatlab, int *imax, int lo)
{
    double *p ;
    int i, k, *C = cs_malloc (n, sizeof (int)) ;
    cs_mex_check (1, n, 1, 0, 0, 1, Imatlab) ;
    p = mxGetPr (Imatlab) ;
    *imax = 0 ;
    for (k = 0 ; k < n ; k++)
    {
        i = p [k] ;
        C [k] = i - 1 ;
        if (i < lo) mexErrMsgTxt ("index out of bounds") ;
        *imax = CS_MAX (*imax, i) ;
    }
    return (C) ;
}

/* return an int array to MATLAB as a flint row vector */
mxArray *cs_mex_put_int (int *p, int n, int offset, int do_free)
{
    mxArray *X = mxCreateDoubleMatrix (1, n, mxREAL) ;
    double *x = mxGetPr (X) ;
    int k ;
    for (k = 0 ; k < n ; k++) x [k] = (p ? p [k] : k) + offset ;
    if (do_free) cs_free (p) ;
    return (X) ;
}
