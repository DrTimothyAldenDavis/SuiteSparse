//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/sputil2.c: utilities for CHOLMOD's MATLAB interface
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Utility routines for the CHOLMOD MATLAB mexFunctions.
//
// If CHOLMOD runs out of memory, MATLAB will terminate the mexFunction
// immediately since it uses mxMalloc (see sputil2_config, below).  Likewise,
// if mxCreate* or mxMalloc (as called in this file) fails, MATLAB will also
// terminate the mexFunction.  When this occurs, MATLAB frees all allocated
// memory, so we don't have to worry about memory leaks.  If this were not the
// case, the routines in this file would suffer from memory leaks whenever an
// error occurred.

#include "sputil2.h"

// This file pointer is used for the mread and mwrite mexFunctions.  It must
// be a global variable, because the file pointer is not passed to the
// sputil2_error_handler function when an error occurs.
FILE *sputil2_file = NULL ;

//------------------------------------------------------------------------------
// sputil2_config
//------------------------------------------------------------------------------

// Define function pointers and other parameters for a mexFunction

void sputil2_config (int64_t spumoni, cholmod_common *cm)
{
    // cholmod_l_solve must return a real or complex X for MATLAB
    cm->prefer_zomplex = false ;

    // printing and error handling
    if (spumoni == 0)
    {
        // do not print anything from within CHOLMOD
        cm->print = -1 ;
        SuiteSparse_config_printf_func_set (NULL) ;
    }
    else
    {
        // spumoni = 1: print warning and error messages.  cholmod_l_print_*
        //      routines will print a one-line summary of each object printed.
        // spumoni = 2: also print a short summary of each object.
        cm->print = spumoni + 2 ;
        // SuiteSparse_config_printf_func_set ((void *) mexPrintf) ;
    }

    // error handler
    cm->error_handler = sputil2_error_handler ;

    // Turn off METIS memory guard.  It is not needed, because mxMalloc will
    // safely terminate the mexFunction and free any workspace without killing
    // all of MATLAB.  This assumes cholmod_make was used to compile CHOLMOD
    // for MATLAB.
    cm->metis_memory = 0.0 ;
}

//------------------------------------------------------------------------------
// sputil2_error_handler
//------------------------------------------------------------------------------

void sputil2_error_handler
(
    int status,
    const char *file,
    int line,
    const char *message
)
{
    if (status < CHOLMOD_OK)
    {
        if (sputil2_file != NULL)
        {
            fclose (sputil2_file) ;
            sputil2_file = NULL ;
        }
        mexErrMsgTxt (message) ;
    }
}

//------------------------------------------------------------------------------
// sputil2_get_sparse
//------------------------------------------------------------------------------

// Create a shallow or deep CHOLMOD sparse copy of a MATLAB sparse or dense
// matrix.  A is returned with A->xtype of CHOLMOD_PATTERN (if Amatlab is
// logical), CHOLMOD_REAL (if Amatlab is real), or CHOLMOD_COMPLEX (if Amatlab
// is complex).  
//
// If the MATLAB matrix is dense, then A is a freshly allocated CHOLMOD sparse
// matrix.
//
// If the MATLAB matrix is sparse, then A == &Astatic, and it is fully or
// partially shallow.  The A->p and A->i arrays are always shallow.  The
// numerical array A->x can be shallow or deep.
//
// The output matrix must be freed by sputil2_free_sparse.  The CHOLMOD matrix
// A must not be modified, except by sputil2_free_sparse.
//
// Example:
//
//      mxArray M = pargin [0] ;
//      cholmod_sparse *A, Astatic ;
//      size_t xsize ;
//      A = sputil2_get_sparse (M, -1, CHOLMOD_SINGLE, &Astatic, &xsize, cm) ;
//      ... use A in CHOLMOD; do not change A or xsize.
//      sputil2_free_sparse (&A, Astatic, xsize, cm) ;

cholmod_sparse *sputil2_get_sparse  // create a CHOLMOD copy of a MATLAB matrix
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse or dense matrix
    int stype,              // assumed A->stype (-1: lower, 0: unsym, 1: upper)
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output, if the MATLAB input matrix is sparse.
    // output:
    size_t *xsize,          // if > 0, A->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_sparse.
    cholmod_common *cm
)
{

    cholmod_sparse *A = NULL ;

    if (!mxIsSparse (Amatlab))
    {
        // get a MATLAB dense matrix
        cholmod_dense Xstatic ;
        size_t X_xsize = 0 ;
        cholmod_dense *X = sputil2_get_dense (Amatlab, dtype, &Xstatic,
            &X_xsize, cm) ;
        // convert it to sparse, with A->stype of 0
        A = cholmod_l_dense_to_sparse (X, 1, cm) ;
        sputil2_free_dense (&X, &Xstatic, X_xsize, cm) ;
        if (stype != 0)
        {
            // convert from 0 to the requested stype
            cholmod_sparse *A2 = cholmod_l_copy (A, stype, 2, cm) ;
            cholmod_l_free_sparse (&A, cm) ;
            A = A2 ;
        }
    }
    else
    {
        // get a MATLAB sparse matrix
        A = sputil2_get_sparse_only (Amatlab, stype, dtype, Astatic, xsize, cm);
    }

    return (A) ;
}

//------------------------------------------------------------------------------
// sputil2_get_sparse_only
//------------------------------------------------------------------------------

// Get a CHOLMOD shallow or deep copy of a MATLAB sparse matrix.

cholmod_sparse *sputil2_get_sparse_only     // returns A = Astatic
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse matrix
    int stype,              // assumed A->stype (-1: lower, 0: unsym, 1: upper)
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output.
    // output:
    size_t *xsize,          // if > 0, A->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_sparse.
    cholmod_common *cm
)
{

    //--------------------------------------------------------------------------
    // fill the static header of A
    //--------------------------------------------------------------------------

    if (!mxIsSparse (Amatlab))
    {
        // the MATLAB matrix must be sparse
        mexErrMsgTxt ("input matrix must be sparse") ;
    }

    cholmod_sparse *A = Astatic ;
    memset (A, 0, sizeof (cholmod_sparse)) ;
    A->nrow = mxGetM (Amatlab) ;
    A->ncol = mxGetN (Amatlab) ;
    A->p = (int64_t *) mxGetJc (Amatlab) ;
    A->i = (int64_t *) mxGetIr (Amatlab) ;
    int64_t *Ap = A->p ;
    int64_t anz = Ap [A->ncol] ;
    A->nzmax = anz ;
    A->packed = true ;
    A->sorted = true ;
    A->nz = NULL ;
    A->itype = CHOLMOD_LONG ;
    A->stype = (stype < 0) ? -1 : (stype == 0 ? 0 : 1) ;
    A->dtype = (dtype == 0) ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE ;
    A->z = NULL ;
    (*xsize) = 0 ;

    //--------------------------------------------------------------------------
    // get the numerical values
    //--------------------------------------------------------------------------

    int Amatlab_class = mxGetClassID (Amatlab) ;
    if (Amatlab_class == mxLOGICAL_CLASS)
    {

        //----------------------------------------------------------------------
        // logical MATLAB sparse matrix
        //----------------------------------------------------------------------

        A->xtype = CHOLMOD_PATTERN ;
        A->x = NULL ;

    }
    else
    {

        //----------------------------------------------------------------------
        // numerical MATLAB sparse matrix
        //----------------------------------------------------------------------

        A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_COMPLEX : CHOLMOD_REAL ;
        size_t e = (A->xtype == CHOLMOD_REAL) ? 1 : 2 ;
        int Amatlab_dtype ;

        if (Amatlab_class == mxSINGLE_CLASS)
        {
            // single or single complex MATLAB sparse matrix
            Amatlab_dtype = CHOLMOD_SINGLE ;
        }
        else if (Amatlab_class == mxDOUBLE_CLASS)
        {
            // double or double complex MATLAB sparse matrix
            Amatlab_dtype = CHOLMOD_DOUBLE ;
        }
        else
        {
            // matrix class is not supported
            mexErrMsgTxt ("get_sparse: class not supported") ;
        }

        //----------------------------------------------------------------------
        // get a deep or shallow copy of A->x
        //----------------------------------------------------------------------

        if (mxIsEmpty (Amatlab))
        {

            //------------------------------------------------------------------
            // the MATLAB matrix is empty
            //------------------------------------------------------------------

            // CHOLMOD requires a non-NULL A->x array, but doesn't access it
            A->x = cholmod_l_calloc (2, sizeof (double), cm) ;
            (*xsize) = 2 * sizeof (double) ;

        }
        else if (A->dtype == Amatlab_dtype)
        {

            //------------------------------------------------------------------
            // the MATLAB dtype matches the requested type
            //------------------------------------------------------------------

            A->x = mxGetData (Amatlab) ;

        }
        else if (A->dtype == CHOLMOD_SINGLE)
        {

            //------------------------------------------------------------------
            // convert a MATLAB double matrix into CHOLMOD single
            //------------------------------------------------------------------

            double *Ax_matlab = (double *) mxGetData (Amatlab) ;
            int64_t anz2 = MAX (anz, 2) ;
            A->x = cholmod_l_malloc (anz2, e * sizeof (float), cm) ;
            (*xsize) = anz2 * e * sizeof (float) ;
            float *Ax = A->x ;
            for (int64_t k = 0 ; k < anz * e ; k++)
            {
                Ax [k] = (float) Ax_matlab [k] ;
            }

        }
        else // A->dtype == CHOLMOD_DOUBLE
        {

            //------------------------------------------------------------------
            // convert a MATLAB single matrix into CHOLMOD double
            //------------------------------------------------------------------

            // NOTE: this does not exist yet in MATLAB.
            float *Ax_matlab = (float *) mxGetData (Amatlab) ;
            int64_t anz2 = MAX (anz, 2) ;
            A->x = cholmod_l_malloc (anz2, e * sizeof (double), cm) ;
            (*xsize) = anz2 * e * sizeof (double) ;
            double *Ax = A->x ;
            for (int64_t k = 0 ; k < anz * e ; k++)
            {
                Ax [k] = (float) Ax_matlab [k] ;
            }
        }
    }
    return (A) ;
}

//------------------------------------------------------------------------------
// sputil2_free_sparse
//------------------------------------------------------------------------------

// Frees any content of a CHOLMOD sparse matrix created by sputil2_get_sparse
// or sputil2_get_sparse_only.

void sputil2_free_sparse    // free a matrix created by sputil2_get_sparse
(
    // input/output:
    cholmod_sparse **Ahandle,   // matrix created by sputil2_get_sparse
    cholmod_sparse *Astatic,
    // input:
    size_t xsize,               // from sputil2_get_sparse when A was created
    cholmod_common *cm
)
{

    if (Ahandle == NULL || *Ahandle == NULL)
    {
        // nothing to do
        return ;
    }

    cholmod_sparse *A = (*Ahandle) ;

    if (A == Astatic)
    {
        // A has a shallow header
        if (xsize > 0)
        {
            cholmod_l_free (xsize, sizeof (uint8_t), A->x, cm) ;
        }
        memset (A, 0, sizeof (cholmod_sparse)) ;
        (*Ahandle) = NULL ;
    }
    else
    {
        // A is a fully deep matrix
        cholmod_l_free_sparse (Ahandle, cm) ;
    }
}

//------------------------------------------------------------------------------
// sputil2_get_dense
//------------------------------------------------------------------------------

// Create a CHOLMOD dense matrix (single or double) from a MATLAB dense matrix
// (single, double, or logical).

cholmod_dense *sputil2_get_dense // CHOLMOD copy of a MATLAB dense matrix
(
    // input:
    const mxArray *Xmatlab, // MATLAB dense matrix
    int dtype,              // requested X->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_dense *Xstatic, // the header of X on input, contents not
                            // initialized.  Contains the CHOLMOD dense X on
                            // output.
    // output:
    size_t *xsize,          // if > 0, X->x has size xsize bytes, and it is a
                            // deep copy and must be freed by
                            // sputil2_free_dense.
    cholmod_common *cm
)
{

    //--------------------------------------------------------------------------
    // fill the static header of X
    //--------------------------------------------------------------------------

    if (mxIsSparse (Xmatlab))
    {
        // the MATLAB matrix must be dense
        mexErrMsgTxt ("input matrix must be dense") ;
    }

    cholmod_dense *X = Xstatic ;
    memset (X, 0, sizeof (cholmod_dense)) ;
    X->nrow = mxGetM (Xmatlab) ;
    X->ncol = mxGetN (Xmatlab) ;
    X->d = X->nrow ;
    int64_t xnz = X->nrow * X->ncol  ;
    X->nzmax = xnz ;
    X->dtype = (dtype == 0) ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE ;
    X->z = NULL ;
    (*xsize) = 0 ;

    //--------------------------------------------------------------------------
    // get the numerical values
    //--------------------------------------------------------------------------

    X->xtype = mxIsComplex (Xmatlab) ? CHOLMOD_COMPLEX : CHOLMOD_REAL ;
    size_t e = (X->xtype == CHOLMOD_REAL) ? 1 : 2 ;
    int Xmatlab_dtype ;

    int Xmatlab_class = mxGetClassID (Xmatlab) ;
    if (Xmatlab_class == mxSINGLE_CLASS)
    {
        // single or single complex MATLAB dense matrix
        Xmatlab_dtype = CHOLMOD_SINGLE ;
    }
    else if (Xmatlab_class == mxDOUBLE_CLASS)
    {
        // double or double complex MATLAB dense matrix
        Xmatlab_dtype = CHOLMOD_DOUBLE ;
    }
    else
    {
        // logical MATLAB dense matrix, converted to CHOLMOD_REAL
        X->xtype = CHOLMOD_REAL ;
        Xmatlab_dtype = -1 ;
    }

    //--------------------------------------------------------------------------
    // get a deep or shallow copy of X->x
    //--------------------------------------------------------------------------

    if (mxIsEmpty (Xmatlab))
    {

        //----------------------------------------------------------------------
        // the MATLAB matrix is empty
        //----------------------------------------------------------------------

        // CHOLMOD requires a non-NULL X->x array, but doesn't access it
        X->x = cholmod_l_calloc (2, sizeof (double), cm) ;
        (*xsize) = 2 * sizeof (double) ;

    }
    else if (X->dtype == Xmatlab_dtype)
    {

        //----------------------------------------------------------------------
        // the MATLAB dtype matches the requested type
        //----------------------------------------------------------------------

        X->x = mxGetData (Xmatlab) ;

    }
    else if (X->dtype == CHOLMOD_SINGLE)
    {

        //----------------------------------------------------------------------
        // convert a MATLAB double or logical matrix into CHOLMOD single
        //----------------------------------------------------------------------

        int64_t xnz2 = MAX (xnz, 2) ;
        X->x = cholmod_l_malloc (xnz2, e * sizeof (float), cm) ;
        (*xsize) = xnz2 * e * sizeof (float) ;
        float *Xx = X->x ;
        if (Xmatlab_dtype == CHOLMOD_DOUBLE)
        {
            // convert MATLAB double to CHOLMOD single
            double *Xx_matlab = (double *) mxGetData (Xmatlab) ;
            for (int64_t k = 0 ; k < xnz * e ; k++)
            {
                Xx [k] = (float) Xx_matlab [k] ;
            }
        }
        else
        {
            // convert MATLAB logical to CHOLMOD single
            uint8_t *Xx_matlab = (uint8_t *) mxGetData (Xmatlab) ;
            for (int64_t k = 0 ; k < xnz * e ; k++)
            {
                Xx [k] = (float) Xx_matlab [k] ;
            }
        }

    }
    else // X->dtype == CHOLMOD_DOUBLE
    {

        //----------------------------------------------------------------------
        // convert a MATLAB single or logical matrix into CHOLMOD double
        //----------------------------------------------------------------------

        int64_t xnz2 = MAX (xnz, 2) ;
        X->x = cholmod_l_malloc (xnz2, e * sizeof (double), cm) ;
        (*xsize) = xnz2 * e * sizeof (double) ;
        double *Xx = X->x ;
        if (Xmatlab_dtype == CHOLMOD_SINGLE)
        {
            // convert MATLAB single to CHOLMOD double
            float *Xx_matlab = (float *) mxGetData (Xmatlab) ;
            for (int64_t k = 0 ; k < xnz * e ; k++)
            {
                Xx [k] = (double) Xx_matlab [k] ;
            }
        }
        else
        {
            // convert MATLAB logical to CHOLMOD double
            uint8_t *Xx_matlab = (uint8_t *) mxGetData (Xmatlab) ;
            for (int64_t k = 0 ; k < xnz * e ; k++)
            {
                Xx [k] = (double) Xx_matlab [k] ;
            }
        }
    }

    return (X) ;
}

//------------------------------------------------------------------------------
// sputil2_free_dense
//------------------------------------------------------------------------------

// Frees any content of a CHOLMOD dense matrix created by sputil2_get_dense

void sputil2_free_dense    // free a matrix created by sputil2_get_dense
(
    // input/output:
    cholmod_dense **Xhandle,    // matrix created by sputil2_get_dense
    cholmod_dense *Xstatic,
    // input:
    size_t xsize,               // from sputil2_get_dense when X was created
    cholmod_common *cm
)
{

    if (Xhandle == NULL || *Xhandle == NULL)
    {
        // nothing to do
        return ;
    }

    cholmod_dense *X = (*Xhandle) ;

    if (X == Xstatic)
    {
        // X has a shallow header
        if (xsize > 0)
        {
            cholmod_l_free (xsize, sizeof (uint8_t), X->x, cm) ;
        }
        memset (X, 0, sizeof (cholmod_dense)) ;
        (*Xhandle) = NULL ;
    }
    else
    {
        mexErrMsgTxt ("invalid dense matrix") ;
    }
}

//------------------------------------------------------------------------------
// sputil2_get_sparse_pattern
//------------------------------------------------------------------------------

// Create a CHOLMOD_PATTERN sparse matrix for a MATLAB sparse or dense matrix.
// The stype is returned as zero.  The resulting matrix should not be modified,
// except to be freed by sputil2_free_sparse.
//
// Example:
//
//      mxArray M = pargin [0] ;
//      cholmod_sparse *A, Astatic ;
//      A = sputil2_get_sparse_pattern (M, CHOLMOD_SINGLE, &Astatic, cm) ;
//      ... use A in CHOLMOD; do not change A
//      sputil2_free_sparse (&A, Astatic, 0, cm) ;

cholmod_sparse *sputil2_get_sparse_pattern
(
    // input:
    const mxArray *Amatlab, // MATLAB sparse or dense matrix
    int dtype,              // requested A->dtype (0: double, nonzero: single)
    // input/output:
    cholmod_sparse *Astatic,    // a static header of A on input, contents not
                            // initialized.  Contains the CHOLMOD sparse A on
                            // output, if the MATLAB input matrix is sparse.
    cholmod_common *cm
)
{

    cholmod_sparse *A = NULL ;

    if (!mxIsSparse (Amatlab) && mxIsLogical (Amatlab))
    {

        //----------------------------------------------------------------------
        // convert full logical MATLAB matrix into CHOLMOD_PATTERN
        //----------------------------------------------------------------------

        int64_t nrow = mxGetM (Amatlab) ;
        int64_t ncol = mxGetN (Amatlab) ;
        uint8_t *x = (uint8_t *) mxGetData (Amatlab) ;
        int64_t nzmax = nrow * ncol ;

        //----------------------------------------------------------------------
        // count the number of nonzeros in the result
        //----------------------------------------------------------------------

        int64_t nz = 0 ;
        for (int64_t j = 0 ; j < nzmax ; j++)
        {
            if (x [j])
            {
                nz++ ;
            }
        }

        //----------------------------------------------------------------------
        // allocate the result A
        //----------------------------------------------------------------------

        A = cholmod_l_spzeros (nrow, ncol, nz, CHOLMOD_PATTERN + dtype, cm) ;
        int64_t *Ap = A->p ;
        int64_t *Ai = A->i ;

        //----------------------------------------------------------------------
        // copy the full logical matrix into the sparse matrix A
        //----------------------------------------------------------------------

        int64_t p = 0 ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            Ap [j] = p ;
            for (int64_t i = 0 ; i < nrow ; i++)
            {
                if (x [i+j*nrow])
                {
                    Ai [p++] = i ;
                }
            }
        }
        Ap [ncol] = nz ;

    }
    else
    {

        //----------------------------------------------------------------------
        // get the MATLAB matrix as a CHOLMOD sparse matrix
        //----------------------------------------------------------------------

        // use the existing dtype of the MATLAB matrix
        int Amatlab_dtype = (mxGetClassID (Amatlab) == mxSINGLE_CLASS) ?
                CHOLMOD_SINGLE : CHOLMOD_DOUBLE ;

        // get the sparse A
        size_t A_xsize = 0 ;
        A = sputil2_get_sparse (Amatlab, 0, Amatlab_dtype, Astatic, &A_xsize,
            cm) ;

        //----------------------------------------------------------------------
        // convert A to CHOLMOD_PATTERN
        //----------------------------------------------------------------------

        if (A == Astatic)
        {
            // A has a shallow header so free A->x and set A to CHOLMOD_PATTERN
            if (A_xsize > 0)
            {
                cholmod_l_free (A_xsize, sizeof (uint8_t), A->x, cm) ;
            }
            A->xtype = CHOLMOD_PATTERN ;
            A->x = NULL ;
        }
        else
        {
            // A is a fully deep matrix
            cholmod_l_sparse_xtype (CHOLMOD_PATTERN + dtype, A, cm) ;
        }
    }

    return (A) ;
}

//------------------------------------------------------------------------------
// sputil2_put_sparse
//------------------------------------------------------------------------------

// create a MATLAB version of a CHOLMOD sparse matrix.  The CHOLMOD sparse
// matrix is destroyed.

mxArray *sputil2_put_sparse     // return MATLAB version of the matrix
(
    cholmod_sparse **Ahandle,   // CHOLMOD version of the matrix
    mxClassID mxclass,          // requested class of the MATLAB matrix:
                                // mxDOUBLE_CLASS, mxSINGLE_CLASS, or
                                // mxLOGICAL_CLASS
    bool drop,                  // if true, drop explicit zeros
    cholmod_common *cm
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    mxArray *Amatlab ;
    if (Ahandle == NULL || (*Ahandle) == NULL)
    {
        mexErrMsgTxt ("A is null") ;
    }
    cholmod_sparse *A = (*Ahandle) ;

    //--------------------------------------------------------------------------
    // create the MATLAB sparse matrix
    //--------------------------------------------------------------------------

    mxComplexity mxcomplexity ;

    if (mxclass == mxLOGICAL_CLASS)
    {
        mxcomplexity = mxREAL ;
        Amatlab = mxCreateSparseLogicalMatrix (0, 0, 1) ;
    }
    else
    {
        mxcomplexity = (A->xtype <= CHOLMOD_REAL) ? mxREAL : mxCOMPLEX ;
        if (mxclass == mxDOUBLE_CLASS)
        {
            Amatlab = mxCreateSparse (0, 0, 1, mxcomplexity) ;
        }
        else if (mxclass == mxSINGLE_CLASS)
        {
            // not yet supported ...
            // Amatlab = mxCreateSparseSingle? ... (0, 0, 1, mxcomplexity) ;
            mexErrMsgTxt ("MATLAB sparse single matrices not yet supported") ;
        }
        else
        {
            mexErrMsgTxt ("put_sparse: class not supported") ;
        }
    }

    //--------------------------------------------------------------------------
    // convert the CHOLMOD A into the desired xtype and dtype
    //--------------------------------------------------------------------------

    int A_dtype = (mxclass == mxSINGLE_CLASS) ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
    int A_xtype = (mxclass == mxLOGICAL_CLASS) ? CHOLMOD_PATTERN :
            ((mxcomplexity == mxREAL) ? CHOLMOD_REAL : CHOLMOD_COMPLEX) ;

    // does nothing if the matrix is already in the requested xtype and dtype.
    cholmod_l_sparse_xtype (A_xtype + A_dtype, A, cm) ;

    //--------------------------------------------------------------------------
    // drop explicit zeros from A
    //--------------------------------------------------------------------------

    // This is optional, since dropping zeros from a LL' or LDL' factorization
    // will break its chordal property and thus breaks ldlrowmod, ldlupdate,
    // and lsubsolve.

    if (drop)
    {
        // drop zeros
        cholmod_l_drop (0, A, cm) ;
        // reduce the space taken by A->i and A->x
        int64_t anz = cholmod_l_nnz (A, cm) ;
        int64_t anzmax = A->nzmax ;
        if (anz < anzmax && anzmax > 4)
        {
            cholmod_l_reallocate_sparse (anz, A, cm) ;
        }
    }

    //--------------------------------------------------------------------------
    // transplant the pattern from A into Amatlab
    //--------------------------------------------------------------------------

    int64_t *Ap = A->p ;
    size_t anz = Ap [A->ncol] ;
    mxSetM (Amatlab, A->nrow) ;
    mxSetN (Amatlab, A->ncol) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    MXFREE (mxGetJc (Amatlab)) ;
    MXFREE (mxGetIr (Amatlab)) ;
    MXFREE (mxGetData (Amatlab)) ;
    mxSetJc (Amatlab, (mwIndex *) A->p) ;
    mxSetIr (Amatlab, (mwIndex *) A->i) ;
    A->p = NULL ;
    A->i = NULL ;
    mexMakeMemoryPersistent (A->p) ;
    mexMakeMemoryPersistent (A->i) ;

    //--------------------------------------------------------------------------
    // transplant the values from A into Amatlab
    //--------------------------------------------------------------------------

    if (A_xtype == CHOLMOD_PATTERN)
    {
        // give the MATLAB logical sparse matrix values all equal to 1
        uint8_t *Ax = cholmod_l_malloc (anz, sizeof (uint8_t), cm) ;
        memset (Ax, 1, anz * sizeof (uint8_t)) ;
        mxSetData (Amatlab, Ax) ;
        mexMakeMemoryPersistent (Ax) ;
    }
    else
    {
        // transplant A->x as the values of the MATLAB sparse matrix
        mxSetData (Amatlab, A->x) ;
        mexMakeMemoryPersistent (A->x) ;
        A->x = NULL ;
    }

    //--------------------------------------------------------------------------
    // free the CHOLMOD sparse matrix and return the new MATLAB sparse matrix
    //--------------------------------------------------------------------------

    cholmod_l_free_sparse (Ahandle, cm) ;
    return (Amatlab) ;
}

//------------------------------------------------------------------------------
// sputil2_put_dense
//------------------------------------------------------------------------------

// create a MATLAB version of a CHOLMOD dense matrix.  The CHOLMOD dense
// matrix is destroyed.

mxArray *sputil2_put_dense      // return MATLAB version of the matrix
(
    cholmod_dense **Xhandle,    // CHOLMOD version of the matrix
    mxClassID mxclass,          // requested class of the MATLAB matrix:
                                // mxDOUBLE_CLASS, mxSINGLE_CLASS, or
                                // mxLOGICAL_CLASS
    cholmod_common *cm
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    mxArray *Xmatlab ;
    if (Xhandle == NULL || (*Xhandle) == NULL)
    {
        mexErrMsgTxt ("X is null") ;
    }
    cholmod_dense *X = (*Xhandle) ;

    //--------------------------------------------------------------------------
    // create the MATLAB dense matrix
    //--------------------------------------------------------------------------

    mxComplexity mxcomplexity ;
    int dtype ;

    if (mxclass == mxLOGICAL_CLASS)
    {
        // logical case not yet supported
        mexErrMsgTxt ("put_dense: logical class not supported") ;
    }
    else
    {
        mxcomplexity = (X->xtype <= CHOLMOD_REAL) ? mxREAL : mxCOMPLEX ;
        if (mxclass == mxDOUBLE_CLASS || mxclass == mxSINGLE_CLASS)
        {
            Xmatlab = mxCreateNumericMatrix (0, 0, mxclass, mxcomplexity) ;
        }
        else
        {
            // other cases not yet supported
            mexErrMsgTxt ("put_dense: class not supported") ;
        }
    }

    //--------------------------------------------------------------------------
    // convert the CHOLMOD X into the desired xtype and dtype
    //--------------------------------------------------------------------------

    int X_dtype = (mxclass == mxSINGLE_CLASS) ? CHOLMOD_SINGLE : CHOLMOD_DOUBLE;
    int X_xtype = (mxcomplexity == mxREAL) ? CHOLMOD_REAL : CHOLMOD_COMPLEX ;

    // does nothing if the matrix is already in the requested xtype and dtype.
    cholmod_l_dense_xtype (X_xtype + X_dtype, X, cm) ;

    //--------------------------------------------------------------------------
    // transplant the matrix from X into Xmatlab
    //--------------------------------------------------------------------------

    mxSetM (Xmatlab, X->nrow) ;
    mxSetN (Xmatlab, X->ncol) ;
    MXFREE (mxGetData (Xmatlab)) ;
    mxSetData (Xmatlab, X->x) ;
    mexMakeMemoryPersistent (X->x) ;
    X->x = NULL ;

    //--------------------------------------------------------------------------
    // free the CHOLMOD dense matrix and return the new MATLAB dense matrix
    //--------------------------------------------------------------------------

    cholmod_l_free_dense (Xhandle, cm) ;
    return (Xmatlab) ;
}

//------------------------------------------------------------------------------
// sputil2_put_int
//------------------------------------------------------------------------------

// Convert a int64_t vector into a double mxArray

mxArray *sputil2_put_int
(
    int64_t *P,         // vector to convert
    int64_t n,          // length of P
    int64_t one_based   // 1 if convert from 0-based to 1-based, 0 otherwise
)
{
    mxArray *Q = mxCreateDoubleMatrix (1, n, mxREAL) ;
    double *p = (double *) mxGetData (Q) ;
    for (int64_t i = 0 ; i < n ; i++)
    {
        p [i] = (double) (P [i] + one_based) ;
    }
    return (Q) ;
}

//------------------------------------------------------------------------------
// sputil2_trim
//------------------------------------------------------------------------------

// Remove columns k to n-1 from a sparse matrix S, leaving columns 0 to k-1.
// S must be packed (there can be no S->nz array).

void sputil2_trim
(
    cholmod_sparse *S,
    int64_t k,
    cholmod_common *cm
)
{

    if (S == NULL)
    {
        return ;
    }

    if (!S->packed)
    {
        mexErrMsgTxt ("invalid matrix") ;
    }

    int64_t ncol = S->ncol ;
    if (k < 0 || k >= ncol)
    {
        // do not modify S
        return ;
    }

    // reduce S->p in size.  This cannot fail.
    size_t n1 = ncol + 1 ;
    S->p = cholmod_l_realloc (k+1, sizeof (int64_t), S->p, &n1, cm) ;

    // get the new number of entries in S
    int64_t *Sp = S->p ;
    size_t nznew = Sp [k] ;

    // reduce S->i, S->x, and S->z (if present) to size nznew
    cholmod_l_reallocate_sparse (nznew, S, cm) ;

    // S now has only k columns
    S->ncol = k ;
}

//------------------------------------------------------------------------------
// sputil2_extract_zeros
//------------------------------------------------------------------------------

// Create a sparse binary (real) matrix Z that contains the pattern
// of explicit zeros in the sparse real/complex/zomplex double matrix A.
// A must be packed.

cholmod_sparse *sputil2_extract_zeros
(
    cholmod_sparse *A,
    cholmod_common *cm
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (A == NULL)
    {
        mexErrMsgTxt ("A is null") ;
    }

    if (!A->packed)
    {
        mexErrMsgTxt ("invalid matrix") ;
    }

    int dtype = A->dtype ;
    int64_t ncol = A->ncol ;
    int64_t nrow = A->nrow ;
    int64_t *Ap = A->p ;
    int64_t *Ai = A->i ;

    //--------------------------------------------------------------------------
    // count the number of zeros in a sparse matrix A
    //--------------------------------------------------------------------------

    int64_t nzeros = 0 ;
    bool is_complex = (A->xtype == CHOLMOD_COMPLEX) ;
    bool is_zomplex = (A->xtype == CHOLMOD_ZOMPLEX) ;

    if (A->xtype == CHOLMOD_PATTERN)
    {
        // Z is empty
        ;
    }
    else if (dtype == CHOLMOD_DOUBLE)
    {
        double *Ax = A->x ;
        double *Az = A->z ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                if (is_complex ? (Ax [2*p] == 0 && Ax [2*p+1] == 0) :
                   (is_zomplex ? (Ax [p] == 0 && Az [p] == 0) : (Ax [p] == 0)))
                {
                    nzeros++ ;
                }
            }
        }
    }
    else
    {
        float *Ax = A->x ;
        float *Az = A->z ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                if (is_complex ? (Ax [2*p] == 0 && Ax [2*p+1] == 0) :
                   (is_zomplex ? (Ax [p] == 0 && Az [p] == 0) : (Ax [p] == 0)))
                {
                    nzeros++ ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // allocate the Z matrix with space for all the zero entries
    //--------------------------------------------------------------------------

    cholmod_sparse *Z = cholmod_l_spzeros (nrow, ncol, nzeros,
        CHOLMOD_REAL + dtype, cm) ;
    if (nzeros == 0)
    {
        // Z is empty
        return (Z) ;
    }

    //--------------------------------------------------------------------------
    // extract the zeros from A and store them in Z as binary values
    //--------------------------------------------------------------------------

    int64_t pz = 0 ;
    int64_t *Zp = Z->p ;
    int64_t *Zi = Z->i ;

    if (dtype == CHOLMOD_DOUBLE)
    {
        double *Ax = A->x ;
        double *Az = A->z ;
        double *Zx = Z->x ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            Zp [j] = pz ;
            for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                if (is_complex ? (Ax [2*p] == 0 && Ax [2*p+1] == 0) :
                   (is_zomplex ? (Ax [p] == 0 && Az [p] == 0) : (Ax [p] == 0)))
                {
                    Zi [pz] = Ai [p] ;
                    Zx [pz] = 1 ;
                    pz++ ;
                }
            }
        }
    }
    else
    {
        float *Ax = A->x ;
        float *Az = A->z ;
        float *Zx = Z->x ;
        for (int64_t j = 0 ; j < ncol ; j++)
        {
            Zp [j] = pz ;
            for (int64_t p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                if (is_complex ? (Ax [2*p] == 0 && Ax [2*p+1] == 0) :
                   (is_zomplex ? (Ax [p] == 0 && Az [p] == 0) : (Ax [p] == 0)))
                {
                    Zi [pz] = Ai [p] ;
                    Zx [pz] = 1 ;
                    pz++ ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last column of Z and return result
    //--------------------------------------------------------------------------

    Zp [ncol] = pz ;
    return (Z) ;
}

