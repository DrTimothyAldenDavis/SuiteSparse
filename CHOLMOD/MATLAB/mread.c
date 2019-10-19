/* ========================================================================== */
/* === CHOLMOD/MATLAB/mread mexFunction ===================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Version 1.2.  Copyright (C) 2005-2006,
 * Timothy A. Davis
 * The CHOLMOD/MATLAB Module is licensed under Version 2.0 of the GNU
 * General Public License.  See gpl.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Read a sparse matrix from a file, in "triplet" format.  Most MatrixMarket
 * format matrices are supported (the "array" format is not supported).
 * The Matrix Market "integer" format is converted into real, but the values
 * are preserved.  The "pattern" format is converted into real.  If the matrix
 * is unsymmetric, all of its values are equal to one.  If it is symmetric, the
 * kth diagonal entry is set to one plus the number of off-diagonal nonzeros in
 * row/column k, and off-diagonal entries are set to -1.
 */

#include "cholmod_matlab.h"

#define MAXLEN 1025

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    cholmod_sparse *A ;
    cholmod_triplet *T ;
    cholmod_common Common, *cm ;
    int *Ti, *Tj ;
    double *Tx, *Tz ;
    FILE *f ;
    char filename [MAXLEN] ;
    int nz, i, j, k, is_complex ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 1)
    {
	mexErrMsgTxt ("usage: A = mread (file)") ;
    }
    if (!mxIsChar (pargin [0]))
    {
	mexErrMsgTxt ("mread requires a filename") ;
    }
    mxGetString (pargin [0], filename, MAXLEN) ;
    f = fopen (filename, "r") ;
    if (f == NULL)
    {
	mexErrMsgTxt ("cannot open file") ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the triplet matrix */
    /* ---------------------------------------------------------------------- */

    T = cholmod_read_triplet (f, cm) ;
    fclose (f) ;
    if (T == NULL)
    {
	mexErrMsgTxt ("could not read file") ;
    }
    if (T->xtype == CHOLMOD_COMPLEX)
    {
	cholmod_triplet_xtype (CHOLMOD_ZOMPLEX, T, cm) ;
    }
    is_complex = (T->xtype == CHOLMOD_ZOMPLEX) ;

    /* ---------------------------------------------------------------------- */
    /* convert to unsymmetric, if necessary */
    /* ---------------------------------------------------------------------- */

    /* this could be done with cholmod_copy for the real case, but that
     * does not work for the zomplex case. */
    if (T->stype != 0)
    {
	cholmod_reallocate_triplet (2*(T->nzmax), T, cm) ;
	Ti = T->i ;
	Tj = T->j ;
	Tx = T->x ;
	Tz = T->z ;
	nz = T->nnz ;
	for (k = 0 ; k < T->nnz ; k++)
	{
	    i = Ti [k] ;
	    j = Tj [k] ;
	    if (i != j)
	    {
		Ti [nz] = j ;
		Tj [nz] = i ;
		Tx [nz] = Tx [k] ;
		if (is_complex)
		{
		    Tz [nz] = -Tz [k] ;
		}
		nz++ ;
	    }
	}
	T->nnz = nz ;
	T->stype = 0 ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert to sparse matrix */
    /* ---------------------------------------------------------------------- */

    A = cholmod_triplet_to_sparse (T, 0, cm) ;
    cholmod_free_triplet (&T, cm) ;

    /* ---------------------------------------------------------------------- */
    /* return result to MATLAB */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_sparse (&A, cm) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD A, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    cholmod_finish (cm) ;
    cholmod_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != (3 + mxIsComplex (pargout[0]))) mexErrMsgTxt ("!") ;
    */
}
