/* ========================================================================== */
/* === Tcov/cmread ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Version 1.2.  Copyright (C) 2005-2006, Timothy A. Davis
 * The CHOLMOD/Tcov Module is licensed under Version 2.0 of the GNU
 * General Public License.  See gpl.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Read in a matrix from a file and print it out.
 *
 * Usage:
 *	cmread matrixfile
 *	cmread < matrixfile
 */

#include "cholmod.h"

int main (int argc, char **argv)
{
    cholmod_sparse *A ;
    FILE *f ;
    cholmod_common Common, *cm ;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    if (argc > 1)
    {
	if ((f = fopen (argv [1], "r")) == NULL)
	{
	    printf ("cannot open file: %s\n", argv [1]) ;
	    return (0) ;
	}
    }
    else
    {
	f = stdin ;
    }

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD, read the matrix, print it, and free it */
    /* ---------------------------------------------------------------------- */

#ifdef DLONG

    cm = &Common ;
    cholmod_l_start (cm) ;
    cm->print = 5 ;
    A = cholmod_l_read_sparse (f, cm) ;
    if (argc > 1) fclose (f) ;
    cholmod_l_print_sparse (A, "A", cm) ;
    cholmod_l_free_sparse (&A, cm) ;
    cholmod_l_finish (cm) ;

#else

    cm = &Common ;
    cholmod_start (cm) ;
    cm->print = 5 ;
    A = cholmod_read_sparse (f, cm) ;
    if (argc > 1) fclose (f) ;
    cholmod_print_sparse (A, "A", cm) ;
    cholmod_free_sparse (&A, cm) ;
    cholmod_finish (cm) ;

#endif

    return (0) ;
}
