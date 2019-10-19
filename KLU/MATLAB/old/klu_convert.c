/* ========================================================================== */
/* === klu_convert ========================================================== */
/* ========================================================================== */

/* Convert KLU's LU factors to a conventional compressed column matrix,
 * containing both L and U.  Used by the MATLAB interface to KLU.
 */

#include "klu_internal.h"

/* assumes Xi and Xx are allocated appropriate size */
void KLU_convert (double *LU1, int *Xip, int *Xlen, int *Xi, double *Xx1,
	double *Xdiag1, int n)
{
    Entry *LU, *Xx, *Xdiag ;

    LU = (Entry *) LU1 ;
    Xx = (Entry *) Xx1 ;
    Xdiag = (Entry *) Xdiag1 ;

    int p, begin = 0, index = 0, len, k ;
    int *Xitemp ;
    Entry *Xxtemp ;
    for (p = 0 ; p < n ; p++)
    {
	GET_POINTER (LU, Xip, Xlen, Xitemp, Xxtemp, p, len) ;
	Xip [p] = begin ; /* changing Xip to Xp inplace */

	for (k = 0 ; k < len ; k++)
	{
	    Xi [index] = Xitemp [k] ;
	    Xx [index] = Xxtemp [k] ;
	    index++;
	}
	Xi [index] = p ;
	if (Xdiag == (Entry *) NULL)
	{
	    CLEAR (Xx [index]) ;
	    REAL (Xx [index]) = 1 ;
	}
	else
	{
	    Xx [index] = Xdiag [p] ;
	}
	index++ ;
 	len++ ;

	begin += len ;
    }
    Xip [n] = begin ;
}
