/* ------------------------------------------------------------------------- */
/* AMD Copyright (c) by Timothy A. Davis,				     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

#include <stdio.h>
#include "amd.h"

int32_t n = 5 ;
int32_t Ap [ ] = { 0,   2,       6,       10,  12, 14} ;
int32_t Ai [ ] = { 0,1, 0,1,2,4, 1,2,3,4, 2,3, 1,4   } ;
int32_t P [5] ;

int main (void)
{
    int32_t k ;
    (void) amd_order (n, Ap, Ai, P, (double *) NULL, (double *) NULL) ;
    for (k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, P [k]) ;
    return (0) ;
}

